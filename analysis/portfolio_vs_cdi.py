"""
portfolio_vs_cdi.py
-------------------
Simula o desempenho da carteira de estratégias ativas vs CDI.

Usa:
  - monitor_status.csv  → quais estratégias estão ativas
  - retornos.csv        → retornos diários dos ativos
  - cdi.csv             → benchmark CDI acumulado
  - position_sizing.py  → sizing via Kelly + vol + drawdown + correlação

Saída:
  - Console      : métricas comparativas (CAGR, Sharpe, drawdown...)
  - reports/portfolio_vs_cdi.png : equity curve + drawdown + distribuição
  - reports/portfolio_vs_cdi.csv : série diária com capital portfólio e CDI

Correção de direção:
  - MOM + sobe: LONG B   (B sobe com A)
  - MOM + cai:  SHORT B  (B cai com A)     → ret = -ret_B
  - CTR + sobe: SHORT B  (B nao segue A)   → ret = -ret_B
  - CTR + cai:  LONG B   (B nao cai com A) → ret = +ret_B
"""

import sys
import os
import re
import warnings
import time

# Força UTF-8 no stdout do Windows para evitar UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto, relativo a este arquivo
REPORTS = os.path.join(BASE, "reports")
DATA    = os.path.join(BASE, "data", "raw")

sys.path.insert(0, os.path.join(BASE, "analysis"))
from position_sizing import PortfolioSizer, CFG
from regime_detector import (
    integrate_backtest_historico, calibrar_fatores,
    MacroRegimeDetector, classificar_ativo,
)


# ── Slippage overnight por cenário ───────────────────────────────────────────
# Modela a diferença entre o fechamento do dia t (sinal) e a abertura do dia t+1
# (execução real). É um custo ADICIONAL ao spread/corretagem já no backtester.
#
# Dois tipos de ativo com slippage estruturalmente diferente:
#   "br"     : ações e fundos BR (.SA) — mercado menor, gap overnight maior
#   "global" : ETFs e índices globais (SPY, QQQ, ^VIX...) — mais líquidos
#
# Cenários (slippage como % do nocional por trade):
SLIPPAGE_CENARIOS = {
    "zero":        {"br": 0.00, "global": 0.00},  # sem slippage (base do backtest atual)
    "otimista":    {"br": 0.05, "global": 0.02},  # abertura próxima ao fechamento
    "base":        {"br": 0.15, "global": 0.05},  # realista — mix de ativos, algum gap
    "conservador": {"br": 0.30, "global": 0.10},  # BR ilíquido, gap overnight frequente
    "pessimista":  {"br": 0.50, "global": 0.20},  # pior caso: stress + ativos ilíquidos
}

# Thresholds do circuit breaker a comparar (None = sem CB)
CB_THRESHOLDS = [None, -0.05, -0.08, -0.12, -0.15, -0.20]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def carregar_dados() -> tuple:
    print("Carregando dados...")

    retornos = pd.read_csv(
        os.path.join(DATA, "retornos.csv"),
        index_col=0, parse_dates=True,
    )
    print(f"  retornos  : {retornos.shape[0]} dias x {retornos.shape[1]} ativos "
          f"({retornos.index[0].date()} - {retornos.index[-1].date()})")

    cdi = pd.read_csv(
        os.path.join(DATA, "cdi.csv"),
        index_col=0, parse_dates=True,
    )
    print(f"  cdi       : {cdi.index[0].date()} - {cdi.index[-1].date()}")

    monitor = pd.read_csv(os.path.join(REPORTS, "monitor_status.csv"))
    n_ativas = (monitor["status"] == "ativa").sum()
    print(f"  monitor   : {len(monitor)} estrategias | {n_ativas} ativas")

    exp_df = None
    for nome in ["exp3_contrarian.csv", "exp2_threshold_sweep.csv"]:
        caminho = os.path.join(REPORTS, nome)
        if os.path.exists(caminho):
            exp_df = pd.read_csv(caminho)
            print(f"  {nome}: {len(exp_df)} linhas")
            break

    return retornos, cdi, monitor, exp_df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_sizes(retornos, monitor, exp_df, capital) -> pd.DataFrame:
    sizer = PortfolioSizer(retornos=retornos, capital=capital)
    print("\nCalculando position sizing...")
    t0 = time.time()
    sizes = sizer.calcular(monitor, exp_df, verbose=True)

    # ── Teto por ativo B (corrige diluição com 600+ estratégias) ─────────────
    # Com centenas de estratégias, o reescalamento para 100% deixa cada posição
    # minúscula (~0.16%), gerando exposição efetiva de ~5% — insuficiente para
    # atingir IPCA+10. A solução é limitar quanto capital vai para cada ativo B.
    TETO_POR_ATIVO_B = 2.0   # % do capital por ativo B (ajuste aqui se quiser)
    ORCAMENTO_TOTAL  = 200.0  # % teto global após teto por ativo B

    ativas_mask = sizes["status_sizing"] == "ativa"

    # Extrai ativo B do id
    ID_RE_B = re.compile(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr[\d.]+_h\d+(?:_m\d+)?$')
    def _b(est_id):
        m = ID_RE_B.match(str(est_id))
        return m.group(2) if m else "__"

    sizes["_ativo_b"] = sizes["id"].apply(_b)

    # Desfaz o reescalamento original (que comprimiu tudo para 100%)
    fator_orig = float(sizes["fator_escala"].iloc[0]) if "fator_escala" in sizes.columns else 1.0
    if fator_orig > 0 and fator_orig < 1.0:
        # Restaura o size pré-reescalamento, limitado pelo cap individual (10%)
        sizes.loc[ativas_mask, "size_final_pct"] = (
            sizes.loc[ativas_mask, "size_final_pct"] / fator_orig
        ).clip(upper=10.0)

    # Aplica teto por ativo B proporcionalmente dentro de cada grupo
    for ativo_b, grp in sizes[ativas_mask].groupby("_ativo_b"):
        exp_b = grp["size_final_pct"].sum()
        if exp_b > TETO_POR_ATIVO_B:
            fator_b = TETO_POR_ATIVO_B / exp_b
            sizes.loc[grp.index, "size_final_pct"] *= fator_b

    sizes["size_final_pct"] = sizes["size_final_pct"].round(4)

    # Reaplica filtro de mínimo após redistribuição
    SIZE_MIN = 0.02
    mask_min = ativas_mask & (sizes["size_final_pct"] < SIZE_MIN)
    sizes.loc[mask_min, "status_sizing"] = "abaixo_minimo"
    sizes.loc[mask_min, "size_final_pct"] = 0.0

    # Teto global (safety net)
    ativas_mask = sizes["status_sizing"] == "ativa"
    exp_total = sizes.loc[ativas_mask, "size_final_pct"].sum()
    if exp_total > ORCAMENTO_TOTAL:
        fator_global = ORCAMENTO_TOTAL / exp_total
        sizes.loc[ativas_mask, "size_final_pct"] *= fator_global
        sizes["size_final_pct"] = sizes["size_final_pct"].round(4)

    # ── Teto de exposição EFETIVA (exposicao x tempo em posicao) ─────────────
    # Esta funcao refaz o sizing por conta propria (teto por ativo B e
    # orcamento nominal), entao precisa reaplicar tambem o teto efetivo do
    # position_sizing.py — senao o controle que impede holdings longos de
    # alavancarem o portfolio seria descartado justamente aqui.
    ORCAMENTO_EFETIVO = CFG.get("orcamento_efetivo_pct", 35.0)
    ocupacao = sizer._calcular_ocupacao(sizes)
    sizes["ocupacao"] = ocupacao
    ativas_mask = sizes["status_sizing"] == "ativa"
    exp_efetiva = (sizes.loc[ativas_mask, "size_final_pct"] * ocupacao[ativas_mask]).sum()
    if exp_efetiva > ORCAMENTO_EFETIVO and exp_efetiva > 0:
        fator_ef = ORCAMENTO_EFETIVO / exp_efetiva
        sizes.loc[ativas_mask, "size_final_pct"] *= fator_ef
        sizes["size_final_pct"] = sizes["size_final_pct"].round(4)
        print(f"  Exposicao efetiva {exp_efetiva:.1f}% > teto {ORCAMENTO_EFETIVO:.0f}% "
              f"-> reescalado por {fator_ef:.4f}")
    sizes["exposicao_efetiva_pct"] = sizes["size_final_pct"] * sizes["ocupacao"]

    # Relatório
    ativas_final = sizes[sizes["status_sizing"] == "ativa"]
    exp_final = ativas_final["size_final_pct"].sum()
    print(f"\n  [Teto por ativo B = {TETO_POR_ATIVO_B}% | Orçamento = {ORCAMENTO_TOTAL}%]")
    print(f"  Exposição total ajustada : {exp_final:.1f}% (nominal)")
    print(f"  Exposição efetiva        : {ativas_final['exposicao_efetiva_pct'].sum():.1f}% "
          f"(teto {ORCAMENTO_EFETIVO:.0f}%) | ocupacao media {ativas_final['ocupacao'].mean():.1%}")
    print(f"  Size mediano             : {ativas_final['size_final_pct'].median():.3f}%")
    print(f"  Estratégias ativas       : {len(ativas_final)}")

    # Top concentrações por ativo B
    conc = (
        ativas_final.groupby("_ativo_b")["size_final_pct"]
        .sum().sort_values(ascending=False).head(8)
    )
    print("  Top ativos B por exposição:")
    for ab, exp in conc.items():
        n = (ativas_final["_ativo_b"] == ab).sum()
        print(f"    {ab:<40} {exp:.2f}%  ({n} estratégias)")

    sizes.to_csv(os.path.join(REPORTS, "position_sizes.csv"), index=False)
    print(f"  position_sizes.csv salvo ({time.time()-t0:.1f}s)")
    return sizes


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULAÇÃO CORRIGIDA
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXTRAÇÃO DE TRADES HISTÓRICOS (para calibração de regime)
# ═══════════════════════════════════════════════════════════════════════════════

_ID_RE = re.compile(
    r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr([\d.]+)_h(\d+)(?:_m(\d+))?$'
)


def _sinal_da_estrategia(p: dict, ra_s: pd.Series) -> pd.Series:
    """
    Série booleana de dias em que a estratégia dispara.

    Tipos 1-3: movimento diário de A acima do threshold.
    Tipo 4 (mom > 0): momentum acumulado dos últimos N meses — precisa
    ser tratado à parte porque seu threshold é 0. Sem isso, `ra_s > 0.0`
    dispararia em ~metade de todos os dias, o que não é o sinal e ainda
    gera exposição absurda quando combinado com holding de 10-42 dias.
    """
    mom = p.get("mom", 0)
    if mom > 0:
        janela = mom * 21
        preco  = (1 + ra_s / 100).cumprod()
        # shift(1): o sinal do dia t usa apenas dados até t-1 (sem lookahead)
        mom_acum = preco.pct_change(janela).shift(1)
        base = mom_acum > 0 if p["direcao"] == "sobe" else mom_acum < 0
        return base.fillna(False)
    return ra_s > p["thr"] if p["direcao"] == "sobe" else ra_s < -p["thr"]


def _datas_sinal_sem_sobreposicao(sinal: pd.Series, idx: pd.Index, h: int,
                                   bloquear: bool):
    """
    Datas em que a estratégia de fato ABRE posição.

    Com `bloquear=True` (tipo 4), só abre nova posição depois que a
    anterior fechou. Sinais lentos ficam ligados por semanas seguidas;
    sem o bloqueio, uma posição nova por dia com holding de N dias
    equivale a manter ~N posições simultâneas da mesma estratégia —
    alavancagem que não existiria na prática (foi o que produziu um
    retorno de 10^21% no primeiro run overnight com o tipo 4).
    """
    datas = sinal[sinal].index
    if not bloquear:
        return datas
    saida, livre_em = [], -1
    for dt in datas:
        pos = idx.get_loc(dt)
        if pos < livre_em:
            continue
        saida.append(dt)
        livre_em = pos + 1 + h
    return pd.DatetimeIndex(saida)

def extrair_trades_historicos(
    sizes_df: pd.DataFrame,
    retornos: pd.DataFrame,
    custo: float = 0.15,
) -> pd.DataFrame:
    """
    Extrai todos os trades históricos das estratégias ativas sem aplicar
    fatores de regime. Retorna DataFrame usado pela calibração.
    """
    ativas = sizes_df[sizes_df["status_sizing"] == "ativa"]
    trades = []
    n_skip = 0

    print(f"\nExtraindo trades historicos ({len(ativas)} estrategias)...")
    t0 = time.time()

    for i, (_, row) in enumerate(ativas.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(ativas)}  |  {time.time()-t0:.0f}s", end="\r")

        p = _parsear_id(row["id"])
        if p is None:
            n_skip += 1
            continue

        a, b    = p["a"], p["b"]
        direcao = p["direcao"]
        tipo    = p["tipo"]
        thr     = p["thr"]
        h       = p["h"]

        if a not in retornos.columns or b not in retornos.columns:
            n_skip += 1
            continue

        categoria = classificar_ativo(b)
        ra  = retornos[a].dropna()
        rb  = retornos[b].dropna()
        idx = ra.index.intersection(rb.index)

        if len(idx) < 30:
            n_skip += 1
            continue

        ra_s   = ra[idx]
        rb_s   = rb[idx]
        sinal  = _sinal_da_estrategia(p, ra_s)
        long_b = (direcao == "sobe") != (tipo == "CTR")

        for dt in _datas_sinal_sem_sobreposicao(sinal, idx, h, p["mom"] > 0):
            pos  = idx.get_loc(dt)
            prox = idx[pos + 1 : pos + 1 + h]
            if len(prox) == 0:
                continue
            ret = float(rb_s[prox].sum())
            if not long_b:
                ret = -ret
            trades.append({
                "data_entrada": dt,
                "ativo_b":      b,
                "categoria_b":  categoria,
                "retorno":      ret - _custo_ativo(b, custo),
            })

    print(f"\n  {len(trades):,} trades extraidos, {n_skip} estrategias ignoradas "
          f"({time.time()-t0:.1f}s)")
    return pd.DataFrame(trades)


def _parsear_id(est_id: str) -> dict | None:
    m = _ID_RE.match(est_id)
    if not m:
        return None
    return {
        "a":       m.group(1),
        "b":       m.group(2),
        "direcao": m.group(3),
        "tipo":    m.group(4),
        "thr":     float(m.group(5)),
        "h":       int(m.group(6)),
        "mom":     int(m.group(7)) if m.group(7) else 0,
    }


def _custo_ativo(ativo_b: str, custo_base: float = 0.15) -> float:
    """Retorna custo por trade: 0.30% para ativos BR (.SA), custo_base para os demais."""
    eh_br = ativo_b.endswith(".SA") or ativo_b in {
        "Guepardo Institucional Master FIA",
        "Alaska Black Master FIF Acoes",
        "Constellation Compounders Master FIA",
    }
    return 0.30 if eh_br else custo_base


def _slippage_ativo(ativo_b: str, slippage_extra) -> float:
    """Slippage overnight adicional por tipo de ativo.
    Aceita float fixo ou dict {"br": X, "global": Y} (ver SLIPPAGE_CENARIOS).
    """
    if not slippage_extra:
        return 0.0
    if isinstance(slippage_extra, (int, float)):
        return float(slippage_extra)
    eh_br = ativo_b.endswith(".SA")
    return float(slippage_extra.get("br" if eh_br else "global", 0.0))


def simular_portfolio(
    sizes_df: pd.DataFrame,
    retornos: pd.DataFrame,
    capital: float,
    custo: float = 0.15,
    fatores_regime: pd.DataFrame = None,
    cdi_df: pd.DataFrame = None,
    slippage_extra=0.0,   # float ou dict {"br": X, "global": Y} — ver SLIPPAGE_CENARIOS
) -> pd.DataFrame:
    """
    Reconstrói a curva de capital do portfólio com direção correta de posição.

    Regra de direção (long_b = True significa comprar B):
      MOM + sobe  →  long  B   (momentum: B sobe com A)
      MOM + cai   →  short B   (momentum: B cai com A)
      CTR + sobe  →  short B   (contrarian: B nao sobe junto)
      CTR + cai   →  long  B   (contrarian: B nao cai junto)

    Equivale a: long_b = (direcao == "sobe") XOR (tipo == "CTR")
    """
    # ── Parâmetros de vol targeting (mesmos do CFG em position_sizing.py) ──────
    _VOL_ALVO = 1.5
    _VOL_JAN  = 20
    _VOL_MIN  = 0.3
    _VOL_MAX  = 2.0

    ativas = sizes_df[sizes_df["status_sizing"] == "ativa"].copy()
    print(f"\nSimulando {len(ativas)} estrategias ativas...")
    t0 = time.time()

    # Pré-computa série de vol rolling para cada ativo B único.
    # Isso permite aplicar vol_adj na DATA DO SINAL, sem lookahead:
    # em vez de usar vol dos últimos 20 dias do fim da série (2026),
    # cada trade usará a vol dos 20 dias anteriores à data do sinal.
    ativos_b_unicos = {
        _parsear_id(r["id"])["b"]
        for _, r in ativas.iterrows()
        if _parsear_id(r["id"]) is not None
    }
    vol_rolling_b = {
        b: retornos[b].rolling(_VOL_JAN, min_periods=_VOL_JAN // 2).std()
        for b in ativos_b_unicos if b in retornos.columns
    }
    print(f"  Vol rolling pre-computada para {len(vol_rolling_b)} ativos B")

    curvas      = []
    pos_periods = []  # (entry_date, exit_date, size) para CDI overlay
    _entries_br_dict:     dict = {}  # signal_date → Σ size_eff entrando em ativos BR
    _entries_global_dict: dict = {}  # signal_date → Σ size_eff entrando em ativos globais
    n_ok   = 0
    n_skip = 0

    for i, (_, row) in enumerate(ativas.iterrows(), 1):
        if i % 50 == 0:
            print(f"  {i}/{len(ativas)}  |  {time.time()-t0:.0f}s", end="\r")

        p = _parsear_id(row["id"])
        if p is None:
            n_skip += 1
            continue

        a, b = p["a"], p["b"]
        if a not in retornos.columns or b not in retornos.columns:
            n_skip += 1
            continue

        # Size base sem vol_adj: remove o vol estático calculado com dados do fim
        # da série, para reaplicá-lo dinamicamente na data de cada sinal.
        static_vol_adj = float(row.get("vol_adj", 1.0) or 1.0)
        if static_vol_adj <= 0:
            static_vol_adj = 1.0
        size_base = row.get("size_regime_pct", row["size_final_pct"]) / static_vol_adj / 100
        thr     = p["thr"]
        h       = p["h"]
        direcao = p["direcao"]
        tipo    = p["tipo"]

        # long_b: True = compra B, False = vende B
        long_b = (direcao == "sobe") != (tipo == "CTR")
        # MOM+sobe→True, MOM+cai→False, CTR+sobe→False, CTR+cai→True

        ra = retornos[a].dropna()
        rb = retornos[b].dropna()
        idx = ra.index.intersection(rb.index)
        if len(idx) < 30:
            n_skip += 1
            continue

        ra_s = ra[idx]
        rb_s = rb[idx]

        sinal = _sinal_da_estrategia(p, ra_s)

        dias_sinal = _datas_sinal_sem_sobreposicao(sinal, idx, h, p["mom"] > 0)
        if len(dias_sinal) < 5:
            n_skip += 1
            continue

        ret_strat = []
        datas     = []
        idx_arr   = idx  # DatetimeIndex

        # Série de vol rolling para este ativo B (lookup por data no loop)
        _vs = vol_rolling_b.get(b)

        for dt in dias_sinal:
            pos = idx_arr.get_loc(dt)
            prox = idx_arr[pos + 1 : pos + 1 + h]
            if len(prox) == 0:
                continue
            ret = float(rb_s[prox].sum())
            if not long_b:
                ret = -ret
            fator_reg = 1.0
            if fatores_regime is not None and b in fatores_regime.columns:
                if dt in fatores_regime.index:
                    fator_reg = float(fatores_regime.at[dt, b])
                else:
                    prev = fatores_regime.index[fatores_regime.index <= dt]
                    if len(prev) > 0:
                        fator_reg = float(fatores_regime.at[prev[-1], b])

            # Vol dinâmico na data do sinal: sem lookahead
            if _vs is not None and dt in _vs.index:
                _vol_dt = float(_vs.at[dt])
                vol_adj_din = float(np.clip(_VOL_ALVO / _vol_dt, _VOL_MIN, _VOL_MAX)) \
                              if not np.isnan(_vol_dt) and _vol_dt > 0 else 1.0
            else:
                vol_adj_din = 1.0

            size_eff     = size_base * vol_adj_din
            custo_b      = _custo_ativo(b, custo) + _slippage_ativo(b, slippage_extra)
            ret_ajustado = ret - custo_b
            ret_strat.append(ret_ajustado * size_eff * fator_reg)
            datas.append(prox[-1])
            pos_periods.append((dt, prox[-1], size_eff))
            if b.endswith(".SA"):
                _entries_br_dict[dt] = _entries_br_dict.get(dt, 0.0) + size_eff
            else:
                _entries_global_dict[dt] = _entries_global_dict.get(dt, 0.0) + size_eff

        if len(ret_strat) < 5:
            n_skip += 1
            continue

        serie = pd.Series(ret_strat, index=pd.DatetimeIndex(datas))
        serie = serie.groupby(level=0).sum()  # agrega trades com mesma data de saída
        curvas.append(serie)
        n_ok += 1

    print(f"\n  Concluido: {n_ok} estrategias simuladas, {n_skip} ignoradas ({time.time()-t0:.1f}s)")

    if not curvas:
        return pd.DataFrame()

    df_curvas     = pd.concat(curvas, axis=1).fillna(0)
    ret_portfolio = df_curvas.sum(axis=1).sort_index()

    # ── CDI Overlay: capital ocioso rende CDI ────────────────────────────────
    # Usa a técnica de diferenças para calcular capital deployado por dia em O(n_trades).
    if cdi_df is not None and pos_periods:
        cdi_col   = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
        cdi_daily = cdi_df[cdi_col].pct_change().dropna()  # taxa diária como fração

        all_days   = ret_portfolio.index
        deploy_arr = np.zeros(len(all_days))

        for entry, exit_d, sz in pos_periods:
            i_open  = all_days.searchsorted(entry,  side="right")
            i_close = all_days.searchsorted(exit_d, side="right")
            if i_open  < len(all_days): deploy_arr[i_open]  += sz
            if i_close < len(all_days): deploy_arr[i_close] -= sz

        daily_deployed = pd.Series(deploy_arr, index=all_days).cumsum().clip(lower=0)
        idle_frac      = (1.0 - daily_deployed).clip(lower=0.0, upper=1.0)
        cdi_overlay    = idle_frac * cdi_daily.reindex(all_days).fillna(0) * 100

        idle_medio = float(idle_frac.mean())
        print(f"  Capital ocioso medio : {idle_medio:.1%}  →  CDI overlay ativo")
    else:
        cdi_overlay    = pd.Series(0.0, index=ret_portfolio.index)
        daily_deployed = pd.Series(0.0, index=ret_portfolio.index)

    capital_serie = (1 + ret_portfolio / 100).cumprod() * capital

    roll_max  = capital_serie.cummax()
    dd        = (capital_serie - roll_max) / roll_max * 100
    n_trades  = int(df_curvas.astype(bool).sum().sum())
    ret_total = float(capital_serie.iloc[-1] / capital - 1) * 100
    vol_d     = float(ret_portfolio.std())
    sharpe    = float(ret_portfolio.mean() / vol_d * np.sqrt(252)) if vol_d > 0 else 0

    print(f"\n{'='*62}")
    print(f"  SIMULACAO DO PORTFOLIO ({n_ok} estrategias)")
    print(f"{'='*62}")
    print(f"  Periodo       : {capital_serie.index[0].date()} - {capital_serie.index[-1].date()}")
    print(f"  Retorno total : {ret_total:+.1f}%")
    print(f"  Sharpe        : {sharpe:.2f}")
    print(f"  Drawdown max  : {float(dd.min()):.1f}%")
    print(f"  Trades        : {n_trades:,}")
    print(f"{'='*62}")

    entries_br_s  = pd.Series(_entries_br_dict).reindex(capital_serie.index).fillna(0)
    entries_glo_s = pd.Series(_entries_global_dict).reindex(capital_serie.index).fillna(0)

    return pd.DataFrame({
        "data":               capital_serie.index,
        "capital_portfolio":  capital_serie.values,
        "retorno_diario_pct": ret_portfolio.reindex(capital_serie.index).fillna(0).values,
        "cdi_overlay_pct":    cdi_overlay.reindex(capital_serie.index).fillna(0).values,
        "daily_deployed":     daily_deployed.reindex(capital_serie.index).fillna(0).values,
        "drawdown_pct":       dd.values,
        "entries_br_pct":     entries_br_s.values,
        "entries_global_pct": entries_glo_s.values,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

def aplicar_circuit_breaker(
    ret: pd.Series,
    threshold_dd: float = -0.08,
    fator_reducao: float = 0.65,
    dias_recuperacao: int = 5,
) -> tuple:
    """
    Circuit breaker no nivel do portfolio.

    Quando o drawdown acumulado ultrapassa |threshold_dd|, escala os retornos
    por fator_reducao. Reativa apos dias_recuperacao dias com drawdown
    acima de threshold/2 (ou seja, jah recuperado a metade da perda).

    Parametros calibrados em 2026-09: auditoria encontrou apenas 1 ativacao
    real em 10 anos de backtest (12/03/2020, no fundo do panico da COVID) —
    nos 20 dias uteis seguintes o sinal puro subiu +15,6%, ou seja, o corte
    de exposicao custou parte de uma recuperacao em V. Como e uma amostra de
    1 evento, nao da pra calibrar com precisao — a mudanca de 0.50->0.65 e
    10->5 dias eh deliberadamente conservadora (ainda reduz exposicao de
    verdade em crise, so reengata mais rapido) em vez de tentar otimizar
    para esse unico caso historico.

    Retorna (ret_ajustado, n_dias_em_corte, mask_corte).
    """
    ret_adj      = ret.copy().astype(float)
    capital      = 100.0
    peak         = 100.0
    em_corte     = False
    dias_corte   = 0
    n_dias_corte = 0
    em_corte_arr = np.zeros(len(ret), dtype=bool)

    for i in range(len(ret)):
        r = float(ret.iloc[i])

        if em_corte:
            em_corte_arr[i] = True
            ret_adj.iloc[i] = r * fator_reducao
            r_eff = r * fator_reducao
            dias_corte   += 1
            n_dias_corte += 1
        else:
            r_eff      = r
            dias_corte = 0

        capital = capital * (1 + r_eff / 100)
        peak    = max(peak, capital)
        dd      = (capital - peak) / peak

        if not em_corte and dd < threshold_dd:
            em_corte   = True
            dias_corte = 0

        if em_corte and dias_corte >= dias_recuperacao and dd > threshold_dd / 2:
            em_corte = False

    mask_corte = pd.Series(em_corte_arr, index=ret.index)
    return ret_adj, n_dias_corte, mask_corte


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TREND FOLLOWING
# ═══════════════════════════════════════════════════════════════════════════════

def simular_trend_following(
    retornos:         pd.DataFrame,
    capital_fraction: float = 0.30,
    sma_rapida:       int   = 50,
    sma_lenta:        int   = 200,
    custo:            float = 0.10,
    permitir_short:   bool  = True,
) -> pd.Series:
    """
    Trend following com SMA crossover em indices/ETFs principais.

    Long quando SMA_rapida > SMA_lenta. Se permitir_short=True (default),
    SHORT quando SMA_rapida < SMA_lenta, em vez de só ficar flat.

    Por que mudar de long-only para long/short (2026-09): a versão
    long-only só conseguia, na melhor das hipoteses, ficar de fora de uma
    queda (retorno zero) — nunca lucrar com ela. Isso foi identificado como
    uma das causas dos períodos em que o portfolio geral ficou abaixo do
    CDI (a fatia de trend following não protegia, só "não ajudava" durante
    tendências de baixa sustentadas). Com short habilitado, uma tendência
    de baixa clara (ex: 2022) passa a contribuir positivamente em vez de
    zerar.

    Simplificação assumida: os ativos (ETFs líquidos SPY/QQQ/TLT/GLD/IWM/
    EFA) são vendíveis a descoberto sem custo de aluguel extra. ^BVSP é um
    índice, não um ativo negociável — na prática esse short exigiria um
    proxy (ex: BOVA11 inverso ou futuro de índice); o backtest trata como
    se fosse direto, o que é uma simplificação otimista para essa perna.

    Retorna serie de retornos diarios nas mesmas unidades que ret_portfolio.
    """
    candidatos  = ["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA", "^BVSP"]
    disponiveis = [a for a in candidatos if a in retornos.columns]

    if not disponiveis:
        print("  Trend following: nenhum ativo disponivel.")
        return pd.Series(dtype=float)

    size_por_ativo = capital_fraction / len(disponiveis)  # fracao do capital
    curvas = []
    pcts_long, pcts_short = [], []

    for ativo in disponiveis:
        ret    = retornos[ativo].fillna(0)
        # retornos vem em formato percentual (1.5 = 1.5%) — dividir por 100
        # antes de compor. Sem isso, um dia de -1.26 vira fator (1-1.26) =
        # -0.26 (preco negativo), e a serie de "preco" reconstruida oscila
        # de sinal a cada retorno negativo grande, destruindo qualquer
        # sinal de tendencia real (confirmado: SPY terminava em ~1e-17
        # em vez de ~4.2x, o crescimento real do periodo).
        prices = (1 + ret / 100).cumprod()
        sma_r  = prices.rolling(sma_rapida, min_periods=sma_rapida // 2).mean()
        sma_l  = prices.rolling(sma_lenta,  min_periods=sma_lenta  // 2).mean()

        if permitir_short:
            signal = np.sign(sma_r - sma_l).fillna(0)   # +1 long, -1 short, 0 indefinido
        else:
            signal = (sma_r > sma_l).astype(float)      # 1 long, 0 flat (comportamento antigo)
        sig_lag = signal.shift(1).fillna(0)  # evita lookahead

        # Custo nas mudancas de posicao
        mudancas = (sig_lag != sig_lag.shift(1)).fillna(False).astype(float)
        ret_tf   = (sig_lag * ret - mudancas * custo) * size_por_ativo

        curvas.append(ret_tf)
        pcts_long.append(float((signal > 0).mean()))
        pcts_short.append(float((signal < 0).mean()))

    ret_tf_total = pd.concat(curvas, axis=1).fillna(0).sum(axis=1).sort_index()

    pct_tempo_long  = float(np.mean(pcts_long))  if pcts_long  else 0.0
    pct_tempo_short = float(np.mean(pcts_short)) if pcts_short else 0.0

    print(f"\n  Trend following: {len(disponiveis)} ativos "
          f"({', '.join(disponiveis)})  |  short {'habilitado' if permitir_short else 'desabilitado'}")
    print(f"  Capital alocado: {capital_fraction:.0%}  |  "
          f"% tempo long: {pct_tempo_long:.0%}  |  % tempo short: {pct_tempo_short:.0%}")

    return ret_tf_total


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CDI BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_cdi(cdi_df, data_inicio, data_fim, capital) -> pd.Series:
    mask = (cdi_df.index >= data_inicio) & (cdi_df.index <= data_fim)
    cdi_p = cdi_df.loc[mask, "CDI"]
    if cdi_p.empty:
        return pd.Series(dtype=float)
    return (cdi_p / cdi_p.iloc[0] * capital).rename("CDI")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_metricas(capital_serie: pd.Series, capital_inicial: float, nome: str) -> dict:
    if len(capital_serie) < 10:
        return {"nome": nome}

    # Usa span de calendário real para CAGR (evita distorção por lacunas na série)
    idx = pd.DatetimeIndex(capital_serie.index)
    n_anos_cal = (idx[-1] - idx[0]).days / 365.25

    ret_d    = capital_serie.pct_change().dropna()
    n_anos   = max(n_anos_cal, 0.01)
    cagr     = (capital_serie.iloc[-1] / capital_inicial) ** (1 / n_anos) - 1

    # Para Sharpe: resample para frequência diária (forward-fill) antes de calcular
    capital_diario = capital_serie.resample("B").last().ffill()
    ret_d_diario   = capital_diario.pct_change().dropna()
    vol    = ret_d_diario.std() * np.sqrt(252)
    sharpe = ret_d_diario.mean() / ret_d_diario.std() * np.sqrt(252) if ret_d_diario.std() > 0 else 0

    roll_max = capital_serie.cummax()
    dd_max   = float(((capital_serie - roll_max) / roll_max).min())
    ret_tot  = (capital_serie.iloc[-1] / capital_inicial - 1) * 100

    return {
        "nome":            nome,
        "retorno_total_%": round(ret_tot, 2),
        "cagr_%":          round(cagr * 100, 2),
        "vol_anual_%":     round(vol * 100, 2),
        "sharpe":          round(sharpe, 2),
        "drawdown_max_%":  round(dd_max * 100, 2),
        "n_dias":          len(capital_serie),
    }


def imprimir_resumo(met_port: dict, met_cdi: dict):
    print(f"\n{'='*62}")
    print(f"{'PORTFOLIO QUANT  vs  CDI':^62}")
    print(f"{'='*62}")
    print(f"{'Metrica':<30} {'Portfolio':>14} {'CDI':>14}")
    print(f"{'─'*62}")
    campos = [
        ("retorno_total_%", "Retorno total (%)"),
        ("cagr_%",          "CAGR (% a.a.)"),
        ("vol_anual_%",     "Volatilidade anual (%)"),
        ("sharpe",          "Sharpe ratio"),
        ("drawdown_max_%",  "Drawdown maximo (%)"),
    ]
    for key, label in campos:
        vp = met_port.get(key, "—")
        vc = met_cdi.get(key, "—")
        fp = f"{vp:+.2f}" if isinstance(vp, (int, float)) else str(vp)
        fc = f"{vc:+.2f}" if isinstance(vc, (int, float)) else str(vc)
        print(f"{label:<30} {fp:>14} {fc:>14}")
    print(f"{'='*62}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRAFICO
# ═══════════════════════════════════════════════════════════════════════════════

def plotar(port_serie, cdi_serie, sizes_df, met_port, met_cdi, capital) -> str:
    COR_PORT = "#1565C0"
    COR_CDI  = "#E65100"
    COR_DD   = "#B71C1C"

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[3, 1.5, 1.5],
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.91, bottom=0.06,
    )
    ax_eq   = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, :])
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_men  = fig.add_subplot(gs[2, 1])

    # ── Equity ───────────────────────────────────────────────────────────────
    ax_eq.plot(port_serie.index, port_serie / 1000,
               color=COR_PORT, lw=1.8, label="Portfolio Quant", zorder=3)
    if not cdi_serie.empty:
        ax_eq.plot(cdi_serie.index, cdi_serie / 1000,
                   color=COR_CDI, lw=1.8, ls="--", label="CDI", zorder=2)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"R$ {x:.0f}k"))
    ax_eq.set_title("Curva de Capital", fontsize=11, pad=8)
    ax_eq.legend(fontsize=10, loc="upper left")
    ax_eq.grid(True, alpha=0.25)
    ax_eq.set_facecolor("#fafafa")

    ax_eq.annotate(
        f"  Port: {met_port.get('retorno_total_%', 0):+.1f}%  (CAGR {met_port.get('cagr_%', 0):+.1f}%)",
        xy=(port_serie.index[-1], port_serie.iloc[-1] / 1000),
        fontsize=8.5, color=COR_PORT, va="center",
    )
    if not cdi_serie.empty and met_cdi:
        ax_eq.annotate(
            f"  CDI: {met_cdi.get('retorno_total_%', 0):+.1f}%  (CAGR {met_cdi.get('cagr_%', 0):+.1f}%)",
            xy=(cdi_serie.index[-1], cdi_serie.iloc[-1] / 1000),
            fontsize=8.5, color=COR_CDI, va="center",
        )

    # ── Drawdown ──────────────────────────────────────────────────────────────
    roll_max = port_serie.cummax()
    dd_port  = (port_serie - roll_max) / roll_max * 100
    ax_dd.fill_between(dd_port.index, dd_port, 0, color=COR_DD, alpha=0.55, label="Drawdown")
    ax_dd.axhline(0, color="black", lw=0.5)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.set_title("Drawdown do Portfolio", fontsize=11, pad=8)
    ax_dd.legend(fontsize=9, loc="lower right")
    ax_dd.grid(True, alpha=0.25)
    ax_dd.set_facecolor("#fafafa")
    dd_min_idx = dd_port.idxmin()
    ax_dd.annotate(
        f"Max: {dd_port.min():.1f}%",
        xy=(dd_min_idx, dd_port.min()),
        xytext=(10, -15), textcoords="offset points",
        fontsize=8, color=COR_DD,
        arrowprops=dict(arrowstyle="->", color=COR_DD, lw=0.8),
    )

    # ── Distribuição de tamanhos ──────────────────────────────────────────────
    col = "size_final_pct"
    ativas = sizes_df[sizes_df.get("status_sizing", pd.Series()) == "ativa"] \
        if "status_sizing" in sizes_df.columns else pd.DataFrame()

    if not ativas.empty and col in ativas.columns:
        med = ativas[col].median()
        ax_dist.hist(ativas[col], bins=40, color="#43A047", alpha=0.75,
                     edgecolor="white", lw=0.4)
        ax_dist.axvline(med, color="red", ls="--", lw=1.4,
                        label=f"Mediana: {med:.2f}%")
        ax_dist.set_xlabel(
            f"Tamanho (% capital)  |  Exposicao total: {ativas[col].sum():.0f}%",
            fontsize=8.5,
        )
        ax_dist.set_ylabel("No de estrategias", fontsize=9)
        ax_dist.set_title("Distribuicao de Tamanhos", fontsize=11, pad=8)
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.25, axis="y")
        ax_dist.set_facecolor("#fafafa")

    # ── Retorno mensal ────────────────────────────────────────────────────────
    ret_men = (port_serie.pct_change().dropna() * 100).resample("ME").sum()
    cores = ["#2E7D32" if r >= 0 else "#C62828" for r in ret_men]
    ax_men.bar(ret_men.index, ret_men, color=cores, width=20, edgecolor="white", lw=0.3)
    ax_men.axhline(0, color="black", lw=0.5)
    ax_men.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_men.set_title("Retorno Mensal Agregado", fontsize=11, pad=8)
    ax_men.grid(True, alpha=0.25, axis="y")
    ax_men.set_facecolor("#fafafa")
    ax_men.legend(handles=[
        Patch(facecolor="#2E7D32", label="Positivo"),
        Patch(facecolor="#C62828", label="Negativo"),
    ], fontsize=8, loc="lower right")

    # ── Título ────────────────────────────────────────────────────────────────
    fig.suptitle("Portfolio Quant  vs  CDI", fontsize=15, fontweight="bold", y=0.975)
    n_est = len(ativas) if not ativas.empty else "—"
    fig.text(
        0.5, 0.948,
        f"Capital: R$ {capital:,.0f}  |  "
        f"Sharpe: {met_port.get('sharpe', 0):.2f}  |  "
        f"CAGR: {met_port.get('cagr_%', 0):+.1f}%  |  "
        f"Max DD: {met_port.get('drawdown_max_%', 0):.1f}%  |  "
        f"{n_est} estrategias ativas",
        ha="center", fontsize=9, color="#444444", style="italic",
    )

    caminho = os.path.join(REPORTS, "portfolio_vs_cdi.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return caminho


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global REPORTS
    import argparse
    import sys as _sys
    _sys.path.append(BASE)
    from modo_teste import adicionar_flag, banner

    parser = argparse.ArgumentParser(description="Simulacao portfolio vs CDI")
    adicionar_flag(parser)
    parser.add_argument("--capital", type=float, default=CFG["capital_inicial"])
    parser.add_argument("--usar-grafo", action="store_true",
                        help="Ajusta o sizing por centralidade no grafo de correlação "
                             "antes de simular (Fase 6, opcional — ver graph_score.py)")
    args = parser.parse_args()
    capital = args.capital

    if args.teste:
        REPORTS = os.path.join(BASE, "reports", "teste")
    banner("portfolio_vs_cdi.py", args.teste)

    retornos, cdi_df, monitor, exp_df = carregar_dados()

    sizes_df = calcular_sizes(retornos, monitor, exp_df, capital)

    if args.usar_grafo:
        from graph_score import montar_grafo, calcular_centralidade, aplicar_em_sizes
        caminho_padroes = os.path.join(REPORTS, "padroes.csv")
        if os.path.exists(caminho_padroes):
            padroes_df = pd.read_csv(caminho_padroes)
            G = montar_grafo(padroes_df)
            cent = calcular_centralidade(G)
            sizes_df = aplicar_em_sizes(sizes_df, cent, col_size="size_final_pct")
            print(f"\n🕸️  Score estrutural aplicado ({G.number_of_nodes()} ativos, "
                  f"{G.number_of_edges()} conexões) antes da simulação")
        else:
            print(f"\n⚠️  --usar-grafo pedido mas {caminho_padroes} não encontrado — ignorado")

    # Corte de treino: walk-forward foi treinado até 2022-12-31.
    # Calibrar regime apenas com trades pré-2023 elimina o lookahead
    # que inflava artificialmente o Sharpe da camada de regime.
    CORTE_TREINO = pd.Timestamp("2023-01-01")

    trades_df     = extrair_trades_historicos(sizes_df, retornos)
    trades_treino = trades_df[
        pd.to_datetime(trades_df["data_entrada"]) < CORTE_TREINO
    ].copy()

    detector     = MacroRegimeDetector(retornos)
    serie_regime = detector.classificar_serie()
    print(f"\nCalibrando fatores de regime — APENAS dados de treino "
          f"({len(trades_treino):,} de {len(trades_df):,} trades, pré-{CORTE_TREINO.date()})...")
    fatores_cal  = calibrar_fatores(trades_treino, serie_regime, verbose=True)

    fatores_regime = integrate_backtest_historico(
        sizes_df, retornos, fatores_calibrados=fatores_cal,
    )

    portfolio_df = simular_portfolio(
        sizes_df, retornos, capital,
        fatores_regime=fatores_regime, cdi_df=cdi_df,
    )
    if portfolio_df.empty:
        print("\nERRO: simulacao retornou portfolio vazio.")
        return

    portfolio_df["data"] = pd.to_datetime(portfolio_df["data"])
    datas      = portfolio_df["data"].values
    ret_sinal  = pd.Series(portfolio_df["retorno_diario_pct"].values, index=datas)
    cdi_overlay = pd.Series(portfolio_df["cdi_overlay_pct"].values,   index=datas)
    entries_br  = pd.Series(portfolio_df["entries_br_pct"].values,     index=datas)
    entries_glo    = pd.Series(portfolio_df["entries_global_pct"].values, index=datas)
    daily_deployed = pd.Series(portfolio_df["daily_deployed"].values,     index=datas)
    _cdi_col       = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
    cdi_daily_raw  = cdi_df[_cdi_col].pct_change().reindex(
        pd.DatetimeIndex(datas)
    ).fillna(0)

    # ── Trend following ────────────────────────────────────────────────────────
    ret_tf_raw = simular_trend_following(retornos, capital_fraction=0.30)
    ret_tf     = ret_tf_raw.reindex(ret_sinal.index).fillna(0)

    # ── Configuracoes a comparar ───────────────────────────────────────────────
    # CB default: -15% (nao otimizado in-sample; -8% era inflado por calibracao no treino)
    ret_cb, n_cb, cb_mask_sinal = aplicar_circuit_breaker(ret_sinal, threshold_dd=-0.15)
    extra_cdi_cb_sinal = (daily_deployed * 0.50 * cdi_daily_raw * 100).where(cb_mask_sinal, 0.0)
    ret_sinal_tf  = ret_sinal + ret_tf
    ret_tudo_cb, _, cb_mask_tudo = aplicar_circuit_breaker(ret_sinal_tf, threshold_dd=-0.15)
    extra_cdi_cb_tudo = (daily_deployed * 0.50 * cdi_daily_raw * 100).where(cb_mask_tudo, 0.0)
    ret_tudo      = ret_tudo_cb + cdi_overlay + extra_cdi_cb_tudo

    configs = {
        "Regime calibrado (base)":    ret_sinal,
        "+ Circuit Breaker (-15%)":   ret_cb + extra_cdi_cb_sinal,
        "+ CDI Overlay":              ret_sinal + cdi_overlay,
        "+ Trend Following (30%)":    ret_sinal_tf,
        "Tudo combinado":             ret_tudo,
    }

    print(f"\n{'='*72}")
    print(f"  {'COMPARACAO DE CONFIGURACOES':^68}")
    print(f"{'='*72}")
    print(f"  {'Configuracao':<35} {'Retorno':>8} {'CAGR':>7} {'Sharpe':>7} {'DD Max':>8}")
    print(f"  {'─'*68}")

    for nome_cfg, ret_cfg in configs.items():
        cap = (1 + ret_cfg / 100).cumprod() * capital
        m   = calcular_metricas(cap, capital, nome_cfg)
        print(f"  {nome_cfg:<35} {m['retorno_total_%']:>7.1f}%  "
              f"{m['cagr_%']:>5.1f}%  {m['sharpe']:>6.2f}  "
              f"{m['drawdown_max_%']:>7.1f}%")

    print(f"{'='*72}")
    if n_cb > 0:
        print(f"  Circuit breaker ativado por {n_cb} dias ({n_cb/len(ret_sinal)*100:.1f}% do periodo)")

    # ── Tabela de sensibilidade: circuit breaker threshold ────────────────────
    print(f"\n{'='*72}")
    print(f"  {'SENSIBILIDADE: CIRCUIT BREAKER THRESHOLD':^68}")
    print(f"{'='*72}")
    print(f"  {'Threshold CB':<20} {'Retorno':>8} {'CAGR':>7} {'Sharpe':>7} {'DD Max':>8} {'% em corte':>10}")
    print(f"  {'─'*68}")
    for cb_thr in CB_THRESHOLDS:
        if cb_thr is None:
            _ret_cb_s  = ret_sinal_tf
            _n_corte   = 0
            _cb_mask_t = pd.Series(False, index=ret_sinal_tf.index)
            _label     = "Sem CB"
        else:
            _ret_cb_s, _n_corte, _cb_mask_t = aplicar_circuit_breaker(ret_sinal_tf, threshold_dd=cb_thr)
            _label = f"{cb_thr*100:.0f}%"
        _extra_cdi = (daily_deployed * 0.50 * cdi_daily_raw * 100).where(_cb_mask_t, 0.0)
        _ret_c = _ret_cb_s + cdi_overlay + _extra_cdi
        _cap_c = (1 + _ret_c / 100).cumprod() * capital
        _m     = calcular_metricas(_cap_c, capital, _label)
        _pct   = _n_corte / len(ret_sinal_tf) * 100 if cb_thr is not None else 0.0
        print(f"  {_label:<20} {_m['retorno_total_%']:>7.1f}%  "
              f"{_m['cagr_%']:>5.1f}%  {_m['sharpe']:>6.2f}  "
              f"{_m['drawdown_max_%']:>7.1f}%  {_pct:>9.1f}%")
    print(f"{'='*72}")

    # ── Tabela de sensibilidade: slippage overnight ───────────────────────────
    # entries_br/glo são a fração do capital entrando em novas posições por dia.
    # Cada cenário deduz adicionalmente: entries_br * slip_br + entries_glo * slip_glo
    # (unidades: fraction * percent = percent de retorno do portfolio, mesmo que ret_tudo)
    print(f"\n{'='*72}")
    print(f"  {'SENSIBILIDADE: SLIPPAGE OVERNIGHT':^68}")
    _br_med  = float(entries_br.mean())
    _glo_med = float(entries_glo.mean())
    print(f"  (entrada media diaria: BR={_br_med*100:.3f}% capital | "
          f"Global={_glo_med*100:.3f}% capital)")
    print(f"{'='*72}")
    print(f"  {'Cenario':<20} {'Slip BR':>7} {'Slip Glo':>9} "
          f"{'Retorno':>8} {'CAGR':>7} {'Sharpe':>7} {'DD Max':>8}")
    print(f"  {'─'*68}")
    for _nome_slip, _slip in SLIPPAGE_CENARIOS.items():
        _slip_cost = (entries_br * _slip["br"] + entries_glo * _slip["global"]) \
                     .reindex(ret_tudo.index).fillna(0)
        _ret_slip  = ret_tudo - _slip_cost
        _cap_slip  = (1 + _ret_slip / 100).cumprod() * capital
        _ms        = calcular_metricas(_cap_slip, capital, _nome_slip)
        print(f"  {_nome_slip:<20} {_slip['br']*100:>6.2f}%  {_slip['global']*100:>7.2f}%  "
              f"{_ms['retorno_total_%']:>7.1f}%  {_ms['cagr_%']:>5.1f}%  "
              f"{_ms['sharpe']:>6.2f}  {_ms['drawdown_max_%']:>7.1f}%")
    print(f"{'='*72}")

    # ── Grafico e CSV com "Tudo combinado" ────────────────────────────────────
    cap_tudo   = (1 + ret_tudo / 100).cumprod() * capital
    port_serie = cap_tudo / cap_tudo.iloc[0] * capital

    data_inicio = pd.Timestamp(port_serie.index[0])
    data_fim    = pd.Timestamp(port_serie.index[-1])
    cdi_serie   = benchmark_cdi(cdi_df, data_inicio, data_fim, capital)

    if not cdi_serie.empty:
        inicio_cdi = cdi_serie.index[0]
        port_sob   = port_serie[port_serie.index >= inicio_cdi]
        port_sob   = port_sob / port_sob.iloc[0] * capital
        cdi_sob    = cdi_serie / cdi_serie.iloc[0] * capital
        met_port   = calcular_metricas(port_sob, capital, "Portfolio (Tudo)")
        met_cdi    = calcular_metricas(cdi_sob,  capital, "CDI")
    else:
        met_port = calcular_metricas(port_serie, capital, "Portfolio (Tudo)")
        met_cdi  = {}

    imprimir_resumo(met_port, met_cdi)

    # ── Análise in-sample vs out-of-sample ────────────────────────────────────
    # Estratégias foram selecionadas com treino até 2022-12-31.
    # Tudo antes de 2023 é in-sample; a partir de 2023 é out-of-sample real.
    CORTE_OOS = pd.Timestamp("2023-01-01")
    port_is  = cap_tudo[cap_tudo.index < CORTE_OOS]
    port_oos = cap_tudo[cap_tudo.index >= CORTE_OOS]

    print(f"\n{'='*72}")
    print(f"  {'ANALISE IN-SAMPLE vs OUT-OF-SAMPLE':^68}")
    print(f"  (treino walk-forward: 2016-2022 | teste: 2023-2026)")
    print(f"{'='*72}")
    print(f"  {'Periodo':<35} {'Retorno':>8} {'CAGR':>7} {'Sharpe':>7} {'DD Max':>8}")
    print(f"  {'─'*68}")

    if len(port_is) > 10:
        met_is = calcular_metricas(port_is, float(port_is.iloc[0]), "In-sample (2016-2022)")
        print(f"  {'In-sample  2016-2022 (treino)':<35} "
              f"{met_is['retorno_total_%']:>7.1f}%  "
              f"{met_is['cagr_%']:>5.1f}%  {met_is['sharpe']:>6.2f}  "
              f"{met_is['drawdown_max_%']:>7.1f}%")

    if len(port_oos) > 10:
        met_oos = calcular_metricas(port_oos, float(port_oos.iloc[0]), "Out-of-sample (2023+)")
        print(f"  {'Out-of-sample  2023+ (teste real)':<35} "
              f"{met_oos['retorno_total_%']:>7.1f}%  "
              f"{met_oos['cagr_%']:>5.1f}%  {met_oos['sharpe']:>6.2f}  "
              f"{met_oos['drawdown_max_%']:>7.1f}%")
    else:
        print("  Out-of-sample: sem dados pos-2023 suficientes")

    print(f"{'='*72}")
    if len(port_is) > 10 and len(port_oos) > 10:
        delta_sharpe = met_oos["sharpe"] - met_is["sharpe"]
        delta_cagr   = met_oos["cagr_%"] - met_is["cagr_%"]
        nota = ("✅ performance out-of-sample consistente"
                if met_oos["sharpe"] >= 0.5 else
                "⚠️  degradacao significativa fora da amostra")
        print(f"  Delta Sharpe (OOS - IS): {delta_sharpe:+.2f}  |  "
              f"Delta CAGR: {delta_cagr:+.1f}%  →  {nota}")
        print(f"{'='*72}")

    # Salva CSV
    out = pd.DataFrame({
        "data":               datas,
        "retorno_sinal_pct":  ret_sinal.values,
        "retorno_tudo_pct":   ret_tudo.values,
        "capital_tudo":       cap_tudo.values,
        "cdi_overlay_pct":    cdi_overlay.values,
        "ret_tf_pct":         ret_tf.values,
    })
    if not cdi_serie.empty:
        out["capital_cdi"] = cdi_serie.reindex(datas).values
    out.to_csv(os.path.join(REPORTS, "portfolio_vs_cdi.csv"), index=False)
    print(f"\n  portfolio_vs_cdi.csv salvo")

    # Grafico
    caminho_png = plotar(port_serie, cdi_serie, sizes_df, met_port, met_cdi, capital)
    print(f"  Grafico salvo: {caminho_png}")
    os.startfile(caminho_png)


if __name__ == "__main__":
    main()