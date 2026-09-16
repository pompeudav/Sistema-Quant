"""
paper_trading.py
----------------
Paper trading: registra sinais diários e acompanha resultados reais.

Modo atual (sem API de corretora):
  - Usa retornos.csv atualizado pelo collector (dados reais do yfinance)
  - Detecta quais estratégias disparariam sinal hoje (após fechamento)
  - Registra em sinais_log.csv com status "aberto"
  - No dia t+holding, preenche o retorno real e fecha a posição
  - Acumula P&L virtual como se você tivesse executado

Quando a API da corretora for integrada (Fase 5.5+):
  - Adicionar função executar_ordem(sinal) que envia a ordem real
  - O log continua existindo, mas o retorno "realizado" vem do preço
    efetivo de execução (não do yfinance)
  - Custo passa a incluir corretagem real + slippage medido

Regra de timing (alinhada ao backtester):
  - Dia t  : A fecha com retorno > threshold → sinal detectado
  - Dias t+1 a t+h: posição em B aberta
  - Retorno = soma dos retornos diários de B de t+1 a t+h (fechamentos)

Arquivos gerados:
  reports/sinais_log.csv              — todos os sinais (abertos e fechados)
  reports/paper_trading_report.txt    — relatório diário em texto

Uso:
  python analysis/paper_trading.py                   # processa hoje
  python analysis/paper_trading.py --data 2026-05-15 # processa data específica
  python analysis/paper_trading.py --dry-run         # sem salvar, só mostra
  python analysis/paper_trading.py --relatorio       # só imprime relatório atual
"""

import sys
import os
import re
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto, relativo a este arquivo
REPORTS = os.path.join(BASE, "reports")
DATA    = os.path.join(BASE, "data", "raw")

sys.path.insert(0, BASE)

_ID_RE = re.compile(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr([\d\.]+)_h(\d+)(?:_m(\d+))?$')

# Custo realista por tipo de ativo (% do nocional por trade, entrada + saída)
CUSTO_BR_ILIQUIDO = 0.30   # ativos .SA e fundos CVM (spread maior, liquidez menor)
CUSTO_GLOBAL      = 0.15   # ETFs e índices globais (SPY, QQQ, ^VIX, etc.)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_id(est_id: str) -> dict | None:
    m = _ID_RE.match(str(est_id))
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


def _custo_ativo(ativo_b: str) -> float:
    """Retorna o custo por trade conforme o tipo de ativo."""
    eh_br = ativo_b.endswith(".SA") or ativo_b in {
        "Guepardo Institucional Master FIA",
        "Alaska Black Master FIF Ações",
        "Constellation Compounders Master FIA",
    }
    return CUSTO_BR_ILIQUIDO if eh_br else CUSTO_GLOBAL


def _proximo_dia_util(idx: pd.DatetimeIndex, data: pd.Timestamp) -> pd.Timestamp | None:
    """Retorna o próximo dia de negociação após `data` no índice."""
    pos = idx.searchsorted(data, side="right")
    return idx[pos] if pos < len(idx) else None


def _nth_dia_util(idx: pd.DatetimeIndex, data: pd.Timestamp, n: int) -> pd.Timestamp | None:
    """Retorna o n-ésimo dia de negociação após `data` (n=1 = próximo dia)."""
    pos = idx.searchsorted(data, side="right")
    target = pos + n - 1
    return idx[target] if target < len(idx) else None


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE SINAIS
# ─────────────────────────────────────────────────────────────────────────────

def detectar_sinais(
    retornos: pd.DataFrame,
    monitor: pd.DataFrame,
    data_hoje: pd.Timestamp,
    sizes_df: pd.DataFrame | None = None,
    log_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Verifica quais estratégias disparam sinal no fechamento de `data_hoje`.

    Para cada estratégia ativa no monitor:
      - Tipos 1-3: confere se o retorno de A hoje passou do threshold
      - Tipo 4: confere se o momentum acumulado de N meses está na direção
        esperada, e se não há posição já aberta desta mesma estratégia
      - Se sim, registra o sinal com entrada (t+1) e saída esperada (t+h)

    `log_df` é o sinais_log.csv atual — usado para saber quais estratégias
    já têm posição aberta (relevante só para o tipo 4, cujo sinal fica
    ligado por semanas seguidas).

    Retorna lista de dicts prontos para inserir no sinais_log.csv.
    """
    if data_hoje not in retornos.index:
        return []

    trading_idx = retornos.index
    ativas = monitor[monitor["status"] == "ativa"]
    sinais = []

    # No sinais_log.csv a coluna do id da estratégia chama-se
    # "estrategia_id" (no monitor_status.csv ela chama-se "id").
    ids_abertos = set()
    if (log_df is not None and not log_df.empty
            and "status" in log_df.columns and "estrategia_id" in log_df.columns):
        ids_abertos = set(log_df.loc[log_df["status"] == "aberto", "estrategia_id"])

    def _tem_posicao_aberta(est_id: str) -> bool:
        return est_id in ids_abertos

    # Índice de sizes por id (para busca rápida)
    size_por_id = {}
    if sizes_df is not None and not sizes_df.empty and "id" in sizes_df.columns:
        ativas_sz = sizes_df[sizes_df.get("status_sizing", pd.Series()) == "ativa"] \
            if "status_sizing" in sizes_df.columns else sizes_df
        size_por_id = dict(zip(ativas_sz["id"], ativas_sz["size_final_pct"]))

    for _, row in ativas.iterrows():
        p = _parsear_id(row["id"])
        if p is None:
            continue

        a, b     = p["a"], p["b"]
        direcao  = p["direcao"]
        thr      = p["thr"]
        h        = p["h"]
        long_b   = (direcao == "sobe") != (p["tipo"] == "CTR")

        if a not in retornos.columns or b not in retornos.columns:
            continue

        if p["mom"] > 0:
            # Tipo 4 (momentum): o gatilho é o retorno acumulado dos
            # últimos N meses, não o movimento de hoje. Com thr=0, a regra
            # de threshold abaixo dispararia em qualquer dia de alta.
            janela = p["mom"] * 21
            hist = retornos[a].loc[:data_hoje].dropna()
            if len(hist) < janela + 2:
                continue
            preco = (1 + hist / 100).cumprod()
            # Exclui o próprio dia de hoje do cálculo (usa até ontem), do
            # mesmo modo que o backtest faz com shift(1).
            mom_acum = preco.iloc[-2] / preco.iloc[-2 - janela] - 1
            disparou = (mom_acum > 0) if direcao == "sobe" else (mom_acum < 0)

            # Sinal lento fica ligado por semanas: não reabre posição se já
            # existe uma aberta desta mesma estratégia (senão acumularia
            # uma posição nova por dia durante toda a tendência).
            if disparou and _tem_posicao_aberta(row["id"]):
                continue
        else:
            ret_a = retornos.at[data_hoje, a]
            if pd.isna(ret_a):
                continue
            disparou = (ret_a > thr) if direcao == "sobe" else (ret_a < -thr)

        if not disparou:
            continue

        data_entrada = _proximo_dia_util(trading_idx, data_hoje)
        if data_entrada is None:
            continue

        data_saida = _nth_dia_util(trading_idx, data_hoje, h)

        # Size: usa position_sizes.csv se disponível, senão usa sharpe_hist
        # como proxy de qualidade (size proporcional entre 0.05% e 0.30%)
        if row["id"] in size_por_id:
            size_pct = float(size_por_id[row["id"]])
        else:
            sharpe_ref = max(float(row.get("sharpe_hist", 0.3)), 0.1)
            size_pct = min(sharpe_ref * 0.10, 0.30)

        custo = _custo_ativo(b)

        sinais.append({
            "data_sinal":          data_hoje.strftime("%Y-%m-%d"),
            "estrategia_id":       row["id"],
            "ativo_a":             a,
            "ativo_b":             b,
            "direcao_a":           direcao,
            "long_b":              long_b,
            "threshold":           thr,
            "holding_days":        h,
            "size_pct":            round(size_pct, 4),
            "retorno_a_pct":       round(float(ret_a), 4),
            "data_entrada":        data_entrada.strftime("%Y-%m-%d"),
            "data_saida_esperada": data_saida.strftime("%Y-%m-%d") if data_saida else None,
            "retorno_b_acum_pct":  None,
            "custo_pct":           custo,
            "retorno_liq_pct":     None,
            "contrib_portfolio_%": None,
            "status":              "aberto",
        })

    return sinais


# ─────────────────────────────────────────────────────────────────────────────
# FECHAMENTO DE POSIÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def fechar_posicoes(
    log_df: pd.DataFrame,
    retornos: pd.DataFrame,
    data_hoje: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """
    Fecha posições cujo data_saida_esperada <= data_hoje.

    Calcula o retorno acumulado real de B entre data_entrada e data_saida,
    aplica o custo e registra a contribuição ao portfólio.

    Retorna (log_df atualizado, n_fechados).
    """
    if log_df.empty:
        return log_df, 0

    abertos_mask = log_df["status"] == "aberto"
    if not abertos_mask.any():
        return log_df, 0

    n_fechados = 0

    trading_idx = retornos.index

    for idx, row in log_df[abertos_mask].iterrows():
        data_saida_str = row.get("data_saida_esperada")

        if pd.isna(data_saida_str) or data_saida_str is None:
            # A data de saída fica vazia quando o sinal foi aberto perto do
            # fim dos dados disponíveis: `_nth_dia_util` não encontra
            # holding_days pregões à frente e devolve None. Antes, este
            # `continue` deixava a posição aberta PARA SEMPRE — em 14/09 as
            # 719 posições abertas estavam nesse estado, 627 já vencidas,
            # e nenhuma seria fechada nunca. Agora recalcula com o índice
            # atual, que já tem os pregões que faltavam na abertura.
            data_sinal = pd.Timestamp(row["data_sinal"])
            h = int(row["holding_days"])
            recalc = _nth_dia_util(trading_idx, data_sinal, h)
            if recalc is None:
                continue  # ainda não há pregões suficientes; tenta amanhã
            log_df.at[idx, "data_saida_esperada"] = recalc.strftime("%Y-%m-%d")
            data_saida_str = recalc.strftime("%Y-%m-%d")

        data_saida = pd.Timestamp(data_saida_str)
        if data_saida > data_hoje:
            continue  # posição ainda aberta

        b = str(row["ativo_b"])
        if b not in retornos.columns:
            log_df.at[idx, "status"] = "sem_dados"
            continue

        data_entrada = pd.Timestamp(row["data_entrada"])
        mask = (retornos.index >= data_entrada) & (retornos.index <= data_saida)
        ret_b_acum = float(retornos.loc[mask, b].sum())

        long_b = bool(row["long_b"])
        custo  = float(row["custo_pct"])
        size   = float(row["size_pct"]) / 100.0

        ret_liq = (ret_b_acum if long_b else -ret_b_acum) - custo
        contrib = ret_liq * size  # contribuição em % do capital

        log_df.at[idx, "retorno_b_acum_pct"]  = round(ret_b_acum, 4)
        log_df.at[idx, "retorno_liq_pct"]     = round(ret_liq, 4)
        log_df.at[idx, "contrib_portfolio_%"] = round(contrib, 6)
        log_df.at[idx, "status"]              = "fechado"
        n_fechados += 1

    return log_df, n_fechados


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE PAPER TRADING
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(log_df: pd.DataFrame) -> dict:
    """Calcula métricas de performance do paper trading."""
    if log_df.empty:
        return {}

    fechados = log_df[log_df["status"] == "fechado"].copy()
    if fechados.empty:
        return {
            "n_fechados": 0,
            "n_abertos": int((log_df["status"] == "aberto").sum()),
        }

    rets    = fechados["retorno_liq_pct"].dropna()
    contribs = fechados["contrib_portfolio_%"].dropna()
    n_tot   = len(fechados)

    win_rate  = float((rets > 0).mean()) if len(rets) > 0 else 0.0
    total_ret = float(contribs.sum())

    # ── Sharpe do PORTFÓLIO, líquido do CDI ─────────────────────────────
    #
    # A versão anterior fazia rets.mean()/rets.std()*sqrt(252) sobre os
    # retornos POR TRADE. Isso está errado por três motivos, e todos
    # empurram o número para cima:
    #   1. trata cada trade como se fosse um dia independente — com 3350
    #      trades em 30 dias, o fator sqrt(252) infla massivamente;
    #   2. trades sobrepostos não são observações independentes;
    #   3. não desconta a taxa livre de risco, que num CDI de ~14% a.a.
    #      é a maior parte do retorno.
    # O resultado prático: reportava Sharpe 2.46 e "PRONTO PARA GO-LIVE"
    # quando o Sharpe real, líquido do CDI, era 0.20 — um falso positivo
    # no exato critério que existe para proteger capital real.
    #
    # A forma correta: agregar as contribuições por DIA (que é o retorno
    # do portfólio), descontar o CDI diário e anualizar pelo número real
    # de observações por ano de calendário.
    def _sharpe_portfolio(df_trades: pd.DataFrame) -> float:
        if df_trades.empty or "contrib_portfolio_%" not in df_trades.columns:
            return 0.0
        d = df_trades.copy()
        d["dia"] = pd.to_datetime(d["data_entrada"], errors="coerce").dt.normalize()
        d["contrib"] = pd.to_numeric(d["contrib_portfolio_%"], errors="coerce")
        d = d.dropna(subset=["dia", "contrib"])
        if d.empty:
            return 0.0
        diario = d.groupby("dia")["contrib"].sum().sort_index()
        if len(diario) < 5 or diario.std() == 0:
            return 0.0

        # Desconta o CDI do mesmo dia (retorno em excesso)
        excesso = diario
        try:
            caminho_cdi = os.path.join(DATA, "cdi.csv")
            if os.path.exists(caminho_cdi):
                cdi = pd.read_csv(caminho_cdi, index_col=0, parse_dates=True)
                col = "CDI" if "CDI" in cdi.columns else cdi.columns[0]
                cdi_d = cdi[col].pct_change().reindex(diario.index).fillna(0) * 100
                excesso = diario - cdi_d
        except Exception:
            pass  # sem CDI, reporta o Sharpe bruto (mais otimista)

        if excesso.std() == 0:
            return 0.0
        anos = max((diario.index[-1] - diario.index[0]).days / 365.25, 1e-9)
        obs_ano = len(diario) / anos
        return float(excesso.mean() / excesso.std() * np.sqrt(obs_ano))

    sharpe = _sharpe_portfolio(fechados)

    # Janela de 30 dias
    cutoff_30d = pd.Timestamp.today() - pd.Timedelta(days=30)
    recentes   = fechados[pd.to_datetime(fechados["data_sinal"]) >= cutoff_30d]
    rets_30d   = recentes["retorno_liq_pct"].dropna()
    n_30d      = len(recentes)
    wr_30d     = float((rets_30d > 0).mean()) if len(rets_30d) > 0 else 0.0
    sharpe_30d = _sharpe_portfolio(recentes)

    # ── Critério de go-live ─────────────────────────────────────────────
    #
    # O roteiro pedia "Sharpe > 0.8 nos últimos 30 dias com >= 20 trades".
    # Medido sobre os dados reais, esse critério isolado aprova em 45% dos
    # momentos: o Sharpe de 30 dias oscila entre -3.7 e +9.3 (mediana 0.61)
    # porque 30 dias é amostra pequena demais para separar sorte de edge.
    # Aprovaria capital real com base num mês bom.
    #
    # Condições adicionadas, todas necessárias:
    #   - histórico mínimo de 60 dias (não decide com um mês)
    #   - Sharpe do período INTEIRO >= 0.5 (não só da janela recente)
    #   - o de 30 dias >= 0.8 e >= 20 trades (o critério original)
    dias_hist = 0
    if not fechados.empty:
        _dias = pd.to_datetime(fechados["data_entrada"], errors="coerce").dropna()
        if len(_dias) > 1:
            dias_hist = (_dias.max() - _dias.min()).days

    MIN_DIAS_HIST      = 60
    MIN_SHARPE_TOTAL   = 0.5
    go_live_ok = (
        sharpe_30d >= 0.8
        and n_30d >= 20
        and dias_hist >= MIN_DIAS_HIST
        and sharpe >= MIN_SHARPE_TOTAL
    )
    go_live_motivo = ""
    if not go_live_ok:
        faltas = []
        if sharpe_30d < 0.8:
            faltas.append(f"Sharpe 30d {sharpe_30d:.2f} < 0.80")
        if n_30d < 20:
            faltas.append(f"trades 30d {n_30d} < 20")
        if dias_hist < MIN_DIAS_HIST:
            faltas.append(f"historico {dias_hist}d < {MIN_DIAS_HIST}d")
        if sharpe < MIN_SHARPE_TOTAL:
            faltas.append(f"Sharpe total {sharpe:.2f} < {MIN_SHARPE_TOTAL}")
        go_live_motivo = " | ".join(faltas)

    return {
        "n_fechados":        n_tot,
        "n_abertos":         int((log_df["status"] == "aberto").sum()),
        "win_rate":          win_rate,
        "sharpe_total":      sharpe,
        "retorno_portfolio_%": total_ret,
        "n_30d":             n_30d,
        "win_rate_30d":      wr_30d,
        "sharpe_30d":        sharpe_30d,
        "go_live_pronto":    go_live_ok,
        "go_live_motivo":    go_live_motivo,
        "dias_historico":    dias_hist,
    }


def imprimir_relatorio(log_df: pd.DataFrame, metricas: dict, data: pd.Timestamp):
    """Imprime relatório de paper trading no console."""
    print(f"\n{'='*70}")
    print(f"  PAPER TRADING — RELATÓRIO  {data.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    if not metricas:
        print("  Sem dados ainda.")
        return

    n_fech = metricas.get("n_fechados", 0)
    n_ab   = metricas.get("n_abertos", 0)

    print(f"\n  ACUMULADO (total)")
    print(f"  {'─'*40}")
    print(f"  Trades fechados  : {n_fech:>6}")
    print(f"  Posições abertas : {n_ab:>6}")
    if n_fech > 0:
        print(f"  Win rate         : {metricas['win_rate']:>6.1%}")
        print(f"  Sharpe (anual.)  : {metricas['sharpe_total']:>6.2f}")
        print(f"  Retorno portf.   : {metricas['retorno_portfolio_%']:>+7.3f}% do capital")

    print(f"\n  ÚLTIMOS 30 DIAS  (critério go-live: Sharpe ≥ 0.8 | trades ≥ 20)")
    print(f"  {'─'*40}")
    print(f"  Trades           : {metricas.get('n_30d', 0):>6}")
    if metricas.get("n_30d", 0) > 0:
        print(f"  Win rate         : {metricas.get('win_rate_30d', 0):>6.1%}")
        print(f"  Sharpe           : {metricas.get('sharpe_30d', 0):>6.2f}")

    go = metricas.get("go_live_pronto", False)
    status_go = "✅ PRONTO" if go else "❌ ainda não"
    print(f"\n  Status go-live   : {status_go}")
    print(f"  Sharpe total     : {metricas.get('sharpe_total', 0):>6.2f} "
          f"(liquido do CDI, minimo 0.50)")
    print(f"  Historico        : {metricas.get('dias_historico', 0):>4}d "
          f"(minimo 60d)")
    if not go and metricas.get("go_live_motivo"):
        print(f"  Falta            : {metricas['go_live_motivo']}")

    # Posições abertas
    abertas = log_df[log_df["status"] == "aberto"]
    if not abertas.empty:
        print(f"\n  POSIÇÕES ABERTAS ({len(abertas)})")
        print(f"  {'─'*66}")
        print(f"  {'Estratégia':<45} {'Entrada':>10} {'Saída Esp.':>10} {'Size%':>6}")
        print(f"  {'─'*66}")
        for _, r in abertas.sort_values("data_entrada").iterrows():
            nome = str(r["estrategia_id"])[:43]
            print(f"  {nome:<45} {str(r['data_entrada']):>10} "
                  f"{str(r['data_saida_esperada'] or '?'):>10} "
                  f"{float(r['size_pct']):>5.2f}%")

    # Últimos 10 fechados
    fechados = log_df[log_df["status"] == "fechado"].sort_values("data_sinal")
    if not fechados.empty:
        ult = fechados.tail(10)
        print(f"\n  ÚLTIMOS {len(ult)} TRADES FECHADOS")
        print(f"  {'─'*70}")
        print(f"  {'Estratégia':<40} {'Data':>10} {'Ret.Liq%':>9} {'Contrib%':>9}")
        print(f"  {'─'*70}")
        for _, r in ult.iterrows():
            nome = str(r["estrategia_id"])[:38]
            ret  = float(r["retorno_liq_pct"]) if pd.notna(r["retorno_liq_pct"]) else 0.0
            con  = float(r["contrib_portfolio_%"]) if pd.notna(r["contrib_portfolio_%"]) else 0.0
            sinal_cor = "+" if ret >= 0 else ""
            print(f"  {nome:<40} {str(r['data_sinal']):>10} "
                  f"{sinal_cor}{ret:>8.2f}% {con:>+9.4f}%")

    print(f"\n{'='*70}")


def salvar_relatorio_txt(log_df: pd.DataFrame, metricas: dict, data: pd.Timestamp):
    """Salva relatório em arquivo texto."""
    import io
    buf = io.StringIO()

    old_stdout = sys.stdout
    sys.stdout = buf
    imprimir_relatorio(log_df, metricas, data)
    sys.stdout = old_stdout

    caminho = os.path.join(REPORTS, "paper_trading_report.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return caminho


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper trading — registra sinais e P&L virtual")
    parser.add_argument("--data", help="Data de referência (YYYY-MM-DD). Default: hoje.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Não salva nada, apenas mostra o que seria registrado.")
    parser.add_argument("--relatorio", action="store_true",
                        help="Apenas imprime o relatório do log existente.")
    args = parser.parse_args()

    data_hoje = pd.Timestamp(args.data) if args.data else pd.Timestamp.today().normalize()

    # ── Carrega dados ─────────────────────────────────────────────────────────
    retornos_path = os.path.join(DATA, "retornos.csv")
    monitor_path  = os.path.join(REPORTS, "monitor_status.csv")
    log_path      = os.path.join(REPORTS, "sinais_log.csv")
    sizes_path    = os.path.join(REPORTS, "position_sizes.csv")

    if not os.path.exists(retornos_path):
        print(f"ERRO: {retornos_path} não encontrado. Execute collector.py primeiro.")
        return
    if not os.path.exists(monitor_path):
        print(f"ERRO: {monitor_path} não encontrado. Execute walk_forward.py ou gerar_monitor.py primeiro.")
        return

    retornos = pd.read_csv(retornos_path, index_col=0, parse_dates=True)
    monitor  = pd.read_csv(monitor_path)

    sizes_df = None
    if os.path.exists(sizes_path):
        sizes_df = pd.read_csv(sizes_path)

    # ── Carrega ou inicializa log ──────────────────────────────────────────────
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        # Garante que colunas de data são strings (armazenadas como texto no CSV)
    else:
        log_df = pd.DataFrame(columns=[
            "data_sinal", "estrategia_id", "ativo_a", "ativo_b",
            "direcao_a", "long_b", "threshold", "holding_days", "size_pct",
            "retorno_a_pct", "data_entrada", "data_saida_esperada",
            "retorno_b_acum_pct", "custo_pct", "retorno_liq_pct",
            "contrib_portfolio_%", "status",
        ])

    # ── Modo relatório apenas ─────────────────────────────────────────────────
    if args.relatorio:
        metricas = calcular_metricas(log_df)
        imprimir_relatorio(log_df, metricas, data_hoje)
        return

    # ── Quais pregões processar ────────────────────────────────────────────────
    #
    # Antes, processava só `data_hoje`. Dois problemas apareceram em produção:
    #   1. Rodando às 19h30, os dados muitas vezes só vão até ONTEM (yfinance
    #      publica o fechamento com atraso). Procurar sinal em "hoje" achava
    #      zero sinais, e o pregão de ontem nunca era processado.
    #   2. Quando a tarefa falhava por alguns dias (aconteceu 11 dias seguidos
    #      em set/2026), esses pregões ficavam perdidos para sempre.
    #
    # Agora processa TODOS os pregões que existem nos dados e ainda não foram
    # registrados no log, do mais antigo ao mais recente. Idempotente: rodar
    # duas vezes no mesmo dia não duplica nada.
    if args.data:
        pregoes = [data_hoje] if data_hoje in retornos.index else []
    else:
        ultimo_no_log = None
        if not log_df.empty and "data_sinal" in log_df.columns:
            ultimo_no_log = pd.to_datetime(log_df["data_sinal"], errors="coerce").max()
        candidatos = retornos.index[retornos.index <= data_hoje]
        if pd.notna(ultimo_no_log) if ultimo_no_log is not None else False:
            candidatos = candidatos[candidatos > ultimo_no_log]
        # Limita a recuperação a 30 pregões para não reprocessar meses caso o
        # log seja apagado por engano.
        pregoes = list(candidatos[-30:])

    print(f"\n{'='*62}")
    print(f"  PAPER TRADING — {data_hoje.strftime('%Y-%m-%d')}")
    print(f"{'='*62}")
    n_ativas = int((monitor["status"] == "ativa").sum())
    print(f"  Estratégias ativas : {n_ativas}")
    print(f"  Dados até          : {retornos.index[-1].date()}")
    if pregoes:
        print(f"  Pregões a processar: {len(pregoes)} "
              f"({pregoes[0].date()} -> {pregoes[-1].date()})")
    else:
        print(f"  Pregões a processar: 0 (log já está em dia)")

    n_fechados_total = 0
    novos_total = []
    for dia in pregoes:
        # 1. Fecha posições vencidas até este dia
        log_df, n_f = fechar_posicoes(log_df, retornos, dia)
        n_fechados_total += n_f

        # 2. Detecta novos sinais neste dia. log_df vai depois de
        # fechar_posicoes: uma posição que fechou hoje já libera a
        # estratégia para abrir de novo (relevante no tipo 4).
        novos = detectar_sinais(retornos, monitor, dia, sizes_df, log_df)
        if novos:
            novo_df = pd.DataFrame(novos)
            log_df = pd.concat([log_df, novo_df], ignore_index=True) \
                     if not log_df.empty else novo_df
            novos_total.extend(novos)

    # Fechamento final com a data de referência, para pegar vencimentos
    # que caem depois do último pregão processado.
    log_df, n_f = fechar_posicoes(log_df, retornos, data_hoje)
    n_fechados_total += n_f

    print(f"  Posições fechadas  : {n_fechados_total}")
    print(f"  Novos sinais       : {len(novos_total)}")

    if novos_total and len(novos_total) <= 40:
        print(f"\n  SINAIS DISPARADOS:")
        print(f"  {'─'*66}")
        for s in novos_total:
            dir_str = "LONG " if s["long_b"] else "SHORT"
            print(f"  {s['data_sinal']} {s['estrategia_id'][:38]:<38} "
                  f"{dir_str} {s['ativo_b']:<12} "
                  f"saída={s['data_saida_esperada'] or '?'}")
    elif novos_total:
        print(f"  ({len(novos_total)} sinais — lista omitida por tamanho)")

    # ── 3. Métricas ────────────────────────────────────────────────────────────
    metricas = calcular_metricas(log_df)
    if metricas and metricas.get("n_fechados", 0) > 0:
        print(f"\n  PERFORMANCE ACUMULADA:")
        print(f"  {'─'*40}")
        print(f"  Trades fechados  : {metricas['n_fechados']:>6}")
        print(f"  Win rate         : {metricas['win_rate']:>6.1%}")
        print(f"  Sharpe (estimado): {metricas['sharpe_total']:>6.2f}")
        print(f"  Retorno portf.   : {metricas['retorno_portfolio_%']:>+7.3f}% do capital")
        print(f"  Últimos 30d      : {metricas['n_30d']:>3} trades | "
              f"Sharpe={metricas['sharpe_30d']:.2f} | "
              f"WR={metricas['win_rate_30d']:.0%}")
        go = "✅ PRONTO PARA GO-LIVE" if metricas["go_live_pronto"] else "⏳ aguardando critérios"
        print(f"  Go-live status   : {go}")
        if not metricas["go_live_pronto"] and metricas.get("go_live_motivo"):
            print(f"  Falta            : {metricas['go_live_motivo']}")

    # ── 4. Salva ───────────────────────────────────────────────────────────────
    if not args.dry_run:
        if not log_df.empty:
            log_df.to_csv(log_path, index=False)
            print(f"\n  sinais_log.csv salvo ({len(log_df)} registros total)")
        caminho_rel = salvar_relatorio_txt(log_df, metricas, data_hoje)
        print(f"  Relatório salvo  : {caminho_rel}")
    else:
        print("\n  [dry-run] Nada foi salvo.")

    print(f"{'='*62}")


if __name__ == "__main__":
    main()
