"""
regime_detector.py
------------------
Fase 4 — Análise de regime macro + ajuste de pesos por regime.

Responsabilidades:
  1. MacroRegimeDetector  — classifica o mercado em 4 regimes usando
                            VIX, DXY, TLT/IEF e CDI (sinais já coletados)
  2. RegimeWeightAdjuster — multiplica o size_final_pct de cada estratégia
                            por um fator de regime (0.0–1.5)
  3. RobustnessAnalyzer   — mede Sharpe e retorno por janela anual,
                            penaliza inconsistência temporal
  4. integrate_with_portfolio_sizer() — integração com position_sizing.py

Os 4 regimes:
  RISK_ON       — VIX baixo, DXY neutro/fraco, TLT caindo (yields subindo)
                  Favorece: MOM em ações, correlações emergentes/US
  RISK_OFF      — VIX alto (>25), TLT subindo (flight to quality)
                  Reduz: emergentes, cripto, ações cíclicas
  INFLACIONARIO — DXY forte, TLT/IEF caindo, commodities subindo
                  Favorece: commodities, energia; reduz: bonds, crescimento
  DEFLACIONARIO — DXY fraco, TLT subindo, commodities caindo
                  Favorece: bonds, ações de qualidade; reduz: commodities

═══════════════════════════════════════════════════════════
COMO INTEGRAR NO portfolio_vs_cdi.py
═══════════════════════════════════════════════════════════

1. No topo do arquivo, adicione o import:

    from regime_detector import integrate_with_portfolio_sizer

2. Na função main(), após a linha:
    sizes_df = calcular_sizes(retornos, monitor, exp_df, capital)

   adicione:
    sizes_df = integrate_with_portfolio_sizer(sizes_df, retornos)

3. Na função simular_portfolio(), troque a linha:
    size = row["size_final_pct"] / 100

   por:
    col_size = "size_regime_pct" if "size_regime_pct" in row.index else "size_final_pct"
    size = row[col_size] / 100

Pronto. O backtest histórico vai rodar com os tamanhos ajustados pelo regime
do ÚLTIMO dia disponível (modo produção). Para backtest dia a dia, use a
função integrate_backtest_historico() — mais preciso, mais lento.
═══════════════════════════════════════════════════════════
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from enum import Enum

warnings.filterwarnings("ignore")

# ── Caminhos (mesma convenção do pipeline) ───────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto, relativo a este arquivo
DATA    = os.path.join(BASE, "data", "raw")
REPORTS = os.path.join(BASE, "reports")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEFINIÇÃO DE REGIMES
# ═══════════════════════════════════════════════════════════════════════════════

class Regime(str, Enum):
    RISK_ON       = "risk_on"
    RISK_OFF      = "risk_off"
    INFLACIONARIO = "inflacionario"
    DEFLACIONARIO = "deflacionario"
    INDEFINIDO    = "indefinido"


# Fatores multiplicativos por regime e categoria de ativo.
# 1.0 = mantém size original | 0.5 = corta pela metade | 1.3 = aumenta 30%
#
# Categorias:
#   "br"      — ações/ETFs/índices brasileiros (.SA, ^BVSP, etc.)
#   "us_eq"   — ações/ETFs americanos (SPY, QQQ, ações individuais US)
#   "crypto"  — criptoativos (BTC, ETH, etc.)
#   "comm"    — commodities (CL=F, GC=F, etc.)
#   "bond"    — renda fixa (TLT, IEF, LQD, etc.)
#   "fx"      — câmbio (BRL=X, EUR=X, DX-Y.NYB)
#   "default" — qualquer outro

FATORES_REGIME: dict = {
    Regime.RISK_ON: {
        "br":      1.2,   # Brasil se beneficia de apetite por risco
        "us_eq":   1.3,   # Ações US lideram em risk-on
        "crypto":  1.2,
        "comm":    1.1,
        "bond":    0.6,   # Bonds caem quando yield sobe
        "fx":      1.0,
        "default": 1.0,
    },
    Regime.RISK_OFF: {
        "br":      0.4,   # Emergentes sofrem muito — fuga de capital
        "us_eq":   0.6,
        "crypto":  0.3,   # Cripto cai forte em pânico
        "comm":    0.6,
        "bond":    1.3,   # Flight to quality
        "fx":      0.8,
        "default": 0.7,
    },
    Regime.INFLACIONARIO: {
        "br":      0.8,   # Selic sobe, comprime valuation
        "us_eq":   0.7,   # Múltiplos comprimem com juro alto
        "crypto":  0.6,
        "comm":    1.4,   # Commodities são o hedge de inflação
        "bond":    0.4,   # Bonds tombam com inflação
        "fx":      1.1,
        "default": 0.8,
    },
    Regime.DEFLACIONARIO: {
        "br":      0.9,
        "us_eq":   1.1,   # Ações de qualidade se saem bem
        "crypto":  0.7,
        "comm":    0.5,   # Commodities caem
        "bond":    1.3,   # Bonds sobem com desinflação
        "fx":      1.0,
        "default": 0.9,
    },
    Regime.INDEFINIDO: {
        "br": 1.0, "us_eq": 1.0, "crypto": 1.0,
        "comm": 1.0, "bond": 1.0, "fx": 1.0, "default": 1.0,
    },
}

# Thresholds para classificação
CFG_REGIME = {
    "vix_risk_off":       25.0,   # VIX > 25 → sinal de risk-off
    "vix_risk_on":        18.0,   # VIX < 18 → sinal de risk-on
    "vix_janela":         10,     # dias para suavizar VIX reconstruído
    "tlt_janela":         20,     # janela de tendência TLT/IEF
    "dxy_janela":         20,     # janela de tendência DXY
    "cdi_alto":           10.0,   # CDI anualizado > 10% → regime restritivo BR
    "comm_janela":        20,     # janela de tendência do composto de commodities
    "blend_risk_off_min": 2,      # mínimo de sinais para classificar risk-off
    "blend_infl_min":     2,      # mínimo de sinais para classificar inflação
}

# Em empate de score entre regimes, decide por esta ordem de prioridade.
# RISK_ON fica por último de propósito: é o regime "nada de especial está
# acontecendo" e não deveria vencer empate contra um regime mais específico
# (ver bug documentado na Fase 4 — antes disso, RISK_ON vencia todo empate
# só por ser o primeiro no dict, o que fazia INFLACIONARIO nunca aparecer).
_PRIORIDADE_EMPATE = [
    Regime.INFLACIONARIO, Regime.DEFLACIONARIO, Regime.RISK_OFF, Regime.RISK_ON,
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DETECTOR DE REGIME
# ═══════════════════════════════════════════════════════════════════════════════

class MacroRegimeDetector:
    """
    Classifica cada dia em um dos 4 regimes usando sinais de mercado.

    Sinais:
      ^VIX      — medo/complacência (já coletado pelo collector.py)
      DX-Y.NYB  — força do dólar   (já coletado)
      TLT       — treasuries 20yr  (já coletado)
      IEF       — treasuries 7-10yr(já coletado)
      CDI       — taxa básica BR   (já coletado)
    """

    def __init__(self, retornos: pd.DataFrame, cfg: dict = None):
        self.ret = retornos
        self.cfg = cfg or CFG_REGIME
        self._cache: Optional[pd.Series] = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get(self, *tickers) -> Optional[pd.Series]:
        for t in tickers:
            if t in self.ret.columns:
                return self.ret[t]
        return None

    def _tendencia(self, serie: pd.Series, janela: int) -> pd.Series:
        """
        +1 se preço acumulado está acima da média móvel (tendência alta),
        -1 se abaixo, 0 se dados insuficientes.

        `serie` vem de self.ret, que guarda retornos em formato percentual
        (5.0 = 5%, não 0.05 — ver collector.py). Por isso divide por 100
        antes de compor; sem isso, um retorno de "5.0" era tratado como
        +500% num único dia, e o preço acumulado explodia em poucas
        semanas, tornando a tendência reconstruída um artefato numérico
        sem relação com o ativo real.
        """
        preco = (1 + serie.fillna(0) / 100).cumprod()
        ma    = preco.rolling(janela, min_periods=max(3, janela // 2)).mean()
        return np.sign(preco - ma)

    def _composto(self, *tickers) -> Optional[pd.Series]:
        """
        Média simples dos retornos diários dos tickers disponíveis — usado
        para montar um proxy de 'commodities em geral' a partir de futuros
        individuais (ouro, petróleo, cobre), já que não existe um índice
        único de commodities entre os dados coletados.
        """
        series = [self.ret[t] for t in tickers if t in self.ret.columns]
        if not series:
            return None
        return pd.concat(series, axis=1).mean(axis=1)

    def _nivel_vix(self) -> Optional[pd.Series]:
        """
        Reconstrói nível aproximado do VIX a partir de retornos.
        Fallback: vol realizada anualizada do S&P (já em pontos percentuais).

        Mesma correção de unidade do `_tendencia`: `vix` vem em formato
        percentual (5.0 = 5%), então precisa dividir por 100 antes de
        compor — sem isso o nível reconstruído explode nas primeiras
        semanas e o threshold de 25/18 nunca reflete um VIX real.
        """
        vix = self._get("^VIX", "VIX")
        if vix is not None:
            # Base 20 em 2016-01-01 (nível histórico médio do VIX)
            nivel = 20.0 * (1 + vix.fillna(0) / 100).cumprod()
            return nivel.rolling(self.cfg["vix_janela"], min_periods=3).mean()

        sp = self._get("^GSPC", "SPY")
        if sp is not None:
            # sp já está em % ao dia; std diário × sqrt(252) já anualiza em
            # pontos percentuais — não precisa de outro ×100.
            return sp.rolling(self.cfg["vix_janela"], min_periods=3).std() * np.sqrt(252)
        return None

    # ── Classificação ─────────────────────────────────────────────────────────

    def classificar_serie(self) -> pd.Series:
        """
        Classifica cada dia do histórico.
        Retorna pd.Series[Regime] com índice de datas.
        Usa cache interno.
        """
        if self._cache is not None:
            return self._cache

        idx     = self.ret.index
        regimes = pd.Series(Regime.INDEFINIDO, index=idx, dtype=object)

        vix_nivel = self._nivel_vix()
        tlt       = self._get("TLT")
        ief       = self._get("IEF")
        dxy       = self._get("DX-Y.NYB", "DXY")
        cdi       = self._get("CDI")

        tend_tlt = self._tendencia(tlt, self.cfg["tlt_janela"]) if tlt is not None else None
        tend_ief = self._tendencia(ief, self.cfg["tlt_janela"]) if ief is not None else None
        tend_dxy = self._tendencia(dxy, self.cfg["dxy_janela"]) if dxy is not None else None
        # cdi já vem em formato percentual (0.0525 = 0.0525% ao dia), então
        # média×252 já resulta no número certo em pontos percentuais anuais
        # (ex: 9.1 = 9.1% a.a.). O ×100 que existia aqui fazia cdi_anual
        # ficar sempre > 900, disparando "CDI alto" em 100% dos dias dos
        # últimos 10 anos, independente do CDI real estar em 2% ou 14%.
        cdi_anual = (
            cdi.rolling(21, min_periods=5).mean() * 252
            if cdi is not None else None
        )

        # Composto de commodities (ouro, petróleo WTI/Brent, cobre) — sinal
        # exclusivo de inflação/deflação. Antes desta correção, o detector
        # nunca olhava para commodities (apesar do docstring da classe dizer
        # que deveria), e todo sinal que apontava para inflação também
        # apontava, em igual medida, para risk-on ou risk-off — então
        # "inflacionario" nunca tinha um sinal só seu e nunca vencia.
        comm = self._composto("GC=F", "CL=F", "BZ=F", "HG=F")
        tend_comm = self._tendencia(comm, self.cfg["comm_janela"]) if comm is not None else None

        for i, dt in enumerate(idx):
            if i < 30:
                continue

            ro, rf, infl, defl = 0, 0, 0, 0

            # VIX
            if vix_nivel is not None and dt in vix_nivel.index:
                v = vix_nivel[dt]
                if pd.notna(v):
                    if v > self.cfg["vix_risk_off"]:
                        rf += 2     # sinal forte de risk-off
                    elif v < self.cfg["vix_risk_on"]:
                        ro += 2     # sinal forte de risk-on

            # TLT e IEF são quase o mesmo sinal (treasuries de prazos
            # diferentes, andam quase sempre juntos) — conta como UM voto,
            # não dois, senão infla artificialmente o lado que "ganhar".
            # subindo → flight to quality → risk-off + deflacionário
            # caindo  → yields subindo → inflacionário + risk-on
            tends_bond = [
                tend[dt] for tend in (tend_tlt, tend_ief)
                if tend is not None and dt in tend.index and pd.notna(tend[dt])
            ]
            if tends_bond:
                t_bond = np.mean(tends_bond)
                if t_bond > 0:
                    rf   += 1
                    defl += 1
                elif t_bond < 0:
                    infl += 1
                    ro   += 1

            # DXY
            # forte → fuga para dólar → risk-off OU inflação
            # fraco  → apetite por risco → risk-on OU deflação
            if tend_dxy is not None and dt in tend_dxy.index:
                d = tend_dxy[dt]
                if pd.notna(d):
                    if d > 0:
                        rf   += 1
                        infl += 1
                    elif d < 0:
                        ro   += 1
                        defl += 1

            # CDI alto (contexto Brasil)
            if cdi_anual is not None and dt in cdi_anual.index:
                c = cdi_anual[dt]
                if pd.notna(c) and c > self.cfg["cdi_alto"]:
                    infl += 1

            # Commodities — sinal EXCLUSIVO de inflação/deflação (não
            # dividido com risk-on/risk-off). Peso 1, igual a TLT/IEF/DXY:
            # uma simples tendência de preço (acima/abaixo da média móvel)
            # é verdadeira quase todo dia, então não pode valer sozinha o
            # mínimo de 2 pontos — precisa de corroboração de outro sinal
            # (DXY, TLT/IEF ou CDI alto), senão passa a dominar a
            # classificação (como aconteceu na primeira tentativa, com
            # peso 2: inflacionário/deflacionário passaram a ser >80% dos
            # dias). Só o VIX tem esse "voto sozinho" — porque seu
            # threshold (>25 ou <18) é um nível extremo, não uma simples
            # tendência que é verdadeira metade do tempo por construção.
            if tend_comm is not None and dt in tend_comm.index:
                tc = tend_comm[dt]
                if pd.notna(tc):
                    if tc > 0:
                        infl += 1
                    elif tc < 0:
                        defl += 1

            # Decide regime — em empate, usa a ordem de prioridade (ver
            # _PRIORIDADE_EMPATE): regimes mais específicos vencem RISK_ON.
            scores = {
                Regime.RISK_OFF:      rf,
                Regime.RISK_ON:       ro,
                Regime.INFLACIONARIO: infl,
                Regime.DEFLACIONARIO: defl,
            }
            score_max = max(scores.values())
            empatados = [r for r, s in scores.items() if s == score_max]
            vencedor  = next(r for r in _PRIORIDADE_EMPATE if r in empatados)
            minimo    = (
                self.cfg["blend_risk_off_min"]
                if vencedor in (Regime.RISK_OFF, Regime.RISK_ON)
                else self.cfg["blend_infl_min"]
            )

            regimes[dt] = vencedor if score_max >= minimo else Regime.INDEFINIDO

        self._cache = regimes
        return regimes

    def classificar_dia(self, data: pd.Timestamp) -> Regime:
        """Retorna o regime de um dia específico (uso em produção)."""
        serie = self.classificar_serie()
        return serie.get(data, Regime.INDEFINIDO)

    def resumo(self) -> pd.DataFrame:
        """Distribuição de regimes por ano — para diagnóstico."""
        serie = self.classificar_serie()
        df    = pd.DataFrame({"regime": [r.value for r in serie]}, index=serie.index)
        df["ano"] = df.index.year
        pivot = df.groupby(["ano", "regime"]).size().unstack(fill_value=0)
        pct   = (pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
        pct.columns = [f"{c}_pct%" for c in pct.columns]
        return pd.concat([pivot, pct], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICADOR DE ATIVOS
# ═══════════════════════════════════════════════════════════════════════════════

_CRYPTO_TOKENS = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
                  "MATIC", "LTC", "LINK", "UNI", "ATOM", "DOT", "COIN"}
_BOND_TICKERS  = {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK",
                  "IRFM11", "IMAB5", "IMAB5P"}
_COMM_TICKERS  = {"GLD", "SLV", "IAU", "USO", "UNG", "CORN", "SOYB", "WEAT"}
_FX_TICKERS    = {"BRL=X", "EUR=X", "JPY=X", "GBP=X", "DX-Y.NYB", "USDBRL=X"}


def classificar_ativo(ticker: str) -> str:
    """Retorna a categoria do ativo para seleção do fator de regime."""
    t    = ticker.upper()
    base = t.replace(".SA", "").replace("-USD", "").replace("=X", "")

    if any(c in base for c in _CRYPTO_TOKENS) or t.endswith("-USD"):
        return "crypto"
    if t in _BOND_TICKERS or base in _BOND_TICKERS:
        return "bond"
    if t in _COMM_TICKERS or base in _COMM_TICKERS or ticker.endswith("=F"):
        return "comm"
    if t in _FX_TICKERS or "=X" in t or "=Y" in t:
        return "fx"
    if ticker.endswith(".SA") or t in {"^BVSP", "BVSP", "IBOV"}:
        return "br"
    # Heurística para ações US: ticker curto, só letras maiúsculas, sem sufixo
    if len(base) <= 5 and base.isalpha() and not ticker.endswith(".SA"):
        return "us_eq"
    return "default"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AJUSTADOR DE PESOS POR REGIME
# ═══════════════════════════════════════════════════════════════════════════════

_ID_RE = re.compile(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr([\d.]+)_h(\d+)(?:_m\d+)?$')


class RegimeWeightAdjuster:
    """
    Recebe o sizes_df da Fase 2 e a série de regimes,
    e devolve sizes_df com size_regime_pct = size_final_pct × fator_regime.

    Não sobrescreve size_final_pct — adiciona colunas novas para transparência.
    """

    def __init__(self, fatores: dict = None, cap_pct: float = 100.0):
        self.fatores = fatores or FATORES_REGIME
        self.cap     = cap_pct

    def _extrair_b(self, est_id: str) -> str:
        m = _ID_RE.match(est_id)
        return m.group(2) if m else est_id

    def ajustar(
        self,
        sizes_df:     pd.DataFrame,
        serie_regime: pd.Series,
        data_ref:     Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Aplica fatores do regime vigente em data_ref (ou último dia).
        Retorna sizes_df com colunas:
          categoria_b, regime_atual, fator_regime, size_regime_pct
        """
        df = sizes_df.copy()

        if data_ref is None:
            data_ref = serie_regime.index[-1]

        regime      = serie_regime.get(data_ref, Regime.INDEFINIDO)
        fatores_dia = self.fatores.get(regime, self.fatores[Regime.INDEFINIDO])

        df["categoria_b"]  = df["id"].apply(lambda x: classificar_ativo(self._extrair_b(x)))
        df["regime_atual"] = regime.value
        df["fator_regime"] = df["categoria_b"].map(
            lambda cat: fatores_dia.get(cat, fatores_dia["default"])
        )
        df["size_regime_pct"] = df["size_final_pct"] * df["fator_regime"]

        # Reescala se ultrapassar orçamento
        total = df.loc[df.get("status_sizing", pd.Series(dtype=str)) == "ativa",
                       "size_regime_pct"].sum()
        if total > self.cap:
            df["size_regime_pct"] *= self.cap / total

        return df

    def fatores_historicos(
        self,
        sizes_df:     pd.DataFrame,
        serie_regime: pd.Series,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame [data × ativo_b] com o fator de cada dia.
        Use para backtest histórico dia a dia.
        """
        ativos_b = {
            self._extrair_b(row["id"])
            for _, row in sizes_df.iterrows()
        }
        result = {}
        for ab in ativos_b:
            cat = classificar_ativo(ab)
            result[ab] = serie_regime.map(
                lambda r: self.fatores.get(
                    r, self.fatores[Regime.INDEFINIDO]
                ).get(cat, self.fatores[Regime.INDEFINIDO]["default"])
            )
        return pd.DataFrame(result, index=serie_regime.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANALISADOR DE ROBUSTEZ TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

class RobustnessAnalyzer:
    """
    Mede Sharpe por janela anual e penaliza inconsistência temporal.

    Score de robustez = sharpe_médio_anual - penalidade × std_sharpes_anuais

    Alta std entre anos = estratégia funciona em alguns anos mas não em outros
    = não é robusta = Kelly deve ser reduzido.
    """

    def __init__(self, penalidade_std: float = 0.5):
        self.penalidade = penalidade_std

    def analisar(self, retornos_serie: pd.Series, nome: str = "") -> dict:
        """Analisa uma série de retornos diários e retorna score de robustez."""
        ret = retornos_serie.dropna()
        ret.index = pd.to_datetime(ret.index)

        if len(ret) < 50:
            return {"nome": nome, "score_robustez": 0.0, "n_anos": 0}

        anos            = ret.index.year.unique()
        sharpes_anuais  = []
        retornos_anuais = []

        for ano in anos:
            ra = ret[ret.index.year == ano]
            if len(ra) < 20 or ra.std() == 0:
                continue
            sharpes_anuais.append((ra.mean() / ra.std()) * np.sqrt(252))
            retornos_anuais.append(float((1 + ra).prod() - 1))

        if not sharpes_anuais:
            return {"nome": nome, "score_robustez": 0.0, "n_anos": 0}

        sm    = float(np.mean(sharpes_anuais))
        ss    = float(np.std(sharpes_anuais))
        score = sm - self.penalidade * ss

        return {
            "nome":               nome,
            "sharpe_medio":       round(sm, 3),
            "sharpe_std":         round(ss, 3),
            "ret_medio_anual_%":  round(float(np.mean(retornos_anuais)) * 100, 2),
            "pct_anos_positivos": round(float(np.mean([s > 0 for s in sharpes_anuais])), 3),
            "score_robustez":     round(score, 3),
            "n_anos":             len(sharpes_anuais),
            "sharpes_por_ano":    [round(s, 2) for s in sharpes_anuais],
        }

    def adicionar_scores(
        self,
        sizes_df: pd.DataFrame,
        retornos: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adiciona coluna score_robustez ao sizes_df.
        Usa o retorno do ativo B como proxy da estratégia.
        """
        df = sizes_df.copy()
        scores = []
        for _, row in df.iterrows():
            m = _ID_RE.match(row["id"])
            b = m.group(2) if m else ""
            if b in retornos.columns:
                r = self.analisar(retornos[b], nome=row["id"])
                scores.append(r.get("score_robustez", 0.0))
            else:
                scores.append(0.0)
        df["score_robustez"] = scores
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FUNÇÕES DE INTEGRAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_with_portfolio_sizer(
    sizes_df: pd.DataFrame,
    retornos: pd.DataFrame,
    data_ref: Optional[pd.Timestamp] = None,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Integração principal — chame no portfolio_vs_cdi.py após calcular_sizes().

    Classifica regimes, aplica fatores ao size_final_pct, salva CSVs de diagnóstico.
    Retorna sizes_df com coluna size_regime_pct adicionada.
    """
    if verbose:
        print("\nFase 4 — Detector de Regime Macro...")

    detector = MacroRegimeDetector(retornos)
    serie    = detector.classificar_serie()

    os.makedirs(REPORTS, exist_ok=True)
    pd.DataFrame({
        "data":   serie.index,
        "regime": [r.value for r in serie],
    }).to_csv(os.path.join(REPORTS, "regime_historico.csv"), index=False)
    detector.resumo().to_csv(os.path.join(REPORTS, "regime_resumo_anual.csv"))

    if verbose:
        contagem = pd.Series([r.value for r in serie]).value_counts()
        total    = len(serie)
        for reg, n in contagem.items():
            print(f"  {reg:<20} {n:>5} dias  ({n/total*100:.1f}%)")
        dr = data_ref or serie.index[-1]
        print(f"  Regime em {dr.date()}: {serie.get(dr, Regime.INDEFINIDO).value.upper()}")

    adjuster  = RegimeWeightAdjuster()
    sizes_adj = adjuster.ajustar(sizes_df, serie, data_ref=data_ref)

    if verbose:
        mask  = sizes_df["status_sizing"] == "ativa"
        orig  = sizes_df.loc[mask, "size_final_pct"].sum()
        novo  = sizes_adj.loc[mask, "size_regime_pct"].sum()
        print(f"  Exposição original : {orig:.1f}%")
        print(f"  Exposição ajustada : {novo:.1f}%")

    return sizes_adj


def calibrar_fatores(
    trades_df: pd.DataFrame,
    serie_regime: pd.Series,
    n_min: int = 30,
    clip_min: float = 0.3,
    clip_max: float = 1.5,
    verbose: bool = True,
) -> dict:
    """
    Calibra FATORES_REGIME contra o histórico real de trades das estratégias.

    trades_df   : DataFrame com colunas [data_entrada, categoria_b, retorno]
    serie_regime: pd.Series[Regime] indexado por data (de classificar_serie())
    n_min       : mínimo de trades para calibrar uma célula (senão usa 1.0)
    clip_min/max: limites do fator calibrado

    Retorna dicionário no mesmo formato de FATORES_REGIME.

    Nota: calibração in-sample — use para diagnóstico de sinal,
    não como estimativa de performance futura.
    """
    trades = trades_df.copy()
    trades["regime"] = trades["data_entrada"].map(
        lambda dt: serie_regime.get(dt, Regime.INDEFINIDO)
    )

    media_global = trades["retorno"].mean()
    if abs(media_global) < 1e-6:
        if verbose:
            print("  Aviso: retorno medio global proximo de zero — fatores neutros")
        return FATORES_REGIME

    categorias       = ["br", "us_eq", "crypto", "comm", "bond", "fx", "default"]
    regimes_calibrar = [Regime.RISK_ON, Regime.RISK_OFF,
                        Regime.INFLACIONARIO, Regime.DEFLACIONARIO]

    if verbose:
        print(f"\n  Retorno medio global : {media_global:.3f}%  "
              f"({len(trades):,} trades)")
        print(f"\n  {'Regime':<16} {'Categoria':<10} {'N':>6} "
              f"{'Ret medio':>10} {'Fator':>8}")
        print(f"  {'─'*54}")

    fatores_cal: dict = {}
    for regime in regimes_calibrar:
        fatores_cal[regime] = {}
        df_reg = trades[trades["regime"] == regime]

        for cat in categorias:
            df_cell = df_reg[df_reg["categoria_b"] == cat]
            n = len(df_cell)

            if n < n_min:
                fator   = 1.0
                ret_med = float("nan")
            else:
                ret_med = df_cell["retorno"].mean()
                fator   = float(np.clip(ret_med / media_global, clip_min, clip_max))

            fatores_cal[regime][cat] = round(fator, 3)

            if verbose and n >= n_min:
                print(f"  {regime.value:<16} {cat:<10} {n:>6} "
                      f"{ret_med:>10.3f}% {fator:>8.3f}")

    fatores_cal[Regime.INDEFINIDO] = {cat: 1.0 for cat in categorias}

    if verbose:
        print()

    return fatores_cal


def integrate_backtest_historico(
    sizes_df: pd.DataFrame,
    retornos: pd.DataFrame,
    verbose:  bool = True,
    fatores_calibrados: dict = None,
) -> pd.DataFrame:
    """
    Para backtest histórico dia a dia — retorna DataFrame [data × ativo_b]
    com o fator multiplicativo de cada dia.

    fatores_calibrados: dict opcional no formato FATORES_REGIME.
    Se None, usa os fatores manuais padrão (FATORES_REGIME).

    Integre no loop de simular_portfolio():
        fatores = integrate_backtest_historico(sizes_df, retornos)
        ...
        fator_dia = fatores.at[dt, b] if b in fatores.columns else 1.0
        ret_ajustado = ret_ajustado * fator_dia
    """
    if verbose:
        print("\nCalculando fatores de regime historico (dia a dia)...")

    detector = MacroRegimeDetector(retornos)
    serie    = detector.classificar_serie()
    adjuster = RegimeWeightAdjuster(fatores=fatores_calibrados)
    fatores  = adjuster.fatores_historicos(sizes_df, serie)

    if verbose:
        print(f"  {fatores.shape[0]} dias x {fatores.shape[1]} ativos B")

    return fatores


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN — diagnóstico standalone
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("REGIME DETECTOR — Fase 4 (diagnóstico)")
    print("=" * 65)

    caminho_ret = os.path.join(DATA, "retornos.csv")
    if not os.path.exists(caminho_ret):
        print(f"\nERRO: {caminho_ret} não encontrado.")
        print("Execute o collector.py antes.")
        return

    print("\nCarregando retornos...")
    ret = pd.read_csv(caminho_ret, index_col=0, parse_dates=True)
    print(f"  {ret.shape[0]} dias × {ret.shape[1]} ativos  "
          f"({ret.index[0].date()} → {ret.index[-1].date()})")

    print("\nClassificando regimes...")
    detector = MacroRegimeDetector(ret)
    serie    = detector.classificar_serie()

    resumo = detector.resumo()
    print("\nDistribuição por ano (%):")
    pct_cols = [c for c in resumo.columns if "_pct%" in str(c)]
    print(resumo[pct_cols].fillna(0).to_string())

    print(f"\nRegime do último dia ({serie.index[-1].date()}): "
          f"{serie.iloc[-1].value.upper()}")

    os.makedirs(REPORTS, exist_ok=True)
    pd.DataFrame({
        "data":   serie.index,
        "regime": [r.value for r in serie],
    }).to_csv(os.path.join(REPORTS, "regime_historico.csv"), index=False)
    resumo.to_csv(os.path.join(REPORTS, "regime_resumo_anual.csv"))
    print(f"\nSalvo em {REPORTS}:")
    print("  regime_historico.csv")
    print("  regime_resumo_anual.csv")

    caminho_ps = os.path.join(REPORTS, "position_sizes.csv")
    if os.path.exists(caminho_ps):
        print("\nIntegrando com position_sizes.csv...")
        sizes     = pd.read_csv(caminho_ps)
        sizes_adj = integrate_with_portfolio_sizer(sizes, ret)
        ativas    = sizes_adj[sizes_adj.get("status_sizing", pd.Series()) == "ativa"]
        if not ativas.empty:
            print("\n  Amostra (primeiras 10 estratégias):")
            cols = ["id", "categoria_b", "fator_regime", "size_final_pct", "size_regime_pct"]
            print(ativas[cols].head(10).to_string(index=False))
        sizes_adj.to_csv(os.path.join(REPORTS, "position_sizes_regime.csv"), index=False)
        print("\n  position_sizes_regime.csv salvo")
    else:
        print("\n  (position_sizes.csv não encontrado — rode portfolio_vs_cdi.py primeiro)")

    print("\nFase 4 concluída.")


if __name__ == "__main__":
    main()
