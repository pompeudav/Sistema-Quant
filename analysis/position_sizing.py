"""
position_sizing.py
------------------
Fase 2 — Kelly Criterion + gestão de risco por estratégia.

Responsabilidades:
  1. KellyCalculator    — Kelly fracionário por estratégia
  2. VolatilityTargeter — ajusta o tamanho pela vol recente do ativo B
  3. DrawdownGuard      — suspende estratégia se drawdown exceder limite em janela móvel
  4. CorrelationLimiter — limita posições abertas simultaneamente em ativos correlacionados
  5. PortfolioSizer     — orquestra tudo e entrega o tamanho final de cada posição

Entrada principal: monitor_status.csv (gerado pelo RegimeMonitor do walk_forward.py)
Saída            : position_sizes.csv — uma linha por estratégia ativa com:
                     - kelly_f        : fração Kelly pura (entre 0 e 1)
                     - kelly_frac     : Kelly fracionário (kelly_f × fracao)
                     - size_bruto_pct : kelly_frac × 100 (% do capital, antes dos ajustes)
                     - vol_adj        : fator de ajuste por volatilidade (0.3–2.0)
                     - corr_adj       : fator de ajuste por correlação (0.0–1.0)
                     - size_final_pct : tamanho final como % do capital
                     - status_sizing  : "ativa" | "abaixo_minimo" | "suspensa_drawdown"
                     - motivo_sizing  : motivo de suspensão (se houver)

Lógica de orçamento (Passo 6):
  Com centenas de estratégias correlacionadas, o Kelly individual de cada uma não
  pode ser somado diretamente — a correlação entre ativos B reduz a independência
  efetiva do portfólio. O sistema usa dois controles complementares:

  a) Teto por ativo B (teto_por_ativo_b_pct):
     Nenhum ativo B pode receber mais do que X% do capital no total,
     independente de quantas estratégias apontam para ele.
     Isso controla a concentração real (ex: ^VIX tem 60 estratégias).

  b) Orçamento total (orcamento_risco_pct):
     Teto de exposição bruta somada. Só reescala se a soma APÓS o teto
     por ativo ainda ultrapassar o limite global.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS (ajustáveis antes de rodar)
# ═══════════════════════════════════════════════════════════════

CFG = {
    # Kelly
    "kelly_fracao":         0.25,    # Kelly fracionário: usa 1/4 do Kelly puro
                                      # Razão: Kelly puro assume estimativas perfeitas;
                                      # 1/4 é o padrão conservador da literatura

    # Teto por ativo B  ← CONTROLE PRINCIPAL DE CONCENTRAÇÃO
    # Com centenas de estratégias, muitos ativos B aparecem dezenas de vezes.
    # (ex: ^VIX aparece como B em 60 estratégias diferentes)
    # Este parâmetro limita quanto capital total pode ser alocado em cada ativo B,
    # somando todas as estratégias que o usam.
    # 2.0% = no máximo 2% do capital em qualquer ativo B único.
    # Isso resolve a correlação sem precisar descartar estratégias boas.
    "teto_por_ativo_b_pct": 2.0,

    # Orçamento de risco total — teto global de exposição bruta NOMINAL.
    # Só reescala se a soma APÓS o teto por ativo ainda ultrapassar este limite.
    # Com teto_por_ativo_b_pct=2.0 e ~96 ativos B únicos, a exposição natural
    # fica em torno de 150-200% — compatível com estratégias de curto prazo
    # onde as posições raramente ficam abertas ao mesmo tempo.
    # Para operar sem alavancagem real, deixe em 200%.
    "orcamento_risco_pct":  200.0,

    # Orçamento de exposição EFETIVA (adicionado 2026-09-02).
    #
    # Por que existe: o orçamento nominal acima soma tamanhos sem olhar
    # POR QUANTO TEMPO cada posição fica aberta. Isso funciona enquanto
    # todas as estratégias têm holding de 1-3 dias e raramente coincidem
    # (200% nominal viravam ~43% de capital realmente empregado). Quando
    # entraram estratégias de momentum com holding de 10-42 dias, elas
    # passaram a ficar posicionadas quase sempre: o capital efetivamente
    # ocupado saltou 6x (528 -> 3352 %-dias) sem que nenhum teto nominal
    # fosse violado, e o backtest creditou essa alavancagem como retorno
    # (vol anual de 68% contra 9.8% do baseline).
    #
    # exposicao_efetiva = size x fracao_do_tempo_posicionada
    # É o capital médio realmente em risco. Este teto é o que impede
    # holdings longos de alavancarem o portfólio silenciosamente.
    #
    # Calibrado em 35%: é o nível que a carteira validada (761 estratégias
    # de holding curto, CAGR 20.8% / Sharpe 1.91 / vol 9.8%) já praticava
    # — medido, não arbitrado. Portanto NÃO altera aquela configuração, e
    # passa a ser o teto que qualquer estratégia nova precisa respeitar.
    # Consequência prática: adicionar sinais lentos não aumenta o risco
    # total, apenas disputa espaço com os existentes — que é exatamente
    # o comportamento desejado para comparar variantes a risco constante.
    "orcamento_efetivo_pct": 35.0,

    # Tamanhos de posição
    "size_max_pct":         10.0,    # nunca mais de 10% do capital em uma posição
    "size_min_pct":          0.02,   # mínimo reduzido — estratégias com teto apertado
                                      # naturalmente ficam com sizes menores

    # Win rate mínimo para o Kelly ter edge positivo
    "win_rate_minimo":       0.50,

    # Volatility targeting
    "vol_alvo_pct":          1.5,    # volatilidade diária alvo por posição (%)
    "vol_janela_dias":        20,    # janela para calcular vol recente de B
    "vol_adj_min":           0.3,    # fator mínimo de ajuste
    "vol_adj_max":           2.0,    # fator máximo de ajuste

    # Drawdown guard
    "drawdown_limite_pct":   15.0,   # suspende se drawdown > 15% em janela_dd dias
    "drawdown_janela_dias":  30,     # janela de observação do drawdown

    # Correlação
    "corr_limite":            0.7,   # limite de correlação entre posições simultâneas
    "corr_janela_dias":       60,    # janela para calcular correlação entre B's
    "corr_penalidade":        0.5,   # reduz posição para 50% se B correlacionado

    # Capital de referência (para simular P&L em R$)
    "capital_inicial":   100_000.0,  # R$ 100k por padrão
}


# ═══════════════════════════════════════════════════════════════
# 1. KELLY CALCULATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class KellyResultado:
    estrategia_id: str
    win_rate: float          # P(ganho) — do walk-forward out-of-sample
    payoff_ratio: float      # retorno médio ganho / |retorno médio perda|
    kelly_f: float           # Kelly puro: W - (1-W)/R
    kelly_frac: float        # Kelly fracionário: kelly_f * fracao
    size_bruto_pct: float    # kelly_frac * 100 (antes dos outros ajustes)
    observacoes: str = ""


class KellyCalculator:
    """
    Calcula o Kelly Criterion fracionário para cada estratégia.

    Fórmula: f* = W - (1 - W) / R
      onde W = win_rate, R = payoff_ratio (ganho_médio / |perda_média|)

    Implementação defensiva:
      - Usa win_rate e payoff_ratio do período *de teste* do walk-forward
        (out-of-sample), nunca do período de treino
      - Aplica fracionamento de 1/4 por padrão (configurável via CFG)
      - Kelly negativo → não opera (f* = 0)
      - Kelly > 1 → limita a 1 (alavancagem não implementada aqui)
    """

    def __init__(self, fracao: float = None):
        self.fracao = fracao or CFG["kelly_fracao"]

    def calcular(
        self,
        estrategia_id: str,
        win_rate: float,
        retorno_medio_ganho: float,   # % médio nos trades vencedores
        retorno_medio_perda: float,   # % médio nos trades perdedores (negativo)
    ) -> KellyResultado:
        """
        Calcula Kelly para uma estratégia.

        Parâmetros:
          win_rate            : fração de trades lucrativos (0–1)
          retorno_medio_ganho : ex: 0.8 (significa +0.8% por trade vencedor)
          retorno_medio_perda : ex: -0.5 (significa -0.5% por trade perdedor)
        """
        obs = []

        # Validações
        if win_rate <= 0 or win_rate >= 1:
            obs.append(f"win_rate={win_rate:.2f} fora do range válido")
            return KellyResultado(
                estrategia_id=estrategia_id,
                win_rate=win_rate,
                payoff_ratio=0.0,
                kelly_f=0.0,
                kelly_frac=0.0,
                size_bruto_pct=0.0,
                observacoes="; ".join(obs),
            )

        if retorno_medio_ganho <= 0:
            obs.append("retorno_medio_ganho <= 0: estratégia não lucrativa")
            return KellyResultado(
                estrategia_id=estrategia_id,
                win_rate=win_rate,
                payoff_ratio=0.0,
                kelly_f=0.0,
                kelly_frac=0.0,
                size_bruto_pct=0.0,
                observacoes="; ".join(obs),
            )

        perda_abs = abs(retorno_medio_perda) if retorno_medio_perda != 0 else 0.001
        payoff_ratio = retorno_medio_ganho / perda_abs

        # Fórmula de Kelly
        kelly_f = win_rate - (1 - win_rate) / payoff_ratio

        if kelly_f <= 0:
            obs.append(f"Kelly puro={kelly_f:.3f} ≤ 0: edge insuficiente — não opera")
            kelly_frac = 0.0
        else:
            kelly_f = min(kelly_f, 1.0)  # sem alavancagem
            kelly_frac = kelly_f * self.fracao
            if kelly_f > 0.5:
                obs.append(f"Kelly puro={kelly_f:.2f} alto — verifique as estimativas")

        size_bruto = kelly_frac * 100  # converte para %

        return KellyResultado(
            estrategia_id=estrategia_id,
            win_rate=win_rate,
            payoff_ratio=round(payoff_ratio, 3),
            kelly_f=round(kelly_f, 4),
            kelly_frac=round(kelly_frac, 4),
            size_bruto_pct=round(size_bruto, 2),
            observacoes="; ".join(obs),
        )

    def calcular_do_monitor(
        self,
        monitor_df: pd.DataFrame,
        exp_df: pd.DataFrame,  # exp3_contrarian.csv ou exp2_threshold_sweep.csv
        win_rate_minimo: float = 0.50,
    ) -> pd.DataFrame:
        """
        Calcula Kelly para todas as estratégias do monitor.

        Usa win_rate do monitor (período recente) e extrai
        retorno_medio_ganho/perda dos experimentos (backtest completo).

        win_rate_minimo: estratégias abaixo deste valor têm Kelly=0 (sem edge).
        Retorna DataFrame com kelly_f e kelly_frac por estratégia.
        """
        resultados = []

        for _, row in monitor_df.iterrows():
            est_id = row["id"]

            # Win rate: hierarquia de fontes (mais recente = mais confiável)
            # 1. winrate_90d  (dados reais recentes — mais confiável)
            # 2. winrate_60d  (um pouco menos recente)
            # 3. winrate_hist (out-of-sample do walk-forward — fallback conservador)
            wr = row.get("winrate_90d", np.nan)
            if pd.isna(wr):
                wr = row.get("winrate_60d", np.nan)
            if pd.isna(wr):
                wr = row.get("winrate_hist", np.nan)

            if pd.isna(wr):
                resultados.append({
                    "id": est_id,
                    "win_rate_usado": np.nan,
                    "payoff_ratio": 0.0,
                    "kelly_f": 0.0,
                    "kelly_frac": 0.0,
                    "size_bruto_pct": 0.0,
                    "obs_kelly": "sem winrate disponivel (60d, 90d e hist = NaN)",
                })
                continue

            wr = float(wr)

            # Aplica piso mínimo de win_rate — abaixo disso Kelly é negativo
            if wr < win_rate_minimo:
                resultados.append({
                    "id": est_id,
                    "win_rate_usado": round(wr, 3),
                    "payoff_ratio": 0.0,
                    "kelly_f": 0.0,
                    "kelly_frac": 0.0,
                    "size_bruto_pct": 0.0,
                    "obs_kelly": f"win_rate={wr:.1%} < mínimo {win_rate_minimo:.0%}: sem edge",
                })
                continue

            partes = est_id.split("_")
            payoff = self._estimar_payoff(est_id, exp_df)

            # Retorno médio ganho/perda: estimativa conservadora baseada no payoff
            ganho_medio = payoff * 0.5  # trade vencedor típico
            perda_media = -0.5          # trade perdedor típico (com custo de 0.15%)

            kr = self.calcular(est_id, wr, ganho_medio, perda_media)
            resultados.append({
                "id": est_id,
                "win_rate_usado": round(wr, 3),
                "payoff_ratio": kr.payoff_ratio,
                "kelly_f": kr.kelly_f,
                "kelly_frac": kr.kelly_frac,
                "size_bruto_pct": kr.size_bruto_pct,
                "obs_kelly": kr.observacoes,
            })

        return pd.DataFrame(resultados)

    def _estimar_payoff(self, est_id: str, exp_df: pd.DataFrame) -> float:
        """
        Estima payoff ratio a partir dos experimentos unificados.
        Usa win_rate_% e retorno_liq_% para derivar ganho/perda médios.
        """
        if exp_df is None or exp_df.empty:
            return 1.5

        partes = est_id.split("_")
        if len(partes) < 2:
            return 1.5

        a = partes[0]
        b = partes[1]

        match = exp_df[
            (exp_df["ativo_a"] == a) &
            (exp_df["ativo_b"] == b)
        ] if "ativo_a" in exp_df.columns else pd.DataFrame()

        if match.empty:
            return 1.5

        # Usa a linha com melhor sharpe (mais representativa)
        row = match.loc[match["sharpe"].idxmax()]
        wr  = float(row["win_rate_%"]) / 100
        ret = float(row["retorno_liq_%"])

        if wr <= 0 or wr >= 1 or ret == 0:
            return 1.5

        # Payoff derivado de: E[P&L] = wr × ganho - (1-wr) × |perda|
        # Assumindo ganho/perda simétricamente distribuídos ao redor da média:
        #   ganho_medio ≈ ret_medio / wr  (aproximação razoável para swing trade)
        #   perda_media ≈ |ret_medio| / (1-wr)
        # Se ret > 0: sistema lucrativo, payoff > 1
        ret_por_trade_est = ret / max(row.get("trades", 50), 10)  # retorno médio por trade
        ganho = ret_por_trade_est / wr if wr > 0 else 1.0
        perda = abs(ret_por_trade_est) / (1 - wr) if (1 - wr) > 0 else 0.5
        payoff = abs(ganho / perda) if perda > 0 else 1.5

        return round(float(np.clip(payoff, 0.5, 5.0)), 3)


# ═══════════════════════════════════════════════════════════════
# 2. VOLATILITY TARGETER
# ═══════════════════════════════════════════════════════════════

class VolatilityTargeter:
    """
    Ajusta o tamanho da posição pela volatilidade recente do ativo B.

    Lógica:
      - Calcula o desvio padrão diário de B nos últimos `janela` dias
      - Se B está mais volátil que o alvo → reduz posição
      - Se B está menos volátil que o alvo → aumenta posição (até limite)

    Fórmula: vol_adj = vol_alvo / vol_recente
      Exemplo: alvo=1.5%, vol_recente=3.0% → vol_adj=0.5 (posição na metade)
               alvo=1.5%, vol_recente=0.8% → vol_adj=1.875, clipa em 2.0

    Por que volátilidade de B e não de A?
      A é o disparador do sinal, B é onde a posição é aberta.
      O risco da posição vem de B, não de A.
    """

    def __init__(
        self,
        vol_alvo_pct: float = None,
        janela_dias: int = None,
        adj_min: float = None,
        adj_max: float = None,
    ):
        self.vol_alvo  = vol_alvo_pct or CFG["vol_alvo_pct"]
        self.janela    = janela_dias  or CFG["vol_janela_dias"]
        self.adj_min   = adj_min      or CFG["vol_adj_min"]
        self.adj_max   = adj_max      or CFG["vol_adj_max"]

    def calcular_fator(
        self,
        ativo_b: str,
        retornos: pd.DataFrame,
        data_referencia: str = None,
    ) -> float:
        """
        Calcula o fator de ajuste de volatilidade para ativo_b.

        Retorna fator entre [adj_min, adj_max].
        Retorna 1.0 se não há dados suficientes.
        """
        if ativo_b not in retornos.columns:
            return 1.0

        serie = retornos[ativo_b].dropna()

        if data_referencia:
            serie = serie[serie.index <= data_referencia]

        if len(serie) < self.janela:
            return 1.0  # dados insuficientes — não ajusta

        vol_recente = serie.tail(self.janela).std()

        if vol_recente <= 0:
            return 1.0

        fator = self.vol_alvo / vol_recente
        return round(float(np.clip(fator, self.adj_min, self.adj_max)), 3)

    def calcular_todos(
        self,
        monitor_df: pd.DataFrame,
        retornos: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula fatores vol para todas as estratégias do monitor.
        Retorna DataFrame com colunas: id, ativo_b, vol_recente_pct, vol_adj.
        """
        resultados = []
        for _, row in monitor_df.iterrows():
            est_id  = row["id"]
            partes  = est_id.split("_")
            ativo_b = partes[1] if len(partes) > 1 else ""

            if ativo_b not in retornos.columns:
                vol_rec = np.nan
                fator   = 1.0
            else:
                serie = retornos[ativo_b].dropna().tail(self.janela)
                vol_rec = round(serie.std(), 4) if len(serie) >= 5 else np.nan
                fator   = self.calcular_fator(ativo_b, retornos)

            resultados.append({
                "id":              est_id,
                "ativo_b":         ativo_b,
                "vol_recente_pct": vol_rec,
                "vol_adj":         fator,
            })

        return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════════
# 3. DRAWDOWN GUARD
# ═══════════════════════════════════════════════════════════════

class DrawdownGuard:
    """
    Suspende estratégias que excedem o limite de drawdown em janela móvel.

    Diferente do RegimeMonitor (que olha Sharpe), o DrawdownGuard olha
    o P&L acumulado nos últimos N dias — mais direto para gestão de risco
    quando estamos operando com capital real.

    Limite padrão: -15% em 30 dias → suspende a estratégia.
    """

    def __init__(
        self,
        limite_pct: float = None,
        janela_dias: int = None,
    ):
        self.limite = -(abs(limite_pct or CFG["drawdown_limite_pct"]))
        self.janela = janela_dias or CFG["drawdown_janela_dias"]

    def verificar(
        self,
        estrategia_id: str,
        historico_trades: list,  # lista de retornos individuais (%) em ordem cronológica
    ) -> tuple[bool, str]:
        """
        Verifica se a estratégia deve ser suspensa por drawdown.

        Retorna (suspender: bool, motivo: str).
        """
        if len(historico_trades) < 3:
            return False, ""

        # Considera apenas os trades mais recentes dentro da janela
        # (aproximação: usa os últimos N trades em vez de dias corridos)
        trades_recentes = historico_trades[-self.janela:]

        curva = np.cumprod(1 + np.array(trades_recentes) / 100)
        peak  = np.maximum.accumulate(curva)
        dd    = (curva - peak) / peak * 100
        dd_max = float(dd.min())

        if dd_max < self.limite:
            return True, f"drawdown={dd_max:.1f}% < {self.limite:.1f}% em {len(trades_recentes)} trades"

        return False, ""

    def verificar_monitor(
        self,
        monitor_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Verifica todas as estratégias do monitor.

        Usa sharpe_60d como proxy de degradação quando não há
        historico_trades individual disponível.

        Retorna DataFrame com colunas: id, suspender_dd, motivo_dd.
        """
        resultados = []

        for _, row in monitor_df.iterrows():
            est_id = row["id"]

            # Proxy: usa sharpe_60d negativo como sinal de drawdown severo
            sh60 = row.get("sharpe_60d", np.nan)
            sh90 = row.get("sharpe_90d", np.nan)

            suspender = False
            motivo    = ""

            if not pd.isna(sh60):
                # Sharpe muito negativo em 60d indica drawdown severo
                if sh60 < -2.0:
                    suspender = True
                    motivo = f"sharpe_60d={sh60:.2f} < -2.0 (drawdown severo)"
                elif not pd.isna(sh90) and sh60 < -1.0 and sh90 < -0.5:
                    suspender = True
                    motivo = f"sharpe_60d={sh60:.2f} e sharpe_90d={sh90:.2f} — degradação persistente"

            resultados.append({
                "id":          est_id,
                "suspender_dd": suspender,
                "motivo_dd":   motivo,
            })

        return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════════
# 4. CORRELATION LIMITER
# ═══════════════════════════════════════════════════════════════

class CorrelationLimiter:
    """
    Limita posições simultâneas em ativos altamente correlacionados.

    Problema: com 530+ estratégias, várias apontam para o mesmo ativo B
    (ex: 20 estratégias com B=MPWR). Operar todas com tamanho cheio é
    equivalente a uma posição concentrada enorme, mesmo que os sinais
    pareçam independentes.

    Solução:
      1. Calcula a correlação entre todos os ativos B ativos
      2. Para pares com correlação > limite, reduz a posição do segundo
         (ordem por sharpe histórico — o melhor fica com tamanho cheio)
      3. Também limita a exposição total a um único ativo B

    Retorna um fator de ajuste [0, 1] por estratégia.
    """

    def __init__(
        self,
        limite_corr: float = None,
        janela_dias: int = None,
        penalidade: float = None,
    ):
        self.limite    = limite_corr or CFG["corr_limite"]
        self.janela    = janela_dias or CFG["corr_janela_dias"]
        self.penalidade = penalidade or CFG["corr_penalidade"]

    def calcular_matriz(
        self,
        ativos_b: list,
        retornos: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calcula matriz de correlação entre os ativos B."""
        ativos_validos = [a for a in ativos_b if a in retornos.columns]
        if len(ativos_validos) < 2:
            return pd.DataFrame()

        ret_recentes = retornos[ativos_validos].tail(self.janela).dropna(axis=1, how="all")
        return ret_recentes.corr()

    def calcular_fatores(
        self,
        monitor_df: pd.DataFrame,
        retornos: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula fator de ajuste por correlação para cada estratégia.

        Estratégias com B mais correlacionado com outros B ativos
        recebem fator < 1.0. A estratégia de melhor sharpe histórico
        no grupo mantém fator 1.0.

        Retorna DataFrame com: id, ativo_b, n_corr_alto, corr_adj.
        """
        # Extrai ativo_b de cada ID
        monitor_df = monitor_df.copy()
        monitor_df["ativo_b_parsed"] = monitor_df["id"].apply(
            lambda x: x.split("_")[1] if len(x.split("_")) > 1 else ""
        )

        ativos_b_unicos = monitor_df["ativo_b_parsed"].unique().tolist()
        matriz_corr = self.calcular_matriz(ativos_b_unicos, retornos)

        resultados = []

        for _, row in monitor_df.iterrows():
            est_id   = row["id"]
            ativo_b  = row["ativo_b_parsed"]
            sh_hist  = row.get("sharpe_hist", 1.0)

            if matriz_corr.empty or ativo_b not in matriz_corr.columns:
                resultados.append({
                    "id": est_id, "ativo_b": ativo_b,
                    "n_corr_alto": 0, "corr_adj": 1.0,
                })
                continue

            # Conta quantos outros ativos B têm correlação > limite
            corrs = matriz_corr[ativo_b].drop(ativo_b, errors="ignore")
            n_alto = int((corrs.abs() > self.limite).sum())

            if n_alto == 0:
                corr_adj = 1.0
            else:
                # Verifica se esta estratégia tem o melhor Sharpe no grupo
                # de correlação alta
                corr_alto_ativos = corrs[corrs.abs() > self.limite].index.tolist()
                sharpes_grupo = monitor_df[
                    monitor_df["ativo_b_parsed"].isin(corr_alto_ativos + [ativo_b])
                ]["sharpe_hist"].fillna(0)

                sh_max_grupo = sharpes_grupo.max()

                if sh_hist >= sh_max_grupo * 0.95:
                    corr_adj = 1.0  # é a melhor no grupo — mantém tamanho cheio
                else:
                    # Penalidade proporcional ao número de correlações altas
                    corr_adj = max(
                        self.penalidade,
                        1.0 / (1 + n_alto * 0.3)
                    )

            resultados.append({
                "id":          est_id,
                "ativo_b":     ativo_b,
                "n_corr_alto": n_alto,
                "corr_adj":    round(corr_adj, 3),
            })

        return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════════
# 5. PORTFOLIO SIZER (orquestrador)
# ═══════════════════════════════════════════════════════════════

class PortfolioSizer:
    """
    Orquestra todos os componentes e gera o tamanho final de cada posição.

    Pipeline:
      1. Kelly           → size_bruto (% do capital)
      2. Volatility adj  → size_bruto × vol_adj
      3. Drawdown guard  → suspende se drawdown severo
      4. Correlation adj → size × corr_adj
      5. Cap máximo      → min(size, size_max_pct)
      6. Cap mínimo      → se size < size_min_pct → não opera

    Saída: position_sizes.csv
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        capital: float = None,
        cfg: dict = None,
    ):
        self.retornos = retornos
        self.capital  = capital or CFG["capital_inicial"]
        self.cfg      = cfg or CFG

        self.kelly    = KellyCalculator(fracao=self.cfg["kelly_fracao"])
        self.vol      = VolatilityTargeter(
            vol_alvo_pct=self.cfg["vol_alvo_pct"],
            janela_dias =self.cfg["vol_janela_dias"],
            adj_min     =self.cfg["vol_adj_min"],
            adj_max     =self.cfg["vol_adj_max"],
        )
        self.dd_guard = DrawdownGuard(
            limite_pct  =self.cfg["drawdown_limite_pct"],
            janela_dias =self.cfg["drawdown_janela_dias"],
        )
        self.corr     = CorrelationLimiter(
            limite_corr =self.cfg["corr_limite"],
            janela_dias =self.cfg["corr_janela_dias"],
            penalidade  =self.cfg["corr_penalidade"],
        )

    def calcular(
        self,
        monitor_df: pd.DataFrame,
        exp_df: pd.DataFrame = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula tamanhos de posição para todas as estratégias ativas.

        monitor_df : DataFrame do monitor_status.csv
        exp_df     : DataFrame do exp3_contrarian.csv (opcional, melhora Kelly)
        """
        ativas = monitor_df[monitor_df["status"] == "ativa"].copy()

        if verbose:
            print(f"\n{'='*65}")
            print(f"POSITION SIZING — {len(ativas)} estratégias ativas")
            print(f"{'='*65}")

        # ── Passo 1: Kelly ────────────────────────────────────────
        if verbose:
            print("\n[1/4] Calculando Kelly Criterion...")
        df_kelly = self.kelly.calcular_do_monitor(
            ativas, exp_df,
            win_rate_minimo=self.cfg.get("win_rate_minimo", 0.50),
        )

        # ── Passo 2: Volatility targeting ────────────────────────
        if verbose:
            print("[2/4] Calculando ajuste de volatilidade...")
        df_vol = self.vol.calcular_todos(ativas, self.retornos)

        # ── Passo 3: Drawdown guard ───────────────────────────────
        if verbose:
            print("[3/4] Verificando drawdowns...")
        df_dd = self.dd_guard.verificar_monitor(ativas)

        # ── Passo 4: Correlação ───────────────────────────────────
        if verbose:
            print("[4/4] Calculando ajustes de correlação...")
        df_corr = self.corr.calcular_fatores(ativas, self.retornos)

        # ── Merge de todos os ajustes ─────────────────────────────
        result = ativas[["id", "status", "sharpe_hist"]].copy()
        result = result.merge(df_kelly[["id", "win_rate_usado", "kelly_f",
                                         "kelly_frac", "size_bruto_pct", "obs_kelly"]],
                               on="id", how="left")
        result = result.merge(df_vol[["id", "vol_recente_pct", "vol_adj"]],
                               on="id", how="left")
        result = result.merge(df_dd[["id", "suspender_dd", "motivo_dd"]],
                               on="id", how="left")
        result = result.merge(df_corr[["id", "n_corr_alto", "corr_adj"]],
                               on="id", how="left")

        # ── Passo 5: Cálculo final ────────────────────────────────
        result["vol_adj"]  = result["vol_adj"].fillna(1.0)
        result["corr_adj"] = result["corr_adj"].fillna(1.0)
        result["size_bruto_pct"] = result["size_bruto_pct"].fillna(0.0)

        # Aplica ajustes sequencialmente
        result["size_pos_vol"]  = result["size_bruto_pct"] * result["vol_adj"]
        result["size_pos_corr"] = result["size_pos_vol"]   * result["corr_adj"]
        result["size_final_pct"] = result["size_pos_corr"].clip(
            upper=self.cfg["size_max_pct"]
        )

        # Determina status final
        motivos = []
        statuses = []

        for _, row in result.iterrows():
            if row.get("suspender_dd", False):
                statuses.append("suspensa_drawdown")
                motivos.append(row.get("motivo_dd", "drawdown excedido"))
            elif row["size_final_pct"] < self.cfg["size_min_pct"]:
                statuses.append("abaixo_minimo")
                motivos.append(f"size={row['size_final_pct']:.2f}% < mínimo {self.cfg['size_min_pct']}%")
            elif row.get("kelly_f", 0) <= 0:
                statuses.append("sem_edge")
                motivos.append(row.get("obs_kelly", "Kelly ≤ 0"))
            else:
                statuses.append("ativa")
                motivos.append("")

        result["status_sizing"] = statuses
        result["motivo_sizing"] = motivos

        # Zera posições suspensas
        mask_suspensa = result["status_sizing"] != "ativa"
        result.loc[mask_suspensa, "size_final_pct"] = 0.0

        # ── Passo 5.5: Teto por ativo B ──────────────────────────
        # (o cálculo de ocupação usado no passo 5.7 está em
        #  _calcular_ocupacao, logo abaixo desta função)
        # Muitos ativos B aparecem em dezenas de estratégias (ex: ^VIX em 60).
        # Sem controle, a exposição real a um único ativo explode.
        # Este passo redistribui o orçamento de cada ativo B entre suas estratégias,
        # preservando a proporção relativa dos Kellys entre elas.
        teto_b = self.cfg.get("teto_por_ativo_b_pct", 2.0)

        # Extrai ativo B de cada id (formato: A_B_dir_tipo_thr_h)
        import re as _re
        _ID_RE_B = _re.compile(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr[\d.]+_h\d+(?:_m\d+)?$')

        def _extrair_b(est_id):
            m = _ID_RE_B.match(str(est_id))
            return m.group(2) if m else "__desconhecido__"

        result["_ativo_b"] = result["id"].apply(_extrair_b)

        ativas_mask = result["status_sizing"] == "ativa"
        for ativo_b, grupo in result[ativas_mask].groupby("_ativo_b"):
            exposicao_b = grupo["size_final_pct"].sum()
            if exposicao_b > teto_b:
                # Reescala proporcionalmente dentro do grupo
                fator_b = teto_b / exposicao_b
                result.loc[grupo.index, "size_final_pct"] *= fator_b
                result["size_final_pct"] = result["size_final_pct"].round(4)

        # Após teto por ativo B, reaplica filtro de mínimo
        mask_abaixo_b = (
            (result["status_sizing"] == "ativa") &
            (result["size_final_pct"] < self.cfg["size_min_pct"])
        )
        result.loc[mask_abaixo_b, "status_sizing"] = "abaixo_minimo"
        result.loc[mask_abaixo_b, "motivo_sizing"] = (
            result.loc[mask_abaixo_b, "size_final_pct"].apply(
                lambda x: f"size={x:.3f}% < mínimo {self.cfg['size_min_pct']}% após teto_b"
            )
        )
        result.loc[mask_abaixo_b, "size_final_pct"] = 0.0

        # ── Passo 5.7: Orçamento de exposição EFETIVA ────────────
        # Pondera cada tamanho pela fração do tempo em que a posição
        # fica de fato aberta. Ver "orcamento_efetivo_pct" no CFG para
        # o porquê. Sem isso, estratégias de holding longo alavancam o
        # portfólio sem violar nenhum teto nominal.
        result["ocupacao"] = self._calcular_ocupacao(result)
        ativas_mask = result["status_sizing"] == "ativa"
        result["exposicao_efetiva_pct"] = result["size_final_pct"] * result["ocupacao"]

        orc_efetivo = self.cfg.get("orcamento_efetivo_pct", 60.0)
        exp_efetiva = result.loc[ativas_mask, "exposicao_efetiva_pct"].sum()
        fator_efetivo = 1.0

        if exp_efetiva > orc_efetivo and exp_efetiva > 0:
            fator_efetivo = orc_efetivo / exp_efetiva
            result.loc[ativas_mask, "size_final_pct"] *= fator_efetivo
            result["size_final_pct"] = result["size_final_pct"].round(4)
            result["exposicao_efetiva_pct"] = (
                result["size_final_pct"] * result["ocupacao"]
            )
            mask_ab = (
                (result["status_sizing"] == "ativa") &
                (result["size_final_pct"] < self.cfg["size_min_pct"])
            )
            result.loc[mask_ab, "status_sizing"] = "abaixo_minimo"
            result.loc[mask_ab, "motivo_sizing"] = "abaixo do mínimo após orçamento efetivo"
            result.loc[mask_ab, "size_final_pct"] = 0.0

        self._exp_efetiva_antes = exp_efetiva
        self._fator_efetivo     = fator_efetivo

        # ── Passo 6: Orçamento de risco total ────────────────────
        # Teto global de exposição bruta. Após o teto por ativo B,
        # a exposição normalmente já está controlada. Este passo só
        # reescala se a soma ainda ultrapassar o limite global.
        orcamento = self.cfg["orcamento_risco_pct"]
        exposicao_atual = result[result["status_sizing"] == "ativa"]["size_final_pct"].sum()

        if exposicao_atual > orcamento:
            fator_escala = orcamento / exposicao_atual
            result.loc[result["status_sizing"] == "ativa", "size_final_pct"] *= fator_escala
            result["size_final_pct"] = result["size_final_pct"].round(4)
            # Reaplica o mínimo após reescalamento global
            mask_abaixo = (
                (result["status_sizing"] == "ativa") &
                (result["size_final_pct"] < self.cfg["size_min_pct"])
            )
            result.loc[mask_abaixo, "status_sizing"] = "abaixo_minimo"
            result.loc[mask_abaixo, "motivo_sizing"] = (
                result.loc[mask_abaixo, "size_final_pct"].apply(
                    lambda x: f"size={x:.3f}% < mínimo {self.cfg['size_min_pct']}% após reescalamento"
                )
            )
            result.loc[mask_abaixo, "size_final_pct"] = 0.0
            result["orcamento_aplicado"] = True
        else:
            fator_escala = 1.0
            result["orcamento_aplicado"] = False

        # Exposição final após todos os controles
        exposicao_final = result[result["status_sizing"] == "ativa"]["size_final_pct"].sum()
        result["fator_escala"] = round(fator_escala, 4)

        # Valor em R$
        result["valor_posicao_brl"] = (
            result["size_final_pct"] / 100 * self.capital
        ).round(2)

        if verbose:
            self._imprimir_resumo(result)

        return result

    def _calcular_ocupacao(self, result: pd.DataFrame) -> pd.Series:
        """
        Fração do tempo em que cada estratégia fica de fato posicionada.

        1.0 = sempre em posição; 0.05 = posicionada em 5% dos pregões.

        Calculada reproduzindo o sinal sobre o histórico e contando os
        dias ocupados (n_trades x holding), com bloqueio de sobreposição
        para os sinais lentos — o mesmo critério do backtester, para que
        o número usado no risco seja o mesmo que o backtest simula.

        Sem os retornos disponíveis, cai num fallback conservador baseado
        só no holding (assume que um holding longo ocupa muito tempo),
        que é o comportamento seguro: prefere subestimar a capacidade a
        subestimar o risco.
        """
        import re as _re
        _RE = _re.compile(
            r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr([\d.]+)_h(\d+)(?:_m(\d+))?$'
        )

        ocupacoes = []
        for est_id in result["id"]:
            m = _RE.match(str(est_id))
            if not m:
                ocupacoes.append(1.0)     # desconhecido: assume o pior caso
                continue

            a       = m.group(1)
            direcao = m.group(3)
            thr     = float(m.group(5))
            h       = int(m.group(6))
            mom     = int(m.group(7)) if m.group(7) else 0

            if self.retornos is None or self.retornos.empty or a not in self.retornos.columns:
                # Fallback: holding longo => ocupacao alta. Limita em 1.0.
                ocupacoes.append(min(1.0, h / 21.0))
                continue

            ra = self.retornos[a].dropna()
            n_dias = len(ra)
            if n_dias < 60:
                ocupacoes.append(min(1.0, h / 21.0))
                continue

            if mom > 0:
                janela = mom * 21
                if n_dias < janela * 2:
                    ocupacoes.append(min(1.0, h / 21.0))
                    continue
                preco = (1 + ra / 100).cumprod()
                sinal_serie = preco.pct_change(janela).shift(1)
                sinal = (sinal_serie > 0) if direcao == "sobe" else (sinal_serie < 0)
                sinal = sinal.fillna(False)
            else:
                sinal = (ra > thr) if direcao == "sobe" else (ra < -thr)

            pos_sinal = sinal.values.nonzero()[0]
            if len(pos_sinal) == 0:
                ocupacoes.append(0.0)
                continue

            # Conta trades sem sobreposicao (mesma regra do backtester)
            n_trades, livre_em = 0, -1
            for pos in pos_sinal:
                if pos < livre_em:
                    continue
                n_trades += 1
                livre_em = pos + 1 + h

            ocupacoes.append(min(1.0, (n_trades * h) / n_dias))

        return pd.Series(ocupacoes, index=result.index)

    def _imprimir_resumo(self, result: pd.DataFrame):
        """Imprime resumo do position sizing."""
        ativas    = result[result["status_sizing"] == "ativa"]
        suspensas = result[result["status_sizing"] != "ativa"]
        fator_escala       = result["fator_escala"].iloc[0] if "fator_escala" in result.columns else 1.0
        orcamento_aplicado = result["orcamento_aplicado"].iloc[0] if "orcamento_aplicado" in result.columns else False

        print(f"\n{'─'*65}")
        print(f"  Estratégias com posição ativa : {len(ativas)}")
        print(f"  Sem edge / suspensas          : {len(suspensas)}")

        if len(ativas) > 0:
            exposicao = ativas['size_final_pct'].sum()
            teto_b    = self.cfg.get("teto_por_ativo_b_pct", 2.0)
            print(f"\n  Teto por ativo B             : {teto_b:.1f}% do capital")
            print(f"  Orçamento global             : {self.cfg['orcamento_risco_pct']:.0f}%")
            print(f"  Exposição total (nominal)    : {exposicao:.1f}% do capital")
            if orcamento_aplicado:
                print(f"  Reescalamento global aplicado: fator = {fator_escala:.4f}")

            # Exposição efetiva: o capital que fica de fato em risco, já
            # descontando que posições de holding curto ficam fechadas na
            # maior parte do tempo. É este número que limita alavancagem.
            if "exposicao_efetiva_pct" in ativas.columns:
                exp_ef = ativas["exposicao_efetiva_pct"].sum()
                orc_ef = self.cfg.get("orcamento_efetivo_pct", 60.0)
                print(f"\n  Orçamento EFETIVO            : {orc_ef:.0f}% do capital")
                print(f"  Exposição efetiva            : {exp_ef:.1f}% "
                      f"(capital medio realmente em risco)")
                antes = getattr(self, "_exp_efetiva_antes", None)
                fator = getattr(self, "_fator_efetivo", 1.0)
                if antes is not None and fator < 1.0:
                    print(f"  -> reescalado por exposicao-tempo: {antes:.1f}% -> "
                          f"{exp_ef:.1f}% (fator {fator:.4f})")
                if "ocupacao" in ativas.columns:
                    print(f"  Ocupacao media das posicoes  : {ativas['ocupacao'].mean():.1%} "
                          f"do tempo")
            print(f"\n  Distribuição de tamanho (% capital):")
            print(f"    Mínimo   : {ativas['size_final_pct'].min():.3f}%")
            print(f"    Mediana  : {ativas['size_final_pct'].median():.3f}%")
            print(f"    Máximo   : {ativas['size_final_pct'].max():.3f}%")
            print(f"    Total R$ : R$ {ativas['valor_posicao_brl'].sum():,.0f} de R$ {self.capital:,.0f}")

            # Concentração por ativo B
            if "_ativo_b" in result.columns:
                conc = (
                    ativas.groupby("_ativo_b")["size_final_pct"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                print(f"\n  Top 10 ativos B por exposição total:")
                for ativo, exp in conc.items():
                    n = len(ativas[ativas["_ativo_b"] == ativo])
                    print(f"    {ativo:<40} {exp:.2f}%  ({n} estratégias)")

            print(f"\n  Top 15 por tamanho de posição:")
            top = ativas.nlargest(15, "size_final_pct")[
                ["id", "size_final_pct", "kelly_frac", "vol_adj", "corr_adj", "sharpe_hist"]
            ]
            print(top.to_string(index=False))

        if len(suspensas) > 0:
            print(f"\n  Suspensas ({len(suspensas)}):")
            for _, row in suspensas.head(10).iterrows():
                print(f"    {row['id'][:60]:60s} → {row['status_sizing']} "
                      f"({row['motivo_sizing']})")
            if len(suspensas) > 10:
                print(f"    ... e mais {len(suspensas)-10}")

    def simular_com_sizing(
        self,
        monitor_df: pd.DataFrame,
        exp_df: pd.DataFrame = None,
        pasta_saida: str = None,
    ) -> pd.DataFrame:
        """
        Simula o P&L do portfólio completo usando os tamanhos reais de posição.

        Esta é a função que torna o backtest realista: em vez de simular
        cada estratégia com 100% do capital, usa o size_final_pct calculado.

        Retorna DataFrame com curva de capital agregada do portfólio.
        """
        sizes = self.calcular(monitor_df, exp_df, verbose=False)
        ativas = sizes[sizes["status_sizing"] == "ativa"]

        if ativas.empty:
            print("⚠️  Nenhuma estratégia ativa para simular.")
            return pd.DataFrame()

        # Para cada estratégia, reconstrói a curva de retornos ponderada
        curvas = []
        capital_alocado = 0.0

        for _, row in ativas.iterrows():
            partes  = row["id"].split("_")
            if len(partes) < 4:
                continue

            ativo_a = partes[0]
            ativo_b = partes[1]
            direcao = partes[2]
            tipo    = partes[3]  # CTR ou MOM
            size    = row["size_final_pct"] / 100

            if ativo_a not in self.retornos.columns or \
               ativo_b not in self.retornos.columns:
                continue

            # Extrai parâmetros do ID
            try:
                thr_str = [p for p in partes if p.startswith("thr")]
                hld_str = [p for p in partes if p.startswith("h")]
                thr = float(thr_str[0].replace("thr", "")) if thr_str else 0.3
                hld = int(hld_str[0].replace("h", ""))     if hld_str else 1
            except Exception:
                thr, hld = 0.3, 1

            # Reconstrói série de retornos da estratégia
            ra = self.retornos[ativo_a].dropna()
            rb = self.retornos[ativo_b].dropna()
            idx = ra.index.intersection(rb.index)

            if len(idx) < 30:
                continue

            ra_s = ra[idx]
            rb_s = rb[idx]
            sinal = ra_s > thr if direcao == "sobe" else ra_s < -thr

            ret_strat = []
            datas = []

            for i, dt in enumerate(idx[:-hld]):
                if not sinal[dt]:
                    continue
                prox = idx[i+1:i+1+hld]
                if len(prox) == 0:
                    continue
                ret = rb_s[prox].sum()
                if tipo == "CTR":
                    ret = -ret  # contrarian: inverte direção
                ret_ajustado = ret - 0.15  # custo
                ret_strat.append(ret_ajustado * size)
                datas.append(prox[-1])

            if len(ret_strat) < 5:
                continue

            serie = pd.Series(ret_strat, index=pd.DatetimeIndex(datas))
            curvas.append(serie)
            capital_alocado += size * self.capital

        if not curvas:
            print("⚠️  Nenhuma curva reconstruída.")
            return pd.DataFrame()

        # Agrega todas as curvas por data (soma dos retornos ponderados)
        df_curvas = pd.concat(curvas, axis=1).fillna(0)
        ret_portfolio = df_curvas.sum(axis=1)  # soma (já ponderado por size)

        # Curva de capital do portfólio
        capital_serie = (1 + ret_portfolio / 100).cumprod() * self.capital
        capital_serie.name = "capital_portfolio"

        # Métricas do portfólio
        n_trades_total = int(df_curvas.astype(bool).sum().sum())
        ret_total      = float(capital_serie.iloc[-1] / self.capital - 1) * 100
        vol_diaria     = float(ret_portfolio.std())
        sharpe_port    = float(
            ret_portfolio.mean() / vol_diaria * np.sqrt(252)
            if vol_diaria > 0 else 0
        )

        rolling_max = capital_serie.cummax()
        dd          = (capital_serie - rolling_max) / rolling_max * 100
        dd_max      = float(dd.min())

        print(f"\n{'='*65}")
        print(f"SIMULAÇÃO COM POSITION SIZING REAL")
        print(f"{'='*65}")
        print(f"  Estratégias na simulação : {len(curvas)}")
        print(f"  Capital inicial          : R$ {self.capital:,.0f}")
        print(f"  Capital alocado (teórico): R$ {capital_alocado:,.0f} "
              f"({capital_alocado/self.capital*100:.1f}%)")
        print(f"  Retorno total simulado   : {ret_total:+.1f}%")
        print(f"  Sharpe do portfólio      : {sharpe_port:.2f}")
        print(f"  Drawdown máximo          : {dd_max:.1f}%")
        print(f"  Trades simulados total   : {n_trades_total}")
        print(f"{'='*65}")

        resultado = pd.DataFrame({
            "data":                capital_serie.index,
            "capital_portfolio":   capital_serie.values,
            "retorno_diario_pct":  ret_portfolio.reindex(capital_serie.index).fillna(0).values,
            "drawdown_pct":        dd.values,
        })

        if pasta_saida:
            os.makedirs(pasta_saida, exist_ok=True)
            caminho = os.path.join(pasta_saida, "portfolio_simulado.csv")
            resultado.to_csv(caminho, index=False)
            print(f"\n💾 portfolio_simulado.csv salvo em {pasta_saida}")

        return resultado


# ═══════════════════════════════════════════════════════════════
# MAIN — execução standalone
# ═══════════════════════════════════════════════════════════════

def main():
    """
    Execução standalone do position sizing.
    Lê os arquivos gerados pela Fase 1 e produz position_sizes.csv.
    """
    import argparse
    import sys as _sys
    # Raiz do projeto relativa a este arquivo (analysis/position_sizing.py),
    # para o script rodar em qualquer maquina, nao so na que o criou.
    _RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.append(_RAIZ)
    from modo_teste import adicionar_flag, banner

    parser = argparse.ArgumentParser(
        description="Fase 2 — Position sizing com Kelly + volatility targeting"
    )
    adicionar_flag(parser)
    parser.add_argument("--pasta-reports", default=None,
                        help="Pasta com os CSVs da Fase 1 (default: reports/, ou reports/teste com --teste)")
    parser.add_argument("--pasta-dados",   default=os.path.join(_RAIZ, "data"),
                        help="Pasta com os dados de retornos (retornos.parquet)")
    parser.add_argument("--capital",       type=float, default=100_000.0,
                        help="Capital inicial em R$ (default: 100.000)")
    parser.add_argument("--simular",       action="store_true",
                        help="Roda simulação completa do portfólio com sizing real")
    parser.add_argument("--usar-grafo",    action="store_true",
                        help="Ajusta o tamanho da posição por centralidade no grafo de "
                             "correlação (Fase 6, opcional — ver graph_score.py)")
    args = parser.parse_args()

    banner("position_sizing.py", args.teste)
    if args.pasta_reports is None:
        args.pasta_reports = (
            os.path.join(_RAIZ, "reports", "teste") if args.teste
            else os.path.join(_RAIZ, "reports")
        )

    # ── Carrega monitor ──────────────────────────────────────
    caminho_monitor = os.path.join(args.pasta_reports, "monitor_status.csv")
    if not os.path.exists(caminho_monitor):
        print(f"❌ monitor_status.csv não encontrado em {args.pasta_reports}")
        return

    monitor_df = pd.read_csv(caminho_monitor)
    print(f"✅ monitor_status.csv carregado: {len(monitor_df)} estratégias")

    # ── Carrega experimentos e normaliza em tabela única ─────
    # exp3 já tem ativo_a/ativo_b separados.
    # exp1/exp2 usam coluna "padrao" (ex: "SPY cai → ^VIX") — precisam ser parseados.
    # Combinamos os 3 em uma única tabela de referência de payoff para o Kelly.
    frames = []
    for nome in ["exp3_contrarian.csv", "exp2_threshold_sweep.csv", "exp1_holding_sweep.csv"]:
        caminho_exp = os.path.join(args.pasta_reports, nome)
        if not os.path.exists(caminho_exp):
            continue
        df_exp = pd.read_csv(caminho_exp)

        if "ativo_a" in df_exp.columns:
            # exp3: já normalizado
            df_exp = df_exp[["ativo_a", "ativo_b", "direcao_a",
                              "win_rate_%", "retorno_liq_%", "sharpe"]].copy()
        elif "padrao" in df_exp.columns:
            # exp1/exp2: parsear "ATIVO_A dir → ATIVO_B" para colunas separadas
            def _parse_padrao(p):
                try:
                    esq, dir_ = str(p).split(" → ")
                    partes = esq.strip().rsplit(" ", 1)
                    return partes[0].strip(), dir_.strip(), partes[1].strip().lower()
                except Exception:
                    return None, None, None

            parsed = df_exp["padrao"].apply(_parse_padrao)
            df_exp["ativo_a"]   = parsed.apply(lambda x: x[0])
            df_exp["ativo_b"]   = parsed.apply(lambda x: x[1])
            df_exp["direcao_a"] = parsed.apply(lambda x: x[2])
            df_exp = df_exp.dropna(subset=["ativo_a", "ativo_b"])
            df_exp = df_exp[["ativo_a", "ativo_b", "direcao_a",
                              "win_rate_%", "retorno_liq_%", "sharpe"]].copy()
        else:
            continue

        df_exp["fonte"] = nome
        frames.append(df_exp)
        print(f"✅ {nome} carregado: {len(df_exp)} linhas")

    exp_df = pd.concat(frames, ignore_index=True) if frames else None
    if exp_df is not None:
        print(f"   → tabela unificada: {len(exp_df)} linhas de {len(frames)} experimentos")

    # ── Carrega retornos ─────────────────────────────────────
    retornos = None
    # Collector salva em data/raw/retornos.csv
    candidatos_ret = [
        os.path.join(args.pasta_dados, "raw", "retornos.csv"),
        os.path.join(args.pasta_dados, "retornos.csv"),
        os.path.join(args.pasta_dados, "retornos.parquet"),
    ]
    for caminho_ret in candidatos_ret:
        if os.path.exists(caminho_ret):
            retornos = (pd.read_parquet(caminho_ret) if caminho_ret.endswith(".parquet")
                        else pd.read_csv(caminho_ret, index_col=0, parse_dates=True))
            print(f"✅ {os.path.basename(caminho_ret)} carregado: {retornos.shape}")
            break

    if retornos is None:
        print("⚠️  retornos não encontrados — vol/corr usarão fatores neutros (1.0)")
        retornos = pd.DataFrame()

    # ── Roda position sizing ──────────────────────────────────
    sizer = PortfolioSizer(
        retornos=retornos,
        capital=args.capital,
    )

    sizes = sizer.calcular(monitor_df, exp_df, verbose=True)

    # ── Ajuste estrutural via grafo (Fase 6, opcional) ─────────
    if args.usar_grafo:
        from graph_score import montar_grafo, calcular_centralidade, aplicar_em_sizes
        caminho_padroes = os.path.join(args.pasta_reports, "padroes.csv")
        if os.path.exists(caminho_padroes):
            padroes = pd.read_csv(caminho_padroes)
            G = montar_grafo(padroes)
            cent = calcular_centralidade(G)
            sizes = aplicar_em_sizes(sizes, cent)
            print(f"\n🕸️  Score estrutural aplicado ({G.number_of_nodes()} ativos no grafo, "
                  f"{G.number_of_edges()} conexões) — sizes ajustados 0.85x-1.15x por centralidade")
        else:
            print(f"\n⚠️  --usar-grafo pedido mas {caminho_padroes} não encontrado — ignorado")

    # ── Salva resultado ───────────────────────────────────────
    caminho_out = os.path.join(args.pasta_reports, "position_sizes.csv")
    sizes.to_csv(caminho_out, index=False)
    print(f"\n💾 position_sizes.csv salvo em {args.pasta_reports}")

    # ── Simulação completa (opcional) ─────────────────────────
    if args.simular and not retornos.empty:
        sizer.simular_com_sizing(monitor_df, exp_df, pasta_saida=args.pasta_reports)


if __name__ == "__main__":
    main()