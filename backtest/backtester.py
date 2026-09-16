"""
backtester.py
-------------
Simula o resultado financeiro de operar padrões detectados pelo correlator.

Lógica central:
  - Para cada padrão (tipo 1 ou 2), simula uma estratégia onde:
      → Quando A dispara o padrão (ex: A cai > threshold%), abrimos
        uma posição em B na direção esperada (compra ou venda a descoberto)
      → Fechamos a posição no fechamento do dia seguinte (holding = 1 dia)
        ou após N dias configurável
  - Calcula retorno bruto, líquido (com custos), drawdown máximo,
    Sharpe ratio, win rate, e compara com benchmark (buy & hold)

Custos simulados (configuráveis):
  - Spread/slippage : 0.05% por operação (entrada + saída = 0.10%)
  - Corretagem      : 0.025% por operação (entrada + saída = 0.05%)
  - Total por trade : ~0.15% (realista para swing trade BR/US)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    from tqdm import tqdm
except ImportError:
    # Fallback simples se tqdm não estiver instalado
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        try:
            total = total or len(iterable)
        except TypeError:
            total = "?"
        print(f"\n▶ {desc} ({total} itens)")
        return iterable
matplotlib.use("Agg")  # sem GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ResultadoBacktest:
    """Resultado completo de um backtest de padrão."""
    # Identificação
    ativo_a: str
    ativo_b: str
    direcao_a: str
    direcao_b: str
    tipo_padrao: int
    threshold: float
    holding_days: int

    # Métricas de operação
    n_trades: int
    win_rate: float           # % de trades lucrativos
    retorno_total: float      # retorno acumulado da estratégia (%)
    retorno_medio: float      # retorno médio por trade (%)
    retorno_anual: float      # retorno anualizado (%)
    vol_anual: float          # volatilidade anualizada dos retornos (%)
    sharpe: float             # Sharpe ratio bruto (sem taxa livre de risco)
    sharpe_ajustado: float    # Sharpe ratio ajustado pelo CDI ou FedFunds
    drawdown_max: float       # máximo drawdown (%)
    retorno_liquido: float    # retorno total após custos (%)

    # Benchmark
    retorno_bh_a: float       # buy & hold de A no mesmo período (%)
    retorno_bh_b: float       # buy & hold de B no mesmo período (%)
    retorno_cdi: float        # CDI no período (%)

    # Série temporal
    curva_capital: pd.Series  # evolução do capital ao longo do tempo

    def __str__(self):
        direcao = f"{self.direcao_a.upper()}({self.ativo_a}) → {self.direcao_b.upper()}({self.ativo_b})"
        return (
            f"\n{'='*60}\n"
            f"BACKTEST: {direcao}\n"
            f"{'='*60}\n"
            f"  Trades          : {self.n_trades}\n"
            f"  Win rate        : {self.win_rate:.1%}\n"
            f"  Retorno total   : {self.retorno_total:+.1f}%\n"
            f"  Retorno líquido : {self.retorno_liquido:+.1f}%  (após custos)\n"
            f"  Retorno anual   : {self.retorno_anual:+.1f}%\n"
            f"  Volatilidade    : {self.vol_anual:.1f}% a.a.\n"
            f"  Sharpe bruto    : {self.sharpe:.2f}\n"
            f"  Sharpe ajustado : {self.sharpe_ajustado:.2f}  (descontado CDI/FedFunds)\n"
            f"  Drawdown máx.   : {self.drawdown_max:.1f}%\n"
            f"  ─────────────────────────────────────\n"
            f"  Buy & hold A    : {self.retorno_bh_a:+.1f}%\n"
            f"  Buy & hold B    : {self.retorno_bh_b:+.1f}%\n"
            f"  CDI período     : {self.retorno_cdi:+.1f}%\n"
        )

    def resumo_linha(self) -> dict:
        return {
            "sinal":             f"{self.ativo_a} {self.direcao_a}",
            "posicao":           f"{self.ativo_b} {self.direcao_b}",
            "trades":            self.n_trades,
            "win_rate_%":        round(self.win_rate * 100, 1),
            "retorno_liq_%":     round(self.retorno_liquido, 1),
            "retorno_anual_%":   round(self.retorno_anual, 1),
            "sharpe":            round(self.sharpe, 2),
            "sharpe_ajustado":   round(self.sharpe_ajustado, 2),
            "drawdown_max_%":    round(self.drawdown_max, 1),
            "bh_b_%":            round(self.retorno_bh_b, 1),
            "cdi_%":             round(self.retorno_cdi, 1),
            "alpha_%":           round(self.retorno_liquido - self.retorno_cdi, 1),
        }


# ═══════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════

class Backtester:
    """
    Simula estratégias de trading baseadas em padrões de correlação.

    Uso:
        bt = Backtester(retornos, cdi=df_cdi)
        resultado = bt.rodar(ativo_a="BZ=F", ativo_b="PRIO3.SA",
                             direcao_a="sobe", direcao_b="sobe")
        print(resultado)
        bt.plotar(resultado)
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        cdi: Optional[pd.DataFrame] = None,
        fedfunds: Optional[pd.DataFrame] = None,
        custo_por_trade: float = 0.15,   # % total (entrada + saída)
        threshold: float = 0.3,           # % mínimo de movimento de A
        holding_days: int = 1,            # dias que mantém a posição
    ):
        """
        Parâmetros
        ----------
        retornos        : DataFrame com retornos diários em % (todas as fontes)
        cdi             : DataFrame com coluna 'CDI' (índice acumulado BCB)
        fedfunds        : DataFrame com coluna 'FedFunds' (índice acumulado FRED)
        custo_por_trade : custo total em % por operação (spread + corretagem)
        threshold       : variação mínima de A para disparar o sinal (%)
        holding_days    : quantos dias manter a posição após o sinal

        Sharpe ajustado:
          - Ativos BR (.SA, fundos CVM): usa CDI como taxa livre de risco
          - Ativos US e outros         : usa Fed Funds como taxa livre de risco
          - Commodities e cripto       : usa Fed Funds (dolarizados)
        """
        self.retornos = retornos.copy()
        self.cdi = cdi
        self.fedfunds = fedfunds
        self.custo = custo_por_trade
        self.threshold = threshold
        self.holding_days = holding_days

    def rodar(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,   # "sobe" ou "cai"
        direcao_b: str,   # "sobe" ou "cai" (direção da posição em B)
        holding_days: int = None,
        momentum_meses: int = 0,   # >0 → sinal tipo 4 (momentum), não threshold
    ) -> Optional[ResultadoBacktest]:
        """
        Roda o backtest de um padrão específico.

        Lógica (tipos 1-3, momentum_meses=0):
          Dia t  : A dispara o sinal (sobe/cai > threshold)
          Dia t+1: abrimos posição em B na direção esperada
          Dia t+N: fechamos a posição (N = holding_days)
          Retorno do trade = retorno de B no período (com sinal se short)

        Lógica (tipo 4, momentum_meses>0):
          O gatilho não é um movimento de um dia, e sim o retorno acumulado
          dos últimos `momentum_meses` meses estar positivo ("sobe") ou
          negativo ("cai"). O resto (abrir em t+1, fechar em t+N) é igual.
          Como o sinal fica ligado por períodos longos, isso gera muito
          mais trades sobrepostos que os tipos 1-3 — é esperado, é o que
          torna esse sinal capaz de operar em mercado calmo.
        """
        holding = holding_days or self.holding_days

        if ativo_a not in self.retornos.columns:
            print(f"⚠️  {ativo_a} não encontrado nos retornos")
            return None
        if ativo_b not in self.retornos.columns:
            print(f"⚠️  {ativo_b} não encontrado nos retornos")
            return None

        ret_a = self.retornos[ativo_a].dropna()
        ret_b = self.retornos[ativo_b].dropna()
        idx   = ret_a.index.intersection(ret_b.index)

        if len(idx) < 50:
            print(f"⚠️  Poucos dados em comum: {len(idx)} dias")
            return None

        ra = ret_a[idx]
        rb = ret_b[idx]

        # Sinal: dias em que A dispara o padrão
        if momentum_meses > 0:
            # Tipo 4 — momentum acumulado. shift(1) evita lookahead: o
            # sinal do dia t só usa informação até o fechamento de t-1.
            janela = momentum_meses * 21
            if len(ra) < janela * 2:
                print(f"⚠️  Dados insuficientes para momentum de {momentum_meses}m")
                return None
            preco = (1 + ra / 100).cumprod()
            mom   = preco.pct_change(janela).shift(1)
            sinal = (mom > 0) if direcao_a == "sobe" else (mom < 0)
            sinal = sinal.fillna(False)
        elif direcao_a == "sobe":
            sinal = ra > self.threshold
        else:
            sinal = ra < -self.threshold

        # Retorno de B nos N dias após o sinal
        # Se direcao_b == "cai", fazemos short em B (retorno invertido)
        sinal_b = 1.0 if direcao_b == "sobe" else -1.0

        trades_ret = []
        trades_datas = []

        # Sinais lentos (tipo 4) ficam ligados por semanas seguidas. Sem
        # bloquear sobreposição, um holding de N dias abriria uma posição
        # NOVA a cada dia, e a curva de capital comporia o mesmo período de
        # retorno N vezes — o que produzia retornos impossíveis (da ordem
        # de 10^44%) e um "Sharpe" que não significava nada. Com o bloqueio,
        # o capital fica alocado em uma posição por vez, que é o que
        # aconteceria de verdade.
        # Para os tipos 1-3 (sinal raro, holding curto) o comportamento
        # segue o de antes, para não alterar resultados já validados.
        evitar_sobreposicao = momentum_meses > 0
        livre_a_partir_de = -1

        for i, data_sinal in enumerate(idx[:-holding]):
            if not sinal[data_sinal]:
                continue
            if evitar_sobreposicao and i < livre_a_partir_de:
                continue

            # Retorno acumulado de B nos próximos `holding` dias
            prox_dias = idx[i+1 : i+1+holding]
            if len(prox_dias) == 0:
                continue

            ret_trade = rb[prox_dias].sum()  # soma de retornos diários
            ret_ajustado = sinal_b * ret_trade - self.custo

            trades_ret.append(ret_ajustado)
            trades_datas.append(prox_dias[-1])
            livre_a_partir_de = i + 1 + holding

        if len(trades_ret) < 5:
            print(f"⚠️  Poucos trades: {len(trades_ret)}")
            return None

        trades_ret   = np.array(trades_ret)
        trades_datas = pd.DatetimeIndex(trades_datas)

        # ── Curva de capital ──────────────────────
        curva = pd.Series(
            (1 + trades_ret / 100).cumprod() * 100,
            index=trades_datas,
            name="capital"
        )

        # ── Métricas ──────────────────────────────
        n_trades      = len(trades_ret)
        win_rate      = (trades_ret > 0).mean()
        retorno_total = curva.iloc[-1] - 100
        retorno_medio = trades_ret.mean()
        n_anos        = (trades_datas[-1] - trades_datas[0]).days / 365.25
        retorno_anual = ((1 + retorno_total/100) ** (1/max(n_anos, 0.1)) - 1) * 100
        vol_anual     = trades_ret.std() * np.sqrt(252 / holding) if trades_ret.std() > 0 else 0

        # Sharpe bruto (sem taxa livre de risco)
        sharpe = (retorno_medio / trades_ret.std() * np.sqrt(252 / holding)
                  if trades_ret.std() > 0 else 0)

        # Drawdown máximo
        rolling_max   = curva.cummax()
        drawdown_ser  = (curva - rolling_max) / rolling_max * 100
        drawdown_max  = drawdown_ser.min()

        retorno_liquido = retorno_total  # já incluiu custos no loop

        # ── Benchmarks ────────────────────────────
        periodo_inicio = str(idx[0].date())
        periodo_fim    = str(idx[-1].date())

        # Sharpe ajustado — desconta CDI (ativos BR) ou FedFunds (ativos US)
        # Calculado após periodo_inicio/fim estarem definidos
        rf_diaria = self._rf_diaria(ativo_b, periodo_inicio, periodo_fim, holding)
        retorno_medio_ajustado = retorno_medio - rf_diaria * holding
        sharpe_ajustado = (retorno_medio_ajustado / trades_ret.std() * np.sqrt(252 / holding)
                           if trades_ret.std() > 0 else 0)

        bh_a = self._retorno_bh(ativo_a, periodo_inicio, periodo_fim)
        bh_b = self._retorno_bh(ativo_b, periodo_inicio, periodo_fim)
        ret_cdi = self._retorno_cdi(periodo_inicio, periodo_fim)

        return ResultadoBacktest(
            ativo_a=ativo_a, ativo_b=ativo_b,
            direcao_a=direcao_a, direcao_b=direcao_b,
            tipo_padrao=4 if momentum_meses > 0 else 1,
            threshold=self.threshold,
            holding_days=holding,
            n_trades=n_trades,
            win_rate=win_rate,
            retorno_total=retorno_total,
            retorno_medio=retorno_medio,
            retorno_anual=retorno_anual,
            vol_anual=vol_anual,
            sharpe=round(sharpe, 4),
            sharpe_ajustado=round(sharpe_ajustado, 4),
            drawdown_max=abs(drawdown_max),
            retorno_liquido=retorno_liquido,
            retorno_bh_a=bh_a,
            retorno_bh_b=bh_b,
            retorno_cdi=ret_cdi,
            curva_capital=curva,
        )

    def rodar_lista(
        self,
        padroes_df: pd.DataFrame,
        top_n: int = 20,
        holding_days: int = None,
    ) -> pd.DataFrame:
        """
        Roda backtests para os top_n padrões de um DataFrame de padrões.

        padroes_df: DataFrame retornado por correlator.relatorio()
        """
        print(f"\n{'='*60}")
        print(f"BACKTEST DOS TOP {top_n} PADRÕES")
        print(f"{'='*60}")

        resultados = []
        df = padroes_df.head(top_n)

        for _, row in df.iterrows():
            a, b = row["ativo_a"], row["ativo_b"]
            da, db = row["direcao_a"], row["direcao_b"]
            print(f"  {a} {da} → {b} {db}...", end=" ", flush=True)

            r = self.rodar(a, b, da, db, holding_days)
            if r:
                print(f"✅  liq={r.retorno_liquido:+.1f}% | sharpe={r.sharpe:.2f} | wr={r.win_rate:.0%}")
                resultados.append(r.resumo_linha())
            else:
                print("⚠️  ignorado")

        if not resultados:
            return pd.DataFrame()

        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values("sharpe", ascending=False)
        return df_res

    def plotar(
        self,
        resultado: ResultadoBacktest,
        salvar: bool = True,
        pasta: str = None,
    ) -> str:
        """
        Gera gráfico com 3 painéis:
          1. Curva de capital da estratégia vs CDI vs buy & hold B
          2. Retorno por trade (barras)
          3. Drawdown ao longo do tempo
        """
        fig = plt.figure(figsize=(14, 9), facecolor="#0f1117")
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.2],
                                hspace=0.35, figure=fig)

        cor_est   = "#4FC3F7"   # azul claro — estratégia
        cor_bh    = "#81C784"   # verde — buy & hold B
        cor_cdi   = "#FFB74D"   # laranja — CDI
        cor_pos   = "#4FC3F7"
        cor_neg   = "#EF5350"
        cor_dd    = "#EF5350"
        bg        = "#0f1117"
        bg_painel = "#1a1e2e"
        txt       = "#e0e0e0"
        txt_dim   = "#7a8099"
        grid_c    = "#2a2e3e"

        titulo = (
            f"{resultado.ativo_a} {resultado.direcao_a.upper()} "
            f"→ {resultado.ativo_b} {resultado.direcao_b.upper()} "
            f"| holding {resultado.holding_days}d"
        )

        # ── Painel 1: Curva de capital ─────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(bg_painel)

        curva = resultado.curva_capital
        ax1.plot(curva.index, curva.values,
                 color=cor_est, linewidth=1.8, label="Estratégia", zorder=3)

        # Buy & hold B normalizado
        rb = self.retornos[resultado.ativo_b].dropna()
        rb = rb[rb.index >= curva.index[0]]
        bh_norm = 100 * (1 + rb / 100).cumprod()
        ax1.plot(bh_norm.index, bh_norm.values,
                 color=cor_bh, linewidth=1.2, linestyle="--",
                 alpha=0.7, label=f"Buy & hold {resultado.ativo_b}")

        # CDI normalizado
        if self.cdi is not None and "CDI" in self.cdi.columns:
            cdi_p = self.cdi["CDI"]
            cdi_p = cdi_p[cdi_p.index >= curva.index[0]]
            cdi_norm = cdi_p / cdi_p.iloc[0] * 100
            ax1.plot(cdi_norm.index, cdi_norm.values,
                     color=cor_cdi, linewidth=1.0, linestyle=":",
                     alpha=0.6, label="CDI")

        ax1.axhline(100, color=grid_c, linewidth=0.8, linestyle="-")
        ax1.set_title(titulo, color=txt, fontsize=13, pad=10, fontweight="bold")
        ax1.set_ylabel("Capital (base 100)", color=txt_dim, fontsize=10)
        ax1.tick_params(colors=txt_dim)
        ax1.spines[["top","right","left","bottom"]].set_visible(False)
        ax1.yaxis.grid(True, color=grid_c, linewidth=0.5)
        ax1.set_axisbelow(True)
        leg = ax1.legend(fontsize=9, facecolor=bg_painel,
                         labelcolor=txt, edgecolor=grid_c)

        # Métricas no canto
        metr = (
            f"Retorno líq.: {resultado.retorno_liquido:+.1f}%  |  "
            f"Sharpe bruto: {resultado.sharpe:.2f}  |  "
            f"Sharpe adj.: {resultado.sharpe_ajustado:.2f}  |  "
            f"Win rate: {resultado.win_rate:.0%}  |  "
            f"Trades: {resultado.n_trades}  |  "
            f"Drawdown máx.: {resultado.drawdown_max:.1f}%"
        )
        ax1.text(0.01, 0.03, metr, transform=ax1.transAxes,
                 fontsize=8.5, color=txt_dim,
                 bbox=dict(facecolor=bg, alpha=0.6, edgecolor="none", pad=3))

        # ── Painel 2: Retorno por trade ────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(bg_painel)

        curva_ret = resultado.curva_capital.pct_change() * 100
        curva_ret = curva_ret.dropna()
        cores_bar = [cor_pos if v >= 0 else cor_neg for v in curva_ret]
        ax2.bar(curva_ret.index, curva_ret.values,
                color=cores_bar, width=2, alpha=0.85)
        ax2.axhline(0, color=grid_c, linewidth=0.8)
        ax2.set_ylabel("Ret. por trade (%)", color=txt_dim, fontsize=9)
        ax2.tick_params(colors=txt_dim)
        ax2.spines[["top","right","left","bottom"]].set_visible(False)
        ax2.yaxis.grid(True, color=grid_c, linewidth=0.5)
        ax2.set_axisbelow(True)

        # ── Painel 3: Drawdown ─────────────────────
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(bg_painel)

        rolling_max = resultado.curva_capital.cummax()
        dd = (resultado.curva_capital - rolling_max) / rolling_max * 100
        ax3.fill_between(dd.index, dd.values, 0,
                         color=cor_dd, alpha=0.4)
        ax3.plot(dd.index, dd.values, color=cor_dd, linewidth=0.8)
        ax3.set_ylabel("Drawdown (%)", color=txt_dim, fontsize=9)
        ax3.tick_params(colors=txt_dim)
        ax3.spines[["top","right","left","bottom"]].set_visible(False)
        ax3.yaxis.grid(True, color=grid_c, linewidth=0.5)
        ax3.set_axisbelow(True)

        plt.tight_layout()

        if salvar:
            pasta = pasta or os.path.join(
                os.path.dirname(__file__), "..", "reports"
            )
            os.makedirs(pasta, exist_ok=True)
            nome = (
                f"bt_{resultado.ativo_a.replace('/','-')}_"
                f"{resultado.ativo_b.replace('/','-').replace('.','')}_"
                f"{resultado.direcao_a}_{resultado.direcao_b}.png"
            )
            caminho = os.path.join(pasta, nome)
            plt.savefig(caminho, dpi=150, bbox_inches="tight",
                        facecolor=bg)
            plt.close()
            print(f"   📊 Gráfico salvo: {caminho}")
            return caminho

        plt.show()
        plt.close()
        return ""

    # ═══════════════════════════════════════════════════════════════
    # VARIAÇÃO 1 — HOLDING SWEEP
    # Testa múltiplos períodos de holding para um padrão
    # ═══════════════════════════════════════════════════════════════

    def holding_sweep(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        holdings: list = None,
        threshold: float = None,
    ) -> pd.DataFrame:
        """
        Testa o mesmo padrão com diferentes holding periods e retorna
        uma tabela comparativa.

        Útil para identificar em quantos dias o sinal se esgota ou
        se fortalece — como encontramos que BZ=F→PRIO3 vira positivo
        apenas com holding=3d.
        """
        holdings = holdings or [1, 2, 3, 5, 7, 10]
        thr_orig = self.threshold
        if threshold:
            self.threshold = threshold

        linhas = []
        for h in holdings:
            r = self.rodar(ativo_a, ativo_b, direcao_a, direcao_b, holding_days=h)
            if r:
                linhas.append({
                    "holding_days":     h,
                    "trades":           r.n_trades,
                    "win_rate_%":       round(r.win_rate * 100, 1),
                    "retorno_liq_%":    round(r.retorno_liquido, 1),
                    "retorno_anual_%":  round(r.retorno_anual, 1),
                    "sharpe":           round(r.sharpe, 2),
                    "drawdown_max_%":   round(r.drawdown_max, 1),
                })

        self.threshold = thr_orig
        df = pd.DataFrame(linhas)
        if df.empty:
            return df

        # ── Score de estabilidade de parâmetros ──────────────────
        # Evita cherry-picking do melhor holding isolado.
        # Prefere regiões estáveis onde múltiplos holdings vizinhos
        # também têm Sharpe positivo.
        #
        # Fórmula:
        #   estabilidade = media(sharpes_positivos) / (1 + std(todos_sharpes))
        #
        # Interpretação:
        #   Alto = região estável (vários holdings funcionam)
        #   Baixo = pico isolado (só um holding funciona, risco de overfitting)
        sharpes = df["sharpe"].values
        sharpes_pos = sharpes[sharpes > 0]
        media_pos = sharpes_pos.mean() if len(sharpes_pos) > 0 else 0.0
        std_todos  = sharpes.std() if len(sharpes) > 1 else 1.0
        df["estabilidade"] = round(media_pos / (1 + std_todos), 4)

        # Conta quantos holdings consecutivos têm Sharpe > 0
        n_positivos = int((sharpes > 0).sum())
        df["n_holdings_positivos"] = n_positivos

        return df

    # ═══════════════════════════════════════════════════════════════
    # VARIAÇÃO 2 — THRESHOLD SWEEP
    # Testa múltiplos thresholds de magnitude do sinal
    # ═══════════════════════════════════════════════════════════════

    def threshold_sweep(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        thresholds: list = None,
        holding_days: int = None,
    ) -> pd.DataFrame:
        """
        Testa o mesmo padrão filtrando apenas movimentos maiores de A.

        Hipótese: movimentos pequenos de A são ruído; apenas movimentos
        grandes (>1.5%, >2%) carregam informação preditiva suficiente
        para justificar uma operação em B.

        Efeito colateral esperado: menos trades, mas win rate maior.
        """
        thresholds = thresholds or [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        holding = holding_days or self.holding_days
        thr_orig = self.threshold

        linhas = []
        for thr in thresholds:
            self.threshold = thr
            r = self.rodar(ativo_a, ativo_b, direcao_a, direcao_b, holding_days=holding)
            if r:
                linhas.append({
                    "threshold_%":      thr,
                    "trades":           r.n_trades,
                    "win_rate_%":       round(r.win_rate * 100, 1),
                    "retorno_liq_%":    round(r.retorno_liquido, 1),
                    "retorno_anual_%":  round(r.retorno_anual, 1),
                    "sharpe":           round(r.sharpe, 2),
                    "drawdown_max_%":   round(r.drawdown_max, 1),
                })

        self.threshold = thr_orig
        df = pd.DataFrame(linhas)
        if df.empty:
            return df

        # ── Score de estabilidade de parâmetros ──────────────────
        # Mesma lógica do holding_sweep: evita escolher o threshold de
        # pico isolado (fácil de ser sorte/overfitting) em vez de uma
        # região onde vários thresholds vizinhos também funcionam.
        sharpes = df["sharpe"].values
        sharpes_pos = sharpes[sharpes > 0]
        media_pos = sharpes_pos.mean() if len(sharpes_pos) > 0 else 0.0
        std_todos  = sharpes.std() if len(sharpes) > 1 else 1.0
        df["estabilidade"] = round(media_pos / (1 + std_todos), 4)
        df["n_thresholds_positivos"] = int((sharpes > 0).sum())

        return df

    # ═══════════════════════════════════════════════════════════════
    # VARIAÇÃO 3 — MEAN REVERSION (direção contrária)
    # Testa se o mercado sobrerreage e reverte
    # ═══════════════════════════════════════════════════════════════

    def rodar_contrarian(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        holding_days: int = None,
        threshold: float = None,
    ) -> Optional[ResultadoBacktest]:
        """
        Estratégia contrarian (mean reversion):
          Quando A sobe muito (> threshold), apostamos que B VAI CAI
          (ou vice-versa) — ou seja, invertemos a direção esperada.

        Hipótese: o mercado sobrerreage a movimentos grandes de A,
        e B reverte na direção oposta nos dias seguintes.

        Isso é diferente da estratégia momentum (que segue a direção de A).
        Funciona melhor em ativos com alta volatilidade e baixa tendência.

        A direção de B é automaticamente invertida em relação a direcao_a:
          - direcao_a = "sobe" → posição SHORT em B
          - direcao_a = "cai"  → posição LONG em B
        """
        # Direção contrária à de A
        direcao_b_contrarian = "cai" if direcao_a == "sobe" else "sobe"

        thr_orig = self.threshold
        if threshold:
            self.threshold = threshold

        resultado = self.rodar(
            ativo_a, ativo_b,
            direcao_a, direcao_b_contrarian,
            holding_days=holding_days,
        )

        self.threshold = thr_orig

        if resultado:
            # Marca no resultado que é estratégia contrarian
            resultado.direcao_b = f"CONTRARIAN({direcao_b_contrarian})"

        return resultado

    def varrer_contrarian(
        self,
        padroes: list,
        thresholds: list = None,
        holdings: list = None,
    ) -> pd.DataFrame:
        """
        Varre uma lista de padrões testando estratégia contrarian
        com múltiplos thresholds e holdings.

        padroes: lista de tuplas (ativo_a, ativo_b, direcao_a)
        """
        thresholds = thresholds or [1.0, 1.5, 2.0, 3.0]
        holdings   = holdings   or [1, 2, 3]
        thr_orig   = self.threshold

        linhas = []
        for a, b, da in tqdm(padroes, desc="Exp 3 — Contrarian", unit="padrão"):
            for thr in thresholds:
                for h in holdings:
                    self.threshold = thr
                    r = self.rodar_contrarian(a, b, da, holding_days=h)
                    if r and r.n_trades >= 10:
                        linhas.append({
                            "ativo_a":         a,
                            "ativo_b":         b,
                            "direcao_a":       da,
                            "estrategia":      "contrarian",
                            "threshold_%":     thr,
                            "holding_days":    h,
                            "trades":          r.n_trades,
                            "win_rate_%":      round(r.win_rate * 100, 1),
                            "retorno_liq_%":   round(r.retorno_liquido, 1),
                            "retorno_anual_%": round(r.retorno_anual, 1),
                            "sharpe":          round(r.sharpe, 2),
                            "drawdown_max_%":  round(r.drawdown_max, 1),
                        })

        self.threshold = thr_orig
        df = pd.DataFrame(linhas)
        if not df.empty:
            df = df.sort_values("sharpe", ascending=False)
        return df

    # ── Helpers ───────────────────────────────────

    def _rf_diaria(
        self,
        ativo_b: str,
        inicio: str,
        fim: str,
        holding: int,
    ) -> float:
        """
        Retorna a taxa livre de risco média diária (%) para o período,
        escolhendo CDI ou FedFunds conforme o tipo de ativo B.

        Regra:
          - Ativo BR (.SA ou fundo CVM): usa CDI
          - Qualquer outro             : usa FedFunds
        """
        # Escolhe a fonte certa
        eh_br = ativo_b.endswith(".SA") or ativo_b in {
            "Guepardo Institucional Master FIA",
            "Alaska Black Master FIF Ações",
            "Constellation Compounders Master FIA",
        }

        if eh_br and self.cdi is not None:
            col = "taxa_dia_%" if "taxa_dia_%" in self.cdi.columns else None
            if col is None and "CDI" in self.cdi.columns:
                # Deriva taxa diária do índice acumulado
                s = self.cdi["CDI"].dropna()
                s = s[(s.index >= inicio) & (s.index <= fim)]
                if len(s) > 1:
                    retornos_diarios = s.pct_change().dropna() * 100
                    return float(retornos_diarios.mean())
                return 0.0
            elif col:
                s = self.cdi[col].dropna()
                s = s[(s.index >= inicio) & (s.index <= fim)]
                return float(s.mean()) if len(s) > 0 else 0.0

        elif not eh_br and self.fedfunds is not None:
            col = "taxa_dia_%" if "taxa_dia_%" in self.fedfunds.columns else None
            if col is None and "FedFunds" in self.fedfunds.columns:
                s = self.fedfunds["FedFunds"].dropna()
                s = s[(s.index >= inicio) & (s.index <= fim)]
                if len(s) > 1:
                    retornos_diarios = s.pct_change().dropna() * 100
                    return float(retornos_diarios.mean())
                return 0.0
            elif col:
                s = self.fedfunds[col].dropna()
                s = s[(s.index >= inicio) & (s.index <= fim)]
                return float(s.mean()) if len(s) > 0 else 0.0

        return 0.0

    def _retorno_bh(self, ativo: str, inicio: str, fim: str) -> float:
        if ativo not in self.retornos.columns:
            return 0.0
        s = self.retornos[ativo].dropna()
        s = s[(s.index >= inicio) & (s.index <= fim)]
        if len(s) < 2:
            return 0.0
        return float(((1 + s / 100).prod() - 1) * 100)

    def _retorno_cdi(self, inicio: str, fim: str) -> float:
        if self.cdi is None:
            return 0.0
        col = "CDI" if "CDI" in self.cdi.columns else self.cdi.columns[0]
        s = self.cdi[col].dropna()
        s = s[(s.index >= inicio) & (s.index <= fim)]
        if len(s) < 2:
            return 0.0
        # Se é índice acumulado
        if s.mean() > 10:
            return float((s.iloc[-1] / s.iloc[0] - 1) * 100)
        # Se é taxa diária
        return float(((1 + s / 100).prod() - 1) * 100)


# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA — EXPERIMENTOS COMPLETOS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from analysis.correlator import _carregar_retornos_completos

    _BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_BASE)
    from modo_teste import adicionar_flag, HOLDINGS_TESTE, THRESHOLDS_TESTE, banner

    parser = argparse.ArgumentParser(description="Backtester — Exp1 (holding sweep), Exp2 (threshold sweep), Exp3 (contrarian)")
    adicionar_flag(parser)
    args = parser.parse_args()
    banner("backtester.py", args.teste)

    PASTA_RAW     = os.path.join(_BASE, "data", "raw")
    PASTA_REPORTS = os.path.join(_BASE, "reports", "teste") if args.teste \
                    else os.path.join(_BASE, "reports")
    os.makedirs(PASTA_REPORTS, exist_ok=True)

    HOLDINGS_SWEEP   = HOLDINGS_TESTE   if args.teste else [1, 2, 3, 5, 7, 10]
    THRESHOLDS_SWEEP_EXP2 = THRESHOLDS_TESTE if args.teste else [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    THRESHOLDS_SWEEP_EXP3 = THRESHOLDS_TESTE if args.teste else [0.5, 1.0, 1.5, 2.0, 3.0]
    HOLDINGS_SWEEP_EXP3   = HOLDINGS_TESTE[:2] if args.teste else [1, 2, 3]
    MAX_PADROES_TESTE = 15  # em modo teste, roda só os N melhores por score — não todos

    # ── Carrega dados ─────────────────────────────
    retornos = _carregar_retornos_completos(PASTA_RAW)

    cdi_path = os.path.join(PASTA_RAW, "cdi.csv")
    cdi = pd.read_csv(cdi_path, index_col=0, parse_dates=True) if os.path.exists(cdi_path) else None

    ff_path = os.path.join(PASTA_RAW, "fedfunds.csv")
    fedfunds = pd.read_csv(ff_path, index_col=0, parse_dates=True) if os.path.exists(ff_path) else None

    bt = Backtester(
        retornos=retornos,
        cdi=cdi,
        fedfunds=fedfunds,
        custo_por_trade=0.15,
        threshold=0.3,
        holding_days=1,
    )

    # ── Carrega padrões do correlator ────────────
    # Lê automaticamente o padroes.csv — funciona com qualquer modo
    # (MODO_TESTE, MODO_RAPIDO ou completo) pois usa o que estiver disponível.
    padroes_path = os.path.join(PASTA_REPORTS, "padroes.csv")
    if not os.path.exists(padroes_path):
        raise FileNotFoundError(
            f"padroes.csv não encontrado em {PASTA_REPORTS}\n"
            "Execute o correlator.py primeiro para gerar os padrões."
        )

    df_padroes = pd.read_csv(padroes_path)
    print(f"\n📂 Carregados {len(df_padroes)} padrões de {padroes_path}")
    print(f"   Tipos: {df_padroes['tipo'].value_counts().to_dict()}")

    # Guarda os tipo 4 (momentum) ANTES do filtro de qualidade abaixo.
    # O filtro (score ≥ 0.85, n_ocorrencias ≥ 100) foi calibrado para os
    # tipos 1-3, que contam ocorrências DIÁRIAS. O tipo 4 usa observações
    # não-sobrepostas de ~1 mês, então em 10 anos gera no máximo ~120
    # ocorrências e score naturalmente mais baixo — aplicar o mesmo corte
    # eliminaria 100% deles por um artefato de unidade, não por qualidade.
    df_padroes_full = df_padroes.copy()

    # ── FILTRO DE QUALIDADE ESTATÍSTICA ──────────────────────────
    # Aplica antes de qualquer outra coisa para reduzir volume
    # Padrões com score < 0.85 ou n_ocorrencias < 100 dificilmente
    # sobreviveriam ao walk-forward (exige sharpe > 0.3 e winrate > 50%
    # fora da amostra) — removê-los aqui não descarta oportunidades reais.
    SCORE_MIN      = 0.85   # frequência de confirmação mínima (~65% ajustado BH)
    OCORRENCIAS_MIN = 100   # mínimo de eventos históricos para ser estatisticamente robusto

    antes = len(df_padroes)
    df_padroes = df_padroes[
        (df_padroes["tipo"] != 4) &                    # tipo 4 tem filtro próprio (Exp 4)
        (df_padroes["score"] >= SCORE_MIN) &
        (df_padroes["n_ocorrencias"] >= OCORRENCIAS_MIN)
    ].copy()
    print(f"\n🔍 Filtro de qualidade (score ≥ {SCORE_MIN}, n_ocorrencias ≥ {OCORRENCIAS_MIN}):")
    print(f"   {antes:,} → {len(df_padroes):,} padrões  ({antes - len(df_padroes):,} removidos, {len(df_padroes)/antes*100:.1f}% mantidos)")
    print(f"   Tipos restantes: {df_padroes['tipo'].value_counts().to_dict()}")

    # Filtra apenas padrões cujos ativos existem na base de retornos
    ativos_disponiveis = set(retornos.columns)
    mask = (
        df_padroes["ativo_a"].isin(ativos_disponiveis) &
        df_padroes["ativo_b"].isin(ativos_disponiveis)
    )
    df_padroes = df_padroes[mask].copy()
    print(f"   Aprovados (ativos disponíveis): {len(df_padroes):,}")

    # Monta lista de tuplas (ativo_a, ativo_b, direcao_a)
    # Deduplica: para o backtester, só precisamos do sinal de A (direcao_a)
    # A direcao_b é testada automaticamente nos experimentos
    vistos = set()
    PADROES = []
    for _, row in df_padroes.iterrows():
        chave = (row["ativo_a"], row["ativo_b"], row["direcao_a"])
        if chave not in vistos:
            vistos.add(chave)
            PADROES.append(chave)

    print(f"   Padrões únicos para backtest: {len(PADROES)}")

    if args.teste and len(PADROES) > MAX_PADROES_TESTE:
        # Em modo teste, roda só os N com maior score — o objetivo é
        # confirmar que o código funciona, não descobrir a melhor estratégia.
        df_padroes_sorted = df_padroes.sort_values("score", ascending=False)
        vistos_teste, PADROES_TESTE = set(), []
        for _, row in df_padroes_sorted.iterrows():
            chave = (row["ativo_a"], row["ativo_b"], row["direcao_a"])
            if chave not in vistos_teste:
                vistos_teste.add(chave)
                PADROES_TESTE.append(chave)
            if len(PADROES_TESTE) >= MAX_PADROES_TESTE:
                break
        PADROES = PADROES_TESTE
        print(f"   🧪 Modo teste: reduzido para os {len(PADROES)} melhores padrões (por score)")

    # ════════════════════════════════════════════════
    # EXPERIMENTO 1 — HOLDING SWEEP
    # Pergunta: em quantos dias o sinal se converte em lucro?
    # ════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("EXPERIMENTO 1 — HOLDING SWEEP (todos os padrões)")
    print("═"*65)

    resultados_holding = []
    for i, (a, b, da) in enumerate(tqdm(PADROES, desc="Exp 1 — Holding Sweep", unit="padrão")):
        # Para momentum (direção seguida)
        db = da  # mesma direção
        df_sw = bt.holding_sweep(a, b, da, db, holdings=HOLDINGS_SWEEP)
        if not df_sw.empty:
            # Usa estabilidade para escolher o melhor holding,
            # não só o Sharpe máximo isolado
            if "estabilidade" in df_sw.columns:
                melhor = df_sw.loc[df_sw["estabilidade"].idxmax()]
            else:
                melhor = df_sw.loc[df_sw["sharpe"].idxmax()]
            resultados_holding.append({
                "padrao":                f"{a} {da} → {b}",
                "melhor_holding":        int(melhor["holding_days"]),
                "trades":                int(melhor["trades"]),
                "win_rate_%":            melhor["win_rate_%"],
                "retorno_liq_%":         melhor["retorno_liq_%"],
                "sharpe":                melhor["sharpe"],
                "drawdown_max_%":        melhor["drawdown_max_%"],
                "estabilidade":          round(melhor.get("estabilidade", 0), 4),
                "n_holdings_positivos":  int(melhor.get("n_holdings_positivos", 0)),
            })

        # Checkpoint a cada 50 padrões — não perde progresso se interromper
        if (i + 1) % 50 == 0 and resultados_holding:
            _ckpt_path = os.path.join(PASTA_REPORTS, "exp1_holding_sweep_checkpoint.csv")
            pd.DataFrame(resultados_holding).to_csv(_ckpt_path, index=False)
            # tqdm já mostra o progresso — não precisamos de print aqui

    df_h = pd.DataFrame(resultados_holding).sort_values("sharpe", ascending=False)
    print("\n\n── RANKING: MELHOR HOLDING POR PADRÃO ──")
    print(df_h.to_string(index=False))
    df_h.to_csv(os.path.join(PASTA_REPORTS, "exp1_holding_sweep.csv"), index=False)
    print(f"\n💾 exp1_holding_sweep.csv salvo")

    # ════════════════════════════════════════════════
    # EXPERIMENTO 2 — THRESHOLD SWEEP
    # Pergunta: movimentos maiores de A têm mais poder preditivo?
    # ════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("EXPERIMENTO 2 — THRESHOLD SWEEP (todos os padrões, holding=3d)")
    print("═"*65)

    resultados_threshold = []
    for i, (a, b, da) in enumerate(tqdm(PADROES, desc="Exp 2 — Threshold Sweep", unit="padrão")):
        db = da
        df_thr = bt.threshold_sweep(a, b, da, db,
                                    thresholds=THRESHOLDS_SWEEP_EXP2,
                                    holding_days=3)
        if not df_thr.empty:
            # Usa estabilidade para escolher o threshold, não só o Sharpe
            # máximo isolado — mesma lógica do Exp1 (holding sweep).
            if "estabilidade" in df_thr.columns:
                melhor = df_thr.loc[df_thr["estabilidade"].idxmax()]
            else:
                melhor = df_thr.loc[df_thr["sharpe"].idxmax()]
            resultados_threshold.append({
                "padrao":                  f"{a} {da} → {b}",
                "melhor_threshold":        melhor["threshold_%"],
                "trades":                  int(melhor["trades"]),
                "win_rate_%":              melhor["win_rate_%"],
                "retorno_liq_%":           melhor["retorno_liq_%"],
                "sharpe":                  melhor["sharpe"],
                "drawdown_max_%":          melhor["drawdown_max_%"],
                "estabilidade":            round(melhor.get("estabilidade", 0), 4),
                "n_thresholds_positivos":  int(melhor.get("n_thresholds_positivos", 0)),
            })

        # Checkpoint a cada 50 padrões
        if (i + 1) % 50 == 0 and resultados_threshold:
            _ckpt_path = os.path.join(PASTA_REPORTS, "exp2_threshold_sweep_checkpoint.csv")
            pd.DataFrame(resultados_threshold).to_csv(_ckpt_path, index=False)

    df_t = pd.DataFrame(resultados_threshold).sort_values("sharpe", ascending=False)
    print("\n\n── RANKING: MELHOR THRESHOLD POR PADRÃO ──")
    print(df_t.to_string(index=False))
    df_t.to_csv(os.path.join(PASTA_REPORTS, "exp2_threshold_sweep.csv"), index=False)
    print(f"\n💾 exp2_threshold_sweep.csv salvo")

    # ════════════════════════════════════════════════
    # EXPERIMENTO 3 — CONTRARIAN (mean reversion)
    # Pergunta: o mercado sobrerreage e B reverte na direção oposta?
    # ════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("EXPERIMENTO 3 — CONTRARIAN / MEAN REVERSION")
    print("═"*65)

    df_contra = bt.varrer_contrarian(
        padroes=PADROES,
        thresholds=THRESHOLDS_SWEEP_EXP3,
        holdings=HOLDINGS_SWEEP_EXP3,
    )

    if not df_contra.empty:
        print("\nTodos os resultados contrarian positivos (sharpe > 0):")
        positivos = df_contra[df_contra["sharpe"] > 0]
        if not positivos.empty:
            print(positivos.to_string(index=False))
        else:
            print("  Nenhum resultado contrarian com sharpe > 0.")

        print(f"\nTop 20 contrarian (por sharpe):")
        print(df_contra.head(20).to_string(index=False))

        df_contra.to_csv(os.path.join(PASTA_REPORTS, "exp3_contrarian.csv"), index=False)

    # ════════════════════════════════════════════════
    # EXPERIMENTO 4 — MOMENTUM DE MÉDIO PRAZO (tipo 4)
    # Pergunta: uma tendência de 3-12 meses continua nas próximas semanas?
    # Diferente dos Exp 1-3: o sinal não depende de choque de curto prazo,
    # então funciona também em mercado calmo (ver correlator.detectar_tipo4).
    # ════════════════════════════════════════════════
    df_padroes_t4 = df_padroes_full[df_padroes_full["tipo"] == 4] \
                    if "tipo" in df_padroes_full.columns else pd.DataFrame()

    if not df_padroes_t4.empty:
        print("\n" + "═"*65)
        print(f"EXPERIMENTO 4 — MOMENTUM DE MÉDIO PRAZO ({len(df_padroes_t4)} padrões tipo 4)")
        print("═"*65)

        HOLDINGS_T4 = [10, 21] if args.teste else [10, 21, 42]
        resultados_mom = []

        for _, row in tqdm(list(df_padroes_t4.iterrows()),
                           desc="Exp 4 — Momentum", unit="padrão"):
            a  = str(row["ativo_a"])
            da = str(row["direcao_a"])
            mm = int(row.get("momentum_meses", 0) or 0)
            if not a or mm <= 0 or a not in retornos.columns:
                continue

            for h in HOLDINGS_T4:
                r = bt.rodar(a, a, da, str(row["direcao_b"]),
                             holding_days=h, momentum_meses=mm)
                if r:
                    resultados_mom.append({
                        "ativo_a":        a,
                        "ativo_b":        a,
                        "direcao_a":      da,
                        "direcao_b":      str(row["direcao_b"]),
                        "momentum_meses": mm,
                        "holding_days":   h,
                        "trades":         r.n_trades,
                        "win_rate_%":     round(r.win_rate * 100, 1),
                        "retorno_liq_%":  round(r.retorno_liquido, 1),
                        "sharpe":         round(r.sharpe, 2),
                        "drawdown_max_%": round(r.drawdown_max, 1),
                    })

        if resultados_mom:
            df_mom = pd.DataFrame(resultados_mom).sort_values("sharpe", ascending=False)
            print("\n── RANKING: MOMENTUM DE MÉDIO PRAZO ──")
            print(df_mom.head(30).to_string(index=False))
            df_mom.to_csv(os.path.join(PASTA_REPORTS, "exp4_momentum.csv"), index=False)
            print(f"\n💾 exp4_momentum.csv salvo ({len(df_mom)} linhas)")
        else:
            print("  Nenhum resultado de momentum válido.")
    else:
        print("\n(Nenhum padrão tipo 4 em padroes.csv — Exp 4 pulado. "
              "Rode o correlator novamente para gerá-los.)")