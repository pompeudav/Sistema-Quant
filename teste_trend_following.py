"""
teste_trend_following.py
------------------------
TESTE DE TESE — um trend following diversificado tem edge nos dados
disponíveis, ou não?

Por que este teste existe
=========================
O motor de correlação de pares (correlator tipos 1/2/3) falhou em três
medições independentes:
  - win rate caiu de 57.8% (período de seleção) para 50.2% (dados novos)
  - Sharpe líquido do CDI no paper trading: 0.20
  - walk-forward de descoberta: -5.2pp ao ano contra o CDI
A razão é conceitual: não há mecanismo econômico que explique por que um
ativo preveria outro com dias de antecedência. Era mineração de dados
sobre 96 mil pares.

Trend following é diferente: tem mecanismo (investidores reagem devagar a
informação nova e rápido demais a preço), tem 30+ anos de evidência
publicada, e foi a única família que mostrou sinal aqui (o Tipo 4 teve
win rate 53.7% e 4.95x retorno por unidade de custo contra 1.39x dos pares).

REGRAS DO TESTE (desenhadas para NÃO repetir o viés anterior)
============================================================
1. UMA regra só: momentum de K meses -> long se positivo, short se negativo.
2. Parâmetros da literatura, não otimizados: K = 3, 6 e 12 meses.
3. Reporta TODAS as janelas testadas. Nada de escolher a melhor e mostrar
   só ela — foi exatamente assim que o sistema anterior se enganou.
4. Sizing por inversa da volatilidade (risk parity simples), com alvo de
   volatilidade fixo. Sem Kelly, sem otimização.
5. Rebalanceamento mensal, para manter custos baixos.
6. Nenhum ativo é escolhido por desempenho: opera o universo inteiro.

Se uma regra tão simples e sem ajuste fino entregar Sharpe > 0.5, a tese
se sustenta e vale construir em cima. Se não entregar, o caminho não é
esse — e o teste custou horas, não meses.

Uso:
    python teste_trend_following.py
    python teste_trend_following.py --vol-alvo 10 --custo 0.10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "analysis"))

# Universo: 37 ativos em 6 classes. Escolhido por REPRESENTATIVIDADE de
# classe de ativo, não por desempenho histórico — critério que não pode
# introduzir viés de seleção.
UNIVERSO = {
    "indices":     ["^GSPC", "^NDX", "^BVSP", "^RUT", "^FTSE", "^N225",
                    "EFA", "EEM", "SPY", "QQQ", "IWM"],
    "juros":       ["TLT", "IEF", "HYG", "LQD", "IMAB11.SA", "IRFM11.SA"],
    "moedas":      ["BRL=X", "EUR=X", "JPY=X", "DX-Y.NYB"],
    "commodities": ["GC=F", "SI=F", "CL=F", "BZ=F", "HG=F", "NG=F",
                    "ZS=F", "ZC=F", "ZW=F", "KC=F", "SB=F", "CT=F"],
    "cripto":      ["BTC-USD", "ETH-USD", "SOL-USD"],
}

JANELAS_MOMENTUM = [3, 6, 12]   # meses — as janelas clássicas da literatura
DIAS_MES = 21


def simular(retornos: pd.DataFrame, ativos: list, janela_meses: int,
            vol_alvo_anual: float = 10.0, custo_pct: float = 0.10,
            rebal_dias: int = DIAS_MES, permitir_short: bool = True) -> dict:
    """
    Simula o trend following e devolve a série de retorno diário do
    conjunto, já líquida de custos.

    Sinal (sem lookahead): no dia t, olha o retorno acumulado dos
    `janela_meses` anteriores, encerrado em t-1. Long se positivo, short
    se negativo. A posição vale até o próximo rebalanceamento.

    Sizing: inversa da volatilidade de cada ativo, escalada para que cada
    posição carregue aproximadamente `vol_alvo_anual` de risco. Isso
    impede que cripto (vol ~60%) domine o portfólio sobre juros (vol ~10%).
    """
    disp = [a for a in ativos if a in retornos.columns]
    if not disp:
        return {}

    ret = retornos[disp].fillna(0.0)
    janela = janela_meses * DIAS_MES
    if len(ret) < janela * 2:
        return {}

    precos = (1 + ret / 100).cumprod()

    # Momentum defasado em 1 dia: decisão de hoje usa dados até ontem.
    momentum = precos.pct_change(janela).shift(1)

    # Volatilidade recente (60d), também defasada — usada para dimensionar.
    vol = ret.rolling(60, min_periods=30).std().shift(1)
    vol = vol.replace(0, np.nan)

    # Peso alvo: (vol_alvo_diaria / vol_do_ativo), limitado para evitar
    # posições enormes em ativos de vol muito baixa.
    vol_alvo_diaria = vol_alvo_anual / np.sqrt(252)
    peso_bruto = (vol_alvo_diaria / vol).clip(upper=3.0)

    if permitir_short:
        direcao = np.sign(momentum)
    else:
        direcao = (momentum > 0).astype(float)

    pesos_alvo = (direcao * peso_bruto).fillna(0.0)

    # Rebalanceamento periódico: mantém os pesos entre datas de rebalance,
    # em vez de reposicionar todo dia (o que multiplicaria os custos).
    pesos = pesos_alvo.copy()
    mask = np.zeros(len(pesos), dtype=bool)
    mask[::rebal_dias] = True
    pesos = pesos.where(pd.Series(mask, index=pesos.index), np.nan).ffill().fillna(0.0)

    # Divide pelo número de ativos: capital repartido, não alavancado
    pesos = pesos / max(len(disp), 1)

    # Retorno bruto do dia
    ret_bruto = (pesos * ret).sum(axis=1)

    # Custo: cobrado sobre o giro (mudança de posição) nos rebalanceamentos
    turnover = pesos.diff().abs().sum(axis=1).fillna(0.0)
    custo_dia = turnover * custo_pct
    ret_liq = ret_bruto - custo_dia

    return {
        "retorno": ret_liq,
        "turnover_total": float(turnover.sum()),
        "custo_total": float(custo_dia.sum()),
        "exposicao_media": float(pesos.abs().sum(axis=1).mean()),
        "n_ativos": len(disp),
    }


def metricas(serie: pd.Series, cdi_df: pd.DataFrame = None) -> dict:
    if serie is None or serie.empty or serie.std() == 0:
        return {}
    cap = (1 + serie / 100).cumprod()
    anos = (serie.index[-1] - serie.index[0]).days / 365.25
    obs_ano = len(serie) / anos
    m = {
        "cagr": (cap.iloc[-1] ** (1 / anos) - 1) * 100,
        "vol": serie.std() * np.sqrt(obs_ano),
        "sharpe": serie.mean() / serie.std() * np.sqrt(obs_ano),
        "dd": ((cap - cap.cummax()) / cap.cummax()).min() * 100,
        "anos": anos,
    }
    # Sharpe líquido do CDI — o número que importa no Brasil
    if cdi_df is not None and not cdi_df.empty:
        col = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
        cdi_d = cdi_df[col].pct_change().reindex(serie.index).fillna(0) * 100
        exc = serie - cdi_d
        if exc.std() > 0:
            m["sharpe_liq"] = exc.mean() / exc.std() * np.sqrt(obs_ano)
        c = cdi_df[col].loc[serie.index[0]:serie.index[-1]].dropna()
        if len(c) > 1:
            m["cdi_cagr"] = ((c.iloc[-1] / c.iloc[0]) ** (1 / anos) - 1) * 100
    return m


def main():
    ap = argparse.ArgumentParser(description="Teste de tese: trend following diversificado")
    ap.add_argument("--vol-alvo", type=float, default=10.0,
                    help="Volatilidade alvo por posicao, %% ao ano (default 10)")
    ap.add_argument("--custo", type=float, default=0.10,
                    help="Custo por unidade de giro, %% (default 0.10)")
    ap.add_argument("--long-only", action="store_true",
                    help="Desabilita short (so compra em tendencia de alta)")
    args = ap.parse_args()

    from correlator import _carregar_retornos_completos
    retornos = _carregar_retornos_completos(os.path.join(BASE, "data", "raw"))

    caminho_cdi = os.path.join(BASE, "data", "raw", "cdi.csv")
    cdi_df = (pd.read_csv(caminho_cdi, index_col=0, parse_dates=True)
              if os.path.exists(caminho_cdi) else None)

    todos = [a for lista in UNIVERSO.values() for a in lista]
    disp = [a for a in todos if a in retornos.columns]

    print("=" * 74)
    print("TESTE DE TESE — TREND FOLLOWING DIVERSIFICADO")
    print("=" * 74)
    print(f"  Universo   : {len(disp)} ativos em {len(UNIVERSO)} classes")
    print(f"  Historico  : {retornos.index[0].date()} -> {retornos.index[-1].date()}")
    print(f"  Vol alvo   : {args.vol_alvo}% a.a. por posicao")
    print(f"  Custo      : {args.custo}% por unidade de giro")
    print(f"  Direcao    : {'long-only' if args.long_only else 'long/short'}")
    print(f"  Rebalance  : mensal (a cada {DIAS_MES} pregoes)")
    print("\n  Sem otimizacao: todas as janelas testadas sao reportadas.")

    print("\n" + "=" * 74)
    print(f"  {'Janela':>8} {'CAGR':>8} {'Vol':>7} {'Sharpe':>8} "
          f"{'Sh.liq':>8} {'DD max':>8} {'vs CDI':>9}")
    print("  " + "-" * 70)

    resultados = {}
    for k in JANELAS_MOMENTUM:
        r = simular(retornos, disp, k, vol_alvo_anual=args.vol_alvo,
                    custo_pct=args.custo, permitir_short=not args.long_only)
        if not r:
            continue
        m = metricas(r["retorno"], cdi_df)
        if not m:
            continue
        resultados[k] = (r, m)
        vs_cdi = m["cagr"] - m.get("cdi_cagr", 0)
        print(f"  {k:>6}m {m['cagr']:>7.1f}% {m['vol']:>6.1f}% "
              f"{m['sharpe']:>8.2f} {m.get('sharpe_liq', 0):>8.2f} "
              f"{m['dd']:>7.1f}% {vs_cdi:>+8.1f}pp")

    if not resultados:
        print("\n  Nenhum resultado — dados insuficientes.")
        return

    # Combinação das três janelas: diversificação de parâmetro, que é a
    # forma honesta de "escolher" — usa todas em vez de eleger a melhor.
    combinado = sum(r["retorno"] for r, _ in resultados.values()) / len(resultados)
    m_comb = metricas(combinado, cdi_df)
    print("  " + "-" * 70)
    vs = m_comb["cagr"] - m_comb.get("cdi_cagr", 0)
    print(f"  {'COMBIN.':>7} {m_comb['cagr']:>7.1f}% {m_comb['vol']:>6.1f}% "
          f"{m_comb['sharpe']:>8.2f} {m_comb.get('sharpe_liq', 0):>8.2f} "
          f"{m_comb['dd']:>7.1f}% {vs:>+8.1f}pp")

    cdi_cagr = m_comb.get("cdi_cagr", 0)
    print(f"\n  CDI no periodo: {cdi_cagr:.1f}% a.a. ({m_comb['anos']:.1f} anos)")

    # ── Estrutura de capital: a diferenca que muda tudo ─────────────────
    print("\n" + "=" * 74)
    print("  ESTRUTURA DE CAPITAL — o mesmo sinal, duas formas de operar")
    print("=" * 74)
    print(f"  (a) Capital SAI do CDI para operar  : {m_comb['cagr']:>6.1f}% a.a.")
    # Com futuros, o capital fica em CDI e a posicao usa margem: soma-se
    # o CDI integral ao resultado da estrategia.
    ret_com_cdi = combinado.copy()
    if cdi_df is not None:
        col = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
        cdi_d = cdi_df[col].pct_change().reindex(combinado.index).fillna(0) * 100
        ret_com_cdi = combinado + cdi_d
    m_fut = metricas(ret_com_cdi, cdi_df)
    print(f"  (b) Capital em CDI + futuros/margem : {m_fut['cagr']:>6.1f}% a.a.  "
          f"(Sharpe {m_fut['sharpe']:.2f}, DD {m_fut['dd']:.1f}%)")
    print(f"      -> ganho da mudanca estrutural  : {m_fut['cagr']-m_comb['cagr']:>+6.1f}pp ao ano")

    # ── Veredito ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    sh_liq = m_comb.get("sharpe_liq", 0)
    print(f"  VEREDITO DA TESE")
    print(f"  Sharpe liquido do CDI (combinado): {sh_liq:.2f}")
    if sh_liq > 0.5:
        print("  >> TESE SE SUSTENTA. Vale construir em cima (Sharpe liq > 0.5).")
    elif sh_liq > 0.2:
        print("  >> TESE FRACA. Ha sinal, mas pequeno. Investigar antes de construir.")
    else:
        print("  >> TESE NAO SE SUSTENTA nos dados. Nao vale construir.")
    print("=" * 74)

    # Por classe de ativo — mostra de onde vem (ou nao vem) o resultado
    print("\n  DECOMPOSICAO POR CLASSE (janela combinada):")
    for classe, lista in UNIVERSO.items():
        ativos_c = [a for a in lista if a in retornos.columns]
        if not ativos_c:
            continue
        series = []
        for k in JANELAS_MOMENTUM:
            rc = simular(retornos, ativos_c, k, vol_alvo_anual=args.vol_alvo,
                         custo_pct=args.custo, permitir_short=not args.long_only)
            if rc:
                series.append(rc["retorno"])
        if not series:
            continue
        mc = metricas(sum(series) / len(series), cdi_df)
        if mc:
            print(f"    {classe:<12} ({len(ativos_c):2d} ativos): "
                  f"CAGR {mc['cagr']:>6.1f}% | Sharpe {mc['sharpe']:>5.2f} | "
                  f"DD {mc['dd']:>6.1f}%")

    saida = os.path.join(BASE, "reports", "teste_trend_following.csv")
    pd.DataFrame({
        "retorno_combinado_pct": combinado,
        "capital": (1 + combinado / 100).cumprod(),
    }).to_csv(saida)
    print(f"\n  Serie salva em {saida}")


if __name__ == "__main__":
    main()
