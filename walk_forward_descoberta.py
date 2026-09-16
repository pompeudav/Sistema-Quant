"""
walk_forward_descoberta.py
--------------------------
Backtest SEM viés de seleção — responde à crítica de que o sistema
"aloca olhando pros dados".

O problema que isto resolve
===========================
O pipeline normal roda nesta ordem:

    correlator (histórico INTEIRO) -> backtester (histórico INTEIRO)
    -> walk_forward divide treino/teste

Quando o walk-forward vai "testar" em 2023-2026, os candidatos que
chegam até ele já foram descobertos e tiveram seus parâmetros
(holding, threshold) escolhidos com dados de 2023-2026. O período de
teste foi usado para decidir quem sobrevive, então deixou de ser teste.

Evidência disso nos dados do projeto (setembro/2026):
    Sharpe médio no treino 2016-2022 ......... +0.51
    Sharpe médio no "teste" 2023-2026 ........ +1.16   <- usado p/ selecionar
    Sharpe médio em maio-set/2026 (novo) ..... +0.43
    Win rate no período de seleção ........... 57.8%
    Win rate no período novo ................. 50.2%   <- moeda justa

Como este script funciona
=========================
Simula o processo REAL de operar o sistema ao longo do tempo:

    para cada ponto de corte T (a cada `passo_meses`):
        1. descobre padrões usando SÓ dados até T
        2. valida em walk-forward interno, também só até T
        3. opera os `passo_meses` seguintes com o que foi descoberto
        4. avança T e repete, esquecendo o que veio depois

A curva final é a costura dos períodos operados. Nenhum trade usa
informação que não estaria disponível na data em que foi aberto —
inclusive a decisão de QUAIS estratégias existir.

Custo: cada ciclo roda um correlator + backtester reduzidos. Com passo
de 6 meses sobre 2016-2026 são ~14 ciclos. Rodar overnight.

Uso:
    python walk_forward_descoberta.py --teste            # 2 ciclos, rápido
    python walk_forward_descoberta.py --passo-meses 6    # completo
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "analysis"))

from correlator import Correlator, _carregar_retornos_completos  # noqa: E402
from backtest.backtester import Backtester                        # noqa: E402


def descobrir_ate(retornos_ate_T: pd.DataFrame, cfg: dict) -> list:
    """
    Roda a descoberta usando SOMENTE dados até T e devolve as estratégias
    aprovadas. Espelha o pipeline normal (correlator -> sweep -> validação),
    mas com todo o horizonte cortado em T.
    """
    c = Correlator(
        retornos=retornos_ate_T,
        min_frequencia=cfg["min_frequencia"],
        min_ocorrencias=cfg["min_ocorrencias"],
        p_valor_max=0.05,
        threshold_movimento=0.3,
    )

    padroes = c.detectar_tipo1() + c.detectar_tipo2() + c.detectar_tipo4()
    padroes = c._corrigir_bh(padroes)
    padroes = [p for p in padroes if p.p_valor_bh < 1.0]
    padroes.sort(key=lambda x: x.score, reverse=True)
    padroes = padroes[: cfg["max_padroes"]]

    if not padroes:
        return []

    # Validação interna: divide o próprio período disponível em
    # treino/teste. Continua havendo seleção aqui — mas toda ela usa
    # apenas dados anteriores a T, que é o ponto.
    corte = retornos_ate_T.index[int(len(retornos_ate_T) * 0.7)]
    ret_tr = retornos_ate_T[retornos_ate_T.index <= corte]
    ret_te = retornos_ate_T[retornos_ate_T.index > corte]
    if len(ret_te) < 60:
        return []

    bt_tr = Backtester(retornos=ret_tr, custo_por_trade=0.15)
    bt_te = Backtester(retornos=ret_te, custo_por_trade=0.15)

    aprovadas = []
    for p in padroes:
        mom = getattr(p, "momentum_meses", 0) or 0
        holdings = [21] if mom > 0 else [1, 3]
        for h in holdings:
            r_tr = bt_tr.rodar(p.ativo_a, p.ativo_b, p.direcao_a, p.direcao_b,
                               holding_days=h, momentum_meses=mom)
            if not r_tr or r_tr.n_trades < 5 or r_tr.sharpe <= 0:
                continue
            r_te = bt_te.rodar(p.ativo_a, p.ativo_b, p.direcao_a, p.direcao_b,
                               holding_days=h, momentum_meses=mom)
            if not r_te or r_te.n_trades < cfg["min_trades_val"]:
                continue
            if r_te.sharpe < cfg["min_sharpe_val"] or r_te.win_rate < 0.5:
                continue
            aprovadas.append({
                "ativo_a": p.ativo_a, "ativo_b": p.ativo_b,
                "direcao_a": p.direcao_a, "direcao_b": p.direcao_b,
                "holding": h, "momentum_meses": mom,
                "sharpe_val": r_te.sharpe,
            })
            break  # um holding por padrão
    return aprovadas


def operar_periodo(estrategias: list, retornos_periodo: pd.DataFrame,
                   cdi_df: pd.DataFrame = None, custo: float = 0.15,
                   exp_efetiva_alvo: float = 35.0,
                   size_max_pct: float = 2.0) -> tuple:
    """
    Opera as estratégias descobertas no período seguinte COM as camadas de
    proteção do sistema real, e devolve (retorno_diario, capital_ocupado).

    O que é simulado aqui, além do sinal bruto:
      - position sizing por estratégia (proporcional ao Sharpe da validação,
        limitado por size_max_pct e reescalado para respeitar o teto de
        exposição efetiva)
      - alocação em base DIÁRIA: cada trade ocupa `size` do capital entre a
        entrada e a saída, e o retorno do dia é a soma ponderada do que
        estava aberto naquele dia
      - CDI overlay: o capital NÃO ocupado rende CDI, que no sistema real é
        o maior contribuidor isolado de retorno

    Tudo aqui usa apenas informação do próprio período operado ou anterior
    a ele — o `sharpe_val` que define o peso veio da validação feita antes
    de T, nunca do período que está sendo operado.
    """
    vazio = (pd.Series(dtype=float), pd.Series(dtype=float))
    if not estrategias or retornos_periodo.empty:
        return vazio

    bt = Backtester(retornos=retornos_periodo, custo_por_trade=custo)
    dias = retornos_periodo.index

    # ── Sizing: peso proporcional à qualidade medida na validação ────────
    # Usa o Sharpe out-of-sample do ciclo (já conhecido em T) como proxy de
    # convicção, do mesmo modo que o Kelly usa win rate/payoff históricos.
    sharpes = np.array([max(e.get("sharpe_val", 0.0), 0.0) for e in estrategias])
    if sharpes.sum() <= 0:
        sharpes = np.ones(len(estrategias))
    pesos = sharpes / sharpes.sum()
    sizes = np.minimum(pesos * exp_efetiva_alvo, size_max_pct) / 100.0

    # ── Aloca cada trade nos dias em que a posição fica aberta ───────────
    ret_diario = pd.Series(0.0, index=dias)
    ocupado    = pd.Series(0.0, index=dias)

    for e, size in zip(estrategias, sizes):
        if size <= 0:
            continue
        r = bt.rodar(e["ativo_a"], e["ativo_b"], e["direcao_a"], e["direcao_b"],
                     holding_days=e["holding"], momentum_meses=e["momentum_meses"])
        if r is None or r.curva_capital is None or len(r.curva_capital) < 2:
            continue

        h = e["holding"]
        # curva_capital é indexada pela data de SAÍDA de cada trade.
        rets_trade = r.curva_capital.pct_change().dropna() * 100
        for data_saida, ret_trade in rets_trade.items():
            if data_saida not in dias:
                continue
            i_saida = dias.get_loc(data_saida)
            i_entra = max(0, i_saida - h + 1)
            janela = dias[i_entra: i_saida + 1]
            if len(janela) == 0:
                continue
            # Distribui o retorno do trade pelos dias em que ficou aberto,
            # em vez de creditar tudo no dia da saída: isso evita picos
            # artificiais de volatilidade diária.
            ret_diario.loc[janela] += (ret_trade * size) / len(janela)
            ocupado.loc[janela]    += size

    ocupado = ocupado.clip(upper=1.0)

    # ── CDI sobre o capital ocioso ───────────────────────────────────────
    if cdi_df is not None and not cdi_df.empty:
        col = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
        cdi_diario = cdi_df[col].pct_change().reindex(dias).fillna(0.0)
        ocioso = (1.0 - ocupado).clip(lower=0.0)
        ret_diario = ret_diario + ocioso * cdi_diario * 100

    return ret_diario.sort_index(), ocupado.sort_index()


def main():
    from modo_teste import adicionar_flag, banner

    ap = argparse.ArgumentParser(
        description="Walk-forward de DESCOBERTA (sem viés de seleção)"
    )
    adicionar_flag(ap)
    ap.add_argument("--passo-meses", type=int, default=6,
                    help="De quantos em quantos meses redescobrir (default 6)")
    ap.add_argument("--treino-min-meses", type=int, default=36,
                    help="Histórico mínimo antes do primeiro ciclo (default 36)")
    args = ap.parse_args()
    banner("walk_forward_descoberta.py", args.teste)

    pasta_raw = os.path.join(BASE, "data", "raw")
    retornos = _carregar_retornos_completos(pasta_raw)

    caminho_cdi = os.path.join(pasta_raw, "cdi.csv")
    cdi_df = (pd.read_csv(caminho_cdi, index_col=0, parse_dates=True)
              if os.path.exists(caminho_cdi) else None)
    if cdi_df is None:
        print("AVISO: cdi.csv nao encontrado - overlay de CDI desativado")

    if args.teste:
        from modo_teste import ATIVOS_TESTE
        cols = [c for c in ATIVOS_TESTE if c in retornos.columns]
        retornos = retornos[cols]
        cfg = {"min_frequencia": 0.55, "min_ocorrencias": 30, "max_padroes": 40,
               "min_trades_val": 3, "min_sharpe_val": 0.0}
        passo, treino_min = 12, 48
    else:
        cfg = {"min_frequencia": 0.55, "min_ocorrencias": 30, "max_padroes": 400,
               "min_trades_val": 10, "min_sharpe_val": 0.3}
        passo, treino_min = args.passo_meses, args.treino_min_meses

    inicio = retornos.index[0]
    fim    = retornos.index[-1]
    T = inicio + pd.DateOffset(months=treino_min)

    print("=" * 70)
    print("WALK-FORWARD DE DESCOBERTA — sem viés de seleção")
    print("=" * 70)
    print(f"  Dados      : {inicio.date()} -> {fim.date()} ({retornos.shape[1]} ativos)")
    print(f"  Treino min : {treino_min} meses | Redescoberta a cada {passo} meses")

    retornos_ciclo = []
    linhas = []
    n = 0
    t0 = time.time()

    while T < fim:
        T_prox = min(T + pd.DateOffset(months=passo), fim)
        n += 1

        ret_ate_T = retornos[retornos.index <= T]
        ret_periodo = retornos[(retornos.index > T) & (retornos.index <= T_prox)]
        if len(ret_periodo) < 5:
            break

        print(f"\n[Ciclo {n}] descobre com dados ate {T.date()} -> "
              f"opera {T.date()} a {T_prox.date()}")
        ests = descobrir_ate(ret_ate_T, cfg)
        print(f"          {len(ests)} estrategias aprovadas")

        serie, ocupado = operar_periodo(ests, ret_periodo, cdi_df=cdi_df)
        if not serie.empty:
            retornos_ciclo.append(serie)
            ret_tot = ((1 + serie / 100).prod() - 1) * 100
            ocup_med = float(ocupado.mean()) if not ocupado.empty else 0.0
            print(f"          retorno do periodo: {ret_tot:+.2f}%  "
                  f"| capital ocupado medio: {ocup_med:.1%}")
        else:
            ret_tot, ocup_med = 0.0, 0.0
            print(f"          sem trades no periodo")

        linhas.append({
            "ciclo": n,
            "descoberta_ate": T.date(),
            "opera_ate": T_prox.date(),
            "n_estrategias": len(ests),
            "retorno_periodo_%": round(ret_tot, 3),
            "capital_ocupado_medio": round(ocup_med, 4),
        })
        T = T_prox

    if not retornos_ciclo:
        print("\nNenhum ciclo produziu trades.")
        return

    serie_total = pd.concat(retornos_ciclo).sort_index()
    serie_total = serie_total.groupby(serie_total.index).mean()

    # ── Circuit breaker: reduz exposicao apos drawdown severo ────────────
    # Mesma regra do portfolio_vs_cdi (threshold -15%, fator 0.65).
    # Reativo por construcao: so olha o drawdown ja realizado, entao nao
    # introduz lookahead.
    cap_cb, pico, em_corte, dias_corte = 1.0, 1.0, False, 0
    serie_cb = serie_total.copy()
    for i, (dt, r) in enumerate(serie_total.items()):
        if em_corte:
            r_eff = r * 0.65
            dias_corte += 1
        else:
            r_eff = r
            dias_corte = 0
        serie_cb.iloc[i] = r_eff
        cap_cb *= (1 + r_eff / 100)
        pico = max(pico, cap_cb)
        dd_atual = (cap_cb - pico) / pico
        if not em_corte and dd_atual < -0.15:
            em_corte, dias_corte = True, 0
        if em_corte and dias_corte >= 5 and dd_atual > -0.075:
            em_corte = False

    def _metricas(serie):
        cap = (1 + serie / 100).cumprod()
        anos_ = (serie.index[-1] - serie.index[0]).days / 365.25
        obs_ano_ = len(serie) / anos_
        return {
            "cap": cap,
            "ret": (cap.iloc[-1] - 1) * 100,
            "cagr": (cap.iloc[-1] ** (1 / anos_) - 1) * 100,
            "vol": serie.std() * np.sqrt(obs_ano_),
            "sharpe": (serie.mean() / serie.std() * np.sqrt(obs_ano_)
                       if serie.std() > 0 else 0),
            "dd": ((cap - cap.cummax()) / cap.cummax()).min() * 100,
            "anos": anos_,
        }

    m_sem = _metricas(serie_total)
    m_com = _metricas(serie_cb)
    capital = m_com["cap"]

    # CDI no mesmo periodo, para a comparacao que interessa
    cagr_cdi = None
    if cdi_df is not None and not cdi_df.empty:
        col = "CDI" if "CDI" in cdi_df.columns else cdi_df.columns[0]
        c = cdi_df[col].loc[serie_total.index[0]:serie_total.index[-1]].dropna()
        if len(c) > 1:
            ret_cdi = c.iloc[-1] / c.iloc[0] - 1
            cagr_cdi = ((1 + ret_cdi) ** (1 / m_com["anos"]) - 1) * 100

    print("\n" + "=" * 70)
    print("RESULTADO SEM VIÉS DE SELEÇÃO — com sizing + CDI overlay")
    print("=" * 70)
    print(f"  Ciclos de redescoberta : {n}")
    print(f"  Período operado        : {serie_total.index[0].date()} -> "
          f"{serie_total.index[-1].date()}  ({m_com['anos']:.1f} anos)")
    print(f"\n  {'':24} {'sem CB':>12} {'com CB':>12}")
    print(f"  {'Retorno total':24} {m_sem['ret']:>11.1f}% {m_com['ret']:>11.1f}%")
    print(f"  {'CAGR':24} {m_sem['cagr']:>11.1f}% {m_com['cagr']:>11.1f}%")
    print(f"  {'Volatilidade':24} {m_sem['vol']:>11.1f}% {m_com['vol']:>11.1f}%")
    print(f"  {'Sharpe':24} {m_sem['sharpe']:>12.2f} {m_com['sharpe']:>12.2f}")
    print(f"  {'Drawdown maximo':24} {m_sem['dd']:>11.1f}% {m_com['dd']:>11.1f}%")

    if cagr_cdi is not None:
        print(f"\n  CDI no periodo         : {cagr_cdi:+.1f}% a.a.")
        print(f"  DIFERENCA (com CB)     : {m_com['cagr'] - cagr_cdi:+.1f}pp ao ano")
        veredito = "BATEU o CDI" if m_com["cagr"] > cagr_cdi else "PERDEU do CDI"
        print(f"  VEREDITO               : {veredito}")

    print(f"\n  Tempo de execucao      : {(time.time()-t0)/60:.1f} min")

    pasta = os.path.join(BASE, "reports", "teste" if args.teste else "")
    os.makedirs(pasta, exist_ok=True)
    pd.DataFrame(linhas).to_csv(
        os.path.join(pasta, "walk_forward_descoberta_ciclos.csv"), index=False)
    capital.rename("capital").to_frame().to_csv(
        os.path.join(pasta, "walk_forward_descoberta_curva.csv"))
    print(f"\n  Salvo em {pasta}")
    print("\n  COMPARE com o backtest normal: a diferença é o tamanho do")
    print("  viés de seleção que existia no número anterior.")


if __name__ == "__main__":
    main()
