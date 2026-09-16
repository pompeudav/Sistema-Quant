"""
monitor_degradacao.py
---------------------
Desliga estratégias que pararam de funcionar, usando os resultados REAIS
do paper trading (sinais_log.csv) em vez de re-rodar backtests.

Por que não usar o RegimeMonitor do walk_forward.py:
  Ele re-roda um backtest por estratégia por janela (30/60/90d). Com 1711
  estratégias isso são 5133 backtests por dia — inviável para rodar no
  fluxo diário. E, mais importante, ele mede o passado simulado de novo,
  enquanto aqui já temos o resultado que as estratégias de fato tiveram
  no paper trading, que é a evidência limpa (ver discussão sobre viés de
  seleção: o backtest pode estar contaminado, o log de paper trading não).

Critérios de suspensão (todos avaliados sobre trades FECHADOS):
  - Win rate abaixo de `min_winrate` na janela recente
  - Retorno acumulado negativo abaixo de `max_perda_acum`
  - Exige `min_trades` fechados na janela para julgar (senão não há
    evidência suficiente e a estratégia é mantida)

Estratégias suspensas vão para reports/estrategias_suspensas.csv, que o
gerar_monitor.py lê para não reativá-las ao regenerar o monitor.

Uso:
    python analysis/monitor_degradacao.py
    python analysis/monitor_degradacao.py --janela 60 --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto, relativo a este arquivo
REPORTS = os.path.join(BASE, "reports")

# Critérios padrão. Deliberadamente frouxos: o objetivo é cortar o que
# está claramente quebrado, não perseguir ruído de curto prazo. Uma
# estratégia com win rate de 45% em 20 trades pode ser só variância.
CFG_DEGRADACAO = {
    "janela_dias":     60,     # olha os últimos N dias de paper trading
    "min_trades":      10,     # mínimo de trades fechados para julgar
    "min_winrate":     0.40,   # abaixo disso, suspende
    "max_perda_acum": -5.0,    # perda acumulada (%) que dispara suspensão
}


def carregar_log() -> pd.DataFrame:
    caminho = os.path.join(REPORTS, "sinais_log.csv")
    if not os.path.exists(caminho):
        print(f"⚠️  {caminho} não encontrado — paper trading ainda não gerou dados.")
        return pd.DataFrame()
    return pd.read_csv(caminho)


def avaliar(log_df: pd.DataFrame, cfg: dict = None, hoje: pd.Timestamp = None) -> pd.DataFrame:
    """
    Avalia cada estratégia na janela recente e decide manter ou suspender.
    Retorna DataFrame com uma linha por estratégia avaliada.
    """
    cfg = cfg or CFG_DEGRADACAO
    if log_df.empty:
        return pd.DataFrame()

    hoje = hoje or pd.Timestamp(datetime.today().date())
    corte = hoje - timedelta(days=cfg["janela_dias"])

    df = log_df[log_df["status"] == "fechado"].copy()
    if df.empty:
        return pd.DataFrame()

    df["data_entrada"] = pd.to_datetime(df["data_entrada"], errors="coerce")
    df["ret"] = pd.to_numeric(df["retorno_liq_pct"], errors="coerce")
    df = df.dropna(subset=["data_entrada", "ret"])
    recente = df[df["data_entrada"] >= corte]

    if recente.empty:
        print(f"Nenhum trade fechado nos últimos {cfg['janela_dias']} dias.")
        return pd.DataFrame()

    linhas = []
    for est_id, g in recente.groupby("estrategia_id"):
        n = len(g)
        wr = float((g["ret"] > 0).mean())
        acum = float(g["ret"].sum())

        suspender, motivo = False, ""
        if n < cfg["min_trades"]:
            motivo = f"poucos trades ({n}) — sem evidência para julgar"
        elif wr < cfg["min_winrate"]:
            suspender = True
            motivo = f"winrate {wr:.0%} < {cfg['min_winrate']:.0%} em {n} trades"
        elif acum < cfg["max_perda_acum"]:
            suspender = True
            motivo = f"perda acumulada {acum:.1f}% < {cfg['max_perda_acum']}%"

        linhas.append({
            "id": est_id,
            "trades_janela": n,
            "winrate_janela": round(wr, 3),
            "retorno_acum_%": round(acum, 3),
            "suspender": suspender,
            "motivo": motivo,
        })

    return pd.DataFrame(linhas).sort_values("retorno_acum_%")


def salvar_suspensas(aval: pd.DataFrame, dry_run: bool = False) -> int:
    """
    Grava/atualiza reports/estrategias_suspensas.csv. Suspensões são
    cumulativas: uma estratégia suspensa continua suspensa até ser
    reavaliada explicitamente (o gerar_monitor respeita este arquivo).
    """
    caminho = os.path.join(REPORTS, "estrategias_suspensas.csv")
    novas = aval[aval["suspender"]].copy()
    if novas.empty:
        print("Nenhuma estratégia nova para suspender.")
        return 0

    novas["data_suspensao"] = datetime.today().strftime("%Y-%m-%d")
    cols = ["id", "data_suspensao", "motivo", "trades_janela",
            "winrate_janela", "retorno_acum_%"]
    novas = novas[cols]

    if os.path.exists(caminho):
        antigas = pd.read_csv(caminho)
        combinado = pd.concat([antigas, novas], ignore_index=True)
        combinado = combinado.drop_duplicates(subset=["id"], keep="first")
    else:
        combinado = novas

    if dry_run:
        print(f"[dry-run] {len(novas)} suspensões NÃO foram salvas.")
        return len(novas)

    combinado.to_csv(caminho, index=False)
    print(f"💾 {caminho}: {len(combinado)} estratégias suspensas no total "
          f"({len(novas)} novas hoje)")
    return len(novas)


def main():
    sys.path.append(BASE)
    from modo_teste import adicionar_flag, banner

    ap = argparse.ArgumentParser(description="Monitor de degradação (Fase 5)")
    adicionar_flag(ap)
    ap.add_argument("--janela", type=int, default=CFG_DEGRADACAO["janela_dias"],
                    help="Janela em dias de paper trading a avaliar")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mostra o que seria suspenso sem salvar")
    args = ap.parse_args()
    banner("monitor_degradacao.py", args.teste)

    global REPORTS
    if args.teste:
        REPORTS = os.path.join(BASE, "reports", "teste")

    cfg = dict(CFG_DEGRADACAO)
    cfg["janela_dias"] = args.janela

    print("=" * 62)
    print(f"MONITOR DE DEGRADAÇÃO — janela de {cfg['janela_dias']} dias")
    print("=" * 62)

    log_df = carregar_log()
    if log_df.empty:
        return

    aval = avaliar(log_df, cfg)
    if aval.empty:
        return

    n_susp = int(aval["suspender"].sum())
    n_ok   = len(aval) - n_susp
    print(f"\n  Estratégias avaliadas : {len(aval)}")
    print(f"  Mantidas              : {n_ok}")
    print(f"  A suspender           : {n_susp}")

    if n_susp:
        print(f"\n  PIORES (top 15 por perda acumulada):")
        piores = aval[aval["suspender"]].head(15)
        print(piores[["id", "trades_janela", "winrate_janela",
                      "retorno_acum_%", "motivo"]].to_string(index=False))

    salvar_suspensas(aval, dry_run=args.dry_run)

    caminho_aval = os.path.join(REPORTS, "degradacao_avaliacao.csv")
    if not args.dry_run:
        aval.to_csv(caminho_aval, index=False)
        print(f"💾 {caminho_aval} salvo")


if __name__ == "__main__":
    main()
