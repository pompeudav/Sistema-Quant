"""
replay_paper_trading.py
------------------------
Roda o paper_trading dia a dia sobre um período passado, construindo o
sinais_log.csv como se o sistema tivesse rodado todos os dias.

Para que serve: as estratégias em produção foram descobertas com dados
até 2026-04-30. Ao atualizar a base (agora vai até setembro), os meses
de maio em diante são dados que o sistema NUNCA viu — out-of-sample
genuíno. Este replay reconstrói o que o paper trading teria registrado
nesse período, em vez de esperar 4 meses para ter a mesma informação.

IMPORTANTE — o que isto NÃO é:
  Não substitui paper trading de verdade. Aqui os preços são de
  fechamento históricos e a execução é perfeita. O paper trading ao vivo
  é o que revela slippage real, horário de execução, falha de API e
  liquidez. Isto valida as ESTRATÉGIAS em dados novos; o paper trading
  ao vivo valida a OPERAÇÃO.

Uso:
    python replay_paper_trading.py --de 2026-05-01 --ate 2026-09-01
"""

import argparse
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "analysis"))
sys.path.insert(0, BASE)

from paper_trading import (  # noqa: E402
    detectar_sinais, fechar_posicoes, REPORTS, DATA,
)


def carregar_dados():
    """
    Carrega retornos, monitor e sizes. O paper_trading.py faz isso dentro
    do main(), então replicamos aqui em vez de refatorar aquele arquivo —
    ele está em uso e mexer nele agora só aumentaria a superfície de erro.
    """
    retornos = pd.read_csv(os.path.join(DATA, "retornos.csv"),
                           index_col=0, parse_dates=True)
    monitor = pd.read_csv(os.path.join(REPORTS, "monitor_status.csv"))
    caminho_sizes = os.path.join(REPORTS, "position_sizes.csv")
    sizes_df = pd.read_csv(caminho_sizes) if os.path.exists(caminho_sizes) else None
    return retornos, monitor, sizes_df


def main():
    ap = argparse.ArgumentParser(description="Replay do paper trading sobre período passado")
    ap.add_argument("--de",  required=True, help="Data inicial YYYY-MM-DD")
    ap.add_argument("--ate", required=True, help="Data final YYYY-MM-DD")
    ap.add_argument("--saida", default="sinais_log.csv",
                    help="Nome do arquivo de log (default: sinais_log.csv)")
    args = ap.parse_args()

    retornos, monitor, sizes_df = carregar_dados()

    d0 = pd.Timestamp(args.de)
    d1 = pd.Timestamp(args.ate)
    dias = [d for d in retornos.index if d0 <= d <= d1]

    print(f"\nReplay de {len(dias)} pregões: {dias[0].date()} → {dias[-1].date()}")
    print(f"Estratégias ativas no monitor: {(monitor['status'] == 'ativa').sum()}")

    log_df = pd.DataFrame()
    total_sinais = 0

    for i, dia in enumerate(dias, 1):
        log_df, _ = fechar_posicoes(log_df, retornos, dia)
        novos = detectar_sinais(retornos, monitor, dia, sizes_df, log_df)
        if novos:
            log_df = pd.concat([log_df, pd.DataFrame(novos)], ignore_index=True)
            total_sinais += len(novos)
        if i % 20 == 0 or i == len(dias):
            abertos = int((log_df["status"] == "aberto").sum()) if not log_df.empty else 0
            print(f"  [{i:3d}/{len(dias)}] {dia.date()} | "
                  f"sinais acumulados: {total_sinais:5d} | abertos: {abertos}")

    if log_df.empty:
        print("\nNenhum sinal gerado no período.")
        return

    caminho = os.path.join(REPORTS, args.saida)
    log_df.to_csv(caminho, index=False)
    print(f"\n{caminho} salvo ({len(log_df)} registros)")

    fechados = log_df[log_df["status"] == "fechado"]
    if fechados.empty:
        print("Nenhuma posição fechada ainda.")
        return

    col_ret = "retorno_liq_pct" if "retorno_liq_pct" in fechados.columns else "retorno_pct"
    r = pd.to_numeric(fechados[col_ret], errors="coerce").dropna()
    if r.empty:
        print("Sem retornos calculados.")
        return

    print("\n" + "=" * 62)
    print("RESULTADO DO PERÍODO OUT-OF-SAMPLE")
    print("=" * 62)
    print(f"  Trades fechados : {len(r)}")
    print(f"  Win rate        : {(r > 0).mean():.1%}")
    print(f"  Retorno médio   : {r.mean():+.4f}% por trade")
    print(f"  Retorno somado  : {r.sum():+.2f}% (soma simples, sem sizing)")
    if r.std() > 0:
        print(f"  Sharpe (aprox)  : {r.mean() / r.std() * (252 ** 0.5):.2f}")
    print(f"  Melhor / pior   : {r.max():+.2f}% / {r.min():+.2f}%")


if __name__ == "__main__":
    main()
