"""
gerar_monitor.py
----------------
Gera monitor_status.csv a partir dos resultados já existentes do walk-forward.
Muito mais rápido que re-rodar walk_forward.py.

Fontes (em ordem de prioridade):
  1. walk_forward.csv           — aprovados na janela única
  2. walk_forward_deslizante.csv — filtra para aprovados em ambos (se existir)

Enriquecimento opcional (--enriquecer):
  Roda backtests nas últimas 3 janelas (30d, 60d, 90d) para as estratégias
  aprovadas e preenche sharpe_60d, winrate_60d, etc.
  Pode demorar vários minutos dependendo da quantidade.

Uso:
  python analysis/gerar_monitor.py                # rápido (sem métricas recentes)
  python analysis/gerar_monitor.py --enriquecer   # com métricas recentes
  python analysis/gerar_monitor.py --apenas-ambos # só aprovados nos dois WFs
"""

import sys
import os
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto, relativo a este arquivo
REPORTS = os.path.join(BASE, "reports")
DATA    = os.path.join(BASE, "data", "raw")

sys.path.insert(0, os.path.join(BASE, "analysis"))
sys.path.insert(0, BASE)


# ── Gera ID no mesmo formato do monitor ──────────────────────────────────────
def _gerar_id(row) -> str:
    tipo = "CTR" if row["contrarian"] else "MOM"
    thr  = row["threshold_%"]
    hld  = int(row["holding_days"])
    thr_fmt = f"{thr:.1f}" if thr == int(thr) else str(thr)
    est_id = (f"{row['ativo_a']}_{row['ativo_b']}_{row['direcao_a']}_"
              f"{tipo}_thr{thr_fmt}_h{hld}")
    # Tipo 4 (momentum): sufixo desambigua janelas diferentes do mesmo
    # ativo, que senão colidiriam (thr=0 e ativo_a == ativo_b).
    # Precisa bater exatamente com walk_forward._gerar_id.
    mom = int(row.get("momentum_meses", 0) or 0) if hasattr(row, "get") else 0
    if mom > 0:
        est_id += f"_m{mom}"
    return est_id


# ── Carrega candidatos aprovados ──────────────────────────────────────────────
def carregar_aprovados(apenas_ambos: bool) -> pd.DataFrame:
    caminho_wf = os.path.join(REPORTS, "walk_forward.csv")
    if not os.path.exists(caminho_wf):
        raise FileNotFoundError(f"walk_forward.csv não encontrado em {REPORTS}")

    wf = pd.read_csv(caminho_wf)
    aprovados = wf[wf["aprovado"]].copy()
    print(f"  walk_forward.csv       : {len(wf)} total | {len(aprovados)} aprovados")

    if apenas_ambos:
        caminho_d = os.path.join(REPORTS, "walk_forward_deslizante.csv")
        if os.path.exists(caminho_d):
            deslizante = pd.read_csv(caminho_d)
            ap_d = deslizante[deslizante["aprovado"]]
            # Filtra aprovados que também passaram no deslizante
            chaves_d = set(
                zip(ap_d["ativo_a"], ap_d["ativo_b"], ap_d["direcao_a"],
                    ap_d["contrarian"], ap_d["threshold_%"], ap_d["holding_days"])
            )
            mask = aprovados.apply(
                lambda r: (r["ativo_a"], r["ativo_b"], r["direcao_a"],
                           r["contrarian"], r["threshold_%"], r["holding_days"])
                          in chaves_d,
                axis=1,
            )
            aprovados = aprovados[mask]
            print(f"  walk_forward_deslizante: {len(deslizante)} total | "
                  f"{len(ap_d)} aprovados | {len(aprovados)} em ambos")
        else:
            print("  walk_forward_deslizante.csv não encontrado — usando apenas janela única")

    return aprovados


# ── Enriquece com métricas recentes ──────────────────────────────────────────
def enriquecer_metricas(df_mon: pd.DataFrame, retornos: pd.DataFrame,
                        cdi: pd.DataFrame) -> pd.DataFrame:
    from backtest.backtester import Backtester

    janelas = [30, 60, 90]
    hoje = datetime.today()

    print(f"\n  Enriquecendo {len(df_mon)} estratégias com métricas recentes...")
    print(f"  (janelas: {janelas} dias)\n")

    for idx, row in df_mon.iterrows():
        est_id = row["id"]
        # Extrai parâmetros do ID
        # Formato: A_B_direcao_TIPO_thrX_hY
        import re
        m = re.match(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr([\d.]+)_h(\d+)(?:_m\d+)?$', est_id)
        if not m:
            continue

        a, b     = m.group(1), m.group(2)
        direcao  = m.group(3)
        contrarian = m.group(4) == "CTR"
        thr      = float(m.group(5))
        hld      = int(m.group(6))

        if a not in retornos.columns or b not in retornos.columns:
            continue

        for janela in janelas:
            data_corte = (hoje - timedelta(days=janela)).strftime("%Y-%m-%d")
            ret_janela = retornos[retornos.index >= data_corte]

            if len(ret_janela) < 10:
                continue

            # Custo realista: 0.30% para ativos BR (.SA), 0.15% para globais
            _eh_br = b.endswith(".SA") or "_" not in b
            _custo = 0.30 if _eh_br else 0.15
            bt = Backtester(
                retornos=ret_janela, cdi=cdi,
                custo_por_trade=_custo,
                threshold=thr, holding_days=hld,
            )

            if contrarian:
                r = bt.rodar_contrarian(a, b, direcao, holding_days=hld)
            else:
                db = "cai" if direcao == "sobe" else "sobe"
                r = bt.rodar(a, b, direcao, db, holding_days=hld)

            if r and r.n_trades >= 3:
                df_mon.at[idx, f"sharpe_{janela}d"]  = round(r.sharpe, 2)
                df_mon.at[idx, f"winrate_{janela}d"] = round(r.win_rate, 3)
                df_mon.at[idx, f"trades_{janela}d"]  = r.n_trades

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(df_mon)} processadas...", end="\r")

    print(f"  {len(df_mon)}/{len(df_mon)} processadas.   ")
    return df_mon


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global REPORTS
    sys.path.insert(0, BASE)
    from modo_teste import adicionar_flag, banner

    parser = argparse.ArgumentParser(description="Gera monitor_status.csv")
    adicionar_flag(parser)
    parser.add_argument("--enriquecer",  action="store_true",
                        help="Calcula métricas recentes (30d/60d/90d) — mais lento")
    parser.add_argument("--apenas-ambos", action="store_true",
                        help="Inclui apenas estratégias aprovadas nos dois walk-forwards")
    args = parser.parse_args()

    if args.teste:
        REPORTS = os.path.join(BASE, "reports", "teste")
    banner("gerar_monitor.py", args.teste)

    print("=" * 60)
    print("GERADOR DE MONITOR_STATUS.CSV")
    print("=" * 60)

    # Carrega aprovados
    print("\nCarregando aprovados do walk-forward...")
    aprovados = carregar_aprovados(args.apenas_ambos)

    if aprovados.empty:
        print("ERRO: nenhuma estratégia aprovada encontrada.")
        return

    # Gera IDs
    aprovados["id"] = aprovados.apply(_gerar_id, axis=1)

    # winrate_hist: win rate do período de TESTE do walk-forward (out-of-sample).
    # Usado pelo position_sizing como fallback quando não há dados recentes (60d/90d).
    winrate_hist_col = "teste_winrate_%"
    winrate_hist = (aprovados[winrate_hist_col] / 100.0).values \
                   if winrate_hist_col in aprovados.columns else np.nan

    # Monta DataFrame base
    df_mon = pd.DataFrame({
        "id":           aprovados["id"].values,
        "status":       "ativa",
        "sharpe_hist":  aprovados["teste_sharpe"].values,
        "winrate_hist": winrate_hist,
        "motivo":       "",
        "sharpe_60d":   np.nan,
        "winrate_60d":  np.nan,
        "trades_60d":   np.nan,
        "sharpe_90d":   np.nan,
        "winrate_90d":  np.nan,
        "trades_90d":   np.nan,
    })

    print(f"\n  {len(df_mon)} estratégias prontas para o monitor")

    # ── Respeita suspensões do monitor de degradação ────────────────────────
    # Este arquivo é regenerado do zero a partir do walk_forward.csv toda vez.
    # Sem este passo, uma estratégia desligada ontem por ter parado de
    # funcionar voltaria a operar hoje, e o monitor de degradação seria
    # inócuo. Ver analysis/monitor_degradacao.py.
    caminho_susp = os.path.join(REPORTS, "estrategias_suspensas.csv")
    if os.path.exists(caminho_susp):
        susp = pd.read_csv(caminho_susp)
        ids_susp = set(susp["id"]) if "id" in susp.columns else set()
        mask = df_mon["id"].isin(ids_susp)
        if mask.any():
            motivos = dict(zip(susp["id"], susp.get("motivo", pd.Series(dtype=str))))
            df_mon.loc[mask, "status"] = "suspensa"
            df_mon.loc[mask, "motivo"] = df_mon.loc[mask, "id"].map(motivos).fillna("degradacao")
            print(f"  {int(mask.sum())} suspensas pelo monitor de degradação "
                  f"(de {len(ids_susp)} no arquivo)")
    else:
        print("  (sem estrategias_suspensas.csv — nenhuma suspensão a aplicar)")

    # Enriquecimento opcional com métricas recentes
    if args.enriquecer:
        print("\nCarregando dados para enriquecimento...")
        from analysis.correlator import _carregar_retornos_completos
        retornos = _carregar_retornos_completos(DATA)

        cdi_path = os.path.join(DATA, "cdi.csv")
        cdi = pd.read_csv(cdi_path, index_col=0, parse_dates=True) \
              if os.path.exists(cdi_path) else None

        df_mon = enriquecer_metricas(df_mon, retornos, cdi)

    # Salva
    caminho = os.path.join(REPORTS, "monitor_status.csv")
    df_mon.to_csv(caminho, index=False)

    n_ativas = (df_mon["status"] == "ativa").sum()
    print(f"\n{'='*60}")
    print(f"monitor_status.csv salvo!")
    print(f"  Estratégias : {len(df_mon)}")
    print(f"  Ativas      : {n_ativas}")
    print(f"  Caminho     : {caminho}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
