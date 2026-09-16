"""
walk_forward.py
---------------
Validação walk-forward e monitor de degradação de estratégias.

Estrutura:
  1. WalkForwardTest   — divide dados em janelas treino/teste e valida padrões
  2. RegimeMonitor     — acompanha performance em tempo real e detecta degradação
  3. StrategyManager   — gerencia o ciclo de vida das estratégias (ativa/suspensa/descartada)

Filosofia:
  Um padrão só é considerado válido para operação real se:
    a) Foi detectado pelo correlator no período de TREINO
    b) Continuou funcionando (Sharpe > 0, win rate > 50%) no período de TESTE
    c) Não está em degradação no monitor em tempo real

  Qualquer padrão que falhe em (b) é descartado, independente de quão
  impressionante foi no treino. Isso elimina o overfitting.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import json

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_BASE)

from analysis.correlator import Correlator, _carregar_retornos_completos
from backtest.backtester import Backtester


# ═══════════════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ResultadoWalkForward:
    """Resultado de validação de um padrão no walk-forward."""
    ativo_a: str
    ativo_b: str
    direcao_a: str
    direcao_b: str
    contrarian: bool
    threshold: float
    holding_days: int

    # Métricas no período de TREINO
    treino_inicio: str
    treino_fim: str
    treino_sharpe: float
    treino_winrate: float
    treino_retorno: float
    treino_trades: int

    # Métricas no período de TESTE (fora da amostra)
    teste_inicio: str
    teste_fim: str
    teste_sharpe: float
    teste_winrate: float
    teste_retorno: float
    teste_trades: int

    # Veredito
    aprovado: bool
    motivo_rejeicao: str = ""

    # Tipo 4 (momentum): janela de momentum em meses. 0 = tipos 1-3.
    momentum_meses: int = 0

    def __str__(self):
        status = "✅ APROVADO" if self.aprovado else f"❌ REJEITADO ({self.motivo_rejeicao})"
        direcao = f"{self.ativo_a} {self.direcao_a} → {self.ativo_b}"
        modo = "CONTRARIAN" if self.contrarian else "MOMENTUM"
        return (
            f"\n{status} | {direcao} | {modo} | thr={self.threshold}% | h={self.holding_days}d\n"
            f"  TREINO  ({self.treino_inicio[:7]} → {self.treino_fim[:7]}): "
            f"sharpe={self.treino_sharpe:+.2f} | wr={self.treino_winrate:.0%} | "
            f"ret={self.treino_retorno:+.1f}% | trades={self.treino_trades}\n"
            f"  TESTE   ({self.teste_inicio[:7]} → {self.teste_fim[:7]}): "
            f"sharpe={self.teste_sharpe:+.2f} | wr={self.teste_winrate:.0%} | "
            f"ret={self.teste_retorno:+.1f}% | trades={self.teste_trades}"
        )

    def resumo(self) -> dict:
        return {
            "ativo_a": self.ativo_a,
            "ativo_b": self.ativo_b,
            "direcao_a": self.direcao_a,
            "direcao_b": self.direcao_b,
            "contrarian": self.contrarian,
            "threshold_%": self.threshold,
            "holding_days": self.holding_days,
            "momentum_meses": self.momentum_meses,
            "treino_sharpe": round(self.treino_sharpe, 2),
            "treino_winrate_%": round(self.treino_winrate * 100, 1),
            "treino_retorno_%": round(self.treino_retorno, 1),
            "treino_trades": self.treino_trades,
            "teste_sharpe": round(self.teste_sharpe, 2),
            "teste_winrate_%": round(self.teste_winrate * 100, 1),
            "teste_retorno_%": round(self.teste_retorno, 1),
            "teste_trades": self.teste_trades,
            "aprovado": self.aprovado,
            "motivo_rejeicao": self.motivo_rejeicao,
            "degradacao_sharpe": round(self.treino_sharpe - self.teste_sharpe, 2),
        }


@dataclass
class StatusEstrategia:
    """Estado atual de uma estratégia no monitor de degradação."""
    id: str                          # chave única da estratégia
    aprovada_em: str                 # data em que passou no walk-forward
    status: str                      # "ativa", "suspensa", "descartada"
    sharpe_historico: float          # sharpe no período de teste
    winrate_historico: float
    ultimos_30d_sharpe: float = 0.0
    ultimos_60d_sharpe: float = 0.0
    ultimos_90d_sharpe: float = 0.0
    dias_em_degradacao: int = 0
    motivo_suspensao: str = ""


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD TEST
# ═══════════════════════════════════════════════════════════════

class WalkForwardTest:
    """
    Valida padrões dividindo os dados em treino e teste.

    Lógica:
      - Treino: detecta padrões e encontra parâmetros ótimos
      - Teste : verifica se os mesmos padrões continuam funcionando
                em dados que o sistema nunca viu

    Critérios de aprovação (configuráveis):
      - Sharpe no teste > 0.3
      - Win rate no teste > 50%
      - Mínimo de 10 trades no teste
      - Degradação de Sharpe (treino-teste) < 1.5
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        cdi: Optional[pd.DataFrame] = None,
        fedfunds: Optional[pd.DataFrame] = None,
        treino_inicio: str = "2016-01-04",
        treino_fim: str   = "2022-12-31",
        teste_inicio: str = "2023-01-01",
        teste_fim: str    = "2026-05-01",
        # Critérios de aprovação
        min_sharpe_teste: float = 0.3,
        min_winrate_teste: float = 0.50,
        min_trades_teste: int = 10,
        max_degradacao_sharpe: float = 1.5,
    ):
        self.retornos = retornos
        self.cdi = cdi
        self.fedfunds = fedfunds
        self.treino_inicio = treino_inicio
        self.treino_fim    = treino_fim
        self.teste_inicio  = teste_inicio
        self.teste_fim     = teste_fim
        self.min_sharpe_teste     = min_sharpe_teste
        self.min_winrate_teste    = min_winrate_teste
        self.min_trades_teste     = min_trades_teste
        self.max_degradacao_sharpe = max_degradacao_sharpe

        # Fatia os dados
        self.ret_treino = retornos[
            (retornos.index >= treino_inicio) &
            (retornos.index <= treino_fim)
        ].copy()
        self.ret_teste = retornos[
            (retornos.index >= teste_inicio) &
            (retornos.index <= teste_fim)
        ].copy()

        print(f"\n📊 Walk-Forward configurado:")
        print(f"   TREINO : {treino_inicio} → {treino_fim} "
              f"({len(self.ret_treino)} dias)")
        print(f"   TESTE  : {teste_inicio} → {teste_fim} "
              f"({len(self.ret_teste)} dias)")
        print(f"   Critérios: sharpe>{min_sharpe_teste} | "
              f"wr>{min_winrate_teste:.0%} | "
              f"trades≥{min_trades_teste} | "
              f"degradação<{max_degradacao_sharpe}")

    def _rodar_backtest(
        self,
        retornos: pd.DataFrame,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        contrarian: bool,
        threshold: float,
        holding_days: int,
        momentum_meses: int = 0,
    ):
        """Roda um backtest em um slice de retornos."""
        bt = Backtester(
            retornos=retornos,
            cdi=self.cdi,
            fedfunds=self.fedfunds,
            custo_por_trade=0.15,
            threshold=threshold,
            holding_days=holding_days,
        )
        if contrarian:
            return bt.rodar_contrarian(ativo_a, ativo_b, direcao_a,
                                       holding_days=holding_days)
        else:
            return bt.rodar(ativo_a, ativo_b, direcao_a, direcao_b,
                           holding_days=holding_days,
                           momentum_meses=momentum_meses)

    def validar_padrao(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        contrarian: bool,
        threshold: float,
        holding_days: int,
        momentum_meses: int = 0,
    ) -> Optional[ResultadoWalkForward]:
        """Valida um padrão específico no walk-forward."""

        # Roda no treino
        r_treino = self._rodar_backtest(
            self.ret_treino, ativo_a, ativo_b,
            direcao_a, direcao_b, contrarian, threshold, holding_days,
            momentum_meses,
        )
        if not r_treino or r_treino.n_trades < 5:
            return None

        # Roda no teste
        r_teste = self._rodar_backtest(
            self.ret_teste, ativo_a, ativo_b,
            direcao_a, direcao_b, contrarian, threshold, holding_days,
            momentum_meses,
        )
        if not r_teste or r_teste.n_trades < self.min_trades_teste:
            motivo = f"poucos trades no teste ({r_teste.n_trades if r_teste else 0} < {self.min_trades_teste})"
            return ResultadoWalkForward(
                ativo_a=ativo_a, ativo_b=ativo_b,
                direcao_a=direcao_a, direcao_b=direcao_b,
                contrarian=contrarian, threshold=threshold,
                holding_days=holding_days,
                treino_inicio=self.treino_inicio, treino_fim=self.treino_fim,
                treino_sharpe=r_treino.sharpe, treino_winrate=r_treino.win_rate,
                treino_retorno=r_treino.retorno_liquido, treino_trades=r_treino.n_trades,
                teste_inicio=self.teste_inicio, teste_fim=self.teste_fim,
                teste_sharpe=0.0, teste_winrate=0.0,
                teste_retorno=0.0, teste_trades=0,
                aprovado=False, motivo_rejeicao=motivo,
                momentum_meses=momentum_meses,
            )

        # Critérios de aprovação
        aprovado = True
        motivo = ""
        degradacao = r_treino.sharpe - r_teste.sharpe

        if r_teste.sharpe < self.min_sharpe_teste:
            aprovado = False
            motivo = f"sharpe teste={r_teste.sharpe:.2f} < {self.min_sharpe_teste}"
        elif r_teste.win_rate < self.min_winrate_teste:
            aprovado = False
            motivo = f"winrate teste={r_teste.win_rate:.0%} < {self.min_winrate_teste:.0%}"
        elif degradacao > self.max_degradacao_sharpe:
            aprovado = False
            motivo = f"degradação sharpe={degradacao:.2f} > {self.max_degradacao_sharpe}"

        return ResultadoWalkForward(
            ativo_a=ativo_a, ativo_b=ativo_b,
            direcao_a=direcao_a, direcao_b=direcao_b,
            contrarian=contrarian, threshold=threshold,
            holding_days=holding_days,
            treino_inicio=self.treino_inicio, treino_fim=self.treino_fim,
            treino_sharpe=round(r_treino.sharpe, 2),
            treino_winrate=r_treino.win_rate,
            treino_retorno=round(r_treino.retorno_liquido, 1),
            treino_trades=r_treino.n_trades,
            teste_inicio=self.teste_inicio, teste_fim=self.teste_fim,
            teste_sharpe=round(r_teste.sharpe, 2),
            teste_winrate=r_teste.win_rate,
            teste_retorno=round(r_teste.retorno_liquido, 1),
            teste_trades=r_teste.n_trades,
            aprovado=aprovado, motivo_rejeicao=motivo,
            momentum_meses=momentum_meses,
        )

    def validar_lista(
        self,
        candidatos: list,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Valida uma lista de candidatos.

        candidatos: lista de dicts com chaves:
            ativo_a, ativo_b, direcao_a, direcao_b,
            contrarian, threshold, holding_days
        """
        print(f"\n{'='*65}")
        print(f"WALK-FORWARD TEST — {len(candidatos)} candidatos")
        print(f"{'='*65}")

        resultados = []
        aprovados = 0

        for i, c in enumerate(candidatos, 1):
            a   = c["ativo_a"]
            b   = c["ativo_b"]
            da  = c["direcao_a"]
            db  = c.get("direcao_b", da)
            con = c.get("contrarian", False)
            thr = c.get("threshold", 0.3)
            hld = c.get("holding_days", 1)
            mom = int(c.get("momentum_meses", 0) or 0)

            modo = "MOMENTUM" if mom > 0 else ("CTR" if con else "MOM")
            desc = (f"mom={mom}m h={hld}d" if mom > 0 else f"thr={thr}% h={hld}d")
            print(f"  [{i:02d}/{len(candidatos)}] {a} {da}→{b} | {modo} "
                  f"{desc}...", end=" ", flush=True)

            r = self.validar_padrao(a, b, da, db, con, thr, hld, mom)
            if r:
                status = "✅" if r.aprovado else "❌"
                print(f"{status} treino={r.treino_sharpe:+.2f} "
                      f"teste={r.teste_sharpe:+.2f}")
                if verbose and r.aprovado:
                    print(r)
                resultados.append(r.resumo())
                if r.aprovado:
                    aprovados += 1
            else:
                print("⚠️  sem dados suficientes")

        df = pd.DataFrame(resultados) if resultados else pd.DataFrame()

        print(f"\n{'='*65}")
        print(f"RESULTADO: {aprovados}/{len(candidatos)} padrões aprovados")
        print(f"{'='*65}")

        if not df.empty:
            aprovados_df = df[df["aprovado"]].sort_values(
                "teste_sharpe", ascending=False
            )
            if not aprovados_df.empty:
                print("\nPADRÕES APROVADOS (ordenados por Sharpe no teste):")
                cols = ["ativo_a","ativo_b","direcao_a","contrarian",
                        "threshold_%","holding_days",
                        "treino_sharpe","teste_sharpe",
                        "teste_winrate_%","teste_retorno_%","teste_trades"]
                print(aprovados_df[cols].to_string(index=False))

        return df


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD DESLIZANTE
# ═══════════════════════════════════════════════════════════════

class WalkForwardDeslizante:
    """
    Walk-forward com janelas móveis — mais robusto que a janela única.

    Em vez de um único treino/teste, divide os dados em N janelas
    sobrepostas e valida cada padrão em todas elas. Um padrão só é
    aprovado se funcionar consistentemente na MAIORIA das janelas.

    Exemplo com 3 janelas de 18 meses cada:
      Janela 1: treino jan/21–jun/22 | teste jul/22–dez/22
      Janela 2: treino jan/21–dez/22 | teste jan/23–jun/23
      Janela 3: treino jan/21–jun/23 | teste jul/23–dez/23
      Janela 4: treino jan/21–dez/23 | teste jan/24–abr/26

    Critério de aprovação: Sharpe > 0 em pelo menos 60% das janelas
    E Sharpe médio > 0.3 across todas as janelas.
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        cdi: Optional[pd.DataFrame] = None,
        fedfunds: Optional[pd.DataFrame] = None,
        inicio_dados: str = "2016-01-04",
        fim_dados: str    = "2026-04-30",
        tamanho_teste_meses: int = 12,   # tamanho de cada janela de teste
        min_treino_meses: int   = 36,    # mínimo de meses de treino
        min_sharpe_medio: float = 0.3,
        min_pct_janelas_positivas: float = 0.6,  # 60% das janelas
        min_trades_por_janela: int = 5,
        custo_por_trade: float = 0.15,
    ):
        self.retornos   = retornos
        self.cdi        = cdi
        self.fedfunds   = fedfunds
        self.inicio     = pd.Timestamp(inicio_dados)
        self.fim        = pd.Timestamp(fim_dados)
        self.tam_teste  = tamanho_teste_meses
        self.min_treino = min_treino_meses
        self.min_sharpe_medio = min_sharpe_medio
        self.min_pct_positivas = min_janelas_positivas = min_pct_janelas_positivas
        self.min_trades = min_trades_por_janela
        self.custo      = custo_por_trade

        # Monta as janelas
        self.janelas = self._montar_janelas()
        print(f"\n📊 Walk-Forward Deslizante configurado:")
        print(f"   Dados: {inicio_dados} → {fim_dados}")
        print(f"   Janela de teste: {tamanho_teste_meses} meses")
        print(f"   Treino mínimo: {min_treino_meses} meses")
        print(f"   {len(self.janelas)} janelas geradas:")
        for i, (ti, tf, ei, ef) in enumerate(self.janelas, 1):
            print(f"     Janela {i}: treino {ti[:7]}→{tf[:7]} | teste {ei[:7]}→{ef[:7]}")

    def _montar_janelas(self) -> list:
        """
        Gera lista de (treino_inicio, treino_fim, teste_inicio, teste_fim).
        O treino sempre começa em inicio_dados e vai crescendo.
        O teste avança de tamanho_teste_meses em tamanho_teste_meses.
        """
        janelas = []
        teste_inicio = self.inicio + pd.DateOffset(months=self.min_treino)

        while True:
            teste_fim = teste_inicio + pd.DateOffset(months=self.tam_teste)
            if teste_fim > self.fim:
                # Última janela vai até o fim dos dados
                teste_fim = self.fim
            if teste_inicio >= self.fim:
                break

            treino_fim = teste_inicio - pd.DateOffset(days=1)
            janelas.append((
                self.inicio.strftime("%Y-%m-%d"),
                treino_fim.strftime("%Y-%m-%d"),
                teste_inicio.strftime("%Y-%m-%d"),
                teste_fim.strftime("%Y-%m-%d"),
            ))

            teste_inicio = teste_inicio + pd.DateOffset(months=self.tam_teste)
            if teste_fim >= self.fim:
                break

        return janelas

    def _rodar_janela(
        self,
        janela: tuple,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        contrarian: bool,
        threshold: float,
        holding_days: int,
    ) -> Optional[float]:
        """Roda backtest no período de teste de uma janela. Retorna Sharpe."""
        ti, tf, ei, ef = janela

        ret_teste = self.retornos[
            (self.retornos.index >= ei) &
            (self.retornos.index <= ef)
        ].copy()

        if len(ret_teste) < 20:
            return None

        bt = Backtester(
            retornos=ret_teste,
            cdi=self.cdi,
            fedfunds=self.fedfunds,
            custo_por_trade=self.custo,
            threshold=threshold,
            holding_days=holding_days,
        )

        r = (bt.rodar_contrarian(ativo_a, ativo_b, direcao_a, holding_days=holding_days)
             if contrarian else
             bt.rodar(ativo_a, ativo_b, direcao_a, direcao_b, holding_days=holding_days))

        if r and r.n_trades >= self.min_trades:
            return r.sharpe
        return None

    def validar_padrao(
        self,
        ativo_a: str,
        ativo_b: str,
        direcao_a: str,
        direcao_b: str,
        contrarian: bool,
        threshold: float,
        holding_days: int,
    ) -> dict:
        """
        Valida um padrão em todas as janelas e retorna resultado consolidado.

        Retorna dict com:
          aprovado          : bool
          sharpe_medio      : média dos Sharpes nas janelas
          pct_positivas     : % de janelas com Sharpe > 0
          sharpes_por_janela: lista de Sharpes (None se sem trades)
          motivo_rejeicao   : string explicando por que foi rejeitado
        """
        sharpes = []
        for janela in self.janelas:
            s = self._rodar_janela(
                janela, ativo_a, ativo_b,
                direcao_a, direcao_b, contrarian, threshold, holding_days
            )
            sharpes.append(s)

        sharpes_validos = [s for s in sharpes if s is not None]

        if len(sharpes_validos) == 0:
            return {
                "aprovado": False,
                "sharpe_medio": 0.0,
                "pct_positivas": 0.0,
                "sharpes_por_janela": sharpes,
                "motivo_rejeicao": "nenhuma janela com trades suficientes",
            }

        sharpe_medio = np.mean(sharpes_validos)
        n_positivas  = sum(1 for s in sharpes_validos if s > 0)
        pct_positivas = n_positivas / len(sharpes_validos)

        aprovado = True
        motivo   = ""

        if sharpe_medio < self.min_sharpe_medio:
            aprovado = False
            motivo = f"sharpe médio={sharpe_medio:.2f} < {self.min_sharpe_medio}"
        elif pct_positivas < self.min_pct_positivas:
            aprovado = False
            motivo = f"apenas {pct_positivas:.0%} das janelas positivas < {self.min_pct_positivas:.0%}"

        return {
            "aprovado": aprovado,
            "sharpe_medio": round(sharpe_medio, 2),
            "pct_positivas": round(pct_positivas, 2),
            "sharpes_por_janela": [round(s, 2) if s is not None else None for s in sharpes],
            "motivo_rejeicao": motivo,
        }

    def validar_lista(self, candidatos: list) -> pd.DataFrame:
        """Valida uma lista de candidatos em todas as janelas."""
        print(f"\n{'='*65}")
        print(f"WALK-FORWARD DESLIZANTE — {len(candidatos)} candidatos × {len(self.janelas)} janelas")
        print(f"{'='*65}")

        resultados = []
        for i, c in enumerate(candidatos, 1):
            a   = c["ativo_a"]
            b   = c["ativo_b"]
            da  = c["direcao_a"]
            db  = c.get("direcao_b", da)
            con = c.get("contrarian", False)
            thr = c.get("threshold", 0.3)
            hld = c.get("holding_days", 1)
            modo = "CTR" if con else "MOM"

            print(f"  [{i:02d}/{len(candidatos)}] {a} {da}→{b} | {modo} "
                  f"thr={thr}% h={hld}d...", end=" ", flush=True)

            res = self.validar_padrao(a, b, da, db, con, thr, hld)
            status = "✅" if res["aprovado"] else "❌"
            print(f"{status} sharpe_médio={res['sharpe_medio']:+.2f} | "
                  f"janelas+={res['pct_positivas']:.0%} | "
                  f"sharpes={res['sharpes_por_janela']}")

            resultados.append({
                "ativo_a": a, "ativo_b": b,
                "direcao_a": da, "direcao_b": db,
                "contrarian": con, "threshold_%": thr,
                "holding_days": hld,
                "aprovado": res["aprovado"],
                "sharpe_medio": res["sharpe_medio"],
                "pct_janelas_positivas": res["pct_positivas"],
                "sharpes_por_janela": str(res["sharpes_por_janela"]),
                "motivo_rejeicao": res["motivo_rejeicao"],
            })

        df = pd.DataFrame(resultados)
        n_ap = df["aprovado"].sum()
        print(f"\n{'='*65}")
        print(f"RESULTADO DESLIZANTE: {n_ap}/{len(candidatos)} aprovados")
        print(f"{'='*65}")
        if n_ap > 0:
            ap = df[df["aprovado"]].sort_values("sharpe_medio", ascending=False)
            print(ap[["ativo_a","ativo_b","direcao_a","contrarian",
                       "threshold_%","holding_days","sharpe_medio",
                       "pct_janelas_positivas"]].to_string(index=False))
        return df


# ═══════════════════════════════════════════════════════════════
# MONITOR DE DEGRADAÇÃO
# ═══════════════════════════════════════════════════════════════

class RegimeMonitor:
    """
    Monitora a performance de estratégias aprovadas em tempo real.

    A cada execução (idealmente diária), atualiza as métricas
    das últimas janelas temporais e suspende estratégias em degradação.

    Critérios de suspensão (configuráveis):
      - Win rate dos últimos 30d cai > 10pp abaixo do histórico
      - Sharpe dos últimos 60d < 0
      - Sequência de 5 trades negativos consecutivos
    """

    def __init__(
        self,
        pasta_estado: str,
        queda_winrate_suspensao: float = 0.10,  # 10pp de queda
        sharpe_60d_suspensao: float    = 0.0,
        seq_negativa_suspensao: int    = 5,
    ):
        self.pasta = pasta_estado
        self.queda_winrate    = queda_winrate_suspensao
        self.sharpe_60d_min   = sharpe_60d_suspensao
        self.seq_negativa_max = seq_negativa_suspensao
        os.makedirs(pasta_estado, exist_ok=True)

        self.arquivo_estado = os.path.join(pasta_estado, "estrategias_ativas.json")
        self.estrategias: dict = self._carregar_estado()

    def registrar_estrategia(
        self,
        resultado_wf: ResultadoWalkForward,
    ):
        """Registra uma estratégia aprovada no walk-forward."""
        id_est = self._gerar_id(resultado_wf)
        self.estrategias[id_est] = {
            "id": id_est,
            "ativo_a": resultado_wf.ativo_a,
            "ativo_b": resultado_wf.ativo_b,
            "direcao_a": resultado_wf.direcao_a,
            "direcao_b": resultado_wf.direcao_b,
            "contrarian": bool(resultado_wf.contrarian),
            "threshold": float(resultado_wf.threshold),
            "holding_days": int(resultado_wf.holding_days),
            "momentum_meses": int(getattr(resultado_wf, "momentum_meses", 0) or 0),
            "aprovada_em": datetime.today().strftime("%Y-%m-%d"),
            "status": "ativa",
            "sharpe_historico": float(resultado_wf.teste_sharpe),
            "winrate_historico": float(resultado_wf.teste_winrate),
            "historico_trades": [],   # lista de retornos individuais
            "dias_em_degradacao": 0,
            "motivo_suspensao": "",
        }
        self._salvar_estado()
        print(f"   ✅ Estratégia registrada: {id_est}")

    def atualizar(
        self,
        retornos_recentes: pd.DataFrame,
        cdi: Optional[pd.DataFrame] = None,
        janelas: list = [30, 60, 90],
    ) -> pd.DataFrame:
        """
        Atualiza métricas de todas as estratégias ativas.
        Deve ser chamado diariamente (ou semanalmente no mínimo).

        Retorna DataFrame com status atual de todas as estratégias.
        """
        print(f"\n{'='*60}")
        print(f"MONITOR DE DEGRADAÇÃO — {datetime.today().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        ativas = {k: v for k, v in self.estrategias.items()
                  if v["status"] == "ativa"}
        print(f"  Estratégias ativas: {len(ativas)}")

        relatorio = []

        for id_est, est in ativas.items():
            a   = est["ativo_a"]
            b   = est["ativo_b"]
            da  = est["direcao_a"]
            db  = est["direcao_b"]
            con = est["contrarian"]
            thr = est["threshold"]
            hld = est["holding_days"]

            metricas_janelas = {}
            suspender = False
            motivo    = ""

            for janela in janelas:
                data_corte = (
                    datetime.today() - timedelta(days=janela)
                ).strftime("%Y-%m-%d")
                ret_janela = retornos_recentes[
                    retornos_recentes.index >= data_corte
                ]

                if len(ret_janela) < 10:
                    continue

                bt = Backtester(
                    retornos=ret_janela, cdi=cdi,
                    custo_por_trade=0.15,
                    threshold=thr, holding_days=hld,
                )

                r = (bt.rodar_contrarian(a, b, da, holding_days=hld)
                     if con else
                     bt.rodar(a, b, da, db, holding_days=hld))

                if r and r.n_trades >= 3:
                    metricas_janelas[f"sharpe_{janela}d"] = round(r.sharpe, 2)
                    metricas_janelas[f"winrate_{janela}d"] = round(r.win_rate, 3)
                    metricas_janelas[f"trades_{janela}d"]  = r.n_trades

                    # Armazena retornos individuais da janela de 90d no estado
                    # (usado pelo KellyCalculator para calcular payoff ratio real)
                    if janela == 90 and hasattr(r, "curva_capital"):
                        rets_ind = list(
                            r.curva_capital.pct_change().dropna().values * 100
                        )
                        existente = self.estrategias[id_est].get("historico_trades", [])
                        # Mantém apenas os últimos 252 retornos (1 ano)
                        self.estrategias[id_est]["historico_trades"] = (
                            existente + rets_ind
                        )[-252:]

                    # Verifica critérios de suspensão
                    queda_wr = est["winrate_historico"] - r.win_rate
                    if janela == 60 and r.sharpe < self.sharpe_60d_min:
                        suspender = True
                        motivo = f"sharpe 60d={r.sharpe:.2f} < {self.sharpe_60d_min}"
                    if janela == 30 and queda_wr > self.queda_winrate:
                        suspender = True
                        motivo = f"queda winrate 30d={queda_wr:.0%} > {self.queda_winrate:.0%}"

            if suspender:
                self.estrategias[id_est]["status"] = "suspensa"
                self.estrategias[id_est]["motivo_suspensao"] = motivo
                self.estrategias[id_est]["dias_em_degradacao"] += 1
                print(f"  ⚠️  SUSPENSA: {id_est} — {motivo}")
            else:
                self.estrategias[id_est]["dias_em_degradacao"] = 0

            linha = {
                "id": id_est,
                "status": self.estrategias[id_est]["status"],
                "sharpe_hist": est["sharpe_historico"],
                **metricas_janelas,
                "motivo": self.estrategias[id_est]["motivo_suspensao"],
            }
            relatorio.append(linha)

        self._salvar_estado()
        df = pd.DataFrame(relatorio)
        if not df.empty:
            print(f"\n{df.to_string(index=False)}")
        return df

    def listar_ativas(self) -> list:
        """Retorna lista de estratégias ativas para o sistema de sinais."""
        return [
            v for v in self.estrategias.values()
            if v["status"] == "ativa"
        ]

    def reativar_suspensas(
        self,
        retornos: pd.DataFrame,
        cdi: Optional[pd.DataFrame] = None,
        min_sharpe_reativacao: float = 0.3,
    ):
        """
        Reavalia estratégias suspensas com dados recentes.
        Se voltarem a performar, reativa automaticamente.
        """
        suspensas = [v for v in self.estrategias.values()
                     if v["status"] == "suspensa"]
        print(f"\n🔄 Reavaliando {len(suspensas)} estratégias suspensas...")

        for est in suspensas:
            bt = Backtester(
                retornos=retornos, cdi=cdi,
                custo_por_trade=0.15,
                threshold=est["threshold"],
                holding_days=est["holding_days"],
            )
            r = (bt.rodar_contrarian(est["ativo_a"], est["ativo_b"],
                                     est["direcao_a"],
                                     holding_days=est["holding_days"])
                 if est["contrarian"] else
                 bt.rodar(est["ativo_a"], est["ativo_b"],
                          est["direcao_a"], est["direcao_b"],
                          holding_days=est["holding_days"]))

            if r and r.sharpe >= min_sharpe_reativacao:
                self.estrategias[est["id"]]["status"] = "ativa"
                self.estrategias[est["id"]]["motivo_suspensao"] = ""
                self.estrategias[est["id"]]["dias_em_degradacao"] = 0
                print(f"  ✅ Reativada: {est['id']} (sharpe={r.sharpe:.2f})")
            else:
                sharpe = r.sharpe if r else "N/A"
                print(f"  ❌ Mantida suspensa: {est['id']} (sharpe={sharpe})")

        self._salvar_estado()

    def _gerar_id(self, r: ResultadoWalkForward) -> str:
        modo = "CTR" if r.contrarian else "MOM"
        base = (f"{r.ativo_a}_{r.ativo_b}_{r.direcao_a}_"
                f"{modo}_thr{r.threshold}_h{r.holding_days}")
        # Tipo 4 (momentum) tem threshold 0 e ativo_a == ativo_b, então dois
        # padrões do mesmo ativo com janelas diferentes (3m vs 6m) gerariam
        # IDs idênticos. O sufixo _m{meses} desambigua. Os regexes de ID do
        # projeto aceitam esse sufixo como opcional, então IDs dos tipos 1-3
        # continuam iguais aos de antes.
        mom = getattr(r, "momentum_meses", 0) or 0
        if mom > 0:
            base += f"_m{int(mom)}"
        return base

    def _carregar_estado(self) -> dict:
        """
        Carrega o estado das estratégias registradas.

        Se o arquivo estiver corrompido, começa do zero em vez de
        propagar a exceção. Este JSON é um CACHE de estado — a fonte da
        verdade é o walk_forward.csv — então perdê-lo custa reprocessar,
        nunca vale derrubar um pipeline que já rodou por horas.

        Isso não é hipotético: um crash no meio do json.dump (bug do
        numpy.bool_, corrigido em 2026-09) deixou o arquivo truncado no
        meio de um campo, e ele derrubou o run overnight seguinte na
        etapa 3 de 4, depois de 4h de processamento.
        """
        if not os.path.exists(self.arquivo_estado):
            return {}
        try:
            with open(self.arquivo_estado, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            corrompido = self.arquivo_estado + ".corrompido"
            print(f"   ⚠️  {os.path.basename(self.arquivo_estado)} ilegível ({e}).")
            try:
                os.replace(self.arquivo_estado, corrompido)
                print(f"      Movido para {os.path.basename(corrompido)}; "
                      f"começando estado do zero.")
            except OSError:
                print(f"      Começando estado do zero (arquivo será sobrescrito).")
            return {}

    def _salvar_estado(self):
        def _json_default(o):
            # Campos vindos de DataFrames pandas às vezes carregam tipos
            # numpy (bool_, int64, float64) que o json padrão não serializa.
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        # Escrita atômica: grava num temporário e só então substitui o
        # arquivo real. Se o processo morrer no meio da serialização, o
        # arquivo bom continua intacto em vez de virar um JSON truncado.
        tmp = self.arquivo_estado + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.estrategias, f, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp, self.arquivo_estado)


# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from modo_teste import (
        adicionar_flag, banner,
        WF_TREINO_INICIO_TESTE, WF_TREINO_FIM_TESTE, WF_TESTE_INICIO_TESTE,
    )

    parser = argparse.ArgumentParser(description="Walk-forward — valida estratégias out-of-sample")
    adicionar_flag(parser)
    args = parser.parse_args()
    banner("walk_forward.py", args.teste)

    PASTA_RAW     = os.path.join(_BASE, "data", "raw")
    PASTA_REPORTS = os.path.join(_BASE, "reports", "teste") if args.teste \
                    else os.path.join(_BASE, "reports")
    PASTA_ESTADO  = os.path.join(PASTA_REPORTS, "estado")
    os.makedirs(PASTA_REPORTS, exist_ok=True)
    os.makedirs(PASTA_ESTADO,  exist_ok=True)

    # ── Carrega dados ─────────────────────────────
    print("📂 Carregando dados...")
    retornos = _carregar_retornos_completos(PASTA_RAW)

    cdi_path = os.path.join(PASTA_RAW, "cdi.csv")
    cdi = (pd.read_csv(cdi_path, index_col=0, parse_dates=True)
           if os.path.exists(cdi_path) else None)

    ff_path = os.path.join(PASTA_RAW, "fedfunds.csv")
    fedfunds = (pd.read_csv(ff_path, index_col=0, parse_dates=True)
                if os.path.exists(ff_path) else None)

    # ── Candidatos: lidos dinamicamente do backtester ──
    # Lê exp1, exp2 e exp3, filtra por mínimo de trades,
    # e monta a lista de candidatos para o walk-forward.
    # Assim o walk-forward sempre reflete os últimos resultados
    # do backtester, independente do modo (teste/rápido/completo).
    # Em modo teste há só ~2 anos de dados por padrão → poucos trades
    # possíveis. Um mínimo de 30 zeraria os candidatos quase sempre.
    MIN_TRADES = 5 if args.teste else 30

    CANDIDATOS = []
    vistos = set()

    for arquivo, tipo_exp in [
        ("exp1_holding_sweep.csv", "holding"),
        ("exp2_threshold_sweep.csv", "threshold"),
        ("exp3_contrarian.csv", "contrarian"),
        ("exp4_momentum.csv", "momentum"),
    ]:
        caminho_exp = os.path.join(PASTA_REPORTS, arquivo)
        if not os.path.exists(caminho_exp):
            print(f"   ⚠️  {arquivo} não encontrado — ignorado")
            continue

        df_exp = pd.read_csv(caminho_exp)

        # Filtra trades mínimos
        if "trades" in df_exp.columns:
            df_exp = df_exp[df_exp["trades"] >= MIN_TRADES]

        # Filtra Sharpe positivo
        if "sharpe" in df_exp.columns:
            df_exp = df_exp[df_exp["sharpe"] > 0]

        for _, row in df_exp.iterrows():
            if tipo_exp in ("holding", "threshold"):
                # Extrai ativo_a, direcao_a, ativo_b do campo "padrao"
                try:
                    partes = str(row.get("padrao","")).split(" → ")
                    parte_a = partes[0].split()
                    a  = " ".join(parte_a[:-1])
                    da = parte_a[-1].lower()
                    b  = partes[1].strip()
                    db = da
                    con = False
                    thr = float(row.get("melhor_threshold", 0.3) or 0.3)
                    hld = int(row.get("melhor_holding", 1) or 1)
                except Exception:
                    continue
            elif tipo_exp == "momentum":
                # Tipo 4: ativo_a == ativo_b, sinal é o momentum acumulado
                a   = str(row.get("ativo_a",""))
                b   = str(row.get("ativo_b", a))
                da  = str(row.get("direcao_a","sobe"))
                db  = str(row.get("direcao_b", da))
                con = False
                thr = 0.0
                hld = int(row.get("holding_days", 21))
                mom = int(row.get("momentum_meses", 0) or 0)
            else:  # contrarian
                a   = str(row.get("ativo_a",""))
                b   = str(row.get("ativo_b",""))
                da  = str(row.get("direcao_a","cai"))
                db  = "sobe" if da == "cai" else "cai"
                con = True
                thr = float(row.get("threshold_%", 1.0))
                hld = int(row.get("holding_days", 1))

            if tipo_exp != "momentum":
                mom = 0

            if not a or not b:
                continue

            chave = (a, b, da, con, thr, hld, mom)
            if chave in vistos:
                continue
            vistos.add(chave)

            CANDIDATOS.append({
                "ativo_a": a, "ativo_b": b,
                "direcao_a": da, "direcao_b": db,
                "contrarian": con,
                "threshold": thr,
                "holding_days": hld,
                "momentum_meses": mom,
            })

    print(f"\n📋 {len(CANDIDATOS)} candidatos carregados dos experimentos "
          f"(filtro: trades≥{MIN_TRADES}, sharpe>0)")

    if not CANDIDATOS:
        raise RuntimeError(
            "Nenhum candidato encontrado. "
            "Execute o backtester.py primeiro para gerar exp1/exp2/exp3."
        )

    # ══════════════════════════════════════════════════
    # WALK-FORWARD DE JANELA ÚNICA (validação principal)
    # ══════════════════════════════════════════════════
    if args.teste:
        wf = WalkForwardTest(
            retornos=retornos,
            cdi=cdi,
            fedfunds=fedfunds,
            treino_inicio=WF_TREINO_INICIO_TESTE,
            treino_fim   =WF_TREINO_FIM_TESTE,
            teste_inicio =WF_TESTE_INICIO_TESTE,
            teste_fim    ="2030-01-01",  # além dos dados disponíveis: usa tudo que existir
            # Critérios relaxados: com ~1 ano de teste e poucos candidatos,
            # o objetivo é exercitar o código (aprovar pelo menos alguns
            # candidatos), não validar estratégias de verdade.
            min_sharpe_teste      = 0.0,
            min_winrate_teste     = 0.0,
            min_trades_teste      = 3,
            max_degradacao_sharpe = 3.0,
        )
    else:
        wf = WalkForwardTest(
            retornos=retornos,
            cdi=cdi,
            fedfunds=fedfunds,
            treino_inicio="2016-01-04",
            treino_fim   ="2022-12-31",
            teste_inicio ="2023-01-01",
            teste_fim    ="2026-04-30",
            min_sharpe_teste      = 0.3,
            min_winrate_teste     = 0.50,
            min_trades_teste      = 15,
            max_degradacao_sharpe = 1.5,
        )

    df_resultados = wf.validar_lista(CANDIDATOS, verbose=True)

    # Salva walk_forward.csv com todos os resultados (aprovados e rejeitados)
    if not df_resultados.empty:
        caminho_wf = os.path.join(PASTA_REPORTS, "walk_forward.csv")
        df_resultados.to_csv(caminho_wf, index=False)
        print(f"\n💾 walk_forward.csv salvo ({len(df_resultados)} linhas)")

    # ══════════════════════════════════════════════════
    # WALK-FORWARD DESLIZANTE (validação de consistência)
    # Roda apenas com candidatos aprovados na janela única
    # ══════════════════════════════════════════════════
    aprovados_ambos = pd.DataFrame()

    if args.teste:
        # O deslizante precisa de ~4 anos de dados (36m treino + 12m teste
        # por janela) — o modo teste usa ~2 anos, então não cabe nenhuma
        # janela. Usa direto os aprovados da janela única para exercitar
        # o resto do pipeline (registro no monitor etc.)
        print("\n🧪 Modo teste: pulando walk-forward deslizante "
              "(dados insuficientes para janelas de 36m+12m) — "
              "usando aprovados da janela única.")
        if not df_resultados.empty:
            aprovados_ambos = df_resultados[df_resultados["aprovado"]]
    elif not df_resultados.empty:
        aprovados_janela_unica = df_resultados[df_resultados["aprovado"]]
        if not aprovados_janela_unica.empty:
            print(f"\n🔄 Rodando walk-forward deslizante com "
                  f"{len(aprovados_janela_unica)} aprovados da janela única...")

            candidatos_deslizante = []
            for _, row in aprovados_janela_unica.iterrows():
                candidatos_deslizante.append({
                    "ativo_a":     row["ativo_a"],
                    "ativo_b":     row["ativo_b"],
                    "direcao_a":   row["direcao_a"],
                    "direcao_b":   row["direcao_b"],
                    "contrarian":  row["contrarian"],
                    "threshold":   row["threshold_%"],
                    "holding_days":row["holding_days"],
                })

            wfd = WalkForwardDeslizante(
                retornos=retornos,
                cdi=cdi,
                fedfunds=fedfunds,
                inicio_dados="2016-01-04",
                fim_dados   ="2026-04-30",
                tamanho_teste_meses=12,
                min_treino_meses=36,
                min_sharpe_medio=0.2,
                min_pct_janelas_positivas=0.6,
                min_trades_por_janela=5,
            )

            df_deslizante = wfd.validar_lista(candidatos_deslizante)

            if not df_deslizante.empty:
                caminho_d = os.path.join(PASTA_REPORTS, "walk_forward_deslizante.csv")
                df_deslizante.to_csv(caminho_d, index=False)
                print(f"\n💾 walk_forward_deslizante.csv salvo")

                aprovados_ambos = df_deslizante[df_deslizante["aprovado"]]
                print(f"\n🏆 Aprovados em AMBOS os walk-forwards: "
                      f"{len(aprovados_ambos)}/{len(candidatos_deslizante)}")
        else:
            print("\n⚠️  Nenhum candidato aprovado na janela única — "
                  "walk-forward deslizante não rodado")

    # ── Registra NO MONITOR apenas aprovados em AMBOS ─────────
    # Garante que só entram estratégias com robustez confirmada
    # em múltiplas janelas temporais.
    monitor = RegimeMonitor(pasta_estado=PASTA_ESTADO)

    if not aprovados_ambos.empty:
        print(f"\n{'='*60}")
        print(f"REGISTRANDO {len(aprovados_ambos)} ESTRATÉGIAS NO MONITOR")
        print(f"(aprovadas em janela única E deslizante)")
        print(f"{'='*60}")

        for _, row in aprovados_ambos.iterrows():
            # Busca métricas detalhadas da janela única para este padrão
            match_unica = df_resultados[
                (df_resultados["ativo_a"]      == row["ativo_a"]) &
                (df_resultados["ativo_b"]      == row["ativo_b"]) &
                (df_resultados["direcao_a"]    == row["direcao_a"]) &
                (df_resultados["contrarian"]   == row["contrarian"]) &
                (df_resultados["threshold_%"]  == row["threshold_%"]) &
                (df_resultados["holding_days"] == row["holding_days"])
            ]

            if match_unica.empty:
                continue

            m = match_unica.iloc[0]
            r_wf = ResultadoWalkForward(
                ativo_a=m["ativo_a"], ativo_b=m["ativo_b"],
                direcao_a=m["direcao_a"], direcao_b=m["direcao_b"],
                contrarian=m["contrarian"],
                threshold=m["threshold_%"], holding_days=m["holding_days"],
                treino_inicio=wf.treino_inicio, treino_fim=wf.treino_fim,
                treino_sharpe=m["treino_sharpe"],
                treino_winrate=m["treino_winrate_%"] / 100,
                treino_retorno=m["treino_retorno_%"],
                treino_trades=m["treino_trades"],
                teste_inicio=wf.teste_inicio, teste_fim=wf.teste_fim,
                teste_sharpe=m["teste_sharpe"],
                teste_winrate=m["teste_winrate_%"] / 100,
                teste_retorno=m["teste_retorno_%"],
                teste_trades=m["teste_trades"],
                aprovado=True,
                momentum_meses=int(m.get("momentum_meses", 0) or 0),
            )
            monitor.registrar_estrategia(r_wf)
    else:
        print("\n⚠️  Nenhuma estratégia passou em ambos os walk-forwards — monitor não atualizado")

    # ── Atualiza métricas recentes via monitor (best-effort) ──────────────────
    # O monitor.atualizar() enriquece estratégias já registradas no JSON.
    # Se nenhuma estratégia passou nos dois WFs, o JSON pode estar vazio —
    # o monitor_status.csv será gerado diretamente dos aprovados abaixo.
    print(f"\n{'='*60}")
    print("MONITOR DE DEGRADAÇÃO — métricas dos últimos 90 dias")
    print(f"{'='*60}")

    data_90d = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    ret_recentes = retornos[retornos.index >= data_90d]
    df_monitor_recente = monitor.atualizar(ret_recentes, cdi=cdi)

    # ── Gera monitor_status.csv ────────────────────────────────────────────────
    # Base: TODOS os aprovados no walk-forward de janela única.
    # Enriquecimento: métricas recentes do monitor (quando disponíveis).
    # Isso garante que monitor_status.csv sempre é gerado após o walk-forward,
    # independente do resultado do walk-forward deslizante.
    if not df_resultados.empty and df_resultados["aprovado"].any():

        def _id(row):
            tipo = "CTR" if row["contrarian"] else "MOM"
            est_id = (f"{row['ativo_a']}_{row['ativo_b']}_{row['direcao_a']}_"
                      f"{tipo}_thr{row['threshold_%']}_h{row['holding_days']}")
            # Sem o sufixo de momentum, dois padrões tipo 4 do mesmo ativo
            # com janelas diferentes (3m e 6m) geram IDs idênticos e viram
            # linhas duplicadas no monitor. Precisa bater com _gerar_id.
            mom = int(row.get("momentum_meses", 0) or 0)
            if mom > 0:
                est_id += f"_m{mom}"
            return est_id

        aprovados_wf = df_resultados[df_resultados["aprovado"]].copy()
        aprovados_wf["id"] = aprovados_wf.apply(_id, axis=1)

        df_mon = pd.DataFrame({
            "id":           aprovados_wf["id"].values,
            "status":       "ativa",
            "sharpe_hist":  aprovados_wf["teste_sharpe"].values,
            "winrate_hist": (aprovados_wf["teste_winrate_%"] / 100.0).values,
            "motivo":       "",
            "sharpe_60d":   np.nan,
            "winrate_60d":  np.nan,
            "trades_60d":   np.nan,
            "sharpe_90d":   np.nan,
            "winrate_90d":  np.nan,
            "trades_90d":   np.nan,
        })

        # Enriquece com métricas recentes do monitor (quando disponíveis)
        if not df_monitor_recente.empty:
            cols_rec = [c for c in df_monitor_recente.columns
                        if c not in ("id", "status", "sharpe_hist", "motivo")]
            df_mon = df_mon.merge(
                df_monitor_recente[["id"] + cols_rec],
                on="id", how="left", suffixes=("", "_rec"),
            )
            for c in cols_rec:
                col_rec = c + "_rec"
                if col_rec in df_mon.columns:
                    df_mon[c] = df_mon[col_rec].combine_first(df_mon[c])
                    df_mon.drop(columns=[col_rec], inplace=True)

        # Suspende estratégias que o monitor marcou como degradadas
        if not df_monitor_recente.empty and "status" in df_monitor_recente.columns:
            suspensas = df_monitor_recente.loc[
                df_monitor_recente["status"] == "suspensa", "id"
            ]
            df_mon.loc[df_mon["id"].isin(suspensas), "status"] = "suspensa"
            df_mon.loc[df_mon["id"].isin(suspensas), "motivo"] = (
                df_monitor_recente.set_index("id")
                .reindex(df_mon.loc[df_mon["id"].isin(suspensas), "id"])["motivo"]
                .values
            )

        caminho_mon = os.path.join(PASTA_REPORTS, "monitor_status.csv")
        df_mon.to_csv(caminho_mon, index=False)
        n_ativas_mon = (df_mon["status"] == "ativa").sum()
        print(f"\n💾 monitor_status.csv salvo "
              f"({len(df_mon)} estratégias | {n_ativas_mon} ativas)")
    else:
        print("\n⚠️  Nenhuma estratégia aprovada — monitor_status.csv não atualizado")

    # ── Resumo final ──────────────────────────
    print(f"\n{'='*60}")
    print("RESUMO EXECUTIVO")
    print(f"{'='*60}")
    n_total  = len(df_resultados) if not df_resultados.empty else 0
    n_unica  = int(df_resultados["aprovado"].sum()) if not df_resultados.empty else 0
    n_ambos  = len(aprovados_ambos)
    n_ativas = len(monitor.listar_ativas())
    n_mon    = len(df_mon) if "df_mon" in dir() and not df_mon.empty else 0
    print(f"  Candidatos testados         : {n_total}")
    print(f"  Aprovados (janela única)    : {n_unica} ({n_unica/max(n_total,1):.0%})")
    print(f"  Aprovados (ambos WF)        : {n_ambos}")
    print(f"  Ativas no monitor           : {n_ativas}")
    print(f"  No monitor_status.csv       : {n_mon}")
    print(f"\n  Próximos passos:")
    print(f"  1. Rodar position_sizing.py com monitor_status.csv")
    print(f"  2. Rodar monitor.atualizar() semanalmente")
    print(f"  3. Implementar paper trading com as estratégias ativas")