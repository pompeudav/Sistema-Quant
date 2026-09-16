"""
correlator.py
-------------
Motor de detecção de padrões de correlação entre ativos.

Detecta quatro tipos de padrão:
  Tipo 1 — Correlação direcional simultânea
            "70% das vezes que A sobe, B sobe e C sobe"
  Tipo 2 — Correlação de magnitude
            "queda de X% em A está associada a queda de Y% em B"
  Tipo 3 — Correlação com defasagem temporal
            "60% das vezes que A cai, B sobe t dias depois"
  Tipo 4 — Momentum de médio prazo (sinal lento, sem par A→B)
            "60% das vezes que A acumulou alta em 6 meses, A continua subindo"

Os tipos 1-3 são sinais RÁPIDOS: dependem de um movimento acima de um
threshold num único dia, então só disparam quando há choque de curto prazo.
O tipo 4 é um sinal LENTO: funciona também em mercado calmo, cobrindo os
períodos em que os outros três ficam sem trabalho (ver Fase 4 / roteiro).
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────

@dataclass
class Padrao:
    """Representa um padrão de correlação detectado."""
    tipo: int                        # 1, 2, 3 ou 4
    ativo_a: str                     # ativo que "dispara" o padrão
    ativo_b: str                     # ativo que "reage"
    direcao_a: str                   # "sobe" ou "cai"
    direcao_b: str                   # "sobe" ou "cai"
    frequencia: float                # % de vezes que o padrão ocorre (0-1)
    n_ocorrencias: int               # quantas vezes A disparou
    n_confirmacoes: int              # quantas vezes B confirmou
    defasagem_dias: int = 0          # para tipo 3: quantos dias depois
    magnitude_a: float = 0.0         # para tipo 2: variação média de A (%)
    magnitude_b: float = 0.0         # para tipo 2: variação média de B (%)
    momentum_meses: int = 0          # para tipo 4: janela de momentum em meses
    correlacao_pearson: float = 0.0  # correlação linear
    p_valor: float = 1.0             # significância estatística (sem ajuste)
    p_valor_bh: float = 1.0          # p-valor após correção Benjamini-Hochberg
    score: float = 0.0               # score combinado (qualidade do padrão)

    def __str__(self):
        base = (
            f"[Tipo {self.tipo}] {self.ativo_a} → {self.ativo_b} "
            f"| {self.direcao_a.upper()} de A → {self.direcao_b.upper()} de B"
        )
        if self.defasagem_dias > 0:
            base += f" ({self.defasagem_dias}d depois)"
        base += (
            f"\n         Frequência: {self.frequencia:.1%} "
            f"({self.n_confirmacoes}/{self.n_ocorrencias} casos)"
            f" | p={self.p_valor:.4f} | p_bh={self.p_valor_bh:.4f} | score={self.score:.3f}"
        )
        if self.tipo == 2:
            base += (
                f"\n         Magnitude: A={self.magnitude_a:+.2f}% → "
                f"B={self.magnitude_b:+.2f}%"
            )
        if self.tipo == 4:
            base += f"\n         Momentum: janela de {self.momentum_meses} meses"
        return base


# ─────────────────────────────────────────────
# MOTOR PRINCIPAL
# ─────────────────────────────────────────────

class Correlator:
    """
    Detecta padrões de correlação entre ativos a partir de retornos diários.

    Uso básico:
        c = Correlator(df_retornos)
        padroes = c.detectar_tudo()
        c.relatorio(padroes)
    """

    def __init__(
        self,
        retornos: pd.DataFrame,
        min_frequencia: float = 0.55,   # mínimo 55% de ocorrências
        min_ocorrencias: int = 20,       # mínimo de casos para ser válido
        p_valor_max: float = 0.05,       # significância estatística
        lags_max: int = 5,               # máximo de dias de defasagem (tipo 3)
        threshold_movimento: float = 0.5, # mínima variação % para "contar"
        momentum_janelas_meses: list = None,  # janelas de momentum (tipo 4)
        momentum_horizonte_dias: int = 21,    # horizonte de holding do tipo 4
    ):
        """
        Parâmetros
        ----------
        retornos            : DataFrame com retornos diários em %
        min_frequencia      : frequência mínima para considerar padrão válido
        min_ocorrencias     : mínimo de dias em que A se moveu
        p_valor_max         : limite de significância (0.05 = 95% confiança)
        lags_max            : até quantos dias depois verificar (tipo 3)
        threshold_movimento : variação mínima em % para considerar "movimento"
        momentum_janelas_meses : janelas de momentum a testar no tipo 4
                                 (default: 3, 6 e 12 meses — as janelas
                                 clássicas da literatura de momentum)
        momentum_horizonte_dias: por quantos dias a posição do tipo 4 fica
                                 aberta. A detecção valida o padrão contra
                                 o retorno acumulado desse mesmo horizonte,
                                 para que o que é validado seja o que é
                                 operado (default 21 = ~1 mês).
        """
        self.retornos = retornos.copy().dropna(how="all")
        self.ativos = list(retornos.columns)
        self.min_frequencia = min_frequencia
        self.min_ocorrencias = min_ocorrencias
        self.p_valor_max = p_valor_max
        self.lags_max = lags_max
        self.threshold = threshold_movimento
        self.momentum_janelas_meses  = momentum_janelas_meses or [3, 6, 12]
        self.momentum_horizonte_dias = momentum_horizonte_dias

        print(f"✅ Correlator iniciado: {len(self.ativos)} ativos | "
              f"{len(self.retornos)} dias | "
              f"threshold={threshold_movimento}%")

    # ── Tipo 1: Correlação direcional ─────────

    def detectar_tipo1(self, pares: list[tuple] = None) -> list[Padrao]:
        """
        Detecta padrões direcionais simultâneos.
        "X% das vezes que A sobe/cai, B sobe/cai no mesmo dia."
        """
        print("\n🔍 Tipo 1: Correlações direcionais simultâneas...")
        padroes = []
        pares = pares or list(combinations(self.ativos, 2))
        total = len(pares)

        for i, (a, b) in enumerate(pares):
            if i % 50 == 0:
                print(f"   {i}/{total} pares analisados...", end="\r")

            ret_a = self.retornos[a].dropna()
            ret_b = self.retornos[b].dropna()
            idx_comum = ret_a.index.intersection(ret_b.index)

            if len(idx_comum) < self.min_ocorrencias:
                continue

            ra = ret_a[idx_comum]
            rb = ret_b[idx_comum]

            for dir_a in ["sobe", "cai"]:
                mascara_a = ra > self.threshold if dir_a == "sobe" else ra < -self.threshold
                n_ocorr = mascara_a.sum()

                if n_ocorr < self.min_ocorrencias:
                    continue

                rb_cond = rb[mascara_a]

                for dir_b in ["sobe", "cai"]:
                    mascara_b = rb_cond > self.threshold if dir_b == "sobe" else rb_cond < -self.threshold
                    n_conf = mascara_b.sum()
                    freq = n_conf / n_ocorr

                    if freq < self.min_frequencia:
                        continue

                    # Teste qui-quadrado
                    p_val = self._teste_proporcao(n_conf, n_ocorr, freq_esperada=0.5)

                    if p_val > self.p_valor_max:
                        continue

                    corr, _ = stats.pearsonr(ra, rb)
                    score = self._calcular_score(freq, n_ocorr, p_val)

                    padroes.append(Padrao(
                        tipo=1,
                        ativo_a=a, ativo_b=b,
                        direcao_a=dir_a, direcao_b=dir_b,
                        frequencia=freq,
                        n_ocorrencias=int(n_ocorr),
                        n_confirmacoes=int(n_conf),
                        correlacao_pearson=round(corr, 4),
                        p_valor=round(p_val, 6),
                        score=round(score, 4),
                    ))

        print(f"\n   ✅ {len(padroes)} padrões tipo 1 encontrados")
        return padroes

    # ── Tipo 2: Correlação de magnitude ───────

    def detectar_tipo2(self, pares: list[tuple] = None) -> list[Padrao]:
        """
        Detecta relações de magnitude entre movimentos.
        "Quando A cai X%, B cai em média Y%."
        """
        print("\n🔍 Tipo 2: Correlações de magnitude...")
        padroes = []
        pares = pares or list(combinations(self.ativos, 2))

        # Usa correlação de Pearson com threshold de significância
        for a, b in pares:
            ret_a = self.retornos[a].dropna()
            ret_b = self.retornos[b].dropna()
            idx = ret_a.index.intersection(ret_b.index)

            if len(idx) < self.min_ocorrencias:
                continue

            ra = ret_a[idx]
            rb = ret_b[idx]
            corr, p_val = stats.pearsonr(ra, rb)

            if p_val > self.p_valor_max:
                continue

            # Só padrões com correlação forte (|r| > 0.4)
            if abs(corr) < 0.4:
                continue

            # Estima magnitudes por regressão linear
            slope, intercept, _, _, _ = stats.linregress(ra, rb)

            # Exemplo representativo: movimento de 2% em A
            mag_a = 2.0
            mag_b = slope * mag_a + intercept

            dir_a = "sobe" if mag_a > 0 else "cai"
            dir_b = "sobe" if mag_b > 0 else "cai"

            score = self._calcular_score(abs(corr), len(idx), p_val)

            padroes.append(Padrao(
                tipo=2,
                ativo_a=a, ativo_b=b,
                direcao_a=dir_a, direcao_b=dir_b,
                frequencia=abs(corr),
                n_ocorrencias=len(idx),
                n_confirmacoes=len(idx),
                magnitude_a=round(mag_a, 2),
                magnitude_b=round(mag_b, 2),
                correlacao_pearson=round(corr, 4),
                p_valor=round(p_val, 6),
                score=round(score, 4),
            ))

        print(f"   ✅ {len(padroes)} padrões tipo 2 encontrados")
        return padroes

    # ── Tipo 3: Correlação com defasagem ──────

    def detectar_tipo3(self, pares: list[tuple] = None) -> list[Padrao]:
        """
        Detecta padrões com defasagem temporal.
        "X% das vezes que A cai, B sobe t dias depois."
        """
        print(f"\n🔍 Tipo 3: Correlações com defasagem (até {self.lags_max} dias)...")
        padroes = []
        pares = pares or list(combinations(self.ativos, 2))

        for a, b in pares:
            ret_a = self.retornos[a].dropna()
            ret_b = self.retornos[b].dropna()

            for lag in range(1, self.lags_max + 1):
                # B deslocado t dias para frente
                rb_lag = ret_b.shift(-lag)
                idx = ret_a.index.intersection(rb_lag.dropna().index)

                if len(idx) < self.min_ocorrencias:
                    continue

                ra = ret_a[idx]
                rb = rb_lag[idx]

                for dir_a in ["sobe", "cai"]:
                    mascara_a = ra > self.threshold if dir_a == "sobe" else ra < -self.threshold
                    n_ocorr = mascara_a.sum()

                    if n_ocorr < self.min_ocorrencias:
                        continue

                    rb_cond = rb[mascara_a]

                    for dir_b in ["sobe", "cai"]:
                        mascara_b = rb_cond > self.threshold if dir_b == "sobe" else rb_cond < -self.threshold
                        n_conf = mascara_b.sum()
                        freq = n_conf / n_ocorr

                        if freq < self.min_frequencia:
                            continue

                        p_val = self._teste_proporcao(n_conf, n_ocorr, freq_esperada=0.5)

                        if p_val > self.p_valor_max:
                            continue

                        corr, _ = stats.pearsonr(ra, rb)
                        score = self._calcular_score(freq, n_ocorr, p_val)

                        padroes.append(Padrao(
                            tipo=3,
                            ativo_a=a, ativo_b=b,
                            direcao_a=dir_a, direcao_b=dir_b,
                            frequencia=freq,
                            n_ocorrencias=int(n_ocorr),
                            n_confirmacoes=int(n_conf),
                            defasagem_dias=lag,
                            correlacao_pearson=round(corr, 4),
                            p_valor=round(p_val, 6),
                            score=round(score, 4),
                        ))

        print(f"   ✅ {len(padroes)} padrões tipo 3 encontrados")
        return padroes

    # ── Tipo 4: Momentum de médio prazo ────────

    def detectar_tipo4(self, ativos: list = None) -> list[Padrao]:
        """
        Detecta momentum de médio prazo (autocorrelação de tendência).
        "X% das vezes que A acumulou alta/baixa nos últimos N meses,
         A continua na mesma direção nos dias seguintes."

        Por que este tipo existe (adicionado 2026-09):
          Os tipos 1-3 só disparam quando um ativo se move mais que um
          threshold NUM DIA — ou seja, o sistema só "trabalha" quando há
          choque de curto prazo. Em mercados calmos, poucos sinais
          disparam e o capital fica ocioso (rendendo só CDI), o que
          explicava boa parte dos períodos em que o portfólio andava
          lateral ao CDI.

          O tipo 4 é um sinal LENTO: não precisa de choque nenhum. Ele
          olha para o retorno acumulado de 3-12 meses e pergunta se essa
          tendência tende a continuar. É o fator "momentum" clássico, com
          prêmio histórico bem documentado (Jegadeesh & Titman, 1993).

        Diferença estrutural em relação aos tipos 1-3:
          Aqui ativo_a == ativo_b (o sinal e a posição são no MESMO ativo)
          — não é um par A→B, é a própria série prevendo a si mesma. O
          resto do pipeline (walk-forward, sizing, monitor) trata isso
          normalmente, pois só precisa de (ativo_a, ativo_b, direção).
        """
        print(f"\n🔍 Tipo 4: Momentum de médio prazo "
              f"(janelas de {self.momentum_janelas_meses} meses)...")
        padroes = []
        ativos = ativos or self.ativos

        # CDI e FedFunds são séries de juros acumulados: sobem
        # monotonicamente por construção, então "momentum positivo" é
        # trivialmente verdadeiro ~100% do tempo. Como sinal seriam um
        # falso positivo perfeito, e como ATIVO não são compráveis do jeito
        # que o backtest assume (só existem como taxa livre de risco).
        # Nos tipos 1-3 eles fazem sentido (como ativo_a que dispara sinal
        # em OUTRO ativo); aqui, onde ativo_a == ativo_b, não fazem.
        NAO_OPERAVEIS = {"CDI", "FedFunds"}
        ativos = [a for a in ativos if a not in NAO_OPERAVEIS]

        for ativo in ativos:
            ret = self.retornos[ativo].dropna()
            if len(ret) < self.min_ocorrencias * 2:
                continue

            # Preço reconstruído — retornos estão em % (5.0 = 5%)
            preco = (1 + ret / 100).cumprod()

            for meses in self.momentum_janelas_meses:
                janela = meses * 21  # ~21 dias úteis por mês
                if len(preco) < janela * 2:
                    continue

                # Momentum passado: retorno acumulado nos últimos `janela` dias.
                # shift(1) garante que o sinal do dia t usa só dado até t-1
                # (sem lookahead — o retorno futuro que estamos prevendo
                # começa em t).
                mom_passado = preco.pct_change(janela).shift(1)

                # O que queremos prever: o retorno acumulado do HORIZONTE em
                # que a posição de fato ficará aberta (não o de 1 dia). Um
                # sinal lento é operado com holding de semanas; validá-lo
                # contra o retorno de amanhã mediria uma coisa e operaria
                # outra. pct_change(h).shift(-h) = retorno dos próximos h dias.
                h = self.momentum_horizonte_dias
                ret_futuro = preco.pct_change(h).shift(-h) * 100

                idx = mom_passado.dropna().index.intersection(ret_futuro.dropna().index)
                if len(idx) < self.min_ocorrencias:
                    continue

                # Observações NÃO-SOBREPOSTAS: com horizonte de h dias, o
                # retorno futuro do dia t e o do dia t+1 compartilham h-1
                # dias — são quase a mesma observação. O teste binomial
                # assume independência; alimentá-lo com dados sobrepostos
                # infla o n em ~h vezes e produz p-valores artificialmente
                # minúsculos (falsos positivos que nem o Benjamini-Hochberg
                # corrige, pois o BH corrige múltiplos testes, não
                # dependência dentro de um mesmo teste).
                # Amostrar a cada h dias custa poder estatístico, mas o
                # p-valor resultante é honesto.
                idx = idx[::h]

                # min_ocorrencias do projeto (30+) é calibrado para os tipos
                # 1-3, que contam dias individuais. Aqui cada observação
                # equivale a ~1 mês, então 10 anos inteiros geram no máximo
                # ~120 observações — exigir 30 já é um piso relativamente
                # alto (2,5 anos de dados só nesse lado do sinal).
                min_obs_t4 = max(20, self.min_ocorrencias // 2)
                if len(idx) < min_obs_t4:
                    continue

                mp = mom_passado[idx]
                rf = ret_futuro[idx]

                for dir_mom in ["sobe", "cai"]:
                    # Tendência de alta = momentum acumulado positivo
                    mascara = mp > 0 if dir_mom == "sobe" else mp < 0
                    n_ocorr = int(mascara.sum())
                    if n_ocorr < min_obs_t4:
                        continue

                    rf_cond = rf[mascara]

                    # Continuação: se veio subindo, continua subindo?
                    dir_esperada = dir_mom
                    n_conf = int((rf_cond > 0).sum() if dir_esperada == "sobe"
                                 else (rf_cond < 0).sum())
                    freq = n_conf / n_ocorr if n_ocorr else 0.0

                    if freq < self.min_frequencia:
                        continue

                    p_val = self._teste_proporcao(n_conf, n_ocorr, freq_esperada=0.5)
                    if p_val > self.p_valor_max:
                        continue

                    score = self._calcular_score(freq, n_ocorr, p_val)

                    padroes.append(Padrao(
                        tipo=4,
                        ativo_a=ativo, ativo_b=ativo,
                        direcao_a=dir_mom, direcao_b=dir_esperada,
                        frequencia=freq,
                        n_ocorrencias=n_ocorr,
                        n_confirmacoes=n_conf,
                        momentum_meses=meses,
                        p_valor=round(p_val, 6),
                        score=round(score, 4),
                    ))

        print(f"   ✅ {len(padroes)} padrões tipo 4 encontrados")
        return padroes

    # ── Detecção completa ──────────────────────

    def detectar_tudo(self, pares: list[tuple] = None) -> list[Padrao]:
        """
        Roda os quatro tipos de detecção e retorna todos os padrões,
        ordenados por score (melhor primeiro).
        """
        print("\n" + "="*60)
        print("INICIANDO DETECÇÃO DE PADRÕES")
        print("="*60)

        p1 = self.detectar_tipo1(pares)
        p2 = self.detectar_tipo2(pares)
        p3 = self.detectar_tipo3(pares)
        p4 = self.detectar_tipo4()

        todos = p1 + p2 + p3 + p4

        # Aplica correção de múltiplos testes (Benjamini-Hochberg)
        # ANTES de filtrar — a correção precisa ver todos os p-valores
        print(f"\n🔬 Aplicando correção de múltiplos testes...")
        todos = self._corrigir_bh(todos)

        # Filtra padrões rejeitados pelo BH (p_valor_bh == 1.0)
        n_antes = len(todos)
        todos = [p for p in todos if p.p_valor_bh < 1.0]
        todos.sort(key=lambda x: x.score, reverse=True)

        print(f"\n🏆 Total de padrões válidos após BH: {len(todos)} "
              f"(removidos: {n_antes - len(todos)})")
        return todos

    # ── Relatório ─────────────────────────────

    def relatorio(
        self,
        padroes: list[Padrao],
        top_n: int = 20,
        salvar_csv: bool = True,
        pasta: str = None,
    ) -> pd.DataFrame:
        """
        Exibe e salva os melhores padrões encontrados.

        Parâmetros
        ----------
        padroes   : lista retornada por detectar_tudo()
        top_n     : quantos padrões exibir
        salvar_csv: se True, salva resultados em CSV
        pasta     : pasta de destino (padrão: ../reports/)
        """
        if not padroes:
            print("⚠️  Nenhum padrão encontrado. Tente reduzir min_frequencia.")
            return pd.DataFrame()

        print(f"\n{'='*60}")
        print(f"TOP {min(top_n, len(padroes))} PADRÕES (de {len(padroes)} encontrados)")
        print(f"{'='*60}")

        for i, p in enumerate(padroes[:top_n], 1):
            print(f"\n#{i:02d}  {p}")

        # Converte para DataFrame
        df = pd.DataFrame([{
            "tipo": p.tipo,
            "ativo_a": p.ativo_a,
            "ativo_b": p.ativo_b,
            "direcao_a": p.direcao_a,
            "direcao_b": p.direcao_b,
            "frequencia_%": round(p.frequencia * 100, 1),
            "n_ocorrencias": p.n_ocorrencias,
            "n_confirmacoes": p.n_confirmacoes,
            "defasagem_dias": p.defasagem_dias,
            "magnitude_a_%": p.magnitude_a,
            "magnitude_b_%": p.magnitude_b,
            "momentum_meses": p.momentum_meses,
            "correlacao_pearson": p.correlacao_pearson,
            "p_valor": p.p_valor,
            "p_valor_bh": p.p_valor_bh,
            "score": p.score,
        } for p in padroes])

        if salvar_csv:
            pasta = pasta or os.path.join(
                os.path.dirname(__file__), "..", "reports"
            )
            os.makedirs(pasta, exist_ok=True)
            caminho = os.path.join(pasta, "padroes.csv")
            df.to_csv(caminho, index=False)
            print(f"\n💾 Resultados salvos em: {caminho}")

        return df

    def filtrar(
        self,
        padroes: list[Padrao],
        ativo: str = None,
        tipo: int = None,
        direcao_a: str = None,
        min_score: float = 0.0,
    ) -> list[Padrao]:
        """
        Filtra padrões por ativo, tipo ou direção.

        Uso: c.filtrar(padroes, ativo="PETR4.SA", direcao_a="cai")
        """
        resultado = padroes
        if ativo:
            resultado = [p for p in resultado if ativo in (p.ativo_a, p.ativo_b)]
        if tipo:
            resultado = [p for p in resultado if p.tipo == tipo]
        if direcao_a:
            resultado = [p for p in resultado if p.direcao_a == direcao_a]
        if min_score:
            resultado = [p for p in resultado if p.score >= min_score]
        return resultado

    # ── Correção de múltiplos testes ──────────

    def _corrigir_bh(self, padroes: list) -> list:
        """
        Aplica correção de Benjamini-Hochberg (FDR) aos p-valores de
        uma lista de padrões.

        Problema sem correção: com 95.000 pares testados a p<0.05,
        esperamos ~4.750 falsos positivos por puro acaso estatístico.
        A correção BH controla a taxa de falsas descobertas (FDR) —
        garante que no máximo X% dos padrões aprovados sejam falsos.

        Funciona assim:
          1. Ordena todos os p-valores do menor para o maior
          2. Calcula threshold ajustado para cada posição:
             threshold_i = (i / n_total) * fdr_rate
          3. Aprova até o último p-valor que ainda fica abaixo do threshold
          4. Rejeita todos os demais

        Referência: Benjamini & Hochberg (1995), JRSS-B.
        """
        if not padroes:
            return padroes

        fdr_rate = 0.05  # controla: máximo 5% de falsos positivos
        n = len(padroes)

        # Ordena por p-valor crescente
        ordenados = sorted(enumerate(padroes), key=lambda x: x[1].p_valor)
        p_valores = [p.p_valor for _, p in ordenados]

        # Calcula thresholds BH: (rank/n) * fdr_rate
        thresholds = [(i + 1) / n * fdr_rate for i in range(n)]

        # Encontra o maior rank onde p_valor <= threshold
        ultimo_aprovado = -1
        for i, (pv, thr) in enumerate(zip(p_valores, thresholds)):
            if pv <= thr:
                ultimo_aprovado = i

        # Atribui p_valor_bh: aprovados mantêm o p original,
        # rejeitados recebem p_bh = 1.0 (serão filtrados)
        resultado = list(padroes)  # cópia
        for rank, (idx_orig, padrao) in enumerate(ordenados):
            if rank <= ultimo_aprovado:
                # Aprovado: p_bh = p_valor ajustado (fórmula BH step-up)
                p_bh = min(padrao.p_valor * n / (rank + 1), 1.0)
                resultado[idx_orig].p_valor_bh = round(p_bh, 6)
            else:
                # Rejeitado: marca com p_bh = 1.0
                resultado[idx_orig].p_valor_bh = 1.0

        n_aprovados = sum(1 for p in resultado if p.p_valor_bh < 1.0)
        n_rejeitados = n - n_aprovados
        print(f"   🔬 Benjamini-Hochberg: {n_aprovados} aprovados | "
              f"{n_rejeitados} rejeitados (FDR={fdr_rate:.0%})")

        return resultado

    # ── Estatísticas auxiliares ────────────────

    def _teste_proporcao(
        self, n_sucesso: int, n_total: int, freq_esperada: float = 0.5
    ) -> float:
        """
        Teste binomial: verifica se a frequência observada é
        estatisticamente diferente do acaso (50%).
        Retorna p-valor.
        """
        resultado = stats.binomtest(n_sucesso, n_total, freq_esperada)
        return resultado.pvalue

    def _calcular_score(
        self, frequencia: float, n_ocorrencias: int, p_valor: float
    ) -> float:
        """
        Score combinado que considera:
        - frequência do padrão (mais importante)
        - número de ocorrências (mais dados = mais confiável)
        - significância estatística (p-valor baixo = melhor)
        """
        score_freq = frequencia                           # 0 a 1
        score_n = min(n_ocorrencias / 200, 1.0)          # normalizado em 200
        score_p = 1 - min(p_valor * 20, 1.0)             # penaliza p alto
        return 0.5 * score_freq + 0.3 * score_n + 0.2 * score_p


# ─────────────────────────────────────────────
# HELPERS DE CARREGAMENTO
# ─────────────────────────────────────────────

def _carregar_retornos_completos(pasta_raw: str) -> pd.DataFrame:
    """
    Monta um DataFrame unificado de retornos diários combinando:
      - retornos.csv      (Yahoo Finance — ações, commodities, cripto, FIIs)
      - retornos_fundos.csv (fundos CVM — Guepardo, Alaska, Constellation)
      - cdi.csv           (Banco Central — CDI como ativo comparável)
      - fedfunds.csv      (FRED — Fed Funds como ativo comparável)

    Estratégia de alinhamento de datas:
      Ativos BR negociam em dias úteis BR; ativos US em dias úteis US;
      fundos em dias úteis BR com 1d de defasagem; CDI/FedFunds em todos
      os dias. O merge usa outer join para preservar todos os dias e cada
      método de detecção faz dropna() nos pares individuais — assim dias
      sem sobreposição são ignorados automaticamente sem perder dados.
    """

    def _ler(nome: str) -> pd.DataFrame:
        caminho = os.path.join(pasta_raw, f"{nome}.csv")
        if not os.path.exists(caminho):
            print(f"   ⚠️  {nome}.csv não encontrado — ignorado")
            return pd.DataFrame()
        df = pd.read_csv(caminho, index_col=0, parse_dates=True)
        df.index.name = "data"
        return df

    print("\n📂 Carregando e mesclando todas as fontes de retornos...")

    # 1. Yahoo Finance
    ret_yahoo = _ler("retornos")
    print(f"   Yahoo Finance : {ret_yahoo.shape[1]} ativos | {ret_yahoo.shape[0]} dias")

    # 2. Fundos CVM
    ret_fundos = _ler("retornos_fundos")
    if not ret_fundos.empty:
        print(f"   Fundos CVM    : {ret_fundos.shape[1]} fundos | {ret_fundos.shape[0]} dias")

    # 3. CDI — converte índice acumulado em retorno diário %
    cdi_raw = _ler("cdi")
    ret_cdi = pd.DataFrame()
    if not cdi_raw.empty and "CDI" in cdi_raw.columns:
        ret_cdi = cdi_raw[["CDI"]].pct_change() * 100
        ret_cdi.columns = ["CDI"]
        ret_cdi = ret_cdi.iloc[1:]
        print(f"   CDI (BCB)     : {len(ret_cdi)} dias úteis")

    # 4. Fed Funds — converte índice acumulado em retorno diário %
    ff_raw = _ler("fedfunds")
    ret_ff = pd.DataFrame()
    if not ff_raw.empty and "FedFunds" in ff_raw.columns:
        # Filtra só dias úteis US (dias com variação != 0 ou NaN)
        ret_ff = ff_raw[["FedFunds"]].pct_change() * 100
        ret_ff.columns = ["FedFunds"]
        ret_ff = ret_ff.iloc[1:]
        # Remove fins de semana (retorno idêntico ao dia anterior por ffill)
        ret_ff = ret_ff[ret_ff.index.dayofweek < 5]
        print(f"   FedFunds(FRED): {len(ret_ff)} dias úteis")

    # Mescla tudo com outer join — NaNs são tratados por par em cada detecção
    frames = [f for f in [ret_yahoo, ret_fundos, ret_cdi, ret_ff] if not f.empty]
    if not frames:
        raise RuntimeError("Nenhum arquivo de retornos encontrado em " + pasta_raw)

    df_total = frames[0]
    for f in frames[1:]:
        df_total = df_total.join(f, how="outer")

    df_total = df_total.sort_index()
    print(f"\n   ✅ Base unificada: {df_total.shape[1]} ativos | {df_total.shape[0]} dias")
    return df_total


def _grupos_ativos(ativos: list = None) -> dict:
    """
    Classifica automaticamente qualquer ativo em um grupo
    com base em regras de nomenclatura de ticker.
    Funciona com o universo expandido de 400+ ativos.
    """
    BENCHMARKS = {
        "^BVSP","^GSPC","^NDX","^DJI","^RUT","^FTSE",
        "^N225","^HSI","^STOXX50E","^VIX","CDI","FedFunds",
    }
    COMMODITIES_FUTUROS = {
        "GC=F","SI=F","PL=F","PA=F","HG=F","ALI=F",
        "CL=F","BZ=F","NG=F","RB=F","HO=F",
        "ZS=F","ZC=F","ZW=F","ZR=F","ZO=F",
        "KC=F","CC=F","SB=F","CT=F","OJ=F",
        "GF=F","LE=F","HE=F",
    }
    COMMODITIES_ETFS = {"PICK","GDX","GDXJ","GLD","SLV","USO","UNG"}
    CRIPTO = {"BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
              "ADA-USD","DOGE-USD","AVAX-USD","COIN","MSTR"}
    CAMBIO = {"BRL=X","EUR=X","JPY=X","CNY=X","DX-Y.NYB"}
    ETFS_MERCADO = {"SPY","QQQ","IWM","EFA","EEM","XLE","XLF","XLK",
                    "XLV","XLI","XLB","XLRE","TLT","IEF","HYG","LQD",
                    "VNQ","ARKK"}
    FIIS = {"HGLG11.SA","MXRF11.SA","KNRI11.SA","XPML11.SA",
            "BTLG11.SA","VISC11.SA"}
    ETFS_BR = {"BOVA11.SA","IVVB11.SA","SMAL11.SA","IMAB11.SA","IRFM11.SA"}
    FUNDOS_BR = {"Guepardo Institucional Master FIA",
                 "Alaska Black Master FIF Ações",
                 "Constellation Compounders Master FIA"}
    # ADRs brasileiros listados em NY
    ADRS_BR = {"PBR","ERJ","GGB","BBD","ITUB"}
    # Internacionais conhecidos
    INTL = {"NVO","SHEL","AZN","UL","BP","DEO","SAP","BABA","JD",
            "PDD","BIDU","NTES","TM","HDB","INFY","WIT","BHP","RIO",
            "SNY","MELI","ASML"}

    grupos = {}
    fonte = ativos or []
    for ativo in fonte:
        if ativo in BENCHMARKS:
            grupos[ativo] = "benchmark"
        elif ativo in CAMBIO:
            grupos[ativo] = "cambio"
        elif ativo in COMMODITIES_FUTUROS:
            grupos[ativo] = "commodity"
        elif ativo in COMMODITIES_ETFS:
            grupos[ativo] = "commodity_etf"
        elif ativo in CRIPTO:
            grupos[ativo] = "cripto"
        elif ativo in FIIS:
            grupos[ativo] = "fii"
        elif ativo in ETFS_BR:
            grupos[ativo] = "etf_br"
        elif ativo in ETFS_MERCADO:
            grupos[ativo] = "etf_us"
        elif ativo in FUNDOS_BR:
            grupos[ativo] = "fundo_br"
        elif ativo in ADRS_BR:
            grupos[ativo] = "adr_br"
        elif ativo in INTL:
            grupos[ativo] = "acao_intl"
        elif ativo.endswith(".SA"):
            grupos[ativo] = "acao_br"
        else:
            grupos[ativo] = "acao_us"  # fallback: tickers sem sufixo = US
    return grupos


# ─────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    _BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_BASE)
    from modo_teste import adicionar_flag, ATIVOS_TESTE, DATA_INICIO_TESTE, banner

    parser = argparse.ArgumentParser(description="Correlator — busca pares A→B com edge estatístico")
    adicionar_flag(parser)
    parser.add_argument("--rapido", action="store_true",
                         help="Todos os ativos, mas só tipo 1+2 (pula tipo 3, o mais lento). ~30-40 min.")
    args = parser.parse_args()

    MODO_TESTE  = args.teste
    MODO_RAPIDO = args.rapido
    banner("correlator.py", MODO_TESTE)

    PASTA_RAW    = os.path.join(_BASE, "data", "raw")
    PASTA_REPORTS = os.path.join(_BASE, "reports", "teste") if MODO_TESTE \
                    else os.path.join(_BASE, "reports")
    os.makedirs(PASTA_REPORTS, exist_ok=True)

    # ── Carrega dados ─────────────────────────
    retornos = _carregar_retornos_completos(PASTA_RAW)

    n_ativos = retornos.shape[1]
    n_pares  = n_ativos * (n_ativos - 1) // 2
    print(f"\n📊 Universo: {n_ativos} ativos → {n_pares:,} pares possíveis")
    print(f"   Tipo 1: ~{n_pares*4:,} testes")
    print(f"   Tipo 2: ~{n_pares:,} testes")
    print(f"   Tipo 3: ~{n_pares*4*5:,} testes (4 dirs × 5 lags)")

    # ══════════════════════════════════════════
    # MODO DE EXECUÇÃO:
    #
    #   --teste    → segundos     (~20 ativos, ~2 anos, só tipo 1+2)
    #   --rapido   → ~30-40 min   (todos os ativos, tipo 1+2)
    #   nenhuma    → horas        (todos os ativos, tipo 1+2+3) ← a versão
    #                              que decide as estratégias de verdade
    # ══════════════════════════════════════════

    if MODO_TESTE:
        # Filtra ativos de teste E período curto — reduz tanto o número
        # de pares quanto o número de observações por par.
        cols_teste = [c for c in ATIVOS_TESTE if c in retornos.columns]
        retornos_usar = retornos[cols_teste]
        retornos_usar = retornos_usar[retornos_usar.index >= DATA_INICIO_TESTE]
        print(f"\n🧪 MODO TESTE — {len(cols_teste)} ativos, dados desde {DATA_INICIO_TESTE}")
        print(f"   Pares: {len(cols_teste)*(len(cols_teste)-1)//2}")
        print(f"   Reports em: {PASTA_REPORTS}")
    else:
        retornos_usar = retornos
        n_pares = retornos.shape[1] * (retornos.shape[1]-1) // 2
        print(f"\n📊 {retornos.shape[1]} ativos → {n_pares:,} pares")

    correlator = Correlator(
        retornos=retornos_usar,
        min_frequencia=0.55,
        min_ocorrencias=30,
        p_valor_max=0.05,
        lags_max=3 if (MODO_TESTE or MODO_RAPIDO) else 5,
        threshold_movimento=0.3,
    )

    t0 = time.time()

    if MODO_TESTE or MODO_RAPIDO:
        # Pula só o tipo 3 (defasagem), que é o caro: multiplica pares por
        # 4 direções × N lags. Os tipos 1, 2 e 4 rodam sempre — o tipo 4 é
        # por ATIVO (não por par), então é barato mesmo no universo completo.
        modo_label = "TESTE" if MODO_TESTE else "RÁPIDO"
        print(f"\n⚡ MODO {modo_label} — tipos 1, 2 e 4 (pula tipo 3)")
        p1 = correlator.detectar_tipo1()
        p2 = correlator.detectar_tipo2()
        p4 = correlator.detectar_tipo4()
        todos = p1 + p2 + p4
        # Aplica BH mesmo em modo rápido/teste
        print(f"\n🔬 Aplicando correção de múltiplos testes...")
        todos = correlator._corrigir_bh(todos)
        n_antes = len(todos)
        todos = [p for p in todos if p.p_valor_bh < 1.0]
        padroes = sorted(todos, key=lambda x: x.score, reverse=True)
        print(f"   Padrões após BH: {len(padroes)} "
              f"(removidos: {n_antes - len(padroes)})")
    else:
        padroes = correlator.detectar_tudo()

    elapsed = time.time() - t0
    print(f"\n⏱  Tempo total: {elapsed/60:.1f} minutos")

    # ── Relatório geral ───────────────────────
    df_resultado = correlator.relatorio(
        padroes, top_n=30, pasta=PASTA_REPORTS
    )

    # ── Relatório por grupo (dinâmico) ────────
    grupos = _grupos_ativos(list(retornos.columns))

    # Conta distribuição de grupos
    from collections import Counter
    dist = Counter(grupos.values())
    print("\n" + "="*60)
    print("DISTRIBUIÇÃO POR GRUPO")
    print("="*60)
    for grupo, qtd in sorted(dist.items()):
        print(f"  {grupo:<15}: {qtd} ativos")

    combos_interesse = [
        ("acao_br",      "commodity",      "BR vs Commodities"),
        ("acao_br",      "benchmark",      "BR vs Benchmarks"),
        ("acao_br",      "acao_us",        "BR vs US"),
        ("acao_br",      "cambio",         "BR vs Câmbio"),
        ("acao_us",      "commodity",      "US vs Commodities"),
        ("acao_us",      "commodity_etf",  "US vs ETFs Commodity"),
        ("cripto",       "acao_us",        "Cripto vs US"),
        ("cripto",       "benchmark",      "Cripto vs Benchmarks"),
        ("fundo_br",     "benchmark",      "Fundos BR vs Benchmarks"),
        ("fundo_br",     "acao_br",        "Fundos BR vs Ações BR"),
        ("adr_br",       "acao_br",        "ADRs BR vs Ações BR"),
        ("commodity",    "cambio",         "Commodities vs Câmbio"),
    ]

    print("\n" + "="*60)
    print("PADRÕES POR GRUPO DE ATIVOS (top 5 por grupo)")
    print("="*60)

    for grupo_a, grupo_b, titulo in combos_interesse:
        filtrados = [
            p for p in padroes
            if (grupos.get(p.ativo_a) == grupo_a and grupos.get(p.ativo_b) == grupo_b)
            or (grupos.get(p.ativo_a) == grupo_b and grupos.get(p.ativo_b) == grupo_a)
        ]
        if not filtrados:
            continue
        filtrados.sort(key=lambda x: x.score, reverse=True)
        print(f"\n── {titulo} ({len(filtrados)} padrões) ──")
        for p in filtrados[:5]:
            print(f"  {p}")

    # ── Tipo 3 ────────────────────────────────
    print("\n" + "="*60)
    print("PADRÕES TIPO 3 (DEFASAGEM TEMPORAL)")
    print("="*60)
    tipo3 = correlator.filtrar(padroes, tipo=3)
    if tipo3:
        for p in tipo3[:15]:
            print(f"  {p}")
    else:
        print("  Nenhum padrão tipo 3. Tente MODO_RAPIDO=False com lags_max=5.")

    # ── Filtros rápidos de interesse ──────────
    for ativo_ref, label in [
        ("CDI",   "CDI"),
        ("GC=F",  "Ouro"),
        ("BRL=X", "USD/BRL"),
        ("^VIX",  "VIX"),
        ("BZ=F",  "Petróleo Brent"),
    ]:
        pp = correlator.filtrar(padroes, ativo=ativo_ref)
        if pp:
            print(f"\n── {label} ({len(pp)} padrões) ──")
            for p in pp[:3]:
                print(f"  {p}")