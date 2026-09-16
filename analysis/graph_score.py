"""
graph_score.py
--------------
Fase 6 (opcional) — Score estrutural via grafo de correlação.

De onde veio a ideia: o projeto do amigo no Desafio Quant AI - Itaú Asset
2025 (Insper Quantitative Finance) modela o mercado como um grafo e usa uma
Graph Attention Network (GAT) para achar padrões estruturais na rede de
correlações. A intuição central é boa: um par A→B que conecta dois "hubs"
do mercado (ex.: petróleo → câmbio → ações exportadoras) tende a refletir
uma relação mais estrutural do que um par entre dois ativos periféricos e
isolados — mesmo que ambos tenham passado no mesmo filtro estatístico.

Por que este módulo NÃO treina uma GAT agora:
  Uma GAT de verdade tem dezenas de milhares de parâmetros treináveis.
  Com um universo de ~20-450 ativos e holding de 1-10 dias, este projeto
  gera no máximo dezenas de milhares de observações úteis — regime de
  overfitting garantido para uma rede neural sobre grafos. "Top 10" numa
  competição não prova que uma GAT generaliza fora da amostra; prova que
  ela memorizou bem o período de treino da competição.

O que este módulo faz em vez disso:
  Usa métricas de centralidade de grafo CLÁSSICAS (grau, betweenness,
  eigenvector) — zero parâmetros treináveis, zero risco de overfitting —
  como um FILTRO ADICIONAL sobre os pares que JÁ passaram pelo funil
  estatístico do projeto (correlator + Benjamini-Hochberg + walk-forward).
  Não decide sozinho quais estratégias operar; ajusta um pouco o tamanho
  da posição (Kelly) para cima em pares estruturalmente centrais e para
  baixo em pares periféricos.

Quando revisitar a ideia da GAT de verdade:
  Depois de 30-60 dias de paper trading (Fase 5), quando existir um alvo
  real — retorno ao vivo por estratégia — para treinar/validar um modelo
  sem contaminar a amostra com o próprio backtest que o gerou. Ver roteiro,
  Fase 6.

Uso:
    python analysis/graph_score.py            # usa reports/padroes.csv
    python analysis/graph_score.py --teste     # usa reports/teste/padroes.csv

Integração (opcional, desligada por padrão):
    python analysis/position_sizing.py --usar-grafo
    python analysis/portfolio_vs_cdi.py --usar-grafo
"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("Execute: pip install networkx")

# Regex do formato de ID de estratégia usado no projeto todo:
# {ativo_a}_{ativo_b}_{direcao}_{CTR|MOM}_thr{X.X}_h{Y}
_ID_RE = re.compile(r'^(.+?)_(.+?)_(sobe|cai)_(CTR|MOM)_thr[\d.]+_h\d+(?:_m\d+)?$')


def parse_ativos_do_id(id_str: str):
    """Extrai (ativo_a, ativo_b) de um ID de estratégia. Retorna (None, None) se não casar."""
    m = _ID_RE.match(str(id_str))
    if not m:
        return None, None
    return m.group(1), m.group(2)


# ═══════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO GRAFO
# ═══════════════════════════════════════════════════════════════

def montar_grafo(padroes_df: pd.DataFrame, peso_col: str = "score") -> "nx.Graph":
    """
    Constrói o grafo de mercado a partir dos padrões aprovados pelo
    correlator (padroes.csv, já filtrado por Benjamini-Hochberg):
    nós = ativos, arestas = pares com padrão detectado, peso = score
    do melhor padrão encontrado entre os dois ativos.
    """
    G = nx.Graph()
    if padroes_df.empty:
        return G

    for _, row in padroes_df.iterrows():
        a, b = row.get("ativo_a"), row.get("ativo_b")
        if not a or not b:
            continue
        peso = float(row[peso_col]) if peso_col in row and pd.notna(row[peso_col]) else 1.0
        if G.has_edge(a, b):
            G[a][b]["weight"] = max(G[a][b]["weight"], peso)
        else:
            G.add_edge(a, b, weight=peso)
    return G


def calcular_centralidade(G: "nx.Graph") -> pd.DataFrame:
    """
    Calcula 3 métricas de centralidade por ativo (nó do grafo):

      grau (ponderado)  — quantas conexões relevantes o ativo tem, e quão
                           fortes elas são (soma dos scores das arestas).
      betweenness       — quão "intermediário" o ativo é entre outros
                           pares (ex.: câmbio costuma ligar commodities a
                           ações exportadoras — betweenness alto).
      eigenvector       — quão conectado o ativo está a OUTROS hubs
                           importantes, não só a muitos ativos periféricos
                           (a diferença central que o GAT captura melhor
                           que uma correlação par-a-par simples).

    As três são normalizadas para [0,1] e combinadas num score único.
    Ativos fora do grafo (sem nenhum padrão aprovado) simplesmente não
    aparecem — tratados como neutros (score 0.5) por quem consome isso.
    """
    cols = ["ativo", "grau", "betweenness", "eigenvector", "score_estrutural"]
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=cols)

    grau = dict(G.degree(weight="weight"))
    btw  = nx.betweenness_centrality(G, weight="weight")
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G.nodes()}

    df = pd.DataFrame({
        "ativo":       list(G.nodes()),
        "grau":        [grau.get(n, 0.0) for n in G.nodes()],
        "betweenness": [btw.get(n, 0.0) for n in G.nodes()],
        "eigenvector": [eig.get(n, 0.0) for n in G.nodes()],
    })

    for col in ["grau", "betweenness", "eigenvector"]:
        maxv = df[col].max()
        df[col + "_norm"] = df[col] / maxv if maxv > 0 else 0.0

    df["score_estrutural"] = (
        df["grau_norm"] * 0.30 +
        df["betweenness_norm"] * 0.40 +
        df["eigenvector_norm"] * 0.30
    )
    return df.sort_values("score_estrutural", ascending=False).reset_index(drop=True)


def score_par(centralidade_df: pd.DataFrame, ativo_a: str, ativo_b: str) -> float:
    """
    Multiplicador estrutural de um PAR — média do score estrutural de A e B,
    mapeado para uma faixa modesta: 0.85x (periférico) a 1.15x (hub).

    Deliberadamente contido: a posição no grafo é um desempate, não uma
    tese de investimento por si só. Nunca dobra nem zera uma posição.
    Retorna 1.0 (neutro) se um dos ativos não estiver no grafo.
    """
    if centralidade_df.empty:
        return 1.0
    idx = centralidade_df.set_index("ativo")["score_estrutural"]
    sa = idx.get(ativo_a, np.nan)
    sb = idx.get(ativo_b, np.nan)
    if pd.isna(sa) or pd.isna(sb):
        return 1.0
    bruto = (sa + sb) / 2
    return 0.85 + 0.30 * float(bruto)


def aplicar_em_sizes(sizes_df: pd.DataFrame, centralidade_df: pd.DataFrame,
                      col_size: str = "size_final_pct") -> pd.DataFrame:
    """
    Aplica o multiplicador estrutural sobre um DataFrame de position sizes
    (position_sizes.csv ou equivalente em memória). Espera uma coluna 'id'
    no formato {ativo_a}_{ativo_b}_{direcao}_{CTR|MOM}_thr{X}_h{Y}.

    Adiciona colunas 'mult_estrutural' e substitui col_size pelo valor
    ajustado (guarda o original em '{col_size}_pre_grafo').
    """
    df = sizes_df.copy()
    ativos = df["id"].apply(parse_ativos_do_id)
    df["_ativo_a"] = ativos.apply(lambda x: x[0])
    df["_ativo_b"] = ativos.apply(lambda x: x[1])
    df["mult_estrutural"] = df.apply(
        lambda r: score_par(centralidade_df, r["_ativo_a"], r["_ativo_b"])
        if r["_ativo_a"] else 1.0,
        axis=1,
    )
    df[f"{col_size}_pre_grafo"] = df[col_size]
    df[col_size] = df[col_size] * df["mult_estrutural"]
    df = df.drop(columns=["_ativo_a", "_ativo_b"])
    return df


# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_BASE)
    from modo_teste import adicionar_flag, banner

    parser = argparse.ArgumentParser(
        description="Score estrutural via grafo de correlação (Fase 6, opcional)"
    )
    adicionar_flag(parser)
    args = parser.parse_args()
    banner("graph_score.py", args.teste)

    pasta_reports = os.path.join(_BASE, "reports", "teste") if args.teste \
                    else os.path.join(_BASE, "reports")
    caminho = os.path.join(pasta_reports, "padroes.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"{caminho} não encontrado — rode correlator.py primeiro.")

    padroes = pd.read_csv(caminho)
    G = montar_grafo(padroes)
    print(f"\n📊 Grafo de mercado: {G.number_of_nodes()} ativos, {G.number_of_edges()} conexões")

    cent = calcular_centralidade(G)
    caminho_out = os.path.join(pasta_reports, "graph_centralidade.csv")
    cent.to_csv(caminho_out, index=False)

    print(f"\nTop 15 ativos por score estrutural (posição de 'hub' no grafo):")
    print(cent.head(15).to_string(index=False))
    print(f"\n💾 graph_centralidade.csv salvo em {pasta_reports}")
    print(f"\nPara usar no position sizing:  python analysis/position_sizing.py --usar-grafo")
