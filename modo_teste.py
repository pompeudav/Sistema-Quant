"""
modo_teste.py
-------------
Configuração compartilhada do "modo teste" (flag --teste).

Problema que resolve: com o universo completo (~487 ativos, 2016-2026),
collector -> correlator -> backtester -> walk_forward leva horas. Isso
torna caro descobrir se uma mudança de código quebrou alguma coisa.

O modo teste roda o MESMO pipeline, ponta a ponta, com um universo pequeno
(~20 ativos representativos de cada grupo) e um período curto (~2 anos).
Não serve para encontrar as melhores estratégias (poucos dados, poucos
ativos) — serve para confirmar que o código funciona antes de rodar a
versão completa (que ainda deve ser rodada, tipicamente overnight).

Uso em cada script:
    from modo_teste import adicionar_flag, ATIVOS_TESTE, ...

    parser = argparse.ArgumentParser(...)
    adicionar_flag(parser)
    args = parser.parse_args()
    if args.teste:
        ...usar ATIVOS_TESTE, DATA_INICIO_TESTE, etc.
"""

import argparse

# Ativos representativos de cada grupo do universo completo (BR, US,
# commodities, cripto, câmbio, benchmarks) — poucos o bastante para rodar
# em segundos/minutos, variados o bastante para exercitar todo o código
# (tipo 1/2/3 do correlator, holding/threshold sweep, contrarian, etc.)
ATIVOS_TESTE = [
    "^BVSP", "^GSPC", "^NDX", "BRL=X", "^VIX",
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "PRIO3.SA",
    "AAPL", "NVDA", "AMZN", "JPM", "XOM",
    "GC=F", "CL=F", "BZ=F", "HG=F",
    "BTC-USD", "ETH-USD",
    "CDI", "FedFunds",
]

# ~2 anos e meio: dá dias suficientes para vários trades por par com
# holding de 1-10 dias, e ainda sobra espaço para um corte treino/teste
# dentro da janela no walk-forward.
DATA_INICIO_TESTE = "2023-06-01"

# Sweeps reduzidos (3 valores em vez de 5-7) — só para exercitar o código.
HOLDINGS_TESTE   = [1, 3, 5]
THRESHOLDS_TESTE = [0.5, 1.0, 2.0]

# Walk-forward: uma janela única e curta, sem a versão deslizante
# (que multiplica o tempo por N janelas e não cabe em ~2 anos de dados).
WF_TREINO_INICIO_TESTE = "2023-06-01"
WF_TREINO_FIM_TESTE    = "2024-12-31"
WF_TESTE_INICIO_TESTE  = "2025-01-01"


def adicionar_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adiciona a flag --teste a um parser existente."""
    parser.add_argument(
        "--teste", action="store_true",
        help=(
            "Modo teste: ~20 ativos, ~2 anos de dados, sweeps reduzidos. "
            "Roda em minutos em vez de horas — use para validar que o "
            "pipeline inteiro funciona antes de rodar a versão completa."
        ),
    )
    return parser


def banner(nome_script: str, ligado: bool):
    if ligado:
        print(f"\n🧪 MODO TESTE ativo em {nome_script} — universo e período reduzidos.")
        print("   Resultados aqui NÃO servem para decidir estratégias, só para")
        print("   confirmar que o pipeline roda sem erro. Rode sem --teste para valer.\n")
