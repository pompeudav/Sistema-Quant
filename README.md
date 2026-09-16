# Sistema Quant — descoberta e validação de estratégias

Sistema quantitativo pessoal com o objetivo de bater o CDI de forma consistente
(meta: IPCA+10 a.a.). Cobre 434 ativos (ações BR/US, ETFs, commodities, cripto,
câmbio, juros) com 10,7 anos de histórico diário.

**Estado atual (set/2026):** infraestrutura completa e validada; motor de sinais
original **não demonstrou edge** em testes honestos. Ver [Resultados](#resultados-o-que-funcionou-e-o-que-não).

---

## Como rodar

```bash
pip install -r requirements.txt
python data/collector.py            # baixa ~10 anos de dados (10-15 min)
python analysis/correlator.py --rapido   # descobre padrões (tipos 1, 2 e 4)
python backtest/backtester.py       # testa holding/threshold por padrão (~3h)
python analysis/walk_forward.py     # valida treino/teste + gera monitor
python analysis/portfolio_vs_cdi.py # simula portfólio com sizing e regime
```

**Modo teste** (roda em minutos, ~22 ativos, só para verificar que o código funciona —
os números não significam nada): adicione `--teste` a qualquer script.

**Operação diária** (`run_daily.ps1`, agendado no Task Scheduler às 19h30):
`collector → monitor_degradacao → gerar_monitor → position_sizing → paper_trading`

---

## Arquitetura

```
data/collector.py          Baixa preços (yfinance), CDI (BCB), Fed Funds (FRED), fundos (CVM).
                           Filtra ativos com >15% de dados faltando ou saltos de preço impossíveis.

analysis/correlator.py     Descobre padrões estatísticos entre pares de ativos:
                             Tipo 1 — direcional simultâneo ("A sobe → B sobe no mesmo dia")
                             Tipo 2 — magnitude ("A sobe X% → B sobe Y%")
                             Tipo 3 — defasagem temporal ("A hoje → B daqui a N dias")
                             Tipo 4 — momentum do próprio ativo (3/6/12 meses)
                           Correção de múltiplos testes por Benjamini-Hochberg.

backtest/backtester.py     Para cada padrão, varre holding (1-42d) e threshold (0.3-3%).
                           Mede win rate, Sharpe, drawdown, custo.

analysis/walk_forward.py   Treina em 2016-2022, testa em 2023+. Aprova só o que mantém
                           Sharpe > 0.3 e win rate > 50% fora da amostra.
                           Também roda walk-forward deslizante (janelas de 12 meses).

analysis/position_sizing.py
                           Kelly fracionário × ajuste por volatilidade × teto por ativo (2%)
                           × teto de EXPOSIÇÃO EFETIVA (35% — tamanho ponderado pelo tempo
                           que a posição fica aberta; sem isso holdings longos alavancam
                           o portfólio sem violar nenhum limite nominal).

analysis/regime_detector.py
                           Classifica cada dia em risk-on / risk-off / inflacionário /
                           deflacionário (VIX, treasuries, DXY, CDI, commodities) e ajusta
                           os tamanhos por regime.

analysis/portfolio_vs_cdi.py
                           Simula a carteira dia a dia vs CDI. Inclui CDI overlay sobre
                           capital ocioso, circuit breaker e trend following long/short.

analysis/paper_trading.py  Registra os sinais que dispararam de fato e o resultado que
                           tiveram. Critério de go-live: Sharpe líquido do CDI ≥ 0.5 no
                           período todo E ≥ 0.8 nos últimos 30d, com ≥ 60 dias de histórico.

analysis/monitor_degradacao.py
                           Desliga estratégias que pararam de funcionar no paper trading
                           (win rate < 40% ou perda > 5% em 60 dias).

walk_forward_descoberta.py
                           Backtest SEM viés de seleção: redescobre os padrões a cada
                           6 meses usando só dados até aquela data. É o único backtest
                           deste repositório em que dá para confiar.

teste_trend_following.py   Teste de tese isolado: trend following diversificado em
                           36 ativos, sem otimização de parâmetros.
```

---

## Resultados: o que funcionou e o que não

Este é o resultado mais importante do projeto e o motivo de compartilhá-lo.

### O backtest normal era ilusão

| Métrica | Backtest normal | Sem viés de seleção |
|---|---|---|
| CAGR | +20,8% | **+4,3%** |
| Sharpe | 1,91 | 0,77 |
| vs CDI | +11pp/ano | **-5,2pp/ano** |

O pipeline descobria padrões olhando o histórico **inteiro**, e só depois dividia
treino/teste. Os candidatos que chegavam ao teste já tinham sido escolhidos sabendo
o que aconteceria nele. Testar 7.248 estratégias e ficar com as 1.717 que "passaram"
garante que sobrevivam as que deram sorte no período de teste.

Prova nos dados: win rate de 57,8% no período usado para selecionar → **50,2%**
(cara ou coroa) em dados genuinamente novos.

### O que o paper trading mostra (dados nunca vistos, maio-set/2026)

- Retorno: +6,2% vs CDI +4,95% → **+3,8pp ao ano** sobre o CDI
- Sharpe líquido do CDI: **0,33** (critério de go-live: 0,5 — reprovado)
- Custos consomem **36% do retorno bruto**
- 665 de 1.561 estratégias têm contribuição negativa
- Tipo 4 (momentum) é **3,5x mais eficiente** por unidade de custo que os pares

### Trend following diversificado também não sustentou

Sharpe 0,17 sem custos, -0,03 com custos. O long-only ganha 3% a.a. mas é beta de
bull market, não alpha. Cripto e juros foram as únicas classes positivas.

### Conclusão

Duas teses testadas com rigor, duas falharam. O motor de correlação de pares não tem
mecanismo econômico (por que PETR4 preveria WEGE3 com 3 dias de antecedência?) e o
trend following não se manifestou neste universo/período.

Com o CDI em 14,6% a.a. (juro real ~9%), a barra é excepcionalmente alta. A mudança
estrutural de maior impacto identificada — manter 100% em Tesouro Selic e operar via
futuros com margem — valeu +9,6pp/ano no teste, mas só faz sentido se houver um sinal
com edge para operar por cima.

---

## Lições de engenharia (bugs que custaram dias)

- **`retornos.csv` está em formato percentual** (5.0 = 5%). Quatro módulos compunham
  `(1 + r).cumprod()` sem dividir por 100 — o preço reconstruído explodia e o regime
  detector nunca classificou um dia como "inflacionário" em 10 anos. 2022 aparecia
  como 83% risk-on.
- **Sharpe por trade × √252 infla o número.** O critério de go-live reportava 2,46
  quando o valor real era 0,20. Calcular sempre sobre o retorno diário do portfólio,
  líquido do CDI, anualizado pelo número real de observações por ano (o índice tem
  fim de semana por causa de cripto — não são 252 dias).
- **Exposição nominal ≠ exposição real.** Somar tamanhos sem olhar quanto tempo cada
  posição fica aberta deixou 613 estratégias de momentum multiplicarem o capital em
  risco por 6 sem violar nenhum teto.
- **PowerShell 5.1 lê `.ps1` como ANSI.** Acentos em UTF-8 quebram o parser antes de
  executar qualquer linha. Scripts de automação em ASCII puro.
- **Dados corrompidos passam silenciosamente.** XPML11 saltou de R$0,14 para R$106
  num dia (erro de ajuste do yfinance) e inflou o backtest inteiro. O collector agora
  descarta ativos com fator diário > 20x entre preços positivos.

---

## Backups e dados

`data/raw/` (1,5 GB) e `reports/` são regenerados pelo pipeline e não estão no
repositório. `backup_codigo_*/` são snapshots locais anteriores a cada mudança
estrutural.

