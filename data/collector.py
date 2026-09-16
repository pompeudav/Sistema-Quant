"""
collector.py  (v3)
------------------
Universo expandido: ~487 ativos
  - ~138 ações BR  (Ibovespa + SMLL + IDIV)
  - ~237 ações US  (top 200 S&P500 + top 50 Nasdaq100 sem duplicatas)
  -   28 ADRs internacionais
  -   24 commodities (futuros)
  -   28 ETFs temáticos
  -   10 cripto + proxies
  -   16 benchmarks e câmbio
  -    6 FIIs

Fontes:
  1. Yahoo Finance  — todos os ativos acima
  2. Banco Central  — CDI diário (série BCB #12)
  3. FRED           — Fed Funds Rate diário (série DFF)
  4. CVM local      — fundos BR (opcional, requer arquivos em data/raw/cvm/)

Período fixo: 2016-01-01 → 2026-04-30

Filtro de qualidade automático:
  Ativos com mais de 15% de dias sem dados são descartados para evitar
  que gaps contaminem os cálculos de correlação.
"""

import io
import os
import time
import warnings
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Execute: pip install yfinance pandas numpy requests")

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

# Período de coleta.
#
# DATA_FIM era fixa em "2026-04-30", o que congelou a base em abril e
# fazia o paper trading não encontrar sinal nenhum (ele procura o sinal de
# HOJE numa base que terminava meses atrás). Agora vai até a data corrente:
# para o backtest isso só acrescenta dados out-of-sample, e para a operação
# diária é o que permite o sistema enxergar o pregão de ontem.
#
# Para reproduzir um backtest antigo exatamente, passe fim="2026-04-30"
# explicitamente em coletar_tudo().
DATA_INICIO = "2016-01-01"
DATA_FIM    = datetime.today().strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════
# UNIVERSO DE ATIVOS
# ═══════════════════════════════════════════════════════════════

# ── Ações brasileiras (Ibovespa + SMLL + IDIV) ────────────────
_IBOVESPA = [
    "ABEV3","ALOS3","ALPA4","AMBP3","ANIM3","ARML3","ASAI3","AZUL4",
    "B3SA3","BBAS3","BBDC3","BBDC4","BBSE3","BEEF3","BPAC11","BRAP4",
    "BRFS3","BRKM5","CASH3","CCRO3","CIEL3","CMIG4","CMIN3","COGN3",
    "CPFE3","CPLE6","CRFB3","CSAN3","CVCB3","CXSE3","DXCO3","EGIE3",
    "ELET3","ELET6","EMBR3","ENEV3","ENGI11","EQTL3","EZTC3","FLRY3",
    "GGBR4","GOAU4","GOLL4","HAPV3","HYPE3","IGTI11","IRBR3","ITSA4",
    "ITUB4","JBSS3","JHSF3","KLBN11","LREN3","LWSA3","MGLU3","MRFG3",
    "MRVE3","MULT3","NTCO3","PCAR3","PETR3","PETR4","PETZ3","PRIO3",
    "RADL3","RAIL3","RAIZ4","RDOR3","RENT3","RRRP3","SANB11","SBSP3",
    "SLCE3","SMFT3","SMTO3","SOMA3","STBP3","SUZB3","TAEE11","TIMS3",
    "TOTS3","UGPA3","USIM5","VALE3","VBBR3","VIVT3","WEGE3","YDUQ3",
]
_SMLL_IDIV_EXTRAS = [
    "ALUP11","AURE3","CGAS5","CLSA3","CSMG3","DESK3","FESA4","GRND3",
    "KEPL3","LAVV3","LJQQ3","LOGG3","LPSB3","MDIA3","MILS3","ODPV3",
    "ONCO3","ORVR3","POMO4","PORT3","PTBL3","RECV3","ROMI3","SAPR11",
    "SEER3","SHUL4","TASA4","TGMA3","TUPY3","TTEN3","UNIP6","VLID3",
    "VSTE3","ZAMP3","CPLE3","TRPL4","ENBR3","CESP6","COCE5","EQPA3",
    "LIGT3","NEOE3",
]
_ibov_set = set(_IBOVESPA)
_extras_uniq = [t for t in _SMLL_IDIV_EXTRAS if t not in _ibov_set]
_ACOES_BR_RAW = _IBOVESPA + _extras_uniq

# ── Ações americanas (S&P500 top 200 + Nasdaq100 extras) ───────
_SP500_TOP200 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
    "LLY","XOM","V","UNH","MA","COST","HD","PG","JNJ","ABBV",
    "BAC","MRK","CVX","NFLX","CRM","WMT","AMD","TMO","KO","ORCL",
    "CSCO","PEP","ACN","MCD","ABT","GE","NOW","LIN","TXN","PM",
    "IBM","GS","MS","ISRG","AMGN","RTX","DIS","SPGI","BKNG","UBER",
    "AXP","CAT","BLK","SYK","GILD","T","ADI","VRTX","SCHW","TJX",
    "PFE","REGN","BSX","MDT","CI","ADP","C","MMC","MDLZ","ETN",
    "DE","ZTS","CME","PANW","HCA","BMY","MU","NKE","LRCX","CB",
    "SO","APH","SHW","ICE","PLD","WFC","MCO","ECL","ITW","ANET",
    "AON","WELL","USB","FI","CL","EOG","MCHP","TGT","NOC","MSI",
    "EMR","GD","CEG","PH","COF","PSA","WM","FCX","MET","AFL",
    "CARR","ORLY","OKE","AJG","ALL","DLR","TRV","HLT","SPG","CTAS",
    "HWM","NSC","MMM","D","BK","PAYX","SRE","PWR","TDG","FAST",
    "VRSK","ODFL","CMG","RSG","FICO","TEL","PCG","NEM","LHX","KMB",
    "EW","GWW","CTVA","DG","CDNS","FANG","MNST","WMB","AME","GIS",
    "PCAR","IDXX","CPRT","IQV","ACGL","OTIS","BIIB","RCL","CCI","PEG",
    "EXC","KEYS","KHC","DVN","GEHC","NDAQ","HIG","ROK","VMC","RMD",
    "VICI","TROW","EFX","LVS","CDW","PPG","IR","EBAY","HSY","MPWR",
    "AVB","DOW","GLW","NUE","EIX","VLO","ADSK","FTNT","DDOG","ZS",
]
_NASDAQ_EXTRAS = [
    "MELI","ASML","WDAY","TEAM","SNPS","CRWD","KLAC","NXPI","MRVL",
    "ON","ABNB","MRNA","LULU","ROST","ANSS","ALGN","DXCM","SWKS",
    "OKTA","ZM","DOCU","GRAB","PDD","JD","BIDU","NTES","NICE",
    "MTCH","DLTR","KDP","CINF","EXPD","ILMN","INTU","PYPL","CHTR",
    "SBUX","TMUS","SGEN","WBA","SPLK","TTWO","QRVO","LCID",
]
_sp500_set = set(_SP500_TOP200)
_NASDAQ_UNIQ = [t for t in _NASDAQ_EXTRAS if t not in _sp500_set]
_ACOES_US = _SP500_TOP200 + _NASDAQ_UNIQ

# ── ADRs internacionais (NYSE) ─────────────────────────────────
_ACOES_INTL = [
    "NVO","SHEL","AZN","UL","BP","DEO","SAP",      # Europa
    "BABA","JD","PDD","BIDU","NTES","TM",           # Asia
    "HDB","INFY","WIT",                             # India
    "BHP","RIO","SNY",                              # Mineração / Pharma
    "PBR","ERJ","GGB","BBD","ITUB",                 # Brasil ADRs
]

# ── Commodities ────────────────────────────────────────────────
_COMMODITIES = {
    "GC=F":"Ouro","SI=F":"Prata","PL=F":"Platina","PA=F":"Paládio",
    "HG=F":"Cobre","ALI=F":"Alumínio",
    "CL=F":"Petróleo WTI","BZ=F":"Petróleo Brent",
    "NG=F":"Gás Natural","RB=F":"Gasolina","HO=F":"Óleo Diesel",
    "ZS=F":"Soja","ZC=F":"Milho","ZW=F":"Trigo","ZO=F":"Aveia",
    "KC=F":"Café Arábica","CC=F":"Cacau",
    "SB=F":"Açúcar","CT=F":"Algodão","OJ=F":"Suco de Laranja",
    "GF=F":"Boi Gordo","LE=F":"Gado Vivo","HE=F":"Suíno",
    "PICK":"Mineração Global ETF",
}

# ── ETFs temáticos ─────────────────────────────────────────────
_ETFS = {
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF",
    "IWM":"Russell 2000","EFA":"Desenvolvidos ex-US","EEM":"Emergentes",
    "XLE":"Energia US","XLF":"Financeiro US","XLK":"Tecnologia US",
    "XLV":"Saúde US","XLI":"Industrial US","XLB":"Materiais US",
    "XLRE":"Real Estate US",
    "TLT":"Treasuries 20yr","IEF":"Treasuries 7-10yr",
    "HYG":"High Yield","LQD":"Grau Investimento",
    "VNQ":"REIT ETF",
    "GLD":"Ouro ETF","SLV":"Prata ETF",
    "GDX":"Mineradoras Ouro","GDXJ":"Mineradoras Jr.",
    "USO":"Petróleo ETF","UNG":"Gás Natural ETF",
    "ARKK":"ARK Innovation",
    "BOVA11.SA":"BOVA11","IVVB11.SA":"IVVB11","SMAL11.SA":"SMAL11",
    "IMAB11.SA":"IMAB11",
}

# ── Cripto ─────────────────────────────────────────────────────
_CRIPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum",
    "SOL-USD":"Solana","BNB-USD":"BNB","XRP-USD":"XRP",
    "ADA-USD":"Cardano","DOGE-USD":"Dogecoin","AVAX-USD":"Avalanche",
    "COIN":"Coinbase","MSTR":"MicroStrategy",
}

# ── Benchmarks e câmbio ────────────────────────────────────────
_BENCHMARKS = {
    "^BVSP":"Ibovespa","^GSPC":"S&P 500","^NDX":"Nasdaq 100",
    "^DJI":"Dow Jones","^RUT":"Russell 2000",
    "^FTSE":"FTSE 100","^N225":"Nikkei 225",
    "^HSI":"Hang Seng","^STOXX50E":"Euro Stoxx 50",
    "BRL=X":"USD/BRL","EUR=X":"USD/EUR",
    "JPY=X":"USD/JPY","CNY=X":"USD/CNY",
    "^VIX":"VIX","DX-Y.NYB":"Índice Dólar",
    "IRFM11.SA":"IRFM11 (prefixado BR)",
}

# ── FIIs ───────────────────────────────────────────────────────
_FIIS = {
    "HGLG11.SA":"CSHG Logística","MXRF11.SA":"Maxi Renda",
    "KNRI11.SA":"Kinea Renda","XPML11.SA":"XP Malls",
    "BTLG11.SA":"BTG Logística","VISC11.SA":"Vinci Shoppings",
}

# ── Mapa nome completo (para resumo) ──────────────────────────
NOMES = {
    **_COMMODITIES, **_ETFS, **_CRIPTO, **_BENCHMARKS, **_FIIS,
    **{f"{t}.SA": t for t in _ACOES_BR_RAW},
    **{t: t for t in _ACOES_US + _ACOES_INTL},
}

# Fundos CVM — identificados por CNPJ
FUNDOS_CVM = {
    "14.213.077/0001-54": "Guepardo Institucional Master FIA",
    "12.055.107/0001-16": "Alaska Black Master FIF Ações",
    "35.916.390/0001-29": "Constellation Compounders Master FIA",
}

def _todos_tickers_yahoo() -> list:
    """Retorna lista deduplicada de todos os tickers do Yahoo Finance."""
    tickers = (
        [f"{t}.SA" for t in _ACOES_BR_RAW]
        + _ACOES_US
        + _ACOES_INTL
        + list(_COMMODITIES.keys())
        + list(_ETFS.keys())
        + list(_CRIPTO.keys())
        + list(_BENCHMARKS.keys())
        + list(_FIIS.keys())
    )
    seen, uniq = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ═══════════════════════════════════════════════════════════════
# COLLECTOR
# ═══════════════════════════════════════════════════════════════

class Collector:
    """
    Coleta, normaliza e salva dados históricos de todos os ativos.

    Uso rápido:
        c = Collector()
        dados = c.coletar_tudo()
    """

    def __init__(
        self,
        pasta_saida: str = PASTA_DADOS,
        max_missing_pct: float = 0.15,
    ):
        self.pasta_saida = pasta_saida
        self.max_missing = max_missing_pct
        os.makedirs(pasta_saida, exist_ok=True)

    # ─────────────────────────────────────────
    # 1. YAHOO FINANCE
    # ─────────────────────────────────────────

    def baixar_yahoo(
        self,
        tickers: list,
        inicio: str,
        fim: str,
        intervalo: str = "1d",
        pausa: float = 0.3,
    ) -> pd.DataFrame:
        """
        Baixa preços ajustados de fechamento via Yahoo Finance.
        Usa yf.Ticker().history() — método que funcionou na v2.
        Aplica filtro de qualidade: descarta ativos com >max_missing_pct gaps.
        """
        n = len(tickers)
        print(f"\n📥 Yahoo Finance — {n} tickers")
        print(f"   {inicio} → {fim} | intervalo: {intervalo}")

        resultados = {}
        falhas = []

        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"   [{i:03d}/{n}] {ticker}...", end=" ", flush=True)
                t = yf.Ticker(ticker)
                df = t.history(
                    start=inicio,
                    end=fim,
                    interval=intervalo,
                    auto_adjust=True,
                )
                if df.empty:
                    print("⚠️  sem dados")
                    falhas.append(ticker)
                    continue

                serie = df["Close"].squeeze()
                serie.index = serie.index.tz_localize(None)
                resultados[ticker] = serie
                print(f"✅ {len(serie)}")

            except Exception as e:
                print(f"❌ {e}")
                falhas.append(ticker)

            time.sleep(pausa)

        if falhas:
            print(f"\n   ⚠️  Sem dados ({len(falhas)}): "
                  f"{falhas[:10]}{'...' if len(falhas) > 10 else ''}")

        if not resultados:
            print("   ❌ Nenhum dado obtido")
            return pd.DataFrame()

        # Filtro de qualidade ANTES de montar o DataFrame.
        # Cada série é avaliada individualmente: missing = NaN dentro
        # do período em que o ativo deveria ter dados (primeiro ao último
        # dia com dado). Isso evita que cripto (365 dias) infle o denominador
        # e descarte ações (apenas dias úteis).
        antes = len(resultados)
        aprovados_dict = {}
        descartados_lista = []
        descartados_salto = []
        for ticker, serie in resultados.items():
            if len(serie) == 0:
                descartados_lista.append(ticker)
                continue
            # Conta NaN apenas entre primeiro e último registro próprio
            primeiro = serie.first_valid_index()
            ultimo   = serie.last_valid_index()
            if primeiro is None or ultimo is None:
                descartados_lista.append(ticker)
                continue
            janela = serie.loc[primeiro:ultimo]
            missing_pct = janela.isna().sum() / len(janela)
            if missing_pct > self.max_missing:
                descartados_lista.append(ticker)
                continue

            # Filtro de salto impossível de preço (adicionado 2026-09-02).
            #
            # O yfinance às vezes devolve a série com ajuste de proventos
            # aplicado de forma inconsistente: parte do histórico numa
            # escala e o resto em outra. Foi o caso do XPML11.SA, que
            # saltou de R$ 0,14 para R$ 106,86 de um pregão para o outro
            # (fator 757x) — um retorno diário de +75.845%, que passou por
            # todo o pipeline e inflou o backtest do portfólio inteiro.
            #
            # Nenhum ativo real, nem cripto, multiplica de valor por 20x
            # num único pregão. Quando isso aparece, a série está furada e
            # o ativo inteiro é descartado: não dá para saber qual metade
            # do histórico está na escala certa.
            # Só compara pregões consecutivos em que AMBOS os preços são
            # positivos. Futuros podem ter preço negativo de verdade — o
            # WTI (CL=F) fechou negativo em 20/04/2020 — e a razão entre
            # preços perde sentido nesses casos; tratá-la como salto
            # descartaria um ativo com dado legítimo.
            s_val = janela.dropna()
            s_pos = s_val[s_val > 0]
            if len(s_pos) > 1:
                fator = (s_pos / s_pos.shift(1)).dropna()
                if len(fator) and (fator.max() > 20 or fator.min() < 1 / 20):
                    descartados_salto.append(ticker)
                    continue

            aprovados_dict[ticker] = serie

        descartados = antes - len(aprovados_dict)
        if descartados_salto:
            print(f"\n   ⚠️  {len(descartados_salto)} ativo(s) descartado(s) por salto "
                  f"impossível de preço (dado corrompido na fonte):")
            for t in descartados_salto:
                print(f"        {t}")

        df_out = pd.DataFrame(aprovados_dict)
        df_out.index = pd.to_datetime(df_out.index)
        df_out.index.name = "data"
        df_out = df_out.sort_index()

        print(f"\n   ✅ {df_out.shape[1]} ativos aprovados | {len(df_out)} dias no índice")
        if descartados:
            print(f"   🗑  {descartados} descartados (>{self.max_missing:.0%} missing no próprio calendário)")

        return df_out

    # ─────────────────────────────────────────
    # 2. CDI — BANCO CENTRAL
    # ─────────────────────────────────────────

    def baixar_cdi(self, inicio: str, fim: str) -> pd.DataFrame:
        """
        Baixa a taxa DI diária do BCB (série #12) e converte em índice
        acumulado — permite calcular retornos diários do CDI exatamente
        como se fosse um ativo financeiro.

        Faz o download em chunks anuais para contornar o limite de
        registros da API do BCB (~10 anos por request).
        """
        print("\n📥 CDI — Banco Central do Brasil...")

        inicio_dt = datetime.strptime(inicio, "%Y-%m-%d")
        fim_dt    = datetime.strptime(fim,    "%Y-%m-%d")

        # Divide em janelas anuais
        chunks = []
        ano_atual = inicio_dt
        while ano_atual <= fim_dt:
            ano_fim = min(datetime(ano_atual.year, 12, 31), fim_dt)
            chunks.append((ano_atual, ano_fim))
            ano_atual = datetime(ano_atual.year + 1, 1, 1)

        frames = []
        for chunk_ini, chunk_fim in chunks:
            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
                f"?formato=json"
                f"&dataInicial={_fmt_bcb(chunk_ini.strftime('%Y-%m-%d'))}"
                f"&dataFinal={_fmt_bcb(chunk_fim.strftime('%Y-%m-%d'))}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dados = resp.json()
                if dados:
                    frames.append(pd.DataFrame(dados))
                    print(f"   ✅ {chunk_ini.year}: {len(dados)} dias")
            except Exception as e:
                print(f"   ⚠️  {chunk_ini.year}: erro — {e}")

        if not frames:
            print("   ❌ Nenhum dado CDI obtido")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df = df.set_index("data").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["taxa_dia_%"] = df["valor"].astype(float)
        df["CDI"] = 100 * (1 + df["taxa_dia_%"] / 100).cumprod()

        acum = df["CDI"].iloc[-1] / df["CDI"].iloc[0] * 100 - 100
        print(f"   ✅ Total: {len(df)} dias úteis | CDI acumulado: {acum:.1f}%")
        return df[["taxa_dia_%", "CDI"]]

    # ─────────────────────────────────────────
    # 3. FED FUNDS RATE — FRED
    # ─────────────────────────────────────────

    def baixar_fedfunds(self, inicio: str, fim: str) -> pd.DataFrame:
        """
        Baixa a Fed Funds Rate diária do FRED (série DFF) e converte em
        índice acumulado — equivalente americano do CDI.
        A DFF é publicada em % ao ano (base 360).
        """
        print("\n📥 Fed Funds Rate — FRED (Federal Reserve)...")

        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id=DFF&vintage_date={fim}"
        )

        try:
            df = pd.read_csv(url, parse_dates=["observation_date"],
                             index_col="observation_date")
            df.index.name = "data"
            df.columns = ["taxa_ano_%"]
            df = df[df["taxa_ano_%"] != "."].copy()
            df["taxa_ano_%"] = df["taxa_ano_%"].astype(float)
            df = df[df.index >= inicio].copy()

            # Converte % ao ano (base 360) para % ao dia
            df["taxa_dia_%"] = df["taxa_ano_%"] / 360

            # Preenche fins de semana com última taxa disponível
            idx_completo = pd.date_range(df.index.min(), df.index.max(), freq="D")
            df = df.reindex(idx_completo).ffill()
            df.index.name = "data"
            df = df[df.index.dayofweek < 5]  # só dias úteis

            df["FedFunds"] = 100 * (1 + df["taxa_dia_%"] / 100).cumprod()

            acum = df["FedFunds"].iloc[-1] / df["FedFunds"].iloc[0] * 100 - 100
            print(f"   ✅ {len(df)} dias | Fed Funds acumulado: {acum:.1f}%")
            return df[["taxa_ano_%", "taxa_dia_%", "FedFunds"]]

        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────
    # 4. FUNDOS CVM (opcional)
    # ─────────────────────────────────────────

    def baixar_fundos_cvm(self, inicio: str, fim: str) -> pd.DataFrame:
        """
        Lê cotas diárias de fundos brasileiros a partir de arquivos locais
        baixados manualmente do portal da CVM.

        ESTRUTURA ESPERADA em data/raw/cvm/:
            inf_diario_fi_2021.zip   ← arquivos anuais (HIST)
            inf_diario_fi_202501.csv ← meses recentes

        Se a pasta não existir, retorna DataFrame vazio sem erro.
        """
        pasta_cvm = os.path.join(self.pasta_saida, "cvm")
        if not os.path.exists(pasta_cvm):
            print("\n⚠️  Pasta CVM não encontrada — fundos ignorados")
            return pd.DataFrame()

        print("\n📥 Fundos brasileiros — lendo arquivos locais CVM...")

        cnpjs = {_limpar_cnpj(c): n for c, n in FUNDOS_CVM.items()}
        arquivos = sorted(os.listdir(pasta_cvm))
        frames = []

        for arquivo in arquivos:
            nome_lower = arquivo.lower()
            if "inf_diario_fi" not in nome_lower:
                continue
            if not (nome_lower.endswith(".csv") or nome_lower.endswith(".zip")):
                continue

            caminho = os.path.join(pasta_cvm, arquivo)
            print(f"   {arquivo}...", end=" ", flush=True)

            try:
                if nome_lower.endswith(".zip"):
                    with zipfile.ZipFile(caminho, "r") as z:
                        csvs_dentro = [n for n in z.namelist()
                                       if n.lower().endswith(".csv")]
                        dfs_zip = []
                        for csv_interno in csvs_dentro:
                            with z.open(csv_interno) as f:
                                df_chunk = pd.read_csv(
                                    io.TextIOWrapper(f, encoding="latin-1"),
                                    sep=";", dtype=str, on_bad_lines="skip",
                                )
                                col_cnpj = (
                                    "CNPJ_FUNDO_CLASSE"
                                    if "CNPJ_FUNDO_CLASSE" in df_chunk.columns
                                    else "CNPJ_FUNDO"
                                )
                                df_chunk["_cnpj"] = df_chunk[col_cnpj].str.replace(
                                    r"[.\/\-]", "", regex=True
                                )
                                df_chunk = df_chunk[
                                    df_chunk["_cnpj"].isin(cnpjs.keys())
                                ].copy()
                                if not df_chunk.empty:
                                    dfs_zip.append(df_chunk)
                        df_arq = pd.concat(dfs_zip) if dfs_zip else pd.DataFrame()
                else:
                    df_arq = pd.read_csv(
                        caminho, sep=";", encoding="latin-1",
                        dtype=str, on_bad_lines="skip",
                    )
                    col_cnpj = (
                        "CNPJ_FUNDO_CLASSE"
                        if "CNPJ_FUNDO_CLASSE" in df_arq.columns
                        else "CNPJ_FUNDO"
                    )
                    df_arq["_cnpj"] = df_arq[col_cnpj].str.replace(
                        r"[.\/\-]", "", regex=True
                    )
                    df_arq = df_arq[df_arq["_cnpj"].isin(cnpjs.keys())].copy()

                if df_arq.empty:
                    print("(fundos não encontrados)")
                    continue

                df_arq["DT_COMPTC"] = pd.to_datetime(
                    df_arq["DT_COMPTC"], format="%Y-%m-%d", errors="coerce"
                )
                df_arq["VL_QUOTA"] = pd.to_numeric(
                    df_arq["VL_QUOTA"].str.replace(",", "."), errors="coerce"
                )
                df_arq["nome"] = df_arq["_cnpj"].map(cnpjs)
                frames.append(
                    df_arq[["DT_COMPTC", "nome", "VL_QUOTA"]].dropna()
                )
                print(f"✅ {len(df_arq)} registros")

            except Exception as e:
                print(f"⚠️  {e}")

        if not frames:
            print("   ❌ Nenhum dado nos arquivos locais")
            return pd.DataFrame()

        df_total = pd.concat(frames, ignore_index=True)
        df_total = df_total[
            (df_total["DT_COMPTC"] >= inicio) &
            (df_total["DT_COMPTC"] <= fim)
        ]
        df_pivot = df_total.pivot_table(
            index="DT_COMPTC", columns="nome",
            values="VL_QUOTA", aggfunc="last",
        )
        df_pivot.index.name = "data"
        df_pivot = df_pivot.sort_index()

        print(f"\n   ✅ {df_pivot.shape[1]} fundos | {df_pivot.shape[0]} dias úteis")
        return df_pivot

    # ─────────────────────────────────────────
    # COLETA COMPLETA
    # ─────────────────────────────────────────

    def coletar_tudo(
        self,
        inicio: str = DATA_INICIO,
        fim: str = DATA_FIM,
        intervalo: str = "1d",
        salvar: bool = True,
        tickers: list = None,
        pular_fundos: bool = False,
    ) -> dict:
        """
        Executa a coleta completa de todas as fontes.

        Args:
            tickers: se informado, substitui o universo completo (~487
                ativos) por essa lista — usado pelo modo teste.
            pular_fundos: se True, não baixa os fundos CVM (os zips locais
                são pesados de ler; irrelevante para validar o pipeline).

        Retorna dict com DataFrames:
            precos           — preços de fechamento (Yahoo)
            retornos         — retornos diários % (Yahoo)
            cdi              — índice CDI acumulado + taxa diária
            fedfunds         — índice Fed Funds acumulado + taxa
            fundos           — cotas diárias dos fundos CVM
            retornos_fundos  — retornos diários % dos fundos
        """
        print("=" * 65)
        print(f"COLETA COMPLETA v3  |  {inicio} → {fim}")
        print("=" * 65)

        todos_tickers = tickers if tickers is not None else _todos_tickers_yahoo()
        print(f"\nUniverso: {len(todos_tickers)} tickers únicos")

        df_precos   = self.baixar_yahoo(todos_tickers, inicio, fim, intervalo)
        df_retornos = df_precos.pct_change() * 100
        df_retornos = df_retornos.iloc[1:]

        df_cdi = self.baixar_cdi(inicio, fim)
        df_ff  = self.baixar_fedfunds(inicio, fim)

        if pular_fundos:
            df_fundos     = pd.DataFrame()
            df_ret_fundos = pd.DataFrame()
        else:
            df_fundos     = self.baixar_fundos_cvm(inicio, fim)
            df_ret_fundos = (
                df_fundos.pct_change() * 100
                if not df_fundos.empty else pd.DataFrame()
            )

        resultado = {
            "precos":          df_precos,
            "retornos":        df_retornos,
            "cdi":             df_cdi,
            "fedfunds":        df_ff,
            "fundos":          df_fundos,
            "retornos_fundos": df_ret_fundos,
        }

        if salvar:
            self._salvar(resultado)

        self._resumo_final(resultado, len(todos_tickers))
        return resultado

    # ─────────────────────────────────────────
    # UTILITÁRIOS
    # ─────────────────────────────────────────

    def carregar(self, nome: str) -> pd.DataFrame:
        """Carrega CSV previamente salvo. Ex: c.carregar('retornos')"""
        caminho = os.path.join(self.pasta_saida, f"{nome}.csv")
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"'{caminho}' não encontrado. Rode coletar_tudo() primeiro."
            )
        df = pd.read_csv(caminho, index_col="data", parse_dates=True)
        print(f"📂 {nome}.csv — {df.shape[0]}L × {df.shape[1]}C")
        return df

    def resumo(self, df_precos: pd.DataFrame) -> pd.DataFrame:
        """Tabela resumo com retorno total, volatilidade e % missing."""
        ret = df_precos.pct_change().iloc[1:]
        rows = []
        for col in df_precos.columns:
            s = df_precos[col].dropna()
            if len(s) < 2:
                continue
            rows.append({
                "ativo":            col,
                "nome":             NOMES.get(col, col),
                "inicio":           s.index[0].date(),
                "fim":              s.index[-1].date(),
                "dias":             len(s),
                "missing_%":        round(df_precos[col].isna().mean() * 100, 1),
                "retorno_total_%":  round((s.iloc[-1] / s.iloc[0] - 1) * 100, 1),
                "vol_anual_%":      round(ret[col].std() * np.sqrt(252) * 100, 1),
            })
        return pd.DataFrame(rows).sort_values("vol_anual_%", ascending=False)

    def _salvar(self, dados: dict):
        print(f"\n💾 Salvando em: {self.pasta_saida}")
        for nome, df in dados.items():
            if df is not None and not df.empty:
                caminho = os.path.join(self.pasta_saida, f"{nome}.csv")
                df.to_csv(caminho)
                print(f"   ✅ {nome}.csv  ({df.shape[0]}L × {df.shape[1]}C)")

    def _resumo_final(self, dados: dict, n_tentados: int):
        print("\n" + "=" * 65)
        print("RESUMO FINAL")
        print("=" * 65)
        p = dados["precos"]
        aprovados   = p.shape[1]
        descartados = n_tentados - aprovados
        print(f"  Tentados     : {n_tentados}")
        print(f"  Aprovados    : {aprovados} ({aprovados/n_tentados:.0%})")
        print(f"  Descartados  : {descartados} (sem dados ou >{self.max_missing:.0%} missing)")
        print(f"  Dias coletados: {p.shape[0]}")

        if not dados["cdi"].empty:
            cdi = dados["cdi"]
            acum = cdi["CDI"].iloc[-1] / cdi["CDI"].iloc[0] * 100 - 100
            print(f"  CDI acumulado : {acum:.1f}%")

        if not dados["fedfunds"].empty:
            ff = dados["fedfunds"]
            acum = ff["FedFunds"].iloc[-1] / ff["FedFunds"].iloc[0] * 100 - 100
            print(f"  Fed Funds ac. : {acum:.1f}%")

        if not dados["fundos"].empty:
            f = dados["fundos"]
            print(f"  Fundos CVM    : {f.shape[1]} fundos | {f.shape[0]} dias")
            for col in f.columns:
                s = f[col].dropna()
                if len(s) > 1:
                    acum = (s.iloc[-1] / s.iloc[0] - 1) * 100
                    print(f"    {col}: {acum:.1f}%")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _fmt_bcb(data_iso: str) -> str:
    """'YYYY-MM-DD' → 'DD/MM/YYYY'"""
    return datetime.strptime(data_iso, "%Y-%m-%d").strftime("%d/%m/%Y")

def _limpar_cnpj(cnpj: str) -> str:
    """'05.159.635/0001-00' → '05159635000100'"""
    return cnpj.replace(".", "").replace("/", "").replace("-", "")


# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modo_teste import adicionar_flag, ATIVOS_TESTE, DATA_INICIO_TESTE, banner

    parser = argparse.ArgumentParser(description="Coleta dados históricos (Yahoo, CDI, Fed Funds, CVM)")
    adicionar_flag(parser)
    args = parser.parse_args()
    banner("collector.py", args.teste)

    if args.teste:
        # Universo pequeno, período curto, sem CVM (zips pesados) e salva
        # em pasta separada — nunca sobrescreve os dados completos.
        c = Collector(
            pasta_saida=os.path.join(PASTA_DADOS, "teste"),
            max_missing_pct=0.15,
        )
        dados = c.coletar_tudo(
            inicio=DATA_INICIO_TESTE,
            tickers=ATIVOS_TESTE,
            pular_fundos=True,
            salvar=True,
        )
    else:
        c = Collector(max_missing_pct=0.15)
        dados = c.coletar_tudo(salvar=True)

    print("\n" + "=" * 65)
    print("RESUMO DOS ATIVOS (ordenado por volatilidade)")
    print("=" * 65)
    print(c.resumo(dados["precos"]).to_string(index=False))

    if not dados["fundos"].empty:
        print("\n" + "=" * 65)
        print("COTAS DOS FUNDOS (últimas 5 datas)")
        print("=" * 65)
        print(dados["fundos"].tail())