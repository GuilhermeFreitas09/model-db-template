from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "simple_quant.db"

# Nome atual do dataset salvo no SQLite.
DEFAULT_DATASET_NAME = "base_mineracao"

# Data inicial atual usada para atualizar a base.
DEFAULT_START_DATE = "2025-01-01"

# Universo atual de acoes para gerar base e depois treinar/escorar.
# Valores antigos:
# ["VALE3.SA", "CMIN3.SA", "CSNA3.SA", "GGBR4.SA"]
#
# Valores novos:
# - mantem mineracao/siderurgia
# - adiciona holdings e pares ligados a commodities/exportacao
# - adiciona petroleo/papel e celulose para ampliar o universo ciclico
TICKERS_MINING = [
    "VALE3.SA",
    "CMIN3.SA",
    "CSNA3.SA",
    "GGBR4.SA",
    "USIM5.SA",
    "GOAU4.SA",
    "BRAP4.SA",
    "PETR4.SA",
    "PRIO3.SA",
    "SUZB3.SA",
    "KLBN11.SA",
]

# Fatores de mercado vindos do Yahoo Finance.
# Valores antigos:
# ["^BVSP", "USDBRL=X", "TIO=F"]
#
# Valores novos:
# - ibovespa e dolar continuam como contexto local
# - minerio de ferro continua central para VALE/CSN/CMIN
# - adiciona S&P 500 e VIX para risco global
# - adiciona DXY para dolar global
# - adiciona cobre, ouro e petroleo para sensibilidade a commodities
YFINANCE_FEATURES = {
    "^BVSP": "ibovespa",
    "USDBRL=X": "dolar",
    "TIO=F": "minerio_ferro",
    "^GSPC": "sp500",
    "^VIX": "vix",
    "DX-Y.NYB": "dxy",
    "HG=F": "cobre",
    "GC=F": "ouro",
    "CL=F": "petroleo_wti",
}

# Series do BCB usadas na base.
# Valores atuais:
# - 11: selic diaria
# - 433: IPCA mensal
BCB_FEATURES = {
    "selic_dia": 11,
    "ipca_mensal": 433,
}

# Prefixo das colunas de preco dos papeis no dataset final.
# Exemplo: preco_vale3_sa, preco_petr4_sa.
PRICE_COLUMN_PREFIX = "preco_"

# Colunas macro atuais usadas na modelagem.
# Antes:
# ["ibovespa", "dolar", "minerio_ferro", "selic_dia", "ipca_mensal"]
#
# Agora:
# inclui os novos fatores do Yahoo e mantem os fatores do BCB.
# se eu quiser adicionar dados, posso colocar o nome das colunas aqui após ajustar o data mining para buscar os dados e salvar no banco.
# ajustar data mining -> ajustar config -> ajustar banco (para comportar novas colunas) -> ajustar features se quiser fazer um processamento 
## diferente para as novas colunas. Se não quiser, todas as colunas adicionais serão processadas automaticamente pelas funções de variação e média.
DEFAULT_MACRO_COLS = [
    "ibovespa",
    "dolar",
    "minerio_ferro",
    "sp500",
    "vix",
    "dxy",
    "cobre",
    "ouro",
    "petroleo_wti",
    "selic_dia",
    "ipca_mensal",
]

# Janelas atuais para targets e variacoes.
DEFAULT_WINDOWS = [7, 15, 30, 60]

# Tolerancia para considerar que o preço futuro é maior do que o preço atual, para evitar ruído.
PRICE_INCREASE_THRESHOLD = 0.005

def safe_ticker_slug(ticker: str) -> str:
    return (
        ticker.lower()
        .replace(".", "_")
        .replace("-", "_")
        .replace("=", "_")
        .replace("^", "")
    )


def price_column_name(ticker: str) -> str:
    return f"{PRICE_COLUMN_PREFIX}{safe_ticker_slug(ticker)}"


def moving_average_column_name(ticker: str, window: int) -> str:
    return f"ma{window}_{safe_ticker_slug(ticker)}"
