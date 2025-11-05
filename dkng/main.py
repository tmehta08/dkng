import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from loguru import logger
import sys

# -----------------------------------------------------
# 1. Setup logging
# -----------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("analysis.log", rotation="10 MB")

# -----------------------------------------------------
# 2. CLI Argument: choose ticker
# -----------------------------------------------------
if len(sys.argv) > 1:
    ticker = sys.argv[1].upper()
else:
    ticker = "DKNG"

logger.info(f"Starting {ticker} stock analysis")

end_date = datetime.today()
start_date = end_date - timedelta(days=730)

# -----------------------------------------------------
# 3. Fetch market data
# -----------------------------------------------------
logger.info(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")
dkng = yf.download(ticker, start=start_date, end=end_date, progress=False)
spy = yf.download("SPY", start=start_date, end=end_date, progress=False)

# -----------------------------------------------------
# 4. Flatten yfinance MultiIndex columns
# -----------------------------------------------------
def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    close_cols = [c for c in df.columns if "close" in c.lower()]
    if close_cols:
        df.rename(columns={close_cols[0]: "Close"}, inplace=True)
    return df

dkng = flatten_yf_columns(dkng)
spy = flatten_yf_columns(spy)

if "Close" not in dkng.columns:
    raise KeyError(f"'Close' column not found in {ticker} data. Columns: {dkng.columns.tolist()}")

# -----------------------------------------------------
# 5. Compute daily returns and indicators
# -----------------------------------------------------
dkng["Daily_Return"] = dkng["Close"].pct_change()
spy["Daily_Return"] = spy["Close"].pct_change()

# RSI Calculation
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

dkng["RSI"] = compute_rsi(dkng["Close"])

# Moving Averages
dkng["SMA_20"] = dkng["Close"].rolling(window=20).mean()
dkng["SMA_50"] = dkng["Close"].rolling(window=50).mean()
dkng["SMA_200"] = dkng["Close"].rolling(window=200).mean()

# MACD
exp1 = dkng["Close"].ewm(span=12, adjust=False).mean()
exp2 = dkng["Close"].ewm(span=26, adjust=False).mean()
dkng["MACD"] = exp1 - exp2
dkng["Signal_Line"] = dkng["MACD"].ewm(span=9, adjust=False).mean()

# Volatility & Sharpe
dkng["Volatility"] = dkng["Daily_Return"].rolling(window=20).std() * np.sqrt(252)
sharpe_ratio = (
    dkng["Daily_Return"].mean() / dkng["Daily_Return"].std() * np.sqrt(252)
    if dkng["Daily_Return"].std() != 0 else np.nan
)

# Drawdown
dkng["Cumulative"] = (1 + dkng["Daily_Return"]).cumprod()
dkng["Cum_Max"] = dkng["Cumulative"].cummax()
dkng["Drawdown"] = dkng["Cumulative"] / dkng["Cum_Max"] - 1
max_drawdown = dkng["Drawdown"].min()

# -----------------------------------------------------
# 6. Fundamentals
# -----------------------------------------------------
logger.info("Fetching fundamental 10-K data...")
try:
    info = yf.Ticker(ticker).info
    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")
    eps = info.get("trailingEps")
    sector = info.get("sector")
except Exception as e:
    logger.warning(f"Fundamentals unavailable: {e}")
    market_cap = pe_ratio = eps = sector = None

# -----------------------------------------------------
# 7. Performance vs SPY
# -----------------------------------------------------
relative_performance = (
    (dkng["Cumulative"].iloc[-1] /
     spy["Close"].pct_change().add(1).cumprod().iloc[-1]) - 1
)

# -----------------------------------------------------
# 8. Print summary
# -----------------------------------------------------
logger.info("=" * 60)
logger.info(f"STOCK ANALYSIS - {ticker}")
logger.info("=" * 60)

logger.info(f"Current Price: ${dkng['Close'].iloc[-1]:.2f}")
logger.info(f"52-Week High: ${dkng['Close'].tail(252).max():.2f}")
logger.info(f"52-Week Low:  ${dkng['Close'].tail(252).min():.2f}")
logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
logger.info(f"Max Drawdown: {max_drawdown:.2%}")
logger.info(f"Relative Performance vs SPY: {relative_performance:.2%}")

if market_cap:
    logger.info(f"Market Cap: ${market_cap:,.0f}")
if pe_ratio:
    logger.info(f"P/E Ratio: {pe_ratio:.2f}")
if eps:
    logger.info(f"EPS: {eps:.2f}")
if sector:
    logger.info(f"Sector: {sector}")

# -----------------------------------------------------
# 9. Visualization
# -----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(dkng.index, dkng["Close"], label="Close Price", linewidth=1.8)
plt.plot(dkng.index, dkng["SMA_50"], label="50-Day SMA", linestyle="--")
plt.plot(dkng.index, dkng["SMA_200"], label="200-Day SMA", linestyle="--")
plt.title(f"{ticker} Price with Moving Averages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

logger.info("Analysis complete.")
