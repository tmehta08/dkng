import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download DKNG data (2 years)
ticker = "DKNG"
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

print(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")
dkng = yf.download(ticker, start=start_date, end=end_date, progress=False)
spy = yf.download("SPY", start=start_date, end=end_date, progress=False)

# Calculate returns
dkng["Daily_Return"] = dkng["Close"].pct_change()
spy["Daily_Return"] = spy["Close"].pct_change()

# Calculate moving averages
dkng["MA_20"] = dkng["Close"].rolling(window=20).mean()
dkng["MA_50"] = dkng["Close"].rolling(window=50).mean()
dkng["MA_200"] = dkng["Close"].rolling(window=200).mean()

# Calculate volatility (annualized)
dkng["Volatility"] = dkng["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

dkng["RSI"] = calculate_rsi(dkng["Close"])

# Calculate MACD
exp1 = dkng["Close"].ewm(span=12, adjust=False).mean()
exp2 = dkng["Close"].ewm(span=26, adjust=False).mean()
dkng["MACD"] = exp1 - exp2
dkng["Signal_Line"] = dkng["MACD"].ewm(span=9, adjust=False).mean()
dkng["MACD_Histogram"] = dkng["MACD"] - dkng["Signal_Line"]

# Performance metrics
total_return_dkng = (dkng["Close"].iloc[-1] / dkng["Close"].iloc[0] - 1) * 100
total_return_spy = (spy["Close"].iloc[-1] / spy["Close"].iloc[0] - 1) * 100
annual_volatility = dkng["Daily_Return"].std() * np.sqrt(252) * 100
sharpe_ratio = (dkng["Daily_Return"].mean() * 252) / (dkng["Daily_Return"].std() * np.sqrt(252))
max_drawdown = (dkng["Close"] / dkng["Close"].cummax() - 1).min() * 100

print("\n" + "="*60)
print(f"DKNG STOCK ANALYSIS - {ticker}")
print("="*60)
print(f"\nCurrent Price: ${dkng['Close'].iloc[-1]:.2f}")
print(f"52-Week High: ${dkng['Close'].tail(252).max():.2f}")
print(f"52-Week Low: ${dkng['Close'].tail(252).min():.2f}")
print(f"\nPerformance (2 Years):")
print(f"  DKNG Return: {total_return_dkng:.2f}%")
print(f"  SPY Return: {total_return_spy:.2f}%")
print(f"  Outperformance: {total_return_dkng - total_return_spy:.2f}%")
print(f"\nRisk Metrics:")
print(f"  Annual Volatility: {annual_volatility:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"  Max Drawdown: {max_drawdown:.2f}%")
print(f"\nTechnical Indicators (Latest):")
print(f"  RSI (14): {dkng['RSI'].iloc[-1]:.2f}")
print(f"  MACD: {dkng['MACD'].iloc[-1]:.4f}")
print(f"  Signal Line: {dkng['Signal_Line'].iloc[-1]:.4f}")
print(f"  Close vs MA20: {((dkng['Close'].iloc[-1] / dkng['MA_20'].iloc[-1] - 1) * 100):.2f}%")
print(f"  Close vs MA200: {((dkng['Close'].iloc[-1] / dkng['MA_200'].iloc[-1] - 1) * 100):.2f}%")

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("DKNG Stock Analysis", fontsize=16, fontweight="bold")

# Price and Moving Averages
axes[0, 0].plot(dkng.index, dkng["Close"], label="Close Price", linewidth=2)
axes[0, 0].plot(dkng.index, dkng["MA_20"], label="MA 20", alpha=0.7)
axes[0, 0].plot(dkng.index, dkng["MA_50"], label="MA 50", alpha=0.7)
axes[0, 0].plot(dkng.index, dkng["MA_200"], label="MA 200", alpha=0.7)
axes[0, 0].set_title("Price & Moving Averages")
axes[0, 0].set_ylabel("Price ($)")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Volatility
axes[0, 1].plot(dkng.index, dkng["Volatility"] * 100, color="orange", linewidth=1.5)
axes[0, 1].set_title("20-Day Rolling Volatility (Annualized)")
axes[0, 1].set_ylabel("Volatility (%)")
axes[0, 1].grid(alpha=0.3)

# RSI
axes[1, 0].plot(dkng.index, dkng["RSI"], color="purple", linewidth=1.5)
axes[1, 0].axhline(70, color="r", linestyle="--", alpha=0.5, label="Overbought")
axes[1, 0].axhline(30, color="g", linestyle="--", alpha=0.5, label="Oversold")
axes[1, 0].set_title("Relative Strength Index (RSI)")
axes[1, 0].set_ylabel("RSI")
axes[1, 0].set_ylim(0, 100)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# MACD
axes[1, 1].plot(dkng.index, dkng["MACD"], label="MACD", linewidth=1.5)
axes[1, 1].plot(dkng.index, dkng["Signal_Line"], label="Signal Line", linewidth=1.5)
axes[1, 1].bar(dkng.index, dkng["MACD_Histogram"], label="Histogram", alpha=0.3)
axes[1, 1].set_title("MACD")
axes[1, 1].set_ylabel("MACD Value")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Daily Returns Distribution
axes[2, 0].hist(dkng["Daily_Return"].dropna() * 100, bins=50, alpha=0.7, color="steelblue")
axes[2, 0].set_title("Daily Returns Distribution")
axes[2, 0].set_xlabel("Daily Return (%)")
axes[2, 0].set_ylabel("Frequency")
axes[2, 0].grid(alpha=0.3)

# Cumulative Returns Comparison
dkng_cum = (1 + dkng["Daily_Return"]).cumprod() - 1
spy_cum = (1 + spy["Daily_Return"]).cumprod() - 1
axes[2, 1].plot(dkng.index, dkng_cum * 100, label="DKNG", linewidth=2)
axes[2, 1].plot(spy.index, spy_cum * 100, label="SPY", linewidth=2)
axes[2, 1].set_title("Cumulative Returns vs SPY")
axes[2, 1].set_ylabel("Cumulative Return (%)")
axes[2, 1].legend()
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Analysis complete! Charts displayed.")
print("="*60)