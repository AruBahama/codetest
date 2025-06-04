"""
Download daily OHLCV with yfinance and save one CSV per ticker into data/raw/.
Includes forward-fill to ensure every symbol ends exactly on each quarter’s
last trading day (required for FinanceToolkit fundamentals).
"""
from __future__ import annotations
import pandas as pd, yfinance as yf, os
from datetime import datetime
from . import feature_engineer  # type: ignore
from ..config import RAW_DIR, START_DATE, END_DATE, TICKER_FILE, WINDOW_LENGTH

def align_to_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    # Resample to ‘B’ for business-day, ffill, then slice so that the final row
    # of each quarter is kept (avoids FinanceToolkit misalignment).
    df = df.resample("B").ffill()
    q_end_mask = df.index.isin(df.resample("Q").last().index)
    return df.loc[q_end_mask]

def download_and_save(ticker: str) -> None:
    print(f" → downloading {ticker}")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if data.empty:
        print(f"    ! no data for {ticker}")
        return
    data = align_to_quarter_end(data)
    out_path = RAW_DIR / f"{ticker}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path)
    print(f"    ✓ saved to {out_path}")

def batch_download() -> None:
    tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
    for t in tickers:
        download_and_save(t.strip())

if __name__ == "__main__":
    batch_download()
