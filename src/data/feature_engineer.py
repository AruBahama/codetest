"""
Compute technical indicators (pandas-ta) + fundamental ratios (FinanceToolkit)
and join everything into a single DataFrame per ticker.
"""
from __future__ import annotations
import pandas as pd, pandas_ta as ta
from financetoolkit import Toolkit
from ..config import RAW_DIR, PROC_DIR

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df_ta = ta.strategy(
        ta.Strategy(
            name="all",
            talib=False  # ← allows pure-python fallback
        ),
        df.copy()
    )
    return pd.concat([df, df_ta], axis=1).dropna()

def add_fundamentals(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    tk = Toolkit(ticker, start=df.index.min(), end=df.index.max())
    funda = tk.ratios.collect()
    funda = funda.ffill().reindex(df.index)      # align on date index
    return pd.concat([df, funda], axis=1).dropna()

def engineer(ticker: str) -> None:
    raw_path = RAW_DIR / f"{ticker}.csv"
    proc_path = PROC_DIR / f"{ticker}.parquet"
    df = pd.read_csv(raw_path, parse_dates=True, index_col=0)
    df = add_technicals(df)
    df = add_fundamentals(ticker, df)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_path)

def batch_engineer() -> None:
    for path in RAW_DIR.glob("*.csv"):
        engineer(path.stem)

if __name__ == "__main__":
    batch_engineer()
