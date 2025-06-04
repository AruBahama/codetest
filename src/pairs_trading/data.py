"""Data downloading utilities for pairs trading research."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency for linting
    yf = None  # type: ignore

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency for linting
    ta = None  # type: ignore


def download_ohlcv(
    csv_path: Path | str,
    data_dir: Path | str = "data/raw",
    max_retries: int = 5,
) -> None:
    """Download daily OHLCV bars for tickers listed in a CSV.

    Parameters
    ----------
    csv_path : Path | str
        Path to CSV containing tickers in column "a".
    data_dir : Path | str, optional
        Directory to write parquet files to, by default "data/raw".
    max_retries : int, optional
        Maximum number of retries per ticker when encountering HTTP 429 errors.
    """

    if yf is None:
        # pragma: no cover - safeguard for environments without yfinance
        raise ImportError("yfinance is required to download data")

    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = pd.read_csv(csv_path)["a"].dropna().unique().tolist()

    progress_columns: Iterable = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*progress_columns) as progress:
        task_id = progress.add_task("Downloading", total=len(tickers))
        for ticker in tickers:
            file_path = data_dir / f"{ticker}.parquet"
            wait = 1
            attempt = 0
            while attempt <= max_retries:
                try:
                    df = yf.download(ticker, progress=False)
                except Exception as exc:  # pragma: no cover - network errors
                    if "429" in str(exc):
                        time.sleep(wait)
                        wait = min(wait * 2, 60)
                        attempt += 1
                        continue
                    raise
                if not df.empty:
                    df = df.reset_index().loc[
                        :,
                        [
                            "Date",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Adj Close",
                            "Volume",
                        ],
                    ]
                    df.to_parquet(file_path)
                break
            progress.advance(task_id)


def build_features(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "data/processed",
) -> None:
    """Read raw OHLCV parquet files, add technical indicators, and save.

    Parameters
    ----------
    data_dir : Path | str, optional
        Directory containing raw parquet files, by default ``"data/raw"``.
    output_dir : Path | str, optional
        Directory to write processed parquet files to, by default
        ``"data/processed"``.
    """

    if ta is None:  # pragma: no cover - skip when pandas_ta is unavailable
        raise ImportError("pandas_ta is required to build features")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(data_dir.glob("*.parquet"))

    progress_columns: Iterable = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*progress_columns) as progress:
        task_id = progress.add_task("Building", total=len(files))
        for file_path in files:
            df = pd.read_parquet(file_path)

            df.ta.sma(length=10, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(append=True)
            df.ta.atr(length=14, append=True)

            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].astype("float32")

            out_path = output_dir / file_path.name
            df.to_parquet(out_path)
            progress.advance(task_id)
