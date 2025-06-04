"""Tests for the preprocess routine."""

import pandas as pd

from src.pairs_trading.data import preprocess


def test_preprocess(tmp_path):
    """Ensure preprocess fills holes and writes a scaler."""
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2016-12-31", periods=9, freq="Y"),
            "Open": [None, 2, 3, 4, 5, None, 7, 8, 9],
            "Close": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    file_path = tmp_path / "AAA.parquet"
    df.to_parquet(file_path)

    preprocess(data_dir=tmp_path)

    out_df = pd.read_parquet(file_path)
    assert not out_df.isna().any().any()

    scaler_file = tmp_path / "scalers" / "AAA_scaler.joblib"
    assert scaler_file.exists()
