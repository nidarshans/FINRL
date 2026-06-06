"""Tests for Polars market data schemas and storage."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from finrl.data.schema import OHLCV_COLUMNS, enforce_ohlcv_schema
from finrl.data.storage import read_parquet, write_parquet
from finrl.data.validation import validate_ohlcv_data


def _toy_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
            "ticker": ["AAA", "AAA"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "adj_close": [100.5, 101.5],
            "volume": [1_000, 1_100],
        }
    )


def test_enforce_ohlcv_schema_returns_polars_dataframe() -> None:
    data = enforce_ohlcv_schema(_toy_ohlcv())

    assert isinstance(data, pl.DataFrame)
    assert data.columns == list(OHLCV_COLUMNS)
    assert data.schema["date"] == pl.Date
    assert data.schema["volume"] == pl.Int64


def test_enforce_ohlcv_schema_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing OHLCV columns"):
        enforce_ohlcv_schema(_toy_ohlcv().drop("open"))


def test_validate_ohlcv_data_rejects_nulls() -> None:
    data = _toy_ohlcv().with_columns(
        pl.when(pl.col("date") == "2024-01-08")
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )

    with pytest.raises(ValueError, match="contains nulls"):
        validate_ohlcv_data(data)


def test_validate_ohlcv_data_rejects_missing_expected_tickers() -> None:
    with pytest.raises(ValueError, match="Missing expected tickers"):
        validate_ohlcv_data(_toy_ohlcv(), expected_tickers=("AAA", "BBB"))


def test_data_cache_round_trip_preserves_schema(tmp_path: Path) -> None:
    path = tmp_path / "ohlcv.parquet"
    data = enforce_ohlcv_schema(_toy_ohlcv())

    write_parquet(data, path)
    loaded = read_parquet(path)

    assert loaded.schema == data.schema
    assert loaded.to_dicts() == data.to_dicts()
