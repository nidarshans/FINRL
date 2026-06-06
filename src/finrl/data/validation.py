"""Validation helpers for ingested market data."""

from __future__ import annotations

import polars as pl

from finrl.data.schema import OHLCV_COLUMNS, enforce_ohlcv_schema


def validate_ohlcv_data(
    data: pl.DataFrame,
    expected_tickers: tuple[str, ...] | None = None,
) -> None:
    """Validate canonical OHLCV data before feature generation."""

    ohlcv = enforce_ohlcv_schema(data)
    null_counts = ohlcv.select(pl.all().null_count()).row(0, named=True)
    null_columns = {
        column: count
        for column, count in null_counts.items()
        if column in OHLCV_COLUMNS and count > 0
    }
    if null_columns:
        raise ValueError(f"OHLCV data contains nulls: {null_columns}")
    if expected_tickers is not None:
        actual = set(ohlcv.get_column("ticker").unique().to_list())
        expected = set(expected_tickers)
        missing = sorted(expected.difference(actual))
        if missing:
            raise ValueError(f"Missing expected tickers: {missing}")
    duplicate_count = ohlcv.select(["date", "ticker"]).is_duplicated().sum()
    if duplicate_count:
        raise ValueError("OHLCV data contains duplicate ticker/date rows.")
