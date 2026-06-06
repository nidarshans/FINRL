"""Polars schemas for market data ingestion."""

from __future__ import annotations

import polars as pl

OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

RETURNS_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "execution_date",
    "next_execution_date",
    "ticker",
    "open",
    "next_open",
    "return",
)

OHLCV_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Int64,
}

RETURNS_SCHEMA: dict[str, pl.DataType] = {
    "decision_date": pl.Date,
    "execution_date": pl.Date,
    "next_execution_date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "next_open": pl.Float64,
    "return": pl.Float64,
}


def enforce_ohlcv_schema(data: pl.DataFrame) -> pl.DataFrame:
    """Return OHLCV data with canonical column order and dtypes."""

    missing = [column for column in OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return data.select(
        [
            pl.col("date").cast(pl.Date),
            pl.col("ticker").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("adj_close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ]
    )


def enforce_returns_schema(data: pl.DataFrame) -> pl.DataFrame:
    """Return open-to-open returns with canonical column order and dtypes."""

    missing = [column for column in RETURNS_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing returns columns: {missing}")
    return data.select(
        [
            pl.col("decision_date").cast(pl.Date),
            pl.col("execution_date").cast(pl.Date),
            pl.col("next_execution_date").cast(pl.Date),
            pl.col("ticker").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64),
            pl.col("next_open").cast(pl.Float64),
            pl.col("return").cast(pl.Float64),
        ]
    )
