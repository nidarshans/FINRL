"""Macro feature engineering from ingested macro proxy series."""

from __future__ import annotations

import polars as pl


def _macro_column_name(ticker: str) -> str:
    normalized = (
        ticker.lower()
        .replace("^", "")
        .replace("-", "_")
        .replace("=", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    return f"macro_{normalized}"


def compute_macro_features(
    macro_data: pl.DataFrame,
    rolling_window: int = 20,
    include_levels: bool = False,
) -> pl.DataFrame:
    """Compute causal stationarity-oriented macro proxy features.

    For each ingested macro proxy series, this returns first differences,
    percent changes, log returns, and trailing z-scores. These transformations
    use only current and prior observations. Scaling and imputation remain
    deferred to the offline preprocessing phase.
    """

    if macro_data.is_empty():
        return pl.DataFrame({"date": []}).with_columns(pl.col("date").cast(pl.Date))
    required = {"date", "ticker", "value"}
    missing = required.difference(macro_data.columns)
    if missing:
        raise ValueError(f"Missing macro columns: {sorted(missing)}")
    wide = (
        macro_data.select(
            [
                pl.col("date").cast(pl.Date),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("value").cast(pl.Float64),
            ]
        )
        .pivot(index="date", on="ticker", values="value", aggregate_function="first")
        .sort("date")
        .rename(
            {
                column: _macro_column_name(column)
                for column in macro_data.get_column("ticker").unique().to_list()
            }
        )
    )
    value_columns = [column for column in wide.columns if column != "date"]
    transformed = wide.with_columns(
        [
            pl.col(column).diff().alias(f"{column}_diff")
            for column in value_columns
        ]
        + [
            pl.col(column).pct_change().alias(f"{column}_pct_change")
            for column in value_columns
        ]
        + [
            pl.when((pl.col(column) > 0.0) & (pl.col(column).shift(1) > 0.0))
            .then((pl.col(column) / pl.col(column).shift(1)).log())
            .otherwise(None)
            .alias(f"{column}_log_return")
            for column in value_columns
        ]
        + [
            (
                (pl.col(column) - pl.col(column).rolling_mean(window_size=rolling_window, min_samples=2))
                / pl.col(column).rolling_std(window_size=rolling_window, min_samples=2)
            ).alias(f"{column}_zscore")
            for column in value_columns
        ]
    )
    output_columns = ["date"]
    if include_levels:
        output_columns.extend(value_columns)
    for suffix in ("diff", "pct_change", "log_return", "zscore"):
        output_columns.extend(f"{column}_{suffix}" for column in value_columns)
    return transformed.select(output_columns)
