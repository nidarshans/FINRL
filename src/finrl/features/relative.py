"""Cross-sectional relative feature calculations."""

from __future__ import annotations

import polars as pl


def cross_sectional_percentile_rank(
    values_by_date: pl.DataFrame,
    value_col: str,
    rank_col: str | None = None,
) -> pl.DataFrame:
    """Rank values within each date only, avoiding global fitting or leakage."""

    if "date" not in values_by_date.columns:
        raise ValueError("Input must contain a 'date' column.")
    if value_col not in values_by_date.columns:
        raise ValueError(f"Input must contain value column '{value_col}'.")

    output_col = rank_col or f"{value_col}_percentile_rank"
    count_by_date = pl.len().over("date")
    return values_by_date.with_columns(
        pl.when(count_by_date <= 1)
        .then(1.0)
        .otherwise((pl.col(value_col).rank(method="average").over("date") - 1.0) / (count_by_date - 1.0))
        .alias(output_col)
    )
