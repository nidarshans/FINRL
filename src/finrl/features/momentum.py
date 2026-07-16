"""Causal momentum and price-location candidate features."""

from __future__ import annotations

import polars as pl


def compute_momentum_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add trailing daily momentum features independently for each ticker.

    ``mom_126_21d`` intentionally skips the most recent 21 completed sessions,
    so it measures the return from t-126 to t-21 and never reads future bars.
    """

    return data.with_columns(
        (
            pl.col("close") / pl.col("close").shift(21).over("ticker") - 1.0
        ).alias("mom_21d"),
        (
            pl.col("close").shift(21).over("ticker")
            / pl.col("close").shift(126).over("ticker")
            - 1.0
        ).alias("mom_126_21d"),
        (
            pl.col("close")
            / pl.col("close")
            .rolling_max(window_size=252, min_samples=1)
            .over("ticker")
            - 1.0
        ).alias("near_52w_high"),
    )
