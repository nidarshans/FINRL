"""Causal price-position features relative to exponential moving averages."""

from __future__ import annotations

import polars as pl


def compute_ema_gap_features(
    data: pl.DataFrame,
    *,
    fast_span: int,
    medium_span: int,
    long_span: int,
    slow_span: int,
) -> pl.DataFrame:
    """Add dimensionless close-to-EMA gaps independently for each ticker."""

    if min(fast_span, medium_span, long_span, slow_span) <= 0:
        raise ValueError("EMA spans must be positive.")
    if not fast_span < medium_span < long_span < slow_span:
        raise ValueError("EMA spans must be strictly increasing.")
    output = data.with_columns(
        pl.col("close").ewm_mean(span=fast_span, adjust=False).over("ticker").alias("_ema20"),
        pl.col("close").ewm_mean(span=medium_span, adjust=False).over("ticker").alias("_ema50"),
        pl.col("close").ewm_mean(span=long_span, adjust=False).over("ticker").alias("_ema100"),
        pl.col("close").ewm_mean(span=slow_span, adjust=False).over("ticker").alias("_ema200"),
    )
    return output.with_columns(
        (pl.col("close") / (pl.col("_ema20") + 1e-9) - 1.0).alias("close_ema20_gap"),
        (pl.col("close") / (pl.col("_ema50") + 1e-9) - 1.0).alias("close_ema50_gap"),
        (pl.col("close") / (pl.col("_ema200") + 1e-9) - 1.0).alias("close_ema200_gap"),
        ((pl.col("_ema20") - pl.col("_ema50")) / (pl.col("close").abs() + 1e-9)).alias("ema20_ema50_distance"),
        ((pl.col("_ema50") - pl.col("_ema100")) / (pl.col("close").abs() + 1e-9)).alias("ema50_ema100_distance"),
        ((pl.col("_ema20") - pl.col("_ema100")) / (pl.col("close").abs() + 1e-9)).alias("ema20_ema100_distance"),
    )
