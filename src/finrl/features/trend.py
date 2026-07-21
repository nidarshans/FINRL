"""Causal price-position features relative to exponential moving averages."""

from __future__ import annotations

import polars as pl


def compute_ema_gap_features(
    data: pl.DataFrame,
    *,
    fast_span: int,
    medium_span: int,
    slow_span: int,
    slope_window: int,
) -> pl.DataFrame:
    """Add causal EMA price gaps, distances, and trailing endpoint slopes."""

    if min(fast_span, medium_span, slow_span, slope_window) <= 0:
        raise ValueError("EMA spans and slope window must be positive.")
    if not fast_span < medium_span < slow_span:
        raise ValueError("EMA spans must be strictly increasing.")
    output = data.with_columns(
        pl.col("close").ewm_mean(span=fast_span, adjust=False).over("ticker").alias("_ema20"),
        pl.col("close").ewm_mean(span=medium_span, adjust=False).over("ticker").alias("_ema50"),
        pl.col("close").ewm_mean(span=slow_span, adjust=False).over("ticker").alias("_ema200"),
    )
    return output.with_columns(
        (pl.col("close") - pl.col("_ema20")).alias("close_ema20_gap"),
        (pl.col("close") - pl.col("_ema50")).alias("close_ema50_gap"),
        (pl.col("close") - pl.col("_ema200")).alias("close_ema200_gap"),
        (pl.col("_ema20") - pl.col("_ema50")).alias("ema20_ema50_distance"),
        (pl.col("_ema50") - pl.col("_ema200")).alias("ema50_ema200_distance"),
        (pl.col("_ema20") - pl.col("_ema200")).alias("ema20_ema200_distance"),
        (
            pl.col("_ema20")
            - pl.col("_ema20").shift(slope_window).over("ticker")
        ).truediv(float(slope_window)).alias("ema20_slope"),
        (
            pl.col("_ema50")
            - pl.col("_ema50").shift(slope_window).over("ticker")
        ).truediv(float(slope_window)).alias("ema50_slope"),
        (
            pl.col("_ema200")
            - pl.col("_ema200").shift(slope_window).over("ticker")
        ).truediv(float(slope_window)).alias("ema200_slope"),
    )
