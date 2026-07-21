"""Causal volatility and drawdown candidate features."""

from __future__ import annotations

import polars as pl


def compute_risk_features(
    data: pl.DataFrame,
    *,
    atr_window: int,
    realized_vol_window: int,
    historical_vol_window: int,
    downside_vol_window: int,
    drawdown_window: int,
) -> pl.DataFrame:
    """Add trailing, dimensionless risk features independently per ticker."""

    if min(
        atr_window,
        realized_vol_window,
        historical_vol_window,
        downside_vol_window,
        drawdown_window,
    ) <= 0:
        raise ValueError("Risk windows must be positive.")
    previous_close = pl.col("close").shift(1).over("ticker")
    output = data.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - previous_close).abs(),
            (pl.col("low") - previous_close).abs(),
        ).alias("_true_range"),
        pl.when(pl.col("_return") < 0.0).then(pl.col("_return")).otherwise(0.0).alias("_downside_return"),
    )
    return output.with_columns(
        (
            pl.col("_true_range").rolling_mean(atr_window, min_samples=1).over("ticker")
            / (pl.col("close").abs() + 1e-9)
        ).alias("natr_20"),
        (
            pl.col("_return").rolling_std(realized_vol_window, min_samples=2).over("ticker")
            * (252.0**0.5)
        ).alias("realized_vol_20"),
        (
            pl.col("_return")
            .rolling_std(
                historical_vol_window,
                min_samples=historical_vol_window,
            )
            .over("ticker")
            * (252.0**0.5)
        ).alias("realized_vol_126"),
        (
            pl.col("_downside_return")
            .pow(2)
            .rolling_mean(downside_vol_window, min_samples=2)
            .over("ticker")
            .sqrt()
            * (252.0**0.5)
        ).alias("downside_vol_60"),
        (
            pl.col("close")
            / pl.col("close").rolling_max(drawdown_window, min_samples=1).over("ticker")
            - 1.0
        ).alias("max_drawdown_126"),
    )
