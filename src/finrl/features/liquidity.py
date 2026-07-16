"""Causal daily OHLCV liquidity and capacity proxy features."""

from __future__ import annotations

import polars as pl


def compute_liquidity_features(data: pl.DataFrame, window: int) -> pl.DataFrame:
    """Add trailing liquidity features independently for each ticker.

    Zero dollar-volume observations do not imply zero price impact, so their
    Amihud contribution is left unavailable rather than converted to zero.
    """

    if window <= 0:
        raise ValueError("Liquidity window must be positive.")
    output = data.with_columns(
        (pl.col("close") * pl.col("volume")).alias("_dollar_volume"),
        pl.when(pl.col("close") * pl.col("volume") > 0.0)
        .then(pl.col("_return").abs() / (pl.col("close") * pl.col("volume")))
        .otherwise(None)
        .alias("_amihud_daily"),
    )
    return output.with_columns(
        pl.when(
            pl.col("_dollar_volume")
            .rolling_mean(window_size=window, min_samples=1)
            .over("ticker")
            > 0.0
        )
        .then(
            pl.col("_dollar_volume")
            .rolling_mean(window_size=window, min_samples=1)
            .over("ticker")
            .log()
        )
        .otherwise(None)
        .alias("log_adv_20"),
        (
            (
                pl.col("volume")
                - pl.col("volume")
                .rolling_mean(window_size=window, min_samples=2)
                .over("ticker")
            )
            / (
                pl.col("volume")
                .rolling_std(window_size=window, min_samples=2)
                .over("ticker")
                + 1e-9
            )
        ).alias("volume_z_20"),
        pl.col("_amihud_daily")
        .rolling_mean(window_size=window, min_samples=1)
        .over("ticker")
        .alias("amihud_20"),
    )
