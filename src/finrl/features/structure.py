"""Confirmation-delayed market-structure features from daily OHLCV bars."""

from __future__ import annotations

import polars as pl


def compute_structure_features(
    data: pl.DataFrame, *, atr_window: int, swing_left: int, swing_right: int
) -> pl.DataFrame:
    """Add causal pivot, reference-distance, and daily-bar AVWAP features.

    A pivot at p is published only at p + ``swing_right``.  The AVWAP anchor
    begins at that confirmation date, avoiding retrospective use of bars that
    were not known to be part of a swing when they occurred.
    """

    if min(atr_window, swing_left, swing_right) <= 0:
        raise ValueError("ATR and swing windows must be positive.")
    previous_close = pl.col("close").shift(1).over("ticker")
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    output = data.with_columns(
        true_range.alias("_true_range"),
        pl.lit(True).alias("_swing_high_candidate"),
        pl.lit(True).alias("_swing_low_candidate"),
    )
    for offset in range(1, swing_left + 1):
        output = output.with_columns(
            (pl.col("_swing_high_candidate") & (pl.col("high") > pl.col("high").shift(offset).over("ticker"))).alias("_swing_high_candidate"),
            (pl.col("_swing_low_candidate") & (pl.col("low") < pl.col("low").shift(offset).over("ticker"))).alias("_swing_low_candidate"),
        )
    for offset in range(1, swing_right + 1):
        output = output.with_columns(
            (pl.col("_swing_high_candidate") & (pl.col("high") >= pl.col("high").shift(-offset).over("ticker"))).alias("_swing_high_candidate"),
            (pl.col("_swing_low_candidate") & (pl.col("low") <= pl.col("low").shift(-offset).over("ticker"))).alias("_swing_low_candidate"),
        )
    output = output.with_columns(
        pl.col("_true_range").rolling_mean(window_size=atr_window, min_samples=1).over("ticker").alias("_atr"),
        pl.when(pl.col("_swing_high_candidate").shift(swing_right).over("ticker"))
        .then(pl.col("high").shift(swing_right).over("ticker"))
        .otherwise(None)
        .alias("_confirmed_high"),
        pl.when(pl.col("_swing_low_candidate").shift(swing_right).over("ticker"))
        .then(pl.col("low").shift(swing_right).over("ticker"))
        .otherwise(None)
        .alias("_confirmed_low"),
    ).with_columns(
        pl.col("_confirmed_high").forward_fill().over("ticker").alias("_resistance"),
        pl.col("_confirmed_low").forward_fill().over("ticker").alias("_support"),
        pl.when(pl.col("_confirmed_low").is_not_null()).then(1).otherwise(0).cum_sum().over("ticker").alias("_low_group"),
    ).with_columns(
        pl.when(pl.col("_atr") > 0.0)
        .then((pl.col("close") - pl.col("_support")) / pl.col("_atr"))
        .otherwise(None)
        .alias("support_distance_atr"),
        pl.when(pl.col("_atr") > 0.0)
        .then((pl.col("_resistance") - pl.col("close")) / pl.col("_atr"))
        .otherwise(None)
        .alias("resistance_distance_atr"),
        (pl.col("_confirmed_low").is_not_null().cast(pl.Float64) - pl.col("_confirmed_high").is_not_null().cast(pl.Float64)).alias("confirmed_structure_score"),
    ).with_columns(
        pl.when(pl.col("_low_group") > 0)
        .then(pl.int_range(pl.len()).over(["ticker", "_low_group"]))
        .otherwise(None)
        .alias("bars_since_swing_low"),
        (pl.col("_confirmed_low").is_not_null().cast(pl.Int64)).cum_sum().over("ticker").alias("_avwap_group"),
    ).with_columns(
        pl.when(pl.col("_avwap_group") > 0)
        .then(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0 * pl.col("volume"))
            .cum_sum()
            .over(["ticker", "_avwap_group"])
            / pl.col("volume").cum_sum().over(["ticker", "_avwap_group"])
        )
        .otherwise(None)
        .alias("_swing_avwap"),
    )
    return output.with_columns(
        pl.when(pl.col("_atr") > 0.0)
        .then((pl.col("close") - pl.col("_swing_avwap")) / pl.col("_atr"))
        .otherwise(None)
        .alias("swing_avwap_distance_atr")
    )
