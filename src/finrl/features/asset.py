"""Trailing features used by the direct-allocation policy."""

from __future__ import annotations

import polars as pl

from finrl.data.schema import enforce_ohlcv_schema
from finrl.features.columns import DIRECT_ALLOCATION_FEATURE_COLUMNS
from finrl.features.schema import FeatureConfig


def rolling_normalized_slope(
    column: str,
    window: int,
    output: str,
    *,
    by: str = "ticker",
) -> pl.Expr:
    """Return a trailing endpoint slope normalized by trailing variation."""

    slope = (
        pl.col(column) - pl.col(column).shift(window - 1).over(by)
    ) / float(window - 1)
    variation = (
        pl.col(column)
        .diff()
        .rolling_std(window_size=window, min_samples=2)
        .over(by)
    )
    return (slope / (variation + 1e-9)).alias(output)


def compute_asset_features(data: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Compute only the trailing features routed to direct allocation."""

    features = enforce_ohlcv_schema(data).sort(["ticker", "date"])
    features = features.with_columns(
        pl.col("close").pct_change().over("ticker").alias("_return"),
        pl.when((pl.col("high") - pl.col("low")).abs() > 1e-9)
        .then(
            ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close")))
            / ((pl.col("high") - pl.col("low")) + 1e-9)
        )
        .otherwise(0.0)
        .alias("_cmf_multiplier"),
        pl.col("close")
        .ewm_mean(span=config.macd_fast_span, adjust=False)
        .over("ticker")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=config.macd_slow_span, adjust=False)
        .over("ticker")
        .alias("_ema_slow"),
        pl.col("close")
        .ewm_mean(span=config.mr_ewma_span, adjust=False)
        .over("ticker")
        .alias("_mr_ewma"),
        (
            pl.col("volume")
            * pl.when(pl.col("close").diff().over("ticker") > 0.0)
            .then(1.0)
            .when(pl.col("close").diff().over("ticker") < 0.0)
            .then(-1.0)
            .otherwise(0.0)
        ).alias("_signed_volume"),
    )
    features = features.with_columns(
        (pl.col("_cmf_multiplier") * pl.col("volume")).alias("_cmf_volume"),
        pl.col("_return")
        .rolling_std(window_size=config.mr_vol_window, min_samples=2)
        .over("ticker")
        .alias("_mr_realized_vol"),
        (
            (
                pl.col("close")
                / pl.col("close")
                .shift(config.momentum_quality_window)
                .over("ticker")
                - 1.0
            )
            / (
                pl.col("_return")
                .rolling_var(
                    window_size=config.momentum_quality_window,
                    min_samples=2,
                )
                .over("ticker")
                + 1e-9
            )
        ).alias("acc_momentum_quality"),
        ((pl.col("_mr_ewma") - pl.col("close")) / pl.col("close")).alias("_mr_gap"),
        (pl.col("_mr_ewma") + 1e-9).log().alias("_log_ewma50"),
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("_macd"),
        pl.col("_signed_volume")
        .ewm_mean(span=config.klinger_fast_span, adjust=False)
        .over("ticker")
        .alias("_klinger_fast"),
        pl.col("_signed_volume")
        .ewm_mean(span=config.klinger_slow_span, adjust=False)
        .over("ticker")
        .alias("_klinger_slow"),
    )
    features = features.with_columns(
        (
            pl.col("_cmf_volume")
            .rolling_sum(window_size=config.cmf_window, min_samples=2)
            .over("ticker")
            / (
                pl.col("volume")
                .rolling_sum(window_size=config.cmf_window, min_samples=2)
                .over("ticker")
                + 1e-9
            )
        ).alias("cmf"),
        (pl.col("_mr_gap") / (pl.col("_mr_realized_vol") + 1e-9)).alias(
            "mr_ewma50_vol_gap"
        ),
        rolling_normalized_slope(
            "_log_ewma50", config.accumulation_window, "ewma50_slope"
        ),
        pl.col("_macd")
        .ewm_mean(span=config.macd_signal_span, adjust=False)
        .over("ticker")
        .alias("acc_macd_signal"),
        (pl.col("_klinger_fast") - pl.col("_klinger_slow")).alias("_klinger"),
    )
    features = features.with_columns(
        pl.col("_klinger")
        .ewm_mean(span=config.klinger_signal_span, adjust=False)
        .over("ticker")
        .alias("acc_klinger_signal"),
        rolling_normalized_slope(
            "cmf", config.accumulation_window, "cmf_slope"
        ),
        rolling_normalized_slope(
            "acc_macd_signal", config.accumulation_window, "_macd_signal_slope"
        ),
        (
            (
                (pl.col("cmf").shift(1).over("ticker") <= 0.0)
                & (pl.col("cmf") > 0.0)
            )
            .fill_null(False)
            .cast(pl.Int64)
        ).alias("_cmf_cross_up"),
        (
            (
                (pl.col("cmf").shift(1).over("ticker") >= 0.0)
                & (pl.col("cmf") < 0.0)
            )
            .fill_null(False)
            .cast(pl.Int64)
        ).alias("_cmf_cross_down"),
    )
    features = features.with_columns(
        rolling_normalized_slope(
            "acc_klinger_signal",
            config.accumulation_window,
            "_klinger_signal_slope",
        ),
        (pl.col("acc_macd_signal") * pl.col("_macd_signal_slope")).alias(
            "macd_signal_strength"
        ),
        (pl.col("_cmf_cross_up") - pl.col("_cmf_cross_down"))
        .cast(pl.Float64)
        .alias("cmf_cross_signal"),
        (pl.col("_cmf_cross_up") + pl.col("_cmf_cross_down")).alias(
            "_cmf_cross_event"
        ),
    )
    features = features.with_columns(
        (pl.col("acc_klinger_signal") * pl.col("_klinger_signal_slope")).alias(
            "klinger_signal_strength"
        ),
        pl.col("_cmf_cross_event")
        .cum_sum()
        .over("ticker")
        .alias("_cmf_cross_group"),
    )
    features = features.with_columns(
        pl.when(pl.col("_cmf_cross_group") > 0)
        .then(pl.int_range(pl.len()).over(["ticker", "_cmf_cross_group"]))
        .otherwise(None)
        .alias("cmf_days_since_cross")
    )
    return features.select(
        "date",
        "ticker",
        *DIRECT_ALLOCATION_FEATURE_COLUMNS,
    )
