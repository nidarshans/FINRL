"""Asset-level trailing feature engineering."""

from __future__ import annotations

import polars as pl

from finrl.data.schema import enforce_ohlcv_schema
from finrl.features.schema import FeatureConfig


def _sort_ohlcv(data: pl.DataFrame) -> pl.DataFrame:
    return enforce_ohlcv_schema(data).sort(["ticker", "date"])


def compute_returns(data: pl.DataFrame) -> pl.DataFrame:
    """Compute close-to-close returns using only prior close per ticker."""

    return _sort_ohlcv(data).with_columns(
        pl.col("close").pct_change().over("ticker").alias("return")
    )


def compute_rsi(data: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    """Compute trailing RSI per ticker."""

    features = compute_returns(data).with_columns(
        pl.col("close").diff().over("ticker").alias("_delta")
    )
    features = features.with_columns(
        pl.when(pl.col("_delta") > 0.0)
        .then(pl.col("_delta"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0.0)
        .then(-pl.col("_delta"))
        .otherwise(0.0)
        .alias("_loss"),
    )
    features = features.with_columns(
        pl.col("_gain")
        .rolling_mean(window_size=window, min_samples=window)
        .over("ticker")
        .alias("_avg_gain"),
        pl.col("_loss")
        .rolling_mean(window_size=window, min_samples=window)
        .over("ticker")
        .alias("_avg_loss"),
    )
    features = features.with_columns(
        pl.when(pl.col("_avg_loss") == 0.0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
        .alias("rsi")
    )
    return features.drop(
        ["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"], strict=False
    )


def compute_macd(
    data: pl.DataFrame,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pl.DataFrame:
    """Compute MACD line, signal, and histogram per ticker."""

    features = _sort_ohlcv(data).with_columns(
        pl.col("close")
        .ewm_mean(span=fast_span, adjust=False)
        .over("ticker")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=slow_span, adjust=False)
        .over("ticker")
        .alias("_ema_slow"),
    )
    features = features.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd")
    )
    features = features.with_columns(
        pl.col("macd")
        .ewm_mean(span=signal_span, adjust=False)
        .over("ticker")
        .alias("macd_signal")
    )
    features = features.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    )
    return features.drop(["_ema_fast", "_ema_slow"], strict=False)


def compute_trend_slope(data: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Compute trailing percent slope from the oldest close in the window."""

    return _sort_ohlcv(data).with_columns(
        (
            (pl.col("close") / pl.col("close").shift(window - 1).over("ticker") - 1.0)
            / float(window - 1)
        ).alias("trend_slope")
    )


def compute_dollar_volume(data: pl.DataFrame) -> pl.DataFrame:
    """Compute dollar volume from close and volume."""

    return _sort_ohlcv(data).with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )


def compute_amihud_illiquidity(data: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Compute trailing Amihud illiquidity from absolute return over dollar volume."""

    features = compute_returns(data).with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )
    features = features.with_columns(
        (pl.col("return").abs() / pl.col("dollar_volume")).alias("_amihud_raw")
    )
    return features.with_columns(
        pl.col("_amihud_raw")
        .rolling_mean(window_size=window, min_samples=1)
        .over("ticker")
        .alias("amihud_illiquidity")
    ).drop("_amihud_raw")


def compute_turnover_feature(data: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Compute a trailing volume turnover proxy.

    True share turnover requires shares outstanding or float data, which Phase 5
    yfinance OHLCV does not provide. This proxy is volume divided by trailing
    mean volume and is explicitly an offline liquidity feature, not portfolio
    turnover.
    """

    return _sort_ohlcv(data).with_columns(
        (
            pl.col("volume")
            / pl.col("volume")
            .rolling_mean(window_size=window, min_samples=1)
            .over("ticker")
        ).alias("turnover_feature")
    )


def compute_volume_momentum(data: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Compute trailing volume momentum versus the value `window` days ago."""

    return _sort_ohlcv(data).with_columns(
        (pl.col("volume") / pl.col("volume").shift(window).over("ticker") - 1.0).alias(
            "volume_momentum"
        )
    )


def compute_volume_acceleration(data: pl.DataFrame) -> pl.DataFrame:
    """Compute change in one-period volume growth per ticker."""

    features = _sort_ohlcv(data).with_columns(
        pl.col("volume").pct_change().over("ticker").alias("_volume_growth")
    )
    return features.with_columns(
        pl.col("_volume_growth").diff().over("ticker").alias("volume_acceleration")
    ).drop("_volume_growth")


def compute_asset_features(data: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Compute the Phase 6 asset feature table from OHLCV data."""

    features = compute_returns(data)
    features = features.with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )
    features = features.with_columns(
        (pl.col("return").abs() / pl.col("dollar_volume")).alias("_amihud_raw"),
        pl.col("close").diff().over("ticker").alias("_delta"),
        (
            (pl.col("close") / pl.col("close").shift(config.trend_window - 1).over("ticker") - 1.0)
            / float(config.trend_window - 1)
        ).alias("trend_slope"),
        (
            pl.col("volume")
            / pl.col("volume")
            .rolling_mean(window_size=config.volume_window, min_samples=1)
            .over("ticker")
        ).alias("turnover_feature"),
        (
            pl.col("volume") / pl.col("volume").shift(config.volume_window).over("ticker") - 1.0
        ).alias("volume_momentum"),
    )
    features = features.with_columns(
        pl.when(pl.col("_delta") > 0.0)
        .then(pl.col("_delta"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0.0)
        .then(-pl.col("_delta"))
        .otherwise(0.0)
        .alias("_loss"),
        pl.col("_amihud_raw")
        .rolling_mean(window_size=config.liquidity_window, min_samples=1)
        .over("ticker")
        .alias("amihud_illiquidity"),
        pl.col("volume").pct_change().over("ticker").alias("_volume_growth"),
        pl.col("close")
        .ewm_mean(span=12, adjust=False)
        .over("ticker")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=26, adjust=False)
        .over("ticker")
        .alias("_ema_slow"),
    )
    features = features.with_columns(
        pl.col("_gain")
        .rolling_mean(window_size=config.rsi_window, min_samples=config.rsi_window)
        .over("ticker")
        .alias("_avg_gain"),
        pl.col("_loss")
        .rolling_mean(window_size=config.rsi_window, min_samples=config.rsi_window)
        .over("ticker")
        .alias("_avg_loss"),
        pl.col("_volume_growth").diff().over("ticker").alias("volume_acceleration"),
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
    )
    features = features.with_columns(
        pl.when(pl.col("_avg_loss") == 0.0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
        .alias("rsi"),
        pl.col("macd").ewm_mean(span=9, adjust=False).over("ticker").alias("macd_signal"),
    )
    features = features.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    )
    return features.drop(
        [
            "_amihud_raw",
            "_delta",
            "_gain",
            "_loss",
            "_avg_gain",
            "_avg_loss",
            "_volume_growth",
            "_ema_fast",
            "_ema_slow",
        ],
        strict=False,
    )
