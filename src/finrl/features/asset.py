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


def rolling_normalized_slope(
    column: str,
    window: int,
    output: str,
    *,
    by: str = "ticker",
) -> pl.Expr:
    """Return a trailing endpoint slope normalized by trailing variation."""

    slope = (
        (pl.col(column) - pl.col(column).shift(window - 1).over(by))
        / float(window - 1)
    )
    variation = pl.col(column).diff().rolling_std(
        window_size=window,
        min_samples=2,
    ).over(by)
    return (slope / (variation + 1e-9)).alias(output)


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
    """Compute trailing per-asset OHLCV component features."""

    features = compute_returns(data)
    features = features.with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )
    features = features.with_columns(
        (pl.col("return").abs() / (pl.col("dollar_volume") + 1e-9)).alias("amihud_raw"),
        pl.col("close").diff().over("ticker").alias("_delta"),
        (
            pl.col("volume")
            * pl.when(pl.col("close").diff().over("ticker") > 0.0)
            .then(1.0)
            .when(pl.col("close").diff().over("ticker") < 0.0)
            .then(-1.0)
            .otherwise(0.0)
        ).alias("acc_signed_volume"),
        pl.col("close").log().alias("_log_close"),
        (pl.col("dollar_volume") + 1.0).log().alias("_log_dollar_volume"),
    )
    features = features.with_columns(
        (pl.col("amihud_raw") + 1e-12).log().alias("_log_amihud_raw"),
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
        pl.col("amihud_raw")
        .rolling_mean(window_size=config.liquidity_window, min_samples=1)
        .over("ticker")
        .alias("liq_amihud_illiquidity"),
        pl.col("volume").pct_change().over("ticker").alias("_volume_growth"),
        pl.col("close")
        .ewm_mean(span=12, adjust=False)
        .over("ticker")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=26, adjust=False)
        .over("ticker")
        .alias("_ema_slow"),
        pl.col("return")
        .rolling_std(window_size=config.realized_vol_window, min_samples=2)
        .over("ticker")
        .alias("acc_realized_vol"),
        rolling_normalized_slope("_log_close", config.accumulation_window, "acc_price_drift"),
        rolling_normalized_slope(
            "_log_dollar_volume",
            config.accumulation_window,
            "acc_liquidity_growth",
        ),
    )
    features = features.with_columns(
        pl.col("acc_realized_vol").alias("realized_vol"),
        pl.col("_gain")
        .rolling_mean(window_size=config.rsi_window, min_samples=config.rsi_window)
        .over("ticker")
        .alias("_avg_gain"),
        pl.col("_loss")
        .rolling_mean(window_size=config.rsi_window, min_samples=config.rsi_window)
        .over("ticker")
        .alias("_avg_loss"),
        rolling_normalized_slope(
            "_log_amihud_raw",
            config.accumulation_window,
            "amihud_trend",
        ),
        rolling_normalized_slope(
            "_log_dollar_volume",
            config.accumulation_window,
            "dollar_volume_trend",
        ),
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("acc_macd"),
        pl.col("acc_signed_volume")
        .ewm_mean(span=config.klinger_fast_span, adjust=False)
        .over("ticker")
        .alias("_klinger_fast"),
        pl.col("acc_signed_volume")
        .ewm_mean(span=config.klinger_slow_span, adjust=False)
        .over("ticker")
        .alias("_klinger_slow"),
    )
    features = features.with_columns(
        (
            -pl.col("acc_realized_vol")
            / (
                pl.col("acc_realized_vol")
                .rolling_median(window_size=config.low_vol_window, min_samples=2)
                .over("ticker")
                + 1e-9
            )
        ).alias("acc_low_vol"),
        (
            -(
                (
                    pl.col("acc_realized_vol")
                    - pl.col("acc_realized_vol")
                    .shift(config.accumulation_window - 1)
                    .over("ticker")
                )
                / float(config.accumulation_window - 1)
            )
            / (
                pl.col("acc_realized_vol")
                .diff()
                .rolling_std(window_size=config.accumulation_window, min_samples=2)
                .over("ticker")
                + 1e-9
            )
        ).alias("acc_vol_compression"),
        (
            (
                (
                    pl.col("acc_realized_vol")
                    - pl.col("acc_realized_vol")
                    .shift(config.accumulation_window - 1)
                    .over("ticker")
                )
                / float(config.accumulation_window - 1)
            )
            / (
                pl.col("acc_realized_vol")
                .diff()
                .rolling_std(window_size=config.accumulation_window, min_samples=2)
                .over("ticker")
                + 1e-9
            )
        ).alias("vol_expansion"),
        (
            pl.col("dollar_volume")
            / (
                pl.col("dollar_volume")
                .rolling_median(window_size=config.liquidity_ratio_window, min_samples=1)
                .over("ticker")
                + 1e-9
            )
        ).alias("liquidity_ratio"),
    )
    features = features.with_columns(
        pl.when(pl.col("_avg_loss") == 0.0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
        .alias("acc_rsi"),
        pl.col("acc_macd")
        .ewm_mean(span=9, adjust=False)
        .over("ticker")
        .alias("acc_macd_signal"),
        (pl.col("_klinger_fast") - pl.col("_klinger_slow")).alias("acc_klinger"),
    )
    features = features.with_columns(
        (pl.col("acc_macd") - pl.col("acc_macd_signal")).alias("acc_macd_hist"),
        pl.col("acc_klinger")
        .ewm_mean(span=config.klinger_signal_span, adjust=False)
        .over("ticker")
        .alias("acc_klinger_signal"),
    )
    features = features.with_columns(
        (pl.col("acc_klinger") - pl.col("acc_klinger_signal")).alias("acc_klinger_hist")
    )
    features = features.with_columns(
        pl.col("amihud_trend").alias("liq_amihud_trend"),
        pl.col("acc_liquidity_growth").alias("liq_liquidity_growth"),
        pl.col("dollar_volume_trend").alias("liq_dollar_volume_trend"),
        (-pl.col("dollar_volume_trend")).alias("liquidity_deterioration"),
        (-rolling_normalized_slope(
            "acc_klinger_hist",
            config.accumulation_window,
            "_klinger_deterioration_raw",
        )).alias("klinger_deterioration"),
        pl.col("vol_expansion").alias("liq_vol_expansion"),
        pl.col("liquidity_ratio").alias("liq_liquidity_ratio"),
        (1.0 - pl.col("liquidity_ratio")).alias("liquidity_shock"),
        rolling_normalized_slope(
            "acc_macd_hist",
            config.accumulation_window,
            "acc_macd_improvement",
        ),
        rolling_normalized_slope(
            "acc_klinger_hist",
            config.accumulation_window,
            "acc_klinger_improvement",
        ),
        (
            1.0
            / (
                1.0
                + (pl.col("acc_macd_signal") / (pl.col("close") + 1e-9)).abs() * 100.0
            )
        ).alias("acc_macd_early"),
        (
            1.0
            / (
                1.0
                + pl.col("acc_klinger_signal").abs()
                / (
                    pl.col("acc_klinger_signal")
                    .abs()
                    .rolling_median(window_size=60, min_samples=2)
                    .over("ticker")
                    + 1e-9
                )
            )
        ).alias("acc_klinger_early"),
        (
            (pl.col("acc_macd_hist") / (pl.col("close") * 0.005 + 1e-9)).tanh()
        ).alias("acc_macd_bullish_hist"),
        (
            (
                pl.col("acc_klinger_hist")
                / (
                    pl.col("acc_klinger_hist")
                    .abs()
                    .rolling_median(window_size=60, min_samples=2)
                    .over("ticker")
                    + 1e-9
                )
            ).tanh()
        ).alias("acc_klinger_bullish_hist"),
    )
    features = features.with_columns(
        pl.col("liquidity_deterioration").alias("liq_liquidity_deterioration"),
        pl.col("klinger_deterioration").alias("liq_klinger_deterioration"),
        pl.col("liquidity_shock").alias("liq_liquidity_shock"),
    )
    return features.drop(
        [
            "_klinger_deterioration_raw",
            "_delta",
            "_gain",
            "_loss",
            "_avg_gain",
            "_avg_loss",
            "_volume_growth",
            "_ema_fast",
            "_ema_slow",
            "_klinger_fast",
            "_klinger_slow",
            "_log_close",
            "_log_dollar_volume",
            "_log_amihud_raw",
        ],
        strict=False,
    )
