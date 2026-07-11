"""Tests for the direct-allocation asset features."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from numpy.testing import assert_allclose

from finrl.features.asset import compute_asset_features
from finrl.features.columns import DIRECT_ALLOCATION_FEATURE_COLUMNS
from finrl.features.schema import FeatureConfig

RTOL = 1e-6
ATOL = 1e-8


def _asset_ohlcv() -> pl.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(10)]
    multipliers = [-0.8, -0.6, 0.9, 0.8, 0.7, -0.9, -0.8, -0.7, 0.9, 0.8]

    def rows(ticker: str, offset: float) -> list[dict[str, object]]:
        return [
            {
                "date": day,
                "ticker": ticker,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset + multiplier,
                "adj_close": 10.0 + offset + multiplier,
                "volume": 100.0 + index,
            }
            for index, (day, multiplier) in enumerate(
                zip(dates, multipliers, strict=True)
            )
        ]

    return pl.DataFrame(rows("AAA", 0.0) + rows("BBB", 20.0))


def _config() -> FeatureConfig:
    return FeatureConfig(
        cmf_window=2,
        accumulation_window=3,
        momentum_quality_window=3,
        macd_fast_span=2,
        macd_slow_span=3,
        macd_signal_span=2,
        klinger_fast_span=2,
        klinger_slow_span=3,
        klinger_signal_span=2,
        mr_ewma_span=3,
        mr_vol_window=3,
    )


def test_compute_asset_features_returns_only_routed_features() -> None:
    features = compute_asset_features(_asset_ohlcv(), _config())

    assert features.columns == [
        "date",
        "ticker",
        *DIRECT_ALLOCATION_FEATURE_COLUMNS,
    ]


def test_cmf_cross_signal_and_days_since_cross() -> None:
    features = compute_asset_features(_asset_ohlcv(), _config()).filter(
        pl.col("ticker") == "AAA"
    )

    assert features.get_column("cmf_cross_signal").to_list()[:8] == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
    ]
    assert features.get_column("cmf_days_since_cross").to_list()[:8] == [
        None,
        None,
        0,
        1,
        2,
        0,
        1,
        2,
    ]


def test_cmf_slope_is_trailing_and_normalized_per_ticker() -> None:
    features = compute_asset_features(_asset_ohlcv(), _config()).filter(
        pl.col("ticker") == "AAA"
    )
    expected = features.select(
        (
            (pl.col("cmf") - pl.col("cmf").shift(2))
            / 2.0
            / (
                pl.col("cmf")
                .diff()
                .rolling_std(window_size=3, min_samples=2)
                + 1e-9
            )
        ).alias("expected")
    )
    valid = features.select("cmf_slope").with_columns(expected).drop_nulls()

    assert_allclose(
        valid.get_column("cmf_slope"),
        valid.get_column("expected"),
        rtol=RTOL,
        atol=ATOL,
    )


def test_mean_reversion_gap_keeps_existing_sign_convention() -> None:
    features = compute_asset_features(_asset_ohlcv(), _config()).filter(
        pl.col("ticker") == "AAA"
    )
    close = _asset_ohlcv().filter(pl.col("ticker") == "AAA").get_column("close")
    ewma = close.ewm_mean(span=3, adjust=False)
    valid = features.with_columns(ewma.alias("_ewma"), close.alias("_close")).drop_nulls(
        "mr_ewma50_vol_gap"
    )

    assert (
        valid.filter(pl.col("_close") < pl.col("_ewma")).get_column(
            "mr_ewma50_vol_gap"
        )
        > 0.0
    ).all()
    assert (
        valid.filter(pl.col("_close") > pl.col("_ewma")).get_column(
            "mr_ewma50_vol_gap"
        )
        < 0.0
    ).all()


def test_direct_features_are_per_ticker_and_trailing_only() -> None:
    data = _asset_ohlcv()
    config = _config()
    combined = compute_asset_features(data, config).sort(["ticker", "date"])
    aaa = combined.filter(pl.col("ticker") == "AAA")
    aaa_only = compute_asset_features(
        data.filter(pl.col("ticker") == "AAA"), config
    ).sort("date")

    assert_allclose(
        aaa.select(DIRECT_ALLOCATION_FEATURE_COLUMNS).to_numpy(),
        aaa_only.select(DIRECT_ALLOCATION_FEATURE_COLUMNS).to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )

    final_date = aaa.get_column("date")[-1]
    changed = data.with_columns(
        pl.when((pl.col("ticker") == "AAA") & (pl.col("date") == final_date))
        .then(pl.col("close") + 5.0)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when((pl.col("ticker") == "AAA") & (pl.col("date") == final_date))
        .then(pl.col("volume") * 10.0)
        .otherwise(pl.col("volume"))
        .alias("volume"),
        pl.when((pl.col("ticker") == "AAA") & (pl.col("date") == final_date))
        .then(pl.col("high") + 10.0)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when((pl.col("ticker") == "AAA") & (pl.col("date") == final_date))
        .then(pl.col("low") - 10.0)
        .otherwise(pl.col("low"))
        .alias("low"),
    )
    changed_prior = compute_asset_features(changed, config).filter(
        (pl.col("ticker") == "AAA") & (pl.col("date") < final_date)
    )
    assert_allclose(
        aaa.filter(pl.col("date") < final_date)
        .select(DIRECT_ALLOCATION_FEATURE_COLUMNS)
        .to_numpy(),
        changed_prior.select(DIRECT_ALLOCATION_FEATURE_COLUMNS).to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )
