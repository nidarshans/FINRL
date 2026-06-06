"""Tests for trailing asset feature calculations."""

from __future__ import annotations

from datetime import date

import polars as pl
from numpy.testing import assert_allclose

from finrl.features.asset import (
    compute_amihud_illiquidity,
    compute_asset_features,
    compute_dollar_volume,
    compute_returns,
    compute_trend_slope,
    compute_turnover_feature,
    compute_volume_acceleration,
    compute_volume_momentum,
)
from finrl.features.relative import cross_sectional_percentile_rank
from finrl.features.schema import FeatureConfig

RTOL = 1e-6
ATOL = 1e-8


def _asset_ohlcv() -> pl.DataFrame:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    return pl.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "open": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "high": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "low": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "close": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "adj_close": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "volume": [100, 110, 120, 130, 200, 190, 210, 220],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def test_compute_returns_uses_prior_close_only() -> None:
    base = _asset_ohlcv()
    changed_future = base.with_columns(
        pl.when((pl.col("ticker") == "AAA") & (pl.col("date") == date(2024, 1, 4)))
        .then(999.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    base_return = compute_returns(base).filter(
        (pl.col("ticker") == "AAA") & (pl.col("date") == date(2024, 1, 3))
    )["return"][0]
    changed_return = compute_returns(changed_future).filter(
        (pl.col("ticker") == "AAA") & (pl.col("date") == date(2024, 1, 3))
    )["return"][0]

    assert_allclose(base_return, 12.0 / 11.0 - 1.0, rtol=RTOL, atol=ATOL)
    assert_allclose(changed_return, base_return, rtol=RTOL, atol=ATOL)


def test_asset_feature_functions_have_hand_checked_values() -> None:
    data = _asset_ohlcv().filter(pl.col("ticker") == "AAA")

    dollar_volume = compute_dollar_volume(data).filter(pl.col("date") == date(2024, 1, 2))
    trend = compute_trend_slope(data, window=3).filter(pl.col("date") == date(2024, 1, 3))
    amihud = compute_amihud_illiquidity(data, window=2).filter(
        pl.col("date") == date(2024, 1, 2)
    )
    turnover = compute_turnover_feature(data, window=2).filter(
        pl.col("date") == date(2024, 1, 2)
    )
    volume_momentum = compute_volume_momentum(data, window=2).filter(
        pl.col("date") == date(2024, 1, 3)
    )
    acceleration = compute_volume_acceleration(data).filter(
        pl.col("date") == date(2024, 1, 3)
    )

    assert_allclose(dollar_volume["dollar_volume"][0], 1210.0, rtol=RTOL, atol=ATOL)
    assert_allclose(trend["trend_slope"][0], 0.1, rtol=RTOL, atol=ATOL)
    assert_allclose(amihud["amihud_illiquidity"][0], (0.1 / 1210.0), rtol=RTOL, atol=ATOL)
    assert_allclose(turnover["turnover_feature"][0], 110.0 / 105.0, rtol=RTOL, atol=ATOL)
    assert_allclose(volume_momentum["volume_momentum"][0], 0.2, rtol=RTOL, atol=ATOL)
    assert acceleration["volume_acceleration"][0] is not None


def test_cross_sectional_percentile_rank_is_per_date() -> None:
    values = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "value": [1.0, 3.0, 100.0, 200.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    ranked = cross_sectional_percentile_rank(values, "value", "rank").sort(["date", "ticker"])

    assert ranked.get_column("rank").to_list() == [0.0, 1.0, 0.0, 1.0]


def test_compute_asset_features_contains_required_columns() -> None:
    features = compute_asset_features(
        _asset_ohlcv(),
        FeatureConfig(rsi_window=2, trend_window=3, volume_window=2, liquidity_window=2),
    )

    expected_columns = {
        "return",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "trend_slope",
        "amihud_illiquidity",
        "dollar_volume",
        "turnover_feature",
        "volume_momentum",
        "volume_acceleration",
    }
    assert expected_columns.issubset(features.columns)
