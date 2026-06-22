"""Tests for trailing asset feature calculations."""

from __future__ import annotations

from datetime import date

import numpy as np
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
        "acc_macd",
        "acc_macd_signal",
        "acc_macd_hist",
        "acc_signed_volume",
        "acc_klinger",
        "acc_klinger_signal",
        "acc_klinger_hist",
        "acc_price_drift",
        "acc_liquidity_growth",
        "liq_liquidity_growth",
        "realized_vol",
        "acc_realized_vol",
        "momentum_quality",
        "acc_momentum_quality",
        "liq_momentum_quality",
        "acc_vol_compression",
        "acc_low_vol",
        "acc_macd_improvement",
        "acc_klinger_improvement",
        "acc_macd_early",
        "acc_klinger_early",
        "acc_macd_bullish_hist",
        "acc_klinger_bullish_hist",
        "acc_rsi",
        "liq_amihud_illiquidity",
        "amihud_raw",
        "amihud_trend",
        "dollar_volume_trend",
        "liquidity_deterioration",
        "klinger_deterioration",
        "vol_expansion",
        "liquidity_ratio",
        "liquidity_shock",
        "liq_amihud_trend",
        "liq_dollar_volume_trend",
        "liq_liquidity_deterioration",
        "liq_klinger_deterioration",
        "liq_vol_expansion",
        "liq_liquidity_ratio",
        "liq_liquidity_shock",
        "dollar_volume",
    }
    assert expected_columns.issubset(features.columns)
    assert "accumulation_score" not in features.columns
    assert "liquidity_exit_score" not in features.columns


def test_requested_asset_component_formulas_are_trailing() -> None:
    dates = [date(2024, 1, day) for day in range(1, 9)]
    close = np.array([10.0, 10.5, 11.0, 10.8, 11.4, 12.0, 11.7, 12.3])
    volume = np.array([100.0, 120.0, 130.0, 125.0, 150.0, 170.0, 160.0, 190.0])
    data = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )
    config = FeatureConfig(
        rsi_window=2,
        accumulation_window=3,
        realized_vol_window=3,
        low_vol_window=4,
        liquidity_window=2,
        momentum_quality_window=3,
        klinger_fast_span=2,
        klinger_slow_span=3,
        klinger_signal_span=2,
        liquidity_ratio_window=4,
    )

    features = compute_asset_features(data, config).sort("date")
    row = features.row(-1, named=True)
    returns = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    dollar_volume = close * volume
    log_close = np.log(close)
    log_dollar_volume = np.log(dollar_volume + 1.0)
    amihud_raw = np.abs(returns) / (dollar_volume + 1e-9)
    log_amihud_raw = np.log(amihud_raw + 1e-12)
    realized_vol = np.full_like(close, np.nan)
    momentum_quality = np.full_like(close, np.nan)
    for index in range(2, len(close)):
        realized_vol[index] = np.std(returns[index - 2 : index + 1], ddof=1)
    for index in range(3, len(close)):
        window_return = close[index] / close[index - 3] - 1.0
        momentum_quality[index] = window_return / (
            np.var(returns[index - 2 : index + 1], ddof=1) + 1e-9
        )

    def normalized_slope(values: np.ndarray, index: int, window: int) -> float:
        slope = (values[index] - values[index - window + 1]) / float(window - 1)
        diffs = np.concatenate([[np.nan], np.diff(values)])
        variation = np.nanstd(diffs[index - window + 1 : index + 1], ddof=1)
        return slope / (variation + 1e-9)

    expected_price_drift = normalized_slope(log_close, len(close) - 1, 3)
    expected_liquidity_growth = normalized_slope(log_dollar_volume, len(close) - 1, 3)
    expected_amihud_trend = normalized_slope(log_amihud_raw, len(close) - 1, 3)
    expected_dollar_volume_trend = normalized_slope(log_dollar_volume, len(close) - 1, 3)
    expected_vol_compression = -normalized_slope(realized_vol, len(close) - 1, 3)
    expected_vol_expansion = normalized_slope(realized_vol, len(close) - 1, 3)
    expected_low_vol = -realized_vol[-1] / (np.median(realized_vol[-4:]) + 1e-9)
    expected_liquidity_ratio = dollar_volume[-1] / (np.median(dollar_volume[-4:]) + 1e-9)
    expected_liquidity_shock = 1.0 - expected_liquidity_ratio
    expected_macd_early = 1.0 / (
        1.0 + abs(row["acc_macd_signal"] / (row["close"] + 1e-9)) * 100.0
    )
    klinger_signal_abs = np.abs(features.get_column("acc_klinger_signal").to_numpy())
    klinger_signal_window = klinger_signal_abs[~np.isnan(klinger_signal_abs)]
    expected_klinger_early = 1.0 / (
        1.0 + abs(row["acc_klinger_signal"]) / (np.median(klinger_signal_window) + 1e-9)
    )
    expected_macd_bullish = np.tanh(row["acc_macd_hist"] / (row["close"] * 0.005 + 1e-9))
    klinger_hist_abs = np.abs(features.get_column("acc_klinger_hist").to_numpy())
    klinger_hist_window = klinger_hist_abs[~np.isnan(klinger_hist_abs)]
    expected_klinger_bullish = np.tanh(
        row["acc_klinger_hist"] / (np.median(klinger_hist_window) + 1e-9)
    )
    expected_klinger_deterioration = -normalized_slope(
        features.get_column("acc_klinger_hist").to_numpy(),
        len(close) - 1,
        3,
    )

    assert_allclose(row["acc_price_drift"], expected_price_drift, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_liquidity_growth"], expected_liquidity_growth, rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_liquidity_growth"], expected_liquidity_growth, rtol=RTOL, atol=ATOL)
    assert_allclose(row["amihud_raw"], amihud_raw[-1], rtol=RTOL, atol=ATOL)
    assert_allclose(row["amihud_trend"], expected_amihud_trend, rtol=RTOL, atol=ATOL)
    assert_allclose(row["dollar_volume_trend"], expected_dollar_volume_trend, rtol=RTOL, atol=ATOL)
    assert_allclose(row["liquidity_deterioration"], -expected_dollar_volume_trend, rtol=RTOL, atol=ATOL)
    assert_allclose(row["realized_vol"], realized_vol[-1], rtol=RTOL, atol=ATOL)
    assert_allclose(row["momentum_quality"], momentum_quality[-1], rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_momentum_quality"], momentum_quality[-1], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_momentum_quality"], momentum_quality[-1], rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_vol_compression"], expected_vol_compression, rtol=RTOL, atol=ATOL)
    assert_allclose(row["vol_expansion"], expected_vol_expansion, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_low_vol"], expected_low_vol, rtol=RTOL, atol=ATOL)
    assert_allclose(row["liquidity_ratio"], expected_liquidity_ratio, rtol=RTOL, atol=ATOL)
    assert_allclose(row["liquidity_shock"], expected_liquidity_shock, rtol=RTOL, atol=ATOL)
    assert_allclose(row["klinger_deterioration"], expected_klinger_deterioration, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_macd_early"], expected_macd_early, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_klinger_early"], expected_klinger_early, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_macd_bullish_hist"], expected_macd_bullish, rtol=RTOL, atol=ATOL)
    assert_allclose(row["acc_klinger_bullish_hist"], expected_klinger_bullish, rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_amihud_trend"], row["amihud_trend"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_dollar_volume_trend"], row["dollar_volume_trend"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_liquidity_deterioration"], row["liquidity_deterioration"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_klinger_deterioration"], row["klinger_deterioration"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_vol_expansion"], row["vol_expansion"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_liquidity_ratio"], row["liquidity_ratio"], rtol=RTOL, atol=ATOL)
    assert_allclose(row["liq_liquidity_shock"], row["liquidity_shock"], rtol=RTOL, atol=ATOL)
