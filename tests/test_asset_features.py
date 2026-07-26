"""Tests for the direct-allocation asset features."""

from __future__ import annotations

from datetime import date, timedelta
import math

import polars as pl
from numpy.testing import assert_allclose

from finrl.features.asset import compute_asset_features
from finrl.features.columns import DIRECT_ALLOCATION_FEATURE_COLUMNS
from finrl.features.liquidity import compute_liquidity_features
from finrl.features.structure import compute_structure_features
from finrl.features.market_relative import compute_market_relative_features
from finrl.features.risk import compute_risk_features
from finrl.features.trend import compute_ema_gap_features
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


def test_compute_asset_features_returns_routed_and_candidate_features() -> None:
    features = compute_asset_features(_asset_ohlcv(), _config())

    assert features.columns == [
        "date",
        "ticker",
        "close",
        *DIRECT_ALLOCATION_FEATURE_COLUMNS,
        "mom_21d",
        "mom_126_21d",
        "near_52w_high",
        "log_adv_20",
        "volume_z_20",
        "close_vwap20_gap",
        "amihud_20",
        "confirmed_structure_score",
        "support_distance_atr",
        "resistance_distance_atr",
        "swing_avwap_distance_atr",
        "bars_since_swing_low",
        "natr_20",
        "realized_vol_20",
        "realized_vol_126",
        "downside_vol_60",
        "max_drawdown_126",
        "close_ema20_gap",
        "close_ema50_gap",
        "close_ema200_gap",
        "ema20_ema50_distance",
        "ema50_ema200_distance",
        "ema20_ema200_distance",
        "ema20_slope",
        "ema50_slope",
        "ema200_slope",
    ]


def test_liquidity_features_match_trailing_dollar_volume_definitions() -> None:
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["AAA", "AAA"],
            "high": [11.0, 13.0],
            "low": [9.0, 10.0],
            "close": [10.0, 11.0],
            "volume": [100.0, 200.0],
            "_return": [None, 0.1],
        }
    )
    actual = compute_liquidity_features(data, window=2).tail(1).row(named=True)

    assert_allclose(actual["log_adv_20"], math.log(1600.0), rtol=RTOL)
    assert_allclose(actual["volume_z_20"], 2**-0.5, rtol=RTOL)
    expected_vwap = (10.0 * 100.0 + (34.0 / 3.0) * 200.0) / 300.0
    assert_allclose(actual["close_vwap20_gap"], 11.0 - expected_vwap, rtol=RTOL)
    assert_allclose(actual["amihud_20"], 0.1 / 2200.0, rtol=RTOL)


def test_structure_pivots_are_published_only_on_the_confirmation_date() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(5)]
    data = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 5,
            "open": [10.0] * 5,
            "high": [10.0, 11.0, 15.0, 12.0, 10.0],
            "low": [5.0, 4.0, 4.0, 4.0, 5.0],
            "close": [8.0, 9.0, 14.0, 11.0, 8.0],
            "adj_close": [8.0, 9.0, 14.0, 11.0, 8.0],
            "volume": [100.0] * 5,
            "_return": [None, 0.1, 0.2, -0.2, -0.1],
        }
    )
    features = compute_structure_features(
        data, atr_window=2, swing_left=1, swing_right=1
    )

    assert features.get_column("_confirmed_high").to_list()[2] is None
    assert features.get_column("_confirmed_high").to_list()[3] == 15.0
    assert features.get_column("confirmed_structure_score").to_list()[3] == -1.0


def test_market_relative_features_require_actual_benchmark_dates() -> None:
    data = _asset_ohlcv().filter(pl.col("ticker") == "AAA")
    benchmark = data.with_columns(pl.lit("SPY").alias("ticker"))
    features = compute_market_relative_features(data, benchmark)

    assert features.get_column("beta_252")[2] is not None
    missing_date_benchmark = benchmark.filter(pl.col("date") != data.get_column("date")[2])
    missing = compute_market_relative_features(data, missing_date_benchmark)
    assert missing.get_column("beta_252")[2] is None


def test_risk_features_use_gap_aware_true_range_and_trailing_drawdown() -> None:
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["AAA"] * 3,
            "open": [10.0, 14.0, 11.0],
            "high": [11.0, 15.0, 12.0],
            "low": [9.0, 13.0, 10.0],
            "close": [10.0, 14.0, 11.0],
            "adj_close": [10.0, 14.0, 11.0],
            "volume": [100.0] * 3,
            "_return": [None, 0.4, 11.0 / 14.0 - 1.0],
        }
    )
    features = compute_risk_features(
        data, atr_window=2, realized_vol_window=2, historical_vol_window=126,
        downside_vol_window=2, drawdown_window=3,
    )

    assert_allclose(features.get_column("natr_20")[1], 3.5 / 14.0, rtol=RTOL)
    assert_allclose(features.get_column("max_drawdown_126")[2], 11.0 / 14.0 - 1.0, rtol=RTOL)


def test_historical_volatility_uses_126_trading_day_window() -> None:
    row_count = 127
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1) + timedelta(days=index) for index in range(row_count)],
            "ticker": ["AAA"] * row_count,
            "high": [101.0] * row_count,
            "low": [99.0] * row_count,
            "close": [100.0] * row_count,
            "_return": [None, *([0.0] * 125), 1.0],
        }
    )

    features = compute_risk_features(
        data,
        atr_window=20,
        realized_vol_window=20,
        historical_vol_window=126,
        downside_vol_window=60,
        drawdown_window=126,
    )

    historical_volatility = features.get_column("realized_vol_126")
    assert historical_volatility[-2] is None
    assert_allclose(historical_volatility[-1], math.sqrt(2.0))


def test_ema_price_gaps_and_slopes_are_trailing() -> None:
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1) + timedelta(days=index) for index in range(3)],
            "ticker": ["AAA"] * 3,
            "close": [10.0, 12.0, 14.0],
        }
    )
    features = compute_ema_gap_features(
        data, fast_span=2, medium_span=3, slow_span=5, slope_window=2
    )

    assert features.get_column("ema20_slope")[0] is None
    assert_allclose(features.get_column("close_ema20_gap")[0], 0.0, atol=ATOL)
    assert_allclose(
        features.get_column("close_ema20_gap")[2],
        features.get_column("close")[2] - features.get_column("_ema20")[2],
        rtol=RTOL,
    )
    assert features.get_column("close_ema20_gap")[2] > 0.0
    assert features.get_column("ema20_ema50_distance")[2] > 0.0
    assert features.get_column("ema20_ema200_distance")[2] > 0.0
    assert_allclose(
        features.get_column("ema20_slope")[2],
        (features.get_column("_ema20")[2] - features.get_column("_ema20")[0]) / 2.0,
        rtol=RTOL,
    )


def test_momentum_features_use_trailing_horizons_and_per_ticker_highs() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(130)]
    rows = [
        {
            "date": day,
            "ticker": "AAA",
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "adj_close": 100.0 + index,
            "volume": 100.0,
        }
        for index, day in enumerate(dates)
    ]
    features = compute_asset_features(pl.DataFrame(rows), _config())
    final = features.tail(1).row(named=True)

    assert_allclose(final["mom_21d"], 229.0 / 208.0 - 1.0, rtol=RTOL)
    assert_allclose(final["mom_126_21d"], 208.0 / 103.0 - 1.0, rtol=RTOL)
    assert_allclose(final["near_52w_high"], 0.0, atol=ATOL)


def test_future_price_mutation_cannot_change_prior_momentum_features() -> None:
    dates = [date(2023, 1, 1) + timedelta(days=index) for index in range(130)]
    data = pl.DataFrame(
        [
            {
                "date": day,
                "ticker": "AAA",
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "adj_close": 100.0 + index,
                "volume": 100.0,
            }
            for index, day in enumerate(dates)
        ]
    )
    baseline = compute_asset_features(data, _config())
    changed = data.with_columns(
        pl.when(pl.col("date") == dates[-1])
        .then(pl.col("close") * 2.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    prior = compute_asset_features(changed, _config()).filter(pl.col("date") < dates[-1])

    assert_allclose(
        baseline.filter(pl.col("date") < dates[-1])
        .select("mom_21d", "mom_126_21d", "near_52w_high")
        .to_numpy(),
        prior.select("mom_21d", "mom_126_21d", "near_52w_high").to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )


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


def test_fip_bollinger_bandwidth_and_ratio_match_trailing_definitions() -> None:
    config = _config()
    features = compute_asset_features(_asset_ohlcv(), config).filter(
        pl.col("ticker") == "AAA"
    )
    source = _asset_ohlcv().filter(pl.col("ticker") == "AAA").with_columns(
        pl.col("close").pct_change().alias("return")
    )
    expected = source.with_columns(
        pl.col("return")
        .sign()
        .fill_null(0.0)
        .rolling_sum(window_size=config.fip_window, min_samples=2)
        .truediv(config.fip_window**0.5)
        .alias("fip_expected"),
        pl.col("close")
        .rolling_mean(window_size=config.bollinger_window, min_samples=2)
        .alias("middle"),
        pl.col("close")
        .rolling_std(window_size=config.bollinger_window, min_samples=2)
        .alias("std"),
    ).with_columns(
        (
            2.0
            * config.bollinger_std_multiplier
            * pl.col("std")
            / (pl.col("middle").abs() + 1e-9)
        ).alias("bandwidth_expected")
    ).with_columns(
        (
            pl.col("fip_expected")
            / (pl.col("bandwidth_expected") + 1e-9)
        ).alias("ratio_expected")
    )

    actual = features.select(
        "frog_in_the_pan", "bollinger_bandwidth", "fip_over_bollinger_bandwidth"
    )
    expected = expected.select(
        "fip_expected", "bandwidth_expected", "ratio_expected"
    )
    assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=True
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
