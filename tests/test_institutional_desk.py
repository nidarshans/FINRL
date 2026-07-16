"""Tests for the causal institutional research desk."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from finrl.research.institutional_desk import (
    DeskConfig,
    analyze_ticker,
    compute_desk_features,
    create_desk_chart,
    rank_universe,
)


def _config() -> DeskConfig:
    return DeskConfig(
        period="1y",
        chart_bars=8,
        ema_fast_span=2,
        ema_mid_span=3,
        ema_slow_span=4,
        atr_window=2,
        realized_vol_window=2,
        downside_vol_window=3,
        drawdown_window=4,
        momentum_short_window=2,
        momentum_medium_window=4,
        momentum_skip_window=1,
        high_lookback=5,
        volume_window=2,
        swing_left=1,
        swing_right=1,
        liquidity_lookback=5,
        zone_lookback=5,
        stress_lookback=4,
        zone_cluster_atr=0.5,
        zone_half_width_atr=0.25,
        max_trigger_distance_atr=10.0,
        max_stop_distance_atr=2.0,
        max_risk_fraction=0.5,
    )


def _ohlcv() -> pd.DataFrame:
    closes = np.array(
        [10.0, 11.0, 12.0, 10.5, 13.0, 11.5, 14.0, 12.5, 15.0, 13.5, 16.0, 14.5]
    )
    return pd.DataFrame(
        {
            "Open": closes - 0.2,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": np.arange(100.0, 112.0) * 1_000.0,
        },
        index=pd.bdate_range(date(2024, 1, 2), periods=len(closes)),
    )


def test_true_range_includes_overnight_gap() -> None:
    data = _ohlcv()
    second = data.index[1]
    data.loc[second, ["Open", "High", "Low", "Close"]] = [15.0, 15.5, 14.5, 15.0]

    features = compute_desk_features(data, _config())

    assert features.loc[second, "true_range"] == 5.5


def test_swing_is_published_on_confirmation_date() -> None:
    features = compute_desk_features(_ohlcv(), _config())
    pivot_date = features.index[2]
    confirmation_date = features.index[3]

    assert bool(features.loc[pivot_date, "swing_high_pivot_visual"])
    assert np.isnan(features.loc[pivot_date, "confirmed_swing_high_event"])
    assert features.loc[confirmation_date, "confirmed_swing_high_event"] == 13.0
    assert features.loc[confirmation_date, "confirmed_swing_high_pivot_pos"] == 2.0


def test_future_bar_change_cannot_change_prior_desk_features() -> None:
    base = _ohlcv()
    changed = base.copy()
    final_date = changed.index[-1]
    changed.loc[final_date, ["Open", "High", "Low", "Close", "Volume"]] = [
        24.0,
        25.0,
        23.0,
        24.5,
        900_000.0,
    ]

    base_features = compute_desk_features(base, _config())
    changed_features = compute_desk_features(changed, _config())
    causal_columns = [
        "structure_score",
        "market_structure",
        "support",
        "resistance",
        "swing_avwap",
        "stress_avwap",
        "bars_since_swing_high",
        "bars_since_swing_low",
    ]

    assert_frame_equal(
        base_features.iloc[:-1][causal_columns],
        changed_features.iloc[:-1][causal_columns],
    )


def test_swing_avwap_matches_pivot_to_confirmation_prices() -> None:
    data = _ohlcv()
    features = compute_desk_features(data, _config())
    confirmation_position = 4
    pivot_position = 3
    typical = (
        features["high"] + features["low"] + features["close"]
    ) / 3.0
    expected = np.average(
        typical.iloc[pivot_position : confirmation_position + 1],
        weights=features["volume"].iloc[pivot_position : confirmation_position + 1],
    )

    assert_allclose(
        features["swing_avwap"].iloc[confirmation_position],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_scenarios_use_r_multiple_targets_and_bounded_stop_distance() -> None:
    data = _ohlcv()

    analysis = analyze_ticker("AAA", _config(), downloader=lambda _ticker, _config: data)
    snapshot = analysis.snapshot
    long = snapshot.long_scenario
    bearish = snapshot.bearish_scenario

    assert long.reward_risk_1 == 2.0
    assert long.reward_risk_2 == 3.0
    assert_allclose(
        long.target_1 - long.trigger,
        2.0 * long.risk_per_share,
        atol=0.02,
    )
    assert long.risk_per_share <= 2.0 * snapshot.atr + 0.02
    assert_allclose(
        bearish.trigger - bearish.target_1,
        2.0 * bearish.risk_per_share,
        atol=0.02,
    )


def test_universe_ranking_is_cross_sectional_and_bounded() -> None:
    scan = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "momentum_21d": [0.3, 0.1, -0.2],
            "momentum_126_21d": [0.4, 0.0, -0.1],
            "near_52w_high": [-0.01, -0.10, -0.30],
            "trend_score": [1.0, 0.0, -1.0],
            "structure_score": [1.0, 0.0, -1.0],
            "realized_vol_20": [0.15, 0.25, 0.50],
            "max_drawdown_126": [-0.02, -0.10, -0.40],
            "log_adv_20": [20.0, 18.0, 16.0],
            "amihud_20": [1e-10, 2e-10, 5e-10],
        }
    )

    ranked = rank_universe(scan)

    assert ranked.iloc[0]["ticker"] == "AAA"
    for column in (
        "momentum_rank",
        "structure_rank",
        "risk_rank",
        "liquidity_rank",
        "research_priority",
    ):
        assert ranked[column].between(0.0, 1.0).all()


def test_inline_chart_contains_price_volume_and_risk_panels() -> None:
    analysis = analyze_ticker(
        "AAA",
        _config(),
        downloader=lambda _ticker, _config: _ohlcv(),
    )

    figure = create_desk_chart(analysis, _config(), show=False)

    trace_names = {trace.name for trace in figure.data}
    assert {"Price", "Daily volume", "Realized volatility", "Rolling drawdown"}.issubset(
        trace_names
    )
    assert figure.layout.xaxis.rangeslider.visible is False
