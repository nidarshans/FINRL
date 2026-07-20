"""Tests for explicit direct-allocation feature routing."""

from __future__ import annotations

import pytest

from finrl.features.columns import (
    DIRECT_ALLOCATION_FEATURE_COLUMNS,
    feature_set_config,
    selected_direct_allocation_indices,
)


def _feature_columns() -> tuple[str, ...]:
    return (
        "date",
        "return",
        "future_return_leakage",
        "acc_macd_signal",
        "mr_ewma50_vol_gap",
        "acc_klinger_signal",
        "ewma50_slope",
        "macd_signal_strength",
        "klinger_signal_strength",
        "acc_momentum_quality",
        "cmf",
        "cmf_slope",
        "cmf_cross_signal",
        "cmf_days_since_cross",
        "frog_in_the_pan",
        "bollinger_bandwidth",
        "fip_over_bollinger_bandwidth",
        "mr_gap_with_rising_ewma",
        "mr_gap_with_falling_ewma",
        "cmf_regime_age",
        "ewma50_slope_up",
        "ewma50_slope_down",
        "unrouted_diagnostic",
    )


def test_direct_allocation_feature_columns_are_explicit() -> None:
    assert DIRECT_ALLOCATION_FEATURE_COLUMNS == (
        "mr_ewma50_vol_gap",
        "ewma50_slope",
        "acc_macd_signal",
        "acc_klinger_signal",
        "macd_signal_strength",
        "klinger_signal_strength",
        "acc_momentum_quality",
        "cmf",
        "cmf_slope",
        "cmf_cross_signal",
        "cmf_days_since_cross",
        "frog_in_the_pan",
        "bollinger_bandwidth",
        "fip_over_bollinger_bandwidth",
    )


def test_selected_direct_allocation_indices_use_exact_allowlist() -> None:
    columns = _feature_columns()
    routing = selected_direct_allocation_indices(columns)

    assert routing.direct_allocation_feature_names == DIRECT_ALLOCATION_FEATURE_COLUMNS
    assert routing.direct_allocation_indices == tuple(
        columns.index(name) for name in DIRECT_ALLOCATION_FEATURE_COLUMNS
    )


def test_missing_direct_allocation_column_raises() -> None:
    columns = tuple(
        column for column in _feature_columns() if column != "acc_macd_signal"
    )

    with pytest.raises(ValueError, match="acc_macd_signal"):
        selected_direct_allocation_indices(columns)


def test_unrouted_columns_are_excluded() -> None:
    columns = _feature_columns()
    routing = selected_direct_allocation_indices(columns)
    selected = {
        columns[index] for index in routing.direct_allocation_indices
    }

    assert "future_return_leakage" not in selected
    assert "unrouted_diagnostic" not in selected
    assert "mr_gap_with_rising_ewma" not in selected
    assert "mr_gap_with_falling_ewma" not in selected
    assert "cmf_regime_age" not in selected
    assert "ewma50_slope_up" not in selected
    assert "ewma50_slope_down" not in selected


def test_named_momentum_feature_set_has_an_exact_ordered_allowlist() -> None:
    feature_set = feature_set_config("baseline_plus_momentum")

    assert feature_set.routed_columns[-3:] == (
        "mom_21d",
        "mom_126_21d",
        "near_52w_high",
    )


def test_named_liquidity_feature_set_has_an_exact_ordered_allowlist() -> None:
    feature_set = feature_set_config("baseline_plus_liquidity")

    assert feature_set.routed_columns[-3:] == (
        "log_adv_20",
        "volume_z_20",
        "amihud_20",
    )


def test_named_structure_feature_set_has_an_exact_ordered_allowlist() -> None:
    feature_set = feature_set_config("baseline_plus_structure")

    assert feature_set.routed_columns[-5:] == (
        "confirmed_structure_score",
        "support_distance_atr",
        "resistance_distance_atr",
        "swing_avwap_distance_atr",
        "bars_since_swing_low",
    )


def test_named_market_relative_feature_set_has_an_exact_ordered_allowlist() -> None:
    feature_set = feature_set_config("baseline_plus_market_relative")

    assert feature_set.routed_columns[-4:] == (
        "relative_strength_63",
        "beta_252",
        "residual_mom_126_21",
        "idio_vol_60",
    )


def test_institutional_core_feature_set_contains_risk_features() -> None:
    feature_set = feature_set_config("institutional_core_v1")

    assert feature_set.routed_columns[-7:-3] == (
        "natr_20",
        "realized_vol_20",
        "downside_vol_60",
        "max_drawdown_126",
    )


def test_volume_ema_feature_set_has_explicit_order() -> None:
    feature_set = feature_set_config("baseline_plus_volume_ema")

    assert feature_set.routed_columns[-4:] == (
        "volume_z_20",
        "close_ema20_gap",
        "close_ema50_gap",
        "close_ema200_gap",
    )
