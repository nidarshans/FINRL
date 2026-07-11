"""Tests for explicit direct-allocation feature routing."""

from __future__ import annotations

import pytest

from finrl.features.columns import (
    DIRECT_ALLOCATION_FEATURE_COLUMNS,
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
