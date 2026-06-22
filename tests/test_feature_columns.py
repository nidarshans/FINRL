"""Tests for explicit score-head feature routing."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from finrl.features.columns import (
    ACCUMULATION_FEATURE_COLUMNS,
    LIQUIDITY_EXIT_FEATURE_COLUMNS,
    selected_feature_indices,
)
from finrl.models.asset_encoder import component_indices, slice_score_head_components


def _feature_columns() -> tuple[str, ...]:
    return (
        "date",
        "return",
        "acc_forbidden_leakage",
        *ACCUMULATION_FEATURE_COLUMNS[:3],
        "liq_amihud_illiquidity",
        *LIQUIDITY_EXIT_FEATURE_COLUMNS[:2],
        "future_return_leakage",
        *ACCUMULATION_FEATURE_COLUMNS[3:],
        *LIQUIDITY_EXIT_FEATURE_COLUMNS[2:],
        "liq_forbidden_extra",
    )


def test_selected_feature_indices_use_exact_allowlists() -> None:
    columns = _feature_columns()
    routing = selected_feature_indices(columns)

    assert routing.accumulation_feature_names == ACCUMULATION_FEATURE_COLUMNS
    assert routing.liquidity_exit_feature_names == LIQUIDITY_EXIT_FEATURE_COLUMNS
    assert routing.accumulation_indices == tuple(
        columns.index(name) for name in ACCUMULATION_FEATURE_COLUMNS
    )
    assert routing.liquidity_exit_indices == tuple(
        columns.index(name) for name in LIQUIDITY_EXIT_FEATURE_COLUMNS
    )


def test_component_index_order_is_stable() -> None:
    columns = _feature_columns()
    acc_indices, liq_indices = component_indices(columns)

    assert tuple(columns[index] for index in acc_indices) == ACCUMULATION_FEATURE_COLUMNS
    assert tuple(columns[index] for index in liq_indices) == LIQUIDITY_EXIT_FEATURE_COLUMNS


def test_selected_feature_indices_raise_for_missing_columns() -> None:
    columns = tuple(
        column for column in _feature_columns() if column != "acc_macd_improvement"
    )

    with pytest.raises(ValueError, match="acc_macd_improvement"):
        selected_feature_indices(columns)


def test_prefix_matched_leakage_columns_are_excluded() -> None:
    columns = _feature_columns()
    routing = selected_feature_indices(columns)
    selected = {
        *(columns[index] for index in routing.accumulation_indices),
        *(columns[index] for index in routing.liquidity_exit_indices),
    }

    assert "acc_forbidden_leakage" not in selected
    assert "liq_forbidden_extra" not in selected
    assert "liq_amihud_illiquidity" not in selected
    assert "future_return_leakage" not in selected


def test_encoder_score_head_slices_have_expected_shapes() -> None:
    columns = _feature_columns()
    routing = selected_feature_indices(columns)
    windows = jnp.arange(2 * 4 * 3 * len(columns), dtype=jnp.float32).reshape(
        2,
        4,
        3,
        len(columns),
    )

    accumulation, liquidity = slice_score_head_components(
        windows,
        routing.accumulation_indices,
        routing.liquidity_exit_indices,
    )

    assert accumulation.shape == (2, 4, 3, 11)
    assert liquidity.shape == (2, 4, 3, 6)
