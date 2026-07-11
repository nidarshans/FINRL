"""Tests for the LightGBM prediction and deterministic allocation policy."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from finrl.features.columns import (
    DIRECT_ALLOCATION_FEATURE_COLUMNS,
    selected_direct_allocation_indices,
)
from finrl.features.panels import AssetFeaturePanel
from finrl.gbt import GBTConfig, build_forward_return_targets, build_training_data, fit_gbt_model, predict_scores, scores_to_weights


FEATURES = ("unused", *DIRECT_ALLOCATION_FEATURE_COLUMNS)


def _panel() -> AssetFeaturePanel:
    values = np.arange(6 * 2 * len(FEATURES), dtype=np.float32).reshape(
        6, 2, len(FEATURES)
    ) / 100.0
    return AssetFeaturePanel(values, tuple(range(6)), ("AAA", "BBB"), FEATURES)


def test_config_rejects_invalid_allocator_parameters() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GBTConfig(temperature=0.0)
    with pytest.raises(ValueError, match="score_clip"):
        GBTConfig(score_clip=0.0)
    with pytest.raises(ValueError, match="target_weights"):
        GBTConfig(target_weights=(1.0, 1.0))


def test_training_data_routes_features_and_uses_weighted_targets() -> None:
    panel = _panel()
    routing = selected_direct_allocation_indices(panel.feature_columns)
    returns = np.arange(12, dtype=np.float32).reshape(6, 2) / 100.0

    data = build_training_data(panel, returns, routing)

    assert data.features.shape == (12, len(DIRECT_ALLOCATION_FEATURE_COLUMNS))
    assert data.targets.shape == (12,)
    assert data.n_times == 6
    assert_allclose(data.targets.reshape(6, 2), returns)


def test_forward_targets_use_only_available_forward_rows() -> None:
    returns = np.array([[0.10], [0.20], [0.30], [0.40]], dtype=np.float32)
    targets = build_forward_return_targets(returns, (1, 2), (1.0, 1.0))

    assert targets.shape == (3, 1)
    expected = np.array([
        (0.10 + (1.10 * 1.20 - 1.0)) / 2.0,
        (0.20 + (1.20 * 1.30 - 1.0)) / 2.0,
        (0.30 + (1.30 * 1.40 - 1.0)) / 2.0,
    ], dtype=np.float32)
    assert_allclose(targets[:, 0], expected, rtol=1e-6)


def test_forward_targets_reject_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        build_forward_return_targets(np.zeros((3, 1)), (2, 1))


def test_unrouted_feature_cannot_change_training_matrix() -> None:
    panel = _panel()
    routing = selected_direct_allocation_indices(panel.feature_columns)
    changed = panel.values.copy()
    changed[..., 0] = 9999.0
    other = AssetFeaturePanel(changed, panel.decision_dates, panel.tickers, panel.feature_columns)
    returns = np.zeros((6, 2), dtype=np.float32)

    assert_allclose(
        build_training_data(panel, returns, routing).features,
        build_training_data(other, returns, routing).features,
    )


def test_softmax_allocator_is_long_only_and_appends_zero_cash() -> None:
    weights = scores_to_weights(np.array([[1.0, 2.0], [0.0, 0.0]]), GBTConfig())

    assert weights.shape == (2, 3)
    assert np.all(weights >= 0.0)
    assert_allclose(weights.sum(axis=1), 1.0)
    assert_allclose(weights[:, -1], 0.0, atol=0.0)
    assert_allclose(weights[1, :2], [0.5, 0.5])


def test_softmax_allocator_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        scores_to_weights(np.array([[np.nan, 0.0]]), GBTConfig())


def test_softmax_allocator_applies_position_cap() -> None:
    weights = scores_to_weights(
        np.array([[10.0, 0.0, 0.0]]), GBTConfig(max_position_weight=0.5)
    )
    assert np.all(weights[:, :-1] <= 0.5 + 1e-6)
    assert_allclose(weights.sum(axis=1), 1.0)


def test_lightgbm_model_fits_and_predicts_deterministically() -> None:
    panel = _panel()
    routing = selected_direct_allocation_indices(panel.feature_columns)
    returns = np.array([[0.01, -0.01], [0.02, -0.02], [0.03, -0.01],
                        [0.04, 0.00], [0.05, 0.01], [0.06, 0.02]], dtype=np.float32)
    config = GBTConfig(n_estimators=10, min_child_samples=2)

    first = fit_gbt_model(panel, returns, routing, config, seed=7)
    second = fit_gbt_model(panel, returns, routing, config, seed=7)

    first_scores = predict_scores(first, panel, routing)
    second_scores = predict_scores(second, panel, routing)
    assert first_scores.shape == (6, 2)
    assert_allclose(first_scores, second_scores, atol=0.0)
