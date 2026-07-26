"""Chronological fit-and-predict runner for one frozen 3M split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.features.panels import AssetFeaturePanel
from finrl.three_m.allocation import allocate_actions
from finrl.three_m.config import LabelConfig, PolicyConfig, TreeConfig
from finrl.three_m.dataset import build_training_data
from finrl.three_m.labels import LabelInputs, build_targets
from finrl.three_m.model import ThreeMModel, ThreeMProbabilities, fit_model, predict_probabilities
from finrl.three_m.policy import decide_actions


@dataclass(frozen=True, slots=True)
class ThreeMSplitOutput:
    """Frozen model and chronological test-time policy outputs."""

    model: ThreeMModel
    probabilities: ThreeMProbabilities
    target_weights: np.ndarray
    actions: np.ndarray


def _evolve_weights(weights: np.ndarray, risky_returns: np.ndarray) -> np.ndarray:
    """Evolve one executed allocation to the next decision state."""

    returns = np.asarray(risky_returns, dtype=np.float64)
    if returns.shape != (weights.size - 1,) or not np.isfinite(returns).all():
        raise ValueError("test_execution_returns must be finite [time, assets].")
    values = np.asarray(weights, dtype=np.float64) * np.concatenate((1.0 + returns, [1.0]))
    if np.any(values < 0.0) or not np.isfinite(values).all() or values.sum() <= 0.0:
        raise ValueError("Execution returns produced an invalid portfolio state.")
    return values / values.sum()


def fit_predict_split(
    train_panel: AssetFeaturePanel,
    test_panel: AssetFeaturePanel,
    train_execution_returns: np.ndarray,
    train_label_inputs: LabelInputs,
    test_execution_returns: np.ndarray,
    tree_config: TreeConfig,
    label_config: LabelConfig,
    policy_config: PolicyConfig,
    seed: int,
    *,
    train_feature_valid_mask: np.ndarray | None = None,
    test_feature_valid_mask: np.ndarray | None = None,
    test_tradable_mask: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
) -> ThreeMSplitOutput:
    """Fit on complete train labels, then execute a frozen test policy causally."""

    if train_panel.feature_columns != test_panel.feature_columns:
        raise ValueError("Train and test panels must use identical feature ordering.")
    train_targets = build_targets(train_execution_returns, train_label_inputs, label_config)
    training_data = build_training_data(
        train_panel, train_targets, feature_valid_mask=train_feature_valid_mask
    )
    model = fit_model(training_data, train_panel.feature_columns, tree_config, seed)
    probabilities = predict_probabilities(model, test_panel)
    test_returns = np.asarray(test_execution_returns, dtype=np.float64)
    expected_returns_shape = test_panel.values.shape[:2]
    if test_returns.shape != expected_returns_shape:
        raise ValueError("test_execution_returns must match the test panel [time, assets].")
    valid = np.ones(expected_returns_shape, dtype=bool)
    for supplied, name in (
        (test_feature_valid_mask, "test_feature_valid_mask"),
        (test_tradable_mask, "test_tradable_mask"),
    ):
        if supplied is not None:
            mask = np.asarray(supplied, dtype=bool)
            if mask.shape != expected_returns_shape:
                raise ValueError(f"{name} must match the test panel [time, assets].")
            valid &= mask
    if test_panel.tradable_mask is not None:
        tradable = np.asarray(test_panel.tradable_mask, dtype=bool)
        if tradable.shape != expected_returns_shape:
            raise ValueError("test panel tradable_mask must match [time, assets].")
        valid &= tradable
    n_assets = expected_returns_shape[1]
    current = (
        np.concatenate((np.zeros(n_assets, dtype=np.float64), [1.0]))
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64).copy()
    )
    if current.shape != (n_assets + 1,) or np.any(current < 0.0) or not np.isclose(current.sum(), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("initial_weights must be a long-only [assets + cash] simplex.")
    weights = np.empty((expected_returns_shape[0], n_assets + 1), dtype=np.float32)
    actions = np.empty(expected_returns_shape, dtype=np.int8)
    for time_index in range(expected_returns_shape[0]):
        row_probabilities = ThreeMProbabilities(
            buy=probabilities.buy[time_index],
            hold=probabilities.hold[time_index],
            sell=probabilities.sell[time_index],
        )
        decision = decide_actions(
            current, row_probabilities, policy_config, tradable_mask=valid[time_index]
        )
        allocation = allocate_actions(
            current, decision, row_probabilities.buy, policy_config
        )
        weights[time_index] = allocation.target_weights
        actions[time_index] = decision.actions
        current = _evolve_weights(allocation.target_weights, test_returns[time_index])
    return ThreeMSplitOutput(
        model=model,
        probabilities=probabilities,
        target_weights=weights,
        actions=actions,
    )
