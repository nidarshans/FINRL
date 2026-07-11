"""Panel conversion and supervised target construction for LightGBM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.features.columns import DirectAllocationRoutingMetadata
from finrl.features.panels import AssetFeaturePanel
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class GBTTrainingData:
    """Flattened pooled observations and schedule-aligned labels."""

    features: Array
    targets: Array
    n_times: int
    n_assets: int


def build_forward_return_targets(
    returns: Array,
    horizons: tuple[int, ...],
    weights: tuple[float, ...] | None = None,
) -> Array:
    """Build weighted compounded forward-return labels without look-ahead.

    Row ``t`` uses only returns from ``t`` through ``t + horizon - 1``. Rows
    whose longest horizon would exceed the available sample are dropped.
    """

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite [time, assets] array.")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive values.")
    if tuple(sorted(horizons)) != tuple(horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be strictly increasing and unique.")
    target_weights = np.ones(len(horizons), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if target_weights.shape != (len(horizons),) or not np.isfinite(target_weights).all() or np.any(target_weights < 0.0) or target_weights.sum() <= 0.0:
        raise ValueError("weights must be finite, non-negative, and match horizons.")
    target_weights /= target_weights.sum()
    max_horizon = horizons[-1]
    n_rows = values.shape[0] - max_horizon + 1
    if n_rows <= 0:
        raise ValueError("returns must contain at least the maximum horizon row count.")
    targets = np.zeros((n_rows, values.shape[1]), dtype=np.float32)
    for horizon, weight in zip(horizons, target_weights):
        compounded = np.stack(
            [np.prod(1.0 + values[index : index + horizon], axis=0) - 1.0 for index in range(n_rows)]
        )
        targets += (weight * compounded).astype(np.float32)
    return targets


def routed_feature_tensor(panel: AssetFeaturePanel,
                          routing: DirectAllocationRoutingMetadata) -> Array:
    """Select the exact routed features without using other panel columns."""

    expected = routing.direct_allocation_feature_names
    actual = tuple(panel.feature_columns[i] for i in routing.direct_allocation_indices)
    if actual != expected:
        raise ValueError("Routed feature order does not match the required allowlist.")
    values = np.take(panel.values, routing.direct_allocation_indices, axis=2)
    if not np.isfinite(values).all():
        raise ValueError("Routed features must contain only finite values.")
    return values.astype(np.float32, copy=False)


def build_training_data(panel: AssetFeaturePanel, target_returns: Array,
                        routing: DirectAllocationRoutingMetadata) -> GBTTrainingData:
    """Build pooled features and weighted forward-return labels."""

    features = routed_feature_tensor(panel, routing)
    targets = np.asarray(target_returns, dtype=np.float32)
    expected_shape = features.shape[:2]
    if targets.shape != expected_shape:
        raise ValueError("target_returns must have shape [time, assets].")
    if not np.isfinite(targets).all():
        raise ValueError("Target returns must contain only finite values.")
    return GBTTrainingData(
        features.reshape(-1, features.shape[-1]),
        targets.reshape(-1),
        features.shape[0],
        features.shape[1],
    )


def build_prediction_matrix(panel: AssetFeaturePanel,
                            routing: DirectAllocationRoutingMetadata) -> Array:
    """Flatten a routed panel while preserving time-major, asset-minor order."""

    values = routed_feature_tensor(panel, routing)
    return values.reshape(-1, values.shape[-1])
