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
