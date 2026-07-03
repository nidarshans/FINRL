"""LightGBM wrapper for pooled cross-sectional forward-return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from finrl.features.columns import DirectAllocationRoutingMetadata
from finrl.features.panels import AssetFeaturePanel
from finrl.gbt.config import GBTConfig
from finrl.gbt.dataset import build_prediction_matrix, build_training_data
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class GBTModel:
    """Fitted model with the feature and shape contract needed for inference."""

    estimator: Any
    config: GBTConfig
    feature_names: tuple[str, ...]
    seed: int


def _lightgbm_regressor(**parameters: object) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("LightGBM is required for the GBT policy.") from exc
    return LGBMRegressor(**parameters)


def fit_gbt_model(panel: AssetFeaturePanel, target_returns: Array,
                  routing: DirectAllocationRoutingMetadata,
                  config: GBTConfig, seed: int) -> GBTModel:
    """Fit one deterministic pooled LightGBM regressor."""

    data = build_training_data(panel, target_returns, routing)
    estimator = _lightgbm_regressor(**config.model_parameters(seed))
    estimator.fit(data.features, data.targets)
    return GBTModel(estimator, config, routing.direct_allocation_feature_names, seed)


def predict_scores(model: GBTModel, panel: AssetFeaturePanel,
                   routing: DirectAllocationRoutingMetadata) -> Array:
    """Predict a score for each time/asset observation."""

    if routing.direct_allocation_feature_names != model.feature_names:
        raise ValueError("Prediction feature names do not match the fitted model.")
    matrix = build_prediction_matrix(panel, routing)
    predictor = getattr(model.estimator, "booster_", model.estimator)
    scores = np.asarray(predictor.predict(matrix), dtype=np.float64)
    if scores.shape != (panel.values.shape[0] * panel.values.shape[1],):
        raise ValueError("LightGBM returned an unexpected prediction shape.")
    if not np.isfinite(scores).all():
        raise ValueError("LightGBM returned non-finite predictions.")
    return scores.reshape(panel.values.shape[:2]).astype(np.float32)
