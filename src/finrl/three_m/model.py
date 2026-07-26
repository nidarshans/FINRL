"""Scikit-learn backed pooled classifiers for 3M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from finrl.features.panels import AssetFeaturePanel
from finrl.three_m.config import TreeConfig
from finrl.three_m.dataset import ThreeMTrainingData


@dataclass(frozen=True, slots=True)
class ThreeMModel:
    """Three fitted shared classifiers and their feature contract."""

    buy_estimator: Any
    hold_estimator: Any
    sell_estimator: Any
    feature_names: tuple[str, ...]
    config: TreeConfig
    seed: int


@dataclass(frozen=True, slots=True)
class ThreeMProbabilities:
    """Per-date, per-asset action probabilities."""

    buy: np.ndarray
    hold: np.ndarray
    sell: np.ndarray


def _classifier(config: TreeConfig, seed: int) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("scikit-learn is required for the 3M policy.") from exc
    return HistGradientBoostingClassifier(**config.estimator_parameters(seed))


def fit_model(
    data: ThreeMTrainingData,
    feature_names: tuple[str, ...],
    config: TreeConfig,
    seed: int,
) -> ThreeMModel:
    """Fit deterministic shared buy, hold, and sell classifiers."""

    if data.features.ndim != 2 or data.features.shape[1] != len(feature_names):
        raise ValueError("Feature names must match the pooled training matrix.")
    if not np.isfinite(data.features).all():
        raise ValueError("Training features must be finite.")
    estimators: list[Any] = []
    for offset, (target, weights) in enumerate(
        (
            (data.buy_targets, data.buy_sample_weight),
            (data.hold_targets, data.hold_sample_weight),
            (data.sell_targets, data.sell_sample_weight),
        )
    ):
        estimator = _classifier(config, seed + offset)
        estimator.fit(data.features, target, sample_weight=weights)
        estimators.append(estimator)
    return ThreeMModel(
        buy_estimator=estimators[0],
        hold_estimator=estimators[1],
        sell_estimator=estimators[2],
        feature_names=feature_names,
        config=config,
        seed=seed,
    )


def predict_probabilities(model: ThreeMModel, panel: AssetFeaturePanel) -> ThreeMProbabilities:
    """Return pooled classifier probabilities reshaped to the asset panel."""

    if panel.feature_columns != model.feature_names:
        raise ValueError("Prediction feature names do not match the fitted 3M model.")
    values = np.asarray(panel.values, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Panel values must be finite [time, assets, features].")
    matrix = values.reshape(-1, values.shape[-1])

    def predict(estimator: Any) -> np.ndarray:
        probability = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
        if probability.shape != (matrix.shape[0], 2):
            raise ValueError("3M classifier returned unexpected probabilities.")
        return probability[:, 1].reshape(values.shape[:2]).astype(np.float32)

    return ThreeMProbabilities(
        buy=predict(model.buy_estimator),
        hold=predict(model.hold_estimator),
        sell=predict(model.sell_estimator),
    )
