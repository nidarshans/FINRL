"""Pooled, feature-valid 3M supervised training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.features.panels import AssetFeaturePanel
from finrl.three_m.labels import ThreeMTargets


@dataclass(frozen=True, slots=True)
class ThreeMTrainingData:
    """Flattened date-major observations and separate binary targets."""

    features: np.ndarray
    buy_targets: np.ndarray
    hold_targets: np.ndarray
    sell_targets: np.ndarray
    buy_sample_weight: np.ndarray
    hold_sample_weight: np.ndarray
    sell_sample_weight: np.ndarray
    valid_mask: np.ndarray


def balanced_binary_sample_weights(target: np.ndarray) -> np.ndarray:
    """Return class-balanced weights for a binary target without resampling."""

    labels = np.asarray(target, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = labels.size - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Each 3M target needs both positive and negative observations.")
    weights = np.empty(labels.shape, dtype=np.float32)
    weights[labels] = labels.size / (2.0 * positive_count)
    weights[~labels] = labels.size / (2.0 * negative_count)
    return weights


def build_training_data(
    panel: AssetFeaturePanel,
    targets: ThreeMTargets,
    feature_valid_mask: np.ndarray | None = None,
) -> ThreeMTrainingData:
    """Flatten complete, finite, tradable observations across assets and dates."""

    values = np.asarray(panel.values, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Panel values must be finite [time, assets, features].")
    if targets.n_times > values.shape[0] or targets.buy.shape != (targets.n_times, values.shape[1]):
        raise ValueError("Targets must align to the leading panel dates and assets.")
    valid = np.ones((targets.n_times, values.shape[1]), dtype=bool)
    if targets.valid_mask is not None:
        target_valid = np.asarray(targets.valid_mask, dtype=bool)
        if target_valid.shape != valid.shape:
            raise ValueError("Target valid_mask must match target [time, assets].")
        valid &= target_valid
    if feature_valid_mask is not None:
        supplied = np.asarray(feature_valid_mask, dtype=bool)
        if supplied.shape != values.shape[:2]:
            raise ValueError("feature_valid_mask must match panel [time, assets].")
        valid &= supplied[: targets.n_times]
    if panel.tradable_mask is not None:
        tradable = np.asarray(panel.tradable_mask, dtype=bool)
        if tradable.shape != values.shape[:2]:
            raise ValueError("tradable_mask must match panel [time, assets].")
        valid &= tradable[: targets.n_times]
    selected = valid.reshape(-1)
    if not selected.any():
        raise ValueError("No feature-valid, tradable training observations.")
    features = values[: targets.n_times].reshape(-1, values.shape[-1])[selected]
    buy = targets.buy.reshape(-1)[selected]
    hold = targets.hold.reshape(-1)[selected]
    sell = targets.sell.reshape(-1)[selected]
    return ThreeMTrainingData(
        features=features,
        buy_targets=buy,
        hold_targets=hold,
        sell_targets=sell,
        buy_sample_weight=balanced_binary_sample_weights(buy),
        hold_sample_weight=balanced_binary_sample_weights(hold),
        sell_sample_weight=balanced_binary_sample_weights(sell),
        valid_mask=valid,
    )
