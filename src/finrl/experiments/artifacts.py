"""Experiment artifact containers."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from finrl.backtest.walk_forward import WalkForwardSplit
from finrl.features.columns import FeatureRoutingMetadata
from finrl.features.preprocessing import FittedPreprocessor
from finrl.features.panels import AssetFeaturePanel
from finrl.features.schema import FeatureBundle
from finrl.dpo_jax.losses import DPOLossMetrics
from finrl.dpo_jax.trainer import DPOTrainState


@dataclass(frozen=True, slots=True)
class RawExperimentData:
    """Prepared feature and return tables consumed by the runner."""

    features: FeatureBundle
    returns: pl.DataFrame
    spy_returns: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    """Frozen artifacts fitted for one walk-forward split."""

    split: WalkForwardSplit
    preprocessor: FittedPreprocessor
    train_features: AssetFeaturePanel
    test_features: AssetFeaturePanel
    feature_routing: FeatureRoutingMetadata | None = None
    dpo_policy_state: DPOTrainState | None = None
    dpo_train_metrics: tuple[DPOLossMetrics, ...] | None = None
