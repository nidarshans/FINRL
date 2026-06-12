"""Experiment artifact containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finrl.backtest.walk_forward import WalkForwardSplit
from finrl.features.columns import FeatureRoutingMetadata
from finrl.features.preprocessing import FittedPreprocessor
from finrl.features.schema import FeatureBundle
from finrl.models.windows import LookbackWindows
from finrl.ppo.flax_trainer import ProductionPPOTrainState, ProductionPPOTrainingResult


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
    train_windows: LookbackWindows
    test_windows: LookbackWindows
    train_spy_returns: np.ndarray | None
    feature_routing: FeatureRoutingMetadata | None = None
    production_ppo_training: ProductionPPOTrainingResult | None = None
    production_policy_state: ProductionPPOTrainState | None = None
