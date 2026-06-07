"""Experiment artifact containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finrl.backtest.walk_forward import WalkForwardSplit
from finrl.features.preprocessing import FittedPreprocessor
from finrl.features.schema import FeatureBundle
from finrl.models.windows import LookbackWindows
from finrl.models.encoder_training import EncoderTrainingResult
from finrl.ppo.flax_trainer import ProductionPPOTrainState, ProductionPPOTrainingResult
from finrl.ppo.trainer import PolicyCheckpoint, PPOTrainingResult
from finrl.regimes.schema import FittedHMM


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
    train_phi: np.ndarray
    test_phi: np.ndarray
    train_regime_probs: np.ndarray
    test_regime_probs: np.ndarray
    train_spy_returns: np.ndarray | None
    fitted_hmm: FittedHMM
    ppo_training: PPOTrainingResult | None
    policy_checkpoint: PolicyCheckpoint | None
    production_encoder_training: EncoderTrainingResult | None = None
    production_ppo_training: ProductionPPOTrainingResult | None = None
    production_policy_state: ProductionPPOTrainState | None = None
