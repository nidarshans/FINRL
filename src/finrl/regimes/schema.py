"""Schemas for Gaussian HMM regime detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from finrl.features.splitsafe import FitWindow

FloatArray = NDArray[np.float64]
CovarianceType = Literal["diag"]


@dataclass(frozen=True, slots=True)
class HMMConfig:
    """Configuration for a diagonal-covariance Gaussian HMM."""

    n_states: int = 4
    covariance_type: CovarianceType = "diag"
    max_iter: int = 50
    tol: float = 1e-4
    min_covar: float = 1e-6
    random_seed: int = 0
    train_start: date | None = None
    train_end: date | None = None

    def __post_init__(self) -> None:
        if self.n_states <= 0:
            raise ValueError("n_states must be positive.")
        if self.covariance_type != "diag":
            raise ValueError("Only diagonal covariance HMMs are supported.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.min_covar <= 0.0:
            raise ValueError("min_covar must be positive.")
        if (self.train_start is None) != (self.train_end is None):
            raise ValueError("train_start and train_end must be provided together.")
        if self.train_start is not None and self.train_end is not None:
            if self.train_start > self.train_end:
                raise ValueError("train_start must be on or before train_end.")


@dataclass(frozen=True, slots=True)
class HMMMetadata:
    """Metadata that documents how an HMM was fitted."""

    n_observations: int
    n_features: int
    n_states: int
    covariance_type: CovarianceType
    train_window: FitWindow | None
    converged: bool
    n_iter: int
    log_likelihood: float
    filtering_only: bool = True


@dataclass(frozen=True, slots=True)
class FittedHMM:
    """Parameters of a fitted diagonal Gaussian HMM."""

    initial_probabilities: FloatArray
    transition_matrix: FloatArray
    means: FloatArray
    variances: FloatArray
    metadata: HMMMetadata

