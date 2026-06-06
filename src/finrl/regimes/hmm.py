"""Diagonal Gaussian HMM fitting for market regime detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray

from finrl.features.splitsafe import FitWindow
from finrl.regimes.filtering import (
    _as_2d_float_array,
    filter_regime_probabilities,
)
from finrl.regimes.schema import FittedHMM, FloatArray, HMMConfig, HMMMetadata


def _train_window_from_config(config: HMMConfig) -> FitWindow | None:
    if config.train_start is None or config.train_end is None:
        return None
    return FitWindow(start=config.train_start, end=config.train_end)


def _diag_variances(covariances: NDArray[np.floating], config: HMMConfig) -> FloatArray:
    covars = np.asarray(covariances, dtype=np.float64)
    if covars.ndim == 2:
        variances = covars
    elif covars.ndim == 3:
        variances = np.diagonal(covars, axis1=1, axis2=2)
    else:
        raise ValueError("Unsupported hmmlearn covariance shape.")
    return np.maximum(variances, config.min_covar)


def _normalize_rows(values: NDArray[np.floating]) -> FloatArray:
    rows = np.asarray(values, dtype=np.float64)
    rows = np.maximum(rows, np.finfo(np.float64).tiny)
    return rows / rows.sum(axis=1, keepdims=True)


def _normalize_vector(values: NDArray[np.floating]) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    vector = np.maximum(vector, np.finfo(np.float64).tiny)
    return vector / vector.sum()


def fit_hmm(
    train_phi: NDArray[np.floating] | list[list[float]],
    config: HMMConfig,
) -> FittedHMM:
    """Fit a diagonal Gaussian HMM with ``hmmlearn`` on train states only."""

    observations = _as_2d_float_array(train_phi)
    if observations.shape[0] < config.n_states:
        raise ValueError("HMM fitting requires at least n_states observations.")

    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.max_iter,
        tol=config.tol,
        min_covar=config.min_covar,
        random_state=config.random_seed,
    )
    model.fit(observations)
    variances = _diag_variances(model.covars_, config)

    metadata = HMMMetadata(
        n_observations=observations.shape[0],
        n_features=observations.shape[1],
        n_states=config.n_states,
        covariance_type=config.covariance_type,
        train_window=_train_window_from_config(config),
        converged=bool(model.monitor_.converged),
        n_iter=int(model.monitor_.iter),
        log_likelihood=float(model.monitor_.history[-1]),
    )
    return FittedHMM(
        initial_probabilities=_normalize_vector(model.startprob_),
        transition_matrix=_normalize_rows(model.transmat_),
        means=np.asarray(model.means_, dtype=np.float64),
        variances=variances,
        metadata=metadata,
    )


def annual_hmm_refit(
    train_phi_by_split: Mapping[object, NDArray[np.floating] | list[list[float]]]
    | Iterable[NDArray[np.floating] | list[list[float]]],
    config: HMMConfig,
) -> tuple[FittedHMM, ...]:
    """Fit one frozen HMM per walk-forward train split."""

    sequences = (
        train_phi_by_split.values()
        if isinstance(train_phi_by_split, Mapping)
        else train_phi_by_split
    )
    return tuple(fit_hmm(train_phi, config) for train_phi in sequences)


__all__ = [
    "FittedHMM",
    "HMMConfig",
    "annual_hmm_refit",
    "filter_regime_probabilities",
    "fit_hmm",
]
