"""Filtering-only inference for Gaussian HMM regime probabilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from finrl.regimes.schema import FittedHMM, FloatArray


def _as_2d_float_array(values: NDArray[np.floating] | list[list[float]]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("HMM observations must be a 2D array.")
    if array.shape[0] == 0:
        raise ValueError("HMM observations must contain at least one row.")
    if not np.isfinite(array).all():
        raise ValueError("HMM observations must be finite.")
    return array


def _logsumexp(values: FloatArray, axis: int | None = None) -> FloatArray:
    return np.asarray(logsumexp(values, axis=axis), dtype=np.float64)


def gaussian_log_likelihood(
    observations: NDArray[np.floating] | list[list[float]],
    means: FloatArray,
    variances: FloatArray,
) -> FloatArray:
    """Return per-observation log likelihoods for diagonal Gaussian states."""

    x = _as_2d_float_array(observations)
    if means.ndim != 2 or variances.ndim != 2:
        raise ValueError("means and variances must be 2D arrays.")
    if means.shape != variances.shape:
        raise ValueError("means and variances must have matching shapes.")
    if x.shape[1] != means.shape[1]:
        raise ValueError("Observation feature count does not match fitted HMM.")
    if np.any(variances <= 0.0):
        raise ValueError("variances must be positive.")

    diff = x[:, None, :] - means[None, :, :]
    log_det = np.sum(np.log(variances), axis=1)
    mahalanobis = np.sum((diff * diff) / variances[None, :, :], axis=2)
    n_features = x.shape[1]
    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det[None, :] + mahalanobis)


def forward_log_probabilities(
    initial_probabilities: FloatArray,
    transition_matrix: FloatArray,
    log_emissions: FloatArray,
) -> tuple[FloatArray, float]:
    """Run a normalized forward pass and return log filtering probabilities."""

    if initial_probabilities.ndim != 1:
        raise ValueError("initial_probabilities must be a 1D array.")
    if transition_matrix.ndim != 2:
        raise ValueError("transition_matrix must be a 2D array.")
    if log_emissions.ndim != 2:
        raise ValueError("log_emissions must be a 2D array.")
    n_states = initial_probabilities.shape[0]
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape must be (n_states, n_states).")
    if log_emissions.shape[1] != n_states:
        raise ValueError("log_emissions state count must match HMM parameters.")
    if np.any(initial_probabilities <= 0.0) or np.any(transition_matrix <= 0.0):
        raise ValueError("HMM probabilities must be strictly positive for log filtering.")

    log_initial = np.log(initial_probabilities)
    log_transition = np.log(transition_matrix)
    log_alpha = np.empty_like(log_emissions)
    log_scales = np.empty(log_emissions.shape[0], dtype=np.float64)

    first = log_initial + log_emissions[0]
    log_scales[0] = float(_logsumexp(first))
    log_alpha[0] = first - log_scales[0]
    for index in range(1, log_emissions.shape[0]):
        predicted = _logsumexp(log_alpha[index - 1][:, None] + log_transition, axis=0)
        current = predicted + log_emissions[index]
        log_scales[index] = float(_logsumexp(current))
        log_alpha[index] = current - log_scales[index]
    return log_alpha, float(np.sum(log_scales))


def filter_regime_probabilities(
    fitted_hmm: FittedHMM,
    phi_sequence: NDArray[np.floating] | list[list[float]],
) -> FloatArray:
    """Return ``P(k_t | x_1:t)`` regime probabilities for each observation."""

    observations = _as_2d_float_array(phi_sequence)
    log_emissions = gaussian_log_likelihood(
        observations,
        fitted_hmm.means,
        fitted_hmm.variances,
    )
    log_alpha, _ = forward_log_probabilities(
        fitted_hmm.initial_probabilities,
        fitted_hmm.transition_matrix,
        log_emissions,
    )
    probabilities = np.exp(log_alpha)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / row_sums


def validate_filtering_only(probabilities: FloatArray, fitted_hmm: FittedHMM) -> None:
    """Validate probability shape, finiteness, normalization, and metadata."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("Regime probabilities must be a 2D array.")
    if probs.shape[1] != fitted_hmm.metadata.n_states:
        raise ValueError("Regime probability state count does not match fitted HMM.")
    if not fitted_hmm.metadata.filtering_only:
        raise ValueError("Regime probabilities must be marked as filtering-only.")
    if not np.isfinite(probs).all():
        raise ValueError("Regime probabilities must be finite.")
    if np.any(probs < 0.0):
        raise ValueError("Regime probabilities must be non-negative.")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("Regime probabilities must sum to 1.")
