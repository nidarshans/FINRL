"""Tests for diagonal Gaussian HMM regime detection."""

from __future__ import annotations

from datetime import date

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from finrl.regimes.filtering import gaussian_log_likelihood
from finrl.regimes import (
    FittedHMM,
    HMMConfig,
    HMMMetadata,
    annual_hmm_refit,
    filter_regime_probabilities,
    fit_hmm,
    validate_filtering_only,
)

RTOL = 1e-6
ATOL = 1e-8


def _toy_phi() -> np.ndarray:
    return np.array(
        [
            [-2.0, -1.0],
            [-1.8, -1.2],
            [-2.2, -0.9],
            [0.0, 0.2],
            [0.1, -0.1],
            [0.2, 0.1],
            [2.0, 1.0],
            [1.8, 1.2],
            [2.2, 0.9],
            [2.1, 1.1],
        ],
        dtype=np.float64,
    )


def test_default_hmm_fit_and_filter_shapes_are_valid() -> None:
    fitted = fit_hmm(_toy_phi(), HMMConfig(max_iter=5))
    probabilities = filter_regime_probabilities(fitted, _toy_phi())

    assert fitted.metadata.n_states == 4
    assert fitted.metadata.covariance_type == "diag"
    assert fitted.means.shape == (4, 2)
    assert fitted.variances.shape == (4, 2)
    assert probabilities.shape == (10, 4)
    validate_filtering_only(probabilities, fitted)
    assert_allclose(probabilities.sum(axis=1), np.ones(10), rtol=RTOL, atol=ATOL)


def test_state_count_is_configurable_and_diagonal_variance_is_positive() -> None:
    fitted = fit_hmm(_toy_phi(), HMMConfig(n_states=3, max_iter=5, min_covar=1e-5))
    probabilities = filter_regime_probabilities(fitted, _toy_phi())

    assert fitted.metadata.n_states == 3
    assert fitted.transition_matrix.shape == (3, 3)
    assert probabilities.shape == (10, 3)
    assert np.all(fitted.variances >= 1e-5)


def test_hmm_metadata_records_train_window() -> None:
    fitted = fit_hmm(
        _toy_phi(),
        HMMConfig(
            n_states=3,
            max_iter=5,
            train_start=date(2020, 1, 3),
            train_end=date(2021, 12, 31),
        ),
    )

    assert fitted.metadata.train_window is not None
    assert fitted.metadata.train_window.start == date(2020, 1, 3)
    assert fitted.metadata.train_window.end == date(2021, 12, 31)
    assert fitted.metadata.n_observations == 10
    assert fitted.metadata.n_features == 2


def test_annual_hmm_refit_returns_one_model_per_train_split() -> None:
    first = _toy_phi()
    second = _toy_phi() + np.array([0.5, -0.25])

    fitted_models = annual_hmm_refit((first, second), HMMConfig(n_states=3, max_iter=3))

    assert len(fitted_models) == 2
    assert all(model.metadata.n_states == 3 for model in fitted_models)


def test_gaussian_log_likelihood_matches_scipy_reference() -> None:
    observations = np.array([[0.0, 0.5], [1.0, -1.0]], dtype=np.float64)
    means = np.array([[0.0, 0.0], [1.0, -0.5]], dtype=np.float64)
    variances = np.array([[0.5, 1.5], [2.0, 0.75]], dtype=np.float64)

    actual = gaussian_log_likelihood(observations, means, variances)
    expected = np.column_stack(
        [
            multivariate_normal.logpdf(
                observations,
                mean=means[state],
                cov=np.diag(variances[state]),
            )
            for state in range(means.shape[0])
        ]
    )

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_forward_filter_matches_hmmlearn_prefix_posteriors() -> None:
    initial = np.array([0.65, 0.35], dtype=np.float64)
    transitions = np.array([[0.82, 0.18], [0.25, 0.75]], dtype=np.float64)
    means = np.array([[-1.0, 0.0], [1.0, 0.5]], dtype=np.float64)
    variances = np.array([[0.4, 0.9], [0.7, 0.5]], dtype=np.float64)
    observations = np.array(
        [
            [-0.8, 0.1],
            [-0.4, 0.2],
            [0.9, 0.4],
            [1.2, 0.7],
        ],
        dtype=np.float64,
    )
    fitted = FittedHMM(
        initial_probabilities=initial,
        transition_matrix=transitions,
        means=means,
        variances=variances,
        metadata=HMMMetadata(
            n_observations=observations.shape[0],
            n_features=observations.shape[1],
            n_states=2,
            covariance_type="diag",
            train_window=None,
            converged=True,
            n_iter=1,
            log_likelihood=0.0,
        ),
    )

    reference = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        init_params="",
        params="",
    )
    reference.n_features = observations.shape[1]
    reference.startprob_ = initial
    reference.transmat_ = transitions
    reference.means_ = means
    reference.covars_ = variances

    actual = filter_regime_probabilities(fitted, observations)
    expected = np.vstack(
        [reference.predict_proba(observations[: index + 1])[-1] for index in range(len(observations))]
    )

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
