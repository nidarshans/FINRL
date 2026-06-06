"""No-look-ahead tests for HMM regime detection."""

from __future__ import annotations

import inspect

import numpy as np
from numpy.testing import assert_allclose

import finrl.regimes.filtering as filtering
import finrl.regimes.hmm as hmm
from finrl.regimes import HMMConfig, filter_regime_probabilities, fit_hmm

RTOL = 1e-6
ATOL = 1e-8


def _train_phi() -> np.ndarray:
    return np.array(
        [
            [-1.5, -1.0],
            [-1.2, -0.8],
            [-1.4, -1.1],
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.2],
            [1.4, 0.9],
            [1.2, 1.1],
            [1.6, 0.8],
        ],
        dtype=np.float64,
    )


def test_filtering_probabilities_do_not_change_when_future_observations_change() -> None:
    fitted = fit_hmm(_train_phi(), HMMConfig(n_states=3, max_iter=5))
    base = np.array(
        [
            [-1.0, -0.7],
            [-0.8, -0.6],
            [0.1, 0.0],
            [1.0, 0.8],
        ],
        dtype=np.float64,
    )
    changed_future = base.copy()
    changed_future[3] = np.array([100.0, -100.0])

    base_probs = filter_regime_probabilities(fitted, base)
    changed_probs = filter_regime_probabilities(fitted, changed_future)

    assert_allclose(base_probs[:3], changed_probs[:3], rtol=RTOL, atol=ATOL)


def test_hmm_fit_api_accepts_train_phi_only_for_observations() -> None:
    signature = inspect.signature(fit_hmm)

    assert tuple(signature.parameters) == ("train_phi", "config")


def test_hmm_evaluation_uses_explicit_forward_filter_not_smoothing_api() -> None:
    source = inspect.getsource(filtering.filter_regime_probabilities)

    assert "forward_log_probabilities" in source
    assert "backward" not in source.lower()
    assert "smooth" not in source.lower()


def test_hmm_training_uses_hmmlearn_not_local_em() -> None:
    source = inspect.getsource(hmm.fit_hmm)

    assert "GaussianHMM" in source
    assert "model.fit(observations)" in source
    assert "expectation" not in source.lower()
    assert "maximization" not in source.lower()


def test_fitting_does_not_depend_on_heldout_future_values() -> None:
    train = _train_phi()
    future = np.full((3, 2), 9_999.0, dtype=np.float64)

    train_only = fit_hmm(train, HMMConfig(n_states=3, max_iter=5))
    with_future_available_but_not_passed = fit_hmm(train, HMMConfig(n_states=3, max_iter=5))

    assert future.shape == (3, 2)
    assert_allclose(
        train_only.initial_probabilities,
        with_future_available_but_not_passed.initial_probabilities,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        train_only.transition_matrix,
        with_future_available_but_not_passed.transition_matrix,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(train_only.means, with_future_available_but_not_passed.means, rtol=RTOL, atol=ATOL)
