import numpy as np
from numpy.testing import assert_allclose

from finrl.portfolio_construction import (
    apply_position_cap,
    cross_sectional_zscore,
    estimate_liquidity_cost,
    shrink_covariance,
    smooth_target_weights,
)


def test_portfolio_construction_utilities() -> None:
    z = cross_sectional_zscore(np.array([[1.0, 2.0, 3.0]]))
    assert_allclose(z.mean(), 0.0, atol=1e-7)
    capped = apply_position_cap(np.array([[0.9, 0.1, 0.0]]), 0.5)
    assert np.all(capped[:, :2] <= 0.5 + 1e-6)
    assert_allclose(capped.sum(axis=1), 1.0)
    smoothed = smooth_target_weights(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 0.5)
    assert_allclose(smoothed, [[0.5, 0.5]])
    covariance = shrink_covariance(np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]))
    assert covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-7
    assert_allclose(estimate_liquidity_cost(np.array([0.04]), np.array([5.0])), [0.0007])
