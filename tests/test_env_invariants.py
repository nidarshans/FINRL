"""Invariant and regression tests for environment accounting."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState, environment_step
from tests.fixtures.accounting_cases import ACCOUNTING_CASES

RTOL = 1e-6
ATOL = 1e-8


def _state_from_case(case) -> EnvState:
    return EnvState(
        weights=jnp.array(case.current_weights, dtype=jnp.float32),
        portfolio_value=jnp.array(case.starting_value, dtype=jnp.float32),
        peak_value=jnp.array(case.starting_peak, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


@pytest.mark.parametrize("case", ACCOUNTING_CASES, ids=[case.name for case in ACCOUNTING_CASES])
def test_hand_computed_weekly_rebalance_cases(case) -> None:
    result = environment_step(
        _state_from_case(case),
        jnp.array(case.target_weights, dtype=jnp.float32),
        jnp.array(case.asset_returns, dtype=jnp.float32),
        jnp.array(case.spy_return, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=case.transaction_cost_rate),
    )

    assert_allclose(result.turnover, case.turnover, rtol=RTOL, atol=ATOL)
    assert_allclose(result.transaction_cost, case.transaction_cost, rtol=RTOL, atol=ATOL)
    assert_allclose(result.gross_return, case.gross_return, rtol=RTOL, atol=ATOL)
    assert_allclose(result.net_return, case.net_return, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.portfolio_value, case.ending_value, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.peak_value, case.ending_peak, rtol=RTOL, atol=ATOL)
    expected_value = np.float32(case.starting_value) * (
        np.float32(1.0) + np.float32(case.net_return)
    )
    expected_peak = np.maximum(np.float32(case.starting_peak), expected_value)
    expected_drawdown = np.float32(1.0) - expected_value / expected_peak
    assert_allclose(result.state.drawdown, expected_drawdown, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.weights, case.target_weights, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("target_weights", "asset_returns", "spy_return", "transaction_cost_rate"),
    (
        ((1.0, 0.0, 0.0), (-0.10, 0.02, 0.001), -0.05, 0.001),
        ((0.0, 1.0, 0.0), (0.00, 0.00, 0.000), 0.02, 0.0),
        ((0.0, 0.0, 1.0), (0.10, -0.10, 0.001), 0.00, 0.001),
        ((0.4, 0.4, 0.2), (0.01, 0.02, 0.001), 0.01, 0.001),
    ),
)
def test_environment_invariants_across_edge_cases(
    target_weights,
    asset_returns,
    spy_return,
    transaction_cost_rate,
) -> None:
    result = environment_step(
        EnvState(
            weights=jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32),
            portfolio_value=jnp.array(100.0, dtype=jnp.float32),
            peak_value=jnp.array(100.0, dtype=jnp.float32),
            drawdown=jnp.array(0.0, dtype=jnp.float32),
            previous_turnover=jnp.array(0.0, dtype=jnp.float32),
            step=jnp.array(0, dtype=jnp.int32),
        ),
        jnp.array(target_weights, dtype=jnp.float32),
        jnp.array(asset_returns, dtype=jnp.float32),
        jnp.array(spy_return, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=transaction_cost_rate),
    )

    assert_allclose(jnp.sum(result.state.weights), 1.0, rtol=RTOL, atol=ATOL)
    assert bool(jnp.all(result.state.weights >= 0.0))
    assert bool(result.state.portfolio_value > 0.0)
    assert bool(result.state.drawdown >= 0.0)
    assert bool(result.state.drawdown <= 1.0)
    assert bool(result.turnover >= 0.0)
    assert bool(result.transaction_cost >= 0.0)
    assert bool(jnp.isfinite(result.reward))


def test_weekly_rebalance_uses_current_weights_before_rebalance() -> None:
    state = EnvState(
        weights=jnp.array([0.90, 0.10, 0.00], dtype=jnp.float32),
        portfolio_value=jnp.array(100.0, dtype=jnp.float32),
        peak_value=jnp.array(100.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )

    result = environment_step(
        state,
        jnp.array([0.10, 0.80, 0.10], dtype=jnp.float32),
        jnp.array([0.00, 0.00, 0.00], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert_allclose(result.turnover, 0.80, rtol=RTOL, atol=ATOL)
    assert_allclose(result.transaction_cost, 0.00080, rtol=RTOL, atol=ATOL)
    assert_allclose(result.net_return, -0.00080, rtol=RTOL, atol=ATOL)
