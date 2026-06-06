"""End-to-end tests for the JAX trading environment step."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from finrl.env.trading_env import (
    EnvConfig,
    EnvState,
    environment_step,
    scan_environment,
)

RTOL = 1e-6
ATOL = 1e-8


def _initial_state() -> EnvState:
    return EnvState(
        weights=jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32),
        portfolio_value=jnp.array(100.0, dtype=jnp.float32),
        peak_value=jnp.array(100.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def test_environment_step_updates_all_accounting_fields() -> None:
    state = _initial_state()
    target_weights = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)
    asset_returns = jnp.array([0.02, -0.01, 0.001], dtype=jnp.float32)
    spy_return = jnp.array(0.004, dtype=jnp.float32)
    config = EnvConfig(transaction_cost_rate=0.001)

    result = environment_step(state, target_weights, asset_returns, spy_return, config)

    assert_allclose(result.turnover, 0.6, rtol=RTOL, atol=ATOL)
    assert_allclose(result.transaction_cost, 0.0006, rtol=RTOL, atol=ATOL)
    assert_allclose(result.gross_return, -0.0007, rtol=RTOL, atol=ATOL)
    assert_allclose(result.net_return, -0.0013, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.portfolio_value, 99.87, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.peak_value, 100.0, rtol=RTOL, atol=ATOL)
    expected_drawdown = np.float32(1.0) - np.float32(99.87) / np.float32(100.0)
    assert_allclose(result.state.drawdown, expected_drawdown, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.weights, target_weights, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.previous_turnover, 0.6, rtol=RTOL, atol=ATOL)
    assert int(result.state.step) == 1
    assert bool(jnp.isfinite(result.reward))


def test_environment_step_invariants_hold_for_valid_inputs() -> None:
    result = environment_step(
        _initial_state(),
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([-0.05, 0.02, 0.001], dtype=jnp.float32),
        jnp.array(-0.03, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert_allclose(jnp.sum(result.state.weights), 1.0, rtol=RTOL, atol=ATOL)
    assert bool(jnp.all(result.state.weights >= 0.0))
    assert bool(result.state.portfolio_value > 0.0)
    assert bool(result.state.drawdown >= 0.0)
    assert bool(result.state.drawdown <= 1.0)
    assert bool(result.turnover >= 0.0)
    assert bool(result.transaction_cost >= 0.0)
    assert bool(jnp.isfinite(result.reward))


def test_environment_step_supports_100_stocks_plus_cash() -> None:
    num_assets = 101
    initial_weights = jnp.ones(num_assets, dtype=jnp.float32) / num_assets
    target_weights = jnp.zeros(num_assets, dtype=jnp.float32).at[-1].set(1.0)
    asset_returns = jnp.zeros(num_assets, dtype=jnp.float32).at[-1].set(0.001)
    state = EnvState(
        weights=initial_weights,
        portfolio_value=jnp.array(100.0, dtype=jnp.float32),
        peak_value=jnp.array(100.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )

    result = environment_step(
        state,
        target_weights,
        asset_returns,
        jnp.array(0.0, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.0, cash_index=100),
    )

    assert result.state.weights.shape == (101,)
    assert_allclose(jnp.sum(result.state.weights), 1.0, rtol=RTOL, atol=ATOL)
    assert_allclose(result.gross_return, 0.001, rtol=RTOL, atol=ATOL)


def test_environment_step_all_cash_allocation_earns_cash_return_less_cost() -> None:
    result = environment_step(
        _initial_state(),
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        jnp.array([0.10, -0.10, 0.001], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.0, cash_index=2),
    )

    assert_allclose(result.gross_return, 0.001, rtol=RTOL, atol=ATOL)
    assert_allclose(result.net_return, 0.001, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.portfolio_value, 100.1, rtol=RTOL, atol=ATOL)


def test_environment_step_high_turnover_with_zero_returns_is_finite() -> None:
    result = environment_step(
        _initial_state(),
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert_allclose(result.turnover, 1.6, rtol=RTOL, atol=ATOL)
    assert_allclose(result.transaction_cost, 0.0016, rtol=RTOL, atol=ATOL)
    assert bool(jnp.isfinite(result.net_return))
    assert bool(jnp.isfinite(result.reward))


def test_environment_step_supports_jit() -> None:
    jitted_step = jax.jit(environment_step)

    result = jitted_step(
        _initial_state(),
        jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32),
        jnp.array([0.02, -0.01, 0.001], dtype=jnp.float32),
        jnp.array(0.004, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert_allclose(result.turnover, 0.6, rtol=RTOL, atol=ATOL)
    assert_allclose(result.state.portfolio_value, 99.87, rtol=RTOL, atol=ATOL)


def test_scan_environment_runs_multiple_weekly_steps() -> None:
    target_weights = jnp.array(
        [
            [0.2, 0.5, 0.3],
            [0.4, 0.4, 0.2],
        ],
        dtype=jnp.float32,
    )
    returns = jnp.array(
        [
            [0.02, -0.01, 0.001],
            [0.01, 0.02, 0.001],
        ],
        dtype=jnp.float32,
    )
    spy_returns = jnp.array([0.004, 0.005], dtype=jnp.float32)

    final_state, results = scan_environment(
        _initial_state(),
        target_weights,
        returns,
        spy_returns,
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert final_state.step == 2
    assert results.reward.shape == (2,)
    assert results.state.weights.shape == (2, 3)
    assert bool(jnp.all(jnp.isfinite(results.reward)))


def test_environment_step_is_deterministic() -> None:
    args = (
        _initial_state(),
        jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32),
        jnp.array([0.02, -0.01, 0.001], dtype=jnp.float32),
        jnp.array(0.004, dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    result_a = environment_step(*args)
    result_b = environment_step(*args)

    assert_allclose(result_a.state.portfolio_value, result_b.state.portfolio_value)
    assert_allclose(result_a.reward, result_b.reward)
