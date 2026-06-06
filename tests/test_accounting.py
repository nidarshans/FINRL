"""Unit tests for pure portfolio accounting calculations."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from finrl.env.accounting import (
    calculate_drawdown,
    calculate_gross_portfolio_return,
    calculate_net_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    normalize_long_only_weights,
    update_portfolio_value,
    update_running_peak,
)
from finrl.env.rewards import RewardConfig, calculate_spy_relative_reward

RTOL = 1e-6
ATOL = 1e-8


def test_calculate_turnover() -> None:
    current = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32)
    target = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)

    turnover = calculate_turnover(current, target)

    assert_allclose(turnover, 0.6, rtol=RTOL, atol=ATOL)


def test_calculate_transaction_cost() -> None:
    cost = calculate_transaction_cost(jnp.array(0.6, dtype=jnp.float32), 0.001)

    assert_allclose(cost, 0.0006, rtol=RTOL, atol=ATOL)


def test_calculate_gross_portfolio_return_includes_cash_return() -> None:
    weights = jnp.array([0.5, 0.25, 0.25], dtype=jnp.float32)
    returns = jnp.array([0.02, -0.01, 0.001], dtype=jnp.float32)

    gross_return = calculate_gross_portfolio_return(weights, returns)

    assert_allclose(gross_return, 0.00775, rtol=RTOL, atol=ATOL)


def test_calculate_net_portfolio_return() -> None:
    net_return = calculate_net_portfolio_return(
        jnp.array(0.00775, dtype=jnp.float32),
        jnp.array(0.0006, dtype=jnp.float32),
    )

    assert_allclose(net_return, 0.00715, rtol=RTOL, atol=ATOL)


def test_update_portfolio_value() -> None:
    new_value = update_portfolio_value(
        jnp.array(100.0, dtype=jnp.float32),
        jnp.array(0.00715, dtype=jnp.float32),
    )

    assert_allclose(new_value, 100.715, rtol=RTOL, atol=ATOL)


def test_update_running_peak() -> None:
    peak = update_running_peak(
        jnp.array(100.0, dtype=jnp.float32),
        jnp.array(100.715, dtype=jnp.float32),
    )

    assert_allclose(peak, 100.715, rtol=RTOL, atol=ATOL)


def test_calculate_drawdown() -> None:
    drawdown = calculate_drawdown(
        jnp.array(90.0, dtype=jnp.float32),
        jnp.array(100.0, dtype=jnp.float32),
    )

    assert_allclose(drawdown, 0.1, rtol=RTOL, atol=ATOL)


def test_calculate_spy_relative_reward_without_penalties() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.02, dtype=jnp.float32),
        spy_return=jnp.array(0.01, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.2, dtype=jnp.float32),
        config=RewardConfig(
            drawdown_limit=0.2,
            drawdown_penalty=1.0,
            turnover_penalty=0.0,
        ),
    )
    expected = np.log1p(0.02) - np.log1p(0.01)

    assert_allclose(reward, expected, rtol=RTOL, atol=ATOL)


def test_calculate_spy_relative_reward_with_drawdown_and_turnover_penalty() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.25, dtype=jnp.float32),
        turnover=jnp.array(0.4, dtype=jnp.float32),
        config=RewardConfig(
            drawdown_limit=0.2,
            drawdown_penalty=2.0,
            turnover_penalty=0.1,
        ),
    )

    assert_allclose(reward, -0.14, rtol=RTOL, atol=ATOL)


def test_calculate_spy_relative_reward_is_finite_when_spy_is_down() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(-0.01, dtype=jnp.float32),
        spy_return=jnp.array(-0.03, dtype=jnp.float32),
        drawdown=jnp.array(0.05, dtype=jnp.float32),
        turnover=jnp.array(0.1, dtype=jnp.float32),
        config=RewardConfig(),
    )

    assert bool(jnp.isfinite(reward))
    assert bool(reward > 0.0)


def test_normalize_long_only_weights_rescales_positive_weights() -> None:
    weights = normalize_long_only_weights(
        jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    )

    assert_allclose(weights, [0.2, 0.3, 0.5], rtol=RTOL, atol=ATOL)
    assert_allclose(jnp.sum(weights), 1.0, rtol=RTOL, atol=ATOL)


def test_normalize_long_only_weights_clips_negative_values() -> None:
    weights = normalize_long_only_weights(
        jnp.array([-1.0, 2.0, 2.0], dtype=jnp.float32)
    )

    assert_allclose(weights, [0.0, 0.5, 0.5], rtol=RTOL, atol=ATOL)
    assert bool(jnp.all(weights >= 0.0))


def test_normalize_long_only_weights_falls_back_to_equal_weight() -> None:
    weights = normalize_long_only_weights(
        jnp.array([-1.0, 0.0, -2.0], dtype=jnp.float32)
    )

    assert_allclose(weights, [1.0 / 3.0] * 3, rtol=RTOL, atol=ATOL)


def test_accounting_functions_support_jit_and_float32() -> None:
    @jax.jit
    def run_accounting(current_weights, target_weights, asset_returns):
        target = normalize_long_only_weights(target_weights)
        turnover = calculate_turnover(current_weights, target)
        transaction_cost = calculate_transaction_cost(turnover, 0.001)
        gross_return = calculate_gross_portfolio_return(target, asset_returns)
        net_return = calculate_net_portfolio_return(gross_return, transaction_cost)
        value = update_portfolio_value(jnp.array(100.0, dtype=jnp.float32), net_return)
        peak = update_running_peak(jnp.array(100.0, dtype=jnp.float32), value)
        drawdown = calculate_drawdown(value, peak)
        return turnover, transaction_cost, gross_return, net_return, value, drawdown

    outputs = run_accounting(
        jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32),
        jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32),
        jnp.array([0.02, -0.01, 0.001], dtype=jnp.float32),
    )

    chex.assert_rank(outputs[0], 0)
    assert outputs[0].dtype == jnp.dtype(jnp.float32)
    for value in outputs:
        assert bool(jnp.isfinite(value))
