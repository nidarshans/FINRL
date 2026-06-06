"""Focused tests for SPY-relative reward behavior."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from finrl.env.rewards import (
    RewardConfig,
    calculate_reward,
    calculate_spy_relative_reward,
)

RTOL = 1e-6
ATOL = 1e-8


def test_reward_positive_when_portfolio_beats_spy() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.02, dtype=jnp.float32),
        spy_return=jnp.array(0.01, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=RewardConfig(),
    )

    assert bool(reward > 0.0)


def test_reward_negative_when_portfolio_trails_spy() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.005, dtype=jnp.float32),
        spy_return=jnp.array(0.01, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=RewardConfig(),
    )

    assert bool(reward < 0.0)


def test_reward_handles_spy_up_period() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.03, dtype=jnp.float32),
        spy_return=jnp.array(0.02, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=RewardConfig(),
    )
    expected = np.log1p(0.03) - np.log1p(0.02)

    assert_allclose(reward, expected, rtol=RTOL, atol=ATOL)


def test_reward_handles_spy_down_period() -> None:
    reward = calculate_spy_relative_reward(
        net_return=jnp.array(-0.01, dtype=jnp.float32),
        spy_return=jnp.array(-0.04, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=RewardConfig(),
    )
    expected = np.log1p(-0.01) - np.log1p(-0.04)

    assert_allclose(reward, expected, rtol=RTOL, atol=ATOL)
    assert bool(reward > 0.0)


def test_drawdown_penalty_is_zero_below_threshold() -> None:
    config = RewardConfig(drawdown_limit=0.2, drawdown_penalty=5.0)

    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.19, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=config,
    )

    assert_allclose(reward, 0.0, rtol=RTOL, atol=ATOL)


def test_drawdown_penalty_is_positive_above_threshold() -> None:
    config = RewardConfig(drawdown_limit=0.2, drawdown_penalty=5.0)

    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.23, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=config,
    )

    assert_allclose(reward, -0.15, rtol=RTOL, atol=ATOL)


def test_turnover_penalty_scales_linearly() -> None:
    config = RewardConfig(turnover_penalty=0.25)

    low_turnover_reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.4, dtype=jnp.float32),
        config=config,
    )
    high_turnover_reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.8, dtype=jnp.float32),
        config=config,
    )

    assert_allclose(high_turnover_reward - low_turnover_reward, -0.1, rtol=RTOL, atol=ATOL)


def test_reward_can_use_alternate_callable() -> None:
    def absolute_net_return_reward(
        net_return,
        spy_return,
        drawdown,
        turnover,
        config,
    ):
        del spy_return, drawdown, turnover, config
        return net_return

    reward = calculate_reward(
        net_return=jnp.array(0.012, dtype=jnp.float32),
        spy_return=jnp.array(0.2, dtype=jnp.float32),
        drawdown=jnp.array(0.9, dtype=jnp.float32),
        turnover=jnp.array(2.0, dtype=jnp.float32),
        config=RewardConfig(drawdown_penalty=100.0, turnover_penalty=100.0),
        reward_fn=absolute_net_return_reward,
    )

    assert_allclose(reward, 0.012, rtol=RTOL, atol=ATOL)
