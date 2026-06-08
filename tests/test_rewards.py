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


def test_hinge_drawdown_penalty_is_zero_below_threshold() -> None:
    config = RewardConfig(
        drawdown_limit=0.2,
        drawdown_penalty=5.0,
        drawdown_penalty_type="hinge",
    )

    reward = calculate_spy_relative_reward(
        net_return=jnp.array(0.0, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.19, dtype=jnp.float32),
        turnover=jnp.array(0.0, dtype=jnp.float32),
        config=config,
    )

    assert_allclose(reward, 0.0, rtol=RTOL, atol=ATOL)


def test_hinge_drawdown_penalty_is_positive_above_threshold() -> None:
    config = RewardConfig(
        drawdown_limit=0.2,
        drawdown_penalty=5.0,
        drawdown_penalty_type="hinge",
    )

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


def test_default_reward_turnover_penalty_is_only_net_return_effect() -> None:
    config = RewardConfig()

    reward_low = calculate_spy_relative_reward(
        net_return=jnp.array(0.01, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.1, dtype=jnp.float32),
        config=config,
    )
    reward_high = calculate_spy_relative_reward(
        net_return=jnp.array(0.01, dtype=jnp.float32),
        spy_return=jnp.array(0.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        turnover=jnp.array(0.9, dtype=jnp.float32),
        config=config,
    )

    assert_allclose(reward_low, reward_high, rtol=RTOL, atol=ATOL)


def test_smooth_drawdown_penalty_is_finite_and_monotonic() -> None:
    config = RewardConfig(
        drawdown_limit=0.2,
        drawdown_penalty=1.0,
        drawdown_penalty_type="smooth",
        drawdown_temp=0.01,
    )

    below = calculate_spy_relative_reward(
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.1, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        config,
    )
    above = calculate_spy_relative_reward(
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.25, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        config,
    )

    assert bool(jnp.isfinite(below))
    assert bool(jnp.isfinite(above))
    assert bool(below > above)
    assert bool(jnp.abs(below) < 1e-4)


def test_active_risk_penalty_is_optional() -> None:
    base = calculate_spy_relative_reward(
        jnp.array(0.03, dtype=jnp.float32),
        jnp.array(0.01, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        RewardConfig(active_risk_penalty=0.0),
    )
    penalized = calculate_spy_relative_reward(
        jnp.array(0.03, dtype=jnp.float32),
        jnp.array(0.01, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        RewardConfig(active_risk_penalty=10.0),
    )

    assert bool(penalized < base)
    assert bool(jnp.isfinite(penalized))


def test_sortino_downside_penalty_ignores_upside_returns() -> None:
    config = RewardConfig(
        sortino_target_return=0.0,
        sortino_downside_penalty=25.0,
    )

    reward = calculate_spy_relative_reward(
        jnp.array(0.02, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        config,
    )
    expected = np.log1p(0.02)

    assert_allclose(reward, expected, rtol=RTOL, atol=ATOL)


def test_sortino_downside_penalty_scales_with_squared_shortfall() -> None:
    base_config = RewardConfig(sortino_downside_penalty=0.0)
    sortino_config = RewardConfig(
        sortino_target_return=0.0,
        sortino_downside_penalty=25.0,
    )

    base = calculate_spy_relative_reward(
        jnp.array(-0.02, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        base_config,
    )
    penalized = calculate_spy_relative_reward(
        jnp.array(-0.02, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        sortino_config,
    )

    assert_allclose(penalized - base, -25.0 * 0.02**2, rtol=RTOL, atol=ATOL)
    assert bool(jnp.isfinite(penalized))


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
