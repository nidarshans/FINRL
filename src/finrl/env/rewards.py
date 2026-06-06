"""Reward functions for the trading environment."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp

from finrl.types import Array


class RewardConfig(NamedTuple):
    """Parameters for SPY-relative reward calculation."""

    drawdown_limit: float = 0.2
    drawdown_penalty: float = 1.0
    turnover_penalty: float = 0.0


class RewardFn(Protocol):
    """Callable interface for pluggable reward functions."""

    def __call__(
        self,
        net_return: Array,
        spy_return: Array,
        drawdown: Array,
        turnover: Array,
        config: RewardConfig,
    ) -> Array:
        """Return a scalar reward for one environment step."""


def calculate_spy_relative_reward(
    net_return: Array,
    spy_return: Array,
    drawdown: Array,
    turnover: Array,
    config: RewardConfig,
) -> Array:
    """Return log excess return versus SPY with drawdown and turnover penalties."""

    drawdown_excess = jnp.maximum(0.0, drawdown - config.drawdown_limit)
    return (
        jnp.log1p(net_return)
        - jnp.log1p(spy_return)
        - config.drawdown_penalty * drawdown_excess
        - config.turnover_penalty * turnover
    )


spy_relative_reward = calculate_spy_relative_reward


def calculate_reward(
    net_return: Array,
    spy_return: Array,
    drawdown: Array,
    turnover: Array,
    config: RewardConfig,
    reward_fn: RewardFn = calculate_spy_relative_reward,
) -> Array:
    """Evaluate the configured reward function."""

    return reward_fn(net_return, spy_return, drawdown, turnover, config)
