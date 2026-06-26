"""Reward functions for the trading environment."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp

from finrl.types import Array


class RewardConfig(NamedTuple):
    """Parameters for SPY-relative reward calculation.

    ``turnover_penalty`` is an optional extra regularizer beyond transaction
    costs already embedded in ``net_return``. ``sortino_downside_penalty`` is a
    per-step Sortino-style downside regularizer: it penalizes squared shortfall
    below ``sortino_target_return`` without penalizing upside volatility.
    """

    drawdown_limit: float = 0.05
    drawdown_penalty: float = 0.0
    drawdown_penalty_type: str | int = "smooth"
    drawdown_temp: float = 0.01
    turnover_penalty: float = 0.0
    active_risk_penalty: float = 0.0
    sortino_target_return: float = 0.0
    sortino_downside_penalty: float = 0.05


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


def calculate_rewards(
    net_return: Array,
    spy_return: Array,
    drawdown: Array,
    turnover: Array,
    config: RewardConfig,
) -> Array:
    """Return the configured portfolio reward."""

    drawdown_excess = drawdown - config.drawdown_limit
    hinge_penalty = jnp.maximum(0.0, drawdown_excess)
    smooth_penalty = config.drawdown_temp * jax_softplus(
        drawdown_excess / config.drawdown_temp
    )
    use_hinge = (
        config.drawdown_penalty_type == "hinge"
        if isinstance(config.drawdown_penalty_type, str)
        else config.drawdown_penalty_type == 1
    )
    drawdown_penalty = jnp.where(use_hinge, hinge_penalty, smooth_penalty)
    log_return = jnp.log1p(net_return)
    active_return = net_return - spy_return
    downside_shortfall = jnp.maximum(config.sortino_target_return - net_return, 0.0)
    return (
        log_return
        # - config.drawdown_penalty * drawdown_penalty
        - config.sortino_downside_penalty * jnp.square(downside_shortfall)
        # - config.turnover_penalty * turnover
        # - config.active_risk_penalty * jnp.square(active_return)
    )


def jax_softplus(x: Array) -> Array:
    """Numerically stable softplus wrapper kept local for reward JIT purity."""

    return jnp.logaddexp(x, 0.0)


calculate_spy_relative_reward = calculate_rewards
spy_relative_reward = calculate_rewards


def calculate_reward(
    net_return: Array,
    spy_return: Array,
    drawdown: Array,
    turnover: Array,
    config: RewardConfig,
    reward_fn: RewardFn = calculate_rewards,
) -> Array:
    """Evaluate the configured reward function."""

    return reward_fn(net_return, spy_return, drawdown, turnover, config)
