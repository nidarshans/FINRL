"""JAX-native weekly portfolio trading environment step."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

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
from finrl.types import Array


class EnvConfig(NamedTuple):
    """Static environment parameters."""

    transaction_cost_rate: float = 0.001
    drawdown_limit: float = 0.2
    drawdown_penalty: float = 1.0
    drawdown_penalty_type: int = 0
    drawdown_temp: float = 0.01
    turnover_penalty: float = 0.0
    active_risk_penalty: float = 0.0
    sortino_target_return: float = 0.0
    sortino_downside_penalty: float = 0.0
    cash_index: int = -1


class EnvState(NamedTuple):
    """Portfolio state carried between weekly environment steps."""

    weights: Array
    portfolio_value: Array
    peak_value: Array
    drawdown: Array
    previous_turnover: Array
    step: Array


class StepResult(NamedTuple):
    """Diagnostics emitted by one environment step."""

    state: EnvState
    reward: Array
    gross_return: Array
    net_return: Array
    transaction_cost: Array
    turnover: Array


def environment_step(
    state: EnvState,
    target_weights: Array,
    asset_returns: Array,
    spy_return: Array,
    config: EnvConfig,
) -> StepResult:
    """Rebalance to target weights, hold for one week, and update state."""

    executed_weights = normalize_long_only_weights(
        target_weights,
        fallback_weights=state.weights,
        cash_index=config.cash_index,
    )
    turnover = calculate_turnover(state.weights, executed_weights)
    transaction_cost = calculate_transaction_cost(
        turnover, config.transaction_cost_rate
    )
    gross_return = calculate_gross_portfolio_return(executed_weights, asset_returns)
    net_return = calculate_net_portfolio_return(gross_return, transaction_cost)
    portfolio_value = update_portfolio_value(state.portfolio_value, net_return)
    peak_value = update_running_peak(state.peak_value, portfolio_value)
    drawdown = calculate_drawdown(portfolio_value, peak_value)
    reward_config = RewardConfig(
        drawdown_limit=config.drawdown_limit,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_penalty_type=config.drawdown_penalty_type,
        drawdown_temp=config.drawdown_temp,
        turnover_penalty=config.turnover_penalty,
        active_risk_penalty=config.active_risk_penalty,
        sortino_target_return=config.sortino_target_return,
        sortino_downside_penalty=config.sortino_downside_penalty,
    )
    reward = calculate_spy_relative_reward(
        net_return=net_return,
        spy_return=spy_return,
        drawdown=drawdown,
        turnover=turnover,
        config=reward_config,
    )
    next_state = EnvState(
        weights=executed_weights,
        portfolio_value=portfolio_value,
        peak_value=peak_value,
        drawdown=drawdown,
        previous_turnover=turnover,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )
    return StepResult(
        state=next_state,
        reward=reward,
        gross_return=gross_return,
        net_return=net_return,
        transaction_cost=transaction_cost,
        turnover=turnover,
    )


def scan_environment(
    initial_state: EnvState,
    target_weights: Array,
    returns: Array,
    spy_returns: Array,
    config: EnvConfig,
) -> tuple[EnvState, StepResult]:
    """Run the environment over a sequence of weekly target weights and returns."""

    def step_fn(carry: EnvState, inputs: tuple[Array, Array, Array]):
        weights_t, returns_t, spy_return_t = inputs
        result = environment_step(carry, weights_t, returns_t, spy_return_t, config)
        return result.state, result

    return jax.lax.scan(step_fn, initial_state, (target_weights, returns, spy_returns))
