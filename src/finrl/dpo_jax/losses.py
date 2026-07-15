"""Differentiable portfolio losses for direct policy optimization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.dpo_jax.config import DPOConfig
from finrl.env.accounting import (
    calculate_transaction_cost,
    calculate_turnover,
    evolve_portfolio_weights,
)
from finrl.types import Array


class DPOLossMetrics(NamedTuple):
    """Diagnostics emitted by the differentiable portfolio loss."""

    mean_net_return: Array
    net_return_volatility: Array
    sharpe_ratio: Array
    mean_turnover: Array
    final_equity: Array


def dpo_loss(
    weights: Array,
    asset_returns: Array,
    initial_weights: Array,
    config: DPOConfig,
) -> tuple[Array, DPOLossMetrics]:
    """Compute the negative net-return Sharpe ratio over a backtest path.

    ``weights`` has shape ``[T, N + 1]`` and includes cash in the final column.
    ``asset_returns`` has shape ``[T, N]`` and excludes cash.
    Transaction costs are included in net returns before calculating Sharpe.
    """

    weights_array = jnp.asarray(weights, dtype=jnp.float32)
    returns_array = jnp.asarray(asset_returns, dtype=jnp.float32)
    initial = jnp.asarray(initial_weights, dtype=jnp.float32)
    transaction_cost_rate = config.transaction_cost_bps / 10000.0

    def step(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        prev_weights, equity = carry
        weights_t, returns_t = inputs
        stock_weights = weights_t[:-1]
        gross_return = jnp.sum(stock_weights * returns_t)
        turnover = calculate_turnover(prev_weights, weights_t)
        transaction_cost = calculate_transaction_cost(
            turnover,
            transaction_cost_rate,
        )
        net_return = gross_return - transaction_cost
        equity = equity * (1.0 + net_return)
        full_returns = jnp.concatenate(
            [returns_t, jnp.zeros((1,), dtype=returns_t.dtype)]
        )
        current_weights = evolve_portfolio_weights(weights_t, full_returns)
        outputs = (net_return, turnover, equity)
        return (current_weights, equity), outputs

    (_, final_equity), outputs = jax.lax.scan(
        step,
        (initial, jnp.array(1.0, dtype=jnp.float32)),
        (weights_array, returns_array),
    )
    net_returns, turnovers, equities = outputs
    mean_net_return = jnp.mean(net_returns)
    net_return_volatility = jnp.std(net_returns)
    sharpe_ratio = mean_net_return / (net_return_volatility + config.eps)
    loss = -sharpe_ratio
    metrics = DPOLossMetrics(
        mean_net_return=mean_net_return,
        net_return_volatility=net_return_volatility,
        sharpe_ratio=sharpe_ratio,
        mean_turnover=jnp.mean(turnovers),
        final_equity=jnp.where(
            equities.shape[0] > 0,
            final_equity,
            jnp.array(1.0, dtype=jnp.float32),
        ),
    )
    return loss, metrics
