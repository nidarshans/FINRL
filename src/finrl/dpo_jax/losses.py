"""Differentiable portfolio losses for direct policy optimization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.dpo_jax.config import DPOConfig
from finrl.types import Array


class DPOLossMetrics(NamedTuple):
    """Diagnostics emitted by the differentiable portfolio loss."""

    mean_log_return: Array
    mean_active_log_return: Array
    mean_turnover: Array
    max_drawdown: Array
    mean_concentration: Array
    final_equity: Array


def dpo_loss(
    weights: Array,
    asset_returns: Array,
    initial_weights: Array,
    config: DPOConfig,
    spy_returns: Array | None = None,
) -> tuple[Array, DPOLossMetrics]:
    """Compute a scan-based differentiable portfolio objective.

    ``weights`` has shape ``[T, N + 1]`` and includes cash in the final column.
    ``asset_returns`` has shape ``[T, N]`` and excludes cash.
    ``spy_returns`` has shape ``[T]`` and is used as the active-return benchmark.
    """

    weights_array = jnp.asarray(weights, dtype=jnp.float32)
    returns_array = jnp.asarray(asset_returns, dtype=jnp.float32)
    benchmark_returns = (
        jnp.zeros((returns_array.shape[0],), dtype=jnp.float32)
        if spy_returns is None
        else jnp.asarray(spy_returns, dtype=jnp.float32)
    )
    if benchmark_returns.shape != (returns_array.shape[0],):
        raise ValueError("spy_returns must have shape [time].")
    initial = jnp.asarray(initial_weights, dtype=jnp.float32)
    transaction_cost_rate = config.transaction_cost_bps / 10000.0

    def step(carry: tuple[Array, Array, Array], inputs: tuple[Array, Array]):
        prev_weights, equity, running_max = carry
        weights_t, returns_t = inputs
        stock_weights = weights_t[:-1]
        gross_return = jnp.sum(stock_weights * returns_t)
        turnover = jnp.sum(jnp.abs(weights_t - prev_weights))
        transaction_cost = turnover * transaction_cost_rate
        net_return = gross_return - transaction_cost
        equity = equity * (1.0 + net_return)
        running_max = jnp.maximum(running_max, equity)
        drawdown = 1.0 - equity / jnp.maximum(running_max, config.eps)
        concentration = jnp.sum(weights_t**2)
        outputs = (net_return, turnover, drawdown, concentration, equity)
        return (weights_t, equity, running_max), outputs

    (_, final_equity, _), outputs = jax.lax.scan(
        step,
        (initial, jnp.array(1.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)),
        (weights_array, returns_array),
    )
    net_returns, turnovers, drawdowns, concentrations, equities = outputs
    log_returns = jnp.log(1.0 + net_returns + config.eps)
    spy_log_returns = jnp.log(1.0 + benchmark_returns + config.eps)
    active_log_returns = log_returns - spy_log_returns

    return_loss = -jnp.mean(active_log_returns)
    turnover_loss = jnp.mean(turnovers)
    drawdown_loss = jnp.mean(drawdowns**2)
    concentration_loss = jnp.mean(concentrations)
    loss = (
        return_loss
        + config.lambda_drawdown * drawdown_loss
    )
    metrics = DPOLossMetrics(
        mean_log_return=jnp.mean(log_returns),
        mean_active_log_return=jnp.mean(active_log_returns),
        mean_turnover=turnover_loss,
        max_drawdown=jnp.max(drawdowns),
        mean_concentration=concentration_loss,
        final_equity=jnp.where(
            equities.shape[0] > 0,
            final_equity,
            jnp.array(1.0, dtype=jnp.float32),
        ),
    )
    return loss, metrics
