"""Pure JAX accounting calculations for portfolio rebalancing."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def calculate_turnover(current_weights: Array, target_weights: Array) -> Array:
    """Return one-way turnover between current and target allocations."""

    return jnp.sum(jnp.abs(target_weights - current_weights))


def calculate_transaction_cost(turnover: Array, cost_rate: float | Array) -> Array:
    """Return transaction cost as a decimal return drag."""

    return jnp.asarray(cost_rate, dtype=jnp.asarray(turnover).dtype) * turnover


def calculate_gross_portfolio_return(weights: Array, asset_returns: Array) -> Array:
    """Return weighted holding-period portfolio return."""

    return jnp.sum(weights * asset_returns)


def calculate_net_portfolio_return(
    gross_return: Array, transaction_cost: Array
) -> Array:
    """Return portfolio return after transaction costs."""

    return gross_return - transaction_cost


def update_portfolio_value(portfolio_value: Array, net_return: Array) -> Array:
    """Update portfolio value using net holding-period return."""

    return portfolio_value * (1.0 + net_return)


def update_running_peak(previous_peak: Array, portfolio_value: Array) -> Array:
    """Return running maximum portfolio value."""

    return jnp.maximum(previous_peak, portfolio_value)


def calculate_drawdown(portfolio_value: Array, peak_value: Array) -> Array:
    """Return drawdown as ``1 - value / peak``."""

    return 1.0 - portfolio_value / peak_value


def normalize_long_only_weights(raw_weights: Array) -> Array:
    """Project nonnegative weights onto the long-only simplex by rescaling."""

    clipped = jnp.maximum(raw_weights, 0.0)
    total = jnp.sum(clipped)
    num_assets = clipped.shape[0]
    equal_weights = jnp.ones_like(clipped) / num_assets
    return jnp.where(total > 0.0, clipped / total, equal_weights)
