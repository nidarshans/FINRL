"""Pure JAX accounting calculations for portfolio rebalancing."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def calculate_turnover(current_weights: Array, target_weights: Array) -> Array:
    """Return institutional one-way turnover between two allocations."""

    return 0.5 * jnp.sum(jnp.abs(target_weights - current_weights))


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


def cash_weights_like(raw_weights: Array, cash_index: int = -1) -> Array:
    """Return a 100% cash allocation with the same shape as ``raw_weights``."""

    weights = jnp.zeros_like(raw_weights)
    return weights.at[cash_index].set(1.0)


def normalize_long_only_weights(
    raw_weights: Array,
    fallback_weights: Array | None = None,
    cash_index: int = -1,
) -> Array:
    """Project finite nonnegative weights onto the long-only simplex.

    Invalid or all-zero inputs fall back to ``fallback_weights`` when supplied,
    otherwise to 100% cash.
    """

    finite = jnp.all(jnp.isfinite(raw_weights))
    cleaned = jnp.where(jnp.isfinite(raw_weights), raw_weights, 0.0)
    clipped = jnp.maximum(cleaned, 0.0)
    total = jnp.sum(clipped)
    fallback = (
        cash_weights_like(raw_weights, cash_index)
        if fallback_weights is None
        else jnp.asarray(fallback_weights, dtype=raw_weights.dtype)
    )
    return jnp.where((total > 0.0) & finite, clipped / total, fallback)
