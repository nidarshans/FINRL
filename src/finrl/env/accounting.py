"""Pure JAX accounting calculations for portfolio rebalancing."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def calculate_turnover(current_weights: Array, target_weights: Array) -> Array:
    """Return full L1 turnover between two allocations."""

    return jnp.sum(jnp.abs(target_weights - current_weights))


def calculate_transaction_cost(turnover: Array, cost_rate: float | Array) -> Array:
    """Return transaction cost as a decimal return drag."""

    return jnp.asarray(cost_rate, dtype=jnp.asarray(turnover).dtype) * turnover


def calculate_liquidity_transaction_cost(
    turnover: Array,
    portfolio_value: Array,
    average_dollar_volume: float,
    spread_bps: float = 0.0,
    impact_bps: float = 0.0,
) -> Array:
    """Estimate spread plus square-root impact cost for traded notional."""

    if average_dollar_volume <= 0.0 or spread_bps < 0.0 or impact_bps < 0.0:
        raise ValueError("Liquidity inputs must be positive volume and non-negative bps.")
    participation = jnp.maximum(
        turnover * portfolio_value / average_dollar_volume, 0.0
    )
    rate = (spread_bps + impact_bps * jnp.sqrt(participation)) / 10_000.0
    return turnover * rate


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


def evolve_portfolio_weights(
    weights: Array,
    asset_returns: Array,
    cash_index: int = -1,
) -> Array:
    """Return end-of-period weights after holdings experience asset returns."""

    current = jnp.asarray(weights)
    returns = jnp.asarray(asset_returns, dtype=current.dtype)
    holding_values = current * (1.0 + returns)
    return normalize_long_only_weights(
        holding_values,
        fallback_weights=current,
        cash_index=cash_index,
    )


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


def keep_top_n_risky_weights(
    weights: Array,
    top_n: int | Array,
    cash_index: int = -1,
) -> Array:
    """Keep the largest ``top_n`` risky weights, preserve cash, and renormalize."""

    n_assets = weights.shape[0]
    resolved_cash_index = jnp.where(
        jnp.asarray(cash_index) >= 0,
        jnp.asarray(cash_index),
        n_assets + jnp.asarray(cash_index),
    )
    risky_mask = jnp.ones((n_assets,), dtype=bool).at[resolved_cash_index].set(False)
    risky_weights = jnp.where(risky_mask, weights, -jnp.inf)
    order = jnp.argsort(risky_weights, descending=True)
    ranks = jnp.empty((n_assets,), dtype=jnp.int32).at[order].set(
        jnp.arange(n_assets, dtype=jnp.int32)
    )
    keep_mask = risky_mask & (ranks < jnp.asarray(top_n, dtype=jnp.int32))
    keep_mask = keep_mask.at[resolved_cash_index].set(True)
    sparse = jnp.where(keep_mask, weights, 0.0)
    return normalize_long_only_weights(sparse, cash_index=cash_index)


def cap_risky_weights(weights: Array, cap: float, cash_index: int = -1) -> Array:
    """Cap risky positions and direct residual allocation to cash."""

    if cap <= 0.0 or cap > 1.0:
        raise ValueError("cap must be in (0, 1].")
    resolved_cash = weights.shape[0] + cash_index if cash_index < 0 else cash_index
    risky = jnp.arange(weights.shape[0]) != resolved_cash
    capped = jnp.where(risky, jnp.minimum(weights, cap), weights)
    residual = jnp.maximum(1.0 - jnp.sum(capped), 0.0)
    return normalize_long_only_weights(
        capped.at[resolved_cash].add(residual), cash_index=cash_index
    )
