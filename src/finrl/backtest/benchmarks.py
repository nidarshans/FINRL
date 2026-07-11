"""Deterministic benchmark allocation policies."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def benchmark_actions(
    n_steps: int,
    n_assets: int,
    policy: str,
) -> Array:
    """Return benchmark target weights with a cash column.

    Supported policies are ``equal_weight``, ``equal_weight_cash``, and
    ``cash``.  The returned shape is ``[time, risky_assets + 1]``.
    """

    if n_steps < 0 or n_assets <= 0:
        raise ValueError("n_steps must be non-negative and n_assets must be positive.")
    if policy == "equal_weight":
        weights = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[:n_assets].set(
            1.0 / n_assets
        )
    elif policy == "equal_weight_cash":
        weights = jnp.ones((n_assets + 1,), dtype=jnp.float32) / (n_assets + 1)
    elif policy == "cash":
        weights = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[-1].set(1.0)
    else:
        raise ValueError(f"Unknown benchmark policy: {policy}")
    return jnp.repeat(weights[None, :], n_steps, axis=0)
