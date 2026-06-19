"""Activation helpers for direct portfolio optimization."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def sparsemax(logits: Array, axis: int = -1) -> Array:
    """Project logits onto the probability simplex with exact zeros."""

    values = jnp.asarray(logits, dtype=jnp.float32)
    axis = axis if axis >= 0 else values.ndim + axis
    moved = jnp.moveaxis(values, axis, -1)
    z = moved - jnp.max(moved, axis=-1, keepdims=True)
    z_sorted = jnp.flip(jnp.sort(z, axis=-1), axis=-1)
    k = jnp.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    z_cumsum = jnp.cumsum(z_sorted, axis=-1)
    support = 1.0 + k * z_sorted > z_cumsum
    k_z = jnp.sum(support, axis=-1, keepdims=True)
    tau_sum = jnp.take_along_axis(
        z_cumsum,
        jnp.maximum(k_z.astype(jnp.int32) - 1, 0),
        axis=-1,
    )
    tau = (tau_sum - 1.0) / k_z
    projected = jnp.maximum(z - tau, 0.0)
    return jnp.moveaxis(projected, -1, axis)
