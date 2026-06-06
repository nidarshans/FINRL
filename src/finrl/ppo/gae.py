"""Generalized advantage estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from finrl.types import Array


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float,
    lambda_: float,
) -> tuple[Array, Array]:
    """Return advantages and value targets for a trajectory."""

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be 1D arrays.")
    if dones.shape != rewards.shape:
        raise ValueError("dones must match rewards shape.")
    if values.shape[0] not in (rewards.shape[0], rewards.shape[0] + 1):
        raise ValueError("values length must be T or T + 1.")
    if values.shape[0] == rewards.shape[0]:
        values = jnp.concatenate([values, jnp.zeros((1,), dtype=values.dtype)])

    def step(next_values: tuple[Array, Array], inputs: tuple[Array, Array, Array, Array]):
        next_advantage, next_value = next_values
        reward, value, done, value_tp1 = inputs
        mask = 1.0 - done
        delta = reward + gamma * value_tp1 * mask - value
        advantage = delta + gamma * lambda_ * mask * next_advantage
        return (advantage, value), advantage

    _, reversed_advantages = jax.lax.scan(
        step,
        (jnp.array(0.0, dtype=rewards.dtype), values[-1]),
        (rewards[::-1], values[:-1][::-1], dones[::-1], values[1:][::-1]),
    )
    advantages = reversed_advantages[::-1]
    returns = advantages + values[:-1]
    return advantages, returns
