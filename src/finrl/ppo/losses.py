"""PPO loss functions."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.types import Array


def ppo_clip_loss(
    new_logprobs: Array,
    old_logprobs: Array,
    advantages: Array,
    clip_epsilon: float,
) -> Array:
    """Return clipped PPO actor loss to minimize."""

    ratio = jnp.exp(new_logprobs - old_logprobs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    return -jnp.mean(objective)


def value_loss(values: Array, returns: Array) -> Array:
    """Return mean squared value prediction loss."""

    return jnp.mean(jnp.square(values - returns))


def entropy_bonus(entropies: Array) -> Array:
    """Return mean policy entropy."""

    return jnp.mean(entropies)

