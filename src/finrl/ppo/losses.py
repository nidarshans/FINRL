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


def huber_value_loss(values: Array, returns: Array, delta: float = 1.0) -> Array:
    """Return Huber value prediction loss for large-return robustness."""

    error = values - returns
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * jnp.square(quadratic) + delta * linear)


def clipped_value_loss(
    values: Array,
    old_values: Array,
    returns: Array,
    clip_epsilon: float,
) -> Array:
    """Return PPO clipped value-function loss."""

    clipped_values = old_values + jnp.clip(
        values - old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    unclipped_loss = jnp.square(values - returns)
    clipped_loss = jnp.square(clipped_values - returns)
    return jnp.mean(jnp.maximum(unclipped_loss, clipped_loss))


def entropy_bonus(entropies: Array) -> Array:
    """Return mean policy entropy."""

    return jnp.mean(entropies)


def ppo_actor_loss(
    new_log_probs: Array,
    old_log_probs: Array,
    advantages: Array,
    clip_epsilon: float,
) -> Array:
    """Return production PPO clipped actor loss."""

    return ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_epsilon)


def critic_loss(
    values: Array,
    returns: Array,
    old_values: Array | None = None,
    clip_epsilon: float = 0.2,
    use_clipping: bool = True,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> Array:
    """Return production critic loss with optional PPO value clipping."""

    if old_values is not None and use_clipping:
        return clipped_value_loss(values, old_values, returns, clip_epsilon)
    if loss_type == "huber":
        return huber_value_loss(values, returns, huber_delta)
    if loss_type != "mse":
        raise ValueError("loss_type must be 'mse' or 'huber'.")
    return value_loss(values, returns)


def ppo_total_loss(
    actor_loss: Array,
    critic_loss_value: Array,
    entropy: Array,
    value_coef: float,
    entropy_coef: float,
) -> Array:
    """Combine actor, critic, and entropy terms into PPO total loss."""

    return actor_loss + value_coef * critic_loss_value - entropy_coef * entropy
