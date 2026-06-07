"""Production PPO scalar metrics."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from finrl.types import Array


@dataclass(frozen=True, slots=True)
class PPOTrainMetrics:
    """Scalar diagnostics reported by production PPO updates."""

    policy_loss: Array
    actor_loss: Array
    critic_loss: Array
    total_loss: Array
    entropy: Array
    approx_kl: Array
    post_update_approx_kl: Array
    clip_fraction: Array
    explained_variance: Array
    grad_norm: Array
    mean_episode_return: Array
    mean_reward: Array
    advantage_mean: Array
    advantage_std: Array
    ratio_mean: Array
    ratio_min: Array
    ratio_max: Array
    mean_turnover: Array
    mean_transaction_cost: Array
    mean_drawdown: Array
    updates_completed: Array
    epochs_completed: Array


def explained_variance(values: Array, returns: Array) -> Array:
    """Return explained variance of value predictions against returns."""

    target_variance = jnp.var(returns)
    residual_variance = jnp.var(returns - values)
    return jnp.where(
        target_variance > 1e-8,
        1.0 - residual_variance / (target_variance + 1e-8),
        jnp.asarray(0.0, dtype=values.dtype),
    )


def approximate_kl(old_log_probs: Array, new_log_probs: Array) -> Array:
    """Return PPO approximate KL from frozen old log-probs."""

    return jnp.mean(old_log_probs - new_log_probs)


def clip_fraction(
    old_log_probs: Array,
    new_log_probs: Array,
    clip_epsilon: float,
) -> Array:
    """Return fraction of policy ratios outside the PPO clip range."""

    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped = jnp.abs(ratio - 1.0) > clip_epsilon
    return jnp.mean(clipped.astype(jnp.float32))
