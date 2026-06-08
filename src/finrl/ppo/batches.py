"""Minibatch helpers for production PPO rollout buffers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from finrl.ppo.rollout import RolloutBatch
from finrl.types import Array


def rollout_length(batch: RolloutBatch) -> int:
    """Return the shared leading time dimension for a rollout batch."""

    return int(batch.rewards.shape[0])


def validate_rollout_batch(batch: RolloutBatch) -> None:
    """Ensure all rollout arrays share the same leading time dimension."""

    length = rollout_length(batch)
    for name, value in batch._asdict().items():
        leaves = jax.tree_util.tree_leaves(value)
        if any(leaf.shape[0] != length for leaf in leaves):
            raise ValueError(f"{name} leading dimension does not match rewards.")


def shuffle_rollout_indices(rng: Array, n_steps: int) -> Array:
    """Return a deterministic permutation for minibatch construction."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    return jax.random.permutation(rng, jnp.arange(n_steps))


def _take_batch(batch: RolloutBatch, indices: Array) -> RolloutBatch:
    return RolloutBatch(
        **{
            name: jax.tree_util.tree_map(lambda leaf: leaf[indices], value)
            for name, value in batch._asdict().items()
        }
    )


def make_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    rng: Array | None = None,
    shuffle: bool = True,
) -> tuple[RolloutBatch, ...]:
    """Split rollout tensors into aligned minibatches."""

    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    validate_rollout_batch(batch)
    n_steps = rollout_length(batch)
    if shuffle:
        if rng is None:
            raise ValueError("rng is required when shuffle is True.")
        indices = shuffle_rollout_indices(rng, n_steps)
    else:
        indices = jnp.arange(n_steps)

    minibatches = []
    for start in range(0, n_steps, minibatch_size):
        stop = min(start + minibatch_size, n_steps)
        minibatches.append(_take_batch(batch, indices[start:stop]))
    return tuple(minibatches)
