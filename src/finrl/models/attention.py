"""JAX attention blocks for cross-asset market encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array
Params: TypeAlias = dict[str, Array]


def _glorot_uniform(key: Array, shape: tuple[int, ...]) -> Array:
    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


@dataclass(frozen=True, slots=True)
class CrossAssetAttention:
    """Single-head self-attention across asset embeddings."""

    hidden_dim: int = 64

    def init(self, key: Array) -> Params:
        """Initialize attention parameters."""

        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        shape = (self.hidden_dim, self.hidden_dim)
        return {
            "w_q": _glorot_uniform(q_key, shape),
            "w_k": _glorot_uniform(k_key, shape),
            "w_v": _glorot_uniform(v_key, shape),
            "w_o": _glorot_uniform(o_key, shape),
        }

    def apply(self, params: Params, asset_embeddings: Array) -> Array:
        """Apply self-attention to ``(n_assets, hidden_dim)`` embeddings."""

        q = asset_embeddings @ params["w_q"]
        k = asset_embeddings @ params["w_k"]
        v = asset_embeddings @ params["w_v"]
        scale = jnp.sqrt(jnp.asarray(asset_embeddings.shape[-1], dtype=asset_embeddings.dtype))
        weights = jax.nn.softmax((q @ k.T) / scale, axis=-1)
        return (weights @ v) @ params["w_o"]


@dataclass(frozen=True, slots=True)
class AttentionPooling:
    """Attention pooling from asset embeddings to one market vector."""

    hidden_dim: int = 64

    def init(self, key: Array) -> Params:
        """Initialize pooling parameters."""

        return {
            "query": jax.random.normal(key, (self.hidden_dim,)) / jnp.sqrt(self.hidden_dim)
        }

    def apply(self, params: Params, asset_embeddings: Array) -> Array:
        """Pool ``(n_assets, hidden_dim)`` embeddings into ``(hidden_dim,)``."""

        logits = asset_embeddings @ params["query"]
        weights = jax.nn.softmax(logits, axis=0)
        return weights @ asset_embeddings

