"""Learned per-asset score heads for PPO-trained OHLCV components."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from finrl.types import Array


class ScoreMLP(nn.Module):
    """Small MLP that maps component features to one scalar score."""

    hidden_dims: tuple[int, ...] = (16, 8)

    @nn.compact
    def __call__(self, components: Array) -> Array:
        x = jnp.asarray(components, dtype=jnp.float32)
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            x = jnp.tanh(x)
        return nn.Dense(1, name="score")(x).squeeze(axis=-1)


class AssetScoreHeads(nn.Module):
    """Compute learned accumulation and liquidity scores."""

    hidden_dims: tuple[int, ...] = (16, 8)

    @nn.compact
    def __call__(
        self,
        accumulation_components: Array,
        liquidity_components: Array,
    ) -> tuple[Array, Array]:
        accumulation = ScoreMLP(
            hidden_dims=self.hidden_dims,
            name="accumulation",
        )(accumulation_components)
        liquidity = ScoreMLP(
            hidden_dims=self.hidden_dims,
            name="liquidity",
        )(liquidity_components)
        return accumulation, liquidity
