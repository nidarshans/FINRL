"""Learned per-asset accumulation and liquidity-exit score heads."""

from __future__ import annotations

import jax
from flax import linen as nn
import jax.numpy as jnp

from finrl.types import Array


class ScoreMLP(nn.Module):
    """Small MLP that maps component features to one scalar score."""

    hidden_dims: tuple[int, ...] = (16, 8)
    use_layer_norm: bool = True
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"

    @nn.compact
    def __call__(self, components: Array) -> Array:
        x = jnp.asarray(components, dtype=jnp.float32)
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"hidden_norm_{index}")(x)
            x = _activation(self.hidden_activation)(x)
        score = nn.Dense(1, name="score")(x).squeeze(axis=-1)
        return _activation(self.output_activation)(score)


class AssetScoreHeads(nn.Module):
    """Compute learned accumulation and liquidity scores."""

    accumulation_hidden_dims: tuple[int, ...] = (16, 8)
    accumulation_use_layer_norm: bool = True
    accumulation_hidden_activation: str = "tanh"
    accumulation_output_activation: str = "sigmoid"
    liquidity_exit_hidden_dims: tuple[int, ...] = (16, 8)
    liquidity_exit_use_layer_norm: bool = True
    liquidity_exit_hidden_activation: str = "tanh"
    liquidity_exit_output_activation: str = "sigmoid"

    @nn.compact
    def __call__(
        self,
        accumulation_components: Array,
        liquidity_components: Array,
    ) -> tuple[Array, Array]:
        accumulation = ScoreMLP(
            hidden_dims=self.accumulation_hidden_dims,
            use_layer_norm=self.accumulation_use_layer_norm,
            hidden_activation=self.accumulation_hidden_activation,
            output_activation=self.accumulation_output_activation,
            name="accumulation",
        )(accumulation_components)
        liquidity = ScoreMLP(
            hidden_dims=self.liquidity_exit_hidden_dims,
            use_layer_norm=self.liquidity_exit_use_layer_norm,
            hidden_activation=self.liquidity_exit_hidden_activation,
            output_activation=self.liquidity_exit_output_activation,
            name="liquidity",
        )(liquidity_components)
        return accumulation, liquidity


def slice_score_head_components(
    asset_features: Array,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
) -> tuple[Array, Array]:
    """Select explicitly routed score-head inputs from an asset panel."""

    features = jnp.asarray(asset_features, dtype=jnp.float32)
    return (
        features[..., jnp.asarray(accumulation_indices)],
        features[..., jnp.asarray(liquidity_indices)],
    )


def _activation(name: str):
    if name == "identity":
        return lambda value: value
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "tanh":
        return jnp.tanh
    if name == "gelu":
        return nn.gelu
    if name == "relu":
        return nn.relu
    raise ValueError(f"Unknown score-head activation: {name}")
