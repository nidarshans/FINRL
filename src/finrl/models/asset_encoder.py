"""Asset-only PPO encoder with learned score heads."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.features.columns import selected_feature_indices
from finrl.models.score_heads import AssetScoreHeads
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class ProductionEncoderConfig:
    """Experiment-level asset-only encoder defaults."""

    lookback: int = 60
    n_assets: int = 100
    asset_feature_dim: int = 1
    asset_hidden_dim: int = 64
    score_hidden_dims: tuple[int, ...] = (16, 8)
    score_use_layer_norm: bool = True
    score_activation: str = "tanh"
    normalize_output: bool = True


@dataclass(frozen=True, slots=True)
class AssetOnlyEncoderConfig:
    """Shape and hidden-size configuration for the asset-only encoder."""

    lookback: int
    n_assets: int
    asset_feature_dim: int
    asset_hidden_dim: int = 64
    score_hidden_dims: tuple[int, ...] = (16, 8)
    score_use_layer_norm: bool = True
    score_activation: str = "tanh"
    normalize_output: bool = True


class _SharedLSTMFinalState(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, sequence: Array) -> Array:
        input_dim = sequence.shape[-1]
        batch_shape = sequence.shape[1:-1]
        gate_dim = 4 * self.hidden_dim
        w_x = self.param("w_x", nn.initializers.xavier_uniform(), (input_dim, gate_dim))
        w_h = self.param("w_h", nn.initializers.orthogonal(), (self.hidden_dim, gate_dim))
        bias = self.param("bias", nn.initializers.zeros, (gate_dim,))
        h = jnp.zeros((*batch_shape, self.hidden_dim), dtype=sequence.dtype)
        c = jnp.zeros_like(h)
        for time_index in range(sequence.shape[0]):
            gates = sequence[time_index] @ w_x + h @ w_h + bias
            input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4, axis=-1)
            c = jax.nn.sigmoid(forget_gate) * c
            c = c + jax.nn.sigmoid(input_gate) * jnp.tanh(candidate)
            h = jax.nn.sigmoid(output_gate) * jnp.tanh(c)
        return h


class AssetOnlyEncoder(nn.Module):
    """Encode raw asset windows to per-asset embeddings without cross-asset inputs."""

    config: AssetOnlyEncoderConfig
    accumulation_indices: tuple[int, ...]
    liquidity_indices: tuple[int, ...]

    @nn.compact
    def __call__(self, asset_windows: Array) -> Array:
        """Return embeddings with shape ``[B, N, D]``."""

        windows = jnp.asarray(asset_windows, dtype=jnp.float32)
        acc_components, liq_components = slice_score_head_components(
            windows,
            self.accumulation_indices,
            self.liquidity_indices,
        )
        acc_score, liq_score = AssetScoreHeads(
            hidden_dims=self.config.score_hidden_dims,
            use_layer_norm=self.config.score_use_layer_norm,
            activation=self.config.score_activation,
            name="score_heads",
        )(acc_components, liq_components)
        augmented = jnp.concatenate(
            [windows, acc_score[..., None], liq_score[..., None]],
            axis=-1,
        )
        sequence = jnp.swapaxes(augmented, 0, 1)
        embeddings = _SharedLSTMFinalState(
            self.config.asset_hidden_dim,
            name="asset_lstm_encoder",
        )(sequence)
        if self.config.normalize_output:
            embeddings = nn.LayerNorm(name="asset_embedding_norm")(embeddings)
        return embeddings


def slice_score_head_components(
    asset_windows: Array,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
) -> tuple[Array, Array]:
    """Select integer-indexed score-head inputs from asset windows."""

    windows = jnp.asarray(asset_windows, dtype=jnp.float32)
    return (
        windows[..., jnp.asarray(accumulation_indices)],
        windows[..., jnp.asarray(liquidity_indices)],
    )


def component_indices(
    feature_columns: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return explicit accumulation and liquidity-exit component indices."""

    metadata = selected_feature_indices(feature_columns)
    return metadata.accumulation_indices, metadata.liquidity_exit_indices
