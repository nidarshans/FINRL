"""Production Flax market encoder.

The production encoder maps no-look-ahead feature windows to the latent
components used by the per-asset allocator:

1. shared asset LSTM over each asset's lookback sequence,
2. cross-asset self-attention,
3. attention pooling over assets into a market vector,
4. macro LSTM over the macro lookback sequence,
5. pass-through current-date spectral state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.types import Array


@dataclass(frozen=True, slots=True)
class ProductionEncoderConfig:
    """Shape and hidden-size configuration for the production encoder."""

    lookback: int = 60
    n_assets: int = 100
    asset_feature_dim: int = 1
    macro_feature_dim: int = 1
    spectral_feature_dim: int = 20
    asset_hidden_dim: int = 64
    macro_hidden_dim: int = 16
    attention_heads: int = 4
    normalize_output: bool = True

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive.")
        if self.n_assets <= 0:
            raise ValueError("n_assets must be positive.")
        if self.asset_feature_dim <= 0 or self.macro_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive.")
        if self.spectral_feature_dim != 20:
            raise ValueError("spectral_feature_dim must be 20.")
        if self.asset_hidden_dim <= 0 or self.macro_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.asset_hidden_dim % self.attention_heads != 0:
            raise ValueError("asset_hidden_dim must be divisible by attention_heads.")


class EncoderOutput(NamedTuple):
    """Per-asset and global market representations for downstream models."""

    asset_embeddings: Array
    market_vector: Array
    macro_state: Array
    spectral_state: Array


class _LSTMFinalState(nn.Module):
    """Small readable LSTM that returns only the final hidden state.

    The leading dimension is time. Any middle dimensions are treated as batch
    dimensions, which lets the same cell encode all assets with shared weights:
    ``(L, N, F) -> (N, H)``.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, sequence: Array) -> Array:
        input_dim = sequence.shape[-1]
        batch_shape = sequence.shape[1:-1]
        gate_dim = 4 * self.hidden_dim

        w_x = self.param(
            "w_x",
            nn.initializers.xavier_uniform(),
            (input_dim, gate_dim),
        )
        w_h = self.param(
            "w_h",
            nn.initializers.orthogonal(),
            (self.hidden_dim, gate_dim),
        )
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


class AssetLSTMEncoder(nn.Module):
    """Shared LSTM encoder for asset windows.

    All assets use the same LSTM parameters. The asset axis is a batch axis, so
    no ticker-specific recurrent parameters are created.
    """

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, asset_window: Array) -> Array:
        """Encode ``(lookback, n_assets, asset_features)`` to ``(n_assets, H)``."""

        return _LSTMFinalState(self.hidden_dim, name="shared_lstm")(asset_window)


class MacroLSTMEncoder(nn.Module):
    """LSTM encoder for macro lookback windows."""

    hidden_dim: int = 16

    @nn.compact
    def __call__(self, macro_window: Array) -> Array:
        """Encode ``(lookback, macro_features)`` to ``(H,)``."""

        return _LSTMFinalState(self.hidden_dim, name="macro_lstm")(macro_window)


class CrossAssetSelfAttention(nn.Module):
    """Multi-head self-attention over the asset dimension."""

    hidden_dim: int = 64
    num_heads: int = 4

    def setup(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

    @nn.compact
    def __call__(self, asset_embeddings: Array) -> Array:
        """Project ``(n_assets, H)`` to attended embeddings with the same shape."""

        n_assets = asset_embeddings.shape[0]
        head_dim = self.hidden_dim // self.num_heads
        qkv = nn.Dense(3 * self.hidden_dim, name="qkv")(asset_embeddings)
        qkv = qkv.reshape(n_assets, 3, self.num_heads, head_dim)
        query, key, value = jnp.moveaxis(qkv, 1, 0)

        scores = jnp.einsum("ihd,jhd->hij", query, key)
        scores = scores / jnp.sqrt(jnp.asarray(head_dim, dtype=asset_embeddings.dtype))
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("hij,jhd->ihd", weights, value)
        attended = attended.reshape(n_assets, self.hidden_dim)
        return nn.Dense(self.hidden_dim, name="output_projection")(attended)


class AttentionPool(nn.Module):
    """Learned attention pooling from per-asset embeddings to one vector."""

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, asset_embeddings: Array) -> Array:
        """Pool ``(n_assets, H)`` to ``(H,)``."""

        scores = nn.Dense(self.hidden_dim, name="score_hidden")(asset_embeddings)
        scores = jnp.tanh(scores)
        logits = nn.Dense(1, use_bias=False, name="score")(scores).squeeze(axis=-1)
        weights = jax.nn.softmax(logits, axis=0)
        return jnp.einsum("n,nh->h", weights, asset_embeddings)


class MarketEncoderFlax(nn.Module):
    """End-to-end production market encoder producing pooled market vectors."""

    config: ProductionEncoderConfig

    @nn.compact
    def __call__(
        self,
        asset_window: Array,
        macro_window: Array,
        spectral_row: Array,
    ) -> Array:
        """Encode one market window into the pooled asset market vector."""

        asset_embeddings = AssetLSTMEncoder(
            hidden_dim=self.config.asset_hidden_dim,
            name="asset_lstm_encoder",
        )(asset_window)
        attended_assets = CrossAssetSelfAttention(
            hidden_dim=self.config.asset_hidden_dim,
            num_heads=self.config.attention_heads,
            name="cross_asset_attention",
        )(asset_embeddings)
        pooled_assets = AttentionPool(
            hidden_dim=self.config.asset_hidden_dim,
            name="asset_attention_pool",
        )(attended_assets)
        if self.config.normalize_output:
            pooled_assets = nn.LayerNorm(name="market_vector_norm")(pooled_assets)
        return pooled_assets

    @nn.compact
    def encode_with_latents(
        self,
        asset_window: Array,
        macro_window: Array,
        spectral_row: Array,
    ) -> EncoderOutput:
        """Encode one market window and expose diagram-level components."""

        asset_embeddings = AssetLSTMEncoder(
            hidden_dim=self.config.asset_hidden_dim,
            name="asset_lstm_encoder",
        )(asset_window)
        attended_assets = CrossAssetSelfAttention(
            hidden_dim=self.config.asset_hidden_dim,
            num_heads=self.config.attention_heads,
            name="cross_asset_attention",
        )(asset_embeddings)
        pooled_assets = AttentionPool(
            hidden_dim=self.config.asset_hidden_dim,
            name="asset_attention_pool",
        )(attended_assets)
        macro_embedding = MacroLSTMEncoder(
            hidden_dim=self.config.macro_hidden_dim,
            name="macro_lstm_encoder",
        )(macro_window)

        if self.config.normalize_output:
            pooled_assets = nn.LayerNorm(name="market_vector_norm")(pooled_assets)
        return EncoderOutput(
            asset_embeddings=attended_assets,
            market_vector=pooled_assets,
            macro_state=macro_embedding,
            spectral_state=spectral_row.astype(jnp.float32),
        )


def init_encoder_variables(
    rng: Array,
    config: ProductionEncoderConfig,
) -> dict[str, object]:
    """Initialize production encoder variables for one example window."""

    module = MarketEncoderFlax(config)
    asset_window = jnp.zeros(
        (config.lookback, config.n_assets, config.asset_feature_dim),
        dtype=jnp.float32,
    )
    macro_window = jnp.zeros(
        (config.lookback, config.macro_feature_dim),
        dtype=jnp.float32,
    )
    spectral_row = jnp.zeros((config.spectral_feature_dim,), dtype=jnp.float32)
    return module.init(
        rng,
        asset_window,
        macro_window,
        spectral_row,
        method=MarketEncoderFlax.encode_with_latents,
    )


def encode_market_state_flax(
    variables: dict[str, object],
    asset_window: Array,
    macro_window: Array,
    spectral_row: Array,
    config: ProductionEncoderConfig,
) -> Array:
    """Apply the production encoder to one feature window."""

    return MarketEncoderFlax(config).apply(variables, asset_window, macro_window, spectral_row)


def encode_market_state_with_latents_flax(
    variables: dict[str, object],
    asset_window: Array,
    macro_window: Array,
    spectral_row: Array,
    config: ProductionEncoderConfig,
) -> EncoderOutput:
    """Apply the production encoder and return per-asset plus global latents."""

    return MarketEncoderFlax(config).apply(
        variables,
        asset_window,
        macro_window,
        spectral_row,
        method=MarketEncoderFlax.encode_with_latents,
    )
