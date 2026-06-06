"""Pure JAX market encoder for asset, macro, and spectral features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

from finrl.models.attention import AttentionPooling, CrossAssetAttention, _glorot_uniform

Array: TypeAlias = jax.Array
Params: TypeAlias = dict[str, object]


class FeatureWindow(NamedTuple):
    """One no-look-ahead encoder input window."""

    asset: Array
    macro: Array
    spectral: Array


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Shape and hidden-size configuration for the market encoder."""

    lookback: int = 60
    n_assets: int = 100
    asset_feature_dim: int = 1
    macro_feature_dim: int = 1
    spectral_feature_dim: int = 20
    asset_hidden_dim: int = 64
    macro_hidden_dim: int = 16
    fusion_hidden_dim: int = 64
    output_dim: int = 32

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
        if self.fusion_hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("fusion and output dimensions must be positive.")


def _zeros(shape: tuple[int, ...]) -> Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def _init_lstm_params(key: Array, input_dim: int, hidden_dim: int) -> dict[str, Array]:
    w_key, b_key = jax.random.split(key)
    del b_key
    return {
        "w": _glorot_uniform(w_key, (input_dim + hidden_dim, 4 * hidden_dim)),
        "b": _zeros((4 * hidden_dim,)),
    }


def _lstm_sequence(params: dict[str, Array], inputs: Array, hidden_dim: int) -> Array:
    """Run an LSTM over leading time dimension and return final hidden state."""

    batch_shape = inputs.shape[1:-1]
    h0 = _zeros((*batch_shape, hidden_dim))
    c0 = _zeros((*batch_shape, hidden_dim))

    def step(carry: tuple[Array, Array], x_t: Array) -> tuple[tuple[Array, Array], Array]:
        h, c = carry
        gates = jnp.concatenate([x_t, h], axis=-1) @ params["w"] + params["b"]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        next_c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        next_h = jax.nn.sigmoid(o) * jnp.tanh(next_c)
        return (next_h, next_c), next_h

    (final_h, _), _ = jax.lax.scan(step, (h0, c0), inputs)
    return final_h


@dataclass(frozen=True, slots=True)
class AssetEncoder:
    """Shared LSTM encoder applied independently to each asset."""

    config: EncoderConfig

    def init(self, key: Array) -> dict[str, Array]:
        """Initialize shared asset LSTM parameters."""

        return _init_lstm_params(
            key,
            self.config.asset_feature_dim,
            self.config.asset_hidden_dim,
        )

    def apply(self, params: dict[str, Array], asset_window: Array) -> Array:
        """Encode ``(lookback, n_assets, asset_feature_dim)`` to ``(n_assets, 64)``."""

        return _lstm_sequence(params, asset_window, self.config.asset_hidden_dim)


@dataclass(frozen=True, slots=True)
class MacroEncoder:
    """LSTM encoder for macro features."""

    config: EncoderConfig

    def init(self, key: Array) -> dict[str, Array]:
        """Initialize macro LSTM parameters."""

        return _init_lstm_params(
            key,
            self.config.macro_feature_dim,
            self.config.macro_hidden_dim,
        )

    def apply(self, params: dict[str, Array], macro_window: Array) -> Array:
        """Encode ``(lookback, macro_feature_dim)`` to ``(16,)``."""

        return _lstm_sequence(params, macro_window, self.config.macro_hidden_dim)


@dataclass(frozen=True, slots=True)
class FusionMLP:
    """Fusion network mapping asset, macro, and spectral embeddings to ``phi_t``."""

    config: EncoderConfig

    def init(self, key: Array) -> dict[str, Array]:
        """Initialize fusion MLP parameters."""

        w1_key, w2_key = jax.random.split(key)
        input_dim = (
            self.config.asset_hidden_dim
            + self.config.macro_hidden_dim
            + self.config.spectral_feature_dim
        )
        return {
            "w1": _glorot_uniform(w1_key, (input_dim, self.config.fusion_hidden_dim)),
            "b1": _zeros((self.config.fusion_hidden_dim,)),
            "w2": _glorot_uniform(
                w2_key,
                (self.config.fusion_hidden_dim, self.config.output_dim),
            ),
            "b2": _zeros((self.config.output_dim,)),
        }

    def apply(self, params: dict[str, Array], fused_input: Array) -> Array:
        """Apply the fusion MLP."""

        hidden = jax.nn.relu(fused_input @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]


@dataclass(frozen=True, slots=True)
class MarketEncoder:
    """End-to-end market state encoder producing ``phi_t in R^32``."""

    config: EncoderConfig

    def init(self, key: Array) -> Params:
        """Initialize all encoder parameters."""

        asset_key, attention_key, pooling_key, macro_key, fusion_key = jax.random.split(key, 5)
        return {
            "asset_encoder": AssetEncoder(self.config).init(asset_key),
            "cross_asset_attention": CrossAssetAttention(
                self.config.asset_hidden_dim
            ).init(attention_key),
            "attention_pooling": AttentionPooling(self.config.asset_hidden_dim).init(
                pooling_key
            ),
            "macro_encoder": MacroEncoder(self.config).init(macro_key),
            "fusion_mlp": FusionMLP(self.config).init(fusion_key),
        }

    def apply(self, params: Params, feature_window: FeatureWindow) -> Array:
        """Encode one feature window into a 32-dimensional market state."""

        return encode_market_state(params, feature_window)


def encode_market_state(params: Params, feature_window: FeatureWindow) -> Array:
    """Encode one feature window using initialized ``MarketEncoder`` parameters."""

    asset_embeddings = _lstm_sequence(
        params["asset_encoder"],  # type: ignore[arg-type]
        feature_window.asset,
        params["asset_encoder"]["b"].shape[0] // 4,  # type: ignore[index]
    )
    attended_assets = CrossAssetAttention(asset_embeddings.shape[-1]).apply(
        params["cross_asset_attention"],  # type: ignore[arg-type]
        asset_embeddings,
    )
    pooled_assets = AttentionPooling(attended_assets.shape[-1]).apply(
        params["attention_pooling"],  # type: ignore[arg-type]
        attended_assets,
    )
    macro_embedding = _lstm_sequence(
        params["macro_encoder"],  # type: ignore[arg-type]
        feature_window.macro,
        params["macro_encoder"]["b"].shape[0] // 4,  # type: ignore[index]
    )
    fused = jnp.concatenate(
        [
            pooled_assets,
            macro_embedding,
            feature_window.spectral.astype(jnp.float32),
        ],
        axis=-1,
    )
    return FusionMLP(
        EncoderConfig(
            asset_hidden_dim=pooled_assets.shape[-1],
            macro_hidden_dim=macro_embedding.shape[-1],
            spectral_feature_dim=feature_window.spectral.shape[-1],
            fusion_hidden_dim=params["fusion_mlp"]["b1"].shape[0],  # type: ignore[index]
            output_dim=params["fusion_mlp"]["b2"].shape[0],  # type: ignore[index]
        )
    ).apply(params["fusion_mlp"], fused)  # type: ignore[arg-type]

