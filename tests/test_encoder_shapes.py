"""Shape and JIT tests for the JAX market encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.models import (
    AssetEncoder,
    AttentionPooling,
    CrossAssetAttention,
    EncoderConfig,
    FeatureWindow,
    MacroEncoder,
    MarketEncoder,
    encode_market_state,
)


def test_encoder_component_shapes_match_architecture() -> None:
    config = EncoderConfig(
        lookback=60,
        n_assets=100,
        asset_feature_dim=3,
        macro_feature_dim=4,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    asset_window = jnp.ones((60, 100, 3), dtype=jnp.float32)
    macro_window = jnp.ones((60, 4), dtype=jnp.float32)

    asset_encoder = AssetEncoder(config)
    asset_embeddings = asset_encoder.apply(asset_encoder.init(keys[0]), asset_window)
    attended = CrossAssetAttention(config.asset_hidden_dim).apply(
        CrossAssetAttention(config.asset_hidden_dim).init(keys[1]),
        asset_embeddings,
    )
    pooled = AttentionPooling(config.asset_hidden_dim).apply(
        AttentionPooling(config.asset_hidden_dim).init(keys[2]),
        attended,
    )
    macro_embedding = MacroEncoder(config).apply(
        MacroEncoder(config).init(keys[3]),
        macro_window,
    )

    assert asset_embeddings.shape == (100, 64)
    assert attended.shape == (100, 64)
    assert pooled.shape == (64,)
    assert macro_embedding.shape == (16,)


def test_market_encoder_outputs_phi_32_under_jit() -> None:
    config = EncoderConfig(
        lookback=6,
        n_assets=5,
        asset_feature_dim=3,
        macro_feature_dim=4,
    )
    encoder = MarketEncoder(config)
    params = encoder.init(jax.random.PRNGKey(1))
    feature_window = FeatureWindow(
        asset=jnp.arange(6 * 5 * 3, dtype=jnp.float32).reshape(6, 5, 3) / 100.0,
        macro=jnp.arange(6 * 4, dtype=jnp.float32).reshape(6, 4) / 50.0,
        spectral=jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float32),
    )

    phi = jax.jit(lambda window: encode_market_state(params, window))(feature_window)

    assert phi.shape == (32,)
    assert jnp.isfinite(phi).all()


def test_market_encoder_supports_vmap_and_scan() -> None:
    config = EncoderConfig(
        lookback=4,
        n_assets=3,
        asset_feature_dim=2,
        macro_feature_dim=2,
    )
    encoder = MarketEncoder(config)
    params = encoder.init(jax.random.PRNGKey(3))
    asset = jnp.arange(2 * 4 * 3 * 2, dtype=jnp.float32).reshape(2, 4, 3, 2) / 100.0
    macro = jnp.arange(2 * 4 * 2, dtype=jnp.float32).reshape(2, 4, 2) / 100.0
    spectral = jnp.ones((2, 20), dtype=jnp.float32)

    vmapped = jax.vmap(
        lambda asset_window, macro_window, spectral_row: encode_market_state(
            params,
            FeatureWindow(asset_window, macro_window, spectral_row),
        )
    )(asset, macro, spectral)

    def scan_step(_: None, inputs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[None, jax.Array]:
        asset_window, macro_window, spectral_row = inputs
        return None, encode_market_state(
            params,
            FeatureWindow(asset_window, macro_window, spectral_row),
        )

    _, scanned = jax.lax.scan(scan_step, None, (asset, macro, spectral))

    assert vmapped.shape == (2, 32)
    assert scanned.shape == (2, 32)
    assert_allclose(vmapped, scanned, rtol=1e-6, atol=1e-7)


def test_encoder_initialization_and_forward_pass_are_deterministic() -> None:
    config = EncoderConfig(
        lookback=4,
        n_assets=3,
        asset_feature_dim=2,
        macro_feature_dim=2,
    )
    encoder = MarketEncoder(config)
    key = jax.random.PRNGKey(2)
    params_a = encoder.init(key)
    params_b = encoder.init(key)
    feature_window = FeatureWindow(
        asset=jnp.ones((4, 3, 2), dtype=jnp.float32),
        macro=jnp.ones((4, 2), dtype=jnp.float32),
        spectral=jnp.ones((20,), dtype=jnp.float32),
    )

    phi_a = encoder.apply(params_a, feature_window)
    phi_b = encoder.apply(params_b, feature_window)

    assert_allclose(phi_a, phi_b, rtol=1e-6, atol=1e-7)


def test_spectral_feature_dimension_must_be_20() -> None:
    with pytest.raises(ValueError, match="spectral_feature_dim must be 20"):
        EncoderConfig(spectral_feature_dim=19)
