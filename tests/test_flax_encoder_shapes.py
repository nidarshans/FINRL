"""Shape, JIT, and determinism tests for the production Flax encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.models import (
    AssetLSTMEncoder,
    AttentionPool,
    CrossAssetSelfAttention,
    MacroLSTMEncoder,
    ProductionEncoderConfig,
    encode_market_state_flax,
    init_encoder_train_state,
    init_encoder_variables,
)


def _tiny_config() -> ProductionEncoderConfig:
    return ProductionEncoderConfig(
        lookback=5,
        n_assets=4,
        asset_feature_dim=3,
        macro_feature_dim=2,
        asset_hidden_dim=8,
        macro_hidden_dim=6,
        attention_heads=2,
        fusion_hidden_dim=10,
        output_dim=7,
    )


def test_flax_encoder_component_shapes_match_architecture() -> None:
    config = _tiny_config()
    asset_window = jnp.arange(5 * 4 * 3, dtype=jnp.float32).reshape(5, 4, 3) / 100.0
    macro_window = jnp.arange(5 * 2, dtype=jnp.float32).reshape(5, 2) / 100.0

    asset_encoder = AssetLSTMEncoder(hidden_dim=config.asset_hidden_dim)
    asset_variables = asset_encoder.init(jax.random.PRNGKey(0), asset_window)
    asset_embeddings = asset_encoder.apply(asset_variables, asset_window)

    attention = CrossAssetSelfAttention(
        hidden_dim=config.asset_hidden_dim,
        num_heads=config.attention_heads,
    )
    attention_variables = attention.init(jax.random.PRNGKey(1), asset_embeddings)
    attended = attention.apply(attention_variables, asset_embeddings)

    pool = AttentionPool(hidden_dim=config.asset_hidden_dim)
    pool_variables = pool.init(jax.random.PRNGKey(2), attended)
    pooled = pool.apply(pool_variables, attended)

    macro_encoder = MacroLSTMEncoder(hidden_dim=config.macro_hidden_dim)
    macro_variables = macro_encoder.init(jax.random.PRNGKey(3), macro_window)
    macro_embedding = macro_encoder.apply(macro_variables, macro_window)

    assert asset_embeddings.shape == (4, 8)
    assert attended.shape == (4, 8)
    assert pooled.shape == (8,)
    assert macro_embedding.shape == (6,)
    assert jnp.isfinite(pooled).all()
    assert jnp.isfinite(macro_embedding).all()


def test_market_encoder_outputs_phi_under_jit() -> None:
    config = _tiny_config()
    variables = init_encoder_variables(jax.random.PRNGKey(4), config)
    asset_window = jnp.ones((5, 4, 3), dtype=jnp.float32)
    macro_window = jnp.ones((5, 2), dtype=jnp.float32)
    spectral_row = jnp.linspace(-1.0, 1.0, 20, dtype=jnp.float32)

    apply_encoder = jax.jit(
        lambda asset, macro, spectral: encode_market_state_flax(
            variables,
            asset,
            macro,
            spectral,
            config,
        )
    )
    phi = apply_encoder(asset_window, macro_window, spectral_row)

    assert phi.shape == (7,)
    assert jnp.isfinite(phi).all()


def test_market_encoder_supports_vmap_over_windows() -> None:
    config = _tiny_config()
    variables = init_encoder_variables(jax.random.PRNGKey(5), config)
    asset = jnp.arange(2 * 5 * 4 * 3, dtype=jnp.float32).reshape(2, 5, 4, 3) / 100.0
    macro = jnp.arange(2 * 5 * 2, dtype=jnp.float32).reshape(2, 5, 2) / 100.0
    spectral = jnp.ones((2, 20), dtype=jnp.float32)

    vmapped = jax.vmap(
        lambda asset_window, macro_window, spectral_row: encode_market_state_flax(
            variables,
            asset_window,
            macro_window,
            spectral_row,
            config,
        )
    )(asset, macro, spectral)

    assert vmapped.shape == (2, 7)
    assert jnp.isfinite(vmapped).all()


def test_same_seed_initializes_identical_encoder_outputs() -> None:
    config = _tiny_config()
    key = jax.random.PRNGKey(6)
    variables_a = init_encoder_variables(key, config)
    variables_b = init_encoder_variables(key, config)
    asset_window = jnp.ones((5, 4, 3), dtype=jnp.float32)
    macro_window = jnp.ones((5, 2), dtype=jnp.float32)
    spectral_row = jnp.ones((20,), dtype=jnp.float32)

    phi_a = encode_market_state_flax(
        variables_a,
        asset_window,
        macro_window,
        spectral_row,
        config,
    )
    phi_b = encode_market_state_flax(
        variables_b,
        asset_window,
        macro_window,
        spectral_row,
        config,
    )

    assert_allclose(phi_a, phi_b, rtol=1e-6, atol=1e-7)


def test_spectral_feature_dimension_must_be_20() -> None:
    with pytest.raises(ValueError, match="spectral_feature_dim must be 20"):
        ProductionEncoderConfig(spectral_feature_dim=19)


def test_asset_hidden_dim_must_match_attention_heads() -> None:
    with pytest.raises(ValueError, match="asset_hidden_dim must be divisible"):
        ProductionEncoderConfig(asset_hidden_dim=10, attention_heads=4)


def test_init_encoder_train_state_wraps_flax_params_with_optax() -> None:
    config = _tiny_config()
    train_state = init_encoder_train_state(
        jax.random.PRNGKey(7),
        config,
        learning_rate=1e-3,
    )

    assert train_state.step == 0
    assert "asset_lstm_encoder" in train_state.params
    assert "cross_asset_attention" in train_state.params
