"""No-look-ahead tests for production Flax encoder usage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.models import (
    ProductionEncoderConfig,
    encode_market_state_flax,
    init_encoder_variables,
)


def test_future_window_changes_do_not_change_prior_flax_encoding() -> None:
    """Encoding one window must not depend on later windows in a batch."""

    config = ProductionEncoderConfig(
        lookback=4,
        n_assets=3,
        asset_feature_dim=2,
        macro_feature_dim=2,
        asset_hidden_dim=8,
        macro_hidden_dim=4,
        attention_heads=2,
    )
    variables = init_encoder_variables(jax.random.PRNGKey(0), config)
    asset = jnp.arange(2 * 4 * 3 * 2, dtype=jnp.float32).reshape(2, 4, 3, 2) / 100.0
    macro = jnp.arange(2 * 4 * 2, dtype=jnp.float32).reshape(2, 4, 2) / 100.0
    spectral = jnp.arange(2 * 20, dtype=jnp.float32).reshape(2, 20) / 100.0

    def encode_batch(
        asset_batch: jax.Array,
        macro_batch: jax.Array,
        spectral_batch: jax.Array,
    ) -> jax.Array:
        return jax.vmap(
            lambda asset_window, macro_window, spectral_row: encode_market_state_flax(
                variables,
                asset_window,
                macro_window,
                spectral_row,
                config,
            )
        )(asset_batch, macro_batch, spectral_batch)

    baseline = encode_batch(asset, macro, spectral)
    changed_future = encode_batch(
        asset.at[1].set(999.0),
        macro.at[1].set(-999.0),
        spectral.at[1].set(500.0),
    )

    assert_allclose(baseline[0], changed_future[0], rtol=1e-6, atol=1e-7)
    assert not jnp.allclose(baseline[1], changed_future[1])
