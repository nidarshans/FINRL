"""Train-state helpers for the production Flax encoder."""

from __future__ import annotations

import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finrl.models.flax_encoder import MarketEncoderFlax, ProductionEncoderConfig
from finrl.types import Array


def init_encoder_train_state(
    rng: Array,
    config: ProductionEncoderConfig,
    learning_rate: float = 1e-3,
) -> TrainState:
    """Initialize encoder parameters and wrap them in an Adam train state."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

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
    variables = module.init(rng, asset_window, macro_window, spectral_row)
    optimizer = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=variables["params"],
        tx=optimizer,
    )
