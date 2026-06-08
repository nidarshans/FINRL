"""Phase A tests for the production Flax/JAX boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from finrl.models import (
    MarketEncoderFlax,
    ProductionEncoderConfig,
    encode_market_state_flax,
    init_encoder_variables,
)
from finrl.ppo import (
    PortfolioActorFlax,
    PortfolioCriticFlax,
    ProductionPPOConfig,
    actor_mean_weights,
    build_structured_ppo_state,
    initialize_ppo_train_state,
)


def test_flax_and_optax_imports_are_available_on_cpu() -> None:
    import flax
    import optax

    assert flax.__version__
    assert optax.__version__
    assert jax.devices()[0].platform in {"cpu", "gpu", "tpu"}


def test_production_encoder_initializes_and_runs_on_tiny_arrays() -> None:
    config = ProductionEncoderConfig(
        lookback=4,
        n_assets=3,
        asset_feature_dim=2,
        macro_feature_dim=2,
    )
    variables = init_encoder_variables(jax.random.PRNGKey(0), config)
    asset_window = jnp.ones((4, 3, 2), dtype=jnp.float32)
    macro_window = jnp.ones((4, 2), dtype=jnp.float32)
    spectral_row = jnp.ones((20,), dtype=jnp.float32)

    market_vectors = encode_market_state_flax(
        variables,
        asset_window,
        macro_window,
        spectral_row,
        config,
    )

    assert market_vectors.shape == (64,)
    assert jnp.isfinite(market_vectors).all()


def test_production_actor_and_critic_initialize_on_tiny_state() -> None:
    config = ProductionPPOConfig(n_assets=4, n_regimes=2)
    state = build_structured_ppo_state(
        asset_embeddings=jnp.ones((3, 64), dtype=jnp.float32),
        market_vector=jnp.ones((config.phi_dim,), dtype=jnp.float32),
        macro_state=jnp.ones((16,), dtype=jnp.float32),
        spectral_state=jnp.ones((20,), dtype=jnp.float32),
        regime_probs=jnp.ones((2,), dtype=jnp.float32) / 2.0,
        prev_weights=jnp.ones((4,), dtype=jnp.float32) / 4.0,
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
    )
    train_state = initialize_ppo_train_state(jax.random.PRNGKey(1), config)

    logits = train_state.actor.apply_fn({"params": train_state.actor.params}, state)
    weights = actor_mean_weights(logits, config)
    value = train_state.critic.apply_fn({"params": train_state.critic.params}, state)

    assert config.state_dim == 108
    assert logits.shape == (4,)
    assert weights.shape == (4,)
    assert jnp.all(weights >= 0.0)
    assert jnp.allclose(jnp.sum(weights), 1.0)
    assert value.shape == ()
    assert jnp.isfinite(value)
