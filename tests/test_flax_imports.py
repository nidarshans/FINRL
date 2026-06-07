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
    create_train_state,
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

    phi = encode_market_state_flax(
        variables,
        asset_window,
        macro_window,
        spectral_row,
        config,
    )

    assert phi.shape == (32,)
    assert jnp.isfinite(phi).all()


def test_production_actor_and_critic_initialize_on_tiny_state() -> None:
    config = ProductionPPOConfig(n_assets=4, n_regimes=2)
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    actor_variables = actor.init(jax.random.PRNGKey(1), state)
    critic_variables = critic.init(jax.random.PRNGKey(2), state)

    logits = actor.apply(actor_variables, state)
    weights = actor_mean_weights(logits, config)
    value = critic.apply(critic_variables, state)

    assert config.state_dim == 40
    assert logits.shape == (4,)
    assert weights.shape == (4,)
    assert jnp.all(weights >= 0.0)
    assert jnp.allclose(jnp.sum(weights), 1.0)
    assert value.shape == ()
    assert jnp.isfinite(value)


def test_create_train_state_wraps_flax_params_with_optax() -> None:
    config = ProductionPPOConfig(n_assets=4, n_regimes=2)
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)
    actor = PortfolioActorFlax(config)
    variables = actor.init(jax.random.PRNGKey(3), state)

    train_state = create_train_state(
        apply_fn=actor.apply,
        params=variables["params"],
        learning_rate=1e-3,
    )

    assert train_state.step == 0
    assert "hidden_0" in train_state.params


def test_existing_smoke_test_encoder_and_ppo_imports_still_work() -> None:
    from finrl.models import EncoderConfig, MarketEncoder
    from finrl.ppo import PPOConfig, PortfolioActor

    smoke_encoder = MarketEncoder(EncoderConfig(lookback=2, n_assets=2))
    smoke_actor = PortfolioActor(PPOConfig(n_assets=3))

    assert smoke_encoder.config.lookback == 2
    assert smoke_actor.config.n_assets == 3

