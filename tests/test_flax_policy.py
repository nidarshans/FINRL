"""Tests for production Flax PPO actor-critic modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.ppo import (
    PortfolioActorFlax,
    PortfolioCriticFlax,
    ProductionPPOConfig,
    actor_mean_weights,
    build_allocation_context,
    build_structured_ppo_state,
    sample_flax_action,
)


def _state(config: ProductionPPOConfig):
    return build_structured_ppo_state(
        asset_embeddings=jnp.ones(
            (config.n_assets - 1, config.asset_latent_dim),
            dtype=jnp.float32,
        ),
        market_vector=jnp.ones((config.phi_dim,), dtype=jnp.float32),
        macro_state=jnp.ones((config.macro_dim,), dtype=jnp.float32),
        spectral_state=jnp.ones((config.spectral_dim,), dtype=jnp.float32),
        regime_probs=jnp.ones((config.n_regimes,), dtype=jnp.float32) / config.n_regimes,
        prev_weights=jnp.ones((config.n_assets,), dtype=jnp.float32) / config.n_assets,
        drawdown=jnp.array(0.05, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )


def test_flax_actor_and_critic_shapes_under_jit() -> None:
    config = ProductionPPOConfig(n_assets=5, n_regimes=3)
    state = _state(config)
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    actor_variables = actor.init(jax.random.PRNGKey(0), state)
    critic_variables = critic.init(jax.random.PRNGKey(1), state)

    logits = jax.jit(lambda x: actor.apply(actor_variables, x))(state)
    value = jax.jit(lambda x: critic.apply(critic_variables, x))(state)
    weights = actor_mean_weights(logits, config)

    assert logits.shape == (5,)
    assert value.shape == ()
    assert weights.shape == (5,)
    assert jnp.all(weights >= 0.0)
    assert_allclose(jnp.sum(weights), 1.0, rtol=1e-6, atol=1e-8)


def test_sampled_flax_action_is_valid_and_reproducible() -> None:
    config = ProductionPPOConfig(n_assets=4, n_regimes=2, dirichlet_concentration=15.0)
    state = _state(config)
    actor = PortfolioActorFlax(config)
    variables = actor.init(jax.random.PRNGKey(2), state)
    key = jax.random.PRNGKey(3)

    action_a = sample_flax_action(variables, state, key, config)
    action_b = sample_flax_action(variables, state, key, config)

    assert action_a.weights.shape == (4,)
    assert jnp.all(action_a.weights >= 0.0)
    assert_allclose(jnp.sum(action_a.weights), 1.0, rtol=1e-6, atol=1e-8)
    assert_allclose(action_a.weights, action_b.weights, rtol=1e-6, atol=1e-8)
    assert jnp.isfinite(action_a.log_prob)
    assert jnp.isfinite(action_a.entropy)


def test_deterministic_flax_action_uses_mean_allocation() -> None:
    config = ProductionPPOConfig(n_assets=4, n_regimes=2)
    state = _state(config)
    actor = PortfolioActorFlax(config)
    variables = actor.init(jax.random.PRNGKey(4), state)
    logits = actor.apply(variables, state)
    expected_weights = actor_mean_weights(logits, config)

    action = sample_flax_action(
        variables,
        state,
        jax.random.PRNGKey(5),
        config,
        deterministic=True,
    )

    assert_allclose(action.weights, expected_weights, rtol=1e-6, atol=1e-8)
    assert jnp.isfinite(action.log_prob)


def test_structured_actor_logits_and_critic_shapes_under_jit() -> None:
    config = ProductionPPOConfig(
        n_assets=4,
        n_regimes=2,
        phi_dim=5,
        asset_latent_dim=3,
        actor_hidden_dims=(6,),
        critic_hidden_dims=(7,),
    )
    state = build_structured_ppo_state(
        asset_embeddings=jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 10.0,
        market_vector=jnp.ones((5,), dtype=jnp.float32),
        macro_state=jnp.ones((16,), dtype=jnp.float32),
        spectral_state=jnp.ones((20,), dtype=jnp.float32),
        regime_probs=jnp.array([0.25, 0.75], dtype=jnp.float32),
        prev_weights=jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        drawdown=jnp.array(0.05, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    actor_variables = actor.init(jax.random.PRNGKey(10), state)
    critic_variables = critic.init(jax.random.PRNGKey(11), state)

    logits = jax.jit(lambda s: actor.apply(actor_variables, s))(state)
    value = jax.jit(lambda s: critic.apply(critic_variables, s))(state)
    weights = actor_mean_weights(logits, config)

    assert logits.shape == (4,)
    assert value.shape == ()
    assert weights.shape == (4,)
    assert jnp.all(weights >= 0.0)
    assert_allclose(jnp.sum(weights), 1.0, rtol=1e-6, atol=1e-8)
    assert "shared_asset_score" in actor_variables["params"]
    assert "cash_score" in actor_variables["params"]


def test_structured_state_preserves_weight_alignment() -> None:
    state = build_structured_ppo_state(
        asset_embeddings=jnp.zeros((2, 3), dtype=jnp.float32),
        market_vector=jnp.zeros((5,), dtype=jnp.float32),
        macro_state=jnp.zeros((16,), dtype=jnp.float32),
        spectral_state=jnp.zeros((20,), dtype=jnp.float32),
        regime_probs=jnp.ones((2,), dtype=jnp.float32) / 2.0,
        prev_weights=jnp.array([0.7, 0.2, 0.1], dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
    )

    assert_allclose(state.prev_weights[:2], jnp.array([0.7, 0.2], dtype=jnp.float32))
    assert_allclose(state.prev_weights[-1], 0.1, rtol=1e-6, atol=1e-8)


def test_allocation_context_matches_diagram_components() -> None:
    context = build_allocation_context(
        market_vector=jnp.ones((5,), dtype=jnp.float32),
        macro_state=2.0 * jnp.ones((3,), dtype=jnp.float32),
        spectral_state=3.0 * jnp.ones((4,), dtype=jnp.float32),
        regime_probs=jnp.array([0.25, 0.75], dtype=jnp.float32),
        drawdown=jnp.array(0.1, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )

    assert context.shape == (16,)
