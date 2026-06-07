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
    build_flax_ppo_state,
    sample_flax_action,
)


def test_production_ppo_state_dimension_matches_components() -> None:
    config = ProductionPPOConfig(n_assets=101, n_regimes=4)
    phi = jnp.ones((32,), dtype=jnp.float32)
    regime_probs = jnp.ones((4,), dtype=jnp.float32) / 4.0
    weights = jnp.ones((101,), dtype=jnp.float32) / 101.0

    state = build_flax_ppo_state(
        phi=phi,
        regime_probs=regime_probs,
        weights=weights,
        drawdown=jnp.array(0.05, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )

    assert config.state_dim == 139
    assert state.shape == (139,)


def test_flax_actor_and_critic_shapes_under_jit() -> None:
    config = ProductionPPOConfig(n_assets=5, n_regimes=3)
    state = jnp.linspace(-1.0, 1.0, config.state_dim, dtype=jnp.float32)
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
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)
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
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)
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
