"""Tests for PPO actor policy and state construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from scipy.stats import dirichlet

from finrl.ppo import (
    PPOConfig,
    PortfolioActor,
    PortfolioContext,
    build_ppo_state,
    dirichlet_concentration,
    portfolio_logprob,
    sample_action,
    temperature_softmax,
)


def test_temperature_softmax_returns_long_only_weights() -> None:
    logits = jnp.array([2.0, 1.0, -1.0], dtype=jnp.float32)

    weights = temperature_softmax(logits, temperature=0.7)

    assert weights.shape == (3,)
    assert jnp.all(weights >= 0.0)
    assert_allclose(jnp.sum(weights), 1.0, rtol=1e-6, atol=1e-8)


def test_actor_outputs_n_plus_one_logits_and_valid_weights() -> None:
    config = PPOConfig(n_assets=4)
    actor = PortfolioActor(config)
    params = actor.init(jax.random.PRNGKey(0))
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)

    logits = actor.apply(params, state)
    action = sample_action(params, state, jax.random.PRNGKey(1), config.temperature)

    assert logits.shape == (4,)
    assert action.weights.shape == (4,)
    assert jnp.all(action.weights >= 0.0)
    assert_allclose(jnp.sum(action.weights), 1.0, rtol=1e-6, atol=1e-8)
    assert jnp.isfinite(action.logprob)
    assert jnp.isfinite(action.entropy)


def test_portfolio_logprob_matches_scipy_dirichlet_reference() -> None:
    logits = jnp.array([0.2, -0.1, 0.4], dtype=jnp.float32)
    action = jnp.array([0.25, 0.35, 0.40], dtype=jnp.float32)
    concentration = dirichlet_concentration(
        logits,
        temperature=0.8,
        concentration_scale=12.0,
        min_concentration=0.1,
    )

    actual = portfolio_logprob(
        logits,
        action,
        temperature=0.8,
        concentration_scale=12.0,
        min_concentration=0.1,
    )
    expected = dirichlet.logpdf(action, concentration)

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_build_ppo_state_dimension_matches_architecture() -> None:
    config = PPOConfig(n_assets=101)
    phi = jnp.ones((32,), dtype=jnp.float32)
    regimes = jnp.ones((4,), dtype=jnp.float32) / 4.0
    context = PortfolioContext(
        weights=jnp.ones((101,), dtype=jnp.float32) / 101.0,
        drawdown=jnp.array(0.05, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )

    state = build_ppo_state(phi, regimes, context)

    assert config.state_dim == 139
    assert state.shape == (139,)


def test_sample_action_is_jittable() -> None:
    config = PPOConfig(n_assets=3)
    params = PortfolioActor(config).init(jax.random.PRNGKey(2))
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)

    weights = jax.jit(
        lambda state_arg: sample_action(
            params,
            state_arg,
            jax.random.PRNGKey(3),
            config.temperature,
        ).weights
    )(state)

    assert weights.shape == (3,)
    assert_allclose(jnp.sum(weights), 1.0, rtol=1e-6, atol=1e-8)
