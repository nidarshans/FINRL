"""Tests for the production Dirichlet simplex policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from scipy.stats import dirichlet

from finrl.ppo import (
    DirichletPortfolioDistribution,
    ProductionPPOConfig,
    action_log_prob,
    policy_entropy,
    validate_simplex_action,
)


def test_dirichlet_distribution_mean_is_on_simplex() -> None:
    config = ProductionPPOConfig(n_assets=3, temperature=0.8)
    distribution = DirichletPortfolioDistribution(
        logits=jnp.array([0.2, -0.1, 0.4], dtype=jnp.float32),
        config=config,
    )

    mean = distribution.mean()
    concentration = distribution.concentration()

    assert mean.shape == (3,)
    assert concentration.shape == (3,)
    assert jnp.all(mean >= 0.0)
    assert jnp.all(concentration > 0.0)
    assert_allclose(jnp.sum(mean), 1.0, rtol=1e-6, atol=1e-8)


def test_action_log_prob_matches_scipy_reference() -> None:
    config = ProductionPPOConfig(
        n_assets=3,
        temperature=0.7,
        dirichlet_concentration=12.0,
        min_concentration=0.1,
    )
    logits = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32)
    action = jnp.array([0.25, 0.35, 0.40], dtype=jnp.float32)
    distribution = DirichletPortfolioDistribution(logits=logits, config=config)

    actual = action_log_prob(logits, action, config)
    expected = dirichlet.logpdf(action, distribution.concentration())

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_policy_entropy_matches_scipy_reference() -> None:
    config = ProductionPPOConfig(
        n_assets=3,
        temperature=1.0,
        dirichlet_concentration=8.0,
        min_concentration=0.2,
    )
    logits = jnp.array([0.1, 0.5, -0.4], dtype=jnp.float32)
    distribution = DirichletPortfolioDistribution(logits=logits, config=config)

    actual = policy_entropy(logits, config)
    expected = dirichlet.entropy(distribution.concentration())

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_sampling_is_reproducible_for_fixed_key() -> None:
    config = ProductionPPOConfig(n_assets=4, dirichlet_concentration=20.0)
    distribution = DirichletPortfolioDistribution(
        logits=jnp.array([0.4, 0.1, -0.2, 0.0], dtype=jnp.float32),
        config=config,
    )
    key = jax.random.PRNGKey(0)

    sample_a = distribution.sample(key)
    sample_b = distribution.sample(key)

    assert_allclose(sample_a, sample_b, rtol=1e-6, atol=1e-8)
    assert jnp.all(sample_a >= 0.0)
    assert_allclose(jnp.sum(sample_a), 1.0, rtol=1e-6, atol=1e-8)


def test_validate_simplex_action_rejects_invalid_weights() -> None:
    validate_simplex_action(jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32))

    for action in (
        jnp.array([0.2, -0.1, 0.9], dtype=jnp.float32),
        jnp.array([0.2, 0.3, 0.4], dtype=jnp.float32),
        jnp.array([0.2, jnp.nan, 0.8], dtype=jnp.float32),
    ):
        try:
            validate_simplex_action(action)
        except ValueError:
            continue
        raise AssertionError("invalid simplex action was accepted")
