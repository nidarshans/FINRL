"""Tests for generalized advantage estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.ppo import compute_gae


def test_compute_gae_matches_hand_calculated_values() -> None:
    rewards = jnp.array([1.0, 1.0], dtype=jnp.float32)
    values = jnp.array([0.5, 0.25, 0.0], dtype=jnp.float32)
    dones = jnp.array([0.0, 1.0], dtype=jnp.float32)

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.9,
        lambda_=0.8,
    )

    assert_allclose(advantages, jnp.array([1.265, 0.75]), rtol=1e-6, atol=1e-8)
    assert_allclose(returns, jnp.array([1.765, 1.0]), rtol=1e-6, atol=1e-8)


def test_compute_gae_is_jittable() -> None:
    rewards = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    values = jnp.array([0.0, 0.1, 0.2, 0.0], dtype=jnp.float32)
    dones = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    advantages, returns = jax.jit(
        lambda r, v, d: compute_gae(r, v, d, gamma=0.99, lambda_=0.95)
    )(rewards, values, dones)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert jnp.isfinite(advantages).all()
    assert jnp.isfinite(returns).all()


def test_compute_gae_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="values length must be T or T \\+ 1"):
        compute_gae(
            rewards=jnp.ones((3,), dtype=jnp.float32),
            values=jnp.ones((5,), dtype=jnp.float32),
            dones=jnp.zeros((3,), dtype=jnp.float32),
            gamma=0.99,
            lambda_=0.95,
        )
