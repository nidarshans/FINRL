"""Tests for PPO loss helpers."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.ppo import (
    clipped_value_loss,
    entropy_bonus,
    ppo_clip_loss,
    ppo_total_loss,
    value_loss,
)


def test_ppo_clip_loss_is_finite_for_synthetic_trajectory() -> None:
    old_logprobs = jnp.array([-1.0, -1.2, -0.9], dtype=jnp.float32)
    new_logprobs = jnp.array([-0.95, -1.3, -0.85], dtype=jnp.float32)
    advantages = jnp.array([0.5, -0.25, 0.1], dtype=jnp.float32)

    loss = ppo_clip_loss(new_logprobs, old_logprobs, advantages, clip_epsilon=0.2)

    assert jnp.isfinite(loss)


def test_value_loss_and_entropy_bonus_are_finite() -> None:
    values = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    returns = jnp.array([0.0, 0.25, 0.5], dtype=jnp.float32)
    entropies = jnp.array([1.0, 0.5, 0.25], dtype=jnp.float32)

    assert jnp.isfinite(value_loss(values, returns))
    assert jnp.isfinite(entropy_bonus(entropies))


def test_clipped_value_loss_uses_larger_ppo_error() -> None:
    values = jnp.array([2.0], dtype=jnp.float32)
    old_values = jnp.array([0.0], dtype=jnp.float32)
    returns = jnp.array([1.0], dtype=jnp.float32)

    loss = clipped_value_loss(values, old_values, returns, clip_epsilon=0.2)

    assert loss == jnp.array(1.0, dtype=jnp.float32)


def test_portfolio_entropy_coefficient_rewards_diversification() -> None:
    low_entropy_loss = ppo_total_loss(
        actor_loss=jnp.array(1.0, dtype=jnp.float32),
        critic_loss_value=jnp.array(0.5, dtype=jnp.float32),
        entropy=jnp.array(0.0, dtype=jnp.float32),
        value_coef=0.5,
        entropy_coef=0.0,
        portfolio_entropy=jnp.array(0.1, dtype=jnp.float32),
        portfolio_entropy_coef=0.2,
    )
    high_entropy_loss = ppo_total_loss(
        actor_loss=jnp.array(1.0, dtype=jnp.float32),
        critic_loss_value=jnp.array(0.5, dtype=jnp.float32),
        entropy=jnp.array(0.0, dtype=jnp.float32),
        value_coef=0.5,
        entropy_coef=0.0,
        portfolio_entropy=jnp.array(1.0, dtype=jnp.float32),
        portfolio_entropy_coef=0.2,
    )
    unchanged = ppo_total_loss(
        actor_loss=jnp.array(1.0, dtype=jnp.float32),
        critic_loss_value=jnp.array(0.5, dtype=jnp.float32),
        entropy=jnp.array(0.0, dtype=jnp.float32),
        value_coef=0.5,
        entropy_coef=0.0,
        portfolio_entropy=jnp.array(1.0, dtype=jnp.float32),
        portfolio_entropy_coef=0.0,
    )

    assert bool(high_entropy_loss < low_entropy_loss)
    assert unchanged == jnp.array(1.25, dtype=jnp.float32)
