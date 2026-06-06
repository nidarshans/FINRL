"""Tests for PPO rollout over the trading environment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState, environment_step
from finrl.ppo import (
    PPOArtifacts,
    PPOConfig,
    collect_train_trajectory,
    initialize_actor_critic,
    load_policy_checkpoint,
    save_policy_checkpoint,
    train_ppo_on_split,
)


def _initial_env_state(n_assets: int) -> EnvState:
    return EnvState(
        weights=jnp.ones((n_assets,), dtype=jnp.float32) / n_assets,
        portfolio_value=jnp.array(1.0, dtype=jnp.float32),
        peak_value=jnp.array(1.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _artifacts(n_steps: int = 3, n_assets: int = 3) -> PPOArtifacts:
    phi = jnp.arange(n_steps * 32, dtype=jnp.float32).reshape(n_steps, 32) / 100.0
    regimes = jnp.ones((n_steps, 4), dtype=jnp.float32) / 4.0
    asset_returns = jnp.array(
        [[0.01, 0.0, 0.0001], [0.0, 0.02, 0.0001], [-0.01, 0.01, 0.0001]],
        dtype=jnp.float32,
    )[:n_steps, :n_assets]
    spy_returns = jnp.array([0.005, 0.004, -0.002], dtype=jnp.float32)[:n_steps]
    return PPOArtifacts(
        phi=phi,
        regime_probs=regimes,
        asset_returns=asset_returns,
        spy_returns=spy_returns,
        initial_env_state=_initial_env_state(n_assets),
        env_config=EnvConfig(transaction_cost_rate=0.0),
    )


def test_collect_train_trajectory_uses_environment_accounting_path() -> None:
    config = PPOConfig(n_assets=3, train_epochs=1)
    artifacts = _artifacts()
    policy = initialize_actor_critic(config, jax.random.PRNGKey(0))

    trajectory = collect_train_trajectory(
        policy,
        artifacts.env_config,
        artifacts,
        config,
        jax.random.PRNGKey(1),
    )
    manual_first = environment_step(
        artifacts.initial_env_state,
        trajectory.actions[0],
        artifacts.asset_returns[0],
        artifacts.spy_returns[0],
        artifacts.env_config,
    )

    assert trajectory.states.shape == (3, config.state_dim)
    assert trajectory.actions.shape == (3, 3)
    assert_allclose(trajectory.rewards[0], manual_first.reward, rtol=1e-6, atol=1e-8)
    assert_allclose(
        trajectory.step_results.state.weights[0],
        trajectory.actions[0],
        rtol=1e-6,
        atol=1e-8,
    )


def test_train_ppo_on_split_returns_finite_losses_and_checkpoint() -> None:
    config = PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4)
    initial = initialize_actor_critic(config, jax.random.PRNGKey(2))

    result = train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(2))

    assert result.checkpoint.state.step == 1
    assert jnp.isfinite(result.actor_loss)
    assert jnp.isfinite(result.critic_loss)
    assert jnp.isfinite(result.total_loss)
    actor_delta = sum(
        float(jnp.sum(jnp.abs(before - after)))
        for before, after in zip(
            jax.tree_util.tree_leaves(initial.actor_params),
            jax.tree_util.tree_leaves(result.checkpoint.state.actor_params),
            strict=True,
        )
    )
    critic_delta = sum(
        float(jnp.sum(jnp.abs(before - after)))
        for before, after in zip(
            jax.tree_util.tree_leaves(initial.critic_params),
            jax.tree_util.tree_leaves(result.checkpoint.state.critic_params),
            strict=True,
        )
    )
    assert actor_delta > 0.0
    assert critic_delta > 0.0


def test_policy_checkpoint_can_be_saved_and_loaded(tmp_path) -> None:
    config = PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4)
    result = train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(6))
    path = tmp_path / "ppo.pkl"

    save_policy_checkpoint(result.checkpoint, path)
    loaded = load_policy_checkpoint(path)

    assert loaded.config == result.checkpoint.config
    assert loaded.train_window == result.checkpoint.train_window
    assert_allclose(
        loaded.state.actor_params["w0"],
        result.checkpoint.state.actor_params["w0"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_train_ppo_on_split_rejects_mismatched_initial_weights() -> None:
    config = PPOConfig(n_assets=3)
    artifacts = _artifacts()
    invalid_state = EnvState(
        weights=jnp.ones((2,), dtype=jnp.float32) / 2.0,
        portfolio_value=artifacts.initial_env_state.portfolio_value,
        peak_value=artifacts.initial_env_state.peak_value,
        drawdown=artifacts.initial_env_state.drawdown,
        previous_turnover=artifacts.initial_env_state.previous_turnover,
        step=artifacts.initial_env_state.step,
    )
    invalid = PPOArtifacts(
        phi=artifacts.phi,
        regime_probs=artifacts.regime_probs,
        asset_returns=artifacts.asset_returns,
        spy_returns=artifacts.spy_returns,
        initial_env_state=invalid_state,
        env_config=artifacts.env_config,
    )

    with pytest.raises(ValueError, match="initial environment weights"):
        train_ppo_on_split(invalid, config, jax.random.PRNGKey(7))
