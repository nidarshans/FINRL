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
    PortfolioActor,
    collect_train_trajectory,
    evaluate_action_logprob,
    freeze_ppo_batch,
    initialize_actor_critic,
    load_policy_checkpoint,
    normalize_advantages,
    save_policy_checkpoint,
    sample_action,
    temperature_softmax,
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
    config = PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4, minibatch_size=3)
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
    assert result.checkpoint.state.optimizer_state is not None
    assert result.diagnostics["updates_completed"] == 1.0
    for value in result.diagnostics.values():
        assert jnp.isfinite(value)


def test_advantage_normalization_is_stable_for_constant_values() -> None:
    advantages = jnp.ones((4,), dtype=jnp.float32)

    normalized = normalize_advantages(advantages)

    assert normalized.shape == advantages.shape
    assert_allclose(normalized, jnp.zeros_like(advantages), rtol=1e-6, atol=1e-8)


def test_freeze_ppo_batch_keeps_old_logprobs_fixed_across_updates() -> None:
    config = PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-3, minibatch_size=3)
    artifacts = _artifacts()
    initial = initialize_actor_critic(config, jax.random.PRNGKey(8))
    trajectory = collect_train_trajectory(
        initial,
        artifacts.env_config,
        artifacts,
        config,
        jax.random.PRNGKey(9),
    )
    batch = freeze_ppo_batch(trajectory, config)

    trained = train_ppo_on_split(artifacts, config, jax.random.PRNGKey(8))
    new_logprobs = jax.vmap(
        lambda state, action: evaluate_action_logprob(
            trained.checkpoint.state.actor_params,
            state,
            action,
            config.temperature,
            config.dirichlet_concentration,
            config.min_concentration,
        )
    )(batch.states, batch.actions)

    assert_allclose(batch.old_logprobs, trajectory.old_logprobs, rtol=1e-6, atol=1e-8)
    assert not bool(jnp.allclose(batch.old_logprobs, new_logprobs))


def test_kl_early_stopping_triggers() -> None:
    config = PPOConfig(
        n_assets=3,
        ppo_epochs=5,
        learning_rate=5e-3,
        minibatch_size=3,
        target_kl=1e-12,
    )

    result = train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(10))

    assert result.diagnostics["epochs_completed"] < config.update_epochs


def test_optimizer_state_persists_across_updates() -> None:
    config = PPOConfig(n_assets=3, ppo_epochs=2, learning_rate=1e-4, minibatch_size=1)

    result = train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(11))

    optimizer_state = result.checkpoint.state.optimizer_state
    assert optimizer_state is not None
    adam_count = optimizer_state[1][0].count
    assert adam_count == result.checkpoint.state.step


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


def test_toy_ppo_training_improves_superior_asset_allocation() -> None:
    n_steps = 12
    config = PPOConfig(
        n_assets=2,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        ppo_epochs=20,
        minibatch_size=12,
        learning_rate=1e-2,
        target_kl=10.0,
        dirichlet_concentration=5.0,
        entropy_coef=0.0,
    )
    artifacts = PPOArtifacts(
        phi=jnp.zeros((n_steps, 32), dtype=jnp.float32),
        regime_probs=jnp.ones((n_steps, 4), dtype=jnp.float32) / 4.0,
        asset_returns=jnp.tile(
            jnp.array([[0.03, 0.0]], dtype=jnp.float32),
            (n_steps, 1),
        ),
        spy_returns=jnp.zeros((n_steps,), dtype=jnp.float32),
        initial_env_state=_initial_env_state(2),
        env_config=EnvConfig(transaction_cost_rate=0.0),
    )
    initial = initialize_actor_critic(config, jax.random.PRNGKey(2))
    initial_trajectory = collect_train_trajectory(
        initial,
        artifacts.env_config,
        artifacts,
        config,
        jax.random.PRNGKey(102),
    )

    result = train_ppo_on_split(artifacts, config, jax.random.PRNGKey(2))
    trained_trajectory = collect_train_trajectory(
        result.checkpoint.state,
        artifacts.env_config,
        artifacts,
        config,
        jax.random.PRNGKey(102),
    )
    final_action = sample_action(
        result.checkpoint.state.actor_params,
        trained_trajectory.states[-1],
        jax.random.PRNGKey(103),
        config.temperature,
        config.dirichlet_concentration,
        config.min_concentration,
    )
    actor = PortfolioActor(config)
    initial_mean_weights = jax.vmap(
        lambda state: temperature_softmax(
            actor.apply(initial.actor_params, state),
            config.temperature,
        )
    )(initial_trajectory.states)
    trained_mean_weights = jax.vmap(
        lambda state: temperature_softmax(
            actor.apply(result.checkpoint.state.actor_params, state),
            config.temperature,
        )
    )(initial_trajectory.states)
    initial_expected_reward = jnp.mean(initial_mean_weights[:, 0] * 0.03)
    trained_expected_reward = jnp.mean(trained_mean_weights[:, 0] * 0.03)

    assert trained_expected_reward > initial_expected_reward
    assert jnp.mean(trained_mean_weights[:, 0]) > 0.60
    assert jnp.all(jnp.isfinite(trained_trajectory.actions))
    assert jnp.all(jnp.isfinite(trained_trajectory.old_logprobs))
    assert jnp.all(jnp.isfinite(trained_trajectory.values))
    assert jnp.all(jnp.isfinite(final_action.weights))


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
