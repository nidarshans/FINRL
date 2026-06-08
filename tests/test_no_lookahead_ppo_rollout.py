"""No-look-ahead tests for production PPO rollout collection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.ppo import (
    ProductionPPOConfig,
    collect_rollout,
    initialize_ppo_train_state,
)


def _initial_state(n_assets: int) -> EnvState:
    return EnvState(
        weights=jnp.ones((n_assets,), dtype=jnp.float32) / n_assets,
        portfolio_value=jnp.array(1.0, dtype=jnp.float32),
        peak_value=jnp.array(1.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def test_rollout_length_excludes_test_observations() -> None:
    config = ProductionPPOConfig(
        n_assets=3,
        n_regimes=2,
        asset_latent_dim=4,
        macro_dim=3,
        spectral_dim=5,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
    )
    train_state = initialize_ppo_train_state(jax.random.PRNGKey(0), config)
    actor_variables = {"params": train_state.actor.params}
    critic_variables = {"params": train_state.critic.params}
    n_steps = 5
    train_steps = 3
    market_vectors = jnp.arange(n_steps * 64, dtype=jnp.float32).reshape(n_steps, 64) / 100.0
    embeddings = (
        jnp.arange(n_steps * 2 * 4, dtype=jnp.float32).reshape(n_steps, 2, 4) / 100.0
    )
    macro = jnp.ones((n_steps, 3), dtype=jnp.float32)
    spectral = jnp.ones((n_steps, 5), dtype=jnp.float32)
    regimes = jnp.ones((n_steps, 2), dtype=jnp.float32) / 2.0
    returns = jnp.array(
        [
            [0.01, 0.0, 0.0001],
            [0.0, 0.02, 0.0001],
            [-0.01, 0.01, 0.0001],
            [0.50, -0.50, 0.0001],
            [-0.75, 0.25, 0.0001],
        ],
        dtype=jnp.float32,
    )
    spy = jnp.array([0.005, 0.004, -0.002, 0.9, -0.9], dtype=jnp.float32)

    baseline = collect_rollout(
        actor_variables,
        critic_variables,
        market_vectors,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(2),
        asset_embeddings=embeddings,
        macro_states=macro,
        spectral_states=spectral,
        rollout_length=train_steps,
    )
    poisoned = collect_rollout(
        actor_variables,
        critic_variables,
        market_vectors.at[train_steps:].set(999.0),
        regimes.at[train_steps:].set(jnp.array([1.0, 0.0], dtype=jnp.float32)),
        returns.at[train_steps:].set(-999.0),
        spy.at[train_steps:].set(999.0),
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(2),
        asset_embeddings=embeddings.at[train_steps:].set(999.0),
        macro_states=macro.at[train_steps:].set(999.0),
        spectral_states=spectral.at[train_steps:].set(999.0),
        rollout_length=train_steps,
    )

    assert baseline.batch.rewards.shape == (train_steps,)
    for left, right in zip(
        jax.tree.leaves(baseline.batch.states),
        jax.tree.leaves(poisoned.batch.states),
        strict=True,
    ):
        assert_allclose(left, right)
    assert_allclose(baseline.batch.actions, poisoned.batch.actions)
    assert_allclose(baseline.batch.rewards, poisoned.batch.rewards)
    assert_allclose(baseline.batch.old_log_probs, poisoned.batch.old_log_probs)
    assert baseline.final_env_state.step == train_steps
