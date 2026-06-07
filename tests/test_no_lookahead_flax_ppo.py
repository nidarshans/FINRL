"""No-look-ahead tests for production Flax PPO training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.ppo import ProductionPPOConfig, train_flax_ppo_on_split


def _initial_state(n_assets: int) -> EnvState:
    return EnvState(
        weights=jnp.ones((n_assets,), dtype=jnp.float32) / n_assets,
        portfolio_value=jnp.array(1.0, dtype=jnp.float32),
        peak_value=jnp.array(1.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _assert_tree_allclose(left: object, right: object) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        assert_allclose(left_leaf, right_leaf, rtol=1e-6, atol=1e-8)


def test_training_rollout_length_excludes_test_observations() -> None:
    config = ProductionPPOConfig(
        n_assets=3,
        n_regimes=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        update_epochs=1,
        minibatch_size=3,
        learning_rate=1e-3,
    )
    n_steps = 5
    train_steps = 3
    phi = jnp.arange(n_steps * 32, dtype=jnp.float32).reshape(n_steps, 32) / 100.0
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

    baseline = train_flax_ppo_on_split(
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(0),
        rollout_length=train_steps,
    )
    poisoned = train_flax_ppo_on_split(
        phi.at[train_steps:].set(999.0),
        regimes.at[train_steps:].set(jnp.array([1.0, 0.0], dtype=jnp.float32)),
        returns.at[train_steps:].set(-999.0),
        spy.at[train_steps:].set(999.0),
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(0),
        rollout_length=train_steps,
    )

    assert baseline.rollout.batch.rewards.shape == (train_steps,)
    assert_allclose(baseline.rollout.batch.rewards, poisoned.rollout.batch.rewards)
    _assert_tree_allclose(baseline.train_state.actor.params, poisoned.train_state.actor.params)
    _assert_tree_allclose(baseline.train_state.critic.params, poisoned.train_state.critic.params)
