"""Tests for production PPO rollout buffers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState, environment_step
from finrl.ppo import (
    PortfolioActorFlax,
    PortfolioCriticFlax,
    ProductionPPOConfig,
    RolloutBatch,
    action_log_prob,
    collect_rollout,
    make_minibatches,
    rollout_length,
    shuffle_rollout_indices,
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


def _config() -> ProductionPPOConfig:
    return ProductionPPOConfig(
        n_assets=3,
        n_regimes=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        dirichlet_concentration=12.0,
    )


def _arrays(n_steps: int = 4) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    phi = jnp.arange(n_steps * 32, dtype=jnp.float32).reshape(n_steps, 32) / 100.0
    regimes = jnp.ones((n_steps, 2), dtype=jnp.float32) / 2.0
    returns = jnp.array(
        [
            [0.01, 0.0, 0.0001],
            [0.0, 0.02, 0.0001],
            [-0.01, 0.01, 0.0001],
            [0.03, -0.02, 0.0001],
        ],
        dtype=jnp.float32,
    )[:n_steps]
    spy = jnp.array([0.005, 0.004, -0.002, 0.01], dtype=jnp.float32)[:n_steps]
    return phi, regimes, returns, spy


def _variables(config: ProductionPPOConfig) -> tuple[dict[str, object], dict[str, object]]:
    state = jnp.ones((config.state_dim,), dtype=jnp.float32)
    actor_variables = PortfolioActorFlax(config).init(jax.random.PRNGKey(0), state)
    critic_variables = PortfolioCriticFlax(config).init(jax.random.PRNGKey(1), state)
    return actor_variables, critic_variables


def _collect(length: int = 4):
    config = _config()
    actor_variables, critic_variables = _variables(config)
    phi, regimes, returns, spy = _arrays(length)
    buffer = collect_rollout(
        actor_variables,
        critic_variables,
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(2),
    )
    return buffer, actor_variables, config, returns, spy


def test_collect_rollout_uses_environment_step_accounting() -> None:
    buffer, _, config, returns, spy = _collect()

    manual_first = environment_step(
        _initial_state(config.action_dim),
        buffer.batch.actions[0],
        returns[0],
        spy[0],
        EnvConfig(transaction_cost_rate=0.0),
    )

    assert rollout_length(buffer.batch) == 4
    assert buffer.batch.states.shape == (4, config.state_dim)
    assert buffer.batch.actions.shape == (4, config.action_dim)
    assert_allclose(buffer.batch.rewards[0], manual_first.reward, rtol=1e-6, atol=1e-8)
    assert_allclose(
        buffer.step_results.state.weights[0],
        buffer.batch.actions[0],
        rtol=1e-6,
        atol=1e-8,
    )


def test_rollout_stores_log_probs_from_collection_policy() -> None:
    buffer, actor_variables, config, _, _ = _collect()
    actor = PortfolioActorFlax(config)
    logits = jax.vmap(lambda state: actor.apply(actor_variables, state))(
        buffer.batch.states
    )
    recomputed = jax.vmap(lambda logit, action: action_log_prob(logit, action, config))(
        logits,
        buffer.batch.actions,
    )

    assert_allclose(buffer.batch.old_log_probs, recomputed, rtol=1e-6, atol=1e-6)


def test_collect_rollout_is_jittable() -> None:
    config = _config()
    actor_variables, critic_variables = _variables(config)
    phi, regimes, returns, spy = _arrays()

    rewards = jax.jit(
        lambda phi_arg, regime_arg, return_arg, spy_arg: collect_rollout(
            actor_variables,
            critic_variables,
            phi_arg,
            regime_arg,
            return_arg,
            spy_arg,
            _initial_state(config.action_dim),
            EnvConfig(transaction_cost_rate=0.0),
            config,
            jax.random.PRNGKey(3),
        ).batch.rewards
    )(phi, regimes, returns, spy)

    assert rewards.shape == (4,)
    assert jnp.isfinite(rewards).all()


def _fake_batch(n_steps: int = 5) -> RolloutBatch:
    ids = jnp.arange(n_steps, dtype=jnp.float32)
    return RolloutBatch(
        states=ids[:, None],
        actions=(ids + 10.0)[:, None],
        old_log_probs=ids + 20.0,
        rewards=ids + 30.0,
        values=ids + 40.0,
        dones=jnp.zeros((n_steps,), dtype=jnp.float32),
        entropies=ids + 50.0,
        turnovers=ids + 60.0,
        transaction_costs=ids + 70.0,
        drawdowns=ids + 80.0,
        net_returns=ids + 90.0,
    )


def test_make_minibatches_preserves_row_alignment() -> None:
    minibatches = make_minibatches(_fake_batch(), minibatch_size=2, shuffle=False)

    assert len(minibatches) == 3
    for minibatch in minibatches:
        ids = minibatch.states[:, 0]
        assert_allclose(minibatch.actions[:, 0], ids + 10.0)
        assert_allclose(minibatch.old_log_probs, ids + 20.0)
        assert_allclose(minibatch.rewards, ids + 30.0)


def test_shuffle_rollout_indices_is_deterministic() -> None:
    key = jax.random.PRNGKey(4)

    first = shuffle_rollout_indices(key, 6)
    second = shuffle_rollout_indices(key, 6)

    assert_allclose(first, second)
    assert sorted(first.tolist()) == list(range(6))
