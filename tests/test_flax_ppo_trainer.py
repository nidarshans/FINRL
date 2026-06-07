"""Tests for the production Flax PPO optimization loop."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.ppo import (
    PPOUpdateBatchFlax,
    ProductionPPOTrainState,
    ProductionPPOConfig,
    RolloutBatch,
    action_log_prob,
    compute_gae,
    critic_loss,
    evaluate_frozen_flax_policy,
    freeze_rollout_batch,
    initialize_ppo_train_state,
    portfolio_allocation_entropy,
    save_policy_checkpoint,
    load_policy_checkpoint,
    train_flax_ppo_on_split,
    update_minibatch,
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


def _config(**overrides: object) -> ProductionPPOConfig:
    values = {
        "n_assets": 3,
        "n_regimes": 2,
        "actor_hidden_dims": (8,),
        "critic_hidden_dims": (8,),
        "update_epochs": 1,
        "minibatch_size": 3,
        "learning_rate": 1e-3,
        "dirichlet_concentration": 12.0,
    }
    values.update(overrides)
    return ProductionPPOConfig(**values)


def _arrays(n_steps: int = 3) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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


def _tree_delta(left: object, right: object) -> float:
    return sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    )


def test_compute_gae_matches_hand_computed_fixture() -> None:
    rewards = jnp.array([1.0, 2.0], dtype=jnp.float32)
    values = jnp.array([0.5, 0.25, 0.0], dtype=jnp.float32)
    dones = jnp.array([0.0, 1.0], dtype=jnp.float32)

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.9, lambda_=0.8)

    expected_advantages = jnp.array([1.985, 1.75], dtype=jnp.float32)
    expected_returns = jnp.array([2.485, 2.0], dtype=jnp.float32)
    assert_allclose(advantages, expected_advantages, rtol=1e-6, atol=1e-6)
    assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)


def test_freeze_rollout_batch_bootstraps_truncated_final_step() -> None:
    config = _config(
        gamma=0.9,
        gae_lambda=0.8,
        normalize_advantages=False,
        phi_dim=1,
    )
    rollout = RolloutBatch(
        states=jnp.zeros((1, config.state_dim), dtype=jnp.float32),
        actions=jnp.ones((1, config.action_dim), dtype=jnp.float32) / config.action_dim,
        old_log_probs=jnp.zeros((1,), dtype=jnp.float32),
        rewards=jnp.array([1.0], dtype=jnp.float32),
        values=jnp.array([0.5], dtype=jnp.float32),
        dones=jnp.array([0.0], dtype=jnp.float32),
        truncations=jnp.array([1.0], dtype=jnp.float32),
        entropies=jnp.zeros((1,), dtype=jnp.float32),
        turnovers=jnp.zeros((1,), dtype=jnp.float32),
        transaction_costs=jnp.zeros((1,), dtype=jnp.float32),
        drawdowns=jnp.zeros((1,), dtype=jnp.float32),
        net_returns=jnp.zeros((1,), dtype=jnp.float32),
    )

    batch = freeze_rollout_batch(
        rollout,
        config,
        bootstrap_value=jnp.array(0.25, dtype=jnp.float32),
    )

    assert_allclose(batch.advantages, jnp.array([0.725], dtype=jnp.float32))
    assert_allclose(batch.returns, jnp.array([1.225], dtype=jnp.float32))


def test_one_minibatch_update_changes_actor_and_critic_params() -> None:
    config = _config()
    phi, regimes, returns, spy = _arrays()
    rng = jax.random.PRNGKey(0)
    init_key, _, _ = jax.random.split(rng, 3)
    result = train_flax_ppo_on_split(
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        rng,
    )
    initial = initialize_ppo_train_state(init_key, config)

    assert _tree_delta(initial.actor.params, result.train_state.actor.params) > 0.0
    assert _tree_delta(initial.critic.params, result.train_state.critic.params) > 0.0
    assert result.metrics.updates_completed == 1.0
    assert jnp.isfinite(result.metrics.total_loss)
    assert jnp.isfinite(result.metrics.policy_loss)
    assert jnp.isfinite(result.metrics.post_update_approx_kl)
    assert jnp.isfinite(result.metrics.mean_episode_return)
    assert jnp.isfinite(result.metrics.advantage_mean)
    assert jnp.isfinite(result.metrics.advantage_std)
    assert result.metrics.ratio_min > 0.0
    assert result.metrics.ratio_max >= result.metrics.ratio_min


def test_portfolio_allocation_entropy_matches_hand_computed_fixture() -> None:
    weights = jnp.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    entropy = portfolio_allocation_entropy(weights)

    assert_allclose(entropy[0], jnp.log(3.0), rtol=1e-6, atol=1e-6)
    assert_allclose(entropy[1], 0.0, rtol=1e-6, atol=1e-6)


def test_portfolio_entropy_penalty_increases_total_loss() -> None:
    base_config = _config(use_value_clipping=False)
    penalized_config = _config(
        use_value_clipping=False,
        portfolio_entropy_coef=0.5,
    )
    state = initialize_ppo_train_state(jax.random.PRNGKey(11), base_config)
    states = jnp.ones((3, base_config.state_dim), dtype=jnp.float32)
    actor_logits = jax.vmap(
        lambda row: state.actor.apply_fn({"params": state.actor.params}, row)
    )(states)
    actions = jnp.ones((3, base_config.action_dim), dtype=jnp.float32)
    actions = actions / base_config.action_dim
    old_log_probs = jax.vmap(
        lambda logit, action: action_log_prob(logit, action, base_config)
    )(actor_logits, actions)
    batch = PPOUpdateBatchFlax(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=jnp.zeros((3,), dtype=jnp.float32),
        returns=jnp.zeros((3,), dtype=jnp.float32),
        old_values=jnp.zeros((3,), dtype=jnp.float32),
        rewards=jnp.zeros((3,), dtype=jnp.float32),
        turnovers=jnp.zeros((3,), dtype=jnp.float32),
        transaction_costs=jnp.zeros((3,), dtype=jnp.float32),
        drawdowns=jnp.zeros((3,), dtype=jnp.float32),
    )

    _, base_metrics = update_minibatch(
        ProductionPPOTrainState(
            actor=state.actor,
            critic=state.critic,
            config=base_config,
        ),
        batch,
    )
    _, penalized_metrics = update_minibatch(
        ProductionPPOTrainState(
            actor=state.actor,
            critic=state.critic,
            config=penalized_config,
        ),
        batch,
    )

    assert_allclose(base_metrics.portfolio_entropy, jnp.log(3.0), rtol=1e-5)
    assert penalized_metrics.total_loss > base_metrics.total_loss


def test_value_loss_decreases_on_tiny_supervised_fixture() -> None:
    config = _config(use_value_clipping=False, learning_rate=5e-3)
    state = initialize_ppo_train_state(jax.random.PRNGKey(1), config)
    states = jnp.ones((3, config.state_dim), dtype=jnp.float32)
    actor_logits = jax.vmap(
        lambda row: state.actor.apply_fn({"params": state.actor.params}, row)
    )(states)
    actions = jnp.ones((3, config.action_dim), dtype=jnp.float32) / config.action_dim
    old_log_probs = jax.vmap(lambda logit, action: action_log_prob(logit, action, config))(
        actor_logits,
        actions,
    )
    batch = PPOUpdateBatchFlax(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=jnp.zeros((3,), dtype=jnp.float32),
        returns=jnp.ones((3,), dtype=jnp.float32),
        old_values=jnp.zeros((3,), dtype=jnp.float32),
        rewards=jnp.zeros((3,), dtype=jnp.float32),
        turnovers=jnp.zeros((3,), dtype=jnp.float32),
        transaction_costs=jnp.zeros((3,), dtype=jnp.float32),
        drawdowns=jnp.zeros((3,), dtype=jnp.float32),
    )

    before_values = jax.vmap(
        lambda row: state.critic.apply_fn({"params": state.critic.params}, row)
    )(states)
    before_loss = critic_loss(before_values, batch.returns, use_clipping=False)
    for _ in range(10):
        state, _ = update_minibatch(state, batch)
    after_values = jax.vmap(
        lambda row: state.critic.apply_fn({"params": state.critic.params}, row)
    )(states)
    after_loss = critic_loss(after_values, batch.returns, use_clipping=False)

    assert after_loss < before_loss


def test_huber_critic_loss_matches_hand_computed_fixture() -> None:
    values = jnp.array([0.0, 3.0], dtype=jnp.float32)
    returns = jnp.array([2.0, 1.0], dtype=jnp.float32)

    actual = critic_loss(
        values,
        returns,
        use_clipping=False,
        loss_type="huber",
        huber_delta=1.0,
    )

    assert_allclose(actual, 1.5, rtol=1e-6, atol=1e-6)


def test_frozen_evaluation_does_not_update_train_state() -> None:
    config = _config()
    phi, regimes, returns, spy = _arrays()
    trained = train_flax_ppo_on_split(
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(2),
    )

    evaluated = evaluate_frozen_flax_policy(
        trained.train_state,
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        jax.random.PRNGKey(3),
    )

    assert _tree_delta(trained.train_state.actor.params, evaluated.train_state.actor.params) == 0.0
    assert (
        _tree_delta(trained.train_state.critic.params, evaluated.train_state.critic.params)
        == 0.0
    )
    assert trained.train_state.actor.step == evaluated.train_state.actor.step
    assert jnp.isfinite(evaluated.rollout.batch.rewards).all()


def test_checkpoint_round_trip_preserves_train_state(tmp_path: Path) -> None:
    config = _config()
    phi, regimes, returns, spy = _arrays()
    result = train_flax_ppo_on_split(
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(4),
    )
    path = tmp_path / "production_ppo.pkl"

    save_policy_checkpoint(result.train_state, path)
    loaded = load_policy_checkpoint(path)

    assert loaded.config == result.train_state.config
    assert loaded.actor.step == result.train_state.actor.step
    assert loaded.critic.step == result.train_state.critic.step
    assert _tree_delta(loaded.actor.params, result.train_state.actor.params) == 0.0
    assert _tree_delta(loaded.critic.params, result.train_state.critic.params) == 0.0
