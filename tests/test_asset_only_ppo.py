"""Tests for asset-only PPO training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from numpy.testing import assert_allclose

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.models.asset_encoder import AssetOnlyEncoder, AssetOnlyEncoderConfig
from finrl.ppo import (
    PortfolioActorFlax,
    PortfolioCriticFlax,
    ProductionPPOConfig,
    build_structured_ppo_state,
    build_update_batch,
    collect_rollout,
    initialize_ppo_train_state,
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


def _configs() -> tuple[ProductionPPOConfig, AssetOnlyEncoderConfig]:
    ppo = ProductionPPOConfig(
        n_assets=3,
        asset_latent_dim=4,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        update_epochs=1,
        minibatch_size=2,
        learning_rate=1e-3,
        dirichlet_concentration=12.0,
    )
    encoder = AssetOnlyEncoderConfig(
        lookback=3,
        n_assets=2,
        asset_feature_dim=4,
        asset_hidden_dim=4,
        score_hidden_dims=(6,),
    )
    return ppo, encoder


def _arrays(n_steps: int = 4) -> tuple[jax.Array, jax.Array, jax.Array]:
    windows = jnp.arange(n_steps * 3 * 2 * 4, dtype=jnp.float32).reshape(n_steps, 3, 2, 4)
    windows = windows / 100.0
    returns = jnp.array(
        [
            [0.01, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [-0.01, 0.01, 0.0],
            [0.03, -0.02, 0.0],
        ],
        dtype=jnp.float32,
    )[:n_steps]
    spy = jnp.array([0.005, 0.004, -0.002, 0.01], dtype=jnp.float32)[:n_steps]
    return windows, returns, spy


def _tree_delta(left: object, right: object) -> float:
    return sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_asset_only_encoder_shape_and_score_head_sensitivity() -> None:
    _, encoder_config = _configs()
    encoder = AssetOnlyEncoder(
        encoder_config,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    windows, _, _ = _arrays(n_steps=2)
    variables = encoder.init(jax.random.PRNGKey(0), windows)
    baseline = encoder.apply(variables, windows)
    changed_params = unfreeze(variables["params"])
    changed_params["score_heads"]["accumulation"]["hidden_0"]["kernel"] = (
        changed_params["score_heads"]["accumulation"]["hidden_0"]["kernel"] + 0.25
    )
    changed = encoder.apply({"params": freeze(changed_params)}, windows)

    assert baseline.shape == (2, 2, 4)
    assert _tree_delta(baseline, changed) > 0.0


def test_asset_only_rollout_stores_raw_windows_not_frozen_states() -> None:
    ppo_config, encoder_config = _configs()
    state = initialize_ppo_train_state(
        jax.random.PRNGKey(0),
        ppo_config,
        encoder_config,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    windows, returns, spy = _arrays()

    rollout = collect_rollout(
        {"params": state.policy.params},
        windows,
        returns,
        spy,
        _initial_state(ppo_config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        ppo_config,
        encoder_config,
        (0, 1),
        (2, 3),
        jax.random.PRNGKey(1),
    )

    assert rollout.batch.asset_windows.shape == windows.shape
    assert not hasattr(rollout.batch, "states")
    assert rollout.batch.actions.shape == (4, 3)


def test_ppo_gradients_reach_encoder_score_heads_actor_and_critic() -> None:
    ppo_config, encoder_config = _configs()
    state = initialize_ppo_train_state(
        jax.random.PRNGKey(0),
        ppo_config,
        encoder_config,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    windows, returns, spy = _arrays()
    rollout = collect_rollout(
        {"params": state.policy.params},
        windows,
        returns,
        spy,
        _initial_state(ppo_config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        ppo_config,
        encoder_config,
        (0, 1),
        (2, 3),
        jax.random.PRNGKey(2),
    )
    batch = build_update_batch(rollout.batch, ppo_config, rollout.bootstrap_value)
    updated, metrics = update_minibatch(state, batch)

    assert _tree_delta(state.policy.params["encoder"]["score_heads"], updated.policy.params["encoder"]["score_heads"]) > 0.0
    assert _tree_delta(state.policy.params["encoder"]["asset_lstm_encoder"], updated.policy.params["encoder"]["asset_lstm_encoder"]) > 0.0
    assert _tree_delta(state.policy.params["actor"], updated.policy.params["actor"]) > 0.0
    assert _tree_delta(state.policy.params["critic"], updated.policy.params["critic"]) > 0.0
    assert metrics.grad_norm > 0.0


def test_train_ppo_on_split_updates_single_policy_state() -> None:
    ppo_config, encoder_config = _configs()
    windows, returns, spy = _arrays(n_steps=3)
    initial = initialize_ppo_train_state(
        jax.random.PRNGKey(0),
        ppo_config,
        encoder_config,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    result = train_flax_ppo_on_split(
        windows,
        returns,
        spy,
        _initial_state(ppo_config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        ppo_config,
        encoder_config,
        (0, 1),
        (2, 3),
        jax.random.PRNGKey(0),
    )

    assert _tree_delta(initial.policy.params, result.train_state.policy.params) > 0.0
    assert result.metrics.updates_completed == 1.0
    assert_allclose(jnp.sum(result.rollout.batch.actions, axis=1), jnp.ones((3,)), rtol=1e-5)


def test_asset_only_policy_is_independent_of_portfolio_context() -> None:
    config, _ = _configs()
    embeddings = jnp.ones((2, 4), dtype=jnp.float32)
    state = build_structured_ppo_state(
        asset_embeddings=embeddings,
        prev_weights=jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32),
        drawdown=jnp.array(0.1, dtype=jnp.float32),
        previous_turnover=jnp.array(0.2, dtype=jnp.float32),
    )
    changed_context_state = build_structured_ppo_state(
        asset_embeddings=jnp.ones((2, 4), dtype=jnp.float32),
        prev_weights=jnp.array([0.8, 0.1, 0.1], dtype=jnp.float32),
        drawdown=jnp.array(0.8, dtype=jnp.float32),
        previous_turnover=jnp.array(1.5, dtype=jnp.float32),
    )
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    actor_variables = actor.init(jax.random.PRNGKey(3), state)
    critic_variables = critic.init(jax.random.PRNGKey(4), state)

    logits = actor.apply(actor_variables, state)
    changed_context_logits = actor.apply(actor_variables, changed_context_state)
    value = critic.apply(critic_variables, state)
    changed_context_value = critic.apply(critic_variables, changed_context_state)

    assert logits.shape == (3,)
    assert value.shape == ()
    assert_allclose(logits, changed_context_logits)
    assert_allclose(value, changed_context_value)
