"""Tests for JAX direct portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.dpo_jax import (
    DPOConfig,
    DirectAllocationHead,
    build_dpo_batch,
    dpo_loss,
    evaluate_dpo,
    initialize_dpo_train_state,
    predict_weights,
    train_step,
)
from finrl.models.asset_encoder import AssetOnlyEncoderConfig


def _tree_delta(left: object, right: object) -> float:
    return sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _encoder_config() -> AssetOnlyEncoderConfig:
    return AssetOnlyEncoderConfig(
        lookback=3,
        n_assets=2,
        asset_feature_dim=4,
        asset_hidden_dim=4,
        score_hidden_dims=(6,),
    )


def _windows_returns() -> tuple[jax.Array, jax.Array]:
    windows = jnp.arange(4 * 3 * 2 * 4, dtype=jnp.float32).reshape(4, 3, 2, 4)
    returns = jnp.array(
        [
            [0.01, 0.00],
            [0.00, 0.02],
            [-0.01, 0.01],
            [0.03, -0.02],
        ],
        dtype=jnp.float32,
    )
    return windows / 100.0, returns


def test_direct_allocation_head_outputs_simplex_weights() -> None:
    head = DirectAllocationHead(hidden_dim=8)
    embeddings = jnp.ones((3, 2, 4), dtype=jnp.float32)
    previous_weights = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    variables = head.init(
        jax.random.PRNGKey(0),
        embeddings,
        previous_weights,
        jnp.zeros((3, 1), dtype=jnp.float32),
        jnp.zeros((3, 1), dtype=jnp.float32),
    )

    weights = head.apply(
        variables,
        embeddings,
        previous_weights,
        jnp.zeros((3, 1), dtype=jnp.float32),
        jnp.zeros((3, 1), dtype=jnp.float32),
    )

    assert weights.shape == (3, 3)
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((3,)), rtol=1e-6)
    assert bool(jnp.all(weights >= 0.0))


def test_dpo_loss_matches_simple_manual_accounting() -> None:
    config = DPOConfig(
        transaction_cost_bps=10.0,
        lambda_turnover=0.0,
        lambda_drawdown=0.0,
        lambda_concentration=0.0,
    )
    weights = jnp.array([[0.5, 0.5], [1.0, 0.0]], dtype=jnp.float32)
    returns = jnp.array([[0.02], [-0.01]], dtype=jnp.float32)
    initial = jnp.array([0.0, 1.0], dtype=jnp.float32)

    loss, metrics = dpo_loss(weights, returns, initial, config)

    expected_net_returns = jnp.array([0.01 - 0.001, -0.01 - 0.001], dtype=jnp.float32)
    expected_equity = jnp.prod(1.0 + expected_net_returns)
    assert_allclose(
        loss,
        -jnp.mean(jnp.log(1.0 + expected_net_returns + config.eps)),
        rtol=1e-6,
    )
    assert_allclose(metrics.mean_turnover, jnp.array(1.0, dtype=jnp.float32), rtol=1e-6)
    assert_allclose(metrics.final_equity, expected_equity, rtol=1e-6)


def test_dpo_train_step_updates_encoder_and_allocation_head() -> None:
    windows, returns = _windows_returns()
    config = DPOConfig(learning_rate=1e-2, transaction_cost_bps=0.0)
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(0),
        config,
        _encoder_config(),
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
        head_hidden_dim=8,
    )
    batch = build_dpo_batch(windows, returns)

    updated, metrics = train_step(state, batch)

    assert (
        _tree_delta(state.policy.params["encoder"], updated.policy.params["encoder"])
        > 0.0
    )
    assert (
        _tree_delta(
            state.policy.params["allocation_head"],
            updated.policy.params["allocation_head"],
        )
        > 0.0
    )
    assert jnp.isfinite(metrics.mean_log_return)


def test_dpo_evaluation_predicts_weights_and_loss() -> None:
    windows, returns = _windows_returns()
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(1),
        DPOConfig(transaction_cost_bps=0.0),
        _encoder_config(),
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
        head_hidden_dim=8,
    )
    batch = build_dpo_batch(windows, returns)

    weights = predict_weights(state, batch)
    loss, metrics = evaluate_dpo(state, batch)

    assert weights.shape == (4, 3)
    assert jnp.isfinite(loss)
    assert metrics.final_equity > 0.0
