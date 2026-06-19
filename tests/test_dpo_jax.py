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
    sparsemax,
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


def test_sparsemax_shape() -> None:
    logits = jnp.ones((4, 11), dtype=jnp.float32)
    weights = sparsemax(logits)

    assert weights.shape == (4, 11)


def test_sparsemax_sums_to_one() -> None:
    logits = jnp.array([[3.0, 1.0, -2.0]], dtype=jnp.float32)
    weights = sparsemax(logits)

    assert_allclose(jnp.sum(weights, axis=-1), jnp.array([1.0], dtype=jnp.float32))


def test_sparsemax_nonnegative() -> None:
    logits = jnp.array([[3.0, 1.0, -2.0]], dtype=jnp.float32)
    weights = sparsemax(logits)

    assert bool(jnp.all(weights >= 0.0))


def test_sparsemax_can_output_exact_zero() -> None:
    logits = jnp.array([[5.0, 1.0, -10.0]], dtype=jnp.float32)
    weights = sparsemax(logits)

    assert weights[0, -1] == 0.0


def test_direct_allocation_head_outputs_simplex_weights() -> None:
    head = DirectAllocationHead(hidden_dim=8, allocation_activation="sparsemax")
    embeddings = jnp.ones((3, 2, 4), dtype=jnp.float32)
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    weights = head.apply(variables, embeddings)

    assert weights.shape == (3, 3)
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((3,)), atol=1e-5)
    assert bool(jnp.all(weights >= 0.0))


def test_direct_allocation_head_shape() -> None:
    batch_size, n_assets, embedding_dim = 4, 10, 16
    embeddings = jnp.ones((batch_size, n_assets, embedding_dim), dtype=jnp.float32)
    head = DirectAllocationHead(hidden_dim=32, allocation_activation="sparsemax")
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    weights = head.apply(variables, embeddings)

    assert weights.shape == (batch_size, n_assets + 1)


def test_sparsemax_allocation_head_has_gradients() -> None:
    batch_size, n_assets, embedding_dim = 4, 10, 16
    embeddings = jnp.arange(
        batch_size * n_assets * embedding_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, n_assets, embedding_dim)
    embeddings = embeddings / jnp.max(embeddings)
    head = DirectAllocationHead(hidden_dim=32, allocation_activation="sparsemax")
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    def loss_fn(params: dict[str, object]) -> jax.Array:
        weights = head.apply({"params": params}, embeddings)
        return jnp.sum(weights**2)

    grads = jax.grad(loss_fn)(variables["params"])

    assert grads is not None
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))


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


def test_dpo_loss_scalar() -> None:
    n_steps, n_assets = 20, 10
    weights = jnp.ones((n_steps, n_assets + 1), dtype=jnp.float32) / (n_assets + 1)
    returns = jnp.zeros((n_steps, n_assets), dtype=jnp.float32)
    initial_weights = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[-1].set(1.0)

    loss, _metrics = dpo_loss(weights, returns, initial_weights, DPOConfig())

    assert loss.shape == ()


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
