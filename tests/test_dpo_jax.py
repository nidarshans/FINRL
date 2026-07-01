"""Tests for JAX direct portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.dpo_jax import (
    DPOConfig,
    DirectAllocationHead,
    ScoreAllocationPolicy,
    build_dpo_batch,
    dpo_loss,
    evaluate_dpo,
    initialize_dpo_train_state,
    predict_weights,
    sparsemax,
    train_step,
)
from finrl.models.score_heads import AssetScoreHeads


def _tree_delta(left: object, right: object) -> float:
    return sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _features_returns() -> tuple[jax.Array, jax.Array]:
    features = jnp.arange(4 * 2 * 5, dtype=jnp.float32).reshape(4, 2, 5)
    returns = jnp.array(
        [
            [0.01, 0.00],
            [0.00, 0.02],
            [-0.01, 0.01],
            [0.03, -0.02],
        ],
        dtype=jnp.float32,
    )
    return features / 100.0, returns


def _score_policy(allocation_hidden_dims: tuple[int, ...] = ()) -> ScoreAllocationPolicy:
    return ScoreAllocationPolicy(
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
        accumulation_hidden_dims=(6,),
        accumulation_hidden_activation="tanh",
        accumulation_output_activation="sigmoid",
        accumulation_use_layer_norm=True,
        liquidity_exit_hidden_dims=(4,),
        liquidity_exit_hidden_activation="relu",
        liquidity_exit_output_activation="sigmoid",
        liquidity_exit_use_layer_norm=False,
        allocation_hidden_dims=allocation_hidden_dims,
        simplex_activation="softmax",
    )


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
    head = DirectAllocationHead(hidden_dims=(8,), simplex_activation="softmax")
    embeddings = jnp.ones((3, 2, 4), dtype=jnp.float32)
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    weights = head.apply(variables, embeddings)

    assert weights.shape == (3, 3)
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((3,)), atol=1e-5)
    assert bool(jnp.all(weights >= 0.0))


def test_direct_allocation_head_shape() -> None:
    batch_size, n_assets, embedding_dim = 4, 10, 16
    embeddings = jnp.ones((batch_size, n_assets, embedding_dim), dtype=jnp.float32)
    head = DirectAllocationHead(hidden_dims=(32,), simplex_activation="softmax")
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    weights = head.apply(variables, embeddings)

    assert weights.shape == (batch_size, n_assets + 1)


def test_direct_allocation_head_supports_no_hidden_layers_with_softmax() -> None:
    scores = jnp.ones((4, 10, 2), dtype=jnp.float32)
    head = DirectAllocationHead(hidden_dims=(), simplex_activation="softmax")
    variables = head.init(jax.random.PRNGKey(0), scores)

    weights = head.apply(variables, scores)

    assert weights.shape == (4, 11)
    assert "hidden_0" not in variables["params"]
    assert "cash_logit" not in variables["params"]
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((4,)), atol=1e-6)
    assert bool(jnp.all(weights[:, :-1] > 0.0))
    assert_allclose(weights[:, -1], jnp.zeros((4,)), atol=0.0)
    assert_allclose(jnp.sum(weights[:, :-1], axis=-1), jnp.ones((4,)), atol=1e-6)


def test_direct_allocation_head_supports_configurable_hidden_dims() -> None:
    batch_size, n_assets, embedding_dim = 4, 10, 16
    embeddings = jnp.ones((batch_size, n_assets, embedding_dim), dtype=jnp.float32)
    head = DirectAllocationHead(
        hidden_dims=(256, 128, 64),
        simplex_activation="softmax",
    )
    variables = head.init(jax.random.PRNGKey(0), embeddings)

    weights = head.apply(variables, embeddings)

    assert weights.shape == (batch_size, n_assets + 1)
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((batch_size,)), atol=1e-5)
    assert bool(jnp.all(weights >= 0.0))


def test_dpo_config_validates_hidden_dims() -> None:
    with pytest.raises(ValueError, match="accumulation_hidden_dims"):
        DPOConfig(accumulation_hidden_dims=())
    with pytest.raises(ValueError, match="liquidity_exit_hidden_dims"):
        DPOConfig(liquidity_exit_hidden_dims=(0,))
    with pytest.raises(ValueError, match="allocation_hidden_dims"):
        DPOConfig(allocation_hidden_dims=(64, 0))
    with pytest.raises(ValueError, match="accumulation_hidden_activation"):
        DPOConfig(accumulation_hidden_activation="swish")


def test_dpo_config_allows_no_allocation_hidden_layers_and_defaults_to_softmax() -> None:
    config = DPOConfig(allocation_hidden_dims=())

    assert config.allocation_hidden_dims == ()
    assert config.simplex_activation == "softmax"


def test_softmax_allocation_head_has_gradients() -> None:
    batch_size, n_assets, embedding_dim = 4, 10, 16
    embeddings = jnp.arange(
        batch_size * n_assets * embedding_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, n_assets, embedding_dim)
    embeddings = embeddings / jnp.max(embeddings)
    head = DirectAllocationHead(hidden_dims=(32,), simplex_activation="softmax")
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


def test_dpo_loss_optimizes_log_return_relative_to_spy() -> None:
    config = DPOConfig(
        transaction_cost_bps=0.0,
        lambda_turnover=0.0,
        lambda_drawdown=0.0,
        lambda_concentration=0.0,
    )
    weights = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    returns = jnp.array([[0.03], [-0.01]], dtype=jnp.float32)
    spy_returns = jnp.array([0.01, 0.02], dtype=jnp.float32)
    initial = jnp.array([0.0, 1.0], dtype=jnp.float32)

    loss, metrics = dpo_loss(
        weights,
        returns,
        initial,
        config,
        spy_returns,
    )

    portfolio_log_returns = jnp.log1p(returns[:, 0] + config.eps)
    spy_log_returns = jnp.log1p(spy_returns + config.eps)
    expected_active = jnp.mean(portfolio_log_returns - spy_log_returns)
    assert_allclose(loss, -expected_active, rtol=1e-6)
    assert_allclose(metrics.mean_active_log_return, expected_active, rtol=1e-6)


def test_dpo_batch_validates_spy_return_shape() -> None:
    features, returns = _features_returns()

    with pytest.raises(ValueError, match="spy_returns must have shape"):
        build_dpo_batch(features, returns, spy_returns=jnp.zeros((4, 1)))


def test_dpo_loss_scalar() -> None:
    n_steps, n_assets = 20, 10
    weights = jnp.ones((n_steps, n_assets + 1), dtype=jnp.float32) / (n_assets + 1)
    returns = jnp.zeros((n_steps, n_assets), dtype=jnp.float32)
    initial_weights = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[-1].set(1.0)

    loss, _metrics = dpo_loss(weights, returns, initial_weights, DPOConfig())

    assert loss.shape == ()


def test_score_allocation_policy_only_passes_two_scores_to_allocation_head() -> None:
    policy = _score_policy(allocation_hidden_dims=(8,))
    features, _ = _features_returns()
    params = policy.init(jax.random.PRNGKey(0), features)["params"]

    assert params["allocation_head"]["hidden_0"]["kernel"].shape[0] == 2
    assert params["score_heads"]["accumulation"]["hidden_0"]["kernel"].shape[-1] == 6
    assert params["score_heads"]["liquidity"]["hidden_0"]["kernel"].shape[-1] == 4


def test_score_head_sigmoid_outputs_are_bounded() -> None:
    heads = AssetScoreHeads(
        accumulation_hidden_dims=(6,),
        accumulation_output_activation="sigmoid",
        liquidity_exit_hidden_dims=(4,),
        liquidity_exit_output_activation="sigmoid",
    )
    acc_inputs = jnp.arange(22, dtype=jnp.float32).reshape(2, 11)
    exit_inputs = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    variables = heads.init(jax.random.PRNGKey(0), acc_inputs, exit_inputs)

    accumulation, liquidity_exit = heads.apply(variables, acc_inputs, exit_inputs)

    assert bool(jnp.all((accumulation >= 0.0) & (accumulation <= 1.0)))
    assert bool(jnp.all((liquidity_exit >= 0.0) & (liquidity_exit <= 1.0)))


def test_unrouted_raw_feature_cannot_change_allocations() -> None:
    policy = _score_policy(allocation_hidden_dims=(8,))
    features, _ = _features_returns()
    variables = policy.init(jax.random.PRNGKey(0), features)
    changed = features.at[..., 4].set(9_999.0)

    assert_allclose(policy.apply(variables, features), policy.apply(variables, changed))


def test_future_feature_change_does_not_change_prior_allocations() -> None:
    policy = _score_policy(allocation_hidden_dims=(8,))
    features, _ = _features_returns()
    variables = policy.init(jax.random.PRNGKey(0), features)
    changed = features.at[-1].set(9_999.0)

    assert_allclose(
        policy.apply(variables, features)[:-1],
        policy.apply(variables, changed)[:-1],
    )


def test_dpo_train_step_updates_score_heads_and_allocation_head() -> None:
    features, returns = _features_returns()
    config = DPOConfig(
        learning_rate=1e-2,
        transaction_cost_bps=0.0,
        accumulation_hidden_dims=(6,),
        liquidity_exit_hidden_dims=(4,),
        allocation_hidden_dims=(8,),
    )
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(0),
        config,
        2,
        5,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    batch = build_dpo_batch(features, returns)

    updated, metrics = train_step(state, batch)

    assert (
        _tree_delta(
            state.policy.params["score_heads"],
            updated.policy.params["score_heads"],
        )
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
    features, returns = _features_returns()
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(1),
        DPOConfig(
            transaction_cost_bps=0.0,
            accumulation_hidden_dims=(6,),
            liquidity_exit_hidden_dims=(4,),
            allocation_hidden_dims=(8,),
        ),
        2,
        5,
        accumulation_indices=(0, 1),
        liquidity_indices=(2, 3),
    )
    batch = build_dpo_batch(features, returns)

    weights = predict_weights(state, batch)
    loss, metrics = evaluate_dpo(state, batch)

    assert weights.shape == (4, 3)
    assert jnp.isfinite(loss)
    assert metrics.final_equity > 0.0
