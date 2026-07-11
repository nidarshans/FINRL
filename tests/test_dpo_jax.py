"""Tests for JAX direct portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from finrl.dpo_jax import (
    DPOConfig,
    DirectAllocationHead,
    DirectFeatureAllocationPolicy,
    build_dpo_batch,
    dpo_loss,
    evaluate_dpo,
    initialize_dpo_train_state,
    predict_weights,
    sparsemax,
    train_dpo,
    train_step,
)
from finrl.env.trading_env import EnvConfig, EnvState, scan_environment


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
    features = jnp.ones((4, 10, 2), dtype=jnp.float32)
    head = DirectAllocationHead(hidden_dims=(), simplex_activation="softmax")
    variables = head.init(jax.random.PRNGKey(0), features)

    weights = head.apply(variables, features)

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


def test_direct_allocation_head_rejects_restricted_output_logits() -> None:
    head = DirectAllocationHead(output_activation="sigmoid")

    with pytest.raises(ValueError, match="must be 'identity'"):
        head.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 2, 3), dtype=jnp.float32),
        )


def test_dpo_config_validates_hidden_dims() -> None:
    with pytest.raises(ValueError, match="allocation_hidden_dims"):
        DPOConfig(allocation_hidden_dims=(64, 0))
    with pytest.raises(ValueError, match="allocation_hidden_activation"):
        DPOConfig(allocation_hidden_activation="swish")
    with pytest.raises(ValueError, match="must be 'identity'"):
        DPOConfig(allocation_output_activation="sigmoid")


def test_dpo_config_allows_no_allocation_hidden_layers_and_defaults_to_softmax() -> None:
    config = DPOConfig(allocation_hidden_dims=())

    assert config.allocation_hidden_dims == ()
    assert config.simplex_activation == "softmax"
    assert config.transaction_cost_bps == 10.0


def test_direct_feature_policy_returns_normalized_weights() -> None:
    policy = DirectFeatureAllocationPolicy(
        direct_feature_indices=(0, 2, 4),
        allocation_hidden_dims=(8,),
    )
    features = jnp.ones((2, 4, 5), dtype=jnp.float32)
    variables = policy.init(jax.random.PRNGKey(0), features)
    weights = policy.apply(variables, features)

    assert weights.shape == (2, 5)
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert_allclose(jnp.sum(weights, axis=-1), jnp.ones((2,)), atol=1e-6)


def test_direct_feature_policy_initializes_and_predicts() -> None:
    features, returns = _features_returns()
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(0),
        DPOConfig(allocation_hidden_dims=(8,)),
        2,
        5,
        direct_feature_indices=(0, 2, 4),
    )
    weights = predict_weights(state, build_dpo_batch(features, returns))

    assert weights.shape == (4, 3)


def test_direct_feature_mode_requires_routed_indices() -> None:
    with pytest.raises(ValueError, match="requires routed feature indices"):
        initialize_dpo_train_state(
            jax.random.PRNGKey(0),
            DPOConfig(),
            2,
            5,
            direct_feature_indices=(),
        )


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

    drifted_first_weights = jnp.array([0.51 / 1.01, 0.50 / 1.01], dtype=jnp.float32)
    second_turnover = jnp.sum(
        jnp.abs(weights[1] - drifted_first_weights)
    )
    expected_net_returns = jnp.array(
        [0.01 - 0.001, -0.01 - 0.001 * second_turnover],
        dtype=jnp.float32,
    )
    expected_equity = jnp.prod(1.0 + expected_net_returns)
    assert_allclose(
        loss,
        -jnp.mean(jnp.log(1.0 + expected_net_returns + config.eps)),
        rtol=1e-6,
    )
    assert_allclose(
        metrics.mean_turnover,
        (1.0 + second_turnover) / 2.0,
        rtol=1e-6,
    )
    assert_allclose(metrics.final_equity, expected_equity, rtol=1e-6)


def test_dpo_loss_reports_turnover_but_does_not_penalize_it() -> None:
    weights = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    returns = jnp.array([[0.0]], dtype=jnp.float32)
    initial = jnp.array([0.0, 1.0], dtype=jnp.float32)
    base = DPOConfig(
        transaction_cost_bps=0.0,
        lambda_turnover=0.0,
        lambda_drawdown=0.0,
    )
    penalized = DPOConfig(
        transaction_cost_bps=0.0,
        lambda_turnover=100.0,
        lambda_drawdown=0.0,
    )

    base_loss, base_metrics = dpo_loss(weights, returns, initial, base)
    penalized_loss, penalized_metrics = dpo_loss(weights, returns, initial, penalized)

    assert_allclose(penalized_loss, base_loss, rtol=1e-6, atol=1e-8)
    assert_allclose(base_metrics.mean_turnover, 2.0, rtol=1e-6)
    assert_allclose(penalized_metrics.mean_turnover, 2.0, rtol=1e-6)


def test_dpo_loss_uses_drifted_weights_for_next_turnover() -> None:
    weights = jnp.array(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
        dtype=jnp.float32,
    )
    returns = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    initial = weights[0]

    _, metrics = dpo_loss(
        weights,
        returns,
        initial,
        DPOConfig(transaction_cost_bps=0.0, lambda_drawdown=0.0),
    )

    expected_second_turnover = 1.0 / 3.0
    assert_allclose(
        metrics.mean_turnover,
        expected_second_turnover / 2.0,
        rtol=1e-6,
    )


def test_dpo_loss_matches_environment_accounting_path() -> None:
    weights = jnp.array(
        [[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]],
        dtype=jnp.float32,
    )
    risky_returns = jnp.array(
        [[0.10, -0.05], [-0.02, 0.04]],
        dtype=jnp.float32,
    )
    full_returns = jnp.concatenate(
        [risky_returns, jnp.zeros((2, 1), dtype=jnp.float32)],
        axis=1,
    )
    initial = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    config = DPOConfig(
        transaction_cost_bps=10.0,
        lambda_turnover=0.0,
        lambda_drawdown=0.0,
    )

    _, metrics = dpo_loss(weights, risky_returns, initial, config)
    final_state, step_results = scan_environment(
        EnvState(
            weights=initial,
            portfolio_value=jnp.array(1.0, dtype=jnp.float32),
            peak_value=jnp.array(1.0, dtype=jnp.float32),
            drawdown=jnp.array(0.0, dtype=jnp.float32),
            previous_turnover=jnp.array(0.0, dtype=jnp.float32),
            step=jnp.array(0, dtype=jnp.int32),
        ),
        weights,
        full_returns,
        jnp.zeros((2,), dtype=jnp.float32),
        EnvConfig(transaction_cost_rate=0.001),
    )

    assert_allclose(metrics.mean_turnover, jnp.mean(step_results.turnover), rtol=1e-6)
    assert_allclose(metrics.final_equity, final_state.portfolio_value, rtol=1e-6)


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


def test_unrouted_raw_feature_cannot_change_allocations() -> None:
    policy = DirectFeatureAllocationPolicy(
        direct_feature_indices=(0, 2, 3),
        allocation_hidden_dims=(8,),
    )
    features, _ = _features_returns()
    variables = policy.init(jax.random.PRNGKey(0), features)
    changed = features.at[..., 4].set(9_999.0)

    assert_allclose(policy.apply(variables, features), policy.apply(variables, changed))


def test_future_feature_change_does_not_change_prior_allocations() -> None:
    policy = DirectFeatureAllocationPolicy(
        direct_feature_indices=(0, 2, 4),
        allocation_hidden_dims=(8,),
    )
    features, _ = _features_returns()
    variables = policy.init(jax.random.PRNGKey(0), features)
    changed = features.at[-1].set(9_999.0)

    assert_allclose(
        policy.apply(variables, features)[:-1],
        policy.apply(variables, changed)[:-1],
    )


def test_dpo_train_step_updates_allocation_head() -> None:
    features, returns = _features_returns()
    config = DPOConfig(
        learning_rate=1e-2,
        transaction_cost_bps=0.0,
        allocation_hidden_dims=(8,),
    )
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(0),
        config,
        2,
        5,
        direct_feature_indices=(0, 2, 4),
    )
    batch = build_dpo_batch(features, returns)

    updated, metrics = train_step(state, batch)

    assert (
        _tree_delta(
            state.policy.params["allocation_head"],
            updated.policy.params["allocation_head"],
        )
        > 0.0
    )
    assert jnp.isfinite(metrics.mean_log_return)


def test_train_dpo_uses_complete_chronological_path() -> None:
    features, returns = _features_returns()
    config = DPOConfig(
        learning_rate=1e-2,
        num_epochs=2,
        transaction_cost_bps=10.0,
        allocation_hidden_dims=(8,),
    )
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(4),
        config,
        2,
        5,
        direct_feature_indices=(0, 2, 4),
    )

    updated, history = train_dpo(state, build_dpo_batch(features, returns))

    assert len(history) == config.num_epochs
    assert all(jnp.isfinite(metrics.mean_log_return) for metrics in history)
    assert (
        _tree_delta(
            state.policy.params["allocation_head"],
            updated.policy.params["allocation_head"],
        )
        > 0.0
    )


def test_train_dpo_full_batch_matches_one_train_step() -> None:
    features, returns = _features_returns()
    config = DPOConfig(
        learning_rate=1e-2,
        num_epochs=1,
        transaction_cost_bps=0.0,
        allocation_hidden_dims=(8,),
    )
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(5),
        config,
        2,
        5,
        direct_feature_indices=(0, 2, 4),
    )
    batch = build_dpo_batch(features, returns)

    expected, _ = train_step(state, batch)
    actual, history = train_dpo(state, batch)

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected.policy.params),
        jax.tree.leaves(actual.policy.params),
        strict=True,
    ):
        assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-7)
    assert len(history) == 1


def test_dpo_evaluation_predicts_weights_and_loss() -> None:
    features, returns = _features_returns()
    state = initialize_dpo_train_state(
        jax.random.PRNGKey(1),
        DPOConfig(
            transaction_cost_bps=0.0,
            allocation_hidden_dims=(8,),
        ),
        2,
        5,
        direct_feature_indices=(0, 2, 4),
    )
    batch = build_dpo_batch(features, returns)

    weights = predict_weights(state, batch)
    loss, metrics = evaluate_dpo(state, batch)

    assert weights.shape == (4, 3)
    assert jnp.isfinite(loss)
    assert metrics.final_equity > 0.0
