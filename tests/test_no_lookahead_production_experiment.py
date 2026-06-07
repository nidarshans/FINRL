"""No-look-ahead tests for production walk-forward integration."""

from __future__ import annotations

import jax
from numpy.testing import assert_allclose

from finrl.backtest.walk_forward import generate_walk_forward_splits
from finrl.experiments.run_walk_forward import evaluate_test_split, fit_train_artifacts
from tests.test_experiment_runner import synthetic_experiment_data
from tests.test_production_walk_forward import _production_config


def _assert_tree_allclose(left: object, right: object) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        assert_allclose(left_leaf, right_leaf, rtol=1e-6, atol=1e-8)


def test_production_test_evaluation_keeps_policy_frozen() -> None:
    data = synthetic_experiment_data()
    config = _production_config()
    split = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)[0]
    artifacts = fit_train_artifacts(
        split,
        data.features,
        data.returns,
        data.spy_returns,
        config,
        split_index=0,
    )
    assert artifacts.production_policy_state is not None
    actor_before = artifacts.production_policy_state.actor.params
    critic_before = artifacts.production_policy_state.critic.params

    result = evaluate_test_split(
        split,
        artifacts,
        data.returns,
        data.spy_returns,
        config,
        split_index=0,
    )

    actor_after = artifacts.production_policy_state.actor.params
    critic_after = artifacts.production_policy_state.critic.params
    _assert_tree_allclose(actor_before, actor_after)
    _assert_tree_allclose(critic_before, critic_after)
    assert result.test_start == split.test_start
    assert result.test_end == split.test_end


def test_production_train_artifacts_record_split_train_window_only() -> None:
    data = synthetic_experiment_data()
    config = _production_config()
    split = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)[0]

    artifacts = fit_train_artifacts(
        split,
        data.features,
        data.returns,
        data.spy_returns,
        config,
        split_index=0,
    )

    assert artifacts.preprocessor.fit_window.start >= split.train_start
    assert artifacts.preprocessor.fit_window.end <= split.train_end
    assert artifacts.fitted_hmm.metadata.train_window is not None
    assert artifacts.fitted_hmm.metadata.train_window.start == split.train_start
    assert artifacts.fitted_hmm.metadata.train_window.end == split.train_end
    assert artifacts.production_policy_state is not None
    assert artifacts.policy_checkpoint is None
