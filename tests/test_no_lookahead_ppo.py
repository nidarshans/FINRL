"""No-look-ahead tests for PPO train/evaluate boundaries."""

from __future__ import annotations

from datetime import date

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from finrl.features.splitsafe import FitWindow
from finrl.ppo import PPOConfig, evaluate_frozen_policy, train_ppo_on_split
from tests.test_ppo_rollout import _artifacts


def _assert_tree_allclose(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert_allclose(left_leaf, right_leaf, rtol=1e-6, atol=1e-8)


def test_ppo_checkpoint_records_train_window_only() -> None:
    train = _artifacts()
    train = type(train)(
        phi=train.phi,
        regime_probs=train.regime_probs,
        asset_returns=train.asset_returns,
        spy_returns=train.spy_returns,
        initial_env_state=train.initial_env_state,
        env_config=train.env_config,
        fit_window=FitWindow(start=date(2020, 1, 1), end=date(2020, 12, 31)),
    )

    result = train_ppo_on_split(
        train,
        PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4),
        jax.random.PRNGKey(3),
    )

    assert result.checkpoint.train_window == train.fit_window


def test_frozen_policy_evaluation_does_not_update_parameters() -> None:
    config = PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4)
    trained = train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(4))
    before = trained.checkpoint.state

    evaluation = evaluate_frozen_policy(
        trained.checkpoint,
        _artifacts(),
        jax.random.PRNGKey(5),
    )

    _assert_tree_allclose(before.actor_params, evaluation.checkpoint.state.actor_params)
    _assert_tree_allclose(before.critic_params, evaluation.checkpoint.state.critic_params)
    assert before.step == evaluation.checkpoint.state.step
    assert jnp.isfinite(evaluation.trajectory.rewards).all()

