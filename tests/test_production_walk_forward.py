"""Focused tests for production walk-forward integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from finrl.backtest.walk_forward import generate_walk_forward_splits
from finrl.experiments import ExperimentConfig, run_split, run_walk_forward_experiment
from finrl.models.flax_encoder import ProductionEncoderConfig
from finrl.ppo.flax_policy import ProductionPPOConfig
from tests.test_experiment_runner import (
    synthetic_experiment_config,
    synthetic_experiment_data,
)


def _production_config() -> ExperimentConfig:
    base = synthetic_experiment_config(enable_ppo=True)
    return ExperimentConfig(
        walk_forward=base.walk_forward,
        preprocessing=base.preprocessing,
        encoder=base.encoder,
        production_encoder=ProductionEncoderConfig(
            lookback=3,
            n_assets=2,
            asset_feature_dim=2,
            macro_feature_dim=1,
            asset_hidden_dim=8,
            macro_hidden_dim=4,
            attention_heads=2,
            fusion_hidden_dim=12,
            output_dim=8,
        ),
        hmm=base.hmm,
        ppo=base.ppo,
        production_ppo=ProductionPPOConfig(
            phi_dim=8,
            n_regimes=2,
            n_assets=3,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            update_epochs=1,
            minibatch_size=2,
            learning_rate=1e-3,
            dirichlet_concentration=12.0,
        ),
        env=base.env,
        enable_ppo=True,
        use_production_pipeline=True,
        seed=base.seed,
        periods_per_year=base.periods_per_year,
    )


def _tree_delta(left: object, right: object) -> float:
    return sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    )


def test_production_walk_forward_run_is_deterministic_for_fixed_seed() -> None:
    data = synthetic_experiment_data()
    config = _production_config()

    first = run_walk_forward_experiment(data, config)
    second = run_walk_forward_experiment(data, config)

    assert len(first.split_results) == 2
    assert first.portfolio_curve.to_dicts() == second.portfolio_curve.to_dicts()
    assert first.spy_curve.to_dicts() == second.spy_curve.to_dicts()
    assert first.allocations.to_dicts() == second.allocations.to_dicts()


def test_production_split_dates_align_with_test_windows_and_spy() -> None:
    data = synthetic_experiment_data()
    config = _production_config()
    split = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)[0]

    run = run_split(split, data, config, split_index=0)
    test_dates = list(run.artifacts.test_windows.decision_dates)

    assert run.artifacts.production_encoder_training is not None
    assert run.artifacts.production_ppo_training is not None
    assert run.artifacts.production_policy_state is not None
    assert run.result.portfolio_returns["decision_date"].to_list() == test_dates
    assert run.result.spy_returns["decision_date"].to_list() == test_dates
    assert run.result.portfolio_returns.height == len(test_dates)


def test_production_splits_have_independent_train_artifacts() -> None:
    data = synthetic_experiment_data()
    config = _production_config()
    splits = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)

    first = run_split(splits[0], data, config, split_index=0).artifacts
    second = run_split(splits[1], data, config, split_index=1).artifacts

    assert first.production_policy_state is not None
    assert second.production_policy_state is not None
    assert first.fitted_hmm.metadata.train_window is not None
    assert second.fitted_hmm.metadata.train_window is not None
    assert first.fitted_hmm.metadata.train_window != second.fitted_hmm.metadata.train_window
    assert (
        _tree_delta(
            first.production_policy_state.actor.params,
            second.production_policy_state.actor.params,
        )
        > 0.0
    )
