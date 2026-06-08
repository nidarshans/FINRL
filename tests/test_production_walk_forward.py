"""Focused tests for production walk-forward integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import polars as pl
from numpy.testing import assert_allclose

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
        production_encoder=ProductionEncoderConfig(
            lookback=3,
            n_assets=2,
            asset_feature_dim=2,
            macro_feature_dim=1,
            asset_hidden_dim=8,
            macro_hidden_dim=4,
            attention_heads=2,
        ),
        hmm=base.hmm,
        production_ppo=ProductionPPOConfig(
            phi_dim=8,
            asset_latent_dim=8,
            macro_dim=4,
            spectral_dim=20,
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


def test_production_split_adds_zero_return_cash_when_returns_omit_cash() -> None:
    data = synthetic_experiment_data()
    data = type(data)(
        features=data.features,
        returns=data.returns.drop("CASH"),
        spy_returns=data.spy_returns,
    )
    config = _production_config()
    split = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)[0]

    run = run_split(split, data, config, split_index=0)

    assert run.artifacts.production_ppo_training is not None
    assert run.artifacts.production_ppo_training.rollout.batch.actions.shape[-1] == 3
    assert {"AAA", "BBB", "CASH"}.issubset(run.result.allocations.columns)
    allocation_sums = run.result.allocations.select(
        (pl.col("AAA") + pl.col("BBB") + pl.col("CASH")).alias("total")
    )
    assert_allclose(
        allocation_sums.get_column("total").to_numpy(),
        1.0,
        rtol=1e-6,
        atol=1e-8,
    )


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
