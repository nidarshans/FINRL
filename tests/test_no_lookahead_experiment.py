"""No-look-ahead tests for walk-forward experiment artifacts."""

from __future__ import annotations

from finrl.backtest.walk_forward import generate_walk_forward_splits
from finrl.experiments.run_walk_forward import fit_train_artifacts, run_split
from tests.test_experiment_runner import (
    synthetic_experiment_config,
    synthetic_experiment_data,
)


def test_train_fitted_artifacts_match_each_split_train_window() -> None:
    data = synthetic_experiment_data()
    config = synthetic_experiment_config(enable_ppo=True)
    splits = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)

    for split_index, split in enumerate(splits):
        artifacts = fit_train_artifacts(
            split,
            data.features,
            data.returns,
            data.spy_returns,
            config,
            split_index,
        )

        assert artifacts.preprocessor.fit_window.start >= split.train_start
        assert artifacts.preprocessor.fit_window.end <= split.train_end
        assert artifacts.fitted_hmm.metadata.train_window is not None
        assert artifacts.fitted_hmm.metadata.train_window.start == split.train_start
        assert artifacts.fitted_hmm.metadata.train_window.end == split.train_end
        assert artifacts.production_policy_state is not None
        assert artifacts.production_ppo_training is not None
        assert artifacts.production_ppo_training.rollout.batch.rewards.shape[0] == len(
            artifacts.train_windows.decision_dates
        )


def test_run_split_reuses_frozen_artifacts_for_test_evaluation() -> None:
    data = synthetic_experiment_data()
    config = synthetic_experiment_config(enable_ppo=True)
    split = generate_walk_forward_splits(data.features.decision_dates, config.walk_forward)[0]

    run = run_split(split, data, config, split_index=0)

    assert run.artifacts.production_policy_state is not None
    assert run.result.test_start == split.test_start
    assert run.result.test_end == split.test_end
    assert run.result.portfolio_returns.height == len(run.artifacts.test_windows.decision_dates)
    assert run.result.spy_returns.height == run.result.portfolio_returns.height
