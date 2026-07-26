"""Tests for the full walk-forward experiment runner."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from numpy.testing import assert_allclose

from finrl.backtest.walk_forward import WalkForwardConfig, generate_walk_forward_splits
from finrl.dpo_jax import DPOConfig
from finrl.env.trading_env import EnvConfig
from finrl.experiments import (
    ExperimentConfig,
    RawExperimentData,
    build_allocation_figure,
    build_holdings_heatmap_granular,
    build_performance_figure,
    build_regime_portfolio_figure,
    build_spectral_figure,
    metrics_to_frame,
    run_walk_forward_experiment,
)
from finrl.features.columns import DIRECT_ALLOCATION_FEATURE_COLUMNS
from finrl.features.preprocessing import PreprocessingConfig
from finrl.features.schema import FeatureBundle


def _synthetic_dates() -> tuple[date, ...]:
    dates = []
    for year in (2020, 2021, 2022):
        start = date(year, 1, 3)
        dates.extend(start + timedelta(days=7 * index) for index in range(6))
    return tuple(dates)


def synthetic_experiment_data() -> RawExperimentData:
    dates = _synthetic_dates()
    tickers = ("AAA", "BBB")
    dpo_feature_columns = DIRECT_ALLOCATION_FEATURE_COLUMNS
    asset_rows = []
    for day_index, day in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            asset_rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "acc_return": 0.01 * (day_index + 1 + ticker_index),
                    "liq_rank": 0.25 + 0.5 * ticker_index,
                    **{
                        column: 0.001 * (day_index + 1) + 0.01 * ticker_index
                        for column in dpo_feature_columns
                    },
                }
            )
    asset = pl.DataFrame(asset_rows).with_columns(pl.col("date").cast(pl.Date))
    macro = pl.DataFrame(
        {
            "date": dates,
            "macro_rate": [0.001 * index for index in range(len(dates))],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    spectral_columns = tuple(f"spectral_{index}" for index in range(20))
    spectral = pl.DataFrame(
        [
            {
                "date": day,
                **{
                    column: float(day_index + column_index / 100.0)
                    for column_index, column in enumerate(spectral_columns)
                },
            }
            for day_index, day in enumerate(dates)
        ]
    ).with_columns(pl.col("date").cast(pl.Date))
    features = FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=dates,
        tickers=tickers,
        asset_feature_columns=("acc_return", "liq_rank", *dpo_feature_columns),
        macro_feature_columns=("macro_rate",),
        spectral_feature_columns=spectral_columns,
    )
    returns = pl.DataFrame(
        {
            "decision_date": dates,
            "AAA": [0.002 + 0.0001 * index for index in range(len(dates))],
            "BBB": [0.001 - 0.00005 * index for index in range(len(dates))],
            "CASH": [0.0001] * len(dates),
        }
    ).with_columns(pl.col("decision_date").cast(pl.Date))
    spy_returns = pl.DataFrame(
        {
            "decision_date": dates,
            "spy_return": [0.0015 + 0.00002 * index for index in range(len(dates))],
        }
    ).with_columns(pl.col("decision_date").cast(pl.Date))
    return RawExperimentData(features=features, returns=returns, spy_returns=spy_returns)


def synthetic_experiment_config(
    enable_dpo: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        walk_forward=WalkForwardConfig(train_years=1, test_years=1, step_years=1),
        preprocessing=PreprocessingConfig(rolling_window=2),
        dpo=DPOConfig(num_epochs=2, learning_rate=1e-3),
        enable_dpo=enable_dpo,
        seed=7,
        periods_per_year=6,
    )


def test_experiment_frequency_sets_default_annualization() -> None:
    assert ExperimentConfig(rebalance_frequency="weekly").annualization_periods == 52
    assert ExperimentConfig(rebalance_frequency="daily").annualization_periods == 252
    assert ExperimentConfig(
        rebalance_frequency="daily",
        periods_per_year=6,
    ).annualization_periods == 6


def test_experiment_requires_matching_dpo_and_environment_costs() -> None:
    with pytest.raises(ValueError, match="transaction cost rates must match"):
        ExperimentConfig(
            dpo=DPOConfig(transaction_cost_bps=5.0),
            env=EnvConfig(transaction_cost_rate=0.001),
            enable_dpo=True,
        )


def test_experiment_allows_eval_only_top_n_execution_for_dpo() -> None:
    config = ExperimentConfig(
        env=EnvConfig(top_n_positions=1),
        enable_dpo=True,
    )

    assert config.env.top_n_positions == 1


def test_experiment_rejects_position_cap_execution_for_dpo() -> None:
    with pytest.raises(ValueError, match="max_position_weight=None"):
        ExperimentConfig(
            env=EnvConfig(max_position_weight=0.2),
            enable_dpo=True,
        )


def test_walk_forward_experiment_runs_two_splits_with_spy_benchmark() -> None:
    result = run_walk_forward_experiment(
        synthetic_experiment_data(),
        synthetic_experiment_config(enable_dpo=False),
    )

    assert len(result.split_results) == 2
    assert result.portfolio_curve.height == result.spy_curve.height
    assert result.aggregate_metrics.spy_relative_alpha is not None
    assert result.aggregate_benchmark_metrics.spy_relative_alpha == 0.0
    assert "equity" in result.portfolio_curve.columns
    assert "equity" in result.spy_curve.columns
    assert {"AAA", "BBB", "CASH"}.issubset(result.allocations.columns)
    assert set(result.regime_probabilities.columns) == {"decision_date", "split_index"}
    assert set(result.spectral_features.columns) == {"decision_date", "split_index"}
    allocation_sums = result.allocations.select((pl.col("AAA") + pl.col("BBB") + pl.col("CASH")).alias("total"))
    assert_allclose(
        allocation_sums.get_column("total").to_numpy(),
        1.0,
        atol=1e-6,
    )


def test_reporting_helpers_create_plotly_figures_and_metrics_frame() -> None:
    result = run_walk_forward_experiment(
        synthetic_experiment_data(),
        synthetic_experiment_config(enable_dpo=False),
    )

    performance_fig = build_performance_figure(result)
    allocation_fig = build_allocation_figure(result, top_n=2)
    heatmap_fig = build_holdings_heatmap_granular(result, top_n=2)
    regime_fig = build_regime_portfolio_figure(result)
    spectral_fig = build_spectral_figure(result)
    metrics = metrics_to_frame(result)

    assert len(performance_fig.data) == 2
    assert len(allocation_fig.data) == 2
    assert len(heatmap_fig.data) == 1
    assert heatmap_fig.layout.title.text == "Portfolio Holdings Over Time"
    assert len(regime_fig.data) == 1
    assert len(spectral_fig.data) == 0
    assert metrics.height == 2
    assert "spy_cumulative_return" in metrics.columns


def test_walk_forward_experiment_is_reproducible_for_fixed_seed() -> None:
    data = synthetic_experiment_data()
    config = synthetic_experiment_config(enable_dpo=False)

    first = run_walk_forward_experiment(data, config)
    second = run_walk_forward_experiment(data, config)

    assert first.portfolio_curve.to_dicts() == second.portfolio_curve.to_dicts()
    assert first.spy_curve.to_dicts() == second.spy_curve.to_dicts()


def test_environment_only_reports_executed_top_n_allocations() -> None:
    config = replace(
        synthetic_experiment_config(enable_dpo=False),
        env=EnvConfig(top_n_positions=1),
    )

    result = run_walk_forward_experiment(synthetic_experiment_data(), config)

    risky_allocations = result.allocations.select(["AAA", "BBB"]).to_numpy()
    assert bool((risky_allocations > 0.0).sum(axis=1).max() <= 1)
    assert_allclose(result.allocations.get_column("CASH").to_numpy(), 0.5, atol=1e-6)


def test_walk_forward_experiment_runs_with_direct_allocation_policy() -> None:
    result = run_walk_forward_experiment(
        synthetic_experiment_data(),
        synthetic_experiment_config(enable_dpo=True),
    )

    assert len(result.split_results) == 2
    assert result.portfolio_curve.height == result.spy_curve.height
    assert {"AAA", "BBB", "CASH"}.issubset(result.allocations.columns)
    allocation_sums = result.allocations.select(
        (pl.col("AAA") + pl.col("BBB") + pl.col("CASH")).alias("total")
    )
    assert_allclose(
        allocation_sums.get_column("total").to_numpy(),
        1.0,
        atol=1e-6,
    )
    assert_allclose(result.allocations.get_column("CASH").to_numpy(), 0.0, atol=0.0)
    first_turnover = result.split_results[0].portfolio_returns.get_column("turnover")[0]
    second_turnover = result.split_results[1].portfolio_returns.get_column("turnover")[0]
    assert_allclose(first_turnover, 2.0, rtol=1e-6, atol=1e-7)
    assert second_turnover < 2.0
    assert result.split_results[0].train_start == date(2020, 1, 1)
    assert result.split_results[1].train_start == date(2020, 1, 1)
    assert result.split_results[1].train_end == date(2021, 12, 31)


def test_dpo_evaluation_reports_only_top_n_executed_positions() -> None:
    base_config = synthetic_experiment_config(enable_dpo=True)
    config = replace(
        base_config,
        env=EnvConfig(top_n_positions=1),
    )

    result = run_walk_forward_experiment(synthetic_experiment_data(), config)

    risky_allocations = result.allocations.select(["AAA", "BBB"]).to_numpy()
    assert bool((risky_allocations > 0.0).sum(axis=1).max() <= 1)
