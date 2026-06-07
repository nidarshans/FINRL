"""Tests for the full walk-forward experiment runner."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from finrl.backtest.walk_forward import WalkForwardConfig, generate_walk_forward_splits
from finrl.experiments import (
    ExperimentConfig,
    RawExperimentData,
    build_allocation_figure,
    build_performance_figure,
    build_regime_portfolio_figure,
    build_spectral_figure,
    metrics_to_frame,
    run_walk_forward_experiment,
)
from finrl.features.preprocessing import PreprocessingConfig
from finrl.features.schema import FeatureBundle
from finrl.models.encoder import EncoderConfig
from finrl.ppo.policy import PPOConfig
from finrl.regimes.schema import HMMConfig


def _synthetic_dates() -> tuple[date, ...]:
    dates = []
    for year in (2020, 2021, 2022):
        start = date(year, 1, 3)
        dates.extend(start + timedelta(days=7 * index) for index in range(6))
    return tuple(dates)


def synthetic_experiment_data() -> RawExperimentData:
    dates = _synthetic_dates()
    tickers = ("AAA", "BBB")
    asset_rows = []
    for day_index, day in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            asset_rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "asset_return": 0.01 * (day_index + 1 + ticker_index),
                    "asset_rank": 0.25 + 0.5 * ticker_index,
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
        asset_feature_columns=("asset_return", "asset_rank"),
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


def synthetic_experiment_config(enable_ppo: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        walk_forward=WalkForwardConfig(train_years=1, test_years=1, step_years=1),
        preprocessing=PreprocessingConfig(rolling_window=2),
        encoder=EncoderConfig(
            lookback=3,
            n_assets=2,
            asset_feature_dim=2,
            macro_feature_dim=1,
            spectral_feature_dim=20,
        ),
        hmm=HMMConfig(n_states=2, max_iter=5),
        ppo=PPOConfig(n_assets=3, train_epochs=1, learning_rate=1e-4),
        enable_ppo=enable_ppo,
        seed=7,
        periods_per_year=6,
    )


def test_walk_forward_experiment_runs_two_splits_with_spy_benchmark() -> None:
    result = run_walk_forward_experiment(
        synthetic_experiment_data(),
        synthetic_experiment_config(enable_ppo=False),
    )

    assert len(result.split_results) == 2
    assert result.portfolio_curve.height == result.spy_curve.height
    assert result.aggregate_metrics.spy_relative_alpha is not None
    assert result.aggregate_benchmark_metrics.spy_relative_alpha == 0.0
    assert "equity" in result.portfolio_curve.columns
    assert "equity" in result.spy_curve.columns
    assert {"AAA", "BBB", "CASH"}.issubset(result.allocations.columns)
    assert {"regime_0", "regime_1"}.issubset(result.regime_probabilities.columns)
    assert "spectral_0" in result.spectral_features.columns
    allocation_sums = result.allocations.select((pl.col("AAA") + pl.col("BBB") + pl.col("CASH")).alias("total"))
    assert allocation_sums.get_column("total").to_list() == [1.0] * result.allocations.height


def test_reporting_helpers_create_plotly_figures_and_metrics_frame() -> None:
    result = run_walk_forward_experiment(
        synthetic_experiment_data(),
        synthetic_experiment_config(enable_ppo=False),
    )

    performance_fig = build_performance_figure(result)
    allocation_fig = build_allocation_figure(result, top_n=2)
    regime_fig = build_regime_portfolio_figure(result)
    spectral_fig = build_spectral_figure(result, value_columns=("spectral_0", "spectral_1"))
    metrics = metrics_to_frame(result)

    assert len(performance_fig.data) == 2
    assert len(allocation_fig.data) == 2
    assert len(regime_fig.data) == 5
    assert len(spectral_fig.data) == 2
    assert metrics.height == 2
    assert "spy_cumulative_return" in metrics.columns


def test_walk_forward_experiment_is_reproducible_for_fixed_seed() -> None:
    data = synthetic_experiment_data()
    config = synthetic_experiment_config(enable_ppo=False)

    first = run_walk_forward_experiment(data, config)
    second = run_walk_forward_experiment(data, config)

    assert first.portfolio_curve.to_dicts() == second.portfolio_curve.to_dicts()
    assert first.spy_curve.to_dicts() == second.spy_curve.to_dicts()
