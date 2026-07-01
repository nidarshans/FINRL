"""Strict walk-forward experiment runner."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from finrl.backtest.results import (
    SplitResult,
    WalkForwardResult,
    calculate_performance_metrics,
    equity_curve,
)
from finrl.backtest.walk_forward import (
    WalkForwardSplit,
    generate_walk_forward_splits,
    slice_feature_bundle,
)
from finrl.dpo_jax import (
    DPOBatch,
    DPOTrainState,
    build_dpo_batch,
    predict_weights,
    train_dpo,
    initialize_dpo_train_state,
)
from finrl.dpo_jax.losses import DPOLossMetrics
from finrl.env.trading_env import EnvState, scan_environment
from finrl.experiments.artifacts import ExperimentArtifacts, RawExperimentData
from finrl.experiments.config import ExperimentConfig
from finrl.features.columns import (
    DirectAllocationRoutingMetadata,
    selected_direct_allocation_indices,
)
from finrl.features.preprocessing import fit_transform_train_transform_test
from finrl.features.schema import FeatureBundle
from finrl.features.panels import AssetFeaturePanel, build_asset_feature_panel


class SplitRunResult(NamedTuple):
    """Artifacts and result for one split."""

    artifacts: ExperimentArtifacts
    result: SplitResult


CASH_RETURN_COLUMN = "CASH"


def _action_return_columns(panel: AssetFeaturePanel) -> tuple[str, ...]:
    return (*panel.tickers, CASH_RETURN_COLUMN)


def _returns_for_dates(
    frame: pl.DataFrame,
    dates: tuple[object, ...],
    columns: tuple[str, ...],
) -> np.ndarray:
    date_frame = pl.DataFrame({"decision_date": list(dates)}).with_columns(
        pl.col("decision_date").cast(pl.Date)
    )
    requested = []
    for column in columns:
        if column in frame.columns:
            requested.append(pl.col(column))
        elif column == CASH_RETURN_COLUMN:
            requested.append(pl.lit(0.0).alias(CASH_RETURN_COLUMN))
        else:
            raise ValueError(f"Return table is missing required column: {column}.")
    aligned = date_frame.join(frame, on="decision_date", how="left").select(requested)
    if aligned.null_count().row(0) != (0,) * len(columns):
        raise ValueError("Return table is missing one or more aligned decision dates.")
    return aligned.to_numpy().astype(np.float32, copy=False)


def _spy_for_dates(frame: pl.DataFrame, dates: tuple[object, ...]) -> np.ndarray:
    if "spy_return" not in frame.columns:
        raise ValueError("spy_returns must contain 'spy_return'.")
    return _returns_for_dates(frame, dates, ("spy_return",)).reshape(-1)


def _returns_for_panel_tickers(frame: pl.DataFrame, panel: AssetFeaturePanel) -> np.ndarray:
    return _returns_for_dates(frame, panel.decision_dates, _action_return_columns(panel))


def _initial_env_state(n_assets: int) -> EnvState:
    weights = jnp.ones((n_assets,), dtype=jnp.float32) / n_assets
    return EnvState(
        weights=weights,
        portfolio_value=jnp.array(1.0, dtype=jnp.float32),
        peak_value=jnp.array(1.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def fit_dpo_train_artifacts(
    train_features: AssetFeaturePanel,
    train_returns: np.ndarray,
    config: ExperimentConfig,
    split_index: int = 0,
    feature_routing: DirectAllocationRoutingMetadata | None = None,
    train_spy: np.ndarray | None = None,
) -> tuple[DPOTrainState, tuple[DPOLossMetrics, ...]]:
    """Fit direct portfolio optimization from decision-date asset features."""

    routing = feature_routing or selected_direct_allocation_indices(
        train_features.feature_columns
    )
    dpo_state = initialize_dpo_train_state(
        jax.random.PRNGKey(config.seed + split_index),
        config.dpo,
        train_features.values.shape[1],
        train_features.values.shape[2],
        routing.direct_allocation_indices,
    )
    initial_weights = jnp.zeros((train_returns.shape[1],), dtype=jnp.float32).at[-1].set(1.0)
    batch = build_dpo_batch(
        jnp.asarray(train_features.values, dtype=jnp.float32),
        jnp.asarray(train_returns[:, :-1], dtype=jnp.float32),
        spy_returns=(
            None if train_spy is None else jnp.asarray(train_spy, dtype=jnp.float32)
        ),
        initial_weights=initial_weights,
    )
    return train_dpo(dpo_state, batch)


def evaluate_dpo_policy(
    policy_state: DPOTrainState,
    test_features: AssetFeaturePanel,
    test_returns: np.ndarray,
    test_spy: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a frozen DPO policy through the trading environment."""

    initial_weights = jnp.zeros((test_returns.shape[1],), dtype=jnp.float32).at[-1].set(1.0)
    batch = DPOBatch(
        asset_features=jnp.asarray(test_features.values, dtype=jnp.float32),
        asset_returns=jnp.asarray(test_returns[:, :-1], dtype=jnp.float32),
        spy_returns=jnp.asarray(test_spy, dtype=jnp.float32),
        previous_weights=jnp.repeat(initial_weights[None, :], test_returns.shape[0], axis=0),
        drawdowns=jnp.zeros((test_returns.shape[0], 1), dtype=jnp.float32),
        previous_turnovers=jnp.zeros((test_returns.shape[0], 1), dtype=jnp.float32),
        initial_weights=initial_weights,
    )
    actions = predict_weights(policy_state, batch)
    _, step_results = scan_environment(
        _initial_env_state(test_returns.shape[1]),
        actions,
        jnp.asarray(test_returns, dtype=jnp.float32),
        jnp.asarray(test_spy, dtype=jnp.float32),
        config.env,
    )
    return (
        np.asarray(step_results.net_return),
        np.asarray(step_results.turnover),
        np.asarray(step_results.transaction_cost),
        np.asarray(actions),
    )


def _environment_only_actions(n_steps: int, n_assets: int) -> jax.Array:
    weights = jnp.ones((n_assets,), dtype=jnp.float32) / n_assets
    return jnp.repeat(weights[None, :], n_steps, axis=0)


def _allocation_frame(
    dates: tuple[object, ...],
    actions: np.ndarray,
    asset_columns: tuple[str, ...],
    split_index: int,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_date": list(dates),
            "split_index": [split_index] * len(dates),
            **{asset: actions[:, index] for index, asset in enumerate(asset_columns)},
        }
    )


def fit_train_artifacts(
    split: WalkForwardSplit,
    features: FeatureBundle,
    returns: pl.DataFrame,
    config: ExperimentConfig,
    split_index: int = 0,
    spy_returns: pl.DataFrame | None = None,
) -> ExperimentArtifacts:
    """Fit preprocessing and optional direct-allocation DPO on train only."""

    train_features, test_features = slice_feature_bundle(features, split)
    preprocessed = fit_transform_train_transform_test(train_features, test_features, config.preprocessing)
    train_panel = build_asset_feature_panel(preprocessed.train)
    test_panel = build_asset_feature_panel(preprocessed.test)
    train_returns = _returns_for_panel_tickers(returns, train_panel)
    feature_routing = None
    dpo_policy_state = None
    dpo_train_metrics = None
    if config.enable_dpo:
        if spy_returns is None:
            raise ValueError("DPO training requires SPY returns.")
        train_spy = _spy_for_dates(spy_returns, train_panel.decision_dates)
        feature_routing = selected_direct_allocation_indices(
            train_panel.feature_columns
        )
        dpo_policy_state, dpo_train_metrics = fit_dpo_train_artifacts(
            train_panel,
            train_returns,
            config,
            split_index,
            feature_routing,
            train_spy,
        )
    return ExperimentArtifacts(
        split=split,
        preprocessor=preprocessed.preprocessor,
        train_features=train_panel,
        test_features=test_panel,
        feature_routing=feature_routing,
        dpo_policy_state=dpo_policy_state,
        dpo_train_metrics=dpo_train_metrics,
    )


def run_benchmark_suite(
    split: WalkForwardSplit,
    spy_returns: np.ndarray,
    config: ExperimentConfig,
) -> pl.DataFrame:
    """Return the S&P 500 benchmark stream for the split."""

    del config
    return pl.DataFrame(
        {
            "decision_date": list(split.test_decision_dates[-len(spy_returns) :]),
            "spy_return": spy_returns,
        }
    )


def evaluate_test_split(
    split: WalkForwardSplit,
    frozen_artifacts: ExperimentArtifacts,
    returns: pl.DataFrame,
    spy_returns: pl.DataFrame,
    config: ExperimentConfig,
    split_index: int = 0,
) -> SplitResult:
    """Evaluate a split with frozen train-fitted artifacts."""

    del split
    return_columns = _action_return_columns(frozen_artifacts.test_features)
    test_dates = frozen_artifacts.test_features.decision_dates
    test_returns = _returns_for_dates(returns, test_dates, return_columns)
    test_spy = _spy_for_dates(spy_returns, test_dates)
    if config.enable_dpo:
        if frozen_artifacts.dpo_policy_state is None:
            raise ValueError("DPO evaluation requires a policy state.")
        portfolio_returns, turnovers, costs, actions = evaluate_dpo_policy(
            frozen_artifacts.dpo_policy_state,
            frozen_artifacts.test_features,
            test_returns,
            test_spy,
            config,
        )
    else:
        actions = _environment_only_actions(test_returns.shape[0], test_returns.shape[1])
        _, step_results = scan_environment(
            _initial_env_state(test_returns.shape[1]),
            actions,
            jnp.asarray(test_returns, dtype=jnp.float32),
            jnp.asarray(test_spy, dtype=jnp.float32),
            config.env,
        )
        portfolio_returns = np.asarray(step_results.net_return)
        turnovers = np.asarray(step_results.turnover)
        costs = np.asarray(step_results.transaction_cost)
        actions = np.asarray(actions)

    benchmark = run_benchmark_suite(frozen_artifacts.split, test_spy, config)
    metrics = calculate_performance_metrics(
        portfolio_returns,
        test_spy,
        turnovers,
        costs,
        config.annualization_periods,
    )
    benchmark_metrics = calculate_performance_metrics(
        test_spy,
        test_spy,
        np.zeros_like(test_spy),
        np.zeros_like(test_spy),
        config.annualization_periods,
    )
    portfolio_frame = pl.DataFrame(
        {
            "decision_date": list(test_dates),
            "portfolio_return": portfolio_returns,
            "turnover": turnovers,
            "transaction_cost": costs,
            "split_index": [split_index] * len(test_dates),
        }
    )
    allocation_frame = _allocation_frame(test_dates, actions, return_columns, split_index)
    empty_regime_frame = pl.DataFrame(
        {"decision_date": list(test_dates), "split_index": [split_index] * len(test_dates)}
    )
    empty_spectral_frame = pl.DataFrame(
        {"decision_date": list(test_dates), "split_index": [split_index] * len(test_dates)}
    )
    return SplitResult(
        split_index=split_index,
        train_start=frozen_artifacts.split.train_start,
        train_end=frozen_artifacts.split.train_end,
        test_start=frozen_artifacts.split.test_start,
        test_end=frozen_artifacts.split.test_end,
        portfolio_returns=portfolio_frame,
        spy_returns=benchmark,
        allocations=allocation_frame,
        regime_probabilities=empty_regime_frame,
        spectral_features=empty_spectral_frame,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )


def run_split(
    split: WalkForwardSplit,
    raw_data: RawExperimentData,
    config: ExperimentConfig,
    split_index: int = 0,
) -> SplitRunResult:
    """Fit train artifacts and evaluate one frozen test split."""

    frozen_artifacts = fit_train_artifacts(
        split,
        raw_data.features,
        raw_data.returns,
        config,
        split_index,
        spy_returns=raw_data.spy_returns,
    )
    result = evaluate_test_split(
        split,
        frozen_artifacts,
        raw_data.returns,
        raw_data.spy_returns,
        config,
        split_index,
    )
    return SplitRunResult(artifacts=frozen_artifacts, result=result)


def aggregate_walk_forward_results(
    results: tuple[SplitResult, ...],
    config: ExperimentConfig,
) -> WalkForwardResult:
    """Aggregate split results into contiguous curves and metrics."""

    if not results:
        raise ValueError("Cannot aggregate empty walk-forward results.")
    portfolio_returns = pl.concat([result.portfolio_returns for result in results], how="vertical")
    spy_returns = pl.concat([result.spy_returns for result in results], how="vertical")
    allocations = pl.concat([result.allocations for result in results], how="vertical")
    regime_probabilities = pl.concat([result.regime_probabilities for result in results], how="vertical")
    spectral = pl.concat([result.spectral_features for result in results], how="vertical")
    portfolio_curve_values = equity_curve(portfolio_returns["portfolio_return"].to_numpy())
    spy_curve_values = equity_curve(spy_returns["spy_return"].to_numpy())
    portfolio_curve = portfolio_returns.select("decision_date").with_columns(
        pl.Series("equity", portfolio_curve_values)
    )
    spy_curve = spy_returns.select("decision_date").with_columns(
        pl.Series("equity", spy_curve_values)
    )
    aggregate_metrics = calculate_performance_metrics(
        portfolio_returns["portfolio_return"].to_numpy(),
        spy_returns["spy_return"].to_numpy(),
        portfolio_returns["turnover"].to_numpy(),
        portfolio_returns["transaction_cost"].to_numpy(),
        config.annualization_periods,
    )
    aggregate_benchmark_metrics = calculate_performance_metrics(
        spy_returns["spy_return"].to_numpy(),
        spy_returns["spy_return"].to_numpy(),
        np.zeros(spy_returns.height),
        np.zeros(spy_returns.height),
        config.annualization_periods,
    )
    return WalkForwardResult(
        split_results=results,
        portfolio_curve=portfolio_curve,
        spy_curve=spy_curve,
        allocations=allocations,
        regime_probabilities=regime_probabilities,
        spectral_features=spectral,
        aggregate_metrics=aggregate_metrics,
        aggregate_benchmark_metrics=aggregate_benchmark_metrics,
    )


def run_walk_forward_experiment(
    raw_data: RawExperimentData,
    config: ExperimentConfig,
) -> WalkForwardResult:
    """Run all strict walk-forward splits for prepared data."""

    splits = generate_walk_forward_splits(raw_data.features.decision_dates, config.walk_forward)
    split_results = tuple(
        run_split(split, raw_data, config, split_index).result
        for split_index, split in enumerate(splits)
    )
    return aggregate_walk_forward_results(split_results, config)
