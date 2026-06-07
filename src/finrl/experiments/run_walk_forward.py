"""Strict walk-forward experiment runner."""

from __future__ import annotations

from dataclasses import replace
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
    slice_returns,
)
from finrl.env.trading_env import EnvState, scan_environment
from finrl.experiments.artifacts import ExperimentArtifacts, RawExperimentData
from finrl.experiments.config import ExperimentConfig
from finrl.features.preprocessing import fit_transform_train_transform_test
from finrl.features.schema import FeatureBundle
from finrl.features.splitsafe import FitWindow
from finrl.logging.tensorboard import TensorBoardLogger
from finrl.models.encoder import FeatureWindow, MarketEncoder, encode_market_state
from finrl.models.encoder_training import (
    EncoderTrainingConfig,
    fit_encoder_on_train_split,
)
from finrl.models.flax_encoder import (
    MarketEncoderFlax,
    ProductionEncoderConfig,
    init_encoder_variables,
)
from finrl.models.windows import LookbackWindows, build_lookback_windows
from finrl.ppo.flax_policy import ProductionPPOConfig
from finrl.ppo.flax_trainer import (
    ProductionPPOEvaluationResult,
    ProductionPPOTrainState,
    ProductionPPOTrainingResult,
    evaluate_frozen_policy as evaluate_frozen_flax_policy,
    train_ppo_on_split as train_flax_ppo_on_split,
)
from finrl.ppo.trainer import (
    PPOArtifacts,
    evaluate_frozen_policy,
    train_ppo_on_split,
)
from finrl.regimes.filtering import filter_regime_probabilities
from finrl.regimes.hmm import fit_hmm


class SplitRunResult(NamedTuple):
    """Artifacts and result for one split."""

    artifacts: ExperimentArtifacts
    result: SplitResult


def _combine_feature_bundles(train: FeatureBundle, test: FeatureBundle) -> FeatureBundle:
    decision_dates = tuple([*train.decision_dates, *test.decision_dates])
    return FeatureBundle(
        asset_features=pl.concat([train.asset_features, test.asset_features], how="vertical"),
        macro_features=pl.concat([train.macro_features, test.macro_features], how="vertical"),
        spectral_features=pl.concat([train.spectral_features, test.spectral_features], how="vertical"),
        decision_dates=decision_dates,
        tickers=train.tickers,
        asset_feature_columns=train.asset_feature_columns,
        macro_feature_columns=train.macro_feature_columns,
        spectral_feature_columns=train.spectral_feature_columns,
    )


def _split_windows(
    windows: LookbackWindows,
    train_dates: tuple[object, ...],
    test_dates: tuple[object, ...],
) -> tuple[LookbackWindows, LookbackWindows]:
    train_date_set = set(train_dates)
    test_date_set = set(test_dates)
    train_indices = [index for index, day in enumerate(windows.decision_dates) if day in train_date_set]
    test_indices = [index for index, day in enumerate(windows.decision_dates) if day in test_date_set]
    return (
        _select_windows(windows, train_indices),
        _select_windows(windows, test_indices),
    )


def _select_windows(windows: LookbackWindows, indices: list[int]) -> LookbackWindows:
    if not indices:
        raise ValueError("No lookback windows align to requested dates.")
    return LookbackWindows(
        asset=windows.asset[indices],
        macro=windows.macro[indices],
        spectral=windows.spectral[indices],
        decision_dates=tuple(windows.decision_dates[index] for index in indices),
        tickers=windows.tickers,
        asset_feature_columns=windows.asset_feature_columns,
        macro_feature_columns=windows.macro_feature_columns,
        spectral_feature_columns=windows.spectral_feature_columns,
    )


def _encode_windows(
    windows: LookbackWindows,
    config: ExperimentConfig,
    split_index: int,
) -> np.ndarray:
    encoder = MarketEncoder(config.encoder)
    key = jax.random.PRNGKey(config.seed + split_index)
    params = encoder.init(key)

    def encode_one(asset_window, macro_window, spectral_row):
        return encode_market_state(
            params,
            FeatureWindow(asset_window, macro_window, spectral_row),
        )

    phi = jax.vmap(encode_one)(
        jnp.asarray(windows.asset, dtype=jnp.float32),
        jnp.asarray(windows.macro, dtype=jnp.float32),
        jnp.asarray(windows.spectral, dtype=jnp.float32),
    )
    return np.asarray(phi, dtype=np.float64)


def _production_encoder_config(
    train_windows: LookbackWindows,
    config: ExperimentConfig,
) -> ProductionEncoderConfig:
    return replace(
        config.production_encoder,
        lookback=train_windows.asset.shape[1],
        n_assets=train_windows.asset.shape[2],
        asset_feature_dim=train_windows.asset.shape[3],
        macro_feature_dim=train_windows.macro.shape[2],
        spectral_feature_dim=train_windows.spectral.shape[1],
    )


def _encode_windows_flax(
    windows: LookbackWindows,
    encoder_config: ProductionEncoderConfig,
    variables: dict[str, object],
) -> np.ndarray:
    encoder = MarketEncoderFlax(encoder_config)
    phi = jax.vmap(
        lambda asset_window, macro_window, spectral_row: encoder.apply(
            variables,
            asset_window,
            macro_window,
            spectral_row,
        )
    )(
        jnp.asarray(windows.asset, dtype=jnp.float32),
        jnp.asarray(windows.macro, dtype=jnp.float32),
        jnp.asarray(windows.spectral, dtype=jnp.float32),
    )
    return np.asarray(phi, dtype=np.float64)


def _return_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if column != "decision_date")


def _returns_for_dates(
    frame: pl.DataFrame,
    dates: tuple[object, ...],
    columns: tuple[str, ...],
) -> np.ndarray:
    date_frame = pl.DataFrame({"decision_date": list(dates)}).with_columns(
        pl.col("decision_date").cast(pl.Date)
    )
    aligned = date_frame.join(frame, on="decision_date", how="left").select(columns)
    if aligned.null_count().row(0) != (0,) * len(columns):
        raise ValueError("Return table is missing one or more aligned decision dates.")
    return aligned.to_numpy().astype(np.float32, copy=False)


def _spy_for_dates(frame: pl.DataFrame, dates: tuple[object, ...]) -> np.ndarray:
    if "spy_return" not in frame.columns:
        raise ValueError("spy_returns must contain 'spy_return'.")
    return _returns_for_dates(frame, dates, ("spy_return",)).reshape(-1)


def _returns_for_window_tickers(
    frame: pl.DataFrame,
    windows: LookbackWindows,
) -> np.ndarray:
    return _returns_for_dates(frame, windows.decision_dates, windows.tickers)


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


def _make_ppo_artifacts(
    phi: np.ndarray,
    regime_probs: np.ndarray,
    returns: np.ndarray,
    spy_returns: np.ndarray,
    config: ExperimentConfig,
    fit_window: FitWindow | None,
) -> PPOArtifacts:
    return PPOArtifacts(
        phi=jnp.asarray(phi, dtype=jnp.float32),
        regime_probs=jnp.asarray(regime_probs, dtype=jnp.float32),
        asset_returns=jnp.asarray(returns, dtype=jnp.float32),
        spy_returns=jnp.asarray(spy_returns, dtype=jnp.float32),
        initial_env_state=_initial_env_state(returns.shape[1]),
        env_config=config.env,
        fit_window=fit_window,
    )


def fit_encoder_train_artifacts(
    train_windows: LookbackWindows,
    test_windows: LookbackWindows,
    returns: pl.DataFrame,
    config: ExperimentConfig,
    split_index: int = 0,
    logger: TensorBoardLogger | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Fit production encoder on train windows and encode train/test windows."""

    encoder_config = _production_encoder_config(train_windows, config)
    train_key = jax.random.PRNGKey(config.seed + split_index)
    train_returns = _returns_for_window_tickers(returns, train_windows)
    training_config = EncoderTrainingConfig(
        batch_size=min(32, max(1, train_windows.asset.shape[0] - 1)),
        epochs=1,
        learning_rate=config.production_ppo.learning_rate,
    )
    if train_windows.asset.shape[0] <= training_config.label_horizon:
        variables = init_encoder_variables(train_key, encoder_config)
        train_phi = _encode_windows_flax(train_windows, encoder_config, variables)
        test_phi = _encode_windows_flax(test_windows, encoder_config, variables)
        return train_phi, test_phi, None

    training = fit_encoder_on_train_split(
        train_key,
        train_windows,
        train_returns,
        encoder_config,
        training_config,
        train_window_count=train_windows.asset.shape[0],
        logger=logger,
    )
    variables = {"params": training.train_state.params["encoder"]}
    train_phi = _encode_windows_flax(train_windows, encoder_config, variables)
    test_phi = _encode_windows_flax(test_windows, encoder_config, variables)
    return train_phi, test_phi, training


def fit_hmm_train_artifacts(
    train_phi: np.ndarray,
    test_phi: np.ndarray,
    split: WalkForwardSplit,
    config: ExperimentConfig,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Fit HMM on train encodings and filter train/test probabilities."""

    hmm_config = replace(
        config.hmm,
        train_start=split.train_start,
        train_end=split.train_end,
    )
    fitted_hmm = fit_hmm(train_phi, hmm_config)
    train_regime_probs = filter_regime_probabilities(fitted_hmm, train_phi)
    combined_regime_probs = filter_regime_probabilities(
        fitted_hmm,
        np.concatenate([train_phi, test_phi], axis=0),
    )
    test_regime_probs = combined_regime_probs[-test_phi.shape[0] :]
    return fitted_hmm, train_regime_probs, test_regime_probs


def fit_ppo_train_artifacts(
    train_phi: np.ndarray,
    train_regime_probs: np.ndarray,
    train_returns: np.ndarray,
    train_spy: np.ndarray,
    config: ExperimentConfig,
    split_index: int = 0,
    logger: TensorBoardLogger | None = None,
) -> ProductionPPOTrainingResult:
    """Fit production PPO on train arrays only."""

    ppo_config = replace(
        config.production_ppo,
        phi_dim=train_phi.shape[1],
        n_regimes=train_regime_probs.shape[1],
        n_assets=train_returns.shape[1],
        minibatch_size=min(config.production_ppo.minibatch_size, train_phi.shape[0]),
    )
    return train_flax_ppo_on_split(
        jnp.asarray(train_phi, dtype=jnp.float32),
        jnp.asarray(train_regime_probs, dtype=jnp.float32),
        jnp.asarray(train_returns, dtype=jnp.float32),
        jnp.asarray(train_spy, dtype=jnp.float32),
        _initial_env_state(train_returns.shape[1]),
        config.env,
        ppo_config,
        jax.random.PRNGKey(config.seed + split_index),
        rollout_length=train_phi.shape[0],
        logger=logger,
    )


def evaluate_frozen_production_policy(
    policy_state: ProductionPPOTrainState,
    test_phi: np.ndarray,
    test_regime_probs: np.ndarray,
    test_returns: np.ndarray,
    test_spy: np.ndarray,
    config: ExperimentConfig,
    split_index: int = 0,
) -> ProductionPPOEvaluationResult:
    """Evaluate a frozen production PPO policy on test arrays."""

    return evaluate_frozen_flax_policy(
        policy_state,
        jnp.asarray(test_phi, dtype=jnp.float32),
        jnp.asarray(test_regime_probs, dtype=jnp.float32),
        jnp.asarray(test_returns, dtype=jnp.float32),
        jnp.asarray(test_spy, dtype=jnp.float32),
        _initial_env_state(test_returns.shape[1]),
        config.env,
        jax.random.PRNGKey(config.seed + split_index + 10_000),
        rollout_length=test_phi.shape[0],
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
            **{
                asset: actions[:, index]
                for index, asset in enumerate(asset_columns)
            },
        }
    )


def fit_train_artifacts(
    split: WalkForwardSplit,
    features: FeatureBundle,
    returns: pl.DataFrame,
    spy_returns: pl.DataFrame,
    config: ExperimentConfig,
    split_index: int = 0,
    logger: TensorBoardLogger | None = None,
) -> ExperimentArtifacts:
    """Fit preprocessing, encoder states, HMM, and optional PPO on train only."""

    train_features, test_features = slice_feature_bundle(features, split)
    preprocessed = fit_transform_train_transform_test(
        train_features,
        test_features,
        config.preprocessing,
    )
    combined = _combine_feature_bundles(preprocessed.train, preprocessed.test)
    combined_windows = build_lookback_windows(combined, config.encoder.lookback)
    train_windows, test_windows = _split_windows(
        combined_windows,
        preprocessed.train.decision_dates,
        preprocessed.test.decision_dates,
    )
    production_encoder_training = None
    if config.use_production_pipeline:
        train_phi, test_phi, production_encoder_training = fit_encoder_train_artifacts(
            train_windows,
            test_windows,
            returns,
            config,
            split_index,
            logger,
        )
    else:
        train_phi = _encode_windows(train_windows, config, split_index)
        test_phi = _encode_windows(test_windows, config, split_index)
    fitted_hmm, train_regime_probs, test_regime_probs = fit_hmm_train_artifacts(
        train_phi,
        test_phi,
        split,
        config,
    )
    return_columns = _return_columns(returns)
    train_returns = _returns_for_dates(returns, train_windows.decision_dates, return_columns)
    train_spy = _spy_for_dates(spy_returns, train_windows.decision_dates)
    ppo_training = None
    policy_checkpoint = None
    production_ppo_training = None
    production_policy_state = None
    if config.enable_ppo:
        if config.use_production_pipeline:
            production_ppo_training = fit_ppo_train_artifacts(
                train_phi,
                train_regime_probs,
                train_returns,
                train_spy,
                config,
                split_index,
                logger,
            )
            production_policy_state = production_ppo_training.train_state
        else:
            train_artifacts = _make_ppo_artifacts(
                train_phi,
                train_regime_probs,
                train_returns,
                train_spy,
                config,
                preprocessed.preprocessor.fit_window,
            )
            ppo_training = train_ppo_on_split(
                train_artifacts,
                replace(
                    config.ppo,
                    n_assets=train_returns.shape[1],
                    n_regimes=train_regime_probs.shape[1],
                ),
                jax.random.PRNGKey(config.seed + split_index),
                logger,
            )
            policy_checkpoint = ppo_training.checkpoint
    return ExperimentArtifacts(
        split=split,
        preprocessor=preprocessed.preprocessor,
        train_windows=train_windows,
        test_windows=test_windows,
        train_phi=train_phi,
        test_phi=test_phi,
        train_regime_probs=train_regime_probs,
        test_regime_probs=test_regime_probs,
        train_spy_returns=train_spy if config.enable_ppo else None,
        fitted_hmm=fitted_hmm,
        ppo_training=ppo_training,
        policy_checkpoint=policy_checkpoint,
        production_encoder_training=production_encoder_training,
        production_ppo_training=production_ppo_training,
        production_policy_state=production_policy_state,
    )


def run_benchmark_suite(
    split: WalkForwardSplit,
    spy_returns: np.ndarray,
    config: ExperimentConfig,
) -> pl.DataFrame:
    """Return the S&P 500 benchmark stream for the split.

    Additional Phase 4 benchmark strategies plug in here later.
    """

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

    return_columns = _return_columns(returns)
    test_dates = frozen_artifacts.test_windows.decision_dates
    test_returns = _returns_for_dates(returns, test_dates, return_columns)
    test_spy = _spy_for_dates(spy_returns, test_dates)
    if config.enable_ppo:
        if config.use_production_pipeline:
            if frozen_artifacts.production_policy_state is None:
                raise ValueError("Production PPO evaluation requires a policy state.")
            production_evaluation = evaluate_frozen_production_policy(
                frozen_artifacts.production_policy_state,
                frozen_artifacts.test_phi,
                frozen_artifacts.test_regime_probs,
                test_returns,
                test_spy,
                config,
                split_index,
            )
            portfolio_returns = np.asarray(
                production_evaluation.rollout.step_results.net_return
            )
            turnovers = np.asarray(production_evaluation.rollout.step_results.turnover)
            costs = np.asarray(
                production_evaluation.rollout.step_results.transaction_cost
            )
            actions = np.asarray(production_evaluation.rollout.batch.actions)
        else:
            if frozen_artifacts.policy_checkpoint is None:
                raise ValueError("PPO evaluation requires a policy checkpoint.")
            test_artifacts = _make_ppo_artifacts(
                frozen_artifacts.test_phi,
                frozen_artifacts.test_regime_probs,
                test_returns,
                test_spy,
                config,
                None,
            )
            evaluation = evaluate_frozen_policy(
                frozen_artifacts.policy_checkpoint,
                test_artifacts,
                jax.random.PRNGKey(config.seed + split_index + 10_000),
            )
            portfolio_returns = np.asarray(evaluation.trajectory.step_results.net_return)
            turnovers = np.asarray(evaluation.trajectory.step_results.turnover)
            costs = np.asarray(evaluation.trajectory.step_results.transaction_cost)
            actions = np.asarray(evaluation.trajectory.actions)
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

    benchmark = run_benchmark_suite(split, test_spy, config)
    metrics = calculate_performance_metrics(
        portfolio_returns,
        test_spy,
        turnovers,
        costs,
        config.periods_per_year,
    )
    benchmark_metrics = calculate_performance_metrics(
        test_spy,
        test_spy,
        np.zeros_like(test_spy),
        np.zeros_like(test_spy),
        config.periods_per_year,
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
    regime_frame = pl.DataFrame(
        {
            "decision_date": list(test_dates),
            "split_index": [split_index] * len(test_dates),
            **{
                f"regime_{index}": frozen_artifacts.test_regime_probs[:, index]
                for index in range(frozen_artifacts.test_regime_probs.shape[1])
            },
        }
    )
    spectral_frame = pl.DataFrame(
        {
            "decision_date": list(test_dates),
            "split_index": [split_index] * len(test_dates),
            **{
                column: frozen_artifacts.test_windows.spectral[:, index]
                for index, column in enumerate(
                    frozen_artifacts.test_windows.spectral_feature_columns
                )
            },
        }
    )
    return SplitResult(
        split_index=split_index,
        train_start=split.train_start,
        train_end=split.train_end,
        test_start=split.test_start,
        test_end=split.test_end,
        portfolio_returns=portfolio_frame,
        spy_returns=benchmark,
        allocations=allocation_frame,
        regime_probabilities=regime_frame,
        spectral_features=spectral_frame,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )


def _log_evaluation_metrics(
    artifacts: ExperimentArtifacts,
    result: SplitResult,
    config: ExperimentConfig,
    logger: TensorBoardLogger,
    step: int,
) -> None:
    if artifacts.ppo_training is None and artifacts.production_ppo_training is None:
        return
    if artifacts.train_spy_returns is None:
        raise ValueError("Train SPY returns are required for PPO evaluation logging.")
    if artifacts.production_ppo_training is not None:
        train_returns = np.asarray(
            artifacts.production_ppo_training.rollout.step_results.net_return
        )
        train_turnovers = np.asarray(
            artifacts.production_ppo_training.rollout.step_results.turnover
        )
        train_costs = np.asarray(
            artifacts.production_ppo_training.rollout.step_results.transaction_cost
        )
    elif artifacts.ppo_training is not None:
        train_returns = np.asarray(artifacts.ppo_training.trajectory.step_results.net_return)
        train_turnovers = np.asarray(
            artifacts.ppo_training.trajectory.step_results.turnover
        )
        train_costs = np.asarray(
            artifacts.ppo_training.trajectory.step_results.transaction_cost
        )
    else:
        raise ValueError("PPO training artifacts are required for evaluation logging.")
    train_spy = np.asarray(artifacts.train_spy_returns)
    train_metrics = calculate_performance_metrics(
        train_returns,
        train_spy,
        train_turnovers,
        train_costs,
        config.periods_per_year,
    )
    logger.log_scalars(
        {
            "train_return": train_metrics.cumulative_return,
            "test_return": result.metrics.cumulative_return,
            "train_alpha": train_metrics.spy_relative_alpha,
            "test_alpha": result.metrics.spy_relative_alpha,
            "train_max_drawdown": train_metrics.max_drawdown,
            "test_max_drawdown": result.metrics.max_drawdown,
        },
        step,
        "evaluation",
    )


def run_split(
    split: WalkForwardSplit,
    raw_data: RawExperimentData,
    config: ExperimentConfig,
    split_index: int = 0,
) -> SplitRunResult:
    """Fit train artifacts and evaluate one frozen test split."""

    logger = (
        TensorBoardLogger(
            log_dir=config.ppo.log_dir,
            experiment_name=f"split_{split_index}",
            enabled=config.ppo.enable_tensorboard,
        )
        if config.enable_ppo
        else None
    )
    try:
        frozen_artifacts = fit_train_artifacts(
            split,
            raw_data.features,
            raw_data.returns,
            raw_data.spy_returns,
            config,
            split_index,
            logger,
        )
        result = evaluate_test_split(
            split,
            frozen_artifacts,
            raw_data.returns,
            raw_data.spy_returns,
            config,
            split_index,
        )
        if logger is not None:
            _log_evaluation_metrics(frozen_artifacts, result, config, logger, split_index)
    finally:
        if logger is not None:
            logger.close()
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
    regime_probabilities = pl.concat(
        [result.regime_probabilities for result in results],
        how="vertical",
    )
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
        config.periods_per_year,
    )
    aggregate_benchmark_metrics = calculate_performance_metrics(
        spy_returns["spy_return"].to_numpy(),
        spy_returns["spy_return"].to_numpy(),
        np.zeros(spy_returns.height),
        np.zeros(spy_returns.height),
        config.periods_per_year,
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

    splits = generate_walk_forward_splits(
        raw_data.features.decision_dates,
        config.walk_forward,
    )
    split_results = tuple(
        run_split(split, raw_data, config, split_index).result
        for split_index, split in enumerate(splits)
    )
    return aggregate_walk_forward_results(split_results, config)
