"""Chronological LightGBM walk-forward policy runner."""

from __future__ import annotations

import numpy as np
import polars as pl

from finrl.features.columns import DirectAllocationRoutingMetadata
from finrl.features.panels import AssetFeaturePanel
from finrl.backtest.walk_forward import generate_walk_forward_splits, slice_feature_bundle
from finrl.experiments.artifacts import RawExperimentData
from finrl.experiments.config import ExperimentConfig
from finrl.features.panels import build_asset_feature_panel
from finrl.features.preprocessing import fit_transform_train_transform_test
from finrl.features.columns import selected_direct_allocation_indices
from finrl.gbt import (
    GBTConfig,
    build_forward_return_targets,
    fit_gbt_model,
    predict_scores,
    scores_to_weights,
)
from finrl.portfolio_construction import smooth_target_weights


def fit_predict_gbt_split(
    train_panel: AssetFeaturePanel,
    test_panel: AssetFeaturePanel,
    train_returns: np.ndarray,
    routing: DirectAllocationRoutingMetadata,
    config: GBTConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a GBT on a purged train window and predict frozen test scores.

    The final ``max(target_horizons) - 1`` training observations are removed
    because their longest forward label is incomplete. The test panel is never
    used for fitting or target construction.
    """

    returns = np.asarray(train_returns, dtype=np.float32)
    if returns.shape[:2] != train_panel.values.shape[:2]:
        raise ValueError("train_returns must match the train panel time/asset shape.")
    targets = build_forward_return_targets(
        returns,
        config.target_horizons,
        config.target_weights,
    )
    train_rows = targets.shape[0]
    if train_rows < 1:
        raise ValueError("Training window has no complete forward-return labels.")
    fitted_panel = AssetFeaturePanel(
        values=train_panel.values[:train_rows],
        decision_dates=train_panel.decision_dates[:train_rows],
        tickers=train_panel.tickers,
        feature_columns=train_panel.feature_columns,
        tradable_mask=(
            None if train_panel.tradable_mask is None else train_panel.tradable_mask[:train_rows]
        ),
    )
    model = fit_gbt_model(fitted_panel, targets, routing, config, seed)
    scores = predict_scores(model, test_panel, routing)
    weights = scores_to_weights(scores, config)
    if config.smoothing_alpha < 1.0:
        initial = np.zeros_like(weights)
        initial[:, :-1] = 1.0 / (weights.shape[1] - 1)
        weights = smooth_target_weights(initial, weights, config.smoothing_alpha)
    return scores, weights


def run_gbt_walk_forward(
    raw_data: RawExperimentData,
    experiment_config: ExperimentConfig,
    gbt_config: GBTConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Fit and evaluate one frozen GBT policy per outer walk-forward split."""

    splits = generate_walk_forward_splits(raw_data.features.decision_dates, experiment_config.walk_forward)
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    routing = selected_direct_allocation_indices(raw_data.features.asset_feature_columns)
    for split_index, split in enumerate(splits):
        train_bundle, test_bundle = slice_feature_bundle(raw_data.features, split)
        processed = fit_transform_train_transform_test(
            train_bundle, test_bundle, experiment_config.preprocessing
        )
        train_panel = build_asset_feature_panel(processed.train)
        test_panel = build_asset_feature_panel(processed.test)
        date_frame = pl.DataFrame({"decision_date": list(train_panel.decision_dates)})
        train_returns = (
            date_frame.join(raw_data.returns, on="decision_date", how="left")
            .select(list(train_panel.tickers))
            .to_numpy()
        )
        outputs.append(
            fit_predict_gbt_split(
                train_panel,
                test_panel,
                train_returns,
                routing,
                gbt_config,
                experiment_config.seed + split_index,
            )
        )
    return tuple(outputs)
