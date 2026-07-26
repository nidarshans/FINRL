"""Deterministic coverage for the 3M pooled GBT policy."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from finrl.features.columns import THREE_M_FEATURE_COLUMNS
from finrl.features.panels import AssetFeaturePanel
from finrl.features.schema import FeatureBundle
from finrl.three_m import (
    Action,
    LabelConfig,
    LabelInputs,
    PolicyConfig,
    ThreeMTargets,
    TreeConfig,
    allocate_actions,
    build_targets,
    build_training_data,
    decide_actions,
    fit_model,
    fit_predict_split,
    predict_probabilities,
)
from finrl.three_m.features import build_three_m_feature_panel
from finrl.three_m.model import ThreeMProbabilities


def _label_inputs(
    close: np.ndarray,
    ema20_gap: np.ndarray,
    ema50_gap: np.ndarray | None = None,
    vwap_gap: np.ndarray | None = None,
) -> LabelInputs:
    return LabelInputs(
        close=close,
        close_ema20_gap=ema20_gap,
        close_ema50_gap=ema20_gap if ema50_gap is None else ema50_gap,
        close_vwap20_gap=ema20_gap if vwap_gap is None else vwap_gap,
    )


def test_event_targets_use_crossovers_and_forward_outcomes() -> None:
    returns = np.array(
        [[0.00, 0.00], [0.03, -0.06], [0.03, 0.00], [0.00, 0.00]],
        dtype=np.float32,
    )
    close = np.full_like(returns, 100.0)
    ema20_gap = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    targets = build_targets(
        returns,
        _label_inputs(close, ema20_gap, ema50_gap=np.array([[-2.0, 0.0], [2.0, -0.5], [2.0, -2.0], [2.0, -2.0]])),
        LabelConfig(outcome_horizon=2, buy_min_return=0.05, sell_min_drawdown=0.05, ema50_epsilon=0.01),
    )

    assert targets.n_times == 3
    assert targets.buy[1].tolist() == [True, False]
    assert targets.sell[1].tolist() == [False, True]
    assert targets.hold[1].tolist() == [True, True]
    assert not targets.valid_mask[0].any()


def test_targets_reject_insufficient_history() -> None:
    with pytest.raises(ValueError, match="outcome_horizon"):
        build_targets(
            np.zeros((2, 1), dtype=np.float32),
            _label_inputs(np.ones((2, 1)), np.zeros((2, 1))),
            LabelConfig(outcome_horizon=3),
        )


def test_hold_target_allows_configured_ema50_mean_reversion() -> None:
    targets = build_targets(
        np.zeros((3, 1), dtype=np.float32),
        _label_inputs(
            np.full((3, 1), 100.0),
            np.full((3, 1), -2.0),
            ema50_gap=np.array([[-2.0], [-0.5], [-2.0]]),
            vwap_gap=np.full((3, 1), -2.0),
        ),
        LabelConfig(outcome_horizon=1, ema50_epsilon=0.01),
    )

    assert targets.hold[:, 0].tolist() == [False, True, False]


def _panel() -> AssetFeaturePanel:
    values = np.arange(6 * 2 * 2, dtype=np.float32).reshape(6, 2, 2) / 10.0
    return AssetFeaturePanel(values, tuple(range(6)), ("AAA", "BBB"), ("f1", "f2"))


def _targets() -> ThreeMTargets:
    buy = np.array([[True, False], [False, True], [True, False], [False, True]])
    hold = np.array([[True, True], [False, False], [True, False], [False, True]])
    sell = ~buy
    return ThreeMTargets(buy=buy, hold=hold, sell=sell, n_times=4)


def test_three_classifiers_fit_and_predict_deterministically() -> None:
    panel = _panel()
    data = build_training_data(panel, _targets())
    config = TreeConfig(max_iter=5, min_samples_leaf=1, max_leaf_nodes=3)
    first = fit_model(data, panel.feature_columns, config, seed=7)
    second = fit_model(data, panel.feature_columns, config, seed=7)

    first_probabilities = predict_probabilities(first, panel)
    second_probabilities = predict_probabilities(second, panel)

    assert first_probabilities.buy.shape == (6, 2)
    assert_allclose(first_probabilities.buy, second_probabilities.buy, atol=0.0)
    assert_allclose(first_probabilities.hold, second_probabilities.hold, atol=0.0)
    assert_allclose(first_probabilities.sell, second_probabilities.sell, atol=0.0)


def test_invalid_feature_warmup_rows_are_excluded_from_training() -> None:
    panel = _panel()
    data = build_training_data(
        panel,
        _targets(),
        feature_valid_mask=np.array(
            [[False, True], [True, True], [True, True], [True, True], [True, True], [True, True]]
        ),
    )

    assert data.features.shape[0] == 7
    assert not data.valid_mask[0, 0]


def test_state_gate_and_allocation_preserve_hold_and_rank_buys() -> None:
    config = PolicyConfig(
        buy_threshold=0.6,
        hold_threshold=0.5,
        sell_threshold=0.7,
        entry_weight=0.3,
        max_positions=2,
        max_position_weight=0.3,
    )
    probabilities = ThreeMProbabilities(
        buy=np.array([0.1, 0.9, 0.8], dtype=np.float32),
        hold=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        sell=np.array([0.1, 0.1, 0.1], dtype=np.float32),
    )
    current = np.array([0.4, 0.0, 0.0, 0.6], dtype=np.float32)
    decision = decide_actions(current, probabilities, config)
    result = allocate_actions(current, decision, probabilities.buy, config)

    assert decision.actions.tolist() == [Action.HOLD, Action.BUY, Action.BUY]
    assert_allclose(result.target_weights, [0.4, 0.3, 0.0, 0.3])
    assert result.admitted_buy_mask.tolist() == [False, True, False]
    assert result.rejected_buy_mask.tolist() == [False, False, True]


def test_state_gate_sells_held_asset_with_weak_hold_signal() -> None:
    config = PolicyConfig(hold_threshold=0.5)
    decision = decide_actions(
        np.array([0.4, 0.6]),
        ThreeMProbabilities(
            buy=np.array([0.9], dtype=np.float32),
            hold=np.array([0.4], dtype=np.float32),
            sell=np.array([0.1], dtype=np.float32),
        ),
        config,
    )

    assert decision.actions.tolist() == [Action.SELL]


def test_nontradable_held_asset_remains_held() -> None:
    decision = decide_actions(
        np.array([0.4, 0.6]),
        ThreeMProbabilities(
            buy=np.array([0.9], dtype=np.float32),
            hold=np.array([0.0], dtype=np.float32),
            sell=np.array([1.0], dtype=np.float32),
        ),
        PolicyConfig(),
        tradable_mask=np.array([False]),
    )

    assert decision.actions.tolist() == [Action.HOLD]


def test_split_runner_freezes_model_and_evolves_test_state_causally() -> None:
    train_panel = _panel()
    test_panel = AssetFeaturePanel(
        values=train_panel.values[:2],
        decision_dates=(10, 11),
        tickers=train_panel.tickers,
        feature_columns=train_panel.feature_columns,
    )
    output = fit_predict_split(
        train_panel=train_panel,
        test_panel=test_panel,
        train_execution_returns=np.array(
            [[0.0, 0.0], [0.02, -0.02], [0.01, -0.01], [0.0, 0.0], [0.01, -0.01], [0.0, 0.0]],
            dtype=np.float32,
        ),
        train_label_inputs=_label_inputs(
            np.full((6, 2), 100.0, dtype=np.float32),
            np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]),
            ema50_gap=np.array([[-2.0, 2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0]]),
            vwap_gap=np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]),
        ),
        test_execution_returns=np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float32),
        tree_config=TreeConfig(max_iter=5, min_samples_leaf=1, max_leaf_nodes=3),
        label_config=LabelConfig(outcome_horizon=1, buy_min_return=0.01, sell_min_drawdown=0.01),
        policy_config=PolicyConfig(entry_weight=0.2, max_position_weight=0.2, max_positions=2),
        seed=11,
    )

    assert output.target_weights.shape == (2, 3)
    assert output.actions.shape == (2, 2)
    assert_allclose(output.target_weights.sum(axis=1), [1.0, 1.0])


def test_three_m_panel_masks_null_warmup_values() -> None:
    rows: list[dict[str, object]] = []
    for date in ("2024-01-02", "2024-01-03"):
        for ticker in ("AAA", "BBB"):
            rows.append({"date": date, "ticker": ticker, **{column: 1.0 for column in THREE_M_FEATURE_COLUMNS}})
    rows[0]["beta_252"] = None
    asset = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
    bundle = FeatureBundle(
        asset_features=asset,
        macro_features=pl.DataFrame({"date": []}, schema={"date": pl.Date}),
        spectral_features=pl.DataFrame({"date": []}, schema={"date": pl.Date}),
        decision_dates=tuple(asset.get_column("date").unique().sort().to_list()),
        tickers=("AAA", "BBB"),
        asset_feature_columns=THREE_M_FEATURE_COLUMNS,
        macro_feature_columns=(),
        spectral_feature_columns=(),
    )

    output = build_three_m_feature_panel(bundle)

    assert not output.valid_mask[0, 0]
    assert output.panel.values[0, 0, THREE_M_FEATURE_COLUMNS.index("beta_252")] == 0.0
