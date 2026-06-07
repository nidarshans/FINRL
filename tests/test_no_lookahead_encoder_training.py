"""No-look-ahead tests for encoder pretraining batch construction."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from finrl.models import EncoderTrainingConfig, make_encoder_batches
from finrl.models.windows import LookbackWindows


def _windows(n_windows: int = 8) -> LookbackWindows:
    asset = np.arange(n_windows * 3 * 2 * 2, dtype=np.float32).reshape(n_windows, 3, 2, 2)
    macro = np.arange(n_windows * 3 * 2, dtype=np.float32).reshape(n_windows, 3, 2)
    spectral = np.arange(n_windows * 20, dtype=np.float32).reshape(n_windows, 20)
    return LookbackWindows(
        asset=asset / 100.0,
        macro=macro / 100.0,
        spectral=spectral / 100.0,
        decision_dates=tuple(range(n_windows)),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return", "volume"),
        macro_feature_columns=("rate", "inflation"),
        spectral_feature_columns=tuple(f"spectral_{index}" for index in range(20)),
    )


def _returns(n_windows: int = 8) -> np.ndarray:
    return np.asarray(
        [[0.01 * index, -0.01 * index] for index in range(n_windows)],
        dtype=np.float32,
    )


def test_test_split_rows_do_not_influence_encoder_training_batches() -> None:
    train_window_count = 5
    training_config = EncoderTrainingConfig(batch_size=3)
    windows = _windows()
    returns = _returns()

    baseline = make_encoder_batches(
        windows,
        returns,
        training_config,
        train_window_count=train_window_count,
    )

    poisoned_windows = LookbackWindows(
        asset=windows.asset.copy(),
        macro=windows.macro.copy(),
        spectral=windows.spectral.copy(),
        decision_dates=windows.decision_dates,
        tickers=windows.tickers,
        asset_feature_columns=windows.asset_feature_columns,
        macro_feature_columns=windows.macro_feature_columns,
        spectral_feature_columns=windows.spectral_feature_columns,
    )
    poisoned_windows.asset[train_window_count:] = 999.0
    poisoned_windows.macro[train_window_count:] = -999.0
    poisoned_windows.spectral[train_window_count:] = 500.0
    poisoned_returns = returns.copy()
    poisoned_returns[train_window_count:] = -500.0

    poisoned = make_encoder_batches(
        poisoned_windows,
        poisoned_returns,
        training_config,
        train_window_count=train_window_count,
    )

    assert len(baseline) == len(poisoned)
    for clean_batch, poisoned_batch in zip(baseline, poisoned):
        assert clean_batch.decision_dates == poisoned_batch.decision_dates
        assert clean_batch.label_dates == poisoned_batch.label_dates
        assert max(clean_batch.label_dates) < train_window_count
        assert_allclose(clean_batch.asset_window, poisoned_batch.asset_window)
        assert_allclose(clean_batch.macro_window, poisoned_batch.macro_window)
        assert_allclose(clean_batch.spectral_row, poisoned_batch.spectral_row)
        assert_allclose(
            clean_batch.market_return_target,
            poisoned_batch.market_return_target,
        )
        assert_allclose(
            clean_batch.cross_sectional_return_target,
            poisoned_batch.cross_sectional_return_target,
        )
