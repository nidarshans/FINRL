from __future__ import annotations

import numpy as np

from finrl.features.columns import DIRECT_ALLOCATION_FEATURE_COLUMNS, selected_direct_allocation_indices
from finrl.features.panels import AssetFeaturePanel
from finrl.gbt import GBTConfig
from finrl.experiments.gbt_runner import fit_predict_gbt_split


def test_gbt_split_purges_incomplete_forward_labels() -> None:
    columns = ("unused", *DIRECT_ALLOCATION_FEATURE_COLUMNS)
    values = np.arange(8 * 2 * len(columns), dtype=np.float32).reshape(8, 2, len(columns)) / 100.0
    train = AssetFeaturePanel(values, tuple(range(8)), ("AAA", "BBB"), columns)
    test = AssetFeaturePanel(values[:2], (8, 9), ("AAA", "BBB"), columns)
    routing = selected_direct_allocation_indices(columns)
    returns = np.array([[0.01, -0.01]] * 8, dtype=np.float32)

    scores, weights = fit_predict_gbt_split(
        train, test, returns, routing,
        GBTConfig(n_estimators=5, min_child_samples=2, target_horizons=(1, 2, 3)), seed=3
    )

    assert scores.shape == (2, 2)
    assert weights.shape == (2, 3)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
