"""No-look-ahead tests for encoder windowing."""

from __future__ import annotations

import dataclasses

import polars as pl
from numpy.testing import assert_allclose

from finrl.models import build_lookback_windows
from tests.test_encoder_windows import _feature_bundle


def test_future_feature_changes_do_not_change_prior_encoder_windows() -> None:
    features = _feature_bundle(5)
    base = build_lookback_windows(features, lookback=3)
    changed_asset = features.asset_features.with_columns(
        pl.when(pl.col("date") == features.decision_dates[-1])
        .then(9_999.0)
        .otherwise(pl.col("asset_a"))
        .alias("asset_a")
    )
    changed_macro = features.macro_features.with_columns(
        pl.when(pl.col("date") == features.decision_dates[-1])
        .then(9_999.0)
        .otherwise(pl.col("macro_a"))
        .alias("macro_a")
    )
    changed_spectral = features.spectral_features.with_columns(
        pl.when(pl.col("date") == features.decision_dates[-1])
        .then(9_999.0)
        .otherwise(pl.col("spectral_0"))
        .alias("spectral_0")
    )
    changed = dataclasses.replace(
        features,
        asset_features=changed_asset,
        macro_features=changed_macro,
        spectral_features=changed_spectral,
    )

    changed_windows = build_lookback_windows(changed, lookback=3)

    assert_allclose(base.asset[:2], changed_windows.asset[:2], rtol=1e-6, atol=1e-8)
    assert_allclose(base.macro[:2], changed_windows.macro[:2], rtol=1e-6, atol=1e-8)
    assert_allclose(
        base.spectral[:2],
        changed_windows.spectral[:2],
        rtol=1e-6,
        atol=1e-8,
    )

