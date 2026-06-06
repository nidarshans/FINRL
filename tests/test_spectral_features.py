"""Tests for trailing spectral feature engineering."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from finrl.features.spectral import (
    compute_liquidity_eigenspectrum,
    compute_sector_flow_indicators,
    compute_spectral_features,
    compute_volume_eigenspectrum,
)

RTOL = 1e-6
ATOL = 1e-8


def _spectral_input() -> pl.DataFrame:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    return pl.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "return": [0.01, 0.02, 0.03, 0.04, -0.01, -0.02, -0.03, -0.04],
            "dollar_volume": [100.0, 110.0, 120.0, 130.0, 200.0, 210.0, 220.0, 230.0],
            "amihud_illiquidity": [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def test_volume_eigenspectrum_uses_trailing_window_only() -> None:
    base = _spectral_input()
    changed_future = base.with_columns(
        pl.when(pl.col("date") == date(2024, 1, 4))
        .then(99999.0)
        .otherwise(pl.col("dollar_volume"))
        .alias("dollar_volume")
    )

    base_row = compute_volume_eigenspectrum(base, lookback=2, n_components=2).filter(
        pl.col("date") == date(2024, 1, 3)
    )
    changed_row = compute_volume_eigenspectrum(changed_future, lookback=2, n_components=2).filter(
        pl.col("date") == date(2024, 1, 3)
    )

    assert_allclose(
        base_row.select(["volume_eigen_0", "volume_eigen_1"]).to_numpy(),
        changed_row.select(["volume_eigen_0", "volume_eigen_1"]).to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_spectral_components_and_flow_indicators_have_expected_columns() -> None:
    data = _spectral_input()

    volume = compute_volume_eigenspectrum(data, lookback=2, n_components=2)
    liquidity = compute_liquidity_eigenspectrum(data, lookback=2, n_components=2)
    flows = compute_sector_flow_indicators(data)
    spectral = compute_spectral_features(data, lookback=2, spectral_dim=20)

    assert {"volume_eigen_0", "volume_eigen_1"}.issubset(volume.columns)
    assert {"liquidity_eigen_0", "liquidity_eigen_1"}.issubset(liquidity.columns)
    assert "sector_flow_return_mean" in flows.columns
    assert len([column for column in spectral.columns if column != "date"]) == 20


def test_eigenspectrum_treats_non_finite_inputs_as_missing() -> None:
    data = _spectral_input().with_columns(
        pl.when((pl.col("date") == date(2024, 1, 3)) & (pl.col("ticker") == "AAA"))
        .then(float("inf"))
        .when((pl.col("date") == date(2024, 1, 4)) & (pl.col("ticker") == "BBB"))
        .then(float("nan"))
        .otherwise(pl.col("amihud_illiquidity"))
        .alias("amihud_illiquidity")
    )

    liquidity = compute_liquidity_eigenspectrum(data, lookback=3, n_components=2)

    values = liquidity.select(["liquidity_eigen_0", "liquidity_eigen_1"]).to_numpy()
    assert np.isfinite(values).all()
