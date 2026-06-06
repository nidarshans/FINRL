"""Lookback window tests for market encoder inputs."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from finrl.features.schema import FeatureBundle
from finrl.models import build_lookback_windows


def _feature_bundle(n_dates: int = 5) -> FeatureBundle:
    dates = tuple(date(2024, 1, 5) + timedelta(days=7 * index) for index in range(n_dates))
    tickers = ("AAA", "BBB")
    asset_rows = []
    for day_index, day in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            asset_rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "asset_a": float(day_index),
                    "asset_b": float(10 * ticker_index + day_index),
                }
            )
    asset = pl.DataFrame(asset_rows).with_columns(pl.col("date").cast(pl.Date))
    macro = pl.DataFrame(
        {
            "date": dates,
            "macro_a": [float(index) for index in range(n_dates)],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    spectral_columns = tuple(f"spectral_{index}" for index in range(20))
    spectral = pl.DataFrame(
        [
            {
                "date": day,
                **{
                    column: float(day_index * 100 + column_index)
                    for column_index, column in enumerate(spectral_columns)
                },
            }
            for day_index, day in enumerate(dates)
        ]
    ).with_columns(pl.col("date").cast(pl.Date))
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=dates,
        tickers=tickers,
        asset_feature_columns=("asset_a", "asset_b"),
        macro_feature_columns=("macro_a",),
        spectral_feature_columns=spectral_columns,
    )


def test_build_lookback_windows_shapes_and_alignment() -> None:
    features = _feature_bundle(5)

    windows = build_lookback_windows(features, lookback=3)

    assert windows.asset.shape == (3, 3, 2, 2)
    assert windows.macro.shape == (3, 3, 1)
    assert windows.spectral.shape == (3, 20)
    assert windows.decision_dates == features.decision_dates[2:]
    assert windows.tickers == ("AAA", "BBB")


def test_window_for_date_t_contains_only_t_minus_l_plus_1_through_t() -> None:
    features = _feature_bundle(5)

    windows = build_lookback_windows(features, lookback=3)

    assert_allclose(windows.macro[1, :, 0], np.array([1.0, 2.0, 3.0]))
    assert_allclose(windows.asset[1, :, 0, 0], np.array([1.0, 2.0, 3.0]))
    assert_allclose(windows.asset[1, :, 1, 1], np.array([11.0, 12.0, 13.0]))
    assert_allclose(windows.spectral[1], np.arange(300.0, 320.0))


def test_build_lookback_windows_rejects_non_20_spectral_features() -> None:
    features = _feature_bundle(5)
    invalid = FeatureBundle(
        asset_features=features.asset_features,
        macro_features=features.macro_features,
        spectral_features=features.spectral_features.drop("spectral_19"),
        decision_dates=features.decision_dates,
        tickers=features.tickers,
        asset_feature_columns=features.asset_feature_columns,
        macro_feature_columns=features.macro_feature_columns,
        spectral_feature_columns=features.spectral_feature_columns[:-1],
    )

    with pytest.raises(ValueError, match="Spectral feature dimension must be 20"):
        build_lookback_windows(invalid, lookback=3)

