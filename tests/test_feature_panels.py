"""Tests for decision-date asset feature panels."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

from numpy.testing import assert_allclose
import polars as pl

from finrl.features.panels import build_asset_feature_panel
from finrl.features.schema import FeatureBundle


def _feature_bundle() -> FeatureBundle:
    dates = tuple(date(2024, 1, 5) + timedelta(days=7 * i) for i in range(4))
    tickers = ("AAA", "BBB")
    asset = pl.DataFrame(
        [
            {"date": day, "ticker": ticker, "value": day_index * 10 + ticker_index}
            for day_index, day in enumerate(dates)
            for ticker_index, ticker in enumerate(tickers)
        ]
    ).with_columns(pl.col("date").cast(pl.Date))
    empty_by_date = pl.DataFrame({"date": dates}).with_columns(pl.col("date").cast(pl.Date))
    return FeatureBundle(
        asset_features=asset,
        macro_features=empty_by_date,
        spectral_features=empty_by_date,
        decision_dates=dates,
        tickers=tickers,
        asset_feature_columns=("value",),
        macro_feature_columns=(),
        spectral_feature_columns=(),
    )


def test_panel_preserves_every_decision_date_without_lookback_truncation() -> None:
    features = _feature_bundle()

    panel = build_asset_feature_panel(features)

    assert panel.values.shape == (4, 2, 1)
    assert panel.decision_dates == features.decision_dates
    assert_allclose(panel.values[:, 0, 0], [0.0, 10.0, 20.0, 30.0])


def test_future_change_does_not_change_prior_panel_rows() -> None:
    features = _feature_bundle()
    base = build_asset_feature_panel(features)
    changed = dataclasses.replace(
        features,
        asset_features=features.asset_features.with_columns(
            pl.when(pl.col("date") == features.decision_dates[-1])
            .then(9_999.0)
            .otherwise(pl.col("value"))
            .alias("value")
        ),
    )

    changed_panel = build_asset_feature_panel(changed)

    assert_allclose(base.values[:-1], changed_panel.values[:-1])
