"""Split-safety helpers for offline preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from finrl.features.schema import FeatureBundle


@dataclass(frozen=True, slots=True)
class FitWindow:
    """Date range used to fit preprocessing artifacts."""

    start: date
    end: date


def feature_date_range(features: FeatureBundle) -> FitWindow:
    """Return the inclusive date range covered by a feature bundle."""

    dates = features.asset_features.select(pl.col("date").cast(pl.Date))
    if dates.is_empty():
        raise ValueError("Cannot determine date range for empty feature bundle.")
    return FitWindow(
        start=dates.select(pl.min("date")).item(),
        end=dates.select(pl.max("date")).item(),
    )


def validate_train_test_order(train_features: FeatureBundle, test_features: FeatureBundle) -> None:
    """Reject train/test inputs that overlap or are out of temporal order."""

    train_window = feature_date_range(train_features)
    test_window = feature_date_range(test_features)
    if train_window.end >= test_window.start:
        raise ValueError(
            "Train features must end before test features begin to avoid look-ahead."
        )
