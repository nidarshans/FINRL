"""No-look-ahead lookback window construction for market encoder inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

from finrl.features.schema import FeatureBundle


@dataclass(frozen=True, slots=True)
class LookbackWindows:
    """Encoder windows aligned by decision date."""

    asset: np.ndarray
    macro: np.ndarray
    spectral: np.ndarray
    decision_dates: tuple[object, ...]
    tickers: tuple[str, ...]
    asset_feature_columns: tuple[str, ...]
    macro_feature_columns: tuple[str, ...]
    spectral_feature_columns: tuple[str, ...]


def _ordered_dates(dates: Iterable[object]) -> tuple[object, ...]:
    return tuple(sorted(dates))


def _asset_matrix_for_date(
    table: pl.DataFrame,
    date_value: object,
    tickers: tuple[str, ...],
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    ticker_frame = pl.DataFrame({"ticker": list(tickers)})
    rows = (
        ticker_frame.join(
            table.filter(pl.col("date") == date_value),
            on="ticker",
            how="left",
        )
        .select(feature_columns)
        .fill_null(0.0)
    )
    return rows.to_numpy().astype(np.float32, copy=False)


def _feature_row_for_date(
    table: pl.DataFrame,
    date_value: object,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    rows = table.filter(pl.col("date") == date_value).select(feature_columns).fill_null(0.0)
    if rows.height != 1:
        raise ValueError("Expected exactly one feature row per decision date.")
    return rows.to_numpy()[0].astype(np.float32, copy=False)


def build_lookback_windows(
    features: FeatureBundle,
    lookback: int = 60,
) -> LookbackWindows:
    """Build windows where date ``t`` contains only ``t-lookback+1:t`` rows."""

    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    decision_dates = _ordered_dates(features.decision_dates)
    if len(decision_dates) < lookback:
        raise ValueError("Not enough decision dates to build lookback windows.")
    if len(features.spectral_feature_columns) != 20:
        raise ValueError("Spectral feature dimension must be 20.")

    asset_by_date = {
        day: _asset_matrix_for_date(
            features.asset_features,
            day,
            features.tickers,
            features.asset_feature_columns,
        )
        for day in decision_dates
    }
    macro_by_date = {
        day: _feature_row_for_date(
            features.macro_features,
            day,
            features.macro_feature_columns,
        )
        for day in decision_dates
    }
    spectral_by_date = {
        day: _feature_row_for_date(
            features.spectral_features,
            day,
            features.spectral_feature_columns,
        )
        for day in decision_dates
    }

    asset_windows = []
    macro_windows = []
    spectral_rows = []
    window_dates = []
    for end_index in range(lookback - 1, len(decision_dates)):
        window_dates_slice = decision_dates[end_index - lookback + 1 : end_index + 1]
        asset_windows.append(np.stack([asset_by_date[day] for day in window_dates_slice]))
        macro_windows.append(np.stack([macro_by_date[day] for day in window_dates_slice]))
        spectral_rows.append(spectral_by_date[decision_dates[end_index]])
        window_dates.append(decision_dates[end_index])

    return LookbackWindows(
        asset=np.stack(asset_windows),
        macro=np.stack(macro_windows),
        spectral=np.stack(spectral_rows),
        decision_dates=tuple(window_dates),
        tickers=features.tickers,
        asset_feature_columns=features.asset_feature_columns,
        macro_feature_columns=features.macro_feature_columns,
        spectral_feature_columns=features.spectral_feature_columns,
    )

