"""Exact 3M feature routing and warm-up validity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finrl.features.columns import THREE_M_FEATURE_COLUMNS
from finrl.features.panels import AssetFeaturePanel
from finrl.features.schema import FeatureBundle


@dataclass(frozen=True, slots=True)
class ThreeMFeaturePanel:
    """Finite 3M panel plus the pre-imputation feature-validity mask."""

    panel: AssetFeaturePanel
    valid_mask: np.ndarray


def three_m_feature_indices(feature_columns: tuple[str, ...]) -> tuple[int, ...]:
    """Return 3M's exact feature positions or fail before fitting."""

    positions = {name: index for index, name in enumerate(feature_columns)}
    missing = tuple(name for name in THREE_M_FEATURE_COLUMNS if name not in positions)
    if missing:
        raise ValueError("Missing required 3M features: " + ", ".join(missing))
    return tuple(positions[name] for name in THREE_M_FEATURE_COLUMNS)


def build_three_m_feature_panel(features: FeatureBundle) -> ThreeMFeaturePanel:
    """Build a fixed-order panel without treating warm-up nulls as observations."""

    indices = three_m_feature_indices(features.asset_feature_columns)
    selected_columns = tuple(features.asset_feature_columns[index] for index in indices)
    if selected_columns != THREE_M_FEATURE_COLUMNS:
        raise ValueError("3M feature order does not match its allowlist.")
    dates = tuple(sorted(features.decision_dates))
    if not dates:
        raise ValueError("Cannot build a 3M panel without decision dates.")
    ticker_frame = pl.DataFrame({"ticker": list(features.tickers)})
    matrices: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    for day in dates:
        day_rows = features.asset_features.filter(pl.col("date") == day)
        if day_rows.height != len(features.tickers):
            raise ValueError(f"Expected one asset feature row per ticker on {day}.")
        matrix = (
            ticker_frame.join(day_rows, on="ticker", how="left")
            .select(THREE_M_FEATURE_COLUMNS)
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        valid = np.isfinite(matrix).all(axis=1)
        matrices.append(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0))
        valid_rows.append(valid)
    values = np.stack(matrices).astype(np.float32)
    valid_mask = np.stack(valid_rows)
    return ThreeMFeaturePanel(
        panel=AssetFeaturePanel(
            values=values,
            decision_dates=dates,
            tickers=features.tickers,
            feature_columns=THREE_M_FEATURE_COLUMNS,
            tradable_mask=np.ones(valid_mask.shape, dtype=bool),
        ),
        valid_mask=valid_mask,
    )
