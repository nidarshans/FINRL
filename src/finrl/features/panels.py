"""Decision-date asset feature panels for learned allocation policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finrl.features.schema import FeatureBundle


@dataclass(frozen=True, slots=True)
class AssetFeaturePanel:
    """Asset features aligned as ``[time, asset, feature]``."""

    values: np.ndarray
    decision_dates: tuple[object, ...]
    tickers: tuple[str, ...]
    feature_columns: tuple[str, ...]
    tradable_mask: np.ndarray | None = None


def build_asset_feature_panel(features: FeatureBundle) -> AssetFeaturePanel:
    """Build one cross-sectional asset matrix per decision date.

    Feature calculations are already trailing and split-safe. This conversion
    adds no temporal window and therefore cannot mix future rows into a
    decision-date input.
    """

    dates = tuple(sorted(features.decision_dates))
    ticker_frame = pl.DataFrame({"ticker": list(features.tickers)})
    matrices: list[np.ndarray] = []
    for day in dates:
        day_rows = features.asset_features.filter(pl.col("date") == day)
        if (
            day_rows.height != len(features.tickers)
            or day_rows.get_column("ticker").n_unique() != len(features.tickers)
        ):
            raise ValueError(f"Expected one asset feature row per ticker on {day}.")
        matrix = (
            ticker_frame.join(day_rows, on="ticker", how="left")
            .select(features.asset_feature_columns)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Asset feature panel contains non-finite values on {day}.")
        matrices.append(matrix)
    if not matrices:
        raise ValueError("Cannot build an asset feature panel without decision dates.")
    return AssetFeaturePanel(
        values=np.stack(matrices),
        decision_dates=dates,
        tickers=features.tickers,
        feature_columns=features.asset_feature_columns,
        tradable_mask=np.ones((len(dates), len(features.tickers)), dtype=bool),
    )
