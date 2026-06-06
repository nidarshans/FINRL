"""Schemas and configuration for offline feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Configuration for trailing, no-look-ahead feature calculations."""

    return_window: int = 1
    rsi_window: int = 14
    trend_window: int = 20
    liquidity_window: int = 20
    volume_window: int = 20
    spectral_window: int = 20
    spectral_dim: int = 20
    include_hawkes: bool = False


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Feature outputs with explicit date, ticker, and column metadata."""

    asset_features: pl.DataFrame
    macro_features: pl.DataFrame
    spectral_features: pl.DataFrame
    decision_dates: tuple[object, ...]
    tickers: tuple[str, ...]
    asset_feature_columns: tuple[str, ...]
    macro_feature_columns: tuple[str, ...]
    spectral_feature_columns: tuple[str, ...]
