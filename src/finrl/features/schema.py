"""Schemas and configuration for offline feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Configuration for trailing, no-look-ahead feature calculations."""

    cmf_window: int = 20
    accumulation_window: int = 40
    momentum_quality_window: int = 40
    macd_fast_span: int = 12
    macd_slow_span: int = 26
    macd_signal_span: int = 9
    mr_ewma_span: int = 50
    mr_vol_window: int = 50
    klinger_fast_span: int = 34
    klinger_slow_span: int = 55
    klinger_signal_span: int = 13
    spectral_dim: int = 20


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
