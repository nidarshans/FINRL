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
    fip_window: int = 40
    bollinger_window: int = 20
    bollinger_std_multiplier: float = 2.0
    liquidity_window: int = 20
    atr_window: int = 20
    realized_vol_window: int = 20
    downside_vol_window: int = 60
    drawdown_window: int = 126
    swing_left: int = 3
    swing_right: int = 3
    spectral_dim: int = 20
    feature_set: str = "baseline_current_14"
    add_momentum_percentile_ranks: bool = False

    def __post_init__(self) -> None:
        """Reject invalid trailing windows before feature calculation."""

        windows = (
            self.cmf_window,
            self.accumulation_window,
            self.momentum_quality_window,
            self.macd_fast_span,
            self.macd_slow_span,
            self.macd_signal_span,
            self.mr_ewma_span,
            self.mr_vol_window,
            self.klinger_fast_span,
            self.klinger_slow_span,
            self.klinger_signal_span,
            self.fip_window,
            self.bollinger_window,
            self.liquidity_window,
            self.atr_window,
            self.realized_vol_window,
            self.downside_vol_window,
            self.drawdown_window,
            self.swing_left,
            self.swing_right,
            self.spectral_dim,
        )
        if any(window <= 0 for window in windows):
            raise ValueError("Feature windows and dimensions must be positive.")
        if self.bollinger_std_multiplier <= 0.0:
            raise ValueError("bollinger_std_multiplier must be positive.")


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
