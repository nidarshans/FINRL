"""Feature bundle construction pipeline."""

from __future__ import annotations

import polars as pl

from finrl.data.calendar import build_weekly_rebalance_calendar
from finrl.data.sources import MarketDataBundle
from finrl.features.asset import compute_asset_features
from finrl.features.columns import (
    LIQUIDITY_FEATURE_COLUMNS,
    MARKET_RELATIVE_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    feature_set_config,
)
from finrl.features.macro import compute_macro_features
from finrl.features.market_relative import compute_market_relative_features
from finrl.features.relative import cross_sectional_percentile_rank
from finrl.features.schema import FeatureBundle, FeatureConfig


def _decision_dates_from_calendar(calendar: pl.DataFrame) -> pl.Series:
    if "decision_date" in calendar.columns:
        return calendar.select(pl.col("decision_date").cast(pl.Date)).to_series()
    if "date" in calendar.columns:
        if {"ticker", "open", "high", "low", "close", "adj_close", "volume"}.issubset(
            calendar.columns
        ):
            return build_weekly_rebalance_calendar(calendar).select("decision_date").to_series()
        return (
            calendar.select(pl.col("date").cast(pl.Date))
            .unique()
            .sort("date")
            .with_columns(pl.col("date").dt.weekday().alias("weekday"))
            .filter(pl.col("weekday") == 5)
            .drop("weekday")
            .to_series()
        )
    raise ValueError("Calendar must contain either 'decision_date' or 'date'.")


def _filter_to_decision_dates(features: pl.DataFrame, decision_dates: pl.Series) -> pl.DataFrame:
    decision_frame = decision_dates.to_frame("date").with_columns(pl.col("date").cast(pl.Date))
    return features.join(decision_frame, on="date", how="inner").sort("date")


def _dummy_spectral_features(dates: pl.Series, dim: int) -> pl.DataFrame:
    columns = {f"spectral_{index}": [0.0] * len(dates) for index in range(dim)}
    return pl.DataFrame({"date": dates.to_list(), **columns}).with_columns(
        pl.col("date").cast(pl.Date)
    )


def build_feature_bundle(
    raw_data: MarketDataBundle,
    config: FeatureConfig,
) -> FeatureBundle:
    """Build aligned Phase 6 features from raw ingested data."""

    asset_features = compute_asset_features(raw_data.ohlcv, config)
    routed_columns = feature_set_config(config.feature_set).routed_columns
    if any(column in MARKET_RELATIVE_FEATURE_COLUMNS for column in routed_columns):
        market_relative = compute_market_relative_features(
            raw_data.ohlcv, raw_data.spy_ohlcv
        )
        asset_features = asset_features.join(
            market_relative, on=["date", "ticker"], how="left"
        )
    needs_relative_ranks = any(
        column.endswith("_percentile_rank")
        for column in feature_set_config(config.feature_set).routed_columns
    )
    if config.add_momentum_percentile_ranks or needs_relative_ranks:
        for column in (*MOMENTUM_FEATURE_COLUMNS, *LIQUIDITY_FEATURE_COLUMNS):
            asset_features = cross_sectional_percentile_rank(asset_features, column)
    missing_columns = tuple(
        column for column in routed_columns if column not in asset_features.columns
    )
    if missing_columns:
        raise ValueError(
            "Feature set requires unavailable columns: " + ", ".join(missing_columns)
        )
    asset_features = asset_features.select("date", "ticker", *routed_columns)

    macro_features = compute_macro_features(raw_data.macro)
    decision_dates = _decision_dates_from_calendar(raw_data.calendar)
    spectral_features = _dummy_spectral_features(decision_dates, config.spectral_dim)
    asset_features = _filter_to_decision_dates(asset_features, decision_dates)
    spectral_features = _filter_to_decision_dates(spectral_features, decision_dates)
    if "date" in macro_features.columns:
        macro_features = _filter_to_decision_dates(macro_features, decision_dates)

    asset_columns = tuple(
        column for column in asset_features.columns if column not in {"date", "ticker"}
    )
    macro_columns = tuple(column for column in macro_features.columns if column != "date")
    spectral_columns = tuple(column for column in spectral_features.columns if column != "date")
    return FeatureBundle(
        asset_features=asset_features,
        macro_features=macro_features,
        spectral_features=spectral_features,
        decision_dates=tuple(decision_dates.to_list()),
        tickers=tuple(asset_features.get_column("ticker").unique().sort().to_list()),
        asset_feature_columns=asset_columns,
        macro_feature_columns=macro_columns,
        spectral_feature_columns=spectral_columns,
    )
