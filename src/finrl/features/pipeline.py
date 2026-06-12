"""Feature bundle construction pipeline."""

from __future__ import annotations

import polars as pl

from finrl.data.calendar import build_weekly_rebalance_calendar
from finrl.data.sources import MarketDataBundle
from finrl.features.asset import compute_asset_features
from finrl.features.hawkes import compute_hawkes_features
from finrl.features.macro import compute_macro_features
from finrl.features.relative import add_asset_relative_features
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

    asset_features = add_asset_relative_features(
        compute_asset_features(raw_data.ohlcv, config)
    )
    if config.include_hawkes:
        hawkes = compute_hawkes_features(raw_data.ohlcv)
        asset_features = asset_features.join(hawkes, on=["date", "ticker"], how="left")

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
