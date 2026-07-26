"""Deterministic walk-forward split generation and slicing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import polars as pl

from finrl.features.schema import FeatureBundle


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Configuration for annual rolling walk-forward splits."""

    train_years: int = 10
    test_years: int = 1
    step_years: int = 1
    expanding_train_window: bool = False


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One train/test walk-forward split with explicit timing metadata."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_decision_dates: tuple[date, ...]
    test_decision_dates: tuple[date, ...]
    train_execution_dates: tuple[date, ...]
    test_execution_dates: tuple[date, ...]
    train_next_execution_dates: tuple[date, ...]
    test_next_execution_dates: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class ReturnSplit:
    """Train/test return slices for portfolio and SPY returns."""

    train_returns: pl.DataFrame
    test_returns: pl.DataFrame
    train_spy_returns: pl.DataFrame
    test_spy_returns: pl.DataFrame


def _as_date_series(dates: Iterable[date | str] | pl.Series) -> pl.Series:
    if isinstance(dates, pl.Series):
        return dates.cast(pl.Date).unique().sort()
    return pl.Series("date", list(dates)).cast(pl.Date).unique().sort()


def _calendar_from_input(dates: Iterable[date | str] | pl.Series | pl.DataFrame) -> pl.DataFrame:
    if isinstance(dates, pl.DataFrame):
        if "decision_date" in dates.columns:
            select_exprs = [pl.col("decision_date").cast(pl.Date)]
            if "execution_date" in dates.columns:
                select_exprs.append(pl.col("execution_date").cast(pl.Date))
            if "next_execution_date" in dates.columns:
                select_exprs.append(pl.col("next_execution_date").cast(pl.Date))
            calendar = dates.select(select_exprs).sort("decision_date")
            if "execution_date" not in calendar.columns:
                calendar = calendar.with_columns(
                    pl.col("decision_date").alias("execution_date")
                )
            if "next_execution_date" not in calendar.columns:
                calendar = calendar.with_columns(
                    pl.col("execution_date").shift(-1).alias("next_execution_date")
                )
            return calendar.drop_nulls(
                ["decision_date", "execution_date", "next_execution_date"]
            ).sort("decision_date")
        if "date" not in dates.columns:
            raise ValueError("Date DataFrame must contain 'date' or 'decision_date'.")
        series = dates.select(pl.col("date").cast(pl.Date)).to_series()
    else:
        series = _as_date_series(dates)
    return pl.DataFrame(
        {
            "decision_date": series,
            "execution_date": series,
            "next_execution_date": series.shift(-1),
        }
    ).drop_nulls("next_execution_date")


def _year_start(year: int) -> date:
    return date(year, 1, 1)


def _year_end(year: int) -> date:
    return date(year, 12, 31)


def _date_tuple(frame: pl.DataFrame, column: str) -> tuple[date, ...]:
    if column not in frame.columns:
        return ()
    return tuple(frame.get_column(column).drop_nulls().to_list())


def validate_split_boundaries(split: WalkForwardSplit) -> None:
    """Validate temporal ordering and non-overlap for one split."""

    if split.train_start > split.train_end:
        raise ValueError("Train start must be on or before train end.")
    if split.test_start > split.test_end:
        raise ValueError("Test start must be on or before test end.")
    if split.train_end >= split.test_start:
        raise ValueError("Train window must end before test window starts.")
    if set(split.train_decision_dates).intersection(split.test_decision_dates):
        raise ValueError("Train and test decision dates must not overlap.")
    if any(day > split.train_end for day in split.train_decision_dates):
        raise ValueError("Train decision dates exceed train window.")
    if any(day < split.test_start for day in split.test_decision_dates):
        raise ValueError("Test decision dates precede test window.")


def generate_walk_forward_splits(
    dates: Iterable[date | str] | pl.Series | pl.DataFrame,
    config: WalkForwardConfig,
) -> tuple[WalkForwardSplit, ...]:
    """Generate annual rolling or expanding train/test splits.

    In expanding mode, the first train date remains fixed while the train end
    advances with each test period.  This supports DPO's cumulative retraining
    schedule without changing the default rolling behavior used elsewhere.
    """

    if config.train_years <= 0 or config.test_years <= 0 or config.step_years <= 0:
        raise ValueError("train_years, test_years, and step_years must be positive.")

    calendar = _calendar_from_input(dates)
    if calendar.is_empty():
        return ()
    first_year = calendar.select(pl.min("decision_date")).item().year
    last_year = calendar.select(pl.max("decision_date")).item().year
    splits: list[WalkForwardSplit] = []
    split_index = 0
    while True:
        train_start_year = (
            first_year
            if config.expanding_train_window
            else first_year + split_index * config.step_years
        )
        train_end_year = first_year + config.train_years - 1 + split_index * config.step_years
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + config.test_years - 1
        if test_end_year > last_year:
            break

        train_start = _year_start(train_start_year)
        train_end = _year_end(train_end_year)
        test_start = _year_start(test_start_year)
        test_end = _year_end(test_end_year)
        train_calendar = calendar.filter(
            (pl.col("decision_date") >= train_start)
            & (pl.col("decision_date") <= train_end)
        )
        test_calendar = calendar.filter(
            (pl.col("decision_date") >= test_start)
            & (pl.col("decision_date") <= test_end)
        )
        if not train_calendar.is_empty() and not test_calendar.is_empty():
            split = WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_decision_dates=_date_tuple(train_calendar, "decision_date"),
                test_decision_dates=_date_tuple(test_calendar, "decision_date"),
                train_execution_dates=_date_tuple(train_calendar, "execution_date"),
                test_execution_dates=_date_tuple(test_calendar, "execution_date"),
                train_next_execution_dates=_date_tuple(
                    train_calendar, "next_execution_date"
                ),
                test_next_execution_dates=_date_tuple(test_calendar, "next_execution_date"),
            )
            validate_split_boundaries(split)
            splits.append(split)
        split_index += 1
    return tuple(splits)


def _filter_by_dates(frame: pl.DataFrame, date_column: str, dates: tuple[date, ...]) -> pl.DataFrame:
    if date_column not in frame.columns:
        raise ValueError(f"Frame must contain '{date_column}'.")
    date_frame = pl.DataFrame({date_column: list(dates)}).with_columns(
        pl.col(date_column).cast(pl.Date)
    )
    return frame.join(date_frame, on=date_column, how="inner").sort(date_column)


def slice_feature_bundle(
    features: FeatureBundle,
    split: WalkForwardSplit,
) -> tuple[FeatureBundle, FeatureBundle]:
    """Slice a feature bundle into train and test bundles by decision date."""

    train_asset = _filter_by_dates(features.asset_features, "date", split.train_decision_dates)
    test_asset = _filter_by_dates(features.asset_features, "date", split.test_decision_dates)
    train_macro = _filter_by_dates(features.macro_features, "date", split.train_decision_dates)
    test_macro = _filter_by_dates(features.macro_features, "date", split.test_decision_dates)
    train_spectral = _filter_by_dates(
        features.spectral_features, "date", split.train_decision_dates
    )
    test_spectral = _filter_by_dates(
        features.spectral_features, "date", split.test_decision_dates
    )
    train = FeatureBundle(
        asset_features=train_asset,
        macro_features=train_macro,
        spectral_features=train_spectral,
        decision_dates=split.train_decision_dates,
        tickers=features.tickers,
        asset_feature_columns=features.asset_feature_columns,
        macro_feature_columns=features.macro_feature_columns,
        spectral_feature_columns=features.spectral_feature_columns,
    )
    test = FeatureBundle(
        asset_features=test_asset,
        macro_features=test_macro,
        spectral_features=test_spectral,
        decision_dates=split.test_decision_dates,
        tickers=features.tickers,
        asset_feature_columns=features.asset_feature_columns,
        macro_feature_columns=features.macro_feature_columns,
        spectral_feature_columns=features.spectral_feature_columns,
    )
    return train, test


def slice_returns(
    returns: pl.DataFrame,
    spy_returns: pl.DataFrame,
    split: WalkForwardSplit,
) -> ReturnSplit:
    """Slice portfolio and SPY return tables by split decision dates."""

    return ReturnSplit(
        train_returns=_filter_by_dates(
            returns, "decision_date", split.train_decision_dates
        ),
        test_returns=_filter_by_dates(returns, "decision_date", split.test_decision_dates),
        train_spy_returns=_filter_by_dates(
            spy_returns, "decision_date", split.train_decision_dates
        ),
        test_spy_returns=_filter_by_dates(
            spy_returns, "decision_date", split.test_decision_dates
        ),
    )
