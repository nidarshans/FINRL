"""Tests for annual walk-forward split generation and slicing."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from finrl.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardSplit,
    generate_walk_forward_splits,
    slice_feature_bundle,
    slice_returns,
    validate_split_boundaries,
)
from finrl.features.schema import FeatureBundle


def _annual_calendar(start_year: int = 2010, end_year: int = 2023) -> pl.DataFrame:
    rows = []
    for year in range(start_year, end_year + 1):
        rows.append(
            {
                "decision_date": date(year, 6, 30),
                "execution_date": date(year, 7, 1),
                "next_execution_date": date(year, 7, 8),
            }
        )
    return pl.DataFrame(rows)


def _feature_bundle() -> FeatureBundle:
    dates = [date(year, 6, 30) for year in range(2010, 2023)]
    asset = pl.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "return": list(range(len(dates))) + list(range(len(dates))),
        }
    )
    macro = pl.DataFrame({"date": dates, "macro_vix_diff": list(range(len(dates)))})
    spectral = pl.DataFrame({"date": dates, "volume_eigen_0": list(range(len(dates)))})
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=tuple(dates),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return",),
        macro_feature_columns=("macro_vix_diff",),
        spectral_feature_columns=("volume_eigen_0",),
    )


def test_generate_walk_forward_splits_matches_documented_examples() -> None:
    splits = generate_walk_forward_splits(
        _annual_calendar(2010, 2023),
        WalkForwardConfig(train_years=10, test_years=1, step_years=1),
    )

    assert len(splits) == 4
    assert (splits[0].train_start, splits[0].train_end, splits[0].test_start, splits[0].test_end) == (
        date(2010, 1, 1),
        date(2019, 12, 31),
        date(2020, 1, 1),
        date(2020, 12, 31),
    )
    assert (splits[1].train_start, splits[1].train_end, splits[1].test_start, splits[1].test_end) == (
        date(2011, 1, 1),
        date(2020, 12, 31),
        date(2021, 1, 1),
        date(2021, 12, 31),
    )
    assert (splits[2].train_start, splits[2].train_end, splits[2].test_start, splits[2].test_end) == (
        date(2012, 1, 1),
        date(2021, 12, 31),
        date(2022, 1, 1),
        date(2022, 12, 31),
    )


def test_generate_expanding_walk_forward_splits_keeps_initial_train_start() -> None:
    splits = generate_walk_forward_splits(
        _annual_calendar(2010, 2023),
        WalkForwardConfig(
            train_years=10,
            test_years=1,
            step_years=1,
            expanding_train_window=True,
        ),
    )

    assert len(splits) == 4
    assert (splits[0].train_start, splits[0].train_end) == (
        date(2010, 1, 1),
        date(2019, 12, 31),
    )
    assert (splits[1].train_start, splits[1].train_end) == (
        date(2010, 1, 1),
        date(2020, 12, 31),
    )


def test_split_carries_decision_and_execution_dates() -> None:
    split = generate_walk_forward_splits(
        _annual_calendar(2010, 2020),
        WalkForwardConfig(train_years=10, test_years=1),
    )[0]

    assert split.train_decision_dates[0] == date(2010, 6, 30)
    assert split.test_decision_dates == (date(2020, 6, 30),)
    assert split.test_execution_dates == (date(2020, 7, 1),)
    assert split.test_next_execution_dates == (date(2020, 7, 8),)


def test_validate_split_boundaries_rejects_overlap() -> None:
    split = WalkForwardSplit(
        train_start=date(2020, 1, 1),
        train_end=date(2020, 12, 31),
        test_start=date(2020, 12, 31),
        test_end=date(2021, 12, 31),
        train_decision_dates=(date(2020, 6, 30),),
        test_decision_dates=(date(2021, 6, 30),),
        train_execution_dates=(),
        test_execution_dates=(),
        train_next_execution_dates=(),
        test_next_execution_dates=(),
    )

    with pytest.raises(ValueError, match="Train window must end before test"):
        validate_split_boundaries(split)


def test_slice_feature_bundle_preserves_metadata_and_dates() -> None:
    split = generate_walk_forward_splits(
        _annual_calendar(2010, 2022),
        WalkForwardConfig(train_years=10, test_years=1),
    )[0]
    train, test = slice_feature_bundle(_feature_bundle(), split)

    assert train.decision_dates[0] == date(2010, 6, 30)
    assert train.decision_dates[-1] == date(2019, 6, 30)
    assert test.decision_dates == (date(2020, 6, 30),)
    assert train.asset_features.get_column("date").max() < test.asset_features.get_column("date").min()
    assert train.tickers == ("AAA", "BBB")
    assert test.asset_feature_columns == ("return",)


def test_slice_returns_slices_portfolio_and_spy_on_same_decision_dates() -> None:
    split = generate_walk_forward_splits(
        _annual_calendar(2010, 2021),
        WalkForwardConfig(train_years=10, test_years=1),
    )[0]
    dates = [date(year, 6, 30) for year in range(2010, 2022)]
    returns = pl.DataFrame(
        {
            "decision_date": dates * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "return": [0.01] * (2 * len(dates)),
        }
    )
    spy = pl.DataFrame({"decision_date": dates, "return": [0.02] * len(dates)})

    sliced = slice_returns(returns, spy, split)

    assert set(sliced.train_returns.get_column("decision_date").unique().to_list()) == set(
        split.train_decision_dates
    )
    assert sliced.test_spy_returns.get_column("decision_date").to_list() == [
        date(2020, 6, 30)
    ]
