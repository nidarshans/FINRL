"""No-look-ahead tests for walk-forward split boundaries."""

from __future__ import annotations

from datetime import date

import polars as pl

from finrl.backtest.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
    slice_feature_bundle,
    slice_returns,
)
from finrl.features.preprocessing import PreprocessingConfig, fit_preprocessors
from finrl.features.schema import FeatureBundle


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_date": [date(year, 6, 30) for year in range(2010, 2022)],
            "execution_date": [date(year, 7, 1) for year in range(2010, 2022)],
            "next_execution_date": [date(year, 7, 8) for year in range(2010, 2022)],
        }
    )


def _features_with_test_outlier() -> FeatureBundle:
    dates = [date(year, 6, 30) for year in range(2010, 2022)]
    values = [float(index) for index in range(len(dates))]
    values[10] = 999_999.0
    asset = pl.DataFrame(
        {"date": dates, "ticker": ["AAA"] * len(dates), "return": values}
    )
    macro = pl.DataFrame({"date": dates, "macro_vix_diff": values})
    spectral = pl.DataFrame({"date": dates, "volume_eigen_0": values})
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=tuple(dates),
        tickers=("AAA",),
        asset_feature_columns=("return",),
        macro_feature_columns=("macro_vix_diff",),
        spectral_feature_columns=("volume_eigen_0",),
    )


def test_train_and_test_decision_dates_do_not_overlap() -> None:
    split = generate_walk_forward_splits(
        _calendar(), WalkForwardConfig(train_years=10, test_years=1)
    )[0]

    assert split.train_end < split.test_start
    assert not set(split.train_decision_dates).intersection(split.test_decision_dates)
    assert max(split.train_decision_dates) < min(split.test_decision_dates)


def test_preprocessing_fit_receives_train_slice_only() -> None:
    split = generate_walk_forward_splits(
        _calendar(), WalkForwardConfig(train_years=10, test_years=1)
    )[0]
    train_features, test_features = slice_feature_bundle(_features_with_test_outlier(), split)

    fitted = fit_preprocessors(train_features, PreprocessingConfig(rolling_window=3))

    assert fitted.fit_window.end == date(2019, 6, 30)
    assert test_features.asset_features.get_column("return").to_list() == [999_999.0]


def test_return_slices_keep_policy_test_period_frozen_from_train_period() -> None:
    split = generate_walk_forward_splits(
        _calendar(), WalkForwardConfig(train_years=10, test_years=1)
    )[0]
    returns = pl.DataFrame(
        {
            "decision_date": [date(year, 6, 30) for year in range(2010, 2022)],
            "return": [float(year) for year in range(2010, 2022)],
        }
    )
    spy = returns.clone()

    sliced = slice_returns(returns, spy, split)

    assert sliced.train_returns.get_column("decision_date").max() == date(2019, 6, 30)
    assert sliced.test_returns.get_column("decision_date").min() == date(2020, 6, 30)
    assert sliced.train_spy_returns.get_column("decision_date").to_list() == (
        sliced.train_returns.get_column("decision_date").to_list()
    )
    assert sliced.test_spy_returns.get_column("decision_date").to_list() == (
        sliced.test_returns.get_column("decision_date").to_list()
    )
