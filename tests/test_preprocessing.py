"""Tests for chronological rolling preprocessing."""

from __future__ import annotations

from datetime import date

import polars as pl
from numpy.testing import assert_allclose

from finrl.features.preprocessing import (
    PreprocessingConfig,
    fit_preprocessors,
    fit_transform_train_transform_test,
    transform_features,
)
from finrl.features.schema import FeatureBundle

RTOL = 1e-6
ATOL = 1e-8


def _bundle(
    dates: tuple[str, ...],
    asset_values: tuple[float, ...],
    macro_values: tuple[float, ...],
    spectral_values: tuple[float, ...],
) -> FeatureBundle:
    asset_rows = []
    for day, value in zip(dates, asset_values, strict=True):
        asset_rows.append(
            {
                "date": day,
                "ticker": "AAA",
                "return": value,
                "return_percentile_rank": 0.25,
            }
        )
        asset_rows.append(
            {
                "date": day,
                "ticker": "BBB",
                "return": value + 10.0,
                "return_percentile_rank": 0.75,
            }
        )
    asset = pl.DataFrame(asset_rows).with_columns(pl.col("date").cast(pl.Date))
    macro = pl.DataFrame(
        {
            "date": dates,
            "macro_vix_diff": macro_values,
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    spectral = pl.DataFrame(
        {
            "date": dates,
            "volume_eigen_0": spectral_values,
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=tuple(asset.select("date").unique().sort("date").to_series().to_list()),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return", "return_percentile_rank"),
        macro_feature_columns=("macro_vix_diff",),
        spectral_feature_columns=("volume_eigen_0",),
    )


def test_fit_preprocessors_records_train_window_and_rolling_metadata() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )

    fitted = fit_preprocessors(train, PreprocessingConfig(rolling_window=2))

    assert fitted.fit_window.start == date(2024, 1, 5)
    assert fitted.fit_window.end == date(2024, 1, 12)
    assert fitted.asset.group_columns == ("ticker",)
    assert fitted.asset.rolling_window == 2


def test_asset_features_are_standardized_per_ticker_chronologically() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12", "2024-01-19"),
        (1.0, 3.0, 5.0),
        (10.0, 12.0, 14.0),
        (100.0, 110.0, 120.0),
    )
    test = _bundle(("2024-01-26",), (7.0,), (16.0,), (130.0,))

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )
    aaa_values = (
        split.train.asset_features.filter(pl.col("ticker") == "AAA")
        .sort("date")
        .get_column("return")
        .to_list()
    )
    bbb_values = (
        split.train.asset_features.filter(pl.col("ticker") == "BBB")
        .sort("date")
        .get_column("return")
        .to_list()
    )

    assert_allclose(aaa_values, [0.0, 0.707107, 0.707107], rtol=RTOL, atol=ATOL)
    assert_allclose(bbb_values, [0.0, 0.707107, 0.707107], rtol=RTOL, atol=ATOL)
    assert_allclose(
        split.test.asset_features.filter(pl.col("ticker") == "AAA")["return"][0],
        0.707107,
        rtol=RTOL,
        atol=ATOL,
    )


def test_macro_and_spectral_features_roll_over_time() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12", "2024-01-19"),
        (1.0, 3.0, 5.0),
        (10.0, 12.0, 14.0),
        (100.0, 110.0, 120.0),
    )
    test = _bundle(("2024-01-26",), (7.0,), (16.0,), (130.0,))

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )

    assert_allclose(
        split.train.macro_features.get_column("macro_vix_diff").to_list(),
        [0.0, 0.707107, 0.707107],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        split.train.spectral_features.get_column("volume_eigen_0").to_list(),
        [0.0, 0.707107, 0.707107],
        rtol=RTOL,
        atol=ATOL,
    )


def test_future_test_values_do_not_change_prior_train_or_test_rows() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12", "2024-01-19"),
        (1.0, 3.0, 5.0),
        (10.0, 12.0, 14.0),
        (100.0, 110.0, 120.0),
    )
    test = _bundle(
        ("2024-01-26", "2024-02-02"),
        (7.0, 9.0),
        (16.0, 18.0),
        (130.0, 140.0),
    )
    changed_future = _bundle(
        ("2024-01-26", "2024-02-02"),
        (7.0, 9_999.0),
        (16.0, 99_999.0),
        (130.0, 999_999.0),
    )

    base = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )
    changed = fit_transform_train_transform_test(
        train,
        changed_future,
        PreprocessingConfig(rolling_window=2),
    )

    assert_allclose(
        base.train.asset_features.get_column("return").to_list(),
        changed.train.asset_features.get_column("return").to_list(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        base.test.asset_features.filter(pl.col("date") == date(2024, 1, 26))[
            "return"
        ].to_list(),
        changed.test.asset_features.filter(pl.col("date") == date(2024, 1, 26))[
            "return"
        ].to_list(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_shapes_and_identifiers_are_preserved() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = _bundle(("2024-01-19",), (5.0,), (14.0,), (120.0,))

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )

    assert split.train.asset_features.shape == train.asset_features.shape
    assert split.test.asset_features.shape == test.asset_features.shape
    assert split.test.asset_features.select(["date", "ticker"]).to_dicts() == (
        test.asset_features.select(["date", "ticker"]).to_dicts()
    )
    assert split.test.macro_features.get_column("date").to_list() == [
        date(2024, 1, 19)
    ]


def test_rank_columns_are_not_standardized() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = _bundle(("2024-01-19",), (5.0,), (14.0,), (120.0,))

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )

    assert split.train.asset_features.get_column("return_percentile_rank").to_list() == [
        0.25,
        0.25,
        0.75,
        0.75,
    ]
    assert split.test.asset_features.get_column("return_percentile_rank").to_list() == [
        0.25,
        0.75,
    ]


def test_clipping_runs_before_rolling_standardization() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = _bundle(("2024-01-19",), (100.0,), (100.0,), (1_000.0,))

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2, clip_lower=-10.0, clip_upper=10.0),
    )

    assert split.test.asset_features.get_column("return").max() < 10.0


def test_missing_values_are_filled_without_future_rows() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = FeatureBundle(
        asset_features=pl.DataFrame(
            {
                "date": ["2024-01-19", "2024-01-19"],
                "ticker": ["AAA", "BBB"],
                "return": [None, 999.0],
                "return_percentile_rank": [0.25, 0.75],
            }
        ).with_columns(pl.col("date").cast(pl.Date)),
        macro_features=pl.DataFrame(
            {"date": ["2024-01-19"], "macro_vix_diff": [None]}
        ).with_columns(pl.col("date").cast(pl.Date)),
        spectral_features=pl.DataFrame(
            {"date": ["2024-01-19"], "volume_eigen_0": [None]}
        ).with_columns(pl.col("date").cast(pl.Date)),
        decision_dates=(date(2024, 1, 19),),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return", "return_percentile_rank"),
        macro_feature_columns=("macro_vix_diff",),
        spectral_feature_columns=("volume_eigen_0",),
    )

    split = fit_transform_train_transform_test(
        train,
        test,
        PreprocessingConfig(rolling_window=2),
    )

    assert split.test.asset_features.select(pl.col("return").is_null().sum()).item() == 0
    assert split.test.macro_features.select(pl.col("macro_vix_diff").is_null().sum()).item() == 0
    assert split.test.spectral_features.select(pl.col("volume_eigen_0").is_null().sum()).item() == 0


def test_transform_features_applies_rolling_within_supplied_history() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12", "2024-01-19"),
        (1.0, 3.0, 5.0),
        (10.0, 12.0, 14.0),
        (100.0, 110.0, 120.0),
    )
    fitted = fit_preprocessors(train, PreprocessingConfig(rolling_window=2))

    transformed = transform_features(train, fitted)

    assert_allclose(
        transformed.macro_features.get_column("macro_vix_diff").to_list(),
        [0.0, 0.707107, 0.707107],
        rtol=RTOL,
        atol=ATOL,
    )
