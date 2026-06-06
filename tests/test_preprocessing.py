"""Tests for train-window-fitted sklearn preprocessing."""

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
                "return": value + 1.0,
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


def test_fit_preprocessors_records_train_window() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )

    fitted = fit_preprocessors(train, PreprocessingConfig())

    assert fitted.fit_window.start == date(2024, 1, 5)
    assert fitted.fit_window.end == date(2024, 1, 12)


def test_fit_transform_uses_train_statistics_for_test_window() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = _bundle(
        ("2024-01-19",),
        (1_000.0,),
        (9_999.0,),
        (999_999.0,),
    )

    split = fit_transform_train_transform_test(train, test, PreprocessingConfig())
    train_return_values = split.train.asset_features.get_column("return").to_list()

    assert_allclose(
        train_return_values,
        [-1.341641, -0.447214, 0.447214, 1.341641],
        rtol=RTOL,
        atol=ATOL,
    )
    assert split.preprocessor.asset.pipeline is not None
    assert_allclose(
        split.preprocessor.asset.pipeline.named_steps["scaler"].mean_,
        [2.5],
        rtol=RTOL,
        atol=ATOL,
    )


def test_transforming_different_test_values_does_not_change_fit_metadata() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    mild_test = _bundle(("2024-01-19",), (4.0,), (14.0,), (120.0,))
    extreme_test = _bundle(("2024-01-19",), (10_000.0,), (99_999.0,), (999_999.0,))
    fitted = fit_preprocessors(train, PreprocessingConfig())

    transform_features(mild_test, fitted)
    transform_features(extreme_test, fitted)

    assert fitted.asset.pipeline is not None
    assert_allclose(
        fitted.asset.pipeline.named_steps["scaler"].mean_,
        [2.5],
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

    split = fit_transform_train_transform_test(train, test, PreprocessingConfig())

    assert split.train.asset_features.shape == train.asset_features.shape
    assert split.test.asset_features.shape == test.asset_features.shape
    assert split.test.asset_features.select(["date", "ticker"]).to_dicts() == (
        test.asset_features.select(["date", "ticker"]).to_dicts()
    )
    assert split.test.macro_features.get_column("date").to_list() == [
        date(2024, 1, 19)
    ]


def test_rank_columns_are_not_globally_scaled() -> None:
    train = _bundle(
        ("2024-01-05", "2024-01-12"),
        (1.0, 3.0),
        (10.0, 12.0),
        (100.0, 110.0),
    )
    test = _bundle(("2024-01-19",), (5.0,), (14.0,), (120.0,))

    split = fit_transform_train_transform_test(train, test, PreprocessingConfig())

    assert split.train.asset_features.get_column("return_percentile_rank").to_list() == [
        0.25,
        0.75,
        0.25,
        0.75,
    ]
    assert split.test.asset_features.get_column("return_percentile_rank").to_list() == [
        0.25,
        0.75,
    ]


def test_clipping_runs_before_scaling() -> None:
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
        PreprocessingConfig(clip_lower=-10.0, clip_upper=10.0),
    )

    assert split.test.asset_features.get_column("return").max() < 10.0


def test_missing_values_are_imputed_from_train_window() -> None:
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
                "return": [None, 6.0],
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

    split = fit_transform_train_transform_test(train, test, PreprocessingConfig())

    assert split.test.asset_features.select(pl.col("return").is_null().sum()).item() == 0
    assert split.test.macro_features.select(pl.col("macro_vix_diff").is_null().sum()).item() == 0
    assert split.test.spectral_features.select(pl.col("volume_eigen_0").is_null().sum()).item() == 0
