"""No-look-ahead tests for preprocessing APIs."""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from finrl.features.preprocessing import (
    PreprocessingConfig,
    fit_transform_train_transform_test,
)
from finrl.features.schema import FeatureBundle


def _one_ticker_bundle(dates: tuple[str, ...]) -> FeatureBundle:
    asset = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "return": list(range(len(dates))),
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    macro = pl.DataFrame(
        {"date": dates, "macro_vix_diff": list(range(len(dates)))}
    ).with_columns(pl.col("date").cast(pl.Date))
    spectral = pl.DataFrame(
        {"date": dates, "volume_eigen_0": list(range(len(dates)))}
    ).with_columns(pl.col("date").cast(pl.Date))
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=tuple(asset.get_column("date").to_list()),
        tickers=("AAA",),
        asset_feature_columns=("return",),
        macro_feature_columns=("macro_vix_diff",),
        spectral_feature_columns=("volume_eigen_0",),
    )


def test_train_test_order_is_required() -> None:
    train = _one_ticker_bundle(("2024-01-12",))
    test = _one_ticker_bundle(("2024-01-05",))

    with pytest.raises(ValueError, match="Train features must end before test"):
        fit_transform_train_transform_test(train, test, PreprocessingConfig())


def test_production_api_requires_explicit_train_and_test_inputs() -> None:
    signature = inspect.signature(fit_transform_train_transform_test)

    assert tuple(signature.parameters) == (
        "train_features",
        "test_features",
        "config",
    )


def test_preprocessing_and_environment_modules_do_not_import_sklearn() -> None:
    import finrl.env.accounting as accounting
    import finrl.env.rewards as rewards
    import finrl.env.trading_env as trading_env
    import finrl.features.preprocessing as preprocessing

    for module in (accounting, rewards, trading_env, preprocessing):
        source = inspect.getsource(module)
        assert "sklearn" not in source
