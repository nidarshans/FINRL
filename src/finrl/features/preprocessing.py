"""Chronological rolling preprocessing for feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from finrl.features.schema import FeatureBundle
from finrl.features.splitsafe import FitWindow, feature_date_range, validate_train_test_order

TableKind = Literal["asset", "macro", "spectral"]
TransformKind = Literal[
    "passthrough",
    "clipped_passthrough",
    "rolling_zscore",
    "lagged_rolling_zscore",
]


@dataclass(frozen=True, slots=True)
class FeatureTransformSpec:
    """Explicit preprocessing contract for one named feature."""

    name: str
    transform: TransformKind
    rolling_periods: int | None = None
    min_periods: int = 2
    clip_lower: float | None = None
    clip_upper: float | None = None


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for no-look-ahead rolling preprocessing."""

    rolling_window: int = 252
    scale: bool = True
    clip_lower: float | None = None
    clip_upper: float | None = None
    preserve_rank_columns: bool = True
    fill_null_value: float = 0.0
    feature_transforms: tuple[FeatureTransformSpec, ...] = ()

    def __post_init__(self) -> None:
        """Validate numerical preprocessing controls at configuration time."""

        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive.")
        if (
            self.clip_lower is not None
            and self.clip_upper is not None
            and self.clip_lower > self.clip_upper
        ):
            raise ValueError("clip_lower cannot exceed clip_upper.")
        if not np.isfinite(self.fill_null_value):
            raise ValueError("fill_null_value must be finite.")
        names = tuple(spec.name for spec in self.feature_transforms)
        if len(names) != len(set(names)):
            raise ValueError("Feature transform names must be unique.")
        for spec in self.feature_transforms:
            if spec.min_periods <= 0:
                raise ValueError("Feature transform min_periods must be positive.")
            if spec.rolling_periods is not None and spec.rolling_periods <= 0:
                raise ValueError("Feature transform rolling_periods must be positive.")
            if (
                spec.clip_lower is not None
                and spec.clip_upper is not None
                and spec.clip_lower > spec.clip_upper
            ):
                raise ValueError("Feature transform clip bounds are invalid.")


@dataclass(frozen=True, slots=True)
class FittedTablePreprocessor:
    """Metadata for one rolling-preprocessed feature table."""

    kind: TableKind
    id_columns: tuple[str, ...]
    transformed_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]
    clipped_passthrough_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    rolling_window: int
    feature_transforms: tuple[FeatureTransformSpec, ...]


@dataclass(frozen=True, slots=True)
class FittedPreprocessor:
    """Preprocessing metadata and the train fit window.

    No full-window scaler statistics are stored. Rolling statistics are computed
    chronologically at transform time from current and prior rows only.
    """

    asset: FittedTablePreprocessor
    macro: FittedTablePreprocessor
    spectral: FittedTablePreprocessor
    fit_window: FitWindow
    config: PreprocessingConfig


@dataclass(frozen=True, slots=True)
class PreprocessedSplit:
    """Train/test feature bundles after chronological rolling preprocessing."""

    train: FeatureBundle
    test: FeatureBundle
    preprocessor: FittedPreprocessor


def _is_rank_column(column: str) -> bool:
    return column.endswith("_percentile_rank")


def _feature_columns(table: pl.DataFrame, id_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in table.columns if column not in id_columns)


def _split_columns(
    table: pl.DataFrame,
    id_columns: tuple[str, ...],
    config: PreprocessingConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    feature_columns = _feature_columns(table, id_columns)
    specs = {spec.name: spec for spec in config.feature_transforms}
    passthrough = tuple(
        column
        for column in feature_columns
        if specs.get(column, None) is not None
        and specs[column].transform == "passthrough"
        or (config.preserve_rank_columns and _is_rank_column(column))
    )
    clipped_passthrough = tuple(
        column
        for column in feature_columns
        if specs.get(column, None) is not None
        and specs[column].transform == "clipped_passthrough"
    )
    transformed = tuple(
        column
        for column in feature_columns
        if column not in passthrough and column not in clipped_passthrough
    )
    return transformed, passthrough, clipped_passthrough


def _table_preprocessor(
    table: pl.DataFrame,
    kind: TableKind,
    id_columns: tuple[str, ...],
    group_columns: tuple[str, ...],
    config: PreprocessingConfig,
) -> FittedTablePreprocessor:
    transformed_columns, passthrough_columns, clipped_passthrough_columns = _split_columns(
        table, id_columns, config
    )
    return FittedTablePreprocessor(
        kind=kind,
        id_columns=id_columns,
        transformed_columns=transformed_columns,
        passthrough_columns=passthrough_columns,
        clipped_passthrough_columns=clipped_passthrough_columns,
        group_columns=group_columns,
        rolling_window=config.rolling_window,
        feature_transforms=tuple(
            spec for spec in config.feature_transforms if spec.name in table.columns
        ),
    )


def build_asset_preprocessor(
    table: pl.DataFrame,
    config: PreprocessingConfig,
) -> FittedTablePreprocessor:
    """Build rolling metadata for asset features, grouped by ticker."""

    return _table_preprocessor(
        table=table,
        kind="asset",
        id_columns=("date", "ticker"),
        group_columns=("ticker",),
        config=config,
    )


def build_macro_preprocessor(
    table: pl.DataFrame,
    config: PreprocessingConfig,
) -> FittedTablePreprocessor:
    """Build rolling metadata for macro features."""

    return _table_preprocessor(
        table=table,
        kind="macro",
        id_columns=("date",),
        group_columns=(),
        config=config,
    )


def build_spectral_preprocessor(
    table: pl.DataFrame,
    config: PreprocessingConfig,
) -> FittedTablePreprocessor:
    """Build rolling metadata for spectral features."""

    return _table_preprocessor(
        table=table,
        kind="spectral",
        id_columns=("date",),
        group_columns=(),
        config=config,
    )


def fit_preprocessors(
    train_features: FeatureBundle,
    config: PreprocessingConfig,
) -> FittedPreprocessor:
    """Create rolling preprocessing metadata from explicit train features only."""

    return FittedPreprocessor(
        asset=build_asset_preprocessor(train_features.asset_features, config),
        macro=build_macro_preprocessor(train_features.macro_features, config),
        spectral=build_spectral_preprocessor(train_features.spectral_features, config),
        fit_window=feature_date_range(train_features),
        config=config,
    )


def _sort_columns(fitted: FittedTablePreprocessor) -> list[str]:
    if fitted.group_columns:
        return [*fitted.group_columns, "date"]
    return ["date"]


def _spec_for(column: str, fitted: FittedTablePreprocessor) -> FeatureTransformSpec | None:
    return next((spec for spec in fitted.feature_transforms if spec.name == column), None)


def _clip_expr(column: str, config: PreprocessingConfig, spec: FeatureTransformSpec | None = None) -> pl.Expr:
    expr = pl.col(column).cast(pl.Float64)
    lower = spec.clip_lower if spec and spec.clip_lower is not None else config.clip_lower
    upper = spec.clip_upper if spec and spec.clip_upper is not None else config.clip_upper
    if lower is not None:
        expr = expr.clip(lower_bound=lower)
    if upper is not None:
        expr = expr.clip(upper_bound=upper)
    return expr


def _fill_expr(column: str, fitted: FittedTablePreprocessor, config: PreprocessingConfig) -> pl.Expr:
    expr = _clip_expr(column, config, _spec_for(column, fitted))
    expr = pl.when(expr.is_finite()).then(expr).otherwise(None)
    if fitted.group_columns:
        expr = expr.forward_fill().over(fitted.group_columns)
    else:
        expr = expr.forward_fill()
    return expr.fill_null(config.fill_null_value)


def _rolling_mean_expr(column: str, fitted: FittedTablePreprocessor) -> pl.Expr:
    expr = pl.col(column).rolling_mean(
        window_size=fitted.rolling_window,
        min_samples=1,
    )
    if fitted.group_columns:
        return expr.over(fitted.group_columns)
    return expr


def _rolling_std_expr(column: str, fitted: FittedTablePreprocessor) -> pl.Expr:
    expr = pl.col(column).rolling_std(
        window_size=fitted.rolling_window,
        min_samples=2,
    )
    if fitted.group_columns:
        return expr.over(fitted.group_columns)
    return expr


def _rolling_standardize_table(
    table: pl.DataFrame,
    fitted: FittedTablePreprocessor,
    config: PreprocessingConfig,
) -> pl.DataFrame:
    sorted_table = table.sort(_sort_columns(fitted))
    fill_columns = (*fitted.transformed_columns, *fitted.clipped_passthrough_columns)
    working_columns = [f"__{column}_filled" for column in fill_columns]
    mean_columns = [f"__{column}_mean" for column in fitted.transformed_columns]
    std_columns = [f"__{column}_std" for column in fitted.transformed_columns]

    output = sorted_table.with_columns(
        [
            _fill_expr(column, fitted, config).alias(f"__{column}_filled")
            for column in fill_columns
        ]
    )
    if config.scale:
        output = output.with_columns(
            [
                (
                    _rolling_mean_expr(f"__{column}_filled", fitted).shift(1)
                    if (_spec_for(column, fitted) and _spec_for(column, fitted).transform == "lagged_rolling_zscore")
                    else _rolling_mean_expr(f"__{column}_filled", fitted)
                ).alias(f"__{column}_mean")
                for column in fitted.transformed_columns
            ]
            + [
                (
                    _rolling_std_expr(f"__{column}_filled", fitted).shift(1)
                    if (_spec_for(column, fitted) and _spec_for(column, fitted).transform == "lagged_rolling_zscore")
                    else _rolling_std_expr(f"__{column}_filled", fitted)
                ).alias(f"__{column}_std")
                for column in fitted.transformed_columns
            ]
        )
        output = output.with_columns(
            [
                pl.col(f"__{column}_filled").alias(column)
                for column in fitted.clipped_passthrough_columns
            ]
        )
        output = output.with_columns(
            [
                pl.when((pl.col(f"__{column}_std").is_null()) | (pl.col(f"__{column}_std") == 0.0))
                .then(0.0)
                .otherwise(
                    (pl.col(f"__{column}_filled") - pl.col(f"__{column}_mean"))
                    / pl.col(f"__{column}_std")
                )
                .alias(column)
                for column in fitted.transformed_columns
            ]
        )
    else:
        output = output.with_columns(
            [
                pl.col(f"__{column}_filled").alias(column)
                for column in fitted.transformed_columns
            ]
        )

    selected_columns = [
        *fitted.id_columns,
        *fitted.transformed_columns,
        *fitted.clipped_passthrough_columns,
        *fitted.passthrough_columns,
    ]
    if "__split" in output.columns:
        selected_columns.append("__split")
    return output.select(selected_columns).drop(
        working_columns + mean_columns + std_columns,
        strict=False,
    )


def transform_features(
    features: FeatureBundle,
    fitted_preprocessors: FittedPreprocessor,
) -> FeatureBundle:
    """Apply rolling preprocessing within a provided feature bundle."""

    asset = _rolling_standardize_table(
        features.asset_features,
        fitted_preprocessors.asset,
        fitted_preprocessors.config,
    )
    macro = _rolling_standardize_table(
        features.macro_features,
        fitted_preprocessors.macro,
        fitted_preprocessors.config,
    )
    spectral = _rolling_standardize_table(
        features.spectral_features,
        fitted_preprocessors.spectral,
        fitted_preprocessors.config,
    )
    return FeatureBundle(
        asset_features=asset,
        macro_features=macro,
        spectral_features=spectral,
        decision_dates=features.decision_dates,
        tickers=features.tickers,
        asset_feature_columns=tuple(
            column for column in asset.columns if column not in {"date", "ticker"}
        ),
        macro_feature_columns=tuple(column for column in macro.columns if column != "date"),
        spectral_feature_columns=tuple(
            column for column in spectral.columns if column != "date"
        ),
    )


def _combine_train_test_table(
    train_table: pl.DataFrame,
    test_table: pl.DataFrame,
) -> pl.DataFrame:
    return pl.concat(
        [
            train_table.with_columns(pl.lit("train").alias("__split")),
            test_table.with_columns(pl.lit("test").alias("__split")),
        ],
        how="vertical",
    )


def _split_processed_table(table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = table.filter(pl.col("__split") == "train").drop("__split")
    test = table.filter(pl.col("__split") == "test").drop("__split")
    return train, test


def fit_transform_train_transform_test(
    train_features: FeatureBundle,
    test_features: FeatureBundle,
    config: PreprocessingConfig,
) -> PreprocessedSplit:
    """Rolling-standardize train/test chronologically without future leakage."""

    validate_train_test_order(train_features, test_features)
    fitted = fit_preprocessors(train_features, config)

    combined_asset = _combine_train_test_table(
        train_features.asset_features,
        test_features.asset_features,
    )
    combined_macro = _combine_train_test_table(
        train_features.macro_features,
        test_features.macro_features,
    )
    combined_spectral = _combine_train_test_table(
        train_features.spectral_features,
        test_features.spectral_features,
    )

    processed_asset = _rolling_standardize_table(combined_asset, fitted.asset, config)
    processed_macro = _rolling_standardize_table(combined_macro, fitted.macro, config)
    processed_spectral = _rolling_standardize_table(combined_spectral, fitted.spectral, config)

    train_asset, test_asset = _split_processed_table(processed_asset)
    train_macro, test_macro = _split_processed_table(processed_macro)
    train_spectral, test_spectral = _split_processed_table(processed_spectral)

    train = FeatureBundle(
        asset_features=train_asset,
        macro_features=train_macro,
        spectral_features=train_spectral,
        decision_dates=train_features.decision_dates,
        tickers=train_features.tickers,
        asset_feature_columns=tuple(
            column for column in train_asset.columns if column not in {"date", "ticker"}
        ),
        macro_feature_columns=tuple(
            column for column in train_macro.columns if column != "date"
        ),
        spectral_feature_columns=tuple(
            column for column in train_spectral.columns if column != "date"
        ),
    )
    test = FeatureBundle(
        asset_features=test_asset,
        macro_features=test_macro,
        spectral_features=test_spectral,
        decision_dates=test_features.decision_dates,
        tickers=test_features.tickers,
        asset_feature_columns=tuple(
            column for column in test_asset.columns if column not in {"date", "ticker"}
        ),
        macro_feature_columns=tuple(column for column in test_macro.columns if column != "date"),
        spectral_feature_columns=tuple(
            column for column in test_spectral.columns if column != "date"
        ),
    )
    return PreprocessedSplit(train=train, test=test, preprocessor=fitted)
