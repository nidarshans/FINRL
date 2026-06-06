"""Offline sklearn preprocessing fitted on train-window features only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finrl.features.schema import FeatureBundle
from finrl.features.splitsafe import FitWindow, feature_date_range, validate_train_test_order

TableKind = Literal["asset", "macro", "spectral"]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for train-window-only preprocessing."""

    impute_strategy: str = "median"
    scale: bool = True
    clip_lower: float | None = None
    clip_upper: float | None = None
    preserve_rank_columns: bool = True


@dataclass(frozen=True, slots=True)
class FittedTablePreprocessor:
    """Fitted sklearn pipeline and column metadata for one feature table."""

    kind: TableKind
    id_columns: tuple[str, ...]
    transformed_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]
    pipeline: Pipeline | None


@dataclass(frozen=True, slots=True)
class FittedPreprocessor:
    """All fitted preprocessing artifacts and their train fit window."""

    asset: FittedTablePreprocessor
    macro: FittedTablePreprocessor
    spectral: FittedTablePreprocessor
    fit_window: FitWindow


@dataclass(frozen=True, slots=True)
class PreprocessedSplit:
    """Train/test feature bundles after train-fitted preprocessing."""

    train: FeatureBundle
    test: FeatureBundle
    preprocessor: FittedPreprocessor


class ClipTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible numeric clipping transformer."""

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, values, y=None):
        del y
        return self

    def transform(self, values):
        return np.clip(values, self.lower, self.upper)


def _build_pipeline(config: PreprocessingConfig) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.impute_strategy)),
    ]
    if config.clip_lower is not None or config.clip_upper is not None:
        steps.append(("clip", ClipTransformer(config.clip_lower, config.clip_upper)))
    if config.scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_asset_preprocessor(config: PreprocessingConfig) -> Pipeline:
    """Build the sklearn pipeline used for non-rank asset features."""

    return _build_pipeline(config)


def build_macro_preprocessor(config: PreprocessingConfig) -> Pipeline:
    """Build the sklearn pipeline used for macro features."""

    return _build_pipeline(config)


def build_spectral_preprocessor(config: PreprocessingConfig) -> Pipeline:
    """Build the sklearn pipeline used for spectral features."""

    return _build_pipeline(config)


def _is_rank_column(column: str) -> bool:
    return column.endswith("_percentile_rank")


def _feature_columns(table: pl.DataFrame, id_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in table.columns if column not in id_columns)


def _split_columns(
    table: pl.DataFrame,
    id_columns: tuple[str, ...],
    config: PreprocessingConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    feature_columns = _feature_columns(table, id_columns)
    if not config.preserve_rank_columns:
        return feature_columns, ()
    passthrough = tuple(column for column in feature_columns if _is_rank_column(column))
    transformed = tuple(column for column in feature_columns if column not in passthrough)
    return transformed, passthrough


def _fit_table(
    table: pl.DataFrame,
    kind: TableKind,
    id_columns: tuple[str, ...],
    config: PreprocessingConfig,
    pipeline: Pipeline,
) -> FittedTablePreprocessor:
    transformed_columns, passthrough_columns = _split_columns(table, id_columns, config)
    fitted_pipeline: Pipeline | None = None
    if transformed_columns:
        fitted_pipeline = pipeline.fit(table.select(transformed_columns).to_numpy())
    return FittedTablePreprocessor(
        kind=kind,
        id_columns=id_columns,
        transformed_columns=transformed_columns,
        passthrough_columns=passthrough_columns,
        pipeline=fitted_pipeline,
    )


def fit_preprocessors(
    train_features: FeatureBundle,
    config: PreprocessingConfig,
) -> FittedPreprocessor:
    """Fit preprocessing artifacts on an explicit train feature bundle only."""

    return FittedPreprocessor(
        asset=_fit_table(
            train_features.asset_features,
            "asset",
            ("date", "ticker"),
            config,
            build_asset_preprocessor(config),
        ),
        macro=_fit_table(
            train_features.macro_features,
            "macro",
            ("date",),
            config,
            build_macro_preprocessor(config),
        ),
        spectral=_fit_table(
            train_features.spectral_features,
            "spectral",
            ("date",),
            config,
            build_spectral_preprocessor(config),
        ),
        fit_window=feature_date_range(train_features),
    )


def _transform_table(
    table: pl.DataFrame,
    fitted: FittedTablePreprocessor,
) -> pl.DataFrame:
    id_frame = table.select(fitted.id_columns)
    pieces = [id_frame]
    if fitted.transformed_columns:
        if fitted.pipeline is None:
            raise ValueError(f"Missing fitted pipeline for {fitted.kind} features.")
        transformed = fitted.pipeline.transform(table.select(fitted.transformed_columns).to_numpy())
        pieces.append(
            pl.DataFrame(
                transformed,
                schema=list(fitted.transformed_columns),
                orient="row",
            )
        )
    if fitted.passthrough_columns:
        pieces.append(table.select(fitted.passthrough_columns))
    return pl.concat(pieces, how="horizontal")


def transform_features(
    features: FeatureBundle,
    fitted_preprocessors: FittedPreprocessor,
) -> FeatureBundle:
    """Transform features using train-fitted preprocessing artifacts."""

    asset = _transform_table(features.asset_features, fitted_preprocessors.asset)
    macro = _transform_table(features.macro_features, fitted_preprocessors.macro)
    spectral = _transform_table(features.spectral_features, fitted_preprocessors.spectral)
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


def fit_transform_train_transform_test(
    train_features: FeatureBundle,
    test_features: FeatureBundle,
    config: PreprocessingConfig,
) -> PreprocessedSplit:
    """Fit on train features, then transform train and test with the train fit."""

    validate_train_test_order(train_features, test_features)
    fitted = fit_preprocessors(train_features, config)
    return PreprocessedSplit(
        train=transform_features(train_features, fitted),
        test=transform_features(test_features, fitted),
        preprocessor=fitted,
    )
