"""Tests for macro feature engineering."""

from __future__ import annotations

from datetime import date

import polars as pl
from numpy.testing import assert_allclose

from finrl.features.macro import compute_macro_features


RTOL = 1e-6
ATOL = 1e-8


def test_compute_macro_features_returns_stationary_transformations() -> None:
    macro = pl.DataFrame(
        {
            "date": [
                "2024-01-05",
                "2024-01-05",
                "2024-01-12",
                "2024-01-12",
                "2024-01-19",
                "2024-01-19",
            ],
            "ticker": ["^VIX", "GC=F", "^VIX", "GC=F", "^VIX", "GC=F"],
            "value": [10.0, 2000.0, 12.0, 2200.0, 9.0, 1980.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    features = compute_macro_features(macro, rolling_window=2)
    second_row = features.row(1, named=True)

    assert "macro_vix" not in features.columns
    assert "macro_vix_diff" in features.columns
    assert "macro_gc_f_pct_change" in features.columns
    assert "macro_vix_log_return" in features.columns
    assert "macro_vix_zscore" in features.columns
    assert_allclose(second_row["macro_vix_diff"], 2.0, rtol=RTOL, atol=ATOL)
    assert_allclose(second_row["macro_vix_pct_change"], 0.2, rtol=RTOL, atol=ATOL)
    assert_allclose(second_row["macro_gc_f_pct_change"], 0.1, rtol=RTOL, atol=ATOL)


def test_compute_macro_features_can_include_raw_levels_when_requested() -> None:
    macro = pl.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-12"],
            "ticker": ["^VIX", "^VIX"],
            "value": [10.0, 12.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    features = compute_macro_features(macro, include_levels=True)

    assert "macro_vix" in features.columns
    assert "macro_vix_diff" in features.columns


def test_compute_macro_features_uses_only_current_and_prior_values() -> None:
    macro = pl.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-12", "2024-01-19"],
            "ticker": ["^VIX", "^VIX", "^VIX"],
            "value": [10.0, 12.0, 9.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    changed_future = macro.with_columns(
        pl.when(pl.col("date") == date(2024, 1, 19))
        .then(999.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )

    base = compute_macro_features(macro, rolling_window=2).row(1, named=True)
    changed = compute_macro_features(changed_future, rolling_window=2).row(1, named=True)

    assert_allclose(
        base["macro_vix_diff"],
        changed["macro_vix_diff"],
        rtol=RTOL,
        atol=ATOL,
    )


def test_compute_macro_features_handles_empty_input() -> None:
    features = compute_macro_features(
        pl.DataFrame({"date": [], "ticker": [], "value": []})
    )

    assert features.columns == ["date"]
    assert features.height == 0
