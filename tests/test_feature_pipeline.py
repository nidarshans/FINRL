"""Tests for end-to-end feature bundle construction."""

from __future__ import annotations

from datetime import date

import polars as pl

from finrl.data.sources import MarketDataBundle
from finrl.features.pipeline import build_feature_bundle
from finrl.features.schema import FeatureConfig


def _pipeline_ohlcv() -> pl.DataFrame:
    dates = [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    rows = []
    for ticker, base in (("AAA", 10.0), ("BBB", 20.0)):
        for index, day in enumerate(dates):
            price = base + float(index)
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 100 + index,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_build_feature_bundle_aligns_to_friday_decision_dates() -> None:
    ohlcv = _pipeline_ohlcv()
    macro = pl.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-12"],
            "ticker": ["^VIX", "^VIX"],
            "value": [13.0, 14.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    calendar = pl.DataFrame(
        {
            "decision_date": ["2024-01-05", "2024-01-12"],
            "execution_date": ["2024-01-08", "2024-01-15"],
            "next_execution_date": ["2024-01-15", "2024-01-22"],
        }
    ).with_columns(pl.all().cast(pl.Date))
    bundle = MarketDataBundle(
        ohlcv=ohlcv,
        spy_ohlcv=pl.DataFrame(),
        macro=macro,
        calendar=calendar,
    )

    features = build_feature_bundle(
        bundle,
        FeatureConfig(
            rsi_window=2,
            trend_window=3,
            liquidity_window=2,
            volume_window=2,
            spectral_window=2,
            spectral_dim=20,
            use_spectral_features=False,
        ),
    )

    assert features.decision_dates == (date(2024, 1, 5), date(2024, 1, 12))
    assert features.tickers == ("AAA", "BBB")
    assert features.asset_features.height == 4
    assert features.macro_features.height == 2
    assert features.spectral_features.height == 2
    assert len(features.spectral_feature_columns) == 20
    assert features.spectral_features.drop("date").sum().row(0) == (0.0,) * 20
    assert set(features.asset_features.get_column("date").to_list()) == {
        date(2024, 1, 5),
        date(2024, 1, 12),
    }
