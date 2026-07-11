"""Tests for weekly rebalance calendar and open-to-open returns."""

from __future__ import annotations

from datetime import date

import polars as pl
from numpy.testing import assert_allclose

from finrl.data.calendar import (
    align_to_trading_calendar,
    build_daily_rebalance_calendar,
    build_rebalance_calendar,
    build_weekly_rebalance_calendar,
    compute_open_to_open_returns,
)
from finrl.data.schema import enforce_ohlcv_schema

RTOL = 1e-6
ATOL = 1e-8


def _two_week_ohlcv() -> pl.DataFrame:
    dates = [
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-15",
    ]
    opens = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 111.1]
    return enforce_ohlcv_schema(
        pl.DataFrame(
            {
                "date": dates,
                "ticker": ["AAA"] * len(dates),
                "open": opens,
                "high": opens,
                "low": opens,
                "close": opens,
                "adj_close": opens,
                "volume": [1_000] * len(dates),
            }
        )
    )


def test_build_weekly_rebalance_calendar_maps_friday_to_monday() -> None:
    calendar = build_weekly_rebalance_calendar(_two_week_ohlcv())

    assert calendar.to_dicts() == [
        {
            "decision_date": date(2024, 1, 5),
            "execution_date": date(2024, 1, 8),
            "next_execution_date": date(2024, 1, 15),
        }
    ]


def test_build_daily_rebalance_calendar_maps_each_day_to_next_two_sessions() -> None:
    calendar = build_daily_rebalance_calendar(_two_week_ohlcv())

    assert calendar.head(2).to_dicts() == [
        {
            "decision_date": date(2024, 1, 5),
            "execution_date": date(2024, 1, 8),
            "next_execution_date": date(2024, 1, 9),
        },
        {
            "decision_date": date(2024, 1, 8),
            "execution_date": date(2024, 1, 9),
            "next_execution_date": date(2024, 1, 10),
        },
    ]


def test_build_weekly_calendar_uses_thursday_when_friday_is_holiday() -> None:
    dates = [
        "2026-06-12",
        "2026-06-15",
        "2026-06-16",
        "2026-06-17",
        "2026-06-18",
        "2026-06-22",
        "2026-06-23",
        "2026-06-24",
        "2026-06-25",
        "2026-06-26",
        "2026-06-29",
    ]
    prices = enforce_ohlcv_schema(
        pl.DataFrame(
            {
                "date": dates,
                "ticker": ["AAA"] * len(dates),
                "open": [100.0] * len(dates),
                "high": [100.0] * len(dates),
                "low": [100.0] * len(dates),
                "close": [100.0] * len(dates),
                "adj_close": [100.0] * len(dates),
                "volume": [1_000] * len(dates),
            }
        )
    )

    calendar = build_weekly_rebalance_calendar(prices)

    assert calendar.row(1, named=True) == {
        "decision_date": date(2026, 6, 18),
        "execution_date": date(2026, 6, 22),
        "next_execution_date": date(2026, 6, 29),
    }


def test_build_rebalance_calendar_dispatches_frequency() -> None:
    prices = _two_week_ohlcv()

    assert build_rebalance_calendar(prices, "weekly").equals(
        build_weekly_rebalance_calendar(prices)
    )
    assert build_rebalance_calendar(prices, "daily").equals(
        build_daily_rebalance_calendar(prices)
    )


def test_compute_open_to_open_returns_uses_same_weekly_holding_period() -> None:
    prices = _two_week_ohlcv()
    calendar = build_weekly_rebalance_calendar(prices)

    returns = compute_open_to_open_returns(prices, calendar)

    row = returns.row(0, named=True)
    assert row["ticker"] == "AAA"
    assert row["execution_date"] == date(2024, 1, 8)
    assert row["next_execution_date"] == date(2024, 1, 15)
    assert_allclose(row["return"], 0.10, rtol=RTOL, atol=ATOL)


def test_compute_open_to_open_returns_supports_daily_holding_period() -> None:
    prices = _two_week_ohlcv()
    calendar = build_daily_rebalance_calendar(prices)

    returns = compute_open_to_open_returns(prices, calendar)

    row = returns.row(0, named=True)
    assert row["decision_date"] == date(2024, 1, 5)
    assert row["execution_date"] == date(2024, 1, 8)
    assert row["next_execution_date"] == date(2024, 1, 9)
    assert_allclose(row["return"], 102.0 / 101.0 - 1.0, rtol=RTOL, atol=ATOL)


def test_compute_open_to_open_returns_treats_zero_price_as_missing_data() -> None:
    prices = _two_week_ohlcv().with_columns(
        pl.when(
            pl.col("date").is_in([date(2024, 1, 8), date(2024, 1, 9)])
        )
        .then(0.0)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    calendar = build_daily_rebalance_calendar(prices)

    returns = compute_open_to_open_returns(prices, calendar)

    assert returns.head(2).get_column("return").to_list() == [0.0, 0.0]


def test_align_to_trading_calendar_filters_dates() -> None:
    data = _two_week_ohlcv()
    calendar = pl.DataFrame({"date": ["2024-01-05", "2024-01-08"]}).with_columns(
        pl.col("date").cast(pl.Date)
    )

    aligned = align_to_trading_calendar(data, calendar)

    assert aligned.get_column("date").to_list() == [
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
