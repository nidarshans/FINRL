"""Trading calendar alignment and open-to-open return calculations."""

from __future__ import annotations

import polars as pl

from finrl.data.schema import enforce_ohlcv_schema, enforce_returns_schema


def align_to_trading_calendar(data: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Filter OHLCV data to dates present in the supplied trading calendar."""

    ohlcv = enforce_ohlcv_schema(data)
    if "date" not in calendar.columns:
        raise ValueError("Calendar must contain a 'date' column.")
    calendar_dates = calendar.select(pl.col("date").cast(pl.Date)).unique()
    return ohlcv.join(calendar_dates, on="date", how="inner").sort(["ticker", "date"])


def build_weekly_rebalance_calendar(daily_prices: pl.DataFrame) -> pl.DataFrame:
    """Build Friday decision to next-session execution calendar from OHLCV dates."""

    trading_dates = (
        enforce_ohlcv_schema(daily_prices)
        .select(pl.col("date"))
        .unique()
        .sort("date")
        .with_columns(
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").shift(-1).alias("execution_date"),
        )
        .filter(pl.col("weekday") == 5)
        .drop("weekday")
        .rename({"date": "decision_date"})
    )
    return trading_dates.with_columns(
        pl.col("execution_date").shift(-1).alias("next_execution_date")
    ).drop_nulls(["execution_date", "next_execution_date"])


def compute_open_to_open_returns(
    open_prices: pl.DataFrame,
    rebalance_calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Compute holding returns from execution open to next execution open."""

    prices = enforce_ohlcv_schema(open_prices).select(["date", "ticker", "open"])
    required_calendar_cols = {
        "decision_date",
        "execution_date",
        "next_execution_date",
    }
    missing = required_calendar_cols.difference(rebalance_calendar.columns)
    if missing:
        raise ValueError(f"Missing rebalance calendar columns: {sorted(missing)}")

    calendar = rebalance_calendar.select(
        [
            pl.col("decision_date").cast(pl.Date),
            pl.col("execution_date").cast(pl.Date),
            pl.col("next_execution_date").cast(pl.Date),
        ]
    )
    execution = calendar.join(
        prices.rename({"date": "execution_date", "open": "open"}),
        on="execution_date",
        how="inner",
    )
    next_execution = prices.rename(
        {"date": "next_execution_date", "open": "next_open"}
    )
    returns = execution.join(
        next_execution,
        on=["next_execution_date", "ticker"],
        how="inner",
    ).with_columns((pl.col("next_open") / pl.col("open") - 1.0).alias("return"))
    return enforce_returns_schema(returns).sort(["ticker", "decision_date"])
