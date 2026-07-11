"""Market data download functions."""

from __future__ import annotations

import pandas as pd
import polars as pl

from finrl.data.schema import enforce_ohlcv_schema
from finrl.data.sources import MarketDataConfig
from finrl.data.universe import validate_universe


def _empty_macro_frame() -> pl.DataFrame:
    return pl.DataFrame({"date": [], "ticker": [], "value": []}).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("value").cast(pl.Float64),
    )


def _price_frame_to_macro_polars(
    frame: pd.DataFrame, tickers: tuple[str, ...]
) -> pl.DataFrame:
    """Convert yfinance price data into long macro proxy values."""

    if frame.empty:
        raise ValueError("yfinance returned no macro rows.")

    rows: list[pd.DataFrame] = []
    if isinstance(frame.columns, pd.MultiIndex):
        for ticker in tickers:
            ticker_frame = frame.xs(ticker, axis=1, level="Ticker", drop_level=True)
            value_column = "Adj Close" if "Adj Close" in ticker_frame.columns else "Close"
            value_frame = ticker_frame[[value_column]].reset_index()
            value_frame["ticker"] = ticker
            value_frame = value_frame.rename(columns={"Date": "date", value_column: "value"})
            rows.append(value_frame)
        long_frame = pd.concat(rows, ignore_index=True)
    else:
        value_column = "Adj Close" if "Adj Close" in frame.columns else "Close"
        long_frame = frame[[value_column]].reset_index()
        long_frame["ticker"] = tickers[0]
        long_frame = long_frame.rename(columns={"Date": "date", value_column: "value"})

    return pl.from_pandas(long_frame).select(
        [
            pl.col("date").cast(pl.Date),
            pl.col("ticker").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64),
        ]
    )


def _yfinance_frame_to_polars(frame: pd.DataFrame, tickers: tuple[str, ...]) -> pl.DataFrame:
    """Convert a yfinance DataFrame into canonical long OHLCV Polars data."""

    if frame.empty:
        raise ValueError("yfinance returned no OHLCV rows.")

    rows: list[pd.DataFrame] = []
    if isinstance(frame.columns, pd.MultiIndex):
        for ticker in tickers:
            ticker_frame = frame.xs(ticker, axis=1, level="Ticker", drop_level=True)
            ticker_frame = ticker_frame.reset_index()
            ticker_frame["ticker"] = ticker
            rows.append(ticker_frame)
        long_frame = pd.concat(rows, ignore_index=True)
    else:
        long_frame = frame.reset_index()
        long_frame["ticker"] = tickers[0]

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    long_frame = long_frame.rename(columns=rename_map)
    if "adj_close" not in long_frame.columns:
        long_frame["adj_close"] = long_frame["close"]
    polars_frame = enforce_ohlcv_schema(pl.from_pandas(long_frame))
    return (
        polars_frame.with_columns(
            pl.col("open", "high", "low", "close", "adj_close", "volume")
            .fill_null(0)
        )
        .sort(["ticker", "date"])
    )


def download_ohlcv_yfinance(
    tickers: tuple[str, ...] | list[str],
    start: str,
    end: str,
    source_config: MarketDataConfig,
) -> pl.DataFrame:
    """Download ticker OHLCV data from yfinance as a Polars DataFrame."""

    import yfinance as yf

    normalized = validate_universe(tuple(tickers))
    frame = yf.download(
        list(normalized),
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=source_config.auto_adjust,
        actions=source_config.actions,
        progress=False,
        threads=False,
    )
    return _yfinance_frame_to_polars(frame, normalized)


def download_ohlcv(
    tickers: tuple[str, ...] | list[str],
    start: str,
    end: str,
    source_config: MarketDataConfig,
) -> pl.DataFrame:
    """Download ticker OHLCV data from the configured source."""

    if source_config.source != "yfinance":
        raise ValueError(f"Unsupported market data source: {source_config.source}")
    return download_ohlcv_yfinance(tickers, start, end, source_config)


def download_macro_series(
    start: str,
    end: str,
    source_config: MarketDataConfig,
) -> pl.DataFrame:
    """Download configured macro proxy tickers through yfinance."""

    if not source_config.macro_tickers:
        return _empty_macro_frame()
    import yfinance as yf

    tickers = validate_universe(source_config.macro_tickers)
    frame = yf.download(
        list(tickers),
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=source_config.auto_adjust,
        actions=source_config.actions,
        progress=False,
        threads=False,
    )
    return _price_frame_to_macro_polars(frame, tickers)
