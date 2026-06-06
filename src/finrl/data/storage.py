"""Cache helpers for Polars market data."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from finrl.data.download import download_macro_series, download_ohlcv
from finrl.data.sources import MarketDataBundle, MarketDataConfig
from finrl.data.validation import validate_ohlcv_data
from finrl.types import PathLikeStr


def write_parquet(data: pl.DataFrame, path: PathLikeStr) -> None:
    """Write a Polars DataFrame to parquet, creating parents as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path)


def read_parquet(path: PathLikeStr) -> pl.DataFrame:
    """Read a Polars parquet file."""

    return pl.read_parquet(Path(path))


def load_or_cache_raw_data(config: MarketDataConfig) -> MarketDataBundle:
    """Load cached OHLCV data or download and cache it."""

    if config.cache_path.exists():
        ohlcv = read_parquet(config.cache_path)
    else:
        tickers = tuple(
            dict.fromkeys(
                (*config.universe.selected_tickers, config.universe.benchmark_ticker)
            )
        )
        ohlcv = download_ohlcv(
            tickers,
            config.start,
            config.end,
            config,
        )
        write_parquet(ohlcv, config.cache_path)

    expected_tickers = tuple(
        dict.fromkeys((*config.universe.selected_tickers, config.universe.benchmark_ticker))
    )
    validate_ohlcv_data(ohlcv, expected_tickers=expected_tickers)
    spy_ohlcv = ohlcv.filter(pl.col("ticker") == config.universe.benchmark_ticker)
    macro = download_macro_series(config.start, config.end, config)
    calendar = ohlcv.select("date").unique().sort("date")
    return MarketDataBundle(ohlcv=ohlcv, spy_ohlcv=spy_ohlcv, macro=macro, calendar=calendar)
