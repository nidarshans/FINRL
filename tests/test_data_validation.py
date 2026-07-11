"""Tests for configurable universe and yfinance ingestion boundaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from finrl.data.download import (
    _price_frame_to_macro_polars,
    _yfinance_frame_to_polars,
    download_ohlcv,
)
from finrl.data.sources import (
    DEFAULT_MACRO_TICKERS,
    YFINANCE_MACRO_PROXIES,
    MarketDataConfig,
)
from finrl.data.storage import load_or_cache_raw_data, write_parquet
from finrl.data.universe import UniverseConfig, load_universe, validate_universe


def test_validate_universe_supports_configured_n_not_hardcoded_100() -> None:
    tickers = validate_universe(["aaa", "bbb", "ccc"], expected_count=3)

    assert tickers == ("AAA", "BBB", "CCC")


def test_validate_universe_rejects_wrong_configured_n() -> None:
    with pytest.raises(ValueError, match="Expected 4"):
        validate_universe(["AAA", "BBB", "CCC"], expected_count=4)


def test_universe_config_truncates_to_max_stocks_before_cash() -> None:
    config = UniverseConfig(tickers=("AAA", "BBB", "CCC"), max_stocks=2)

    assert config.selected_tickers == ("AAA", "BBB")
    assert config.stock_count == 2
    assert config.asset_count == 3


def test_load_universe_reads_newline_and_comma_separated_tickers(tmp_path: Path) -> None:
    path = tmp_path / "universe.txt"
    path.write_text("aaa, bbb\n# comment\nccc\n", encoding="utf-8")

    assert load_universe(path) == ("AAA", "BBB", "CCC")


def test_yfinance_conversion_returns_polars_ohlcv_dataframe() -> None:
    columns = pd.MultiIndex.from_product(
        [["AAA"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        names=["Ticker", "Price"],
    )
    frame = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.5, 1_000]],
        index=pd.to_datetime(["2024-01-05"]),
        columns=columns,
    )
    frame.index.name = "Date"

    data = _yfinance_frame_to_polars(frame, ("AAA",))

    assert isinstance(data, pl.DataFrame)
    assert data.to_dicts()[0]["ticker"] == "AAA"
    assert data.to_dicts()[0]["open"] == 100.0


def test_yfinance_conversion_zero_fills_dates_without_ohlcv_data() -> None:
    columns = pd.MultiIndex.from_product(
        [
            ["AAA", "LATE"],
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        ],
        names=["Ticker", "Price"],
    )
    frame = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 100.5, 1_000, *([float("nan")] * 6)],
            [
                101.0,
                102.0,
                100.0,
                101.5,
                101.5,
                1_100,
                50.0,
                51.0,
                49.0,
                50.5,
                50.5,
                500,
            ],
        ],
        index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
        columns=columns,
    )
    frame.index.name = "Date"

    data = _yfinance_frame_to_polars(frame, ("AAA", "LATE"))

    missing_date = data.filter(
        (pl.col("ticker") == "LATE") & (pl.col("date") == pl.date(2024, 1, 5))
    )
    assert missing_date.select(
        "open", "high", "low", "close", "adj_close", "volume"
    ).row(0) == (0.0, 0.0, 0.0, 0.0, 0.0, 0)


def test_yfinance_macro_conversion_returns_polars_dataframe() -> None:
    frame = pd.DataFrame(
        {"Close": [20.0]},
        index=pd.to_datetime(["2024-01-05"]),
    )
    frame.index.name = "Date"

    data = _price_frame_to_macro_polars(frame, ("^VIX",))

    assert isinstance(data, pl.DataFrame)
    assert data.to_dicts()[0]["ticker"] == "^VIX"
    assert data.to_dicts()[0]["value"] == 20.0


def test_default_macro_tickers_use_yfinance_proxies(tmp_path: Path) -> None:
    config = MarketDataConfig(
        universe=UniverseConfig(tickers=("AAA",)),
        start="2024-01-01",
        end="2024-02-01",
        cache_dir=tmp_path,
    )

    assert YFINANCE_MACRO_PROXIES == {
        "vix": "^VIX",
        "oil": "CL=F",
        "fed_funds_rate": "ZQ=F",
        "treasury_10y": "ZN=F",
        "gold": "GC=F",
        "copper": "HG=F",
    }
    assert DEFAULT_MACRO_TICKERS == (
        "^VIX",
        "CL=F",
        "ZQ=F",
        "ZN=F",
        "GC=F",
        "HG=F",
    )
    assert config.macro_tickers == DEFAULT_MACRO_TICKERS


def test_download_ohlcv_dispatches_to_yfinance(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_download(tickers, start, end, source_config):
        called["tickers"] = tuple(tickers)
        called["start"] = start
        called["end"] = end
        called["source"] = source_config.source
        return pl.DataFrame(
            {
                "date": [],
                "ticker": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "adj_close": [],
                "volume": [],
            }
        )

    monkeypatch.setattr("finrl.data.download.download_ohlcv_yfinance", fake_download)
    config = MarketDataConfig(
        universe=UniverseConfig(tickers=("AAA",)),
        start="2024-01-01",
        end="2024-02-01",
        cache_dir=tmp_path,
    )

    result = download_ohlcv(("AAA",), config.start, config.end, config)

    assert isinstance(result, pl.DataFrame)
    assert called == {
        "tickers": ("AAA",),
        "start": "2024-01-01",
        "end": "2024-02-01",
        "source": "yfinance",
    }


def test_load_or_cache_raw_data_reads_cache_without_network(monkeypatch, tmp_path: Path) -> None:
    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called when cache exists")

    monkeypatch.setattr("finrl.data.storage.download_ohlcv", fail_download)
    cached = pl.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-05"],
            "ticker": ["AAA", "SPY"],
            "open": [100.0, 200.0],
            "high": [101.0, 201.0],
            "low": [99.0, 199.0],
            "close": [100.5, 200.5],
            "adj_close": [100.5, 200.5],
            "volume": [1_000, 2_000],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    config = MarketDataConfig(
        universe=UniverseConfig(tickers=("AAA",), benchmark_ticker="SPY"),
        start="2024-01-01",
        end="2024-02-01",
        cache_dir=tmp_path,
        macro_tickers=(),
    )
    write_parquet(cached, config.cache_path)

    bundle = load_or_cache_raw_data(config)

    assert bundle.ohlcv.height == 2
    assert bundle.spy_ohlcv.get_column("ticker").to_list() == ["SPY"]
    assert bundle.calendar.get_column("date").to_list() == [cached["date"][0]]
