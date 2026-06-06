"""Data source and market data bundle configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from finrl.data.universe import UniverseConfig
from finrl.types import PathLikeStr

YFINANCE_MACRO_PROXIES: dict[str, str] = {
    "vix": "^VIX",
    "oil": "CL=F",
    "fed_funds_rate": "ZQ=F",
    "treasury_10y": "ZN=F",
    "gold": "GC=F",
    "copper": "HG=F",
}

DEFAULT_MACRO_TICKERS: tuple[str, ...] = tuple(YFINANCE_MACRO_PROXIES.values())


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    """Configuration for raw market data ingestion."""

    universe: UniverseConfig
    start: str
    end: str
    cache_dir: PathLikeStr
    source: str = "yfinance"
    auto_adjust: bool = False
    actions: bool = False
    macro_tickers: tuple[str, ...] = DEFAULT_MACRO_TICKERS

    @property
    def cache_path(self) -> Path:
        """Default OHLCV cache path."""

        return Path(self.cache_dir) / "ohlcv.parquet"


@dataclass(frozen=True, slots=True)
class MarketDataBundle:
    """Aligned raw market data artifacts."""

    ohlcv: pl.DataFrame
    spy_ohlcv: pl.DataFrame
    macro: pl.DataFrame
    calendar: pl.DataFrame
