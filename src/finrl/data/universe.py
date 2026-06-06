"""Configurable stock universe loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finrl.types import PathLikeStr


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Configuration for a user-selected stock universe."""

    tickers: tuple[str, ...]
    max_stocks: int | None = None
    include_cash: bool = True
    cash_ticker: str = "CASH"
    benchmark_ticker: str = "SPY"

    @property
    def stock_count(self) -> int:
        """Number of stock tickers before adding cash."""

        return len(self.selected_tickers)

    @property
    def selected_tickers(self) -> tuple[str, ...]:
        """Configured stock tickers, optionally truncated to ``max_stocks``."""

        unique_tickers = tuple(dict.fromkeys(normalize_ticker(t) for t in self.tickers))
        if self.max_stocks is None:
            return unique_tickers
        return unique_tickers[: self.max_stocks]

    @property
    def asset_count(self) -> int:
        """Total tradable asset count, including cash if configured."""

        return self.stock_count + int(self.include_cash)


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol for ingestion."""

    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker symbols must be non-empty.")
    return normalized


def load_universe(path: PathLikeStr) -> tuple[str, ...]:
    """Load ticker symbols from a newline or comma separated text file."""

    text = Path(path).read_text(encoding="utf-8")
    raw_tokens = text.replace(",", "\n").splitlines()
    tickers = tuple(
        normalize_ticker(token)
        for token in raw_tokens
        if token.strip() and not token.strip().startswith("#")
    )
    return tickers


def validate_universe(
    tickers: tuple[str, ...] | list[str],
    expected_count: int | None = None,
) -> tuple[str, ...]:
    """Validate and normalize an `N`-stock universe before cash is added."""

    normalized = tuple(normalize_ticker(ticker) for ticker in tickers)
    duplicates = sorted(
        {ticker for ticker in normalized if normalized.count(ticker) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate tickers are not allowed: {duplicates}")
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} stock tickers, received {len(normalized)}."
        )
    return normalized
