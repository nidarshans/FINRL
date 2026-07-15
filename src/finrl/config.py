"""Project-wide configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Stable defaults shared across repository phases."""

    num_stocks: int = 100
    include_cash: bool = True
    rebalance_frequency: Literal["daily", "weekly"] = "weekly"
    transaction_cost_bps: float = 0.0
    lookback_days: int = 60
    hmm_states: int = 4
    train_years: int = 10
    test_years: int = 1

    @property
    def num_assets(self) -> int:
        """Total tradable assets, including cash when configured."""

        return self.num_stocks + int(self.include_cash)

    @property
    def transaction_cost_rate(self) -> float:
        """Transaction cost as a decimal rate."""

        return self.transaction_cost_bps / 10_000.0
