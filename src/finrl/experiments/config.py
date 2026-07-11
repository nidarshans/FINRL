"""Configuration for walk-forward experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from finrl.backtest.walk_forward import WalkForwardConfig
from finrl.dpo_jax.config import DPOConfig
from finrl.env.trading_env import EnvConfig
from finrl.features.preprocessing import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a strict walk-forward experiment."""

    walk_forward: WalkForwardConfig = WalkForwardConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    dpo: DPOConfig = DPOConfig()
    env: EnvConfig = EnvConfig()
    enable_dpo: bool = True
    rebalance_frequency: Literal["daily", "weekly"] = "weekly"
    seed: int = 0
    periods_per_year: int | None = None

    def __post_init__(self) -> None:
        if self.rebalance_frequency not in {"daily", "weekly"}:
            raise ValueError("rebalance_frequency must be 'daily' or 'weekly'.")
        if self.periods_per_year is not None and self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        dpo_cost_rate = self.dpo.transaction_cost_bps / 10000.0
        if self.enable_dpo and not math.isclose(
            dpo_cost_rate,
            self.env.transaction_cost_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "DPO and environment transaction cost rates must match."
            )
        if self.enable_dpo and self.env.top_n_positions is not None:
            raise ValueError(
                "DPO requires top_n_positions=None so training and execution match."
            )

    @property
    def annualization_periods(self) -> int:
        """Return an explicit override or the frequency-derived default."""

        if self.periods_per_year is not None:
            return self.periods_per_year
        return 252 if self.rebalance_frequency == "daily" else 52
