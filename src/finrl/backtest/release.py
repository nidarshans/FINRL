"""Final release-validation aggregation for a frozen backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.backtest.results import PerformanceMetrics, calculate_performance_metrics
from finrl.backtest.robustness import CapacityEstimate, estimate_capacity, release_gate, stress_transaction_costs


@dataclass(frozen=True, slots=True)
class ReleaseValidation:
    """Consolidated robustness evidence for a candidate strategy."""

    base_metrics: PerformanceMetrics
    cost_stress: dict[float, PerformanceMetrics]
    capacity: CapacityEstimate
    passed: bool


def validate_release(
    returns: np.ndarray,
    spy_returns: np.ndarray,
    turnovers: np.ndarray,
    costs: np.ndarray,
    dollar_volume: np.ndarray,
    periods_per_year: int,
    max_drawdown: float,
    min_information_ratio: float = 0.0,
) -> ReleaseValidation:
    """Run deterministic release checks on an untouched evaluation stream."""

    base = calculate_performance_metrics(
        returns, spy_returns, turnovers, costs, periods_per_year
    )
    stress = stress_transaction_costs(
        returns, spy_returns, turnovers, costs, periods_per_year
    )
    capacity = estimate_capacity(dollar_volume)
    passed = release_gate(base, max_drawdown, min_information_ratio) and all(
        release_gate(metrics, max_drawdown, min_information_ratio)
        for multiplier, metrics in stress.items()
        if multiplier >= 2.0
    )
    return ReleaseValidation(base, stress, capacity, passed)
