"""Robustness, stress-testing, and capacity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.backtest.results import PerformanceMetrics, calculate_performance_metrics


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Simple ADV participation-based capacity estimate."""

    average_daily_dollar_volume: float
    participation_limit: float
    estimated_capacity: float


def stress_transaction_costs(
    returns: np.ndarray,
    spy_returns: np.ndarray,
    turnovers: np.ndarray,
    costs: np.ndarray,
    periods_per_year: int,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
) -> dict[float, PerformanceMetrics]:
    """Recalculate net metrics under proportional cost multipliers."""

    values = np.asarray(returns, dtype=np.float64)
    cost_values = np.asarray(costs, dtype=np.float64)
    if values.shape != cost_values.shape:
        raise ValueError("returns and costs must have matching shapes.")
    output: dict[float, PerformanceMetrics] = {}
    for multiplier in multipliers:
        if multiplier < 0.0 or not np.isfinite(multiplier):
            raise ValueError("cost multipliers must be finite and non-negative.")
        gross = values + cost_values
        stressed = gross - multiplier * cost_values
        output[float(multiplier)] = calculate_performance_metrics(
            stressed, spy_returns, turnovers, multiplier * cost_values, periods_per_year
        )
    return output


def execution_delay_returns(returns: np.ndarray, delay_periods: int = 1) -> np.ndarray:
    """Apply a conservative signal/execution delay to a return stream."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or delay_periods < 0:
        raise ValueError("returns must be one-dimensional and delay_periods non-negative.")
    if delay_periods == 0:
        return values.copy()
    if delay_periods >= values.size:
        return np.zeros_like(values)
    return np.concatenate((np.zeros(delay_periods), values[:-delay_periods]))


def estimate_capacity(
    dollar_volume: np.ndarray,
    participation_limit: float = 0.10,
) -> CapacityEstimate:
    """Estimate deployable capital from average dollar volume and participation."""

    volume = np.asarray(dollar_volume, dtype=np.float64)
    if volume.ndim != 1 or volume.size == 0 or not np.isfinite(volume).all() or np.any(volume < 0.0):
        raise ValueError("dollar_volume must be finite, non-negative, and one-dimensional.")
    if not 0.0 < participation_limit <= 1.0:
        raise ValueError("participation_limit must be in (0, 1].")
    average = float(np.mean(volume))
    return CapacityEstimate(average, participation_limit, average * participation_limit)


def subperiod_metrics(
    returns: np.ndarray,
    spy_returns: np.ndarray,
    periods_per_year: int,
    block_size: int,
) -> tuple[PerformanceMetrics, ...]:
    """Calculate metrics over contiguous non-overlapping subperiods."""

    values = np.asarray(returns)
    benchmark = np.asarray(spy_returns)
    if values.ndim != 1 or values.shape != benchmark.shape or block_size <= 0:
        raise ValueError("returns, spy_returns, and block_size are incompatible.")
    return tuple(
        calculate_performance_metrics(values[start : start + block_size], benchmark[start : start + block_size], periods_per_year=periods_per_year)
        for start in range(0, values.size - block_size + 1, block_size)
    )


def release_gate(metrics: PerformanceMetrics, max_drawdown: float, min_information_ratio: float = 0.0) -> bool:
    """Return whether headline risk and benchmark-relative thresholds pass."""

    if max_drawdown < 0.0:
        raise ValueError("max_drawdown must be non-negative.")
    return metrics.max_drawdown <= max_drawdown and metrics.information_ratio >= min_information_ratio
