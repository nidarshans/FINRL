"""Walk-forward backtest result containers and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Aggregate performance metrics for one return stream."""

    cumulative_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    mean_turnover: float
    total_transaction_cost: float
    spy_relative_alpha: float


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Result for one walk-forward split."""

    split_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    portfolio_returns: pl.DataFrame
    spy_returns: pl.DataFrame
    allocations: pl.DataFrame
    spectral_features: pl.DataFrame
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregated walk-forward experiment result."""

    split_results: tuple[SplitResult, ...]
    portfolio_curve: pl.DataFrame
    spy_curve: pl.DataFrame
    allocations: pl.DataFrame
    spectral_features: pl.DataFrame
    aggregate_metrics: PerformanceMetrics
    aggregate_benchmark_metrics: PerformanceMetrics


def equity_curve(returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """Return an equity curve from period returns."""

    values = [float(initial_value)]
    for period_return in returns:
        values.append(values[-1] * (1.0 + float(period_return)))
    return np.asarray(values[1:], dtype=np.float64)


def max_drawdown_from_curve(curve: np.ndarray) -> float:
    """Return maximum drawdown for an equity curve."""

    if curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    drawdowns = 1.0 - curve / peaks
    return float(np.max(drawdowns))


def calculate_performance_metrics(
    returns: np.ndarray,
    spy_returns: np.ndarray,
    turnovers: np.ndarray | None = None,
    transaction_costs: np.ndarray | None = None,
    periods_per_year: int = 52,
) -> PerformanceMetrics:
    """Calculate portfolio metrics versus SPY holding-period returns."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    returns = np.asarray(returns, dtype=np.float64)
    spy_returns = np.asarray(spy_returns, dtype=np.float64)
    if returns.shape != spy_returns.shape:
        raise ValueError("returns and spy_returns must have matching shapes.")
    if returns.size == 0:
        raise ValueError("Cannot calculate metrics for empty returns.")
    turnovers_array = (
        np.zeros_like(returns)
        if turnovers is None
        else np.asarray(turnovers, dtype=np.float64)
    )
    costs_array = (
        np.zeros_like(returns)
        if transaction_costs is None
        else np.asarray(transaction_costs, dtype=np.float64)
    )
    if turnovers_array.shape != returns.shape or costs_array.shape != returns.shape:
        raise ValueError("turnovers and transaction_costs must match returns shape.")

    curve = equity_curve(returns)
    spy_curve = equity_curve(spy_returns)
    annualizer = float(periods_per_year)
    cumulative_return = float(curve[-1] - 1.0)
    annualized_return = float(curve[-1] ** (annualizer / returns.size) - 1.0)
    volatility = float(np.std(returns, ddof=0) * np.sqrt(annualizer))
    spy_cumulative_return = float(spy_curve[-1] - 1.0)
    return PerformanceMetrics(
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_drawdown_from_curve(curve),
        mean_turnover=float(np.mean(turnovers_array)),
        total_transaction_cost=float(np.sum(costs_array)),
        spy_relative_alpha=cumulative_return - spy_cumulative_return,
    )
