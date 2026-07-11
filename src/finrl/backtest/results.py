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
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    regression_alpha: float = 0.0


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
    regime_probabilities: pl.DataFrame
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
    regime_probabilities: pl.DataFrame
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
    mean_return = float(np.mean(returns))
    sharpe = mean_return / float(np.std(returns, ddof=0)) * np.sqrt(annualizer) if np.std(returns) > 0 else 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(annualizer))
    sortino = mean_return * annualizer / downside_dev if downside_dev > 0 else 0.0
    max_dd = max_drawdown_from_curve(curve)
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0
    active = returns - spy_returns
    tracking_error = float(np.std(active, ddof=0) * np.sqrt(annualizer))
    information_ratio = float(np.mean(active) * annualizer / tracking_error) if tracking_error > 0 else 0.0
    benchmark_var = float(np.var(spy_returns, ddof=0))
    beta = float(np.cov(returns, spy_returns, ddof=0)[0, 1] / benchmark_var) if benchmark_var > 0 else 0.0
    regression_alpha = float((mean_return - beta * float(np.mean(spy_returns))) * annualizer)
    return PerformanceMetrics(
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_dd,
        mean_turnover=float(np.mean(turnovers_array)),
        total_transaction_cost=float(np.sum(costs_array)),
        spy_relative_alpha=cumulative_return - spy_cumulative_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        beta=beta,
        regression_alpha=regression_alpha,
    )
