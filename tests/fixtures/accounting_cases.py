"""Hand-computed accounting fixtures for Phase 3 tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccountingCase:
    """Expected values for a one-step rebalance case."""

    name: str
    current_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    asset_returns: tuple[float, ...]
    spy_return: float
    transaction_cost_rate: float
    starting_value: float
    starting_peak: float
    turnover: float
    transaction_cost: float
    gross_return: float
    net_return: float
    ending_value: float
    ending_peak: float
    drawdown: float


ACCOUNTING_CASES: tuple[AccountingCase, ...] = (
    AccountingCase(
        name="mixed_stock_cash_rebalance",
        current_weights=(0.50, 0.30, 0.20),
        target_weights=(0.20, 0.50, 0.30),
        asset_returns=(0.020, -0.010, 0.001),
        spy_return=0.004,
        transaction_cost_rate=0.001,
        starting_value=100.0,
        starting_peak=100.0,
        turnover=0.30,
        transaction_cost=0.00030,
        gross_return=-0.00070,
        net_return=-0.00100,
        ending_value=99.90,
        ending_peak=100.0,
        drawdown=0.00100,
    ),
    AccountingCase(
        name="all_cash_no_transaction_cost",
        current_weights=(0.50, 0.30, 0.20),
        target_weights=(0.00, 0.00, 1.00),
        asset_returns=(0.100, -0.100, 0.001),
        spy_return=0.0,
        transaction_cost_rate=0.0,
        starting_value=100.0,
        starting_peak=100.0,
        turnover=0.80,
        transaction_cost=0.0,
        gross_return=0.001,
        net_return=0.001,
        ending_value=100.10,
        ending_peak=100.10,
        drawdown=0.0,
    ),
)
