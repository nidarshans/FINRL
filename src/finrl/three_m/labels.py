"""Crossover and line-maintenance target construction for 3M."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.three_m.config import LabelConfig


@dataclass(frozen=True, slots=True)
class LabelInputs:
    """Decision-date prices and causal distances to the three reference lines."""

    close: np.ndarray
    close_ema20_gap: np.ndarray
    close_ema50_gap: np.ndarray
    close_vwap20_gap: np.ndarray


@dataclass(frozen=True, slots=True)
class ThreeMTargets:
    """Aligned binary labels with complete forward outcomes only."""

    buy: np.ndarray
    hold: np.ndarray
    sell: np.ndarray
    n_times: int
    valid_mask: np.ndarray | None = None


def _validated_inputs(
    execution_returns: np.ndarray, inputs: LabelInputs
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    returns = np.asarray(execution_returns, dtype=np.float64)
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            inputs.close,
            inputs.close_ema20_gap,
            inputs.close_ema50_gap,
            inputs.close_vwap20_gap,
        )
    )
    if returns.ndim != 2 or any(values.shape != returns.shape for values in arrays):
        raise ValueError("Returns and every label input must match [time, assets].")
    if not np.isfinite(returns).all() or any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Returns and label inputs must be finite.")
    if np.any(arrays[0] <= 0.0):
        raise ValueError("Close prices must be positive.")
    return returns, arrays


def _crosses_above(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return (current > 0.0) & (previous <= 0.0)


def _crosses_below(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return (current < 0.0) & (previous >= 0.0)


def build_targets(
    execution_returns: np.ndarray,
    inputs: LabelInputs,
    config: LabelConfig,
) -> ThreeMTargets:
    """Build crossover-conditioned entry/exit labels and line-maintenance holds.

    Buy is positive when price crosses above EMA-20, EMA-50, or VWAP-20 and
    its compounded forward return reaches the configured threshold. Sell is
    positive for a corresponding bearish crossover followed by the configured
    maximum drawdown. Hold is positive when price remains above EMA-20 or
    VWAP-20, or is no more than ``ema50_epsilon`` below EMA-50.
    """

    returns, (close, ema20_gap, ema50_gap, vwap_gap) = _validated_inputs(
        execution_returns, inputs
    )
    n_times = returns.shape[0] - config.outcome_horizon + 1
    if n_times <= 0:
        raise ValueError("Not enough rows for the configured outcome_horizon.")
    n_assets = returns.shape[1]
    buy = np.zeros((n_times, n_assets), dtype=bool)
    hold = np.zeros_like(buy)
    sell = np.zeros_like(buy)
    valid_mask = np.ones((n_times, n_assets), dtype=bool)
    valid_mask[0] = False
    for time_index in range(n_times):
        path = np.cumprod(
            1.0 + returns[time_index : time_index + config.outcome_horizon], axis=0
        ) - 1.0
        bullish_crossover = (
            _crosses_above(ema20_gap[time_index], ema20_gap[time_index - 1])
            | _crosses_above(ema50_gap[time_index], ema50_gap[time_index - 1])
            | _crosses_above(vwap_gap[time_index], vwap_gap[time_index - 1])
        ) if time_index > 0 else np.zeros(n_assets, dtype=bool)
        bearish_crossover = (
            _crosses_below(ema20_gap[time_index], ema20_gap[time_index - 1])
            | _crosses_below(ema50_gap[time_index], ema50_gap[time_index - 1])
            | _crosses_below(vwap_gap[time_index], vwap_gap[time_index - 1])
        ) if time_index > 0 else np.zeros(n_assets, dtype=bool)
        forward_return = path[-1]
        forward_drawdown = path.min(axis=0)
        buy[time_index] = bullish_crossover & (
            forward_return >= config.buy_min_return + config.round_trip_cost
        )
        hold[time_index] = (
            (ema20_gap[time_index] >= 0.0)
            | (vwap_gap[time_index] >= 0.0)
            | (ema50_gap[time_index] >= -config.ema50_epsilon * close[time_index])
        )
        sell[time_index] = bearish_crossover & (
            forward_drawdown <= -(config.sell_min_drawdown + config.round_trip_cost)
        )
    return ThreeMTargets(
        buy=buy,
        hold=hold,
        sell=sell,
        n_times=n_times,
        valid_mask=valid_mask,
    )
