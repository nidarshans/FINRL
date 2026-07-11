"""Deterministic portfolio construction and trading-cost utilities."""

from __future__ import annotations

import numpy as np


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Standardize each time row using only contemporaneous assets."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("values must be a finite [time, assets] array.")
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    return np.divide(array - mean, std, out=np.zeros_like(array), where=std > 0.0).astype(np.float32)


def apply_position_cap(weights: np.ndarray, cap: float, cash_index: int = -1) -> np.ndarray:
    """Cap risky positions and redistribute excess weight deterministically."""

    values = np.asarray(weights, dtype=np.float64).copy()
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must be finite, non-negative, and two-dimensional.")
    if not 0.0 < cap <= 1.0:
        raise ValueError("cap must be in (0, 1].")
    risky = [i for i in range(values.shape[1]) if i != (values.shape[1] + cash_index if cash_index < 0 else cash_index)]
    for row in values:
        row[risky] = np.minimum(row[risky], cap)
        excess = 1.0 - row.sum()
        if excess > 0.0:
            eligible = [i for i in risky if row[i] < cap - 1e-12]
            if eligible:
                row[eligible] += excess / len(eligible)
            else:
                row[cash_index] += excess
        row[:] /= row.sum()
    return values.astype(np.float32)


def smooth_target_weights(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Blend targets toward current holdings to reduce unnecessary turnover."""

    current_array = np.asarray(current, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if current_array.shape != target_array.shape or current_array.ndim != 2:
        raise ValueError("current and target must have matching [time, assets] shapes.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if not np.isfinite(current_array).all() or not np.isfinite(target_array).all():
        raise ValueError("current and target must be finite.")
    blended = (1.0 - alpha) * current_array + alpha * target_array
    return (blended / blended.sum(axis=1, keepdims=True)).astype(np.float32)


def shrink_covariance(returns: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Return a positive-definite covariance estimate shrunk to diagonal."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite [time, assets] array with at least two rows.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(covariance))
    return ((1.0 - shrinkage) * covariance + shrinkage * diagonal).astype(np.float32)


def estimate_liquidity_cost(participation: np.ndarray, spread_bps: np.ndarray, impact_bps: float = 10.0) -> np.ndarray:
    """Estimate per-period cost in decimal units from participation and spread."""

    participation_array = np.asarray(participation, dtype=np.float64)
    spread_array = np.asarray(spread_bps, dtype=np.float64)
    if participation_array.shape != spread_array.shape or np.any(participation_array < 0.0) or np.any(spread_array < 0.0):
        raise ValueError("participation and spread_bps must be matching non-negative arrays.")
    if impact_bps < 0.0:
        raise ValueError("impact_bps must be non-negative.")
    return (spread_array / 10_000.0 + impact_bps / 10_000.0 * np.sqrt(participation_array)).astype(np.float32)
