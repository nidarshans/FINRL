"""Deterministic per-asset 3M action eligibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from finrl.three_m.config import PolicyConfig
from finrl.three_m.model import ThreeMProbabilities


class Action(IntEnum):
    """Allowed asset actions at one decision time."""

    FLAT = 0
    BUY = 1
    HOLD = 2
    SELL = 3


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """Per-asset actions after state and tradability eligibility checks."""

    actions: np.ndarray
    held_mask: np.ndarray


def decide_actions(
    current_weights: np.ndarray,
    probabilities: ThreeMProbabilities,
    config: PolicyConfig,
    tradable_mask: np.ndarray | None = None,
) -> ActionDecision:
    """Gate pooled probabilities by actual holdings without changing weights."""

    weights = np.asarray(current_weights, dtype=np.float64)
    n_assets = probabilities.buy.shape[0]
    if weights.shape != (n_assets + 1,) or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("current_weights must be finite non-negative [assets + cash].")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("current_weights must sum to one.")
    probability_arrays = (probabilities.buy, probabilities.hold, probabilities.sell)
    if any(np.asarray(values).shape != (n_assets,) for values in probability_arrays):
        raise ValueError("Probability arrays must be matching one-dimensional asset vectors.")
    if any(not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)) for values in probability_arrays):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    tradable = np.ones(n_assets, dtype=bool) if tradable_mask is None else np.asarray(tradable_mask, dtype=bool)
    if tradable.shape != (n_assets,):
        raise ValueError("tradable_mask must have one value per risky asset.")
    held = weights[:-1] > config.holding_epsilon
    actions = np.full(n_assets, Action.FLAT, dtype=np.int8)
    actions[held] = Action.HOLD
    buy_eligible = ~held & tradable & (probabilities.buy >= config.buy_threshold)
    actions[buy_eligible] = Action.BUY
    sell_condition = held & tradable & (
        (probabilities.sell >= config.sell_threshold)
        | (probabilities.hold < config.hold_threshold)
    )
    actions[sell_condition] = Action.SELL
    return ActionDecision(actions=actions, held_mask=held)
