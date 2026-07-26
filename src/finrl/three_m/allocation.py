"""Cash-aware 3M portfolio transition preserving held positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finrl.three_m.config import PolicyConfig
from finrl.three_m.policy import Action, ActionDecision


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Target allocation and deterministic outcomes for requested buys."""

    target_weights: np.ndarray
    admitted_buy_mask: np.ndarray
    rejected_buy_mask: np.ndarray


def allocate_actions(
    current_weights: np.ndarray,
    decision: ActionDecision,
    buy_probabilities: np.ndarray,
    config: PolicyConfig,
) -> AllocationResult:
    """Sell first, preserve holds, then fund ranked buys only from cash."""

    current = np.asarray(current_weights, dtype=np.float64)
    n_assets = decision.actions.shape[0]
    if current.shape != (n_assets + 1,) or not np.isfinite(current).all() or np.any(current < 0.0):
        raise ValueError("current_weights must be finite non-negative [assets + cash].")
    if not np.isclose(current.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("current_weights must sum to one.")
    probabilities = np.asarray(buy_probabilities, dtype=np.float64)
    if probabilities.shape != (n_assets,) or not np.isfinite(probabilities).all():
        raise ValueError("buy_probabilities must be finite with one value per risky asset.")
    actions = np.asarray(decision.actions, dtype=np.int8)
    if not np.isin(actions, [action.value for action in Action]).all():
        raise ValueError("actions contain an unknown action code.")
    target = current.copy()
    sell_mask = actions == Action.SELL
    target[-1] += target[:-1][sell_mask].sum()
    target[:-1][sell_mask] = 0.0
    requested_buy = actions == Action.BUY
    existing = target[:-1] > config.holding_epsilon
    requested_buy &= ~existing
    admitted = np.zeros(n_assets, dtype=bool)
    slots = config.max_positions - int(existing.sum())
    candidates = np.flatnonzero(requested_buy)
    # Stable asset-index tie breaking makes allocation reproducible.
    ranked = candidates[np.lexsort((candidates, -probabilities[candidates]))]
    for asset_index in ranked:
        if slots <= 0 or target[-1] + 1e-12 < config.entry_weight:
            continue
        target[asset_index] = config.entry_weight
        target[-1] -= config.entry_weight
        admitted[asset_index] = True
        slots -= 1
    rejected = requested_buy & ~admitted
    if np.any(target[:-1] < -1e-12) or target[-1] < -1e-12:
        raise RuntimeError("3M allocation created negative weights.")
    target = np.maximum(target, 0.0)
    if not np.isclose(target.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("3M allocation failed to conserve capital.")
    return AllocationResult(
        target_weights=target.astype(np.float32),
        admitted_buy_mask=admitted,
        rejected_buy_mask=rejected,
    )
