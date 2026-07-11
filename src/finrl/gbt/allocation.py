"""Deterministic conversion of model scores to portfolio weights."""

from __future__ import annotations

import numpy as np

from finrl.gbt.config import GBTConfig
from finrl.types import Array
from finrl.portfolio_construction import apply_position_cap


def scores_to_weights(scores: Array, config: GBTConfig) -> Array:
    """Convert ``[time, assets]`` scores to long-only weights plus zero cash."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [time, assets].")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values.")
    logits = np.clip(values / config.temperature, -config.score_clip, config.score_clip)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    stock_weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    cash = np.zeros((values.shape[0], 1), dtype=stock_weights.dtype)
    output = np.concatenate((stock_weights, cash), axis=1)
    if config.max_position_weight is not None:
        output = apply_position_cap(output, config.max_position_weight)
    return output.astype(np.float32)
