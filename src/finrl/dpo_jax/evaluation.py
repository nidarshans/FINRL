"""Evaluation helpers for direct portfolio optimization."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.dpo_jax.losses import DPOLossMetrics, dpo_loss
from finrl.dpo_jax.policy import build_score_allocation_policy
from finrl.dpo_jax.trainer import DPOBatch, DPOTrainState
from finrl.types import Array


def predict_weights(state: DPOTrainState, batch: DPOBatch) -> Array:
    """Return direct allocation weights for a batch."""

    policy = build_score_allocation_policy(
        state.config,
        state.accumulation_indices,
        state.liquidity_indices,
    )
    return policy.apply({"params": state.policy.params}, batch.asset_features)


def evaluate_dpo(state: DPOTrainState, batch: DPOBatch) -> tuple[Array, DPOLossMetrics]:
    """Evaluate the DPO objective without applying optimizer updates."""

    weights = predict_weights(state, batch)
    return dpo_loss(
        weights,
        jnp.asarray(batch.asset_returns, dtype=jnp.float32),
        batch.initial_weights,
        state.config,
    )
