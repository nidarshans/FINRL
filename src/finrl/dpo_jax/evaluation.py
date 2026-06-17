"""Evaluation helpers for direct portfolio optimization."""

from __future__ import annotations

import jax.numpy as jnp

from finrl.dpo_jax.allocation_head import DirectAllocationHead
from finrl.dpo_jax.losses import DPOLossMetrics, dpo_loss
from finrl.dpo_jax.trainer import DPOBatch, DPOTrainState
from finrl.models.asset_encoder import AssetOnlyEncoder
from finrl.types import Array


def predict_weights(state: DPOTrainState, batch: DPOBatch) -> Array:
    """Return direct allocation weights for a batch."""

    encoder = AssetOnlyEncoder(
        state.encoder_config,
        accumulation_indices=state.accumulation_indices,
        liquidity_indices=state.liquidity_indices,
    )
    head = DirectAllocationHead(hidden_dim=state.head_hidden_dim)
    embeddings = encoder.apply({"params": state.policy.params["encoder"]}, batch.asset_windows)
    return head.apply(
        {"params": state.policy.params["allocation_head"]},
        embeddings,
        batch.previous_weights,
        batch.drawdowns,
        batch.previous_turnovers,
    )


def evaluate_dpo(state: DPOTrainState, batch: DPOBatch) -> tuple[Array, DPOLossMetrics]:
    """Evaluate the DPO objective without applying optimizer updates."""

    weights = predict_weights(state, batch)
    return dpo_loss(
        weights,
        jnp.asarray(batch.asset_returns, dtype=jnp.float32),
        batch.initial_weights,
        state.config,
    )
