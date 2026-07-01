"""JAX direct portfolio optimization components."""

from finrl.dpo_jax.activations import sparsemax
from finrl.dpo_jax.allocation_head import DirectAllocationHead
from finrl.dpo_jax.config import DPOConfig
from finrl.dpo_jax.evaluation import evaluate_dpo, predict_weights
from finrl.dpo_jax.losses import DPOLossMetrics, dpo_loss
from finrl.dpo_jax.policy import (
    DirectFeatureAllocationPolicy,
    build_allocation_policy,
    slice_direct_allocation_features,
)
from finrl.dpo_jax.trainer import (
    DPOBatch,
    DPOTrainState,
    build_dpo_batch,
    initialize_dpo_train_state,
    train_dpo,
    train_step,
)

__all__ = [
    "DPOBatch",
    "DPOConfig",
    "DPOLossMetrics",
    "DPOTrainState",
    "DirectAllocationHead",
    "DirectFeatureAllocationPolicy",
    "build_dpo_batch",
    "build_allocation_policy",
    "slice_direct_allocation_features",
    "dpo_loss",
    "evaluate_dpo",
    "initialize_dpo_train_state",
    "predict_weights",
    "sparsemax",
    "train_dpo",
    "train_step",
]
