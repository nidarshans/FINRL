"""Minimal JAX trainer for direct portfolio optimization."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from finrl.dpo_jax.allocation_head import DirectAllocationHead
from finrl.dpo_jax.config import DPOConfig
from finrl.dpo_jax.losses import DPOLossMetrics, dpo_loss
from finrl.models.asset_encoder import AssetOnlyEncoder, AssetOnlyEncoderConfig
from finrl.types import Array


class DPOBatch(NamedTuple):
    """One differentiable backtest batch."""

    asset_windows: Array
    asset_returns: Array
    previous_weights: Array
    drawdowns: Array
    previous_turnovers: Array
    initial_weights: Array


@struct.dataclass
class DPOTrainState:
    """Optimizer state for encoder and direct allocation head params."""

    policy: TrainState
    config: DPOConfig = struct.field(pytree_node=False)
    encoder_config: AssetOnlyEncoderConfig = struct.field(pytree_node=False)
    accumulation_indices: tuple[int, ...] = struct.field(pytree_node=False)
    liquidity_indices: tuple[int, ...] = struct.field(pytree_node=False)
    head_hidden_dim: int = struct.field(pytree_node=False, default=32)


def initialize_dpo_train_state(
    rng: Array,
    config: DPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
    head_hidden_dim: int = 32,
) -> DPOTrainState:
    """Initialize encoder and direct allocation head parameters."""

    encoder_key, head_key = jax.random.split(rng)
    dpo_encoder_config = replace(
        encoder_config,
        score_hidden_dims=config.dpo_score_hidden_dims,
        score_use_layer_norm=config.dpo_score_use_layer_norm,
        score_activation=config.dpo_activation,
    )
    encoder = AssetOnlyEncoder(
        dpo_encoder_config,
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
    )
    head = DirectAllocationHead(
        hidden_dims=config.dpo_allocation_hidden_dims,
        hidden_dim=head_hidden_dim,
        allocation_activation=config.allocation_activation,
        activation=config.dpo_activation,
        use_layer_norm=config.dpo_allocation_use_layer_norm,
    )
    example_windows = jnp.zeros(
        (
            1,
            dpo_encoder_config.lookback,
            dpo_encoder_config.n_assets,
            dpo_encoder_config.asset_feature_dim,
        ),
        dtype=jnp.float32,
    )
    encoder_params = encoder.init(encoder_key, example_windows)["params"]
    example_embeddings = encoder.apply({"params": encoder_params}, example_windows)
    params = {
        "encoder": encoder_params,
        "allocation_head": head.init(
            head_key,
            example_embeddings,
        )["params"],
    }
    return DPOTrainState(
        policy=TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params=params,
            tx=optax.adam(config.learning_rate),
        ),
        config=config,
        encoder_config=dpo_encoder_config,
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
        head_hidden_dim=head_hidden_dim,
    )


def build_dpo_batch(
    asset_windows: Array,
    asset_returns: Array,
    initial_weights: Array | None = None,
) -> DPOBatch:
    """Build a simple cash-initialized batch from windows and next returns."""

    windows = jnp.asarray(asset_windows, dtype=jnp.float32)
    returns = jnp.asarray(asset_returns, dtype=jnp.float32)
    n_steps = windows.shape[0]
    n_assets = returns.shape[-1]
    if initial_weights is None:
        initial = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[-1].set(1.0)
    else:
        initial = jnp.asarray(initial_weights, dtype=jnp.float32)
    previous_weights = jnp.broadcast_to(initial, (n_steps, n_assets + 1))
    return DPOBatch(
        asset_windows=windows,
        asset_returns=returns,
        previous_weights=previous_weights,
        drawdowns=jnp.zeros((n_steps, 1), dtype=jnp.float32),
        previous_turnovers=jnp.zeros((n_steps, 1), dtype=jnp.float32),
        initial_weights=initial,
    )


@jax.jit
def train_step(state: DPOTrainState, batch: DPOBatch) -> tuple[DPOTrainState, DPOLossMetrics]:
    """Run one differentiable portfolio optimization update."""

    def loss_fn(params: dict[str, object]) -> tuple[Array, DPOLossMetrics]:
        encoder = AssetOnlyEncoder(
            state.encoder_config,
            accumulation_indices=state.accumulation_indices,
            liquidity_indices=state.liquidity_indices,
        )
        head = DirectAllocationHead(
            hidden_dims=state.config.dpo_allocation_hidden_dims,
            hidden_dim=state.head_hidden_dim,
            allocation_activation=state.config.allocation_activation,
            activation=state.config.dpo_activation,
            use_layer_norm=state.config.dpo_allocation_use_layer_norm,
        )
        embeddings = encoder.apply({"params": params["encoder"]}, batch.asset_windows)
        weights = head.apply(
            {"params": params["allocation_head"]},
            embeddings,
        )
        return dpo_loss(weights, batch.asset_returns, batch.initial_weights, state.config)

    (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.policy.params
    )
    updates, new_opt_state = state.policy.tx.update(
        grads,
        state.policy.opt_state,
        state.policy.params,
    )
    new_policy = state.policy.replace(
        params=optax.apply_updates(state.policy.params, updates),
        opt_state=new_opt_state,
    )
    return DPOTrainState(
        policy=new_policy,
        config=state.config,
        encoder_config=state.encoder_config,
        accumulation_indices=state.accumulation_indices,
        liquidity_indices=state.liquidity_indices,
        head_hidden_dim=state.head_hidden_dim,
    ), metrics


def train_dpo(
    state: DPOTrainState,
    batch: DPOBatch,
) -> tuple[DPOTrainState, tuple[DPOLossMetrics, ...]]:
    """Fit DPO for ``state.config.num_epochs`` over a single batch."""

    metrics: list[DPOLossMetrics] = []
    current = state
    for _ in range(state.config.num_epochs):
        current, epoch_metrics = train_step(current, batch)
        metrics.append(epoch_metrics)
    return current, tuple(metrics)
