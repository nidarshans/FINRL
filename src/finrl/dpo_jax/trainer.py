"""JAX trainer for direct-feature portfolio optimization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from finrl.dpo_jax.config import DPOConfig
from finrl.dpo_jax.losses import DPOLossMetrics, dpo_loss
from finrl.dpo_jax.policy import build_allocation_policy
from finrl.types import Array


class DPOBatch(NamedTuple):
    """One differentiable backtest batch."""

    asset_features: Array
    asset_returns: Array
    spy_returns: Array
    previous_weights: Array
    drawdowns: Array
    previous_turnovers: Array
    initial_weights: Array


@struct.dataclass
class DPOTrainState:
    """Optimizer state for the direct allocation policy."""

    policy: TrainState
    config: DPOConfig = struct.field(pytree_node=False)
    direct_feature_indices: tuple[int, ...] = struct.field(pytree_node=False)


def initialize_dpo_train_state(
    rng: Array,
    config: DPOConfig,
    n_assets: int,
    asset_feature_dim: int,
    direct_feature_indices: tuple[int, ...],
) -> DPOTrainState:
    """Initialize direct allocation policy parameters."""

    policy = build_allocation_policy(
        config,
        direct_feature_indices,
    )
    example_features = jnp.zeros(
        (1, n_assets, asset_feature_dim),
        dtype=jnp.float32,
    )
    params = policy.init(rng, example_features)["params"]
    return DPOTrainState(
        policy=TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params=params,
            tx=optax.adam(config.learning_rate),
        ),
        config=config,
        direct_feature_indices=direct_feature_indices,
    )


def build_dpo_batch(
    asset_features: Array,
    asset_returns: Array,
    initial_weights: Array | None = None,
    spy_returns: Array | None = None,
) -> DPOBatch:
    """Build a cash-initialized batch from decision-date features and returns."""

    features = jnp.asarray(asset_features, dtype=jnp.float32)
    returns = jnp.asarray(asset_returns, dtype=jnp.float32)
    benchmark_returns = (
        jnp.zeros((returns.shape[0],), dtype=jnp.float32)
        if spy_returns is None
        else jnp.asarray(spy_returns, dtype=jnp.float32)
    )
    if features.ndim != 3:
        raise ValueError("asset_features must have shape [time, asset, feature].")
    if returns.ndim != 2:
        raise ValueError("asset_returns must have shape [time, asset].")
    if features.shape[:2] != returns.shape:
        raise ValueError("Asset feature and return time/asset dimensions must match.")
    if benchmark_returns.shape != (returns.shape[0],):
        raise ValueError("spy_returns must have shape [time].")
    n_steps = features.shape[0]
    n_assets = returns.shape[-1]
    if initial_weights is None:
        initial = jnp.zeros((n_assets + 1,), dtype=jnp.float32).at[-1].set(1.0)
    else:
        initial = jnp.asarray(initial_weights, dtype=jnp.float32)
    if initial.shape != (n_assets + 1,):
        raise ValueError("initial_weights must contain one weight per asset plus cash.")
    previous_weights = jnp.broadcast_to(initial, (n_steps, n_assets + 1))
    return DPOBatch(
        asset_features=features,
        asset_returns=returns,
        spy_returns=benchmark_returns,
        previous_weights=previous_weights,
        drawdowns=jnp.zeros((n_steps, 1), dtype=jnp.float32),
        previous_turnovers=jnp.zeros((n_steps, 1), dtype=jnp.float32),
        initial_weights=initial,
    )


@jax.jit
def train_step(state: DPOTrainState, batch: DPOBatch) -> tuple[DPOTrainState, DPOLossMetrics]:
    """Run one differentiable portfolio optimization update."""

    def loss_fn(params: dict[str, object]) -> tuple[Array, DPOLossMetrics]:
        policy = build_allocation_policy(
            state.config,
            state.direct_feature_indices,
        )
        weights = policy.apply({"params": params}, batch.asset_features)
        return dpo_loss(
            weights,
            batch.asset_returns,
            batch.initial_weights,
            state.config,
            batch.spy_returns,
        )

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
        direct_feature_indices=state.direct_feature_indices,
    ), metrics


def train_dpo(
    state: DPOTrainState,
    batch: DPOBatch,
) -> tuple[DPOTrainState, tuple[DPOLossMetrics, ...]]:
    """Fit DPO over one complete chronological backtest per epoch.

    Equity, running peak, drifted holdings, and their gradients must remain on
    one scan path. Splitting optimizer updates into time chunks would reset or
    truncate that state and optimize a different objective.
    """

    metrics: list[DPOLossMetrics] = []
    current = state
    if int(batch.asset_features.shape[0]) == 0:
        raise ValueError("DPO training requires at least one time step.")

    for _ in range(state.config.num_epochs):
        current, _ = train_step(current, batch)
        epoch_metrics = _evaluate_metrics(current, batch)
        metrics.append(epoch_metrics)
    return current, tuple(metrics)


def _predict_weights(state: DPOTrainState, asset_features: Array) -> Array:
    """Predict weights without introducing an evaluation-module import cycle."""

    policy = build_allocation_policy(state.config, state.direct_feature_indices)
    return policy.apply({"params": state.policy.params}, asset_features)


def _evaluate_metrics(state: DPOTrainState, batch: DPOBatch) -> DPOLossMetrics:
    """Evaluate epoch metrics over the complete training path."""

    weights = _predict_weights(state, batch.asset_features)
    _, metrics = dpo_loss(
        weights,
        batch.asset_returns,
        batch.initial_weights,
        state.config,
        batch.spy_returns,
    )
    return metrics
