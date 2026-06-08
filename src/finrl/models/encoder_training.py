"""Train-window-only pretraining for the production Flax encoder.

Phase C uses a multi-task self-supervised objective:

- predict equal-weight next-period market return,
- predict next-period cross-sectional normalized asset returns,
- predict next-period cross-sectional volatility.

The caller may pass full walk-forward arrays, but ``train_window_count`` is
applied before label construction. This keeps feature windows and their
``t + horizon`` labels strictly inside the train split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from finrl.logging.tensorboard import TensorBoardLogger
from finrl.models.encoder_metrics import (
    EncoderTrainMetrics,
    encoder_metrics_to_dict,
)
from finrl.models.encoder_losses import (
    EncoderLossWeights,
    EncoderPredictionHeads,
    encoder_loss,
)
from finrl.models.flax_encoder import MarketEncoderFlax, ProductionEncoderConfig
from finrl.models.windows import LookbackWindows
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class EncoderTrainingConfig:
    """Configuration for local-safe encoder pretraining."""

    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    label_horizon: int = 1
    shuffle_batches: bool = False
    loss_weights: EncoderLossWeights = EncoderLossWeights()

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.label_horizon <= 0:
            raise ValueError("label_horizon must be positive.")


@dataclass(frozen=True, slots=True)
class EncoderBatch:
    """One mini-batch for encoder pretraining."""

    asset_window: Array
    macro_window: Array
    spectral_row: Array
    market_return_target: Array
    volatility_target: Array
    cross_sectional_return_target: Array
    decision_dates: tuple[object, ...]
    label_dates: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class EncoderTrainingResult:
    """Fitted encoder pretraining artifacts."""

    train_state: TrainState
    metrics_by_epoch: tuple[dict[str, float], ...]


def _tree_global_norm(tree: object) -> Array:
    """Return the global L2 norm of a gradient pytree."""

    squared_norms = [
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(jnp.sum(jnp.asarray(squared_norms)))


def _validate_training_arrays(
    windows: LookbackWindows,
    realized_returns: np.ndarray,
    train_window_count: int,
    horizon: int,
) -> None:
    if realized_returns.ndim != 2:
        raise ValueError("realized_returns must have shape (n_windows, n_assets).")
    if realized_returns.shape[0] != windows.asset.shape[0]:
        raise ValueError("realized_returns must align with window decision dates.")
    if realized_returns.shape[1] != windows.asset.shape[2]:
        raise ValueError("realized_returns asset dimension must match windows.")
    if train_window_count > windows.asset.shape[0]:
        raise ValueError("train_window_count cannot exceed available windows.")
    if train_window_count <= horizon:
        raise ValueError("train_window_count must exceed label_horizon.")


def _make_targets(
    realized_returns: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build market, volatility, and normalized cross-sectional labels."""

    market_return = realized_returns.mean(axis=1).astype(np.float32, copy=False)
    centered = realized_returns - market_return[:, None]
    volatility = np.sqrt(np.mean(centered**2, axis=1)).astype(np.float32, copy=False)
    cross_sectional = centered / (volatility[:, None] + epsilon)
    return (
        market_return,
        volatility,
        cross_sectional.astype(np.float32, copy=False),
    )


def make_encoder_batches(
    windows: LookbackWindows,
    realized_returns: np.ndarray,
    training_config: EncoderTrainingConfig,
    train_window_count: int | None = None,
    rng: Array | None = None,
) -> tuple[EncoderBatch, ...]:
    """Create train-only encoder batches.

    ``realized_returns[i]`` is the asset return observed at
    ``windows.decision_dates[i]``. For a feature window at index ``i``, the
    prediction label is ``realized_returns[i + label_horizon]``. The last
    ``label_horizon`` train windows are omitted because their labels would sit
    outside the train split.
    """

    count = windows.asset.shape[0] if train_window_count is None else train_window_count
    horizon = training_config.label_horizon
    _validate_training_arrays(windows, realized_returns, count, horizon)

    input_indices = np.arange(0, count - horizon)
    label_indices = input_indices + horizon
    label_returns = realized_returns[label_indices].astype(np.float32, copy=False)
    market_target, volatility_target, cross_sectional_target = _make_targets(label_returns)

    if training_config.shuffle_batches:
        if rng is None:
            raise ValueError("rng is required when shuffle_batches is True.")
        order = np.asarray(jax.random.permutation(rng, input_indices.shape[0]))
        input_indices = input_indices[order]
        label_indices = label_indices[order]
        market_target = market_target[order]
        volatility_target = volatility_target[order]
        cross_sectional_target = cross_sectional_target[order]

    batches = []
    for start in range(0, input_indices.shape[0], training_config.batch_size):
        stop = min(start + training_config.batch_size, input_indices.shape[0])
        batch_inputs = input_indices[start:stop]
        batch_labels = label_indices[start:stop]
        batches.append(
            EncoderBatch(
                asset_window=jnp.asarray(windows.asset[batch_inputs], dtype=jnp.float32),
                macro_window=jnp.asarray(windows.macro[batch_inputs], dtype=jnp.float32),
                spectral_row=jnp.asarray(windows.spectral[batch_inputs], dtype=jnp.float32),
                market_return_target=jnp.asarray(market_target[start:stop], dtype=jnp.float32),
                volatility_target=jnp.asarray(volatility_target[start:stop], dtype=jnp.float32),
                cross_sectional_return_target=jnp.asarray(
                    cross_sectional_target[start:stop],
                    dtype=jnp.float32,
                ),
                decision_dates=tuple(windows.decision_dates[index] for index in batch_inputs),
                label_dates=tuple(windows.decision_dates[index] for index in batch_labels),
            )
        )
    return tuple(batches)


def init_encoder_pretraining_state(
    rng: Array,
    encoder_config: ProductionEncoderConfig,
    training_config: EncoderTrainingConfig,
) -> TrainState:
    encoder_key, heads_key = jax.random.split(rng)
    encoder = MarketEncoderFlax(encoder_config)
    heads = EncoderPredictionHeads(
        n_assets=encoder_config.n_assets,
        hidden_dim=encoder_config.asset_hidden_dim,
    )
    asset_window = jnp.zeros(
        (encoder_config.lookback, encoder_config.n_assets, encoder_config.asset_feature_dim),
        dtype=jnp.float32,
    )
    macro_window = jnp.zeros(
        (encoder_config.lookback, encoder_config.macro_feature_dim),
        dtype=jnp.float32,
    )
    spectral_row = jnp.zeros((encoder_config.spectral_feature_dim,), dtype=jnp.float32)
    market_vector = jnp.zeros((encoder_config.asset_hidden_dim,), dtype=jnp.float32)
    encoder_variables = encoder.init(
        encoder_key,
        asset_window,
        macro_window,
        spectral_row,
        method=MarketEncoderFlax.encode_with_latents,
    )
    head_variables = heads.init(heads_key, market_vector)
    params = {
        "encoder": encoder_variables["params"],
        "heads": head_variables["params"],
    }
    optimizer = optax.adam(training_config.learning_rate)
    return TrainState.create(apply_fn=encoder.apply, params=params, tx=optimizer)


def _mean_metrics(metrics: Sequence[EncoderTrainMetrics]) -> dict[str, float]:
    keys = encoder_metrics_to_dict(metrics[0]).keys()
    return {
        key: float(
            jnp.mean(
                jnp.asarray([
                    encoder_metrics_to_dict(metric)[key]
                    for metric in metrics
                ])
            )
        )
        for key in keys
    }


def train_encoder_epoch(
    train_state: TrainState,
    batches: Sequence[EncoderBatch],
    encoder_config: ProductionEncoderConfig,
    training_config: EncoderTrainingConfig,
    logger: TensorBoardLogger | None = None,
    epoch: int = 0,
) -> tuple[TrainState, dict[str, float]]:
    """Run one encoder pretraining epoch over prepared train-only batches."""

    if not batches:
        raise ValueError("At least one encoder batch is required.")

    batch_metrics = []
    state = train_state
    for batch in batches:
        def loss_fn(params: dict[str, object]) -> tuple[Array, dict[str, Array]]:
            return encoder_loss(
                params,
                batch,
                encoder_config,
                training_config.loss_weights,
            )

        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_value, metrics), grads = loss_and_grad(
            state.params,
        )
        del loss_value
        metrics = EncoderTrainMetrics(
            loss=metrics["loss"],
            market_loss=metrics["market_loss"],
            volatility_loss=metrics["volatility_loss"],
            cross_sectional_loss=metrics["cross_sectional_loss"],
            l2_penalty=metrics["l2_penalty"],
            grad_norm=_tree_global_norm(grads),
        )
        state = state.apply_gradients(grads=grads)
        batch_metrics.append(metrics)
    epoch_metrics = _mean_metrics(batch_metrics)
    if logger is not None:
        logger.log_scalars(epoch_metrics, epoch, "encoder")
    return state, epoch_metrics


def fit_encoder_on_train_split(
    rng: Array,
    windows: LookbackWindows,
    realized_returns: np.ndarray,
    encoder_config: ProductionEncoderConfig,
    training_config: EncoderTrainingConfig,
    train_window_count: int | None = None,
    logger: TensorBoardLogger | None = None,
) -> EncoderTrainingResult:
    """Fit encoder pretraining heads using only the train split."""

    batches = make_encoder_batches(
        windows,
        realized_returns,
        training_config,
        train_window_count=train_window_count,
        rng=rng,
    )
    state = init_encoder_pretraining_state(rng, encoder_config, training_config)
    metrics = []
    for epoch in range(training_config.epochs):
        state, epoch_metrics = train_encoder_epoch(
            state,
            batches,
            encoder_config,
            training_config,
            logger,
            epoch,
        )
        metrics.append(epoch_metrics)
    return EncoderTrainingResult(train_state=state, metrics_by_epoch=tuple(metrics))
