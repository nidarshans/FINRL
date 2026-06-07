"""Scalar diagnostics for production encoder pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from finrl.types import Array


@dataclass(frozen=True, slots=True)
class EncoderTrainMetrics:
    """Scalar metrics emitted by one encoder pretraining step or epoch."""

    loss: Array
    market_loss: Array
    volatility_loss: Array
    cross_sectional_loss: Array
    l2_penalty: Array
    grad_norm: Array


def encoder_metrics_to_dict(metrics: EncoderTrainMetrics) -> dict[str, Array]:
    """Return TensorBoard-friendly scalar encoder metrics."""

    return {
        "loss": metrics.loss,
        "market_loss": metrics.market_loss,
        "volatility_loss": metrics.volatility_loss,
        "cross_sectional_loss": metrics.cross_sectional_loss,
        "l2_penalty": metrics.l2_penalty,
        "grad_norm": metrics.grad_norm,
    }


def finite_encoder_metrics(metrics: EncoderTrainMetrics) -> Array:
    """Return whether all encoder diagnostics are finite scalars."""

    values = jnp.asarray(list(encoder_metrics_to_dict(metrics).values()))
    return jnp.all(jnp.isfinite(values))
