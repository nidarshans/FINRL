"""Self-supervised losses for production encoder pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.models.flax_encoder import MarketEncoderFlax, ProductionEncoderConfig
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class EncoderLossWeights:
    """Weights for the production encoder pretraining objective."""

    market: float = 1.0
    volatility: float = 0.25
    cross_sectional: float = 0.5
    l2: float = 1e-5
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.market < 0.0 or self.volatility < 0.0 or self.cross_sectional < 0.0:
            raise ValueError("prediction loss weights must be non-negative.")
        if self.l2 < 0.0:
            raise ValueError("l2 loss weight must be non-negative.")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")


class EncoderPredictionHeads(nn.Module):
    """Prediction heads trained on top of the pooled market vector."""

    n_assets: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, market_vector: Array) -> dict[str, Array]:
        """Predict market, volatility, and cross-sectional next-return labels."""

        hidden = nn.Dense(self.hidden_dim, name="hidden")(market_vector)
        hidden = nn.gelu(hidden)
        market = nn.Dense(1, name="market_return")(hidden).squeeze(axis=-1)
        volatility = nn.Dense(1, name="volatility")(hidden).squeeze(axis=-1)
        cross_sectional = nn.Dense(self.n_assets, name="cross_sectional")(hidden)
        return {
            "market_return": market,
            "volatility": volatility,
            "cross_sectional": cross_sectional,
        }


def huber_loss(prediction: Array, target: Array, delta: float = 1.0) -> Array:
    """Return elementwise Huber loss."""

    error = prediction - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


def l2_penalty(params: Mapping[str, object]) -> Array:
    """Return squared L2 norm over floating point parameter leaves."""

    leaves = jax.tree.leaves(params)
    penalties = [
        jnp.sum(jnp.square(leaf))
        for leaf in leaves
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    ]
    if not penalties:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sum(jnp.stack(penalties))


def encoder_loss(
    params: Mapping[str, object],
    batch: object,
    encoder_config: ProductionEncoderConfig,
    loss_weights: EncoderLossWeights,
) -> tuple[Array, dict[str, Array]]:
    """Compute the Phase C multi-task encoder pretraining loss.

    The objective predicts labels at ``t + horizon`` from a feature window that
    ends at ``t``. Batch construction is responsible for ensuring those labels
    remain inside the train split.
    """

    encoder = MarketEncoderFlax(encoder_config)
    heads = EncoderPredictionHeads(
        n_assets=encoder_config.n_assets,
        hidden_dim=encoder_config.asset_hidden_dim,
    )

    market_vectors = jax.vmap(
        lambda asset_window, macro_window, spectral_row: encoder.apply(
            {"params": params["encoder"]},
            asset_window,
            macro_window,
            spectral_row,
        )
    )(batch.asset_window, batch.macro_window, batch.spectral_row)
    predictions = heads.apply({"params": params["heads"]}, market_vectors)

    market_loss = jnp.mean(
        huber_loss(
            predictions["market_return"],
            batch.market_return_target,
            loss_weights.huber_delta,
        )
    )
    volatility_loss = jnp.mean(
        huber_loss(
            predictions["volatility"],
            batch.volatility_target,
            loss_weights.huber_delta,
        )
    )
    cross_sectional_loss = jnp.mean(
        huber_loss(
            predictions["cross_sectional"],
            batch.cross_sectional_return_target,
            loss_weights.huber_delta,
        )
    )
    regularization = l2_penalty(params)
    total = (
        loss_weights.market * market_loss
        + loss_weights.volatility * volatility_loss
        + loss_weights.cross_sectional * cross_sectional_loss
        + loss_weights.l2 * regularization
    )
    metrics = {
        "loss": total,
        "market_loss": market_loss,
        "volatility_loss": volatility_loss,
        "cross_sectional_loss": cross_sectional_loss,
        "l2_penalty": regularization,
    }
    return total, metrics
