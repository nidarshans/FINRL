"""Direct allocation head for differentiable portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.types import Array


class DirectAllocationHead(nn.Module):
    """Map per-asset embeddings and portfolio context to simplex weights."""

    hidden_dim: int = 32

    @nn.compact
    def __call__(
        self,
        asset_embeddings: Array,
        previous_weights: Array,
        drawdown: Array,
        previous_turnover: Array,
    ) -> Array:
        """Return long-only target weights with cash in the final column."""

        embeddings = jnp.asarray(asset_embeddings, dtype=jnp.float32)
        asset_hidden = nn.Dense(self.hidden_dim, name="asset_hidden")(embeddings)
        asset_hidden = jnp.tanh(asset_hidden)
        stock_logits = nn.Dense(1, name="stock_logits")(asset_hidden).squeeze(axis=-1)

        portfolio_state = jnp.concatenate(
            [
                jnp.asarray(previous_weights, dtype=jnp.float32),
                _as_column(drawdown),
                _as_column(previous_turnover),
            ],
            axis=-1,
        )
        cash_hidden = nn.Dense(self.hidden_dim, name="cash_hidden")(portfolio_state)
        cash_hidden = jnp.tanh(cash_hidden)
        cash_logit = nn.Dense(1, name="cash_logit")(cash_hidden)

        logits = jnp.concatenate([stock_logits, cash_logit], axis=-1)
        return jax.nn.softmax(logits, axis=-1)


def _as_column(value: Array) -> Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.ndim == 1:
        return array[:, None]
    return array
