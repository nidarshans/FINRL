"""Direct allocation head for differentiable portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.dpo_jax.activations import sparsemax
from finrl.types import Array


class DirectAllocationHead(nn.Module):
    """Map per-asset embeddings to sparse simplex portfolio weights."""

    hidden_dim: int = 32
    allocation_activation: str = "sparsemax"

    @nn.compact
    def __call__(self, asset_embeddings: Array) -> Array:
        """Return long-only target weights with cash in the final column."""

        embeddings = jnp.asarray(asset_embeddings, dtype=jnp.float32)
        asset_hidden = nn.Dense(self.hidden_dim, name="asset_hidden")(embeddings)
        asset_hidden = jnp.tanh(asset_hidden)
        stock_logits = nn.Dense(1, name="stock_logits")(asset_hidden).squeeze(axis=-1)

        cash_logit_param = self.param(
            "cash_logit",
            nn.initializers.zeros,
            (1,),
        )
        cash_logit = jnp.broadcast_to(cash_logit_param, (stock_logits.shape[0], 1))

        logits = jnp.concatenate([stock_logits, cash_logit], axis=-1)
        if self.allocation_activation == "softmax":
            return jax.nn.softmax(logits, axis=-1)
        if self.allocation_activation == "sparsemax":
            return sparsemax(logits, axis=-1)
        raise ValueError(f"Unknown allocation activation: {self.allocation_activation}")
