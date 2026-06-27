"""Direct allocation head for differentiable portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.dpo_jax.activations import sparsemax
from finrl.types import Array


class DirectAllocationHead(nn.Module):
    """Map each asset's scores independently to jointly normalized weights."""

    hidden_dims: tuple[int, ...] = ()
    simplex_activation: str = "softmax"
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, asset_scores: Array) -> Array:
        """Return long-only target weights with cash in the final column."""

        scores = jnp.asarray(asset_scores, dtype=jnp.float32)
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("Every allocation hidden dimension must be positive.")

        x = scores
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"hidden_norm_{index}")(x)
            x = _activation(self.hidden_activation)(x)
        stock_logits = nn.Dense(1, name="stock_logits")(x).squeeze(axis=-1)
        stock_logits = _activation(self.output_activation)(stock_logits)

        if self.simplex_activation == "softmax":
            stock_weights = jax.nn.softmax(stock_logits, axis=-1)
        elif self.simplex_activation == "sparsemax":
            stock_weights = sparsemax(stock_logits, axis=-1)
        else:
            raise ValueError(f"Unknown simplex activation: {self.simplex_activation}")
        cash_weight = jnp.zeros(
            (*stock_weights.shape[:-1], 1),
            dtype=stock_weights.dtype,
        )
        return jnp.concatenate([stock_weights, cash_weight], axis=-1)


def _activation(name: str):
    if name == "identity":
        return lambda value: value
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "tanh":
        return jnp.tanh
    if name == "gelu":
        return nn.gelu
    if name == "relu":
        return nn.relu
    raise ValueError(f"Unknown allocation activation function: {name}")
