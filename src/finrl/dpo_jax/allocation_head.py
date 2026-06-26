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
    hidden_dims: tuple[int, ...] | None = None
    allocation_activation: str = "sparsemax"
    activation: str = "tanh"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, asset_embeddings: Array) -> Array:
        """Return long-only target weights with cash in the final column."""

        embeddings = jnp.asarray(asset_embeddings, dtype=jnp.float32)
        hidden_dims = self.hidden_dims if self.hidden_dims is not None else (self.hidden_dim,)
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("Every allocation hidden dimension must be positive.")

        x = embeddings
        for index, hidden_dim in enumerate(hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"hidden_norm_{index}")(x)
            x = _activation(self.activation)(x)
        stock_logits = nn.Dense(1, name="stock_logits")(x).squeeze(axis=-1)

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


def _activation(name: str):
    if name == "tanh":
        return jnp.tanh
    if name == "gelu":
        return nn.gelu
    if name == "relu":
        return nn.relu
    raise ValueError(f"Unknown allocation activation function: {name}")
