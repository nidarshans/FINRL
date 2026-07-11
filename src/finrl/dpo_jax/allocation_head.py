"""Direct allocation head for differentiable portfolio optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from finrl.dpo_jax.activations import sparsemax
from finrl.types import Array


class DirectAllocationHead(nn.Module):
    """Map each asset's routed features to jointly normalized weights."""

    hidden_dims: tuple[int, ...] = ()
    simplex_activation: str = "softmax"
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, asset_features: Array, tradable_mask: Array | None = None) -> Array:
        """Return long-only target weights with cash in the final column."""

        features = jnp.asarray(asset_features, dtype=jnp.float32)
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("Every allocation hidden dimension must be positive.")
        if self.output_activation != "identity":
            raise ValueError(
                "Allocation output activation must be 'identity' so logits remain unrestricted."
            )

        x = features
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"hidden_norm_{index}")(x)
            x = _activation(self.hidden_activation)(x)
        stock_logits = nn.Dense(1, name="stock_logits")(x).squeeze(axis=-1)
        stock_logits = _activation(self.output_activation)(stock_logits)

        if tradable_mask is None:
            tradable = jnp.ones_like(stock_logits, dtype=bool)
        else:
            tradable = jnp.asarray(tradable_mask, dtype=bool)
            if tradable.shape != stock_logits.shape:
                raise ValueError("tradable_mask must match [time, assets].")
        valid_count = jnp.sum(tradable, axis=-1, keepdims=True)
        masked_logits = jnp.where(tradable, stock_logits, -jnp.inf)

        if self.simplex_activation == "softmax":
            stock_weights = jax.nn.softmax(masked_logits, axis=-1)
        elif self.simplex_activation == "sparsemax":
            stock_weights = sparsemax(masked_logits, axis=-1)
        else:
            raise ValueError(f"Unknown simplex activation: {self.simplex_activation}")
        stock_weights = jnp.where(valid_count > 0, stock_weights, 0.0)
        cash_weight = jnp.where(
            valid_count > 0,
            0.0,
            1.0,
        ).astype(stock_weights.dtype)
        cash_weight = cash_weight.reshape(*stock_weights.shape[:-1], 1)
        # Keep the cash fallback shape explicit for both batched and unbatched use.
        cash_weight = jnp.where(
            valid_count > 0,
            jnp.zeros_like(cash_weight),
            jnp.ones_like(cash_weight),
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
