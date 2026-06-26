"""Production Flax critic boundary for portfolio PPO."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from finrl.ppo.flax_policy import ProductionPPOConfig
from finrl.ppo.flax_policy import PPOState
from finrl.types import Array


class PortfolioCriticFlax(nn.Module):
    """Flax critic that values the current asset window only."""

    config: ProductionPPOConfig

    @nn.compact
    def __call__(self, state: PPOState) -> Array:
        """Return a scalar value estimate."""

        mean_pool = jnp.mean(state.asset_embeddings, axis=0)
        max_pool = jnp.max(state.asset_embeddings, axis=0)
        x = jnp.concatenate(
            [
                mean_pool,
                max_pool,
            ],
            axis=-1,
        )
        for index, hidden_dim in enumerate(self.config.critic_hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            x = jnp.tanh(x)
        value = nn.Dense(1, name="value")(x)
        return jnp.squeeze(value, axis=-1)
