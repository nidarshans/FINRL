"""Production Flax critic boundary for portfolio PPO."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from finrl.ppo.flax_policy import ProductionPPOConfig
from finrl.types import Array


class PortfolioCriticFlax(nn.Module):
    """Flax critic that maps PPO state vectors to scalar values."""

    config: ProductionPPOConfig

    @nn.compact
    def __call__(self, state: Array) -> Array:
        """Return a scalar value estimate."""

        x = state
        for index, hidden_dim in enumerate(self.config.critic_hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            x = jnp.tanh(x)
        value = nn.Dense(1, name="value")(x)
        return jnp.squeeze(value, axis=-1)

