"""JAX critic network for PPO."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from finrl.ppo.policy import PPOConfig, Params, apply_mlp, init_mlp_params
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class PortfolioCritic:
    """Critic network mapping PPO state to scalar value."""

    config: PPOConfig

    def init(self, key: Array) -> Params:
        """Initialize critic parameters."""

        dims = (self.config.state_dim, *self.config.critic_hidden_dims, 1)
        return init_mlp_params(key, dims)

    def apply(self, params: Params, state: Array) -> Array:
        """Return scalar state value."""

        value = apply_mlp(params, state, len(self.config.critic_hidden_dims) + 1)
        return jnp.squeeze(value, axis=-1)

