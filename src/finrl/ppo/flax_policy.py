"""Production Flax actor boundary for portfolio PPO."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn

from finrl.types import Array


@dataclass(frozen=True, slots=True)
class ProductionPPOConfig:
    """Configuration for production Flax PPO actor-critic models."""

    phi_dim: int = 32
    n_regimes: int = 4
    n_assets: int = 101
    actor_hidden_dims: tuple[int, int] = (128, 128)
    critic_hidden_dims: tuple[int, int] = (128, 64)
    temperature: float = 1.0
    dirichlet_concentration: float = 50.0
    min_concentration: float = 1e-3

    @property
    def state_dim(self) -> int:
        """Return `phi + regimes + weights + drawdown + previous turnover`."""

        return self.phi_dim + self.n_regimes + self.n_assets + 2

    def __post_init__(self) -> None:
        if self.phi_dim <= 0 or self.n_regimes <= 0 or self.n_assets <= 0:
            raise ValueError("PPO dimensions must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.dirichlet_concentration <= 0.0:
            raise ValueError("dirichlet_concentration must be positive.")
        if self.min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive.")


class PortfolioActorFlax(nn.Module):
    """Flax actor that maps PPO state vectors to allocation logits."""

    config: ProductionPPOConfig

    @nn.compact
    def __call__(self, state: Array) -> Array:
        """Return logits for `N + 1` tradable assets."""

        x = state
        for index, hidden_dim in enumerate(self.config.actor_hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{index}")(x)
            x = jnp.tanh(x)
        return nn.Dense(self.config.n_assets, name="logits")(x)


def actor_mean_weights(logits: Array, config: ProductionPPOConfig) -> Array:
    """Return deterministic mean allocation for evaluation."""

    return nn.softmax(logits / config.temperature, axis=-1)

