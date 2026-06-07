"""Production Flax actor boundary for portfolio PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from flax import linen as nn

from finrl.types import Array


@dataclass(frozen=True, slots=True)
class ProductionPPOConfig:
    """Configuration for production Flax PPO actor-critic models.

    ``n_assets`` is the action dimension, including cash. For a 100-stock
    universe plus cash, use ``n_assets=101``.
    """

    phi_dim: int = 32
    n_regimes: int = 4
    n_assets: int = 101
    actor_hidden_dims: tuple[int, int] = (128, 128)
    critic_hidden_dims: tuple[int, int] = (128, 64)
    temperature: float = 1.0
    dirichlet_concentration: float = 50.0
    min_concentration: float = 1e-3

    @property
    def action_dim(self) -> int:
        """Return the number of allocation weights emitted by the actor."""

        return self.n_assets

    @property
    def state_dim(self) -> int:
        """Return ``phi + regimes + weights + drawdown + previous turnover``."""

        return self.phi_dim + self.n_regimes + self.action_dim + 2

    def __post_init__(self) -> None:
        if self.phi_dim <= 0 or self.n_regimes <= 0 or self.n_assets <= 0:
            raise ValueError("PPO dimensions must be positive.")
        if not self.actor_hidden_dims or not self.critic_hidden_dims:
            raise ValueError("actor and critic hidden dimensions cannot be empty.")
        if any(hidden_dim <= 0 for hidden_dim in self.actor_hidden_dims):
            raise ValueError("actor hidden dimensions must be positive.")
        if any(hidden_dim <= 0 for hidden_dim in self.critic_hidden_dims):
            raise ValueError("critic hidden dimensions must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.dirichlet_concentration <= 0.0:
            raise ValueError("dirichlet_concentration must be positive.")
        if self.min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive.")


class ProductionPortfolioAction(NamedTuple):
    """Production Flax policy action and diagnostics."""

    weights: Array
    logits: Array
    log_prob: Array
    entropy: Array


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


def build_ppo_state(
    phi: Array,
    regime_probs: Array,
    weights: Array,
    drawdown: Array,
    previous_turnover: Array,
) -> Array:
    """Build production PPO state from market, regime, and portfolio context."""

    return jnp.concatenate(
        [
            jnp.asarray(phi, dtype=jnp.float32),
            jnp.asarray(regime_probs, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.atleast_1d(jnp.asarray(drawdown, dtype=jnp.float32)),
            jnp.atleast_1d(jnp.asarray(previous_turnover, dtype=jnp.float32)),
        ],
        axis=0,
    )


def sample_action(
    variables: dict[str, object],
    state: Array,
    rng: Array,
    config: ProductionPPOConfig,
    deterministic: bool = False,
) -> ProductionPortfolioAction:
    """Sample or deterministically evaluate a production Flax policy action."""

    from finrl.ppo.simplex_distribution import DirichletPortfolioDistribution

    logits = PortfolioActorFlax(config).apply(variables, state)
    distribution = DirichletPortfolioDistribution(logits=logits, config=config)
    weights = distribution.mean() if deterministic else distribution.sample(rng)
    return ProductionPortfolioAction(
        weights=weights,
        logits=logits,
        log_prob=distribution.log_prob(weights),
        entropy=distribution.entropy(),
    )
