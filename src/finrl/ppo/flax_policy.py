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

    asset_latent_dim: int = 64
    n_assets: int = 101
    actor_hidden_dims: tuple[int, int] = (128, 128)
    critic_hidden_dims: tuple[int, int] = (128, 64)
    temperature: float = 0.7
    dirichlet_concentration: float = 50.0
    min_concentration: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    portfolio_entropy_coef: float = 0.0
    learning_rate: float = 1e-3
    update_epochs: int = 2
    minibatch_size: int = 64
    target_kl: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    use_value_clipping: bool = True
    value_loss_type: str = "mse"
    value_huber_delta: float = 1.0

    @property
    def action_dim(self) -> int:
        """Return the number of allocation weights emitted by the actor."""

        return self.n_assets

    @property
    def state_dim(self) -> int:
        """Return the explicit PPO policy state dimension."""

        return self.asset_latent_dim

    def __post_init__(self) -> None:
        if self.n_assets <= 0:
            raise ValueError("PPO dimensions must be positive.")
        if self.asset_latent_dim <= 0:
            raise ValueError("asset_latent_dim must be positive.")
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
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.clip_epsilon <= 0.0 or self.value_clip_epsilon <= 0.0:
            raise ValueError("clip epsilons must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.portfolio_entropy_coef < 0.0:
            raise ValueError("portfolio_entropy_coef must be non-negative.")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive.")
        if self.target_kl <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("target_kl and max_grad_norm must be positive.")
        if self.value_loss_type not in {"mse", "huber"}:
            raise ValueError("value_loss_type must be 'mse' or 'huber'.")
        if self.value_huber_delta <= 0.0:
            raise ValueError("value_huber_delta must be positive.")


class ProductionPortfolioAction(NamedTuple):
    """Production Flax policy action and diagnostics."""

    weights: Array
    logits: Array
    log_prob: Array
    entropy: Array


class PPOState(NamedTuple):
    """Structured per-step PPO state.

    ``asset_embeddings`` excludes cash and has shape ``[n_assets - 1, D]``.
    Portfolio accounting state is intentionally excluded so each allocation
    decision depends only on the current asset window.
    """

    asset_embeddings: Array


class PortfolioActorFlax(nn.Module):
    """Shared per-asset allocation scorer independent of portfolio context."""

    config: ProductionPPOConfig

    @nn.compact
    def __call__(self, state: PPOState) -> Array:
        """Return logits for `N + 1` tradable assets."""

        x = state.asset_embeddings
        for index, hidden_dim in enumerate(self.config.actor_hidden_dims):
            x = nn.Dense(hidden_dim, name=f"shared_asset_hidden_{index}")(x)
            x = jnp.tanh(x)
        asset_scores = nn.Dense(1, name="shared_asset_score")(x).squeeze(axis=-1)

        pooled = jnp.concatenate(
            [jnp.mean(state.asset_embeddings, axis=0), jnp.max(state.asset_embeddings, axis=0)],
            axis=-1,
        )
        cash = pooled
        for index, hidden_dim in enumerate(self.config.actor_hidden_dims):
            cash = nn.Dense(hidden_dim, name=f"cash_hidden_{index}")(cash)
            cash = jnp.tanh(cash)
        cash_score = nn.Dense(1, name="cash_score")(cash).squeeze(axis=-1)
        return jnp.concatenate([asset_scores, jnp.atleast_1d(cash_score)], axis=-1)


def actor_mean_weights(logits: Array, config: ProductionPPOConfig) -> Array:
    """Return deterministic Dirichlet mean allocation for evaluation."""

    base = nn.softmax(logits / config.temperature, axis=-1)
    alpha = config.dirichlet_concentration * base + config.min_concentration
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def build_structured_ppo_state(
    asset_embeddings: Array,
    prev_weights: Array | None = None,
    drawdown: Array | None = None,
    previous_turnover: Array | None = None,
) -> PPOState:
    """Build a PPO state from asset embeddings only.

    The optional portfolio arguments are accepted for backward-compatible call
    sites that still carry accounting diagnostics through rollout collection.
    They are deliberately ignored by the actor and critic.
    """

    del prev_weights, drawdown, previous_turnover
    return PPOState(
        asset_embeddings=jnp.asarray(asset_embeddings, dtype=jnp.float32),
    )


def sample_action(
    variables: dict[str, object],
    state: PPOState,
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
