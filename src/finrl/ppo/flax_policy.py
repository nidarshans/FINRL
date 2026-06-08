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

    phi_dim: int = 64
    asset_latent_dim: int = 64
    macro_dim: int = 16
    spectral_dim: int = 20
    n_regimes: int = 4
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
        """Return explicit context plus previous portfolio dimensions."""

        return (
            self.phi_dim
            + self.macro_dim
            + self.spectral_dim
            + self.n_regimes
            + self.action_dim
            + 2
        )

    def __post_init__(self) -> None:
        if self.phi_dim <= 0 or self.n_regimes <= 0 or self.n_assets <= 0:
            raise ValueError("PPO dimensions must be positive.")
        if self.asset_latent_dim <= 0 or self.macro_dim <= 0 or self.spectral_dim <= 0:
            raise ValueError("latent and context dimensions must be positive.")
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
    """Structured per-asset PPO state.

    ``asset_embeddings`` excludes cash and has shape ``[n_assets - 1, D]``.
    ``prev_weights`` includes cash and has shape ``[n_assets]``.
    """

    asset_embeddings: Array
    market_vector: Array
    macro_state: Array
    spectral_state: Array
    regime_probs: Array
    prev_weights: Array
    drawdown: Array
    prev_turnover: Array


class PortfolioActorFlax(nn.Module):
    """Context-aware shared per-asset allocation scorer."""

    config: ProductionPPOConfig

    @nn.compact
    def __call__(self, state: PPOState) -> Array:
        """Return logits for `N + 1` tradable assets."""

        n_risky_assets = self.config.n_assets - 1
        context = build_allocation_context(
            state.market_vector,
            state.macro_state,
            state.spectral_state,
            state.regime_probs,
            state.drawdown,
            state.prev_turnover,
        )
        repeated_context = jnp.broadcast_to(context, (n_risky_assets, context.shape[-1]))
        asset_prev_weights = state.prev_weights[:n_risky_assets, None]
        asset_inputs = jnp.concatenate(
            [
                state.asset_embeddings,
                repeated_context,
                asset_prev_weights,
            ],
            axis=-1,
        )
        x = asset_inputs
        for index, hidden_dim in enumerate(self.config.actor_hidden_dims):
            x = nn.Dense(hidden_dim, name=f"shared_asset_hidden_{index}")(x)
            x = jnp.tanh(x)
        asset_scores = nn.Dense(1, name="shared_asset_score")(x).squeeze(axis=-1)

        cash_input = jnp.concatenate(
            [
                context,
                jnp.atleast_1d(state.prev_weights[-1]),
            ],
            axis=-1,
        )
        cash = cash_input
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
    market_vector: Array,
    macro_state: Array,
    spectral_state: Array,
    regime_probs: Array,
    prev_weights: Array,
    drawdown: Array,
    previous_turnover: Array,
) -> PPOState:
    """Build structured per-asset PPO state from market and portfolio context."""

    return PPOState(
        asset_embeddings=jnp.asarray(asset_embeddings, dtype=jnp.float32),
        market_vector=jnp.asarray(market_vector, dtype=jnp.float32),
        macro_state=jnp.asarray(macro_state, dtype=jnp.float32),
        spectral_state=jnp.asarray(spectral_state, dtype=jnp.float32),
        regime_probs=jnp.asarray(regime_probs, dtype=jnp.float32),
        prev_weights=jnp.asarray(prev_weights, dtype=jnp.float32),
        drawdown=jnp.asarray(drawdown, dtype=jnp.float32),
        prev_turnover=jnp.asarray(previous_turnover, dtype=jnp.float32),
    )


def build_allocation_context(
    market_vector: Array,
    macro_state: Array,
    spectral_state: Array,
    regime_probs: Array,
    drawdown: Array,
    previous_turnover: Array,
) -> Array:
    """Concatenate the global allocation context from the architecture diagram."""

    return jnp.concatenate(
        [
            jnp.asarray(market_vector, dtype=jnp.float32),
            jnp.asarray(macro_state, dtype=jnp.float32),
            jnp.asarray(spectral_state, dtype=jnp.float32),
            jnp.asarray(regime_probs, dtype=jnp.float32),
            jnp.atleast_1d(jnp.asarray(drawdown, dtype=jnp.float32)),
            jnp.atleast_1d(jnp.asarray(previous_turnover, dtype=jnp.float32)),
        ],
        axis=-1,
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
