"""Production Dirichlet distribution on the portfolio simplex."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln

from finrl.ppo.flax_policy import ProductionPPOConfig
from finrl.types import Array


def _safe_action(action: Array) -> Array:
    """Clip action weights away from zero for stable log probability."""

    return jnp.clip(action, 1e-12, 1.0)


def validate_simplex_action(action: Array, atol: float = 1e-5) -> None:
    """Raise when an eager action is not a finite long-only simplex vector."""

    action_array = jnp.asarray(action)
    if bool(jnp.any(~jnp.isfinite(action_array))):
        raise ValueError("action contains NaN or infinite values.")
    if bool(jnp.any(action_array < -atol)):
        raise ValueError("action contains negative portfolio weights.")
    if not bool(jnp.isclose(jnp.sum(action_array, axis=-1), 1.0, atol=atol).all()):
        raise ValueError("action weights must sum to 1.")


@dataclass(frozen=True, slots=True)
class DirichletPortfolioDistribution:
    """Dirichlet policy whose mean allocation comes from actor logits."""

    logits: Array
    config: ProductionPPOConfig

    def mean(self) -> Array:
        """Return deterministic evaluation allocation on the simplex."""

        return jax.nn.softmax(self.logits / self.config.temperature, axis=-1)

    def concentration(self) -> Array:
        """Return positive Dirichlet concentration parameters."""

        return (
            self.config.dirichlet_concentration * self.mean()
            + self.config.min_concentration
        )

    def sample(self, rng: Array) -> Array:
        """Sample a long-only allocation."""

        return jax.random.dirichlet(rng, self.concentration())

    def log_prob(self, action: Array) -> Array:
        """Return Dirichlet log probability for ``action``."""

        alpha = self.concentration()
        safe_action = _safe_action(action)
        alpha_sum = jnp.sum(alpha, axis=-1)
        return (
            gammaln(alpha_sum)
            - jnp.sum(gammaln(alpha), axis=-1)
            + jnp.sum((alpha - 1.0) * jnp.log(safe_action), axis=-1)
        )

    def entropy(self) -> Array:
        """Return Dirichlet entropy."""

        alpha = self.concentration()
        alpha_sum = jnp.sum(alpha, axis=-1)
        dimension = alpha.shape[-1]
        return (
            jnp.sum(gammaln(alpha), axis=-1)
            - gammaln(alpha_sum)
            + (alpha_sum - dimension) * digamma(alpha_sum)
            - jnp.sum((alpha - 1.0) * digamma(alpha), axis=-1)
        )


def action_log_prob(logits: Array, action: Array, config: ProductionPPOConfig) -> Array:
    """Return production policy log probability for an allocation."""

    return DirichletPortfolioDistribution(logits=logits, config=config).log_prob(action)


def policy_entropy(logits: Array, config: ProductionPPOConfig) -> Array:
    """Return production policy entropy."""

    return DirichletPortfolioDistribution(logits=logits, config=config).entropy()
