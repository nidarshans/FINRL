"""Portfolio action distribution helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln

from finrl.types import Array


def temperature_softmax(logits: Array, temperature: float | Array) -> Array:
    """Map logits to long-only portfolio weights."""

    if isinstance(temperature, (int, float)) and temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    temp = jnp.asarray(temperature, dtype=jnp.asarray(logits).dtype)
    return jax.nn.softmax(logits / temp, axis=-1)


def dirichlet_concentration(
    logits: Array,
    temperature: float | Array,
    concentration_scale: float | Array,
    min_concentration: float | Array,
) -> Array:
    """Return Dirichlet concentrations centered on temperature-softmax weights."""

    mean_weights = temperature_softmax(logits, temperature)
    scale = jnp.asarray(concentration_scale, dtype=mean_weights.dtype)
    floor = jnp.asarray(min_concentration, dtype=mean_weights.dtype)
    return mean_weights * scale + floor


def sample_dirichlet_portfolio(
    rng: Array,
    logits: Array,
    temperature: float | Array,
    concentration_scale: float | Array,
    min_concentration: float | Array,
) -> Array:
    """Sample long-only portfolio weights from a Dirichlet policy."""

    alpha = dirichlet_concentration(
        logits,
        temperature,
        concentration_scale,
        min_concentration,
    )
    return jax.random.dirichlet(rng, alpha)


def portfolio_logprob(
    logits: Array,
    action_weights: Array,
    temperature: float | Array,
    concentration_scale: float | Array = 100.0,
    min_concentration: float | Array = 1e-3,
) -> Array:
    """Return Dirichlet log probability of portfolio weights under logits."""

    alpha = dirichlet_concentration(
        logits,
        temperature,
        concentration_scale,
        min_concentration,
    )
    action = jnp.clip(action_weights, 1e-12, 1.0)
    alpha0 = jnp.sum(alpha, axis=-1)
    return (
        gammaln(alpha0)
        - jnp.sum(gammaln(alpha), axis=-1)
        + jnp.sum((alpha - 1.0) * jnp.log(action), axis=-1)
    )


def portfolio_entropy(
    logits: Array,
    temperature: float | Array,
    concentration_scale: float | Array = 100.0,
    min_concentration: float | Array = 1e-3,
) -> Array:
    """Return Dirichlet entropy of the portfolio policy."""

    alpha = dirichlet_concentration(
        logits,
        temperature,
        concentration_scale,
        min_concentration,
    )
    alpha0 = jnp.sum(alpha, axis=-1)
    n_assets = alpha.shape[-1]
    return (
        jnp.sum(gammaln(alpha), axis=-1)
        - gammaln(alpha0)
        + (alpha0 - n_assets) * digamma(alpha0)
        - jnp.sum((alpha - 1.0) * digamma(alpha), axis=-1)
    )
