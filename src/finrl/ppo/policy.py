"""JAX actor policy for long-only portfolio PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from finrl.env.trading_env import EnvState
from finrl.ppo.distributions import (
    portfolio_entropy,
    portfolio_logprob,
    sample_dirichlet_portfolio,
    temperature_softmax,
)
from finrl.types import Array

Params = dict[str, Array]


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Configuration for actor-critic PPO on portfolio states."""

    phi_dim: int = 32
    n_regimes: int = 4
    n_assets: int = 101
    actor_hidden_dims: tuple[int, int] = (128, 128)
    critic_hidden_dims: tuple[int, int] = (128, 64)
    temperature: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    learning_rate: float = 1e-3
    ppo_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    use_value_clipping: bool = True
    train_epochs: int | None = None
    enable_tensorboard: bool = False
    log_dir: str | Path = "runs"
    log_frequency: int = 1
    # Logits define a softmax mean; this scale controls Dirichlet exploration.
    dirichlet_concentration: float = 100.0
    min_concentration: float = 1e-3

    @property
    def state_dim(self) -> int:
        """Return architecture-derived PPO state dimension."""

        return self.phi_dim + self.n_regimes + self.n_assets + 2

    @property
    def clip_eps(self) -> float:
        """Alias for the PPO policy clip threshold."""

        return self.clip_epsilon

    @property
    def update_epochs(self) -> int:
        """Return the configured number of PPO epochs.

        ``train_epochs`` is kept as a compatibility alias for older callers.
        """

        return self.ppo_epochs if self.train_epochs is None else self.train_epochs

    def __post_init__(self) -> None:
        if self.phi_dim <= 0 or self.n_regimes <= 0 or self.n_assets <= 0:
            raise ValueError("PPO dimensions must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive.")
        if self.value_clip_eps <= 0.0:
            raise ValueError("value_clip_eps must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive.")
        if self.train_epochs is not None and self.train_epochs <= 0:
            raise ValueError("train_epochs must be positive when provided.")
        if self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive.")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
        if self.dirichlet_concentration <= 0.0:
            raise ValueError("dirichlet_concentration must be positive.")
        if self.min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive.")


class PortfolioContext(NamedTuple):
    """Portfolio context used to build PPO state."""

    weights: Array
    drawdown: Array
    previous_turnover: Array


class PortfolioAction(NamedTuple):
    """Portfolio action diagnostics emitted by the actor."""

    weights: Array
    logits: Array
    logprob: Array
    entropy: Array


class ActorCriticState(NamedTuple):
    """Actor and critic parameters plus update step."""

    actor_params: Params
    critic_params: Params
    step: Array
    optimizer_state: optax.OptState | None = None


def _glorot_uniform(key: Array, shape: tuple[int, int]) -> Array:
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mlp_params(
    key: Array,
    layer_dims: tuple[int, ...],
) -> Params:
    """Initialize MLP parameters."""

    keys = jax.random.split(key, len(layer_dims) - 1)
    params: Params = {}
    for index, layer_key in enumerate(keys):
        in_dim = layer_dims[index]
        out_dim = layer_dims[index + 1]
        params[f"w{index}"] = _glorot_uniform(layer_key, (in_dim, out_dim))
        params[f"b{index}"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def apply_mlp(params: Params, inputs: Array, n_layers: int) -> Array:
    """Apply an MLP with ReLU hidden layers."""

    x = inputs
    for index in range(n_layers):
        x = x @ params[f"w{index}"] + params[f"b{index}"]
        if index < n_layers - 1:
            x = jax.nn.relu(x)
    return x


@dataclass(frozen=True, slots=True)
class PortfolioActor:
    """Actor network mapping PPO state to portfolio logits."""

    config: PPOConfig

    def init(self, key: Array) -> Params:
        """Initialize actor parameters."""

        dims = (self.config.state_dim, *self.config.actor_hidden_dims, self.config.n_assets)
        return init_mlp_params(key, dims)

    def apply(self, params: Params, state: Array) -> Array:
        """Return action logits."""

        return apply_mlp(params, state, len(self.config.actor_hidden_dims) + 1)


def build_ppo_state(
    phi: Array,
    regime_probs: Array,
    portfolio_context: PortfolioContext | EnvState,
) -> Array:
    """Concatenate market, regime, and portfolio context into PPO state."""

    return jnp.concatenate(
        [
            jnp.asarray(phi, dtype=jnp.float32),
            jnp.asarray(regime_probs, dtype=jnp.float32),
            jnp.asarray(portfolio_context.weights, dtype=jnp.float32),
            jnp.atleast_1d(jnp.asarray(portfolio_context.drawdown, dtype=jnp.float32)),
            jnp.atleast_1d(
                jnp.asarray(portfolio_context.previous_turnover, dtype=jnp.float32)
            ),
        ],
        axis=0,
    )


def sample_action(
    params: Params,
    state: Array,
    rng: Array,
    temperature: float | Array = 1.0,
    concentration_scale: float | Array = 100.0,
    min_concentration: float | Array = 1e-3,
) -> PortfolioAction:
    """Sample long-only target weights from the actor's Dirichlet policy."""

    logits = apply_mlp(params, state, len(params) // 2)
    weights = sample_dirichlet_portfolio(
        rng,
        logits,
        temperature,
        concentration_scale,
        min_concentration,
    )
    return PortfolioAction(
        weights=weights,
        logits=logits,
        logprob=portfolio_logprob(
            logits,
            weights,
            temperature,
            concentration_scale,
            min_concentration,
        ),
        entropy=portfolio_entropy(
            logits,
            temperature,
            concentration_scale,
            min_concentration,
        ),
    )


def evaluate_action_logprob(
    params: Params,
    state: Array,
    action: Array,
    temperature: float | Array = 1.0,
    concentration_scale: float | Array = 100.0,
    min_concentration: float | Array = 1e-3,
) -> Array:
    """Evaluate action log probability under actor parameters."""

    logits = apply_mlp(params, state, len(params) // 2)
    return portfolio_logprob(
        logits,
        action,
        temperature,
        concentration_scale,
        min_concentration,
    )
