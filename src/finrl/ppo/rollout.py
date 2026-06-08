"""Production Flax PPO rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.env.trading_env import EnvConfig, EnvState, StepResult, environment_step
from finrl.ppo.flax_policy import (
    PPOState,
    ProductionPPOConfig,
    build_structured_ppo_state,
    sample_action,
)
from finrl.ppo.flax_value import PortfolioCriticFlax
from finrl.types import Array


class RolloutBatch(NamedTuple):
    """Frozen rollout tensors with a shared leading time dimension."""

    states: Array | PPOState
    actions: Array
    old_log_probs: Array
    rewards: Array
    values: Array
    dones: Array
    truncations: Array
    entropies: Array
    turnovers: Array
    transaction_costs: Array
    drawdowns: Array
    net_returns: Array


@dataclass(frozen=True, slots=True)
class RolloutBuffer:
    """Rollout batch plus final environment state and raw step diagnostics."""

    batch: RolloutBatch
    final_env_state: EnvState
    step_results: StepResult
    bootstrap_value: Array


def _validate_rollout_inputs(
    market_vectors: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    config: ProductionPPOConfig,
    rollout_length: int,
    asset_embeddings: Array,
    macro_states: Array,
    spectral_states: Array,
) -> None:
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive.")
    if rollout_length > market_vectors.shape[0]:
        raise ValueError("rollout_length cannot exceed available timesteps.")
    if market_vectors.shape[-1] != config.phi_dim:
        raise ValueError("market vector dimension does not match ProductionPPOConfig.")
    if regime_probs.shape[0] < rollout_length:
        raise ValueError("regime_probs length is shorter than rollout_length.")
    if regime_probs.shape[-1] != config.n_regimes:
        raise ValueError("regime_probs dimension does not match ProductionPPOConfig.")
    if asset_returns.shape[0] < rollout_length:
        raise ValueError("asset_returns length is shorter than rollout_length.")
    if asset_returns.shape[-1] != config.action_dim:
        raise ValueError("asset_returns dimension must match action dimension.")
    if spy_returns.shape[0] < rollout_length:
        raise ValueError("spy_returns length is shorter than rollout_length.")
    if initial_env_state.weights.shape != (config.action_dim,):
        raise ValueError("initial_env_state weights must match action dimension.")
    if asset_embeddings.shape[0] < rollout_length:
        raise ValueError("asset_embeddings length is shorter than rollout_length.")
    if asset_embeddings.shape[1:] != (config.action_dim - 1, config.asset_latent_dim):
        raise ValueError("asset_embeddings shape must be [T, n_assets - 1, latent_dim].")
    if macro_states.shape[0] < rollout_length or macro_states.shape[-1] != config.macro_dim:
        raise ValueError("macro_states shape must be [T, macro_dim].")
    if spectral_states.shape[0] < rollout_length or spectral_states.shape[-1] != config.spectral_dim:
        raise ValueError("spectral_states shape must be [T, spectral_dim].")


def collect_rollout(
    actor_variables: dict[str, object],
    critic_variables: dict[str, object],
    market_vectors: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    ppo_config: ProductionPPOConfig,
    rng: Array,
    asset_embeddings: Array,
    macro_states: Array,
    spectral_states: Array,
    rollout_length: int | None = None,
    deterministic: bool = False,
    bootstrap_market_vector: Array | None = None,
    bootstrap_regime_probs: Array | None = None,
    bootstrap_asset_embeddings: Array | None = None,
    bootstrap_macro_state: Array | None = None,
    bootstrap_spectral_state: Array | None = None,
    terminal: bool = False,
) -> RolloutBuffer:
    """Collect one production rollout using the existing environment step."""

    length = market_vectors.shape[0] if rollout_length is None else rollout_length
    _validate_rollout_inputs(
        market_vectors,
        regime_probs,
        asset_returns,
        spy_returns,
        initial_env_state,
        ppo_config,
        length,
        asset_embeddings,
        macro_states,
        spectral_states,
    )
    if (bootstrap_market_vector is None) != (bootstrap_regime_probs is None):
        raise ValueError(
            "bootstrap_market_vector and bootstrap_regime_probs must be provided together."
        )
    market_vectors = market_vectors[:length]
    regime_probs = regime_probs[:length]
    asset_embeddings = asset_embeddings[:length]
    macro_states = macro_states[:length]
    spectral_states = spectral_states[:length]
    asset_returns = asset_returns[:length]
    spy_returns = spy_returns[:length]
    critic = PortfolioCriticFlax(ppo_config)

    def run_step(
        env_state: EnvState,
        key: Array,
        state_t: Array | PPOState,
        returns_t: Array,
        spy_t: Array,
    ) -> tuple[EnvState, Array, tuple[Array | PPOState, Array, Array, Array, Array, Array, StepResult]]:
        key, action_key = jax.random.split(key)
        action = sample_action(
            actor_variables,
            state_t,
            action_key,
            ppo_config,
            deterministic=deterministic,
        )
        value_t = critic.apply(critic_variables, state_t)
        result = environment_step(env_state, action.weights, returns_t, spy_t, env_config)
        return result.state, key, (
            state_t,
            action.weights,
            action.log_prob,
            result.reward,
            value_t,
            action.entropy,
            result,
        )

    def step_fn(
        carry: tuple[EnvState, Array],
        inputs: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[tuple[EnvState, Array], tuple[PPOState, Array, Array, Array, Array, StepResult]]:
        env_state, key = carry
        embedding_t, market_vector_t, macro_t, spectral_t, regime_t, returns_t, spy_t = inputs
        state_t = build_structured_ppo_state(
            asset_embeddings=embedding_t,
            market_vector=market_vector_t,
            macro_state=macro_t,
            spectral_state=spectral_t,
            regime_probs=regime_t,
            prev_weights=env_state.weights,
            drawdown=env_state.drawdown,
            previous_turnover=env_state.previous_turnover,
        )
        next_env_state, next_key, output = run_step(
            env_state,
            key,
            state_t,
            returns_t,
            spy_t,
        )
        return (next_env_state, next_key), output

    (final_env_state, _), outputs = jax.lax.scan(
        step_fn,
        (initial_env_state, rng),
        (
            asset_embeddings,
            market_vectors,
            macro_states,
            spectral_states,
            regime_probs,
            asset_returns,
            spy_returns,
        ),
    )
    states, actions, old_log_probs, rewards, values, entropies, step_results = outputs
    has_bootstrap = (
        bootstrap_market_vector is not None and bootstrap_regime_probs is not None
    )
    if has_bootstrap:
        if (
            bootstrap_asset_embeddings is None
            or bootstrap_macro_state is None
            or bootstrap_spectral_state is None
        ):
            raise ValueError("structured bootstrap state components are required.")
        bootstrap_state = build_structured_ppo_state(
            asset_embeddings=jnp.asarray(bootstrap_asset_embeddings, dtype=jnp.float32),
            market_vector=jnp.asarray(bootstrap_market_vector, dtype=jnp.float32),
            macro_state=jnp.asarray(bootstrap_macro_state, dtype=jnp.float32),
            spectral_state=jnp.asarray(bootstrap_spectral_state, dtype=jnp.float32),
            regime_probs=jnp.asarray(bootstrap_regime_probs, dtype=jnp.float32),
            prev_weights=final_env_state.weights,
            drawdown=final_env_state.drawdown,
            previous_turnover=final_env_state.previous_turnover,
        )
        bootstrap_value = critic.apply(critic_variables, bootstrap_state)
    else:
        bootstrap_value = jnp.asarray(0.0, dtype=jnp.float32)
    final_done = 1.0 if terminal else 0.0
    final_truncation = 0.0 if terminal else 1.0
    dones = jnp.zeros_like(rewards).at[-1].set(final_done)
    truncations = jnp.zeros_like(rewards).at[-1].set(final_truncation)
    batch = RolloutBatch(
        states=jax.lax.stop_gradient(states),
        actions=jax.lax.stop_gradient(actions),
        old_log_probs=jax.lax.stop_gradient(old_log_probs),
        rewards=jax.lax.stop_gradient(rewards),
        values=jax.lax.stop_gradient(values),
        dones=jax.lax.stop_gradient(dones),
        truncations=jax.lax.stop_gradient(truncations),
        entropies=jax.lax.stop_gradient(entropies),
        turnovers=jax.lax.stop_gradient(step_results.turnover),
        transaction_costs=jax.lax.stop_gradient(step_results.transaction_cost),
        drawdowns=jax.lax.stop_gradient(step_results.state.drawdown),
        net_returns=jax.lax.stop_gradient(step_results.net_return),
    )
    return RolloutBuffer(
        batch=batch,
        final_env_state=final_env_state,
        step_results=step_results,
        bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
    )
