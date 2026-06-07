"""Production Flax PPO rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.env.trading_env import EnvConfig, EnvState, StepResult, environment_step
from finrl.ppo.flax_policy import (
    PortfolioActorFlax,
    ProductionPPOConfig,
    build_ppo_state,
    sample_action,
)
from finrl.ppo.flax_value import PortfolioCriticFlax
from finrl.types import Array


class RolloutBatch(NamedTuple):
    """Frozen rollout tensors with a shared leading time dimension."""

    states: Array
    actions: Array
    old_log_probs: Array
    rewards: Array
    values: Array
    dones: Array
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


def _validate_rollout_inputs(
    phi: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    config: ProductionPPOConfig,
    rollout_length: int,
) -> None:
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive.")
    if rollout_length > phi.shape[0]:
        raise ValueError("rollout_length cannot exceed available timesteps.")
    if phi.shape[-1] != config.phi_dim:
        raise ValueError("phi dimension does not match ProductionPPOConfig.")
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


def collect_rollout(
    actor_variables: dict[str, object],
    critic_variables: dict[str, object],
    phi: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    ppo_config: ProductionPPOConfig,
    rng: Array,
    rollout_length: int | None = None,
) -> RolloutBuffer:
    """Collect one production rollout using the existing environment step."""

    length = phi.shape[0] if rollout_length is None else rollout_length
    _validate_rollout_inputs(
        phi,
        regime_probs,
        asset_returns,
        spy_returns,
        initial_env_state,
        ppo_config,
        length,
    )
    phi = phi[:length]
    regime_probs = regime_probs[:length]
    asset_returns = asset_returns[:length]
    spy_returns = spy_returns[:length]
    critic = PortfolioCriticFlax(ppo_config)

    def step_fn(
        carry: tuple[EnvState, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[EnvState, Array], tuple[Array, Array, Array, Array, Array, StepResult]]:
        env_state, key = carry
        phi_t, regime_t, returns_t, spy_t = inputs
        key, action_key = jax.random.split(key)
        state_t = build_ppo_state(
            phi_t,
            regime_t,
            env_state.weights,
            env_state.drawdown,
            env_state.previous_turnover,
        )
        action = sample_action(
            actor_variables,
            state_t,
            action_key,
            ppo_config,
        )
        value_t = critic.apply(critic_variables, state_t)
        result = environment_step(env_state, action.weights, returns_t, spy_t, env_config)
        return (result.state, key), (
            state_t,
            action.weights,
            action.log_prob,
            result.reward,
            value_t,
            action.entropy,
            result,
        )

    (final_env_state, _), outputs = jax.lax.scan(
        step_fn,
        (initial_env_state, rng),
        (phi, regime_probs, asset_returns, spy_returns),
    )
    states, actions, old_log_probs, rewards, values, entropies, step_results = outputs
    dones = jnp.zeros_like(rewards).at[-1].set(1.0)
    batch = RolloutBatch(
        states=jax.lax.stop_gradient(states),
        actions=jax.lax.stop_gradient(actions),
        old_log_probs=jax.lax.stop_gradient(old_log_probs),
        rewards=jax.lax.stop_gradient(rewards),
        values=jax.lax.stop_gradient(values),
        dones=jax.lax.stop_gradient(dones),
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
    )
