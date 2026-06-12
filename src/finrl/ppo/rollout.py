"""PPO rollout collection over raw asset windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.env.trading_env import EnvConfig, EnvState, StepResult, environment_step
from finrl.models.asset_encoder import AssetOnlyEncoder, AssetOnlyEncoderConfig
from finrl.ppo.flax_policy import (
    PPOState,
    ProductionPPOConfig,
    build_structured_ppo_state,
    sample_action,
)
from finrl.ppo.flax_value import PortfolioCriticFlax
from finrl.types import Array


class RolloutBatch(NamedTuple):
    """Rollout tensors with raw features as the training input."""

    asset_windows: Array
    prev_weights: Array
    drawdowns: Array
    prev_turnovers: Array
    actions: Array
    old_log_probs: Array
    rewards: Array
    values: Array
    dones: Array
    truncations: Array
    entropies: Array
    turnovers: Array
    transaction_costs: Array
    net_returns: Array


@dataclass(frozen=True, slots=True)
class RolloutBuffer:
    """Rollout batch plus final environment state and raw step diagnostics."""

    batch: RolloutBatch
    final_env_state: EnvState
    step_results: StepResult
    bootstrap_value: Array


def _validate_rollout_inputs(
    asset_windows: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    config: ProductionPPOConfig,
    rollout_length: int,
) -> None:
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive.")
    if rollout_length > asset_windows.shape[0]:
        raise ValueError("rollout_length cannot exceed available timesteps.")
    if asset_returns.shape[0] < rollout_length:
        raise ValueError("asset_returns length is shorter than rollout_length.")
    if asset_returns.shape[-1] != config.action_dim:
        raise ValueError("asset_returns dimension must match action dimension.")
    if spy_returns.shape[0] < rollout_length:
        raise ValueError("spy_returns length is shorter than rollout_length.")
    if initial_env_state.weights.shape != (config.action_dim,):
        raise ValueError("initial_env_state weights must match action dimension.")


def collect_rollout(
    variables: dict[str, object],
    asset_windows: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    ppo_config: ProductionPPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
    rng: Array,
    rollout_length: int | None = None,
    deterministic: bool = False,
    terminal: bool = False,
) -> RolloutBuffer:
    """Collect one rollout using current encoder, actor, and critic params."""

    length = asset_windows.shape[0] if rollout_length is None else rollout_length
    _validate_rollout_inputs(
        asset_windows,
        asset_returns,
        spy_returns,
        initial_env_state,
        ppo_config,
        length,
    )
    asset_windows = jnp.asarray(asset_windows[:length], dtype=jnp.float32)
    asset_returns = jnp.asarray(asset_returns[:length], dtype=jnp.float32)
    spy_returns = jnp.asarray(spy_returns[:length], dtype=jnp.float32)
    encoder = AssetOnlyEncoder(
        encoder_config,
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
    )
    critic = PortfolioCriticFlax(ppo_config)

    def step_fn(
        carry: tuple[EnvState, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[EnvState, Array], tuple[Array, ...]]:
        env_state, key = carry
        window_t, returns_t, spy_t = inputs
        key, action_key = jax.random.split(key)
        embeddings_t = encoder.apply(
            {"params": variables["params"]["encoder"]},
            window_t[None, ...],
        )[0]
        state_t: PPOState = build_structured_ppo_state(
            asset_embeddings=embeddings_t,
            prev_weights=env_state.weights,
            drawdown=env_state.drawdown,
            previous_turnover=env_state.previous_turnover,
        )
        action = sample_action(
            {"params": variables["params"]["actor"]},
            state_t,
            action_key,
            ppo_config,
            deterministic=deterministic,
        )
        value_t = critic.apply({"params": variables["params"]["critic"]}, state_t)
        result = environment_step(env_state, action.weights, returns_t, spy_t, env_config)
        return (result.state, key), (
            window_t,
            env_state.weights,
            env_state.drawdown,
            env_state.previous_turnover,
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
        (asset_windows, asset_returns, spy_returns),
    )
    (
        raw_windows,
        prev_weights,
        drawdowns,
        prev_turnovers,
        actions,
        old_log_probs,
        rewards,
        values,
        entropies,
        step_results,
    ) = outputs
    final_done = 1.0 if terminal else 0.0
    final_truncation = 0.0 if terminal else 1.0
    dones = jnp.zeros_like(rewards).at[-1].set(final_done)
    truncations = jnp.zeros_like(rewards).at[-1].set(final_truncation)
    bootstrap_value = jnp.asarray(0.0, dtype=jnp.float32)
    return RolloutBuffer(
        batch=RolloutBatch(
            asset_windows=jax.lax.stop_gradient(raw_windows),
            prev_weights=jax.lax.stop_gradient(prev_weights),
            drawdowns=jax.lax.stop_gradient(drawdowns),
            prev_turnovers=jax.lax.stop_gradient(prev_turnovers),
            actions=jax.lax.stop_gradient(actions),
            old_log_probs=jax.lax.stop_gradient(old_log_probs),
            rewards=jax.lax.stop_gradient(rewards),
            values=jax.lax.stop_gradient(values),
            dones=jax.lax.stop_gradient(dones),
            truncations=jax.lax.stop_gradient(truncations),
            entropies=jax.lax.stop_gradient(entropies),
            turnovers=jax.lax.stop_gradient(step_results.turnover),
            transaction_costs=jax.lax.stop_gradient(step_results.transaction_cost),
            net_returns=jax.lax.stop_gradient(step_results.net_return),
        ),
        final_env_state=final_env_state,
        step_results=step_results,
        bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
    )
