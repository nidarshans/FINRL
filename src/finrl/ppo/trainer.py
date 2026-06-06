"""Small pure-JAX PPO rollout and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from finrl.env.trading_env import EnvConfig, EnvState, StepResult, environment_step
from finrl.features.splitsafe import FitWindow
from finrl.ppo.gae import compute_gae
from finrl.ppo.losses import entropy_bonus, ppo_clip_loss, value_loss
from finrl.ppo.policy import (
    ActorCriticState,
    PPOConfig,
    PortfolioActor,
    build_ppo_state,
    evaluate_action_logprob,
    sample_action,
)
from finrl.ppo.distributions import portfolio_entropy
from finrl.ppo.value import PortfolioCritic
from finrl.types import Array


@dataclass(frozen=True, slots=True)
class PPOArtifacts:
    """Prepared train or test arrays for PPO over one split."""

    phi: Array
    regime_probs: Array
    asset_returns: Array
    spy_returns: Array
    initial_env_state: EnvState
    env_config: EnvConfig = EnvConfig()
    fit_window: FitWindow | None = None


class PPOTrajectory(NamedTuple):
    """Trajectory collected from policy and environment."""

    states: Array
    actions: Array
    old_logprobs: Array
    rewards: Array
    values: Array
    dones: Array
    entropies: Array
    final_env_state: EnvState
    step_results: StepResult


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    """Frozen actor-critic parameters trained on one window."""

    state: ActorCriticState
    train_window: FitWindow | None
    config: PPOConfig


@dataclass(frozen=True, slots=True)
class PPOTrainingResult:
    """Result of fitting PPO on one split."""

    checkpoint: PolicyCheckpoint
    trajectory: PPOTrajectory
    actor_loss: Array
    critic_loss: Array
    total_loss: Array


@dataclass(frozen=True, slots=True)
class PPOEvaluationResult:
    """Frozen-policy evaluation result."""

    checkpoint: PolicyCheckpoint
    trajectory: PPOTrajectory


def initialize_actor_critic(config: PPOConfig, rng: Array) -> ActorCriticState:
    """Initialize actor and critic parameters."""

    actor_key, critic_key = jax.random.split(rng)
    return ActorCriticState(
        actor_params=PortfolioActor(config).init(actor_key),
        critic_params=PortfolioCritic(config).init(critic_key),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _validate_artifacts(artifacts: PPOArtifacts, config: PPOConfig) -> None:
    if artifacts.phi.shape[0] == 0:
        raise ValueError("PPO artifacts must contain at least one timestep.")
    if artifacts.phi.shape[-1] != config.phi_dim:
        raise ValueError("phi dimension does not match PPOConfig.")
    if artifacts.regime_probs.shape != (artifacts.phi.shape[0], config.n_regimes):
        raise ValueError("regime_probs shape does not match PPOConfig.")
    if artifacts.asset_returns.shape != (artifacts.phi.shape[0], config.n_assets):
        raise ValueError("asset_returns shape does not match PPOConfig.")
    if artifacts.spy_returns.shape[0] != artifacts.phi.shape[0]:
        raise ValueError("spy_returns length must match phi length.")
    if artifacts.initial_env_state.weights.shape != (config.n_assets,):
        raise ValueError("initial environment weights must match PPOConfig.")


def collect_train_trajectory(
    policy: ActorCriticState,
    env: EnvConfig,
    train_data: PPOArtifacts,
    config: PPOConfig | None = None,
    rng: Array | None = None,
) -> PPOTrajectory:
    """Collect one train trajectory using existing environment accounting."""

    ppo_config = config or PPOConfig(n_assets=train_data.asset_returns.shape[-1])
    _validate_artifacts(train_data, ppo_config)
    key = jax.random.PRNGKey(0) if rng is None else rng
    critic = PortfolioCritic(ppo_config)

    def step_fn(carry: tuple[EnvState, Array], inputs: tuple[Array, Array, Array, Array]):
        env_state, step_key = carry
        phi_t, regime_t, returns_t, spy_t = inputs
        step_key, action_key = jax.random.split(step_key)
        state_t = build_ppo_state(phi_t, regime_t, env_state)
        action = sample_action(
            policy.actor_params,
            state_t,
            action_key,
            ppo_config.temperature,
            ppo_config.dirichlet_concentration,
            ppo_config.min_concentration,
        )
        result = environment_step(env_state, action.weights, returns_t, spy_t, env)
        value_t = critic.apply(policy.critic_params, state_t)
        return (result.state, step_key), (
            state_t,
            action.weights,
            action.logprob,
            result.reward,
            value_t,
            action.entropy,
            result,
        )

    (final_env_state, _), outputs = jax.lax.scan(
        step_fn,
        (train_data.initial_env_state, key),
        (
            train_data.phi,
            train_data.regime_probs,
            train_data.asset_returns,
            train_data.spy_returns,
        ),
    )
    states, actions, logprobs, rewards, values, entropies, step_results = outputs
    dones = jnp.zeros_like(rewards).at[-1].set(1.0)
    return PPOTrajectory(
        states=states,
        actions=actions,
        old_logprobs=logprobs,
        rewards=rewards,
        values=values,
        dones=dones,
        entropies=entropies,
        final_env_state=final_env_state,
        step_results=step_results,
    )


def _loss_for_params(
    actor_params: dict[str, Array],
    critic_params: dict[str, Array],
    trajectory: PPOTrajectory,
    config: PPOConfig,
) -> tuple[Array, tuple[Array, Array]]:
    critic = PortfolioCritic(config)
    values = jax.vmap(lambda state: critic.apply(critic_params, state))(trajectory.states)
    advantages, returns = compute_gae(
        trajectory.rewards,
        values,
        trajectory.dones,
        config.gamma,
        config.gae_lambda,
    )
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    new_logprobs = jax.vmap(
        lambda state, action: evaluate_action_logprob(
            actor_params,
            state,
            action,
            config.temperature,
            config.dirichlet_concentration,
            config.min_concentration,
        )
    )(trajectory.states, trajectory.actions)
    logits = jax.vmap(lambda state: PortfolioActor(config).apply(actor_params, state))(
        trajectory.states
    )
    entropies = jax.vmap(
        lambda logit: portfolio_entropy(
            logit,
            config.temperature,
            config.dirichlet_concentration,
            config.min_concentration,
        )
    )(logits)
    actor_loss = ppo_clip_loss(
        new_logprobs,
        jax.lax.stop_gradient(trajectory.old_logprobs),
        jax.lax.stop_gradient(advantages),
        config.clip_epsilon,
    )
    critic_loss = value_loss(values, jax.lax.stop_gradient(returns))
    total = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy_bonus(
        entropies
    )
    return total, (actor_loss, critic_loss)


def _sgd_update(params: dict[str, Array], grads: dict[str, Array], learning_rate: float):
    return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)


def train_ppo_on_split(
    train_artifacts: PPOArtifacts,
    config: PPOConfig,
    rng: Array | None = None,
) -> PPOTrainingResult:
    """Train PPO on a single train split and return a frozen checkpoint."""

    _validate_artifacts(train_artifacts, config)
    key = jax.random.PRNGKey(0) if rng is None else rng
    policy_state = initialize_actor_critic(config, key)
    trajectory = collect_train_trajectory(
        policy_state,
        train_artifacts.env_config,
        train_artifacts,
        config,
        key,
    )

    actor_params = policy_state.actor_params
    critic_params = policy_state.critic_params
    actor_loss = jnp.array(0.0, dtype=jnp.float32)
    critic_loss = jnp.array(0.0, dtype=jnp.float32)
    total_loss = jnp.array(0.0, dtype=jnp.float32)
    for _ in range(config.train_epochs):
        def loss_fn(
            current_actor_params: dict[str, Array],
            current_critic_params: dict[str, Array],
        ) -> tuple[Array, tuple[Array, Array]]:
            return _loss_for_params(
                current_actor_params,
                current_critic_params,
                trajectory,
                config,
            )

        (total_loss, (actor_loss, critic_loss)), grads = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(actor_params, critic_params)
        actor_grads, critic_grads = grads
        actor_params = _sgd_update(actor_params, actor_grads, config.learning_rate)
        critic_params = _sgd_update(critic_params, critic_grads, config.learning_rate)

    updated_state = ActorCriticState(
        actor_params=actor_params,
        critic_params=critic_params,
        step=policy_state.step + jnp.array(config.train_epochs, dtype=jnp.int32),
    )
    checkpoint = PolicyCheckpoint(
        state=updated_state,
        train_window=train_artifacts.fit_window,
        config=config,
    )
    return PPOTrainingResult(
        checkpoint=checkpoint,
        trajectory=trajectory,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        total_loss=total_loss,
    )


def evaluate_frozen_policy(
    policy_checkpoint: PolicyCheckpoint,
    test_artifacts: PPOArtifacts,
    rng: Array | None = None,
) -> PPOEvaluationResult:
    """Evaluate a checkpoint without updating policy parameters."""

    trajectory = collect_train_trajectory(
        policy_checkpoint.state,
        test_artifacts.env_config,
        test_artifacts,
        policy_checkpoint.config,
        jax.random.PRNGKey(0) if rng is None else rng,
    )
    return PPOEvaluationResult(checkpoint=policy_checkpoint, trajectory=trajectory)
