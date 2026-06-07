"""Small pure-JAX PPO rollout and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from finrl.env.trading_env import EnvConfig, EnvState, StepResult, environment_step
from finrl.features.splitsafe import FitWindow
from finrl.logging.tensorboard import TensorBoardLogger
from finrl.ppo.gae import compute_gae
from finrl.ppo.losses import (
    clipped_value_loss,
    entropy_bonus,
    ppo_clip_loss,
    value_loss,
)
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
    diagnostics: dict[str, Array]


@dataclass(frozen=True, slots=True)
class PPOEvaluationResult:
    """Frozen-policy evaluation result."""

    checkpoint: PolicyCheckpoint
    trajectory: PPOTrajectory


def initialize_actor_critic(config: PPOConfig, rng: Array) -> ActorCriticState:
    """Initialize actor and critic parameters."""

    actor_key, critic_key = jax.random.split(rng)
    actor_params = PortfolioActor(config).init(actor_key)
    critic_params = PortfolioCritic(config).init(critic_key)
    optimizer = _make_optimizer(config)
    return ActorCriticState(
        actor_params=actor_params,
        critic_params=critic_params,
        step=jnp.array(0, dtype=jnp.int32),
        optimizer_state=optimizer.init((actor_params, critic_params)),
    )


class PPOUpdateBatch(NamedTuple):
    """Frozen tensors used for PPO epochs after rollout collection."""

    states: Array
    actions: Array
    old_logprobs: Array
    advantages: Array
    returns: Array
    old_values: Array
    rewards: Array
    turnovers: Array
    drawdowns: Array


def _make_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    """Build the persistent PPO optimizer."""

    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
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


def normalize_advantages(advantages: Array, epsilon: float = 1e-8) -> Array:
    """Normalize advantages once per rollout with a near-zero variance guard."""

    std = jnp.std(advantages)
    centered = advantages - jnp.mean(advantages)
    return jnp.where(std > epsilon, centered / (std + epsilon), jnp.zeros_like(advantages))


def freeze_ppo_batch(trajectory: PPOTrajectory, config: PPOConfig) -> PPOUpdateBatch:
    """Compute and freeze rollout tensors used across all PPO epochs."""

    advantages, returns = compute_gae(
        trajectory.rewards,
        trajectory.values,
        trajectory.dones,
        config.gamma,
        config.gae_lambda,
    )
    if config.normalize_advantages:
        advantages = normalize_advantages(advantages)
    return PPOUpdateBatch(
        states=jax.lax.stop_gradient(trajectory.states),
        actions=jax.lax.stop_gradient(trajectory.actions),
        old_logprobs=jax.lax.stop_gradient(trajectory.old_logprobs),
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        old_values=jax.lax.stop_gradient(trajectory.values),
        rewards=jax.lax.stop_gradient(trajectory.rewards),
        turnovers=jax.lax.stop_gradient(trajectory.step_results.turnover),
        drawdowns=jax.lax.stop_gradient(trajectory.step_results.state.drawdown),
    )


def _take_batch(batch: PPOUpdateBatch, indices: Array) -> PPOUpdateBatch:
    return jax.tree_util.tree_map(lambda leaf: leaf[indices], batch)


def _explained_variance(values: Array, returns: Array) -> Array:
    target_var = jnp.var(returns)
    return jnp.where(
        target_var > 1e-8,
        1.0 - jnp.var(returns - values) / (target_var + 1e-8),
        jnp.array(0.0, dtype=values.dtype),
    )


def _portfolio_diagnostics(
    trajectory: PPOTrajectory,
    spy_returns: Array,
    config: PPOConfig,
) -> dict[str, Array]:
    actions = trajectory.actions
    returns = trajectory.step_results.net_return
    cash_index = config.n_assets - 1 if config.n_assets > 0 else -1
    if cash_index < 0:
        cash_weight = jnp.array(0.0, dtype=actions.dtype)
    else:
        cash_weight = jnp.mean(actions[:, cash_index])
    return {
        "reward": jnp.mean(trajectory.rewards),
        "alpha_vs_spy": jnp.mean(returns - spy_returns),
        "turnover": jnp.mean(trajectory.step_results.turnover),
        "transaction_cost": jnp.mean(trajectory.step_results.transaction_cost),
        "max_drawdown": jnp.max(trajectory.step_results.state.drawdown),
        "cash_weight": cash_weight,
        "max_position_weight": jnp.mean(jnp.max(actions, axis=1)),
        "effective_positions": jnp.mean(1.0 / jnp.sum(jnp.square(actions), axis=1)),
    }


def _loss_for_params(
    actor_params: dict[str, Array],
    critic_params: dict[str, Array],
    batch: PPOUpdateBatch,
    config: PPOConfig,
) -> tuple[Array, dict[str, Array]]:
    critic = PortfolioCritic(config)
    values = jax.vmap(lambda state: critic.apply(critic_params, state))(batch.states)
    new_logprobs = jax.vmap(
        lambda state, action: evaluate_action_logprob(
            actor_params,
            state,
            action,
            config.temperature,
            config.dirichlet_concentration,
            config.min_concentration,
        )
    )(batch.states, batch.actions)
    logits = jax.vmap(lambda state: PortfolioActor(config).apply(actor_params, state))(
        batch.states
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
        batch.old_logprobs,
        batch.advantages,
        config.clip_epsilon,
    )
    critic_loss = (
        clipped_value_loss(values, batch.old_values, batch.returns, config.value_clip_eps)
        if config.use_value_clipping
        else value_loss(values, batch.returns)
    )
    approx_kl = jnp.mean(batch.old_logprobs - new_logprobs)
    ratio = jnp.exp(new_logprobs - batch.old_logprobs)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32)
    )
    entropy = entropy_bonus(entropies)
    total = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy_bonus(
        entropies
    )
    diagnostics = {
        "policy_loss": actor_loss,
        "value_loss": critic_loss,
        "entropy": entropy,
        "total_loss": total,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": _explained_variance(values, batch.returns),
        "mean_reward": jnp.mean(batch.rewards),
        "mean_turnover": jnp.mean(batch.turnovers),
        "max_drawdown": jnp.max(batch.drawdowns),
    }
    return total, diagnostics


def _empty_diagnostics(dtype: jnp.dtype = jnp.float32) -> dict[str, Array]:
    return {
        name: jnp.array(0.0, dtype=dtype)
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "total_loss",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "grad_norm",
            "mean_reward",
            "mean_turnover",
            "max_drawdown",
            "epochs_completed",
            "updates_completed",
        )
    }


def _merge_diagnostics(diagnostics: list[dict[str, Array]]) -> dict[str, Array]:
    if not diagnostics:
        return _empty_diagnostics()
    names = diagnostics[0].keys()
    return {
        name: jnp.mean(jnp.stack([entry[name] for entry in diagnostics]))
        for name in names
    }


def _train_minibatch(
    actor_params: dict[str, Array],
    critic_params: dict[str, Array],
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: PPOUpdateBatch,
    config: PPOConfig,
) -> tuple[dict[str, Array], dict[str, Array], optax.OptState, dict[str, Array]]:
    def loss_fn(
        current_actor_params: dict[str, Array],
        current_critic_params: dict[str, Array],
    ) -> tuple[Array, dict[str, Array]]:
        return _loss_for_params(
            current_actor_params,
            current_critic_params,
            batch,
            config,
        )

    (total_loss, diagnostics), grads = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(actor_params, critic_params)
    diagnostics = {
        **diagnostics,
        "total_loss": total_loss,
        "grad_norm": optax.global_norm(grads),
    }
    updates, optimizer_state = optimizer.update(
        grads,
        optimizer_state,
        (actor_params, critic_params),
    )
    actor_updates, critic_updates = updates
    actor_params = optax.apply_updates(actor_params, actor_updates)
    critic_params = optax.apply_updates(critic_params, critic_updates)
    return actor_params, critic_params, optimizer_state, diagnostics


def train_ppo_on_split(
    train_artifacts: PPOArtifacts,
    config: PPOConfig,
    rng: Array | None = None,
    logger: TensorBoardLogger | None = None,
    experiment_name: str | None = None,
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
    optimizer = _make_optimizer(config)
    optimizer_state = policy_state.optimizer_state
    if optimizer_state is None:
        optimizer_state = optimizer.init((actor_params, critic_params))
    frozen_batch = freeze_ppo_batch(trajectory, config)
    owned_logger = (
        TensorBoardLogger(
            log_dir=config.log_dir,
            experiment_name=experiment_name,
            enabled=config.enable_tensorboard,
        )
        if logger is None
        else None
    )
    active_logger = logger if logger is not None else owned_logger
    if active_logger is not None:
        active_logger.log_hyperparameters(config)
    portfolio_metrics = _portfolio_diagnostics(
        trajectory,
        train_artifacts.spy_returns,
        config,
    )
    n_steps = int(frozen_batch.rewards.shape[0])
    diagnostics_history: list[dict[str, Array]] = []
    updates_completed = 0
    epochs_completed = 0
    key, shuffle_key = jax.random.split(key)
    for epoch in range(config.update_epochs):
        epoch_key = jax.random.fold_in(shuffle_key, epoch)
        indices = jax.random.permutation(epoch_key, n_steps)
        stop_epoch = False
        for start in range(0, n_steps, config.minibatch_size):
            minibatch_indices = indices[start : start + config.minibatch_size]
            minibatch = _take_batch(frozen_batch, minibatch_indices)
            actor_params, critic_params, optimizer_state, diagnostics = _train_minibatch(
                actor_params,
                critic_params,
                optimizer_state,
                optimizer,
                minibatch,
                config,
            )
            diagnostics_history.append(diagnostics)
            updates_completed += 1
            if active_logger is not None and updates_completed % config.log_frequency == 0:
                active_logger.log_scalars(diagnostics, updates_completed, "ppo")
                active_logger.log_scalars(portfolio_metrics, updates_completed, "portfolio")
                active_logger.log_regime_metrics(
                    train_artifacts.regime_probs,
                    trajectory.actions,
                    updates_completed,
                    "regime",
                )
            if float(diagnostics["approx_kl"]) > config.target_kl:
                stop_epoch = True
                break
        epochs_completed += 1
        if stop_epoch:
            break

    diagnostics = _merge_diagnostics(diagnostics_history)
    diagnostics = {
        **diagnostics,
        "epochs_completed": jnp.array(epochs_completed, dtype=jnp.float32),
        "updates_completed": jnp.array(updates_completed, dtype=jnp.float32),
    }

    updated_state = ActorCriticState(
        actor_params=actor_params,
        critic_params=critic_params,
        step=policy_state.step + jnp.array(updates_completed, dtype=jnp.int32),
        optimizer_state=optimizer_state,
    )
    checkpoint = PolicyCheckpoint(
        state=updated_state,
        train_window=train_artifacts.fit_window,
        config=config,
    )
    if owned_logger is not None:
        owned_logger.close()
    return PPOTrainingResult(
        checkpoint=checkpoint,
        trajectory=trajectory,
        actor_loss=diagnostics["policy_loss"],
        critic_loss=diagnostics["value_loss"],
        total_loss=diagnostics["total_loss"],
        diagnostics=diagnostics,
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
