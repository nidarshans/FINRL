"""Production Flax PPO optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.logging.tensorboard import TensorBoardLogger
from finrl.ppo.flax_policy import PortfolioActorFlax, ProductionPPOConfig
from finrl.ppo.flax_value import PortfolioCriticFlax
from finrl.ppo.gae import compute_gae
from finrl.ppo.losses import critic_loss, ppo_actor_loss, ppo_total_loss
from finrl.ppo.metrics import (
    PPOTrainMetrics,
    approximate_kl,
    clip_fraction,
    explained_variance,
    ppo_metrics_to_dict,
)
from finrl.ppo.rollout import RolloutBatch, RolloutBuffer, collect_rollout
from finrl.ppo.simplex_distribution import action_log_prob, policy_entropy
from finrl.types import Array


class PPOUpdateBatchFlax(NamedTuple):
    """Frozen tensors used for production PPO minibatch updates."""

    states: Array
    actions: Array
    old_log_probs: Array
    advantages: Array
    returns: Array
    old_values: Array
    rewards: Array
    turnovers: Array
    transaction_costs: Array
    drawdowns: Array


@dataclass(frozen=True, slots=True)
class ProductionPPOTrainState:
    """Actor and critic train states for production PPO."""

    actor: TrainState
    critic: TrainState
    config: ProductionPPOConfig


@dataclass(frozen=True, slots=True)
class ProductionPPOTrainingResult:
    """Result from fitting production PPO on one split."""

    train_state: ProductionPPOTrainState
    rollout: RolloutBuffer
    metrics: PPOTrainMetrics


@dataclass(frozen=True, slots=True)
class ProductionPPOEvaluationResult:
    """Frozen production PPO evaluation result."""

    train_state: ProductionPPOTrainState
    rollout: RolloutBuffer


def _optimizer(config: ProductionPPOConfig) -> optax.GradientTransformation:
    return optax.adam(config.learning_rate)


def _tree_global_norm(tree: object) -> Array:
    """Return the global L2 norm of a gradient pytree."""

    squared_norms = [
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(jnp.sum(jnp.asarray(squared_norms)))


def initialize_ppo_train_state(
    rng: Array,
    config: ProductionPPOConfig,
) -> ProductionPPOTrainState:
    """Initialize production Flax actor and critic train states."""

    actor_key, critic_key = jax.random.split(rng)
    state = jnp.zeros((config.state_dim,), dtype=jnp.float32)
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    tx = _optimizer(config)
    actor_variables = actor.init(actor_key, state)
    critic_variables = critic.init(critic_key, state)
    return ProductionPPOTrainState(
        actor=TrainState.create(
            apply_fn=actor.apply,
            params=actor_variables["params"],
            tx=tx,
        ),
        critic=TrainState.create(
            apply_fn=critic.apply,
            params=critic_variables["params"],
            tx=tx,
        ),
        config=config,
    )


def normalize_advantages(advantages: Array, epsilon: float = 1e-8) -> Array:
    """Normalize advantages with a stable zero-variance guard."""

    std = jnp.std(advantages)
    centered = advantages - jnp.mean(advantages)
    return jnp.where(std > epsilon, centered / (std + epsilon), jnp.zeros_like(advantages))


def portfolio_allocation_entropy(weights: Array, epsilon: float = 1e-8) -> Array:
    """Return Shannon entropy of long-only portfolio weights."""

    safe_weights = jnp.clip(weights, epsilon, 1.0)
    return -jnp.sum(weights * jnp.log(safe_weights), axis=-1)


def freeze_rollout_batch(
    rollout: RolloutBatch,
    config: ProductionPPOConfig,
    bootstrap_value: Array | None = None,
) -> PPOUpdateBatchFlax:
    """Compute and freeze advantages/returns from a collected rollout."""

    values = rollout.values
    if bootstrap_value is not None:
        bootstrap = jnp.atleast_1d(
            jnp.asarray(bootstrap_value, dtype=rollout.values.dtype)
        )
        values = jnp.concatenate([rollout.values, bootstrap])
    advantages, returns = compute_gae(
        rollout.rewards,
        values,
        rollout.dones,
        config.gamma,
        config.gae_lambda,
    )
    if config.normalize_advantages:
        advantages = normalize_advantages(advantages)
    return PPOUpdateBatchFlax(
        states=jax.lax.stop_gradient(rollout.states),
        actions=jax.lax.stop_gradient(rollout.actions),
        old_log_probs=jax.lax.stop_gradient(rollout.old_log_probs),
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        old_values=jax.lax.stop_gradient(rollout.values),
        rewards=jax.lax.stop_gradient(rollout.rewards),
        turnovers=jax.lax.stop_gradient(rollout.turnovers),
        transaction_costs=jax.lax.stop_gradient(rollout.transaction_costs),
        drawdowns=jax.lax.stop_gradient(rollout.drawdowns),
    )


def _take_update_batch(batch: PPOUpdateBatchFlax, indices: Array) -> PPOUpdateBatchFlax:
    return PPOUpdateBatchFlax(
        **{
            name: value[indices]
            for name, value in batch._asdict().items()
        }
    )


def _minibatches_from_indices(
    batch: PPOUpdateBatchFlax,
    minibatch_size: int,
    rng: Array,
) -> tuple[PPOUpdateBatchFlax, ...]:
    n_steps = int(batch.rewards.shape[0])
    indices = jax.random.permutation(rng, jnp.arange(n_steps))
    minibatches = []
    for start in range(0, n_steps, minibatch_size):
        stop = min(start + minibatch_size, n_steps)
        minibatches.append(_take_update_batch(batch, indices[start:stop]))
    return tuple(minibatches)


def _loss_for_params(
    actor_params: dict[str, object],
    critic_params: dict[str, object],
    batch: PPOUpdateBatchFlax,
    config: ProductionPPOConfig,
) -> tuple[Array, PPOTrainMetrics]:
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    logits = jax.vmap(lambda state: actor.apply({"params": actor_params}, state))(
        batch.states
    )
    values = jax.vmap(lambda state: critic.apply({"params": critic_params}, state))(
        batch.states
    )
    new_log_probs = jax.vmap(
        lambda logit, action: action_log_prob(logit, action, config)
    )(logits, batch.actions)
    entropies = jax.vmap(lambda logit: policy_entropy(logit, config))(logits)
    actor_loss = ppo_actor_loss(
        new_log_probs,
        batch.old_log_probs,
        batch.advantages,
        config.clip_epsilon,
    )
    value_loss = critic_loss(
        values,
        batch.returns,
        old_values=batch.old_values,
        clip_epsilon=config.value_clip_epsilon,
        use_clipping=config.use_value_clipping,
        loss_type=config.value_loss_type,
        huber_delta=config.value_huber_delta,
    )
    entropy = jnp.mean(entropies)
    allocation_entropy = jnp.mean(portfolio_allocation_entropy(batch.actions))
    ratios = jnp.exp(new_log_probs - batch.old_log_probs)
    total = ppo_total_loss(
        actor_loss,
        value_loss,
        entropy,
        config.value_coef,
        config.entropy_coef,
    ) + config.portfolio_entropy_coef * allocation_entropy
    metrics = PPOTrainMetrics(
        policy_loss=actor_loss,
        actor_loss=actor_loss,
        critic_loss=value_loss,
        total_loss=total,
        entropy=entropy,
        approx_kl=approximate_kl(batch.old_log_probs, new_log_probs),
        post_update_approx_kl=approximate_kl(batch.old_log_probs, new_log_probs),
        clip_fraction=clip_fraction(
            batch.old_log_probs,
            new_log_probs,
            config.clip_epsilon,
        ),
        explained_variance=explained_variance(values, batch.returns),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        mean_episode_return=jnp.sum(batch.rewards),
        mean_reward=jnp.mean(batch.rewards),
        portfolio_entropy=allocation_entropy,
        effective_assets=jnp.exp(allocation_entropy),
        advantage_mean=jnp.mean(batch.advantages),
        advantage_std=jnp.std(batch.advantages),
        ratio_mean=jnp.mean(ratios),
        ratio_min=jnp.min(ratios),
        ratio_max=jnp.max(ratios),
        mean_turnover=jnp.mean(batch.turnovers),
        mean_transaction_cost=jnp.mean(batch.transaction_costs),
        mean_drawdown=jnp.mean(batch.drawdowns),
        updates_completed=jnp.asarray(0.0, dtype=jnp.float32),
        epochs_completed=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return total, metrics


def update_minibatch(
    train_state: ProductionPPOTrainState,
    batch: PPOUpdateBatchFlax,
) -> tuple[ProductionPPOTrainState, PPOTrainMetrics]:
    """Apply one PPO minibatch update."""

    config = train_state.config

    def loss_fn(
        actor_params: dict[str, object],
        critic_params: dict[str, object],
    ) -> Array:
        loss, _ = _loss_for_params(actor_params, critic_params, batch, config)
        return loss

    loss, grads = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1),
    )(train_state.actor.params, train_state.critic.params)
    del loss
    actor_grads, critic_grads = grads
    grad_norm = _tree_global_norm(grads)
    scale = jnp.minimum(
        1.0,
        train_state.config.max_grad_norm / (grad_norm + 1e-8),
    )
    actor_grads = jax.tree_util.tree_map(lambda grad: grad * scale, actor_grads)
    critic_grads = jax.tree_util.tree_map(lambda grad: grad * scale, critic_grads)
    _, pre_update_metrics = _loss_for_params(
        train_state.actor.params,
        train_state.critic.params,
        batch,
        config,
    )
    new_actor = train_state.actor.apply_gradients(grads=actor_grads)
    new_critic = train_state.critic.apply_gradients(grads=critic_grads)
    _, post_update_metrics = _loss_for_params(
        new_actor.params,
        new_critic.params,
        batch,
        config,
    )
    metrics = dataclass_replace(
        pre_update_metrics,
        grad_norm=grad_norm,
        post_update_approx_kl=post_update_metrics.approx_kl,
        updates_completed=jnp.asarray(1.0, dtype=jnp.float32),
    )
    return (
        ProductionPPOTrainState(
            actor=new_actor,
            critic=new_critic,
            config=config,
        ),
        metrics,
    )


def dataclass_replace(metrics: PPOTrainMetrics, **updates: Array) -> PPOTrainMetrics:
    """Return metrics with selected fields replaced."""

    values = {
        name: getattr(metrics, name)
        for name in PPOTrainMetrics.__dataclass_fields__
    }
    values.update(updates)
    return PPOTrainMetrics(**values)


def _mean_metrics(metrics: list[PPOTrainMetrics]) -> PPOTrainMetrics:
    if not metrics:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return PPOTrainMetrics(*(zero for _ in PPOTrainMetrics.__dataclass_fields__))
    values = {}
    for name in PPOTrainMetrics.__dataclass_fields__:
        values[name] = jnp.mean(jnp.stack([getattr(item, name) for item in metrics]))
    return PPOTrainMetrics(**values)


def train_epoch(
    train_state: ProductionPPOTrainState,
    batch: PPOUpdateBatchFlax,
    rng: Array,
    logger: TensorBoardLogger | None = None,
    step_offset: int = 0,
) -> tuple[ProductionPPOTrainState, PPOTrainMetrics]:
    """Train over one frozen rollout for up to ``config.update_epochs``."""

    state = train_state
    all_metrics: list[PPOTrainMetrics] = []
    epochs_completed = 0
    for epoch in range(state.config.update_epochs):
        epoch_key = jax.random.fold_in(rng, epoch)
        minibatches = _minibatches_from_indices(
            batch,
            state.config.minibatch_size,
            epoch_key,
        )
        stop = False
        for minibatch in minibatches:
            state, metrics = update_minibatch(state, minibatch)
            all_metrics.append(metrics)
            if logger is not None:
                logger.log_scalars(
                    ppo_metrics_to_dict(metrics),
                    step_offset + len(all_metrics),
                    "production_ppo",
                )
            post_update_kl = float(jax.device_get(metrics.post_update_approx_kl))
            if post_update_kl > state.config.target_kl:
                stop = True
                break
        epochs_completed += 1
        if stop:
            break
    merged = _mean_metrics(all_metrics)
    merged = dataclass_replace(
        merged,
        updates_completed=jnp.asarray(len(all_metrics), dtype=jnp.float32),
        epochs_completed=jnp.asarray(epochs_completed, dtype=jnp.float32),
    )
    return state, merged


def train_ppo_on_split(
    phi: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    config: ProductionPPOConfig,
    rng: Array,
    rollout_length: int | None = None,
    logger: TensorBoardLogger | None = None,
) -> ProductionPPOTrainingResult:
    """Collect a train rollout and optimize production PPO on that split."""

    init_key, rollout_key, train_key = jax.random.split(rng, 3)
    state = initialize_ppo_train_state(init_key, config)
    rollout = collect_rollout(
        {"params": state.actor.params},
        {"params": state.critic.params},
        phi,
        regime_probs,
        asset_returns,
        spy_returns,
        initial_env_state,
        env_config,
        config,
        rollout_key,
        rollout_length=rollout_length,
    )
    update_batch = freeze_rollout_batch(rollout.batch, config, rollout.bootstrap_value)
    state, metrics = train_epoch(state, update_batch, train_key, logger)
    return ProductionPPOTrainingResult(
        train_state=state,
        rollout=rollout,
        metrics=metrics,
    )


def evaluate_frozen_policy(
    train_state: ProductionPPOTrainState,
    phi: Array,
    regime_probs: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    rng: Array,
    rollout_length: int | None = None,
) -> ProductionPPOEvaluationResult:
    """Evaluate a frozen policy without updating params or optimizer state."""

    rollout = collect_rollout(
        {"params": train_state.actor.params},
        {"params": train_state.critic.params},
        phi,
        regime_probs,
        asset_returns,
        spy_returns,
        initial_env_state,
        env_config,
        train_state.config,
        rng,
        rollout_length=rollout_length,
        deterministic=True,
    )
    return ProductionPPOEvaluationResult(train_state=train_state, rollout=rollout)
