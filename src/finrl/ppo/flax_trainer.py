"""Production Flax PPO optimization loop for asset-only training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.features.columns import FeatureRoutingMetadata
from finrl.logging.tensorboard import TensorBoardLogger
from finrl.models.asset_encoder import AssetOnlyEncoder, AssetOnlyEncoderConfig
from finrl.ppo.flax_policy import (
    PortfolioActorFlax,
    ProductionPPOConfig,
    build_structured_ppo_state,
)
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
    """PPO minibatch tensors that keep raw asset windows in the graph."""

    asset_windows: Array
    prev_weights: Array
    drawdowns: Array
    prev_turnovers: Array
    actions: Array
    old_log_probs: Array
    advantages: Array
    returns: Array
    old_values: Array
    rewards: Array
    turnovers: Array
    transaction_costs: Array


@dataclass(frozen=True, slots=True)
class ProductionPPOTrainState:
    """Single optimizer state for encoder, actor, and critic params."""

    policy: TrainState
    config: ProductionPPOConfig
    encoder_config: AssetOnlyEncoderConfig
    accumulation_indices: tuple[int, ...]
    liquidity_indices: tuple[int, ...]
    feature_routing: FeatureRoutingMetadata | None = None


@dataclass(frozen=True, slots=True)
class ProductionPPOTrainingResult:
    """Result from fitting production PPO on one split."""

    train_state: ProductionPPOTrainState
    rollout: RolloutBuffer
    metrics: PPOTrainMetrics
    feature_routing: FeatureRoutingMetadata | None = None


@dataclass(frozen=True, slots=True)
class ProductionPPOEvaluationResult:
    """Production PPO evaluation result."""

    train_state: ProductionPPOTrainState
    rollout: RolloutBuffer


def _optimizer(config: ProductionPPOConfig) -> optax.GradientTransformation:
    return optax.adam(config.learning_rate)


def _tree_global_norm(tree: object) -> Array:
    squared_norms = [jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.asarray(squared_norms)))


def _modules(
    config: ProductionPPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
) -> tuple[AssetOnlyEncoder, PortfolioActorFlax, PortfolioCriticFlax]:
    return (
        AssetOnlyEncoder(
            encoder_config,
            accumulation_indices=accumulation_indices,
            liquidity_indices=liquidity_indices,
        ),
        PortfolioActorFlax(config),
        PortfolioCriticFlax(config),
    )


def initialize_ppo_train_state(
    rng: Array,
    config: ProductionPPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
    feature_routing: FeatureRoutingMetadata | None = None,
) -> ProductionPPOTrainState:
    """Initialize one train state for encoder, actor, and critic."""

    encoder_key, actor_key, critic_key = jax.random.split(rng, 3)
    encoder, actor, critic = _modules(
        config,
        encoder_config,
        accumulation_indices,
        liquidity_indices,
    )
    example_windows = jnp.zeros(
        (
            1,
            encoder_config.lookback,
            encoder_config.n_assets,
            encoder_config.asset_feature_dim,
        ),
        dtype=jnp.float32,
    )
    embeddings = encoder.init(encoder_key, example_windows)["params"]
    example_state = build_structured_ppo_state(
        asset_embeddings=jnp.zeros((config.action_dim - 1, config.asset_latent_dim), dtype=jnp.float32),
        prev_weights=jnp.zeros((config.action_dim,), dtype=jnp.float32).at[-1].set(1.0),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
    )
    params = {
        "encoder": embeddings,
        "actor": actor.init(actor_key, example_state)["params"],
        "critic": critic.init(critic_key, example_state)["params"],
    }
    return ProductionPPOTrainState(
        policy=TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params=params,
            tx=_optimizer(config),
        ),
        config=config,
        encoder_config=encoder_config,
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
        feature_routing=feature_routing,
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


def build_update_batch(
    rollout: RolloutBatch,
    config: ProductionPPOConfig,
    bootstrap_value: Array | None = None,
) -> PPOUpdateBatchFlax:
    """Compute PPO targets while preserving raw asset windows for the loss."""

    values = rollout.values
    if bootstrap_value is not None:
        bootstrap = jnp.atleast_1d(jnp.asarray(bootstrap_value, dtype=rollout.values.dtype))
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
        asset_windows=rollout.asset_windows,
        prev_weights=rollout.prev_weights,
        drawdowns=rollout.drawdowns,
        prev_turnovers=rollout.prev_turnovers,
        actions=rollout.actions,
        old_log_probs=rollout.old_log_probs,
        advantages=jax.lax.stop_gradient(advantages),
        returns=jax.lax.stop_gradient(returns),
        old_values=rollout.values,
        rewards=rollout.rewards,
        turnovers=rollout.turnovers,
        transaction_costs=rollout.transaction_costs,
    )


def _take_update_batch(batch: PPOUpdateBatchFlax, indices: Array) -> PPOUpdateBatchFlax:
    return PPOUpdateBatchFlax(
        **{
            name: jax.tree_util.tree_map(lambda leaf: leaf[indices], value)
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
    return tuple(
        _take_update_batch(batch, indices[start : min(start + minibatch_size, n_steps)])
        for start in range(0, n_steps, minibatch_size)
    )


def _states_from_windows(
    params: dict[str, object],
    batch: PPOUpdateBatchFlax,
    config: ProductionPPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
):
    encoder = AssetOnlyEncoder(
        encoder_config,
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
    )
    embeddings = encoder.apply({"params": params["encoder"]}, batch.asset_windows)
    return build_structured_ppo_state(
        asset_embeddings=embeddings,
        prev_weights=batch.prev_weights,
        drawdown=batch.drawdowns,
        previous_turnover=batch.prev_turnovers,
    )


def _loss_for_params(
    params: dict[str, object],
    batch: PPOUpdateBatchFlax,
    train_state: ProductionPPOTrainState,
) -> tuple[Array, PPOTrainMetrics]:
    config = train_state.config
    states = _states_from_windows(
        params,
        batch,
        config,
        train_state.encoder_config,
        train_state.accumulation_indices,
        train_state.liquidity_indices,
    )
    actor = PortfolioActorFlax(config)
    critic = PortfolioCriticFlax(config)
    logits = jax.vmap(lambda state: actor.apply({"params": params["actor"]}, state))(states)
    values = jax.vmap(lambda state: critic.apply({"params": params["critic"]}, state))(states)
    new_log_probs = jax.vmap(lambda logit, action: action_log_prob(logit, action, config))(
        logits,
        batch.actions,
    )
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
        allocation_entropy,
        config.portfolio_entropy_coef,
    )
    metrics = PPOTrainMetrics(
        policy_loss=actor_loss,
        actor_loss=actor_loss,
        critic_loss=value_loss,
        total_loss=total,
        entropy=entropy,
        approx_kl=approximate_kl(batch.old_log_probs, new_log_probs),
        post_update_approx_kl=approximate_kl(batch.old_log_probs, new_log_probs),
        clip_fraction=clip_fraction(batch.old_log_probs, new_log_probs, config.clip_epsilon),
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


def dataclass_replace(metrics: PPOTrainMetrics, **updates: Array) -> PPOTrainMetrics:
    values = {name: getattr(metrics, name) for name in PPOTrainMetrics.__dataclass_fields__}
    values.update(updates)
    return PPOTrainMetrics(**values)


def update_minibatch(
    train_state: ProductionPPOTrainState,
    batch: PPOUpdateBatchFlax,
) -> tuple[ProductionPPOTrainState, PPOTrainMetrics]:
    """Apply one PPO minibatch update to encoder, actor, and critic."""

    def loss_fn(params: dict[str, object]) -> Array:
        loss, _ = _loss_for_params(params, batch, train_state)
        return loss

    _, grads = jax.value_and_grad(loss_fn)(train_state.policy.params)
    grad_norm = _tree_global_norm(grads)
    scale = jnp.minimum(1.0, train_state.config.max_grad_norm / (grad_norm + 1e-8))
    grads = jax.tree_util.tree_map(lambda grad: grad * scale, grads)
    _, pre_update_metrics = _loss_for_params(train_state.policy.params, batch, train_state)
    new_policy = train_state.policy.apply_gradients(grads=grads)
    updated = ProductionPPOTrainState(
        policy=new_policy,
        config=train_state.config,
        encoder_config=train_state.encoder_config,
        accumulation_indices=train_state.accumulation_indices,
        liquidity_indices=train_state.liquidity_indices,
    )
    _, post_update_metrics = _loss_for_params(new_policy.params, batch, updated)
    metrics = dataclass_replace(
        pre_update_metrics,
        grad_norm=grad_norm,
        post_update_approx_kl=post_update_metrics.approx_kl,
        updates_completed=jnp.asarray(1.0, dtype=jnp.float32),
    )
    return updated, metrics


def _mean_metrics(metrics: list[PPOTrainMetrics]) -> PPOTrainMetrics:
    if not metrics:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return PPOTrainMetrics(*(zero for _ in PPOTrainMetrics.__dataclass_fields__))
    return PPOTrainMetrics(
        **{
            name: jnp.mean(jnp.stack([getattr(item, name) for item in metrics]))
            for name in PPOTrainMetrics.__dataclass_fields__
        }
    )


def train_epoch(
    train_state: ProductionPPOTrainState,
    batch: PPOUpdateBatchFlax,
    rng: Array,
    logger: TensorBoardLogger | None = None,
    step_offset: int = 0,
) -> tuple[ProductionPPOTrainState, PPOTrainMetrics]:
    """Train over one rollout for up to ``config.update_epochs``."""

    state = train_state
    all_metrics: list[PPOTrainMetrics] = []
    epochs_completed = 0
    for epoch in range(state.config.update_epochs):
        minibatches = _minibatches_from_indices(
            batch,
            state.config.minibatch_size,
            jax.random.fold_in(rng, epoch),
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
            if float(jax.device_get(metrics.post_update_approx_kl)) > state.config.target_kl:
                stop = True
                break
        epochs_completed += 1
        if stop:
            break
    merged = _mean_metrics(all_metrics)
    return state, dataclass_replace(
        merged,
        updates_completed=jnp.asarray(len(all_metrics), dtype=jnp.float32),
        epochs_completed=jnp.asarray(epochs_completed, dtype=jnp.float32),
    )


def train_ppo_on_split(
    asset_windows: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    config: ProductionPPOConfig,
    encoder_config: AssetOnlyEncoderConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
    rng: Array,
    rollout_length: int | None = None,
    logger: TensorBoardLogger | None = None,
    feature_routing: FeatureRoutingMetadata | None = None,
) -> ProductionPPOTrainingResult:
    """Collect a rollout and optimize asset-only PPO on that split."""

    init_key, rollout_key, train_key = jax.random.split(rng, 3)
    state = initialize_ppo_train_state(
        init_key,
        config,
        encoder_config,
        accumulation_indices,
        liquidity_indices,
        feature_routing,
    )
    rollout = collect_rollout(
        {"params": state.policy.params},
        asset_windows,
        asset_returns,
        spy_returns,
        initial_env_state,
        env_config,
        config,
        encoder_config,
        accumulation_indices,
        liquidity_indices,
        rollout_key,
        rollout_length=rollout_length,
    )
    update_batch = build_update_batch(rollout.batch, config, rollout.bootstrap_value)
    state, metrics = train_epoch(state, update_batch, train_key, logger)
    return ProductionPPOTrainingResult(
        train_state=state,
        rollout=rollout,
        metrics=metrics,
        feature_routing=feature_routing,
    )


def evaluate_policy(
    train_state: ProductionPPOTrainState,
    asset_windows: Array,
    asset_returns: Array,
    spy_returns: Array,
    initial_env_state: EnvState,
    env_config: EnvConfig,
    rng: Array,
    rollout_length: int | None = None,
) -> ProductionPPOEvaluationResult:
    """Evaluate a policy without updating params."""

    rollout = collect_rollout(
        {"params": train_state.policy.params},
        asset_windows,
        asset_returns,
        spy_returns,
        initial_env_state,
        env_config,
        train_state.config,
        train_state.encoder_config,
        train_state.accumulation_indices,
        train_state.liquidity_indices,
        rng,
        rollout_length=rollout_length,
        deterministic=True,
    )
    return ProductionPPOEvaluationResult(train_state=train_state, rollout=rollout)
