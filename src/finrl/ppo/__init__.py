"""Production Flax PPO training package."""

from finrl.ppo.batches import (
    make_minibatches,
    rollout_length,
    shuffle_rollout_indices,
    validate_rollout_batch,
)
from finrl.ppo.flax_policy import (
    PortfolioActorFlax,
    PPOState,
    ProductionPortfolioAction,
    ProductionPPOConfig,
    actor_mean_weights,
    build_structured_ppo_state,
    sample_action as sample_flax_action,
)
from finrl.ppo.flax_trainer import (
    build_update_batch,
    PPOUpdateBatchFlax,
    ProductionPPOEvaluationResult,
    ProductionPPOTrainState,
    ProductionPPOTrainingResult,
    evaluate_policy as evaluate_flax_policy,
    initialize_ppo_train_state,
    portfolio_allocation_entropy,
    train_epoch,
    train_ppo_on_split as train_flax_ppo_on_split,
    update_minibatch,
)
from finrl.ppo.flax_value import PortfolioCriticFlax
from finrl.ppo.checkpoints import load_policy_checkpoint, save_policy_checkpoint
from finrl.ppo.gae import compute_gae
from finrl.ppo.losses import (
    clipped_value_loss,
    critic_loss,
    entropy_bonus,
    huber_value_loss,
    ppo_actor_loss,
    ppo_clip_loss,
    ppo_total_loss,
    value_loss,
)
from finrl.ppo.metrics import (
    PPOTrainMetrics,
    approximate_kl,
    clip_fraction,
    explained_variance,
    finite_ppo_metrics,
    ppo_metrics_to_dict,
)
from finrl.ppo.simplex_distribution import (
    DirichletPortfolioDistribution,
    action_log_prob,
    policy_entropy,
    validate_simplex_action,
)
from finrl.ppo.rollout import RolloutBatch, RolloutBuffer, collect_rollout

__all__ = [
    "DirichletPortfolioDistribution",
    "PortfolioActorFlax",
    "PPOState",
    "PortfolioCriticFlax",
    "ProductionPortfolioAction",
    "ProductionPPOConfig",
    "ProductionPPOEvaluationResult",
    "ProductionPPOTrainState",
    "ProductionPPOTrainingResult",
    "PPOTrainMetrics",
    "PPOUpdateBatchFlax",
    "RolloutBatch",
    "RolloutBuffer",
    "actor_mean_weights",
    "action_log_prob",
    "build_structured_ppo_state",
    "build_update_batch",
    "clipped_value_loss",
    "collect_rollout",
    "compute_gae",
    "critic_loss",
    "entropy_bonus",
    "evaluate_flax_policy",
    "explained_variance",
    "finite_ppo_metrics",
    "huber_value_loss",
    "initialize_ppo_train_state",
    "load_policy_checkpoint",
    "make_minibatches",
    "portfolio_allocation_entropy",
    "ppo_clip_loss",
    "ppo_actor_loss",
    "ppo_total_loss",
    "ppo_metrics_to_dict",
    "policy_entropy",
    "approximate_kl",
    "clip_fraction",
    "rollout_length",
    "sample_flax_action",
    "save_policy_checkpoint",
    "shuffle_rollout_indices",
    "train_epoch",
    "train_flax_ppo_on_split",
    "update_minibatch",
    "validate_rollout_batch",
    "validate_simplex_action",
    "value_loss",
]
