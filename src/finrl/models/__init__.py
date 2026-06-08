"""Production Flax market representation models package."""

from finrl.models.encoder_losses import (
    EncoderLossWeights,
    EncoderPredictionHeads,
    encoder_loss,
)
from finrl.models.encoder_metrics import (
    EncoderTrainMetrics,
    encoder_metrics_to_dict,
    finite_encoder_metrics,
)
from finrl.models.encoder_training import (
    EncoderBatch,
    EncoderTrainingConfig,
    EncoderTrainingResult,
    fit_encoder_on_train_split,
    init_encoder_pretraining_state,
    make_encoder_batches,
    train_encoder_epoch,
)
from finrl.models.flax_encoder import (
    AssetLSTMEncoder,
    AttentionPool,
    CrossAssetSelfAttention,
    EncoderOutput,
    MacroLSTMEncoder,
    MarketEncoderFlax,
    ProductionEncoderConfig,
    encode_market_state_flax,
    encode_market_state_with_latents_flax,
    init_encoder_variables,
)
from finrl.models.windows import LookbackWindows, build_lookback_windows

__all__ = [
    "AssetLSTMEncoder",
    "AttentionPool",
    "CrossAssetSelfAttention",
    "EncoderBatch",
    "EncoderLossWeights",
    "EncoderPredictionHeads",
    "EncoderOutput",
    "EncoderTrainingConfig",
    "EncoderTrainingResult",
    "EncoderTrainMetrics",
    "LookbackWindows",
    "MacroLSTMEncoder",
    "MarketEncoderFlax",
    "ProductionEncoderConfig",
    "build_lookback_windows",
    "encode_market_state_flax",
    "encode_market_state_with_latents_flax",
    "encoder_loss",
    "encoder_metrics_to_dict",
    "fit_encoder_on_train_split",
    "finite_encoder_metrics",
    "init_encoder_pretraining_state",
    "init_encoder_variables",
    "make_encoder_batches",
    "train_encoder_epoch",
]
