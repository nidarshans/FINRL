"""Market representation models package."""

from finrl.models.attention import AttentionPooling, CrossAssetAttention
from finrl.models.encoder import (
    AssetEncoder,
    EncoderConfig,
    FeatureWindow,
    FusionMLP,
    MacroEncoder,
    MarketEncoder,
    encode_market_state,
)
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
from finrl.models.encoder_state import init_encoder_train_state
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
    MacroLSTMEncoder,
    MarketEncoderFlax,
    ProductionEncoderConfig,
    encode_market_state_flax,
    init_encoder_variables,
)
from finrl.models.windows import LookbackWindows, build_lookback_windows

__all__ = [
    "AssetEncoder",
    "AssetLSTMEncoder",
    "AttentionPool",
    "AttentionPooling",
    "CrossAssetAttention",
    "CrossAssetSelfAttention",
    "EncoderConfig",
    "EncoderBatch",
    "EncoderLossWeights",
    "EncoderPredictionHeads",
    "EncoderTrainingConfig",
    "EncoderTrainingResult",
    "EncoderTrainMetrics",
    "FeatureWindow",
    "FusionMLP",
    "LookbackWindows",
    "MacroEncoder",
    "MacroLSTMEncoder",
    "MarketEncoder",
    "MarketEncoderFlax",
    "ProductionEncoderConfig",
    "build_lookback_windows",
    "encode_market_state",
    "encode_market_state_flax",
    "encoder_loss",
    "encoder_metrics_to_dict",
    "fit_encoder_on_train_split",
    "finite_encoder_metrics",
    "init_encoder_pretraining_state",
    "init_encoder_train_state",
    "init_encoder_variables",
    "make_encoder_batches",
    "train_encoder_epoch",
]
