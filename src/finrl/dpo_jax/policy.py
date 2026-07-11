"""Direct-feature allocation policy."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from finrl.dpo_jax.allocation_head import DirectAllocationHead
from finrl.dpo_jax.config import DPOConfig
from finrl.types import Array


def slice_direct_allocation_features(
    asset_features: Array,
    direct_indices: tuple[int, ...],
) -> Array:
    """Select explicitly routed direct-allocation inputs from an asset panel."""

    features = jnp.asarray(asset_features, dtype=jnp.float32)
    return features[..., jnp.asarray(direct_indices)]


class DirectFeatureAllocationPolicy(nn.Module):
    """Map explicitly selected raw asset features directly to portfolio weights."""

    direct_feature_indices: tuple[int, ...]
    allocation_hidden_dims: tuple[int, ...]
    allocation_hidden_activation: str = "tanh"
    allocation_output_activation: str = "identity"
    allocation_use_layer_norm: bool = True
    simplex_activation: str = "softmax"

    @nn.compact
    def __call__(self, asset_features: Array, tradable_mask: Array | None = None) -> Array:
        direct_features = slice_direct_allocation_features(
            asset_features,
            self.direct_feature_indices,
        )
        return DirectAllocationHead(
            hidden_dims=self.allocation_hidden_dims,
            simplex_activation=self.simplex_activation,
            hidden_activation=self.allocation_hidden_activation,
            output_activation=self.allocation_output_activation,
            use_layer_norm=self.allocation_use_layer_norm,
            name="allocation_head",
        )(direct_features, tradable_mask=tradable_mask)


def build_allocation_policy(
    config: DPOConfig,
    direct_feature_indices: tuple[int, ...],
) -> DirectFeatureAllocationPolicy:
    """Build the direct-feature allocation policy."""

    if not direct_feature_indices:
        raise ValueError("Direct allocation requires routed feature indices.")
    return DirectFeatureAllocationPolicy(
        direct_feature_indices=direct_feature_indices,
        allocation_hidden_dims=config.allocation_hidden_dims,
        allocation_hidden_activation=config.allocation_hidden_activation,
        allocation_output_activation=config.allocation_output_activation,
        allocation_use_layer_norm=config.allocation_use_layer_norm,
        simplex_activation=config.simplex_activation,
    )
