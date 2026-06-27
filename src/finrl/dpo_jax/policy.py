"""Score-only direct allocation policy."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from finrl.dpo_jax.allocation_head import DirectAllocationHead
from finrl.dpo_jax.config import DPOConfig
from finrl.models.score_heads import AssetScoreHeads, slice_score_head_components
from finrl.types import Array


class ScoreAllocationPolicy(nn.Module):
    """Map raw features to two scores, then map only scores to weights."""

    accumulation_indices: tuple[int, ...]
    liquidity_indices: tuple[int, ...]
    accumulation_hidden_dims: tuple[int, ...]
    accumulation_hidden_activation: str
    accumulation_output_activation: str
    accumulation_use_layer_norm: bool
    liquidity_exit_hidden_dims: tuple[int, ...]
    liquidity_exit_hidden_activation: str
    liquidity_exit_output_activation: str
    liquidity_exit_use_layer_norm: bool
    allocation_hidden_dims: tuple[int, ...]
    allocation_hidden_activation: str = "tanh"
    allocation_output_activation: str = "identity"
    allocation_use_layer_norm: bool = True
    simplex_activation: str = "softmax"

    @nn.compact
    def __call__(self, asset_features: Array) -> Array:
        acc_components, liq_components = slice_score_head_components(
            asset_features,
            self.accumulation_indices,
            self.liquidity_indices,
        )
        accumulation, liquidity_exit = AssetScoreHeads(
            accumulation_hidden_dims=self.accumulation_hidden_dims,
            accumulation_use_layer_norm=self.accumulation_use_layer_norm,
            accumulation_hidden_activation=self.accumulation_hidden_activation,
            accumulation_output_activation=self.accumulation_output_activation,
            liquidity_exit_hidden_dims=self.liquidity_exit_hidden_dims,
            liquidity_exit_use_layer_norm=self.liquidity_exit_use_layer_norm,
            liquidity_exit_hidden_activation=self.liquidity_exit_hidden_activation,
            liquidity_exit_output_activation=self.liquidity_exit_output_activation,
            name="score_heads",
        )(acc_components, liq_components)
        scores = jnp.stack((accumulation, liquidity_exit), axis=-1)
        return DirectAllocationHead(
            hidden_dims=self.allocation_hidden_dims,
            simplex_activation=self.simplex_activation,
            hidden_activation=self.allocation_hidden_activation,
            output_activation=self.allocation_output_activation,
            use_layer_norm=self.allocation_use_layer_norm,
            name="allocation_head",
        )(scores)


def build_score_allocation_policy(
    config: DPOConfig,
    accumulation_indices: tuple[int, ...],
    liquidity_indices: tuple[int, ...],
) -> ScoreAllocationPolicy:
    """Build the policy from its explicit experiment configuration."""

    return ScoreAllocationPolicy(
        accumulation_indices=accumulation_indices,
        liquidity_indices=liquidity_indices,
        accumulation_hidden_dims=config.accumulation_hidden_dims,
        accumulation_hidden_activation=config.accumulation_hidden_activation,
        accumulation_output_activation=config.accumulation_output_activation,
        accumulation_use_layer_norm=config.accumulation_use_layer_norm,
        liquidity_exit_hidden_dims=config.liquidity_exit_hidden_dims,
        liquidity_exit_hidden_activation=config.liquidity_exit_hidden_activation,
        liquidity_exit_output_activation=config.liquidity_exit_output_activation,
        liquidity_exit_use_layer_norm=config.liquidity_exit_use_layer_norm,
        allocation_hidden_dims=config.allocation_hidden_dims,
        allocation_hidden_activation=config.allocation_hidden_activation,
        allocation_output_activation=config.allocation_output_activation,
        allocation_use_layer_norm=config.allocation_use_layer_norm,
        simplex_activation=config.simplex_activation,
    )
