"""Explicit feature routing contracts for learned allocation policies."""

from __future__ import annotations

from dataclasses import dataclass


DIRECT_ALLOCATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "mr_ewma50_vol_gap",
    "ewma50_slope",
    "acc_macd_signal",
    "acc_klinger_signal",
    "macd_signal_strength",
    "klinger_signal_strength",
    "acc_momentum_quality",
    "cmf",
    "cmf_slope",
    "cmf_cross_signal",
    "cmf_days_since_cross",
    "frog_in_the_pan",
    "bollinger_bandwidth",
    "fip_over_bollinger_bandwidth",
)

# The legacy constant remains the default public contract.  Named sets make
# ablations explicit without allowing prefix-based or incidental routing.
BASELINE_CURRENT_14: tuple[str, ...] = DIRECT_ALLOCATION_FEATURE_COLUMNS
MOMENTUM_FEATURE_COLUMNS: tuple[str, ...] = (
    "mom_21d",
    "mom_126_21d",
    "near_52w_high",
)
MOMENTUM_RANK_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_percentile_rank" for column in MOMENTUM_FEATURE_COLUMNS
)
LIQUIDITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_adv_20",
    "volume_z_20",
    "amihud_20",
)
LIQUIDITY_RANK_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_percentile_rank" for column in LIQUIDITY_FEATURE_COLUMNS
)
STRUCTURE_FEATURE_COLUMNS: tuple[str, ...] = (
    "confirmed_structure_score",
    "support_distance_atr",
    "resistance_distance_atr",
    "swing_avwap_distance_atr",
    "bars_since_swing_low",
)
MARKET_RELATIVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "relative_strength_63",
    "beta_252",
    "residual_mom_126_21",
    "idio_vol_60",
)
RISK_FEATURE_COLUMNS: tuple[str, ...] = (
    "natr_20",
    "realized_vol_20",
    "downside_vol_60",
    "max_drawdown_126",
)


@dataclass(frozen=True, slots=True)
class FeatureSetConfig:
    """An ordered, explicitly allowlisted policy feature set."""

    name: str
    routed_columns: tuple[str, ...]


FEATURE_SETS: dict[str, FeatureSetConfig] = {
    "baseline_current_14": FeatureSetConfig(
        "baseline_current_14", BASELINE_CURRENT_14
    ),
    "baseline_plus_momentum": FeatureSetConfig(
        "baseline_plus_momentum", BASELINE_CURRENT_14 + MOMENTUM_FEATURE_COLUMNS
    ),
    "baseline_plus_momentum_ranked": FeatureSetConfig(
        "baseline_plus_momentum_ranked",
        BASELINE_CURRENT_14 + MOMENTUM_RANK_FEATURE_COLUMNS,
    ),
    "baseline_plus_liquidity": FeatureSetConfig(
        "baseline_plus_liquidity", BASELINE_CURRENT_14 + LIQUIDITY_FEATURE_COLUMNS
    ),
    "baseline_plus_liquidity_ranked": FeatureSetConfig(
        "baseline_plus_liquidity_ranked",
        BASELINE_CURRENT_14 + LIQUIDITY_RANK_FEATURE_COLUMNS,
    ),
    "baseline_plus_structure": FeatureSetConfig(
        "baseline_plus_structure", BASELINE_CURRENT_14 + STRUCTURE_FEATURE_COLUMNS
    ),
    "baseline_plus_market_relative": FeatureSetConfig(
        "baseline_plus_market_relative",
        BASELINE_CURRENT_14 + MARKET_RELATIVE_FEATURE_COLUMNS,
    ),
    "baseline_plus_risk": FeatureSetConfig(
        "baseline_plus_risk", BASELINE_CURRENT_14 + RISK_FEATURE_COLUMNS
    ),
    "institutional_core_v1": FeatureSetConfig(
        "institutional_core_v1",
        BASELINE_CURRENT_14
        + MOMENTUM_FEATURE_COLUMNS
        + RISK_FEATURE_COLUMNS
        + LIQUIDITY_FEATURE_COLUMNS,
    ),
    "institutional_core_plus_structure_v1": FeatureSetConfig(
        "institutional_core_plus_structure_v1",
        BASELINE_CURRENT_14
        + MOMENTUM_FEATURE_COLUMNS
        + RISK_FEATURE_COLUMNS
        + LIQUIDITY_FEATURE_COLUMNS
        + STRUCTURE_FEATURE_COLUMNS,
    ),
}


def feature_set_config(name: str) -> FeatureSetConfig:
    """Return a named routing configuration or fail before model creation."""

    try:
        return FEATURE_SETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown feature set: {name}.") from error


@dataclass(frozen=True, slots=True)
class DirectAllocationRoutingMetadata:
    """Selected direct-allocation feature names and integer positions."""

    direct_allocation_feature_names: tuple[str, ...]
    direct_allocation_indices: tuple[int, ...]


def selected_direct_allocation_indices(
    feature_columns: tuple[str, ...],
    feature_set: str = "baseline_current_14",
) -> DirectAllocationRoutingMetadata:
    """Return explicit direct-allocation feature names and indices.

    The fixed allowlist keeps similarly named diagnostics or leakage-prone helper
    columns out of the learned allocation policy.
    """

    selected_columns = feature_set_config(feature_set).routed_columns
    positions = {name: index for index, name in enumerate(feature_columns)}
    missing = tuple(
        name
        for name in selected_columns
        if name not in positions
    )
    if missing:
        raise ValueError(
            "Missing required asset feature columns for policy routing: "
            + ", ".join(missing)
        )
    return DirectAllocationRoutingMetadata(
        direct_allocation_feature_names=selected_columns,
        direct_allocation_indices=tuple(
            positions[name] for name in selected_columns
        ),
    )
