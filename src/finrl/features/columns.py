"""Explicit feature routing contracts for asset score heads."""

from __future__ import annotations

from dataclasses import dataclass


ACCUMULATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "acc_price_drift",
    "acc_liquidity_growth",
    "acc_vol_compression",
    "acc_low_vol",
    "acc_macd_improvement",
    "acc_klinger_improvement",
    "acc_macd_bullish_hist",
    "acc_klinger_bullish_hist",
    "acc_macd_early",
    "acc_klinger_early",
    "acc_momentum_quality",
)

LIQUIDITY_EXIT_FEATURE_COLUMNS: tuple[str, ...] = (
    "liq_amihud_trend",
    "liq_liquidity_deterioration",
    "liq_klinger_deterioration",
    "liq_vol_expansion",
    "liq_liquidity_shock",
    "liq_momentum_quality",
)


@dataclass(frozen=True, slots=True)
class FeatureRoutingMetadata:
    """Selected score-head feature names and integer positions."""

    accumulation_feature_names: tuple[str, ...]
    accumulation_indices: tuple[int, ...]
    liquidity_exit_feature_names: tuple[str, ...]
    liquidity_exit_indices: tuple[int, ...]


def selected_feature_indices(
    feature_columns: tuple[str, ...],
) -> FeatureRoutingMetadata:
    """Return explicit score-head feature names and indices.

    The score heads intentionally use fixed allowlists instead of prefix routing.
    That keeps similarly named diagnostics or leakage-prone helper columns out of
    the learned accumulation and liquidity-exit scores.
    """

    positions = {name: index for index, name in enumerate(feature_columns)}
    missing = tuple(
        name
        for name in (*ACCUMULATION_FEATURE_COLUMNS, *LIQUIDITY_EXIT_FEATURE_COLUMNS)
        if name not in positions
    )
    if missing:
        raise ValueError(
            "Missing required asset feature columns for score-head routing: "
            + ", ".join(missing)
        )
    return FeatureRoutingMetadata(
        accumulation_feature_names=ACCUMULATION_FEATURE_COLUMNS,
        accumulation_indices=tuple(positions[name] for name in ACCUMULATION_FEATURE_COLUMNS),
        liquidity_exit_feature_names=LIQUIDITY_EXIT_FEATURE_COLUMNS,
        liquidity_exit_indices=tuple(positions[name] for name in LIQUIDITY_EXIT_FEATURE_COLUMNS),
    )
