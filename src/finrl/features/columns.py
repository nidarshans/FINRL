"""Explicit feature routing contracts for learned allocation policies."""

from __future__ import annotations

from dataclasses import dataclass


DIRECT_ALLOCATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "mr_ewma50_vol_gap",
    "acc_macd_signal",
    "acc_klinger_signal",
    "macd_signal_strength",
    "klinger_signal_strength",
    "acc_momentum_quality"
)


@dataclass(frozen=True, slots=True)
class DirectAllocationRoutingMetadata:
    """Selected direct-allocation feature names and integer positions."""

    direct_allocation_feature_names: tuple[str, ...]
    direct_allocation_indices: tuple[int, ...]


def selected_direct_allocation_indices(
    feature_columns: tuple[str, ...],
) -> DirectAllocationRoutingMetadata:
    """Return explicit direct-allocation feature names and indices.

    The fixed allowlist keeps similarly named diagnostics or leakage-prone helper
    columns out of the learned allocation policy.
    """

    positions = {name: index for index, name in enumerate(feature_columns)}
    missing = tuple(
        name
        for name in DIRECT_ALLOCATION_FEATURE_COLUMNS
        if name not in positions
    )
    if missing:
        raise ValueError(
            "Missing required asset feature columns for policy routing: "
            + ", ".join(missing)
        )
    return DirectAllocationRoutingMetadata(
        direct_allocation_feature_names=DIRECT_ALLOCATION_FEATURE_COLUMNS,
        direct_allocation_indices=tuple(
            positions[name] for name in DIRECT_ALLOCATION_FEATURE_COLUMNS
        ),
    )
