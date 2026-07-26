"""Immutable configuration for the 3M policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TreeConfig:
    """Parameters passed to each scikit-learn histogram GBT classifier."""

    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    max_depth: int | None = 5
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    max_bins: int = 255

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.max_iter <= 0 or self.max_leaf_nodes < 2:
            raise ValueError("max_iter must be positive and max_leaf_nodes at least two.")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or None.")
        if self.min_samples_leaf <= 0 or not 2 <= self.max_bins <= 255:
            raise ValueError("min_samples_leaf must be positive and max_bins in [2, 255].")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")

    def estimator_parameters(self, seed: int) -> dict[str, object]:
        """Return deterministic, chronology-safe estimator parameters."""

        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "max_bins": self.max_bins,
            "early_stopping": False,
            "random_state": seed,
        }


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Event-based buy, hold, and sell label controls."""

    short_horizon: int = 5
    medium_horizon: int = 20
    long_horizon: int = 60
    outcome_horizon: int = 20
    buy_min_return: float = 0.05
    sell_min_drawdown: float = 0.05
    ema50_epsilon: float = 0.01
    round_trip_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.short_horizon <= self.medium_horizon <= self.long_horizon:
            raise ValueError("Horizons must be positive and increasing.")
        if self.outcome_horizon <= 0:
            raise ValueError("outcome_horizon must be positive.")
        if not 0.0 < self.buy_min_return < 1.0 or not 0.0 < self.sell_min_drawdown < 1.0:
            raise ValueError("Buy-return and sell-drawdown thresholds must be in (0, 1).")
        if not 0.0 <= self.ema50_epsilon < 1.0:
            raise ValueError("ema50_epsilon must be in [0, 1).")
        if self.round_trip_cost < 0.0:
            raise ValueError("round_trip_cost must be non-negative.")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Deterministic state-gate and cash-aware allocation controls."""

    buy_threshold: float = 0.60
    hold_threshold: float = 0.45
    sell_threshold: float = 0.60
    entry_weight: float = 0.05
    max_positions: int = 20
    max_position_weight: float = 0.10
    holding_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("buy_threshold", "hold_threshold", "sell_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.entry_weight <= 0.0 or not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("Entry and maximum position weights must be in (0, 1].")
        if self.entry_weight > self.max_position_weight:
            raise ValueError("entry_weight cannot exceed max_position_weight.")
        if self.max_positions <= 0 or self.holding_epsilon < 0.0:
            raise ValueError("max_positions must be positive and holding_epsilon non-negative.")
