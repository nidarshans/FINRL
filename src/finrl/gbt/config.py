"""Configuration for the gradient-boosted tree policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GBTConfig:
    """LightGBM and deterministic allocation parameters."""

    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 10
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    temperature: float = 0.01
    score_clip: float = 20.0
    target_horizons: tuple[int, ...] = (5, 20, 60)
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    max_position_weight: float | None = None
    smoothing_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least two.")
        if self.max_depth == 0 or self.max_depth < -1:
            raise ValueError("max_depth must be -1 or positive.")
        if self.min_child_samples <= 0:
            raise ValueError("min_child_samples must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.score_clip <= 0.0:
            raise ValueError("score_clip must be positive.")
        if not self.target_horizons or any(horizon <= 0 for horizon in self.target_horizons):
            raise ValueError("target_horizons must contain positive trading-day counts.")
        if tuple(sorted(self.target_horizons)) != tuple(self.target_horizons):
            raise ValueError("target_horizons must be strictly increasing.")
        if len(set(self.target_horizons)) != len(self.target_horizons):
            raise ValueError("target_horizons must be unique.")
        if len(self.target_weights) != len(self.target_horizons):
            raise ValueError("target_weights must match target_horizons.")
        if any(weight < 0.0 for weight in self.target_weights) or sum(self.target_weights) <= 0.0:
            raise ValueError("target_weights must be non-negative with a positive sum.")
        if self.max_position_weight is not None and not 0.0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1].")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1].")

    @property
    def normalized_target_weights(self) -> tuple[float, ...]:
        """Return target weights normalized to sum to one."""

        total = sum(self.target_weights)
        return tuple(weight / total for weight in self.target_weights)

    def model_parameters(self, seed: int) -> dict[str, object]:
        """Return deterministic CPU LightGBM parameters."""

        return {
            "objective": "regression",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": seed,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 1,
            "verbosity": -1,
        }
