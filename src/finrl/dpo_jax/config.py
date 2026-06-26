"""Configuration for JAX direct portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Minimal hyperparameters for differentiable portfolio optimization."""

    learning_rate: float = 3e-4
    num_epochs: int = 5
    batch_size: int = 32

    transaction_cost_bps: float = 5.0

    lambda_turnover: float = 0.01
    lambda_drawdown: float = 0.10
    lambda_concentration: float = 0.01

    allocation_activation: str = "sparsemax"
    dpo_score_hidden_dims: tuple[int, ...] = (64, 32, 16)
    dpo_allocation_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dpo_activation: str = "tanh"
    dpo_score_use_layer_norm: bool = True
    dpo_allocation_use_layer_norm: bool = True

    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.allocation_activation not in {"softmax", "sparsemax"}:
            raise ValueError("allocation_activation must be 'softmax' or 'sparsemax'.")
        _validate_hidden_dims("dpo_score_hidden_dims", self.dpo_score_hidden_dims)
        _validate_hidden_dims(
            "dpo_allocation_hidden_dims",
            self.dpo_allocation_hidden_dims,
        )
        if self.dpo_activation not in {"tanh", "gelu", "relu"}:
            raise ValueError("dpo_activation must be one of 'tanh', 'gelu', or 'relu'.")


def _validate_hidden_dims(name: str, hidden_dims: tuple[int, ...]) -> None:
    if not hidden_dims:
        raise ValueError(f"{name} must be non-empty.")
    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise ValueError(f"Every value in {name} must be positive.")
