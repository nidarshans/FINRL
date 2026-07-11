"""Configuration for JAX direct portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Minimal hyperparameters for differentiable portfolio optimization."""

    learning_rate: float = 3e-4
    num_epochs: int = 5

    transaction_cost_bps: float = 10.0

    lambda_turnover: float = 0.01
    lambda_drawdown: float = 0.10
    lambda_concentration: float = 0.01

    allocation_hidden_dims: tuple[int, ...] = ()
    allocation_hidden_activation: str = "tanh"
    allocation_output_activation: str = "identity"
    allocation_use_layer_norm: bool = True
    simplex_activation: str = "softmax"

    eps: float = 1e-8

    def __post_init__(self) -> None:
        _validate_hidden_dims("allocation_hidden_dims", self.allocation_hidden_dims, allow_empty=True)
        _validate_activation("allocation_hidden_activation", self.allocation_hidden_activation)
        _validate_activation(
            "allocation_output_activation",
            self.allocation_output_activation,
            output=True,
        )
        if self.allocation_output_activation != "identity":
            raise ValueError(
                "allocation_output_activation must be 'identity' so logits remain unrestricted."
            )
        if self.simplex_activation not in {"softmax", "sparsemax"}:
            raise ValueError("simplex_activation must be 'softmax' or 'sparsemax'.")


def _validate_hidden_dims(
    name: str,
    hidden_dims: tuple[int, ...],
    allow_empty: bool = False,
) -> None:
    if not hidden_dims and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise ValueError(f"Every value in {name} must be positive.")


def _validate_activation(name: str, activation: str, output: bool = False) -> None:
    allowed = {"tanh", "gelu", "relu"}
    if output:
        allowed |= {"identity", "sigmoid"}
    if activation not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}.")
