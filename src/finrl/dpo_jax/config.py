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

    eps: float = 1e-8
