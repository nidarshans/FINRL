"""Train-state helpers for production Flax PPO and encoder modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optax
from flax.training.train_state import TrainState


def create_train_state(
    apply_fn: Callable[..., Any],
    params: Any,
    learning_rate: float,
) -> TrainState:
    """Create a small, explicit Flax train state with Adam optimizer."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    optimizer = optax.adam(learning_rate)
    return TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)

