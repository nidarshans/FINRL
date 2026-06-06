"""Checkpoint helpers for PPO actor-critic state."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_policy_checkpoint(checkpoint: Any, path: str | Path) -> None:
    """Save a PPO policy checkpoint."""

    with Path(path).open("wb") as handle:
        pickle.dump(checkpoint, handle)


def load_policy_checkpoint(path: str | Path) -> Any:
    """Load a PPO policy checkpoint."""

    with Path(path).open("rb") as handle:
        return pickle.load(handle)

