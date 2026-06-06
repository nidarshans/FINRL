"""Checkpoint helpers for encoder parameter pytrees."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_encoder_checkpoint(params: Any, path: str | Path) -> None:
    """Save encoder parameters to a local pickle checkpoint."""

    with Path(path).open("wb") as handle:
        pickle.dump(params, handle)


def load_encoder_checkpoint(path: str | Path) -> Any:
    """Load encoder parameters from a local pickle checkpoint."""

    with Path(path).open("rb") as handle:
        return pickle.load(handle)

