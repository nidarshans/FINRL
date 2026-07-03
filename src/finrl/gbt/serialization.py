"""Portable LightGBM model serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from finrl.gbt.config import GBTConfig
from finrl.gbt.model import GBTModel
from finrl.types import PathLikeStr


def save_gbt_model(model: GBTModel, directory: PathLikeStr) -> None:
    """Save the native booster and explicit policy metadata."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    booster = getattr(model.estimator, "booster_", model.estimator)
    booster.save_model(str(path / "model.txt"))
    (path / "metadata.json").write_text(json.dumps({
        "config": asdict(model.config),
        "feature_names": model.feature_names,
        "seed": model.seed,
    }, indent=2), encoding="utf-8")


def load_gbt_model(directory: PathLikeStr) -> GBTModel:
    """Load a native LightGBM booster and its policy metadata."""

    try:
        from lightgbm import Booster
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LightGBM is required for the GBT policy.") from exc
    path = Path(directory)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    config_data = metadata["config"]
    config_data["target_horizons"] = tuple(config_data["target_horizons"])
    config_data["target_weights"] = tuple(config_data["target_weights"])
    return GBTModel(
        Booster(model_file=str(path / "model.txt")),
        GBTConfig(**config_data),
        tuple(metadata["feature_names"]),
        int(metadata["seed"]),
    )
