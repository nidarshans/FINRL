"""Reusable TensorBoard logging helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from finrl.types import Array


def to_cpu_scalar(value: object) -> float:
    """Convert a scalar JAX/NumPy/Python value to a CPU float for logging."""

    scalar = np.asarray(jax.device_get(value), dtype=np.float64)
    if scalar.shape != ():
        raise ValueError("TensorBoard scalar values must be rank-0.")
    return float(scalar)


def _flatten_hparams(value: object, prefix: str = "") -> dict[str, object]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        items: dict[str, object] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten_hparams(child, child_prefix))
        return items
    if isinstance(value, Path):
        return {prefix: str(value)}
    if isinstance(value, tuple):
        return {prefix: ",".join(str(item) for item in value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {prefix: value}
    return {prefix: str(value)}


def _load_summary_writer() -> type[Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter

            return SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging requires torch.utils.tensorboard or tensorboardX."
            ) from exc


class TensorBoardLogger:
    """Small wrapper around TensorBoard SummaryWriter with JAX scalar handling."""

    def __init__(
        self,
        log_dir: str | Path = "runs",
        experiment_name: str | None = None,
        enabled: bool = True,
        writer: Any | None = None,
    ) -> None:
        self.enabled = enabled
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = experiment_name or f"ppo-{timestamp}"
        self.log_dir = Path(log_dir) / name
        self._writer = writer
        if self.enabled and self._writer is None:
            writer_cls = _load_summary_writer()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = writer_cls(str(self.log_dir))

    @property
    def writer(self) -> Any | None:
        """Return the underlying SummaryWriter, if logging is enabled."""

        return self._writer if self.enabled else None

    def log_scalars(
        self,
        metrics: dict[str, object],
        step: int,
        prefix: str | None = None,
    ) -> None:
        """Log scalar metrics after moving values to CPU."""

        if not self.enabled or self._writer is None:
            return
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self._writer.add_scalar(tag, to_cpu_scalar(value), step)

    def log_hyperparameters(self, hparams: object) -> None:
        """Log experiment hyperparameters as text and hparam metadata."""

        if not self.enabled or self._writer is None:
            return
        flattened = _flatten_hparams(hparams)
        text = "\n".join(f"{key}: {value}" for key, value in sorted(flattened.items()))
        self._writer.add_text("hparams", text, 0)
        add_hparams = getattr(self._writer, "add_hparams", None)
        if add_hparams is not None:
            serializable = {
                key: value
                for key, value in flattened.items()
                if isinstance(value, (str, int, float, bool))
            }
            add_hparams(serializable, {})

    def log_regime_metrics(
        self,
        regime_probs: Array,
        actions: Array,
        step: int,
        prefix: str = "regime",
    ) -> None:
        """Log average HMM probabilities and allocation by regime."""

        if not self.enabled:
            return
        probs = np.asarray(jax.device_get(regime_probs), dtype=np.float64)
        allocations = np.asarray(jax.device_get(actions), dtype=np.float64)
        if probs.ndim != 2 or allocations.ndim != 2:
            raise ValueError("regime probabilities and actions must be rank-2 arrays.")
        probability_metrics = {
            f"state_{index}_probability": probability
            for index, probability in enumerate(np.mean(probs, axis=0))
        }
        self.log_scalars(probability_metrics, step, prefix)
        weights = probs[:, :, None]
        denominators = np.sum(probs, axis=0)
        by_regime = np.divide(
            np.sum(weights * allocations[:, None, :], axis=0),
            denominators[:, None],
            out=np.zeros((probs.shape[1], allocations.shape[1]), dtype=np.float64),
            where=denominators[:, None] > 0.0,
        )
        allocation_metrics = {
            f"state_{regime_index}_asset_{asset_index}_allocation": value
            for regime_index, row in enumerate(by_regime)
            for asset_index, value in enumerate(row)
        }
        self.log_scalars(allocation_metrics, step, prefix)

    def close(self) -> None:
        """Flush and close the writer."""

        if self.enabled and self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
