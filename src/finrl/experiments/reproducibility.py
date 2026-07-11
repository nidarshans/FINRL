"""Reproducible experiment metadata and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from finrl.backtest.results import WalkForwardResult
from finrl.experiments.artifacts import RawExperimentData
from finrl.experiments.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ExperimentRunMetadata:
    """Stable provenance attached to one experiment run."""

    run_id: str
    created_at_utc: str
    git_revision: str | None
    seed: int
    tickers: tuple[str, ...]
    decision_start: date
    decision_end: date
    input_fingerprint: str
    config: dict[str, Any]


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported metadata value: {type(value)!r}")


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _table_digest(table: pl.DataFrame) -> bytes:
    return table.write_json().encode("utf-8")


def input_fingerprint(data: RawExperimentData) -> str:
    """Return a stable SHA-256 fingerprint of all raw experiment tables."""

    digest = hashlib.sha256()
    for table in (
        data.features.asset_features,
        data.features.macro_features,
        data.features.spectral_features,
        data.returns,
        data.spy_returns,
    ):
        digest.update(_table_digest(table))
    return digest.hexdigest()


def build_run_metadata(data: RawExperimentData, config: ExperimentConfig) -> ExperimentRunMetadata:
    """Build provenance metadata without mutating experiment inputs."""

    dates = tuple(sorted(data.features.decision_dates))
    if not dates:
        raise ValueError("Cannot build metadata for empty decision dates.")
    config_dict = asdict(config)
    encoded = json.dumps(config_dict, default=_json_default, sort_keys=True)
    fingerprint = input_fingerprint(data)
    run_id = hashlib.sha256(
        (fingerprint + encoded + str(config.seed)).encode("utf-8")
    ).hexdigest()[:16]
    return ExperimentRunMetadata(
        run_id=run_id,
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        git_revision=_git_revision(),
        seed=config.seed,
        tickers=tuple(data.features.tickers),
        decision_start=dates[0],
        decision_end=dates[-1],
        input_fingerprint=fingerprint,
        config=config_dict,
    )


def save_walk_forward_artifacts(
    result: WalkForwardResult,
    metadata: ExperimentRunMetadata,
    directory: str | Path,
) -> Path:
    """Persist provenance, curves, allocations, and split return streams."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(
        json.dumps(asdict(metadata), default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result.portfolio_curve.write_parquet(output / "portfolio_curve.parquet")
    result.spy_curve.write_parquet(output / "spy_curve.parquet")
    result.allocations.write_parquet(output / "allocations.parquet")
    pl.concat([split.portfolio_returns for split in result.split_results]).write_parquet(
        output / "portfolio_returns.parquet"
    )
    pl.concat([split.spy_returns for split in result.split_results]).write_parquet(
        output / "spy_returns.parquet"
    )
    return output
