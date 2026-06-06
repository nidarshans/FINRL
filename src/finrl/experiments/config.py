"""Configuration for walk-forward experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finrl.backtest.walk_forward import WalkForwardConfig
from finrl.env.trading_env import EnvConfig
from finrl.features.preprocessing import PreprocessingConfig
from finrl.models.encoder import EncoderConfig
from finrl.ppo.policy import PPOConfig
from finrl.regimes.schema import HMMConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a strict walk-forward experiment."""

    walk_forward: WalkForwardConfig = WalkForwardConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    encoder: EncoderConfig = EncoderConfig()
    hmm: HMMConfig = HMMConfig()
    ppo: PPOConfig = PPOConfig()
    env: EnvConfig = EnvConfig()
    enable_ppo: bool = True
    seed: int = 0
    periods_per_year: int = 52
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")

