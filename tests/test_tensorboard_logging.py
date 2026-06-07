"""Tests for optional TensorBoard logging."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from finrl.logging import TensorBoardLogger, to_cpu_scalar
from finrl.ppo import PPOConfig, train_ppo_on_split
from tests.test_ppo_rollout import _artifacts


class FakeSummaryWriter:
    """Minimal SummaryWriter stand-in for logger tests."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        self.text.append((tag, text_string, global_step))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_to_cpu_scalar_converts_jax_scalar() -> None:
    assert to_cpu_scalar(jnp.array(1.25, dtype=jnp.float32)) == 1.25


def test_train_ppo_logs_update_metrics_with_fake_writer() -> None:
    writer = FakeSummaryWriter()
    logger = TensorBoardLogger(enabled=True, writer=writer)
    config = PPOConfig(
        n_assets=3,
        train_epochs=1,
        learning_rate=1e-4,
        minibatch_size=3,
        enable_tensorboard=True,
        log_frequency=1,
    )

    train_ppo_on_split(_artifacts(), config, jax.random.PRNGKey(2), logger)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "ppo/policy_loss" in tags
    assert "ppo/grad_norm" in tags
    assert "portfolio/alpha_vs_spy" in tags
    assert "portfolio/effective_positions" in tags
    assert "regime/state_0_probability" in tags
    assert "regime/state_0_asset_0_allocation" in tags
    assert all(isinstance(value, float) for _, value, _ in writer.scalars)
