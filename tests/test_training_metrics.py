"""Tests for production training diagnostics and optional logging."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finrl.env.trading_env import EnvConfig, EnvState
from finrl.logging import TensorBoardLogger
from finrl.models import (
    EncoderTrainMetrics,
    EncoderTrainingConfig,
    ProductionEncoderConfig,
    encoder_metrics_to_dict,
    finite_encoder_metrics,
    fit_encoder_on_train_split,
)
from finrl.models.windows import LookbackWindows
from finrl.ppo import (
    ProductionPPOConfig,
    finite_ppo_metrics,
    ppo_metrics_to_dict,
    train_flax_ppo_on_split,
)


class FakeSummaryWriter:
    """Minimal writer used to verify disabled logging is inert."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _initial_state(n_assets: int) -> EnvState:
    return EnvState(
        weights=jnp.ones((n_assets,), dtype=jnp.float32) / n_assets,
        portfolio_value=jnp.array(1.0, dtype=jnp.float32),
        peak_value=jnp.array(1.0, dtype=jnp.float32),
        drawdown=jnp.array(0.0, dtype=jnp.float32),
        previous_turnover=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _windows(n_windows: int = 5) -> LookbackWindows:
    return LookbackWindows(
        asset=np.arange(n_windows * 3 * 2 * 2, dtype=np.float32).reshape(
            n_windows,
            3,
            2,
            2,
        )
        / 100.0,
        macro=np.ones((n_windows, 3, 2), dtype=np.float32),
        spectral=np.ones((n_windows, 20), dtype=np.float32) / 10.0,
        decision_dates=tuple(range(n_windows)),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return", "volume"),
        macro_feature_columns=("rate", "inflation"),
        spectral_feature_columns=tuple(f"spectral_{index}" for index in range(20)),
    )


def _returns(n_windows: int = 5) -> np.ndarray:
    return np.asarray(
        [[0.01 * index, -0.005 * index] for index in range(n_windows)],
        dtype=np.float32,
    )


def test_ppo_metrics_are_finite_scalar_diagnostics() -> None:
    config = ProductionPPOConfig(
        n_assets=3,
        n_regimes=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        update_epochs=1,
        minibatch_size=3,
        learning_rate=1e-3,
        dirichlet_concentration=12.0,
    )
    phi = jnp.arange(3 * 32, dtype=jnp.float32).reshape(3, 32) / 100.0
    regimes = jnp.ones((3, 2), dtype=jnp.float32) / 2.0
    returns = jnp.array(
        [[0.01, 0.0, 0.0001], [0.0, 0.02, 0.0001], [-0.01, 0.01, 0.0001]],
        dtype=jnp.float32,
    )
    spy = jnp.array([0.005, 0.004, -0.002], dtype=jnp.float32)

    result = train_flax_ppo_on_split(
        phi,
        regimes,
        returns,
        spy,
        _initial_state(config.action_dim),
        EnvConfig(transaction_cost_rate=0.0),
        config,
        jax.random.PRNGKey(0),
    )

    metrics = ppo_metrics_to_dict(result.metrics)
    assert bool(finite_ppo_metrics(result.metrics))
    assert all(jnp.asarray(value).shape == () for value in metrics.values())


def test_encoder_metrics_are_finite_scalar_diagnostics() -> None:
    metrics = EncoderTrainMetrics(
        loss=jnp.array(1.0, dtype=jnp.float32),
        market_loss=jnp.array(0.1, dtype=jnp.float32),
        volatility_loss=jnp.array(0.2, dtype=jnp.float32),
        cross_sectional_loss=jnp.array(0.3, dtype=jnp.float32),
        l2_penalty=jnp.array(0.4, dtype=jnp.float32),
        grad_norm=jnp.array(0.5, dtype=jnp.float32),
    )

    metric_dict = encoder_metrics_to_dict(metrics)

    assert bool(finite_encoder_metrics(metrics))
    assert all(jnp.asarray(value).shape == () for value in metric_dict.values())


def test_encoder_training_logging_can_be_disabled() -> None:
    writer = FakeSummaryWriter()
    logger = TensorBoardLogger(enabled=False, writer=writer)
    encoder_config = ProductionEncoderConfig(
        lookback=3,
        n_assets=2,
        asset_feature_dim=2,
        macro_feature_dim=2,
        asset_hidden_dim=8,
        macro_hidden_dim=4,
        attention_heads=2,
        fusion_hidden_dim=12,
        output_dim=6,
    )

    result = fit_encoder_on_train_split(
        jax.random.PRNGKey(1),
        _windows(),
        _returns(),
        encoder_config,
        EncoderTrainingConfig(batch_size=2, epochs=1),
        train_window_count=5,
        logger=logger,
    )

    assert result.metrics_by_epoch
    assert writer.scalars == []
    assert logger.writer is None
