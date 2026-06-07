"""Tests for production encoder pretraining."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from finrl.models import (
    EncoderTrainingConfig,
    ProductionEncoderConfig,
    encoder_loss,
    fit_encoder_on_train_split,
    init_encoder_pretraining_state,
    make_encoder_batches,
    train_encoder_epoch,
)
from finrl.models.checkpoints import load_encoder_checkpoint, save_encoder_checkpoint
from finrl.models.windows import LookbackWindows


def _windows(n_windows: int = 6) -> LookbackWindows:
    asset = np.arange(n_windows * 3 * 2 * 2, dtype=np.float32).reshape(n_windows, 3, 2, 2)
    macro = np.arange(n_windows * 3 * 2, dtype=np.float32).reshape(n_windows, 3, 2) / 10.0
    spectral = np.arange(n_windows * 20, dtype=np.float32).reshape(n_windows, 20) / 100.0
    return LookbackWindows(
        asset=asset / 100.0,
        macro=macro,
        spectral=spectral,
        decision_dates=tuple(range(n_windows)),
        tickers=("AAA", "BBB"),
        asset_feature_columns=("return", "volume"),
        macro_feature_columns=("rate", "inflation"),
        spectral_feature_columns=tuple(f"spectral_{index}" for index in range(20)),
    )


def _returns(n_windows: int = 6) -> np.ndarray:
    return np.asarray(
        [[0.01 * index, -0.005 * index] for index in range(n_windows)],
        dtype=np.float32,
    )


def _encoder_config() -> ProductionEncoderConfig:
    return ProductionEncoderConfig(
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


def test_make_encoder_batches_builds_expected_targets() -> None:
    training_config = EncoderTrainingConfig(batch_size=2)
    batches = make_encoder_batches(
        _windows(),
        _returns(),
        training_config,
        train_window_count=5,
    )

    assert len(batches) == 2
    assert batches[0].asset_window.shape == (2, 3, 2, 2)
    assert batches[0].decision_dates == (0, 1)
    assert batches[0].label_dates == (1, 2)
    assert batches[1].decision_dates == (2, 3)
    assert batches[1].label_dates == (3, 4)

    expected_market = np.asarray([0.0025, 0.005], dtype=np.float32)
    expected_volatility = np.asarray([0.0075, 0.015], dtype=np.float32)
    expected_cross_sectional = np.asarray(
        [[1.0, -1.0], [1.0, -1.0]],
        dtype=np.float32,
    )
    assert_allclose(batches[0].market_return_target, expected_market)
    assert_allclose(batches[0].volatility_target, expected_volatility)
    assert_allclose(
        batches[0].cross_sectional_return_target,
        expected_cross_sectional,
        rtol=1e-5,
        atol=1e-6,
    )


def test_encoder_loss_is_finite_on_deterministic_data() -> None:
    encoder_config = _encoder_config()
    training_config = EncoderTrainingConfig(batch_size=3)
    batch = make_encoder_batches(
        _windows(),
        _returns(),
        training_config,
        train_window_count=5,
    )[0]
    state = init_encoder_pretraining_state(
        jax.random.PRNGKey(0),
        encoder_config,
        training_config,
    )

    loss, metrics = encoder_loss(
        state.params,
        batch,
        encoder_config,
        training_config.loss_weights,
    )

    assert jnp.isfinite(loss)
    assert set(metrics) == {
        "loss",
        "market_loss",
        "volatility_loss",
        "cross_sectional_loss",
        "l2_penalty",
    }
    assert all(jnp.isfinite(value) for value in metrics.values())


def test_one_encoder_training_epoch_changes_parameters() -> None:
    encoder_config = _encoder_config()
    training_config = EncoderTrainingConfig(batch_size=2, learning_rate=1e-2)
    batches = make_encoder_batches(
        _windows(),
        _returns(),
        training_config,
        train_window_count=5,
    )
    state = init_encoder_pretraining_state(
        jax.random.PRNGKey(1),
        encoder_config,
        training_config,
    )

    new_state, metrics = train_encoder_epoch(
        state,
        batches,
        encoder_config,
        training_config,
    )

    old_leaves = jax.tree.leaves(state.params)
    new_leaves = jax.tree.leaves(new_state.params)
    changed = [not jnp.allclose(old, new) for old, new in zip(old_leaves, new_leaves)]
    assert any(bool(value) for value in changed)
    assert np.isfinite(metrics["loss"])


def test_fit_encoder_on_train_split_returns_epoch_metrics() -> None:
    result = fit_encoder_on_train_split(
        jax.random.PRNGKey(2),
        _windows(),
        _returns(),
        _encoder_config(),
        EncoderTrainingConfig(batch_size=2, epochs=2),
        train_window_count=5,
    )

    assert result.train_state.step > 0
    assert len(result.metrics_by_epoch) == 2
    assert all(np.isfinite(metrics["loss"]) for metrics in result.metrics_by_epoch)


def test_encoder_checkpoint_round_trip_preserves_params(tmp_path: Path) -> None:
    state = init_encoder_pretraining_state(
        jax.random.PRNGKey(3),
        _encoder_config(),
        EncoderTrainingConfig(batch_size=2),
    )
    path = tmp_path / "encoder.pkl"

    save_encoder_checkpoint(state.params, path)
    loaded = load_encoder_checkpoint(path)

    original_leaves = jax.tree.leaves(state.params)
    loaded_leaves = jax.tree.leaves(loaded)
    for original, restored in zip(original_leaves, loaded_leaves):
        assert_allclose(original, restored)
