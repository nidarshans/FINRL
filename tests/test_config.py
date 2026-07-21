"""Tests for Phase 1 repository configuration."""

from dataclasses import FrozenInstanceError

import pytest

from finrl.config import ProjectConfig
from finrl.features.schema import FeatureConfig


def test_default_asset_dimension_includes_cash() -> None:
    config = ProjectConfig()

    assert config.num_stocks == 100
    assert config.include_cash is True
    assert config.num_assets == 101


def test_default_asset_dimension_without_cash() -> None:
    config = ProjectConfig(include_cash=False)

    assert config.num_assets == 100


def test_transaction_cost_bps_converts_to_decimal_rate() -> None:
    config = ProjectConfig(transaction_cost_bps=10.0)

    assert config.transaction_cost_rate == pytest.approx(0.001)


def test_project_config_is_immutable() -> None:
    config = ProjectConfig()

    with pytest.raises(FrozenInstanceError):
        config.num_stocks = 50  # type: ignore[misc]


def test_feature_config_default_trailing_windows() -> None:
    config = FeatureConfig()

    assert config.accumulation_window == 40
    assert config.macd_fast_span == 12
    assert config.macd_slow_span == 26
    assert config.macd_signal_span == 9
    assert config.mr_vol_window == 50
    assert config.historical_vol_window == 126
    assert config.ema_slope_window == 10
