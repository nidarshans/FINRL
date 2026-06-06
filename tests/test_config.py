"""Tests for Phase 1 repository configuration."""

from dataclasses import FrozenInstanceError

import pytest

from finrl.config import ProjectConfig


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
