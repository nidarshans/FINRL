"""Tests for the explicit Hawkes feature TODO."""

from __future__ import annotations

import polars as pl
import pytest

from finrl.features.hawkes import compute_hawkes_features


def test_compute_hawkes_features_requires_clarification() -> None:
    with pytest.raises(NotImplementedError, match="TODO: Hawkes feature construction"):
        compute_hawkes_features(pl.DataFrame({"date": [], "ticker": []}))
