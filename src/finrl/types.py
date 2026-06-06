"""Shared type aliases used by repository modules."""

from __future__ import annotations

from os import PathLike
from typing import Any, TypeAlias

try:
    import jax

    Array: TypeAlias = jax.Array
except ImportError:  # pragma: no cover - package dependency should provide JAX.
    Array: TypeAlias = Any

DateLike: TypeAlias = str
Ticker: TypeAlias = str
PathLikeStr: TypeAlias = str | PathLike[str]
Scalar: TypeAlias = float
