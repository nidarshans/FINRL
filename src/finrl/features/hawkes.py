"""Hawkes feature entry points."""

from __future__ import annotations

import polars as pl


def compute_hawkes_features(data: pl.DataFrame) -> pl.DataFrame:
    """Raise until the Hawkes model specification is clarified.

    The architecture names required Hawkes outputs but does not specify event
    construction, estimation windows, kernel family, or fitting method. Returning
    placeholder values would create misleading research signals, so this remains
    an explicit TODO.
    """

    del data
    raise NotImplementedError(
        "TODO: Hawkes feature construction requires event definition, kernel, "
        "estimation window, and filtering-only update specification."
    )
