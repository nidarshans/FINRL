"""Compatibility training hook for the production market encoder."""

from __future__ import annotations

from finrl.models.encoder_training import fit_encoder_on_train_split


def train_encoder(*args: object, **kwargs: object) -> object:
    """Train the encoder with the Phase C self-supervised objective."""

    return fit_encoder_on_train_split(*args, **kwargs)
