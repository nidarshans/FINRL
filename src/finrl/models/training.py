"""Training hooks for the market encoder.

The encoder objective is intentionally unspecified in the architecture, so this
module does not invent a supervised or self-supervised loss.
"""

from __future__ import annotations


def train_encoder(*args: object, **kwargs: object) -> None:
    """Placeholder for future Colab-only encoder training."""

    del args, kwargs
    raise NotImplementedError("Encoder training objective is not specified yet.")

