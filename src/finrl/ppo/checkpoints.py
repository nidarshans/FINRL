"""Checkpoint helpers for PPO actor-critic state."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def _pack_checkpoint(checkpoint: Any) -> Any:
    from finrl.ppo.flax_trainer import ProductionPPOTrainState

    if not isinstance(checkpoint, ProductionPPOTrainState):
        return checkpoint
    return {
        "kind": "production_ppo_train_state",
        "config": checkpoint.config,
        "encoder_config": checkpoint.encoder_config,
        "accumulation_indices": checkpoint.accumulation_indices,
        "liquidity_indices": checkpoint.liquidity_indices,
        "policy": {
            "step": checkpoint.policy.step,
            "params": checkpoint.policy.params,
            "opt_state": checkpoint.policy.opt_state,
        },
    }


def _unpack_checkpoint(payload: Any) -> Any:
    if not (
        isinstance(payload, dict)
        and payload.get("kind") == "production_ppo_train_state"
    ):
        return payload

    from finrl.ppo.flax_trainer import initialize_ppo_train_state
    import jax

    state = initialize_ppo_train_state(
        rng=jax.random.PRNGKey(0),
        config=payload["config"],
        encoder_config=payload["encoder_config"],
        accumulation_indices=tuple(payload["accumulation_indices"]),
        liquidity_indices=tuple(payload["liquidity_indices"]),
    )
    return type(state)(
        policy=state.policy.replace(
            step=payload["policy"]["step"],
            params=payload["policy"]["params"],
            opt_state=payload["policy"]["opt_state"],
        ),
        config=payload["config"],
        encoder_config=payload["encoder_config"],
        accumulation_indices=tuple(payload["accumulation_indices"]),
        liquidity_indices=tuple(payload["liquidity_indices"]),
    )


def save_policy_checkpoint(checkpoint: Any, path: str | Path) -> None:
    """Save a PPO policy checkpoint."""

    with Path(path).open("wb") as handle:
        pickle.dump(_pack_checkpoint(checkpoint), handle)


def load_policy_checkpoint(path: str | Path) -> Any:
    """Load a PPO policy checkpoint."""

    with Path(path).open("rb") as handle:
        return _unpack_checkpoint(pickle.load(handle))
