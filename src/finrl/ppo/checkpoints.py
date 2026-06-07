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
        "actor": {
            "step": checkpoint.actor.step,
            "params": checkpoint.actor.params,
            "opt_state": checkpoint.actor.opt_state,
        },
        "critic": {
            "step": checkpoint.critic.step,
            "params": checkpoint.critic.params,
            "opt_state": checkpoint.critic.opt_state,
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
    )
    return type(state)(
        actor=state.actor.replace(
            step=payload["actor"]["step"],
            params=payload["actor"]["params"],
            opt_state=payload["actor"]["opt_state"],
        ),
        critic=state.critic.replace(
            step=payload["critic"]["step"],
            params=payload["critic"]["params"],
            opt_state=payload["critic"]["opt_state"],
        ),
        config=payload["config"],
    )


def save_policy_checkpoint(checkpoint: Any, path: str | Path) -> None:
    """Save a PPO policy checkpoint."""

    with Path(path).open("wb") as handle:
        pickle.dump(_pack_checkpoint(checkpoint), handle)


def load_policy_checkpoint(path: str | Path) -> Any:
    """Load a PPO policy checkpoint."""

    with Path(path).open("rb") as handle:
        return _unpack_checkpoint(pickle.load(handle))
