"""Minimal replay: seed, map id and the applied actions reproduce a match exactly."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.config import DEFAULT_RULES, Rules
from ..sim.engine import Game
from ..sim.maps import Map, load_map


@dataclass
class ActionReplay:
    seed: int
    map_name: str
    actions: np.ndarray  # float[T, n, 5]


def save_replay(path: str | Path, replay: ActionReplay) -> None:
    np.savez_compressed(path, seed=replay.seed, map_name=replay.map_name, actions=replay.actions)


def load_replay(path: str | Path) -> ActionReplay:
    with np.load(path) as f:
        return ActionReplay(int(f["seed"]), str(f["map_name"]), f["actions"].copy())


def replay_states(
    replay: ActionReplay, game_map: Map | None = None, rules: Rules = DEFAULT_RULES
) -> Iterator[Game]:
    """Re-simulate a replay, yielding the game after the initial state and every tick."""
    game = Game(game_map or load_map(replay.map_name), seed=replay.seed, rules=rules)
    yield game
    for applied in replay.actions:
        game.step_applied(applied)
        yield game
