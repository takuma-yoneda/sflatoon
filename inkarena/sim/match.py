"""Run a full match between two in-process policies.

The subprocess sandbox (time limits, crash handling) arrives in phase 2; here a
policy is any object with ``act(obs) -> list[dict]``, and an exception from it
counts as invalid output (a no-op for all three agents that tick).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RULES, Rules
from .engine import ACTION_DIM, Game
from .maps import Map


@dataclass
class MatchResult:
    map_name: str
    seed: int
    score: tuple[int, int]
    winner: int | None  # 0 = team A, 1 = team B, None = draw
    kills: tuple[int, ...]
    deaths: tuple[int, ...]
    actions: np.ndarray  # float[T, 6, 5] applied actions, absolute frame
    final_digest: str


def run_match(
    game_map: Map, policy_a, policy_b, seed: int = 0, rules: Rules = DEFAULT_RULES
) -> MatchResult:
    game = Game(game_map, seed=seed, rules=rules)
    actions = np.zeros((rules.ticks, 6, ACTION_DIM))
    while not game.done:
        acts = []
        for team, policy in ((0, policy_a), (1, policy_b)):
            try:
                acts.append(policy.act(game.observe(team)))
            except Exception:
                acts.append(None)
        applied = game.read_actions(*acts)
        actions[game.state.tick] = applied
        game.step_applied(applied)
    s = game.state
    return MatchResult(
        map_name=game_map.name,
        seed=seed,
        score=s.score,
        winner=game.winner(),
        kills=tuple(int(v) for v in s.kills),
        deaths=tuple(int(v) for v in s.deaths),
        actions=actions,
        final_digest=s.digest(),
    )
