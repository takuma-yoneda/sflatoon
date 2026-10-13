"""Run a full match between two players.

A player is any object with ``act(obs) -> list[dict] | None``; ``None`` means a
no-op tick. Players may expose ``crashes`` (an int); when it reaches
``rules.max_crashes`` the player's team forfeits. Plain policies are wrapped in
:class:`InProcessPlayer`, which counts an exception as a crash. The subprocess
sandbox with time limits is :class:`inkarena.arena.sandbox.SandboxedPlayer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RULES, Rules
from .engine import ACTION_DIM, Game
from .maps import Map


class InProcessPlayer:
    """Runs a policy object in this process. No time limits."""

    def __init__(self, policy):
        self.policy = policy
        self.crashes = 0
        self.errors: list[str] = []

    def act(self, obs):
        try:
            return self.policy.act(obs)
        except Exception as e:
            self.crashes += 1
            self.errors.append(f"tick {obs['tick']}: {type(e).__name__}: {e}")
            return None


@dataclass
class MatchResult:
    map_name: str
    seed: int
    score: tuple[int, int]
    winner: int | None  # 0 = team A, 1 = team B, None = draw
    kills: tuple[int, ...]
    deaths: tuple[int, ...]
    actions: np.ndarray  # float[T, n, 5] applied actions, absolute frame
    final_digest: str
    ticks: int  # ticks played (less than rules.ticks after a forfeit)
    forfeit: int | None = None  # team that forfeited, if any
    crashes: tuple[int, int] = (0, 0)
    recorder: object | None = None  # inkarena.replay.recorder.Recorder when record=True


def as_player(p):
    return p if hasattr(p, "crashes") else InProcessPlayer(p)


def run_match(
    game_map: Map,
    player_a,
    player_b,
    seed: int = 0,
    rules: Rules = DEFAULT_RULES,
    record: bool = False,
) -> MatchResult:
    game = Game(game_map, seed=seed, rules=rules)
    recorder = None
    if record:
        from ..replay.recorder import Recorder

        recorder = Recorder(game)
    players = (as_player(player_a), as_player(player_b))
    actions = np.zeros((rules.ticks, game.n, ACTION_DIM))
    forfeit, ended_early = None, False
    while not game.done:
        acts = [players[team].act(game.observe(team)) for team in (0, 1)]
        out = [p.crashes >= rules.max_crashes for p in players]
        if any(out):
            ended_early = True
            forfeit = None if all(out) else out.index(True)  # both at once: a draw
            break
        applied = game.read_actions(*acts)
        actions[game.state.tick] = applied
        if recorder is None:
            game.step_applied(applied)
        else:
            prev = recorder.pre_step()
            recorder.record(applied, game.step_applied(applied), prev)
    s = game.state
    if ended_early:
        winner = None if forfeit is None else 1 - forfeit
    else:
        winner = game.winner()
    return MatchResult(
        map_name=game_map.name,
        seed=seed,
        score=s.score,
        winner=winner,
        kills=tuple(int(v) for v in s.kills),
        deaths=tuple(int(v) for v in s.deaths),
        actions=actions[: s.tick],
        final_digest=s.digest(),
        ticks=s.tick,
        forfeit=forfeit,
        crashes=(players[0].crashes, players[1].crashes),
        recorder=recorder,
    )
