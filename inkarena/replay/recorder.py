"""Record a match into replay arrays and neutral events (design doc sections 4 and 7).

Replay ``.npz`` contents (absolute frame; with m agents per team, agents 0..m-1 are
team A and m..2m-1 team B, n = 2m):

- ``seed``, ``map_name``, ``ticks``
- ``agents`` float32[T+1, n, 8]: x, y, facing, hp, ink, submerged, alive, invisible
  (row 0 is the initial state, row t+1 the state after tick t)
- ``respawn_in`` int16[T+1, 6]
- ``proj`` float32[T+1, P, 4]: x, y, dx, dy, padded with NaN; ``proj_owner`` int8[T+1, P], -1 pad
- ``ink0`` int8[H, W] initial ink; ``ink_diff`` int32[N, 4]: (state index, y, x, new value)
- ``actions`` float64[T, n, 5]: applied actions (move x, move y, aim, fire, swim)
- ``visible`` bool[T+1, 2, 3]: team t sees the other team's agent j
- ``score`` int32[T+1, 2]

Events (one JSON object per line): ``splat``, ``respawn``, ``ink_empty``,
``enemy_spotted``, ``coverage_checkpoint``. Agents are named ``A0``-``A2`` and
``B0``-``B2``; positions are absolute-frame ``[x, y]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sim.engine import Game, TickEvents

AGENT_COLS = ("x", "y", "facing", "hp", "ink", "submerged", "alive", "invisible")
COVERAGE_EVERY = 100
INK_EMPTY = 4.0


def agent_name(k: int, m: int) -> str:
    """A0, A1, ... for team A; B0, B1, ... for team B."""
    return f"{'AB'[k // m]}{k % m}"


class Recorder:
    def __init__(self, game: Game):
        self.game = game
        self.agents: list[np.ndarray] = []
        self.respawn_in: list[np.ndarray] = []
        self.proj: list[np.ndarray] = []
        self.visible: list[np.ndarray] = []
        self.score: list[tuple[int, int]] = []
        self.ink_diff: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.events: list[dict] = []
        self.ink0 = game.state.grid.copy()
        self._last_ink = self.ink0.copy()
        self._snapshot()

    def _snapshot(self) -> None:
        s = self.game.state
        a = np.empty((self.game.n, len(AGENT_COLS)), dtype=np.float32)
        a[:, 0:2] = s.pos
        a[:, 2] = s.facing
        a[:, 3] = s.hp
        a[:, 4] = s.ink
        a[:, 5] = s.submerged
        a[:, 6] = s.alive
        a[:, 7] = s.invisible > 0
        self.agents.append(a)
        self.respawn_in.append(s.respawn_in.astype(np.int16))
        self.proj.append(
            np.array(
                [(p.x, p.y, p.dx, p.dy, p.owner) for p in s.projectiles], dtype=np.float64
            ).reshape(-1, 5)
        )
        self.visible.append(s.visible.copy())
        self.score.append(s.score)

    def record(self, applied: np.ndarray, ev: TickEvents, prev) -> None:
        """Call after ``game.step_applied``; ``prev`` is ``(ink, visible)`` from before the step."""
        g, s = self.game, self.game.state
        t = s.tick - 1  # the tick just simulated
        self.actions.append(applied.copy())
        self._snapshot()
        changed = np.argwhere(s.grid != self._last_ink)
        if len(changed):
            vals = s.grid[changed[:, 0], changed[:, 1]]
            self.ink_diff.append(
                np.column_stack([np.full(len(changed), t + 1), changed, vals]).astype(np.int32)
            )
            self._last_ink = s.grid.copy()

        prev_ink, prev_visible = prev
        for killer, victim, x, y, dist in ev.kills:
            kx, ky = s.pos[killer]
            saw = g._in_view(victim, kx, ky, g.rules.view_range)
            self.events.append(
                {
                    "type": "splat",
                    "tick": t,
                    "killer": agent_name(killer, g.m),
                    "victim": agent_name(victim, g.m),
                    "pos": [round(x, 3), round(y, 3)],
                    "dist": round(dist, 3),
                    "victim_saw_killer": bool(saw),
                }
            )
        for k in ev.respawns:
            self.events.append(
                {
                    "type": "respawn",
                    "tick": t,
                    "agent": agent_name(k, g.m),
                    "pos": [float(s.pos[k, 0]), float(s.pos[k, 1])],
                }
            )
        for k in range(g.n):
            if s.alive[k] and s.ink[k] < INK_EMPTY <= prev_ink[k]:
                self.events.append({"type": "ink_empty", "tick": t, "agent": agent_name(k, g.m)})
        for team in (0, 1):
            for j in range(g.m):
                if s.visible[team, j] and not prev_visible[team, j]:
                    k = g.m * (1 - team) + j
                    self.events.append(
                        {
                            "type": "enemy_spotted",
                            "tick": t,
                            "team": "AB"[team],
                            "enemy": agent_name(k, g.m),
                            "pos": [round(float(s.pos[k, 0]), 3), round(float(s.pos[k, 1]), 3)],
                        }
                    )
        if s.tick % COVERAGE_EVERY == 0 or g.done:
            self.events.append(
                {"type": "coverage_checkpoint", "tick": t, "A": s.score[0], "B": s.score[1]}
            )

    def pre_step(self):
        s = self.game.state
        return s.ink.copy(), s.visible.copy()

    # ------------------------------------------------------------------ output

    def arrays(self) -> dict[str, np.ndarray]:
        n_proj = max((len(p) for p in self.proj), default=0)
        n_states = len(self.agents)
        proj = np.full((n_states, n_proj, 4), np.nan, dtype=np.float32)
        owner = np.full((n_states, n_proj), -1, dtype=np.int8)
        for i, p in enumerate(self.proj):
            proj[i, : len(p)] = p[:, :4]
            owner[i, : len(p)] = p[:, 4]
        diff = np.concatenate(self.ink_diff) if self.ink_diff else np.zeros((0, 4), np.int32)
        actions = np.stack(self.actions) if self.actions else np.zeros((0, self.game.n, 5))
        return {
            "seed": np.int64(self.game.seed),
            "map_name": np.str_(self.game.map.name),
            "ticks": np.int64(len(self.actions)),
            "agents": np.stack(self.agents),
            "respawn_in": np.stack(self.respawn_in),
            "proj": proj,
            "proj_owner": owner,
            "ink0": self.ink0,
            "ink_diff": diff,
            "actions": actions,
            "visible": np.stack(self.visible),
            "score": np.array(self.score, dtype=np.int32),
        }

    def save(self, replay_path: str | Path, events_path: str | Path | None = None) -> None:
        np.savez_compressed(replay_path, **self.arrays())
        if events_path is not None:
            with open(events_path, "w") as f:
                for e in self.events:
                    f.write(json.dumps(e, separators=(",", ":")) + "\n")

    def coverage(self, every: int = COVERAGE_EVERY) -> list[list[int]]:
        """[tick, A, B] every ``every`` ticks (state index = tick count)."""
        return [[i, int(a), int(b)] for i, (a, b) in enumerate(self.score) if i % every == 0]
