"""Replay query library (design doc section 7).

    from inkarena.replay import load
    r = load("replays/round_07/m012.replay.npz")   # events from m012.events.jsonl next to it
    r.events("splat", agent="A1", t0=300, t1=600)
    r.state_at(450)          # dict: agents, projectiles, ink, score, visible
    r.trace("B2", 0, 200)    # [n, 8] rows of AGENT_COLS for ticks t0..t1
    r.coverage()             # int[T+1, 2] painted cells per team after each tick
    r.render(450, "t450.png", vision=True)
    r.heatmap("position", "pos.png")

Everything is in the absolute frame: team A's base is at the bottom (high y).
State index ``i`` is the state after ``i`` ticks (0 = start), and an event with
``tick == t`` happened during tick ``t``, so it is visible in ``state_at(t + 1)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sim.maps import load_map
from .recorder import AGENT_COLS

_CHECKPOINT = 100


def agent_names(n: int) -> tuple[str, ...]:
    m = n // 2
    return tuple(f"{'AB'[k // m]}{k % m}" for k in range(n))


class Replay:
    def __init__(self, data: dict, events: list[dict] | None = None, path: Path | None = None):
        self.path = path
        self.seed = int(data["seed"])
        self.map_name = str(data["map_name"])
        self.ticks = int(data["ticks"])
        self.agents = data["agents"]
        self.respawn_in = data["respawn_in"]
        self.proj = data["proj"]
        self.proj_owner = data["proj_owner"]
        self.ink0 = data["ink0"]
        self.ink_diff = data["ink_diff"]
        self.actions = data["actions"]
        self.visible = data["visible"]
        self.score = data["score"]
        self._events = events or []
        self.map = load_map(self.map_name, validate=False)
        self.n_agents = self.agents.shape[1]
        self.team_size = self.n_agents // 2
        self.names = agent_names(self.n_agents)
        self._diff_start = np.searchsorted(self.ink_diff[:, 0], np.arange(self.ticks + 2))
        self._ink_cache: dict[int, np.ndarray] = {0: self.ink0}

    def agent_index(self, agent) -> int:
        if isinstance(agent, int | np.integer):
            return int(agent)
        return self.names.index(agent)

    @property
    def n_states(self) -> int:
        return self.ticks + 1

    def _check(self, tick: int) -> int:
        if not 0 <= tick <= self.ticks:
            raise IndexError(f"tick {tick} outside 0..{self.ticks}")
        return int(tick)

    # ---------------------------------------------------------------- queries

    def events(self, type: str | None = None, agent=None, t0: int = 0, t1: int | None = None):
        """Events filtered by type, by agent (any field naming it) and tick range [t0, t1]."""
        name = None if agent is None else self.names[self.agent_index(agent)]
        out = []
        for e in self._events:
            if type is not None and e["type"] != type:
                continue
            if e["tick"] < t0 or (t1 is not None and e["tick"] > t1):
                continue
            if name is not None and name not in (
                e.get(k) for k in ("agent", "killer", "victim", "enemy")
            ):
                continue
            out.append(e)
        return out

    def ink_at(self, tick: int) -> np.ndarray:
        tick = self._check(tick)
        base = (tick // _CHECKPOINT) * _CHECKPOINT
        if base not in self._ink_cache:
            prev = max(k for k in self._ink_cache if k <= base)
            self._ink_cache[base] = self._apply(self._ink_cache[prev], prev, base)
        return self._apply(self._ink_cache[base], base, tick)

    def _apply(self, ink: np.ndarray, start: int, end: int) -> np.ndarray:
        ink = ink.copy()
        d = self.ink_diff[self._diff_start[start + 1] : self._diff_start[end + 1]]
        ink[d[:, 1], d[:, 2]] = d[:, 3]
        return ink

    def state_at(self, tick: int) -> dict:
        """Full state after ``tick`` ticks."""
        tick = self._check(tick)
        agents = []
        for k in range(self.n_agents):
            row = self.agents[tick, k]
            a = {c: float(v) for c, v in zip(AGENT_COLS, row, strict=True)}
            for c in ("hp",):
                a[c] = int(a[c])
            for c in ("submerged", "alive", "invisible"):
                a[c] = bool(a[c])
            a["name"] = self.names[k]
            a["respawn_in"] = int(self.respawn_in[tick, k])
            agents.append(a)
        owners = self.proj_owner[tick]
        projectiles = [
            {
                "pos": (float(p[0]), float(p[1])),
                "dir": (float(p[2]), float(p[3])),
                "owner": self.names[o],
            }
            for p, o in zip(self.proj[tick], owners, strict=True)
            if o >= 0
        ]
        return {
            "tick": tick,
            "agents": agents,
            "projectiles": projectiles,
            "ink": self.ink_at(tick),
            "score": tuple(int(v) for v in self.score[tick]),
            "visible": self.visible[tick].copy(),
        }

    def trace(self, agent, t0: int = 0, t1: int | None = None) -> np.ndarray:
        """Rows of AGENT_COLS for one agent over states t0..t1 (inclusive)."""
        t1 = self.ticks if t1 is None else t1
        return self.agents[self._check(t0) : self._check(t1) + 1, self.agent_index(agent)]

    def coverage(self) -> np.ndarray:
        return self.score.copy()

    # -------------------------------------------------------------- rendering

    def render(self, tick: int, out_png, vision: bool = True, scale: int = 32, **kwargs) -> Path:
        from .render import render_state

        img = render_state(self, self._check(tick), vision=vision, scale=scale, **kwargs)
        img.save(out_png)
        return Path(out_png)

    def heatmap(self, kind: str, out_png, team: str | None = None, scale: int = 32) -> Path:
        """``kind``: "position" (time spent per cell), "deaths", or "paint" (ink changes)."""
        from .render import render_heatmap

        render_heatmap(self, kind, team, scale).save(out_png)
        return Path(out_png)

    def video(self, out_mp4, **kwargs) -> Path:
        from .render import render_video

        return render_video(self, out_mp4, **kwargs)


def load(path) -> Replay:
    path = Path(path)
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    events_path = path.with_name(path.name.replace(".replay.npz", ".events.jsonl"))
    events = []
    if events_path != path and events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    return Replay(data, events, path)
