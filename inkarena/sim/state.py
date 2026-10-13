"""Simulator state. Agents 0-2 are team A, agents 3-5 are team B.

All positions and angles here are in the absolute (team A) frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .config import Rules
from .maps import Map

NEVER = -(10**9)  # "tick of last event" before anything happened


@dataclass(slots=True)
class Projectile:
    x: float
    y: float
    dx: float  # unit direction
    dy: float
    remaining: float  # range left, in cells
    owner: int  # agent index


@dataclass
class State:
    tick: int  # ticks simulated so far; the next step simulates this tick
    pos: np.ndarray  # float64[6, 2]
    facing: np.ndarray  # float64[6], radians in [0, 2*pi)
    hp: np.ndarray  # int64[6]
    ink: np.ndarray  # float64[6]
    submerged: np.ndarray  # bool[6]
    alive: np.ndarray  # bool[6]
    respawn_in: np.ndarray  # int64[6], ticks until respawn (dead agents)
    invisible: np.ndarray  # int64[6], ticks of respawn invisibility left
    next_fire: np.ndarray  # int64[6], first tick the agent may fire again
    last_shot: np.ndarray  # int64[6]
    last_damage: np.ndarray  # int64[6]
    grid: np.ndarray  # int8[H, W] ink: +1 team A, -1 team B, 0 empty
    projectiles: list[Projectile] = field(default_factory=list)
    visible: np.ndarray = field(default_factory=lambda: np.zeros((2, 3), dtype=bool))
    score: tuple[int, int] = (0, 0)
    kills: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    deaths: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))

    def digest(self) -> str:
        """Hash of everything that affects future ticks, for determinism checks."""
        h = hashlib.sha256()
        h.update(np.int64(self.tick).tobytes())
        for arr in (
            self.pos,
            self.facing,
            self.hp,
            self.ink,
            self.submerged,
            self.alive,
            self.respawn_in,
            self.invisible,
            self.next_fire,
            self.last_shot,
            self.last_damage,
            self.grid,
            self.visible,
            self.kills,
            self.deaths,
        ):
            h.update(np.ascontiguousarray(arr).tobytes())
        for p in self.projectiles:
            h.update(np.array([p.x, p.y, p.dx, p.dy, p.remaining, p.owner], dtype=float).tobytes())
        return h.hexdigest()


def initial_state(m: Map, rules: Rules) -> State:
    n = 2 * rules.team_size
    pos = np.concatenate([m.spawns[0], m.spawns[1]]).astype(float)
    cx, cy = m.center
    facing = np.arctan2(cy - pos[:, 1], cx - pos[:, 0]) % (2 * np.pi)
    return State(
        tick=0,
        pos=pos,
        facing=facing,
        hp=np.full(n, rules.max_hp, dtype=np.int64),
        ink=np.full(n, rules.max_ink, dtype=float),
        submerged=np.zeros(n, dtype=bool),
        alive=np.ones(n, dtype=bool),
        respawn_in=np.zeros(n, dtype=np.int64),
        invisible=np.zeros(n, dtype=np.int64),
        next_fire=np.zeros(n, dtype=np.int64),
        last_shot=np.full(n, NEVER, dtype=np.int64),
        last_damage=np.full(n, NEVER, dtype=np.int64),
        grid=m.base.copy(),
    )
