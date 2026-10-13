"""Baseline v0 house bot (design doc section 5). Each agent independently:

1. Ink < 20: go to the nearest own-ink cell and stay submerged until refilled.
2. Enemy visible and in range: aim at it and fire.
3. Otherwise: walk to the nearest non-own cell, firing along the movement direction.
"""

from __future__ import annotations

import math

import numpy as np

from inkarena.bots import helpers as h

LOW_INK, FULL_INK = 20.0, 95.0
SHOT_RANGE = 6.0
RETARGET_EVERY = 8


class Policy:
    def __init__(self, seed: int = 0):
        self.refilling: list[bool] = []
        self.waypoints: list[list[tuple[float, float]]] = []
        self.target: list[tuple[int, int] | None] = []
        self.age: list[int] = []
        self.paintable = None

    def act(self, obs):
        walls, ink = obs["walls"], obs["ink"]
        if self.paintable is None:
            # Only bases are painted at the start; the enemy base is our base rotated.
            own_base = ink == 1
            self.paintable = ~walls & ~own_base & ~own_base[::-1, ::-1]
            n = len(obs["team"])
            self.refilling, self.target, self.age = [False] * n, [None] * n, [0] * n
            self.waypoints = [[] for _ in range(n)]
        out = []
        for i, me in enumerate(obs["team"]):
            if not me["alive"]:
                self.waypoints[i], self.target[i] = [], None
                out.append(_noop(me))
                continue
            out.append(self._agent(i, me, obs, walls, ink))
        return out

    def _agent(self, i, me, obs, walls, ink):
        pos = me["pos"]
        x, y = h.cell_of(pos)

        if me["ink"] < LOW_INK:
            self.refilling[i] = True
        if self.refilling[i]:
            if me["ink"] >= FULL_INK:
                self.refilling[i] = False
            elif ink[y, x] == 1:
                return {"move": (0.0, 0.0), "aim": me["facing"], "fire": False, "swim": True}
            else:
                move = self._go(i, pos, ink == 1, walls, ink)
                return {"move": move, "aim": me["facing"], "fire": False, "swim": False}

        enemy = self._enemy_in_range(pos, obs["enemies"], walls)
        move = self._go(i, pos, (ink != 1) & self.paintable, walls, ink)
        if enemy is not None:
            aim = math.atan2(enemy[1] - pos[1], enemy[0] - pos[0])
            return {"move": move, "aim": aim, "fire": True, "swim": False}
        if move == (0.0, 0.0):
            return {"move": move, "aim": me["facing"], "fire": False, "swim": False}
        aim = math.atan2(move[1], move[0])
        return {"move": move, "aim": aim, "fire": True, "swim": False}

    def _enemy_in_range(self, pos, enemies, walls):
        best, best_d = None, SHOT_RANGE
        for e in enemies:
            d = math.dist(pos, e["pos"])
            if d <= best_d and h.los(walls, pos, e["pos"]):
                best, best_d = e["pos"], d
        return best

    def _go(self, i, pos, goal_mask: np.ndarray, walls, ink) -> tuple[float, float]:
        """Step toward the nearest goal cell, replanning when the goal is met or stale."""
        t = self.target[i]
        self.age[i] += 1
        stale = t is None or not goal_mask[t[1], t[0]] or self.age[i] >= RETARGET_EVERY
        if stale or not self.waypoints[i]:
            t = h.nearest(goal_mask, pos, walls)
            self.target[i], self.age[i] = t, 0
            self.waypoints[i] = h.path(walls, pos, h.center_of(t)) if t else []
            if t and not self.waypoints[i]:
                self.waypoints[i] = [h.center_of(t)]  # already in the target cell
        wps = self.waypoints[i]
        while wps and math.dist(pos, wps[0]) < 0.3:
            wps.pop(0)
        if not wps:
            return (0.0, 0.0)
        dx, dy = wps[0][0] - pos[0], wps[0][1] - pos[1]
        n = math.hypot(dx, dy)
        return (dx / n, dy / n)


def _noop(me):
    return {"move": (0.0, 0.0), "aim": me["facing"], "fire": False, "swim": False}
