"""Ink Arena simulator core.

One call to :meth:`Game.step` simulates one tick, in the order fixed by the
design doc (section 4). Each numbered step is its own method:

1. read actions          5. advance projectiles, resolve hits   9. refill ink, regen HP
2. turn facing           6. damage, then deaths                10. respawns
3. move                  7. paint (all at once)                11. score, visibility
4. fire                  8. check submerge validity

Actions arrive in each team's own frame and are converted to the absolute frame
in step 1. The resolved, absolute-frame actions ("applied actions") are what a
replay stores; :meth:`Game.step_applied` replays them exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import frame
from .config import DEFAULT_RULES, Rules
from .geometry import (
    angle_diff,
    cells_in_disc,
    cells_near_segment,
    circle_hits_wall,
    first_wall_t,
    line_of_sight,
    segment_circle_t,
    wrap_angle,
)
from .maps import Map
from .state import Projectile, State, initial_state

# Applied action columns, absolute frame.
MOVE_X, MOVE_Y, AIM, FIRE, SWIM = range(5)
ACTION_DIM = 5

_BISECT_STEPS = 12
_WALL_EPS = 1e-6


@dataclass
class TickEvents:
    hits: list[tuple[int, int]] = field(default_factory=list)  # (shooter, victim)
    # (killer, victim, x, y, dist): victim position and shooter-victim distance at the hit
    kills: list[tuple[int, int, float, float, float]] = field(default_factory=list)
    respawns: list[int] = field(default_factory=list)
    shots: list[int] = field(default_factory=list)


def team_sign(team: int) -> int:
    return 1 if team == 0 else -1


class Game:
    def __init__(self, game_map: Map, seed: int = 0, rules: Rules = DEFAULT_RULES):
        self.map = game_map
        self.rules = rules
        self.seed = seed
        self.m = rules.team_size  # agents per team
        self.n = 2 * rules.team_size  # agents 0..m-1 are team A, m..n-1 team B
        if len(game_map.spawns[0]) < self.m:
            raise ValueError(
                f"map {game_map.name} has {len(game_map.spawns[0])} spawns per team, "
                f"rules need {self.m}"
            )
        # All randomness (spawn jitter, shot spread) comes from this one seeded RNG.
        self.rng = np.random.default_rng(seed)
        self.walls = game_map.walls
        self.base = game_map.base
        self.scorable = game_map.scorable
        self.spawns = self._jitter_spawns()
        self.state: State = initial_state(game_map, rules)
        s, (cx, cy) = self.state, game_map.center
        for k in range(self.n):
            x, y = s.pos[k] = self.spawns[self.team_of(k), k % self.m]
            s.facing[k] = wrap_angle(math.atan2(cy - y, cx - x))
        self._team_walls = [frame.grid_to_team(self.walls, t).copy() for t in (0, 1)]
        for w in self._team_walls:
            w.flags.writeable = False
        self._score_and_visibility()

    def _jitter_spawns(self) -> np.ndarray:
        """This match's spawn points: each map spawn moved by a random offset, mirrored for B.

        An offset that would put either team's agent in a wall is redrawn, up to 10 times,
        then dropped. Agents respawn at the same jittered point all match.
        """
        spawns = self.map.spawns.copy()
        j, rad = self.rules.spawn_jitter, self.rules.radius
        if j <= 0:
            return spawns
        for i in range(spawns.shape[1]):
            for _ in range(10):
                off = self.rng.uniform(-j, j, size=2)
                a, b = spawns[0, i] + off, spawns[1, i] - off
                if not (
                    circle_hits_wall(self.walls, *a, rad) or circle_hits_wall(self.walls, *b, rad)
                ):
                    spawns[0, i], spawns[1, i] = a, b
                    break
        return spawns

    def team_of(self, agent: int) -> int:
        return 0 if agent < self.m else 1

    @property
    def done(self) -> bool:
        return self.state.tick >= self.rules.ticks

    # ------------------------------------------------------------------ public API

    def step(self, actions_a, actions_b) -> TickEvents:
        """Simulate one tick from team-frame action lists (one dict per agent)."""
        return self.step_applied(self.read_actions(actions_a, actions_b))

    def step_applied(self, applied: np.ndarray) -> TickEvents:
        """Simulate one tick from absolute-frame applied actions, shape [n_agents, 5]."""
        if self.done:
            raise RuntimeError("match is over")
        self.last_applied = applied
        ev = TickEvents()
        self._turn(applied)
        self._move(applied)
        new_projectiles = self._fire(applied, ev)
        self.state.projectiles.extend(new_projectiles)
        paint = np.zeros((2,) + self.walls.shape, dtype=bool)
        hits = self._advance_projectiles(paint)
        self._apply_damage(hits, paint, ev)
        self._paint(paint)
        self._check_submerge()
        self._refill_and_regen()
        self._respawn(ev)
        self.state.tick += 1
        self._score_and_visibility()
        return ev

    def winner(self) -> int | None:
        """0 for team A, 1 for team B, None for a draw."""
        a, b = self.state.score
        return 0 if a > b else 1 if b > a else None

    # ---------------------------------------------------------- 1. read actions

    def read_actions(self, actions_a, actions_b) -> np.ndarray:
        """Validate and convert team-frame actions to applied absolute-frame actions.

        Invalid output (wrong list length, missing keys, NaN) is a no-op for that
        agent: facing holds, no move, no fire, stays submerged if still valid.
        Dead agents' actions are ignored.
        """
        s = self.state
        applied = np.zeros((self.n, ACTION_DIM))
        for team, acts in ((0, actions_a), (1, actions_b)):
            valid_list = isinstance(acts, list | tuple) and len(acts) == self.m
            for i in range(self.m):
                k = self.m * team + i
                parsed = _parse_action(acts[i]) if valid_list else None
                if parsed is None or not s.alive[k]:
                    applied[k] = (0.0, 0.0, s.facing[k], 0.0, float(s.submerged[k]))
                    continue
                (dx, dy), aim, fire, swim = parsed
                n = math.hypot(dx, dy)
                if n > 1.0:
                    dx, dy = dx / n, dy / n
                dx, dy = frame.vec_from_team(dx, dy, team)
                applied[k] = (dx, dy, frame.angle_from_team(aim, team), float(fire), float(swim))
        return applied

    # ------------------------------------------------------------ 2. turn facing

    def _turn(self, applied: np.ndarray) -> None:
        s, cap = self.state, self.rules.turn_rate
        for k in range(self.n):
            if not s.alive[k]:
                continue
            d = angle_diff(applied[k, AIM], s.facing[k])
            d = max(-cap, min(cap, d))
            s.facing[k] = wrap_angle(s.facing[k] + d)

    # ------------------------------------------------------------------ 3. move

    def _move(self, applied: np.ndarray) -> None:
        s, r = self.state, self.rules
        for k in range(self.n):
            if not s.alive[k]:
                continue
            sign = team_sign(self.team_of(k))
            x, y = s.pos[k]
            under = self.state.grid[int(y), int(x)]
            # Fire wins over swim; swim is ignored off own ink.
            sub = bool(applied[k, SWIM]) and not applied[k, FIRE] and under == sign
            s.submerged[k] = sub
            dx, dy = applied[k, MOVE_X], applied[k, MOVE_Y]
            if dx == 0.0 and dy == 0.0:
                continue
            if sub:
                speed = r.swim_speed
            else:
                speed = r.walk_speed * (r.enemy_ink_factor if under == -sign else 1.0)
            s.pos[k] = self._slide(x, y, dx * speed, dy * speed, sub, sign)

    def _free(self, x: float, y: float, sub: bool, sign: int) -> bool:
        if circle_hits_wall(self.walls, x, y, self.rules.radius):
            return False
        return not sub or self.state.grid[int(y), int(x)] == sign

    def _slide(self, x: float, y: float, dx: float, dy: float, sub: bool, sign: int):
        """Move with wall sliding: resolve the x then the y component."""
        if self._free(x + dx, y + dy, sub, sign):
            return x + dx, y + dy
        x = self._axis(x, y, dx, 0.0, sub, sign)[0]
        return self._axis(x, y, 0.0, dy, sub, sign)

    def _axis(self, x, y, dx, dy, sub, sign):
        if dx == 0.0 and dy == 0.0:
            return x, y
        if self._free(x + dx, y + dy, sub, sign):
            return x + dx, y + dy
        lo, hi = 0.0, 1.0  # largest free fraction of the step
        for _ in range(_BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if self._free(x + dx * mid, y + dy * mid, sub, sign):
                lo = mid
            else:
                hi = mid
        return x + dx * lo, y + dy * lo

    # ------------------------------------------------------------------ 4. fire

    def _fire(self, applied: np.ndarray, ev: TickEvents) -> list[Projectile]:
        s, r = self.state, self.rules
        out = []
        for k in range(self.n):
            if not (s.alive[k] and applied[k, FIRE]):
                continue
            if s.submerged[k] or s.tick < s.next_fire[k] or s.ink[k] < r.shot_cost:
                continue
            s.ink[k] -= r.shot_cost
            s.next_fire[k] = s.tick + r.fire_cooldown
            s.last_shot[k] = s.tick
            f = s.facing[k]
            if r.shot_spread > 0:
                f += self.rng.uniform(-r.shot_spread, r.shot_spread)
            out.append(
                Projectile(s.pos[k, 0], s.pos[k, 1], math.cos(f), math.sin(f), r.shot_range, k)
            )
            ev.shots.append(k)
        return out

    # ------------------------------------------------- 5. projectiles and hits

    def _advance_projectiles(self, paint: np.ndarray) -> list[tuple[int, int, float, float]]:
        """Move every projectile one tick. Returns hits as (shooter, victim, x, y)."""
        s, r = self.state, self.rules
        shape = self.walls.shape
        hits = []
        keep = []
        for p in s.projectiles:
            team = self.team_of(p.owner)
            length = min(r.shot_speed, p.remaining)
            x0, y0 = p.x, p.y
            x1, y1 = x0 + p.dx * length, y0 + p.dy * length
            t_end, landed = 1.0, p.remaining - length <= 0
            tw = first_wall_t(self.walls, x0, y0, x1, y1)
            if tw is not None:
                t_end, landed = max(0.0, tw - _WALL_EPS / max(length, _WALL_EPS)), True
            victim = None
            for k in range(self.m * (1 - team), self.m * (1 - team) + self.m):
                if not s.alive[k]:
                    continue
                t = segment_circle_t(x0, y0, x1, y1, s.pos[k, 0], s.pos[k, 1], r.radius)
                if t is not None and t <= t_end:
                    t_end, victim, landed = t, k, True
            ex, ey = x0 + (x1 - x0) * t_end, y0 + (y1 - y0) * t_end
            ys, xs = cells_near_segment(shape, x0, y0, ex, ey, r.strip_halfwidth)
            paint[team, ys, xs] = True
            if victim is not None:
                hits.append((p.owner, victim, ex, ey))
            if landed:
                self._paint_disc(paint[team], ex, ey, r.splat_radius)
            else:
                p.x, p.y, p.remaining = x1, y1, p.remaining - length
                keep.append(p)
        s.projectiles = keep
        return hits

    # ------------------------------------------------------- 6. damage, deaths

    def _apply_damage(self, hits, paint: np.ndarray, ev: TickEvents) -> None:
        s, r = self.state, self.rules
        for shooter, victim, _, _ in hits:
            ev.hits.append((shooter, victim))
            if s.hp[victim] <= 0:
                continue  # already killed by an earlier hit this tick
            s.hp[victim] -= r.shot_damage
            s.last_damage[victim] = s.tick
            if s.hp[victim] <= 0:
                vx, vy = s.pos[victim]
                sx, sy = s.pos[shooter]
                ev.kills.append((shooter, victim, vx, vy, math.hypot(vx - sx, vy - sy)))
                s.kills[shooter] += 1
        for _, victim, x, y, _ in ev.kills:
            s.hp[victim] = 0
            s.alive[victim] = False
            s.submerged[victim] = False
            s.respawn_in[victim] = r.respawn_delay
            s.invisible[victim] = 0
            s.deaths[victim] += 1
            self._paint_disc(paint[1 - self.team_of(victim)], x, y, r.burst_radius)

    def _paint_disc(self, mask: np.ndarray, x: float, y: float, radius: float) -> None:
        """Mark cells within ``radius`` that the centre can see (splats don't cross walls)."""
        ys, xs = cells_in_disc(self.walls.shape, x, y, radius)
        for cy, cx in zip(ys, xs, strict=True):
            if not self.walls[cy, cx] and line_of_sight(self.walls, x, y, cx + 0.5, cy + 0.5):
                mask[cy, cx] = True

    # ------------------------------------------------------------------ 7. paint

    def _paint(self, paint: np.ndarray) -> None:
        """Apply all paint at once. Cells painted by both teams keep their value."""
        a, b = paint[0], paint[1]
        g = self.state.grid
        g[a & ~b & self.scorable] = 1
        g[b & ~a & self.scorable] = -1

    # ------------------------------------------------------------- 8. submerge

    def _check_submerge(self) -> None:
        s = self.state
        for k in range(self.n):
            if s.submerged[k]:
                x, y = s.pos[k]
                if s.grid[int(y), int(x)] != team_sign(self.team_of(k)):
                    s.submerged[k] = False

    # ----------------------------------------------------- 9. refill and regen

    def _refill_and_regen(self) -> None:
        s, r = self.state, self.rules
        t = s.tick
        for k in range(self.n):
            if not s.alive[k]:
                continue
            if s.submerged[k]:
                s.ink[k] = min(r.max_ink, s.ink[k] + r.swim_refill)
            elif t - s.last_shot[k] >= r.idle_refill_delay:
                s.ink[k] = min(r.max_ink, s.ink[k] + r.idle_refill)
            if s.hp[k] < r.max_hp:
                since = t - s.last_damage[k]
                if since >= r.regen_delay and (since - r.regen_delay) % r.regen_interval == 0:
                    s.hp[k] += 1

    # --------------------------------------------------------------- 10. respawn

    def _respawn(self, ev: TickEvents) -> None:
        s, r = self.state, self.rules
        killed_now = {victim for _, victim, *_ in ev.kills}
        cx, cy = self.map.center
        for k in range(self.n):
            if s.alive[k]:
                if s.invisible[k] > 0:
                    s.invisible[k] -= 1
                continue
            if k in killed_now:
                continue
            s.respawn_in[k] -= 1
            if s.respawn_in[k] > 0:
                continue
            x, y = self.spawns[self.team_of(k), k % self.m]
            s.pos[k] = (x, y)
            s.facing[k] = wrap_angle(math.atan2(cy - y, cx - x))
            s.hp[k] = r.max_hp
            s.ink[k] = r.max_ink
            s.alive[k] = True
            s.submerged[k] = False
            s.respawn_in[k] = 0
            s.invisible[k] = r.respawn_invisible
            s.next_fire[k] = s.tick + 1
            s.last_damage[k] = s.tick
            ev.respawns.append(k)

    # ------------------------------------------------- 11. score, visibility

    def _score_and_visibility(self) -> None:
        s = self.state
        g = s.grid[self.scorable]
        s.score = (int((g == 1).sum()), int((g == -1).sum()))
        for team in (0, 1):
            for j in range(self.m):
                s.visible[team, j] = self.sees(team, self.m * (1 - team) + j)

    def sees(self, team: int, target: int) -> bool:
        """Whether ``team`` can see agent ``target`` (an enemy) right now."""
        s = self.state
        if not s.alive[target] or s.invisible[target] > 0:
            return False
        tx, ty = s.pos[target]
        limit = self.rules.submerged_view_range if s.submerged[target] else self.rules.view_range
        return any(
            s.alive[k] and self._in_view(k, tx, ty, limit)
            for k in range(self.m * team, self.m * team + self.m)
        )

    def _in_view(self, k: int, tx: float, ty: float, limit: float) -> bool:
        s = self.state
        x, y = s.pos[k]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist > limit:
            return False
        if (
            dist > 0
            and abs(angle_diff(math.atan2(dy, dx), s.facing[k])) > self.rules.view_half_angle
        ):
            return False
        return line_of_sight(self.walls, x, y, tx, ty)

    def point_visible(self, team: int, x: float, y: float) -> bool:
        """Visibility rule for non-agent objects (enemy projectiles)."""
        s = self.state
        return any(
            s.alive[k] and self._in_view(k, x, y, self.rules.view_range)
            for k in range(self.m * team, self.m * team + self.m)
        )

    # ------------------------------------------------------------ observation

    def observe(self, team: int) -> dict:
        """Observation for ``team`` in its own frame (design doc section 3)."""
        s = self.state
        w, h = self.map.width, self.map.height
        team_info = []
        for k in range(self.m * team, self.m * team + self.m):
            team_info.append(
                {
                    "pos": frame.pos_to_team(s.pos[k, 0], s.pos[k, 1], team, w, h),
                    "facing": frame.angle_to_team(s.facing[k], team),
                    "hp": int(s.hp[k]),
                    "ink": float(s.ink[k]),
                    "swimming": bool(s.submerged[k]),
                    "alive": bool(s.alive[k]),
                    "respawn_in": int(s.respawn_in[k]),
                }
            )
        enemies = []
        for j in range(self.m):
            if s.visible[team, j]:
                k = self.m * (1 - team) + j
                enemies.append(
                    {
                        "pos": frame.pos_to_team(s.pos[k, 0], s.pos[k, 1], team, w, h),
                        "facing": frame.angle_to_team(s.facing[k], team),
                        "swimming": bool(s.submerged[k]),
                    }
                )
        enemies.sort(key=lambda e: e["pos"])  # no stable identities
        speed = self.rules.shot_speed
        projectiles = []
        for p in s.projectiles:
            ours = self.team_of(p.owner) == team
            if ours or self.point_visible(team, p.x, p.y):
                projectiles.append(
                    {
                        "pos": frame.pos_to_team(p.x, p.y, team, w, h),
                        "vel": frame.vec_to_team(p.dx * speed, p.dy * speed, team),
                        "ours": ours,
                    }
                )
        ours, theirs = s.score if team == 0 else s.score[::-1]
        return {
            "tick": s.tick,
            "ticks_left": self.rules.ticks - s.tick,
            "walls": self._team_walls[team],
            "ink": frame.ink_to_team(s.grid, team),
            "team": team_info,
            "enemies": enemies,
            "projectiles": projectiles,
            "score": (ours, theirs),
        }


def _parse_action(a) -> tuple[tuple[float, float], float, bool, bool] | None:
    """Parse one agent's action dict, or None if it is invalid."""
    try:
        dx, dy = (float(v) for v in a["move"])
        aim = float(a["aim"])
        fire = bool(a["fire"])
        swim = bool(a["swim"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (dx, dy, aim)):
        return None
    return (dx, dy), aim, fire, swim
