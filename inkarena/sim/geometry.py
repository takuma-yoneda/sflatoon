"""Grid geometry helpers: angles, raycasts, circle and segment queries."""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(a: float) -> float:
    """Wrap to [0, 2*pi)."""
    a = math.fmod(a, TWO_PI)
    return a + TWO_PI if a < 0 else a


def angle_diff(target: float, current: float) -> float:
    """Signed smallest rotation from ``current`` to ``target``, in (-pi, pi]."""
    d = math.fmod(target - current, TWO_PI)
    if d > math.pi:
        d -= TWO_PI
    elif d <= -math.pi:
        d += TWO_PI
    return d


def first_wall_t(walls: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float | None:
    """Parameter t in [0, 1] where segment p0->p1 first enters a wall cell, or None.

    Uses an exact grid traversal (Amanatides-Woo). Returns 0.0 if p0 is already
    inside a wall cell.
    """
    height, width = walls.shape
    cx, cy = math.floor(x0), math.floor(y0)
    if not (0 <= cx < width and 0 <= cy < height) or walls[cy, cx]:
        return 0.0
    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    inf = math.inf
    if dx != 0:
        t_max_x = ((cx + (step_x > 0)) - x0) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = t_delta_x = inf
    if dy != 0:
        t_max_y = ((cy + (step_y > 0)) - y0) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = t_delta_y = inf
    while True:
        if t_max_x < t_max_y:
            t = t_max_x
            if t > 1.0:
                return None
            cx += step_x
            t_max_x += t_delta_x
        else:
            t = t_max_y
            if t > 1.0:
                return None
            cy += step_y
            t_max_y += t_delta_y
        if not (0 <= cx < width and 0 <= cy < height) or walls[cy, cx]:
            return t


def line_of_sight(walls: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    """True when the segment between two points crosses no wall cell."""
    return first_wall_t(walls, x0, y0, x1, y1) is None


def circle_hits_wall(walls: np.ndarray, x: float, y: float, r: float) -> bool:
    """True when a circle overlaps any wall cell (cells outside the grid are wall)."""
    height, width = walls.shape
    for cy in range(math.floor(y - r), math.floor(y + r) + 1):
        for cx in range(math.floor(x - r), math.floor(x + r) + 1):
            if 0 <= cx < width and 0 <= cy < height and not walls[cy, cx]:
                continue
            nx = min(max(x, cx), cx + 1)
            ny = min(max(y, cy), cy + 1)
            if (x - nx) ** 2 + (y - ny) ** 2 < r * r:
                return True
    return False


def segment_circle_t(
    x0: float, y0: float, x1: float, y1: float, cx: float, cy: float, r: float
) -> float | None:
    """Smallest t in [0, 1] where segment p0->p1 is within ``r`` of (cx, cy), or None."""
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - cx, y0 - cy
    c = fx * fx + fy * fy - r * r
    if c <= 0:
        return 0.0
    a = dx * dx + dy * dy
    if a == 0:
        return None
    b = 2 * (fx * dx + fy * dy)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return t if 0.0 <= t <= 1.0 else None


def cells_near_segment(
    shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of cells whose centre is within ``r`` of segment p0->p1 (inclusive)."""
    height, width = shape
    xa = max(0, math.floor(min(x0, x1) - r))
    xb = min(width - 1, math.floor(max(x0, x1) + r))
    ya = max(0, math.floor(min(y0, y1) - r))
    yb = min(height - 1, math.floor(max(y0, y1) + r))
    if xa > xb or ya > yb:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    ys, xs = np.mgrid[ya : yb + 1, xa : xb + 1]
    px = xs + 0.5 - x0
    py = ys + 0.5 - y0
    dx, dy = x1 - x0, y1 - y0
    a = dx * dx + dy * dy
    t = np.clip((px * dx + py * dy) / a, 0.0, 1.0) if a > 0 else np.zeros_like(px, dtype=float)
    ex = px - t * dx
    ey = py - t * dy
    mask = ex * ex + ey * ey <= r * r + 1e-9
    return ys[mask], xs[mask]


def cells_in_disc(shape: tuple[int, int], x: float, y: float, r: float):
    """(ys, xs) of cells whose centre is within ``r`` of (x, y) (inclusive)."""
    return cells_near_segment(shape, x, y, x, y, r)
