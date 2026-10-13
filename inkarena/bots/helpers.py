"""Bot helper library. Identical for every bot; works in the team frame of ``obs``.

All functions take plain arrays from the observation (``obs["walls"]``,
``obs["ink"]``) and positions as ``(x, y)`` floats or cells as ``(x, y)`` ints.
Grids are indexed ``[y, x]``. Paths are 4-connected, which an agent of radius
0.4 can always follow through 1-wide gaps.

- ``path(walls, a, b)``: shortest walkable path from point a to point b
- ``los(walls, a, b)``: True when no wall cell lies between two points
- ``nearest(mask, start, walls=None)``: closest True cell (walking distance if walls given)
- ``frontier(ink, walls=None)``: non-own cells next to own ink
- ``regions(mask)``: 4-connected components of a mask
- ``distance_field(walls, sources)``: walking distance (in cells) from source cells
"""

from __future__ import annotations

import math

import numpy as np

from inkarena.sim.geometry import line_of_sight

UNREACHABLE = np.iinfo(np.int32).max


def cell_of(p) -> tuple[int, int]:
    return int(math.floor(p[0])), int(math.floor(p[1]))


def center_of(c) -> tuple[float, float]:
    return c[0] + 0.5, c[1] + 0.5


def los(walls: np.ndarray, a, b) -> bool:
    return line_of_sight(walls, float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def _neighbours(mask: np.ndarray) -> np.ndarray:
    """Cells 4-adjacent to any True cell of ``mask``."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def distance_field(walls: np.ndarray, sources: np.ndarray, limit: int | None = None) -> np.ndarray:
    """Walking distance from the nearest source cell, ``UNREACHABLE`` where none.

    ``sources`` is a bool mask (or a list of (x, y) cells).
    """
    if not isinstance(sources, np.ndarray) or sources.dtype != bool:
        mask = np.zeros(walls.shape, dtype=bool)
        for x, y in sources:
            mask[y, x] = True
        sources = mask
    free = ~walls
    dist = np.full(walls.shape, UNREACHABLE, dtype=np.int32)
    front = sources & free
    seen = front.copy()
    d = 0
    while front.any():
        dist[front] = d
        if limit is not None and d >= limit:
            break
        front = _neighbours(front) & free & ~seen
        seen |= front
        d += 1
    return dist


def path(walls: np.ndarray, a, b) -> list[tuple[float, float]]:
    """Cell-centre waypoints from point ``a`` to point ``b`` (excluding a's cell).

    Empty when b is unreachable or a and b share a cell.
    """
    ca, cb = cell_of(a), cell_of(b)
    dist = distance_field(walls, [cb])
    if dist[ca[1], ca[0]] == UNREACHABLE:
        return []
    return _descend(dist, ca)


def _descend(dist: np.ndarray, start) -> list[tuple[float, float]]:
    h, w = dist.shape
    x, y = start
    out = []
    while dist[y, x] > 0:
        best = None
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            inside = 0 <= nx < w and 0 <= ny < h
            if inside and dist[ny, nx] < (dist[y, x] if best is None else dist[best[1], best[0]]):
                best = (nx, ny)
        x, y = best
        out.append(center_of((x, y)))
    return out


def nearest(mask: np.ndarray, start, walls: np.ndarray | None = None):
    """Closest True cell of ``mask`` to point ``start`` as (x, y), or None.

    With ``walls``, "closest" is walking distance; otherwise straight-line.
    """
    if not mask.any():
        return None
    if walls is None:
        ys, xs = np.nonzero(mask)
        d = (xs + 0.5 - start[0]) ** 2 + (ys + 0.5 - start[1]) ** 2
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])
    dist = distance_field(walls, [cell_of(start)])
    dist = np.where(mask, dist, UNREACHABLE)
    i = int(np.argmin(dist))
    if dist.flat[i] == UNREACHABLE:
        return None
    y, x = divmod(i, mask.shape[1])
    return x, y


def frontier(ink: np.ndarray, walls: np.ndarray | None = None) -> np.ndarray:
    """Cells that aren't own ink but touch own ink (bool mask)."""
    out = _neighbours(ink == 1) & (ink != 1)
    if walls is not None:
        out &= ~walls
    return out


def regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected components: (labels int32[H, W] with 0 = background, count)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    remaining = mask.copy()
    while remaining.any():
        count += 1
        i = int(np.argmax(remaining))
        comp = np.zeros_like(mask)
        comp.flat[i] = True
        while True:
            grown = (comp | _neighbours(comp)) & mask
            if np.array_equal(grown, comp):
                break
            comp = grown
        labels[comp] = count
        remaining &= ~comp
    return labels, count
