"""ASCII map loading and validation.

Format: one row per line, top row first, so the character at line ``y``,
column ``x`` is cell ``(x, y)``. Symbols:

- ``#`` wall, ``.`` floor
- ``a`` / ``b`` base cells of team A / team B
- ``1``, ``2``, ... team A spawn points (also team A base cells), numbered from 1
  with no gaps. Team B's spawns are the 180-degree images of these and are
  written as ``b`` in the file.

The map border always counts as wall, whether or not it is drawn. A map is valid
when its 180-degree rotation equals itself with teams swapped, and every
non-wall cell is reachable from every other through 4-connected moves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAP_DIR = Path(__file__).resolve().parent.parent / "maps"

WALL, FLOOR, BASE_A, BASE_B = "#", ".", "a", "b"
SPAWN_CHARS = "123456789"
STANDARD_SIZE = (32, 20)


class MapError(ValueError):
    pass


@dataclass(frozen=True)
class Map:
    """A parsed map. Arrays are indexed ``[y, x]``."""

    name: str
    walls: np.ndarray  # bool[H, W]
    base: np.ndarray  # int8[H, W]: +1 team A base, -1 team B base, 0 otherwise
    spawns: np.ndarray  # float[2, k, 2]: team, index, (x, y) at cell centres

    @property
    def width(self) -> int:
        return self.walls.shape[1]

    @property
    def height(self) -> int:
        return self.walls.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    @property
    def scorable(self) -> np.ndarray:
        """Cells that count toward score: neither wall nor base."""
        return ~self.walls & (self.base == 0)


def parse_map(text: str, name: str = "<string>", validate: bool = True) -> Map:
    rows = [line.rstrip("\r") for line in text.splitlines() if line.strip()]
    if not rows:
        raise MapError(f"{name}: empty map")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise MapError(f"{name}: rows have different lengths")
    height = len(rows)
    grid = np.array([list(r) for r in rows])
    allowed = set(WALL + FLOOR + BASE_A + BASE_B + SPAWN_CHARS)
    bad = set(np.unique(grid)) - allowed
    if bad:
        raise MapError(f"{name}: unknown symbols {sorted(bad)}")

    walls = grid == WALL
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True

    base = np.zeros((height, width), dtype=np.int8)
    team_a = (grid == BASE_A) | np.isin(grid, list(SPAWN_CHARS))
    base[team_a] = 1
    base[grid == BASE_B] = -1
    base[walls] = 0

    present = [ch for ch in SPAWN_CHARS if (grid == ch).any()]
    if not present or present != list(SPAWN_CHARS[: len(present)]):
        raise MapError(f"{name}: spawns must be numbered 1..k with no gaps, found {present}")
    spawns = np.zeros((2, len(present), 2))
    for i, ch in enumerate(present):
        ys, xs = np.nonzero(grid == ch)
        if len(xs) != 1:
            raise MapError(f"{name}: spawn '{ch}' must appear exactly once, found {len(xs)}")
        x, y = int(xs[0]), int(ys[0])
        if walls[y, x]:
            raise MapError(f"{name}: spawn '{ch}' is on the border")
        spawns[0, i] = (x + 0.5, y + 0.5)
        spawns[1, i] = (width - x - 0.5, height - y - 0.5)

    m = Map(name=name, walls=walls, base=base, spawns=spawns)
    if validate:
        validate_map(m)
    return m


def validate_map(m: Map, size: tuple[int, int] | None = None) -> None:
    """Raise MapError unless the map is symmetric and fully connected."""
    if size is not None and (m.width, m.height) != size:
        raise MapError(f"{m.name}: size {m.width}x{m.height}, expected {size[0]}x{size[1]}")
    if not np.array_equal(m.walls, m.walls[::-1, ::-1]):
        raise MapError(f"{m.name}: walls are not 180-degree symmetric")
    if not np.array_equal(m.base, -m.base[::-1, ::-1]):
        raise MapError(f"{m.name}: bases are not 180-degree symmetric with teams swapped")
    if not (m.base == 1).any():
        raise MapError(f"{m.name}: no base cells")
    for team in range(2):
        sign = 1 if team == 0 else -1
        for x, y in m.spawns[team]:
            if m.base[int(y), int(x)] != sign:
                raise MapError(f"{m.name}: team {'AB'[team]} spawn ({x}, {y}) is not on its base")
    unreachable = _unreachable(m.walls)
    if unreachable:
        x, y = unreachable[0]
        raise MapError(f"{m.name}: {len(unreachable)} floor cells unreachable, e.g. ({x}, {y})")


def _unreachable(walls: np.ndarray) -> list[tuple[int, int]]:
    height, width = walls.shape
    free = np.argwhere(~walls)
    if len(free) == 0:
        return []
    seen = np.zeros_like(walls)
    y0, x0 = free[0]
    seen[y0, x0] = True
    queue = deque([(y0, x0)])
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < height and 0 <= nx < width and not walls[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return [(int(x), int(y)) for y, x in np.argwhere(~walls & ~seen)]


def load_map(name_or_path: str | Path, validate: bool = True) -> Map:
    """Load a bundled map by name (e.g. ``"duel"``) or from a file path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = MAP_DIR / f"{name_or_path}.txt"
    m = parse_map(path.read_text(), name=path.stem, validate=False)
    if validate:
        validate_map(m, size=STANDARD_SIZE)
    return m


def list_maps() -> list[str]:
    return sorted(p.stem for p in MAP_DIR.glob("*.txt"))
