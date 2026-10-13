# Bot API v1

A bot is a Python module that defines a `Policy` class. One policy controls a whole team,
and the same instance is used for every tick of a match.

```python
class Policy:
    def __init__(self, seed: int): ...  # may use up to 1 s of CPU
    def act(self, obs: dict) -> list[dict]:  # one action per agent, every tick
        ...
```

The rules behind every number here are in [rules.md](rules.md).

## Coordinates

- Positions are `(x, y)` floats in cell units. Cell `(x, y)` covers `[x, x+1) × [y, y+1)`, and its center is `(x + 0.5, y + 0.5)`.
- Grids are numpy arrays indexed `[y, x]`, with shape `[H, W]` (20 x 32 on the bundled maps).
- Angles are radians, measured with `atan2(dy, dx)`: 0 points toward +x, and π/2 points toward +y, which is down the grid.
- Every team sees the map in its own frame, rotated so its base is at the bottom (high `y`). Actions are read back in the same frame, so one bot works on either side without changes.

## Observation

```python
obs = {
    "tick": int,  # ticks played so far
    "ticks_left": int,
    "walls": bool[H, W],  # never changes during a match
    "ink": int8[H, W],  # +1 ours, -1 theirs, 0 unpainted
    "team": [  # our agents, always in the same order
        {
            "pos": (x, y),
            "facing": float,
            "hp": int,
            "ink": float,
            "swimming": bool,
            "alive": bool,
            "respawn_in": int,
        },
    ]
    * team_size,
    "enemies": [  # visible enemies only, sorted by position
        {"pos": (x, y), "facing": float, "swimming": bool},
    ],
    "projectiles": [  # ours, plus visible enemy ones
        {"pos": (x, y), "vel": (vx, vy), "ours": bool},
    ],
    "score": (our_cells, their_cells),
}
```

- `enemies` has no stable identities and no last-seen positions. Any tracking or memory is up to the bot.
- `respawn_in` is the number of ticks until a dead agent respawns, and 0 while it's alive.
- `vel` is in cells per tick.

## Action

`act` returns one dict per agent, in the same order as `obs["team"]`:

```python
{"move": (dx, dy), "aim": float, "fire": bool, "swim": bool}
```

- `move` is a direction. Vectors longer than 1 are clipped to unit length, and shorter ones move proportionally slower. `(0, 0)` stands still.
- `aim` is the target facing. Facing turns toward it by at most 30° per tick, and both shots and the view cone follow facing.
- `fire` shoots along the current facing, when the agent has the ink and its cooldown is over.
- `swim` submerges on own ink. It is ignored elsewhere, and `fire` wins when both are set.
- A missing key, a wrong type or a non-finite number makes that agent's action a no-op for the tick. A no-op holds facing, doesn't move and doesn't fire, and keeps the agent submerged if it still can be.
- Returning a list of the wrong length makes every agent's action a no-op for the tick.
- Raising an exception, or going over the 50 ms per-tick limit, is a crash. A crash makes the tick a no-op for the team and restarts the bot with a fresh `Policy(seed)`, so its memory is lost. After 3 crashes the team forfeits the match.
- A bot over its 5 ms average budget gets a no-op tick, but that doesn't count as a crash.

## Helpers

`inkarena.bots.helpers` is the same for every bot. Every function works in the team frame
of `obs`.

| Function | Returns |
|---|---|
| `path(walls, a, b)` | Shortest walkable path from point `a` to point `b`, as a list of waypoints |
| `los(walls, a, b)` | True when no wall cell lies between two points |
| `nearest(mask, start, walls=None)` | Closest `True` cell of `mask` as `(x, y)`, by walking distance when `walls` is given |
| `frontier(ink, walls=None)` | Mask of cells that aren't ours but touch our ink |
| `regions(mask)` | 4-connected components of a mask, as `(labels, count)` |
| `distance_field(walls, sources)` | Walking distance in cells from the source cells |
| `cell_of(pos)`, `center_of(cell)` | Convert between points and cells |

## Example

This bot sends each agent to the nearest cell that isn't ours, firing where it walks.

```python
import math

from inkarena.bots import helpers as h


class Policy:
    def __init__(self, seed: int):
        pass

    def act(self, obs):
        walls, ink = obs["walls"], obs["ink"]
        out = []
        for me in obs["team"]:
            target = h.nearest((ink != 1) & ~walls, me["pos"], walls)
            route = h.path(walls, me["pos"], h.center_of(target)) if target else []
            if not route:
                out.append({"move": (0.0, 0.0), "aim": me["facing"], "fire": False, "swim": False})
                continue
            dx, dy = route[0][0] - me["pos"][0], route[0][1] - me["pos"][1]
            aim = math.atan2(dy, dx)
            out.append({"move": (dx, dy), "aim": aim, "fire": True, "swim": False})
        return out
```

The house bots in `inkarena/bots/` (`noop`, `random_bot`, `baseline`) are fuller examples.
Run a bot with `python tools/arena.py --a path/to/bot.py --b baseline`.
