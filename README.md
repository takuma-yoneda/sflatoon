# Ink Arena (sflatoon)

A top-down 2D Splatoon-like 3v3 territory shooter where LLMs compete by writing bots.
Bots never run per-tick inside a model; models author a `policy.py` that the simulator runs.

Status: phase 2 (bots and arena). See the design doc and build plan in the project for the roadmap.

## Layout

```
inkarena/
  sim/        numpy simulator: rules (config.py), maps, tick engine, team frame transform
  bots/       house bots (noop, random, baseline) and the bot helper library (helpers.py)
  arena/      bot loader, subprocess sandbox, match runner
  replay/     replays (phase 1: seed + map + applied actions)
  maps/       bundled ASCII maps (64x40, 180-degree symmetric)
tools/
  validate_maps.py   check map symmetry and reachability
  bench.py           matches/min across worker processes
  arena.py           local matches between two bots, with side swapping
  demo.py            play one match and print the board as ASCII
tests/               one test module per rule area
```

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python tools/validate_maps.py
python tools/bench.py --matches 32 --bot random_bot
python tools/arena.py --a baseline --b random -n 20 --out replays/local/
```

A bot is a directory with `policy.py` defining `Policy(seed)` with `act(obs) -> list[dict]`
(design doc section 3). Bots run in a subprocess: 1 s CPU for setup, 50 ms CPU hard limit per
tick (over it kills and restarts the bot and counts as a crash), a 5 ms average budget (over it
the tick becomes a no-op), 1 GiB address space, and sockets disabled. Crashes are no-op ticks;
3 crashes forfeit the match.

```python
from inkarena.sim import load_map, run_match
from inkarena.bots import random_bot

result = run_match(load_map("crossroads"), random_bot.Policy(1), random_bot.Policy(2), seed=0)
print(result.score, result.winner)
```

## Simulator notes

- Tick order follows design doc §4; each step is one method of `inkarena.sim.engine.Game`.
- Arrays are indexed `[y, x]`; map files list rows top to bottom, so line `y`, column `x` is cell
  `(x, y)`. Angles are `atan2(dy, dx)` in these coordinates. Team A's base is at the bottom
  (high `y`); team B sees the world rotated 180 degrees so it also starts at the bottom.
- A replay is the seed, the map name and the applied actions (`float[T, 6, 5]`: move x, move y,
  aim, fire, swim, absolute frame, after no-op substitution). `inkarena.replay.replay_states`
  re-simulates it exactly.

Choices the design doc left open, made here (change them in `engine.py` if they should differ):

- Splats and death bursts only paint cells their centre can see, so they don't cross walls.
- `fire=True` always surfaces the agent, even when the shot itself is blocked by cooldown or ink.
- A respawned agent uses the spawn point matching its index (agent 0 -> spawn `1`).
- The `enemies` list in an observation is sorted by position so its order carries no identity.
- After a crash the bot restarts with a fresh `Policy(seed)`, so its memory is lost.
- The average CPU budget starts with a 0.5 s head start, so a slow first tick isn't punished.
- Network blocking is best-effort (sockets raise inside the worker; on Linux the worker also tries
  an empty network namespace). Real isolation is the harness container's job (phase 5).
- Helper functions take the walls grid explicitly: `path(walls, a, b)` rather than `path(a, b)`.
