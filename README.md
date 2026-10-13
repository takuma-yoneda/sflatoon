# Ink Arena (sflatoon)

A top-down 2D Splatoon-like 3v3 territory shooter where LLMs compete by writing bots.
Bots never run per-tick inside a model; models author a `policy.py` that the simulator runs.

Status: phase 1 (simulator core). See the design doc and build plan in the project for the roadmap.

## Layout

```
inkarena/
  sim/        numpy simulator: rules (config.py), maps, tick engine, team frame transform
  bots/       house bots (noop, random)
  replay/     replays (phase 1: seed + map + applied actions)
  maps/       bundled ASCII maps (64x40, 180-degree symmetric)
tools/
  validate_maps.py   check map symmetry and reachability
  bench.py           matches/min across worker processes
tests/               one test module per rule area
```

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python tools/validate_maps.py
python tools/bench.py --matches 32 --bot random_bot
```

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
- In-process policies that raise count as invalid output for that tick; the subprocess sandbox
  with time limits and crash forfeits is phase 2.
