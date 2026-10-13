# Ink Arena (sflatoon)

A top-down 2D Splatoon-like 2v2 territory shooter where LLMs compete by writing bots.
Bots never run per-tick inside a model; models author a `policy.py` that the simulator runs.

Status: phase 5 (harness pilot). Rules v1 are frozen: the rules are in [docs/rules.md](docs/rules.md) and the bot API in [docs/api.md](docs/api.md). See the design doc and build plan in the project for the roadmap.

## Layout

```
inkarena/
  sim/        numpy simulator: rules (config.py), maps, tick engine, team frame transform
  bots/       house bots (noop, random, baseline) and the bot helper library (helpers.py)
  arena/      bot loader, subprocess sandbox, match runner
  replay/     replay recorder (.npz + events.jsonl), query library, PNG/heatmap/MP4 renderer
  harness/    model workspace, budgeted agent loop (bash + view_image), official eval, pilot
  maps/       bundled ASCII maps (32x20, 180-degree symmetric)
tools/
  validate_maps.py   check map symmetry and reachability
  bench.py           matches/min across worker processes
  arena.py           local matches between two bots, with side swapping
  demo.py            play one match and print the board as ASCII
  replay.py          replay query library + CLI: info, events, render, heatmap, video
  tune.py            bot-vs-bot metrics under rule variants (used to pick the v1 values)
  pilot.py           one model writes and revises a bot, with an official eval per revision
docs/                rules v1 and the bot API
tests/               one test module per rule area
```

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python tools/validate_maps.py
python tools/bench.py --matches 32 --bot random_bot
python tools/arena.py --a baseline --b random -n 20 --out replays/local/
python tools/replay.py render replays/local/m000.replay.npz --tick 450 -o t450.png
python tools/replay.py video replays/local/m000.replay.npz -o m000.mp4   # needs ffmpeg or .[video]
python tools/tune.py --seeds 10                                         # compare rule variants
```

`arena.py --out DIR` writes `summary.json` (per match: id, opponent, map, side, result, score,
coverage every 10 s) plus `mNNN.replay.npz` and `mNNN.events.jsonl` per match. Replay arrays and
event fields are documented in `inkarena/replay/recorder.py`.

A bot is a directory with `policy.py` defining `Policy(seed)` with `act(obs) -> list[dict]`
(design doc section 3). Bots run in a subprocess: 1 s CPU for setup, 50 ms CPU hard limit per
tick (over it kills and restarts the bot and counts as a crash), a 5 ms average budget (over it
the tick becomes a no-op), 1 GiB address space, and sockets disabled. Crashes are no-op ticks;
3 crashes forfeit the match.

```python
from inkarena.sim import load_map, run_match
from inkarena.bots import random_bot

result = run_match(load_map("duel"), random_bot.Policy(1), random_bot.Policy(2), seed=0)
print(result.score, result.winner)
```

## Model pilot

`tools/pilot.py` runs one model through several revisions of its bot (design doc sections 6
and 7). Each revision starts a fresh agent loop in `RUN_DIR/workspace/`, with only `NOTES.md`
carried over. The model gets a `bash` tool and a `view_image` tool, and a budget of 50 turns,
200 local matches and 30 minutes, echoed after every tool result. When it submits, the bot is
frozen and plays 50 official matches against the baseline and 50 against the previous version.

```bash
python -m pip install -e ".[harness]"   # adds the anthropic SDK; needs API credentials
python tools/pilot.py --model MODEL_ID --run-dir runs/pilot1 --revisions 5 --local-shell
```

The model's commands run on the host unless `--shell-prefix` routes them into a sandbox, so
`--local-shell` is only for a disposable container or VM. `RUN_DIR/pilot.jsonl` gets one row
per revision: how the revision ended, turns, local matches, tokens, how many `ANALYSIS.md`
claims cite a match and tick, and the official win rates. Full transcripts are in
`RUN_DIR/transcripts/`. The system prompt is `inkarena/harness/prompts/system.md`.

## Simulator notes

- Current format: 2v2, 60 s matches (600 ticks), 32x20 maps. Team size and match length are
  `Rules.team_size` and `Rules.ticks`; a map needs at least `team_size` spawn digits (`1`, `2`, ...).

- Tick order follows design doc §4; each step is one method of `inkarena.sim.engine.Game`.
- Arrays are indexed `[y, x]`; map files list rows top to bottom, so line `y`, column `x` is cell
  `(x, y)`. Angles are `atan2(dy, dx)` in these coordinates. Team A's base is at the bottom
  (high `y`); team B sees the world rotated 180 degrees so it also starts at the bottom.
- A replay is the seed, the map name and the applied actions (`float[T, n, 5]`: move x, move y,
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
