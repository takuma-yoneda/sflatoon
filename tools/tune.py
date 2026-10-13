"""Rule tuning sweep: play bot-vs-bot matches under rule variants and compare how games play.

python tools/tune.py                                  # default rules plus the built-in variants
python tools/tune.py --set shot_cost=3 --set splat_radius=1.0 --name cheap_small
python tools/tune.py --only default --replays out/    # also save one replay per map

Each variant changes some fields of the default Rules. Metrics, averaged over matches:
  painted   share of scorable cells painted at the end (either team)
  margin    |A - B| / (A + B) at the end; 0 is a dead heat, 1 is a shutout
  leads     lead changes after the first 10 s
  center    ink flips (one team's cell repainted by the other) in the middle third, per match
  kills     splats per match
  swim      share of alive agent-ticks spent submerged
  empty     ink_empty events per match
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from inkarena.arena.loader import load_policy_class
from inkarena.sim import DEFAULT_RULES, Rules, list_maps, load_map, run_match

# One change at a time from the defaults, to see which knobs matter.
VARIANTS: dict[str, dict] = {
    "default": {},
    "cheap_shots": {"shot_cost": 3.0},
    "costly_shots": {"shot_cost": 6.0},
    "fast_refill": {"swim_refill": 8.0},
    "small_splat": {"splat_radius": 1.0, "strip_halfwidth": 0.7},
    "fast_walk": {"walk_speed": 0.33},
    "quick_respawn": {"respawn_delay": 20},
    "long_view": {"view_range": 8.0},
    "long_shot": {"shot_range": 4.0},
}

METRICS = ("painted", "margin", "leads", "center", "kills", "swim", "empty")


def match_metrics(result, game_map) -> dict:
    rec = result.recorder
    arrays = rec.arrays()
    score = arrays["score"].astype(np.int64)
    total = int(game_map.scorable.sum())
    a, b = (int(v) for v in score[-1])

    diff = np.sign(score[100:, 0] - score[100:, 1])
    diff = diff[diff != 0]
    leads = int((diff[1:] != diff[:-1]).sum())

    h = game_map.walls.shape[0]
    ink = arrays["ink0"].astype(np.int8).copy()
    flips = 0
    for _, y, x, v in arrays["ink_diff"]:
        if h // 3 <= y < h - h // 3 and ink[y, x] == -v and v != 0:
            flips += 1
        ink[y, x] = v

    agents = arrays["agents"]  # columns: x, y, facing, hp, ink, submerged, alive, invisible
    alive = agents[:, :, 6] > 0
    swim = float((agents[:, :, 5] > 0)[alive].mean()) if alive.any() else 0.0
    events = rec.events
    return {
        "painted": (a + b) / total,
        "margin": abs(a - b) / max(1, a + b),
        "leads": leads,
        "center": flips,
        "kills": sum(e["type"] == "splat" for e in events),
        "swim": swim,
        "empty": sum(e["type"] == "ink_empty" for e in events),
    }


def _play(job):
    name, overrides, bot_a, bot_b, map_name, seed, replay_dir = job
    rules = dataclasses.replace(DEFAULT_RULES, **overrides)
    game_map = load_map(map_name)
    pa, pb = load_policy_class(bot_a)(2 * seed), load_policy_class(bot_b)(2 * seed + 1)
    result = run_match(game_map, pa, pb, seed=seed, rules=rules, record=True)
    if replay_dir and seed == 0:
        out = Path(replay_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        result.recorder.save(out / f"{map_name}.replay.npz", out / f"{map_name}.events.jsonl")
    m = match_metrics(result, game_map)
    m["winner"] = result.winner
    return name, m


def parse_set(items: list[str]) -> dict:
    fields = {f.name: f.type for f in dataclasses.fields(Rules)}
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if key not in fields:
            raise SystemExit(f"unknown rule {key!r}")
        out[key] = int(value) if fields[key] in (int, "int") else float(value)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", default="baseline")
    p.add_argument("--b", default="baseline")
    p.add_argument("--seeds", type=int, default=10, help="seeds per map")
    p.add_argument("--maps", default="all")
    p.add_argument("--only", help="comma-separated variant names")
    p.add_argument("--set", action="append", default=[], help="rule=value for a custom variant")
    p.add_argument("--name", default="custom")
    p.add_argument("--replays", help="directory for one replay per variant and map (seed 0)")
    p.add_argument("--json", help="write per-variant means here")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    variants = dict(VARIANTS)
    if args.set:
        variants = {args.name: parse_set(args.set)}
    if args.only:
        variants = {k: variants[k] for k in args.only.split(",")}
    maps = list_maps() if args.maps == "all" else args.maps.split(",")
    jobs = [
        (name, ov, args.a, args.b, m, s, args.replays)
        for name, ov in variants.items()
        for m in maps
        for s in range(args.seeds)
    ]
    with Pool(args.workers) as pool:
        results = pool.map(_play, jobs, chunksize=1)

    summary = {}
    print(f"{'variant':16}" + "".join(f"{k:>9}" for k in METRICS) + "   A-B-draw")
    for name, ov in variants.items():
        rows = [m for n, m in results if n == name]
        means = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        w = [r["winner"] for r in rows]
        means["record"] = [w.count(0), w.count(1), w.count(None)]
        means["overrides"] = ov
        summary[name] = means
        cells = "".join(
            f"{means[k]:9.2f}" if k in ("painted", "margin", "swim") else f"{means[k]:9.1f}"
            for k in METRICS
        )
        print(f"{name:16}{cells}   {'-'.join(map(str, means['record']))}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
