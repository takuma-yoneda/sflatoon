"""Simulator throughput: full matches per minute across worker processes.

python tools/bench.py --matches 32 --bot noop --workers 4 [--min-rate 200]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from multiprocessing import Pool

from inkarena.sim import list_maps, load_map, run_match


def _play(args: tuple[str, str, int]) -> tuple[int, int]:
    bot, map_name, seed = args
    module = importlib.import_module(f"inkarena.bots.{bot}")
    result = run_match(load_map(map_name), module.Policy(seed), module.Policy(seed + 1), seed=seed)
    return result.score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matches", type=int, default=32)
    p.add_argument("--bot", default="noop", help="module under inkarena.bots")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--min-rate", type=float, default=None, help="exit 1 below this matches/min")
    args = p.parse_args()

    maps = list_maps()
    jobs = [(args.bot, maps[i % len(maps)], i) for i in range(args.matches)]
    start = time.perf_counter()
    with Pool(args.workers) as pool:
        pool.map(_play, jobs, chunksize=1)
    elapsed = time.perf_counter() - start
    rate = args.matches / elapsed * 60
    print(
        f"{args.matches} matches ({args.bot}) on {args.workers} workers in {elapsed:.1f}s: "
        f"{rate:.0f} matches/min"
    )
    if args.min_rate is not None and rate < args.min_rate:
        print(f"below the required {args.min_rate:.0f} matches/min")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
