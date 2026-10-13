"""Play local matches between two bots.

    python tools/arena.py --a bot/ --b baseline -n 20 --maps all --out replays/local/

``--a``/``--b`` take a bot directory (with policy.py), a .py file, or a house bot
name (noop, random, baseline). Matches alternate sides, so each map and seed is
played once from each side. Results are reported from A's point of view.
"""

from __future__ import annotations

import argparse
import os
import sys

from inkarena.arena.runner import make_jobs, run_jobs, tally, write_summary
from inkarena.sim import list_maps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="bot under test")
    p.add_argument("--b", required=True, help="opponent")
    p.add_argument("-n", type=int, default=20, help="number of matches")
    p.add_argument("--maps", default="all", help="'all' or a comma-separated list")
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--out", default=None, help="directory for summary.json and replays")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--no-sandbox", action="store_true", help="run bots in-process (trusted only)")
    args = p.parse_args()

    maps = list_maps() if args.maps == "all" else args.maps.split(",")
    jobs = make_jobs(args.a, args.b, args.n, maps, not args.no_sandbox, args.out, args.seed)
    results = run_jobs(jobs, args.workers)
    for r in results:
        extra = f"  forfeit by {r['forfeit']}" if r["forfeit"] else ""
        print(
            f"{r['id']}  {r['map']:<12} side {r['side']}  {r['result']:<5} "
            f"{r['score'][0]:>5} - {r['score'][1]:<5}{extra}"
        )
    t = tally(results)
    print(
        f"\n{t['wins']}W {t['draws']}D {t['losses']}L  "
        f"win rate {t['win_rate']:.1%} ± {t['stderr']:.1%}"
    )
    if args.out:
        write_summary(os.path.join(args.out, "summary.json"), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
