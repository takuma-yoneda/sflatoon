"""Harness pilot: one model writes and revises a bot, with an official eval after each revision.

    python tools/pilot.py --model MODEL_ID --run-dir runs/pilot1 --revisions 5 --local-shell

Needs `pip install -e ".[harness]"` and Anthropic API credentials. The model's bash commands
run on this machine unless you pass --shell-prefix (for example
--shell-prefix "docker exec -i -w /ws pilot"), so --local-shell is required to confirm that
this machine is itself a disposable container or VM. Re-running with the same --run-dir
resumes after the last finished revision.
"""

from __future__ import annotations

import argparse
import shlex
import sys

from inkarena.harness.agent import AnthropicModel
from inkarena.harness.pilot import DEFAULT_LIMITS, run_pilot


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--revisions", type=int, default=5)
    p.add_argument("--turns", type=int, default=DEFAULT_LIMITS["max_turns"])
    p.add_argument("--matches", type=int, default=DEFAULT_LIMITS["max_matches"])
    p.add_argument("--minutes", type=float, default=DEFAULT_LIMITS["max_seconds"] / 60)
    p.add_argument("--eval-matches", type=int, default=50)
    p.add_argument("--workers", type=int)
    p.add_argument("--shell-prefix", help="command prefix that runs bash in a sandbox")
    p.add_argument("--local-shell", action="store_true", help="run commands on this machine")
    args = p.parse_args()
    if not args.shell_prefix and not args.local_shell:
        p.error("pass --shell-prefix, or --local-shell if this machine is disposable")

    limits = {
        "max_turns": args.turns,
        "max_matches": args.matches,
        "max_seconds": args.minutes * 60,
    }
    rows = run_pilot(
        AnthropicModel(args.model, effort=args.effort),
        args.run_dir,
        revisions=args.revisions,
        limits=limits,
        eval_matches=args.eval_matches,
        workers=args.workers,
        shell_prefix=shlex.split(args.shell_prefix) if args.shell_prefix else None,
    )
    print(f"{'rev':>3}  {'ended':<12} {'turns':>5} {'local':>5}  vs baseline")
    for r in rows:
        t = r["eval"]["baseline"]
        print(
            f"{r['revision']:>3}  {r.get('ended', '-'):<12} {r.get('turns', 0):>5} "
            f"{r.get('local_matches', 0):>5}  {t['win_rate']:.0%} ± {t['stderr']:.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
