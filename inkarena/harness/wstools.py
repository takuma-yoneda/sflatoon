"""Workspace-side tools. The workspace's tools/arena.py calls :func:`arena_main`."""

from __future__ import annotations

import sys

from ..arena.cli import main
from .budget import BudgetExceeded, charge_matches


def _charge(args) -> None:
    if args.no_sandbox:
        raise SystemExit("--no-sandbox is not available in the workspace")
    try:
        charge_matches(args.n)
    except BudgetExceeded as e:
        raise SystemExit(f"not run: {e}") from None


def arena_main(argv: list[str] | None = None) -> int:
    return main(argv, before_run=_charge)


if __name__ == "__main__":
    sys.exit(arena_main())
