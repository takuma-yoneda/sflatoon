"""The model's workspace (design doc section 7) and bot versions.

workspace/
  bot/policy.py            the model's code (it may add modules under bot/)
  NOTES.md                 its only memory between revisions
  ANALYSIS.md              required with every submission
  docs/rules.md, docs/api.md
  lib/helpers.py           the bot helper library, for reading (bots import inkarena.bots.helpers)
  opponents/rN/            frozen copies of its own past versions, for local matches
  replays/round_N/         official results: <opponent>/summary.json plus replays
  replays/local/           where tools/arena.py writes by default
  tools/arena.py, tools/replay.py
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import inkarena

DOCS = ("rules.md", "api.md")

ARENA_TOOL = '''"""Local matches. Each match counts against this revision's budget.

    python tools/arena.py --a bot/ --b baseline -n 20 --maps all --out replays/local/
    python tools/arena.py --a bot/ --b opponents/r2 -n 20
"""

import sys

from inkarena.harness.wstools import arena_main

if __name__ == "__main__":
    sys.exit(arena_main())
'''

REPLAY_TOOL = '''"""Replay query and render library, plus a CLI (`python tools/replay.py -h`).

    import sys; sys.path.insert(0, "tools")
    from replay import load
    r = load("replays/local/m000.replay.npz")
    r.events("splat", agent="A1"); r.state_at(450); r.trace("B1", 0, 200)
    r.coverage(); r.render(450, "t450.png"); r.heatmap("position", "pos.png")
"""

import sys

from inkarena.replay.cli import Replay, load, main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
'''

NOTES = "# Notes\n\nYour memory between revisions. Nothing else carries over.\n"

ANALYSIS = """# Analysis

What you changed this revision and why. Back every claim about bot behavior with a match ID
and tick, for example "replays/local/m004, tick 312: A0 walks into the wall at (10, 7)".
"""


def repo_docs_dir() -> Path:
    """docs/ next to the inkarena package (an editable or source checkout)."""
    return Path(inkarena.__file__).resolve().parents[1] / "docs"


def create_workspace(ws: Path, start_bot: Path | None = None) -> Path:
    """A fresh workspace. The bot starts as a copy of ``start_bot`` (default: baseline)."""
    ws = Path(ws)
    for d in ("bot", "docs", "lib", "opponents", "replays/local", "tools"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    bots = resources.files("inkarena.bots")
    if start_bot is None:
        (ws / "bot" / "policy.py").write_text(bots.joinpath("baseline.py").read_text())
    else:
        shutil.copytree(start_bot, ws / "bot", dirs_exist_ok=True)
    (ws / "lib" / "helpers.py").write_text(bots.joinpath("helpers.py").read_text())
    (ws / "lib" / "helpers.py").chmod(0o444)
    for name in DOCS:
        shutil.copy(repo_docs_dir() / name, ws / "docs" / name)
    (ws / "tools" / "arena.py").write_text(ARENA_TOOL)
    (ws / "tools" / "replay.py").write_text(REPLAY_TOOL)
    (ws / "NOTES.md").write_text(NOTES)
    (ws / "ANALYSIS.md").write_text(ANALYSIS)
    return ws


def snapshot_bot(ws: Path, dest: Path) -> Path:
    """Freeze the workspace's bot/ into ``dest`` (without caches)."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ws / "bot", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def publish_opponent(version_dir: Path, ws: Path) -> None:
    """Make a frozen version available to the model as ``opponents/<name>``."""
    target = ws / "opponents" / version_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(version_dir, target)
