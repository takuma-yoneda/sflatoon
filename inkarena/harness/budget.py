"""Per-revision budget: agent turns, local matches and wall clock (design doc section 6).

The state lives in a JSON file outside the workspace. The harness charges turns, and the
workspace's ``tools/arena.py`` charges matches through ``INKARENA_BUDGET``, which points at
that file.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ENV = "INKARENA_BUDGET"


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_turns: int = 50
    max_matches: int = 200
    max_seconds: float = 30 * 60
    turns: int = 0
    matches: int = 0
    started: float = 0.0

    @classmethod
    def start(cls, path: Path, **limits) -> Budget:
        b = cls(**limits, started=time.time())
        b.save(path)
        return b

    @classmethod
    def load(cls, path: Path) -> Budget:
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(self)))
        os.replace(tmp, path)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def exhausted(self) -> str | None:
        """Why the revision must stop, or None."""
        if self.turns >= self.max_turns:
            return "turns"
        if self.seconds_left <= 0:
            return "wall clock"
        return None

    def echo(self) -> str:
        used = int(self.elapsed)
        line = (
            f"[budget] turns {self.turns}/{self.max_turns} · "
            f"local matches {self.matches}/{self.max_matches} · "
            f"time {used // 60}m{used % 60:02d}s of {int(self.max_seconds // 60)}m"
        )
        if self.max_turns - self.turns <= 3 or self.seconds_left < 120:
            line += " · almost out: finish bot/policy.py and ANALYSIS.md now"
        return line


def charge_matches(n: int) -> None:
    """Called by the workspace arena before it plays ``n`` matches."""
    path = os.environ.get(ENV)
    if not path:
        return  # outside the harness: no budget
    # Lock so that arena runs started in parallel each get charged.
    with open(Path(path).with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        b = Budget.load(Path(path))
        if b.matches + n > b.max_matches:
            left = b.max_matches - b.matches
            raise BudgetExceeded(f"only {left} local matches left this revision, asked for {n}")
        b.matches += n
        b.save(Path(path))
