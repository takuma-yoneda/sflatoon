"""Find and import a bot's ``Policy`` class.

A bot spec is one of:
- a directory containing ``policy.py`` (the bot may add modules next to it),
- a path to a ``.py`` file,
- the name of a house bot: ``noop``, ``random``, ``baseline``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

HOUSE_BOTS = {
    "noop": "inkarena.bots.noop",
    "random": "inkarena.bots.random_bot",
    "baseline": "inkarena.bots.baseline",
}


def resolve(spec: str) -> str:
    """Normalize a bot spec: house bot names stay, paths become absolute."""
    if spec in HOUSE_BOTS:
        return spec
    path = Path(spec).expanduser().resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"no bot at {spec!r} (expected a dir with policy.py, a .py file, "
            f"or one of {sorted(HOUSE_BOTS)})"
        )
    return str(path)


def load_policy_class(spec: str):
    spec = resolve(spec)
    if spec in HOUSE_BOTS:
        return importlib.import_module(HOUSE_BOTS[spec]).Policy
    path = Path(spec)
    sys.path.insert(0, str(path.parent))  # let the bot import its own modules
    module_spec = importlib.util.spec_from_file_location("policy", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules["policy"] = module
    module_spec.loader.exec_module(module)
    return module.Policy


def label(spec: str) -> str:
    if spec in HOUSE_BOTS:
        return spec
    p = Path(spec)
    return p.parent.name if p.name == "policy.py" else p.stem
