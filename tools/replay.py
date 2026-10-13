"""Thin wrapper; see inkarena/replay/cli.py."""

import sys

from inkarena.replay.cli import *  # noqa: F403  (re-exports the library for `from replay import load`)
from inkarena.replay.cli import main

if __name__ == "__main__":
    sys.exit(main())
