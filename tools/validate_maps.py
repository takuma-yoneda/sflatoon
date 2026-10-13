"""Check that every bundled map (or the given map files) loads and validates.

python tools/validate_maps.py [map.txt ...]
"""

from __future__ import annotations

import sys

from inkarena.sim import MapError, list_maps, load_map


def main(argv: list[str]) -> int:
    targets = argv or list_maps()
    failed = 0
    for target in targets:
        try:
            m = load_map(target)
        except (MapError, OSError) as e:
            print(f"FAIL {target}: {e}")
            failed += 1
            continue
        floor, scorable = int((~m.walls).sum()), int(m.scorable.sum())
        print(f"ok   {m.name}: {m.width}x{m.height}, {floor} floor cells, {scorable} scorable")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
