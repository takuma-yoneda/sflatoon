"""Replay query and render library, plus a small CLI.

Library (see inkarena/replay/query.py for details):

    import sys; sys.path.insert(0, "tools")
    from replay import load
    r = load("replays/local/m000.replay.npz")
    r.events("splat", agent="A1"); r.state_at(450); r.trace("B2", 0, 200)
    r.coverage(); r.render(450, "t450.png"); r.heatmap("position", "pos.png")

CLI:

    python tools/replay.py info    m000.replay.npz
    python tools/replay.py events  m000.replay.npz [--type splat] [--agent A1]
    python tools/replay.py render  m000.replay.npz --tick 450 -o t450.png
    python tools/replay.py heatmap m000.replay.npz --kind position [--team A] -o pos.png
    python tools/replay.py video   m000.replay.npz -o m000.mp4 [--speed 2] [--t0 0 --t1 600]
"""

from __future__ import annotations

import argparse
import json
import sys

from inkarena.replay import Replay, load

__all__ = ["Replay", "load"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("info", "events", "render", "heatmap", "video"):
        sp = sub.add_parser(name)
        sp.add_argument("replay")
        if name == "events":
            sp.add_argument("--type")
            sp.add_argument("--agent")
            sp.add_argument("--t0", type=int, default=0)
            sp.add_argument("--t1", type=int)
        if name == "render":
            sp.add_argument("--tick", type=int, required=True)
            sp.add_argument("--no-vision", action="store_true")
        if name == "heatmap":
            sp.add_argument("--kind", default="position", choices=["position", "deaths", "paint"])
            sp.add_argument("--team", choices=["A", "B"])
        if name == "video":
            sp.add_argument("--speed", type=float, default=1.0)
            sp.add_argument("--fps", type=int, default=60)
            sp.add_argument("--t0", type=int, default=0)
            sp.add_argument("--t1", type=int)
        if name in ("render", "heatmap", "video"):
            sp.add_argument("--scale", type=int, default=32)
            sp.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    r = load(args.replay)
    if args.cmd == "info":
        a, b = r.score[-1]
        print(f"map {r.map_name}  seed {r.seed}  ticks {r.ticks}  final A {a} - B {b}")
        counts: dict[str, int] = {}
        for e in r.events():
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        print("events:", json.dumps(counts))
    elif args.cmd == "events":
        for e in r.events(args.type, args.agent, args.t0, args.t1):
            print(json.dumps(e))
    elif args.cmd == "render":
        print(r.render(args.tick, args.out, vision=not args.no_vision, scale=args.scale))
    elif args.cmd == "heatmap":
        print(r.heatmap(args.kind, args.out, team=args.team, scale=args.scale))
    elif args.cmd == "video":
        print(
            r.video(
                args.out, fps=args.fps, speed=args.speed, t0=args.t0, t1=args.t1, scale=args.scale
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
