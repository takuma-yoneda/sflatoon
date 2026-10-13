"""Play one match and print the board as ASCII.

    python tools/demo.py [--map duel] [--seed 0] [--every 200]

Legend: '#' wall, 'A'/'B' agents, 'o'/'x' team A/B ink, 'a'/'b' bases, '.' unpainted.
Team A uses the random bot; team B stands still, so A should out-paint B.
"""

from __future__ import annotations

import argparse

from inkarena.bots import noop, random_bot
from inkarena.sim import Game, load_map


def board(game: Game) -> str:
    s = game.state
    rows = []
    for y in range(game.map.height):
        row = []
        for x in range(game.map.width):
            if game.walls[y, x]:
                row.append("#")
            elif game.base[y, x]:
                row.append("a" if game.base[y, x] > 0 else "b")
            else:
                row.append({1: "o", -1: "x", 0: "."}[int(s.grid[y, x])])
        rows.append(row)
    for k in range(game.n):
        if s.alive[k]:
            x, y = (int(v) for v in s.pos[k])
            rows[y][x] = "A" if k < game.m else "B"
    return "\n".join("".join(r) for r in rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", default="duel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=200, help="print the board every N ticks")
    args = p.parse_args()

    game = Game(load_map(args.map), seed=args.seed)
    a, b = random_bot.Policy(args.seed), noop.Policy()
    while not game.done:
        game.step(a.act(game.observe(0)), b.act(game.observe(1)))
        if game.state.tick % args.every == 0 or game.done:
            print(f"tick {game.state.tick}  score A {game.state.score[0]}  B {game.state.score[1]}")
            print(board(game), end="\n\n")
    winner = {0: "team A", 1: "team B", None: "draw"}[game.winner()]
    print(f"final score {game.state.score}, winner: {winner}")


if __name__ == "__main__":
    main()
