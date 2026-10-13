import math

from inkarena.sim import Game, parse_map

W, H = 24, 16


def open_map_text(walls=()):
    """An open arena with team A's base at the bottom-left, B's at the top-right.

    ``walls`` is a list of (x, y) cells to wall off (mirrored automatically).
    """
    g = [["." for _ in range(W)] for _ in range(H)]
    for x in range(W):
        g[0][x] = g[H - 1][x] = "#"
    for y in range(H):
        g[y][0] = g[y][W - 1] = "#"
    for y in (13, 14):
        for x in (1, 2, 3, 4):
            g[y][x] = "a"
            g[H - 1 - y][W - 1 - x] = "b"
    g[14][1], g[14][2], g[14][3] = "1", "2", "3"
    for x, y in walls:
        g[y][x] = "#"
        g[H - 1 - y][W - 1 - x] = "#"
    return "\n".join("".join(r) for r in g)


def make_game(walls=(), **rules):
    from inkarena.sim import Rules

    return Game(parse_map(open_map_text(walls), name="open"), rules=Rules(**rules))


def park(game, *agents):
    """Take agents out of play: dead, with a respawn that never comes."""
    s = game.state
    for k in agents:
        s.alive[k] = False
        s.respawn_in[k] = 10**9
        s.pos[k] = (0.5, 0.5)


def place(game, k, x, y, facing=0.0):
    s = game.state
    s.pos[k] = (x, y)
    s.facing[k] = facing
    s.alive[k] = True
    s.hp[k] = 3
    s.invisible[k] = 0


def act(move=(0.0, 0.0), aim=None, fire=False, swim=False):
    return {"move": move, "aim": aim, "fire": fire, "swim": swim}


def step(game, actions=None):
    """Step with absolute-frame per-agent actions given as {agent: act(...)}.

    Agents without an entry hold still and keep their facing. ``aim=None`` keeps
    the current facing. Converts team B's actions into its own frame.
    """
    actions = actions or {}
    teams = [[], []]
    s = game.state
    for k in range(6):
        a = dict(actions.get(k, act()))
        if a["aim"] is None:
            a["aim"] = float(s.facing[k])
        if k >= 3:
            a["move"] = (-a["move"][0], -a["move"][1])
            a["aim"] = a["aim"] + math.pi
        teams[k // 3].append(a)
    return game.step(teams[0], teams[1])


def paint(game, cells, value):
    for x, y in cells:
        game.state.grid[y, x] = value


def fill(game, x0, y0, x1, y1, value):
    game.state.grid[y0 : y1 + 1, x0 : x1 + 1] = value
    game.state.grid[game.walls] = 0
    base = game.base != 0
    game.state.grid[base] = game.base[base]
