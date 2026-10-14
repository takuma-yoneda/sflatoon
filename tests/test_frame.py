import math

import numpy as np
import pytest
from helpers import make_game, park, place

from inkarena.bots import random_bot
from inkarena.sim import Game, Rules, load_map


def test_team_b_observation_is_rotated():
    g = make_game()
    park(g, 1, 2, 4, 5)
    place(g, 0, 6.5, 7.5, facing=0.0)
    place(g, 3, 9.5, 7.5, facing=math.pi / 2)
    g.state.grid[3, 5] = 1
    g.state.grid[10, 20] = -1
    g._score_and_visibility()
    obs = g.observe(1)
    w, h = g.map.width, g.map.height
    me = obs["team"][0]
    assert me["pos"] == (w - 9.5, h - 7.5)
    assert me["facing"] == pytest.approx(3 * math.pi / 2)
    assert obs["ink"][h - 1 - 3, w - 1 - 5] == -1  # team A's ink is "theirs"
    assert obs["ink"][h - 1 - 10, w - 1 - 20] == 1
    assert np.array_equal(obs["walls"], g.walls[::-1, ::-1])
    a, b = g.state.score
    assert obs["score"] == (b, a)


def test_team_b_actions_are_rotated():
    g = make_game()
    park(g, 1, 2, 4, 5)
    place(g, 3, 9.5, 7.5, facing=0.0)
    noop = {"move": (0, 0), "aim": 0.0, "fire": False, "swim": False}
    # In B's frame: move +x and aim at +y. Absolute: move -x, aim at -y.
    b = [{"move": (1.0, 0.0), "aim": math.pi / 2, "fire": False, "swim": False}]
    g.step([noop] * 3, b + [noop] * 2)
    assert g.state.pos[3, 0] == pytest.approx(9.25)
    assert g.state.facing[3] == pytest.approx(math.radians(-30) % (2 * math.pi))


def test_both_bases_start_at_the_bottom_in_own_frame():
    g = Game(load_map("duel"))
    a, b = g.observe(0), g.observe(1)
    assert [m["pos"] for m in a["team"]] == pytest.approx([m["pos"] for m in b["team"]])
    assert np.array_equal(a["ink"], b["ink"])
    assert np.array_equal(a["walls"], b["walls"])
    assert all(y > g.map.height / 2 for _, y in (m["pos"] for m in a["team"]))


def test_mirrored_play_stays_symmetric():
    """Both teams run the same seeded bot; the state must stay 180-degree symmetric.

    Spawn jitter is mirrored, so it keeps the symmetry; shot spread doesn't, so it is off.
    """
    g = Game(load_map("zigzag"), seed=3, rules=Rules(shot_spread=0.0))
    pa, pb = random_bot.Policy(7), random_bot.Policy(7)
    w, h = g.map.width, g.map.height
    for _ in range(600):
        g.step(pa.act(g.observe(0)), pb.act(g.observe(1)))
        s = g.state
        assert np.array_equal(s.grid, -s.grid[::-1, ::-1])
        m = g.m
        mirrored = np.stack([w - s.pos[m:, 0], h - s.pos[m:, 1]], axis=1)
        assert np.allclose(s.pos[:m], mirrored, atol=1e-9)
        assert np.array_equal(s.hp[:m], s.hp[m:])
        assert np.array_equal(s.alive[:m], s.alive[m:])
        assert s.score[0] == s.score[1]


def test_default_format_is_2v2_for_60_seconds():
    g = Game(load_map("duel"))
    assert g.n == 4 and g.rules.ticks == 600
    obs = g.observe(1)
    assert len(obs["team"]) == 2 and obs["ticks_left"] == 600
    assert g.state.visible.shape == (2, 2)
    # Ranges are half the original 64x40 values, to match the half-size maps.
    assert g.rules.view_range == 6.0 and g.rules.shot_range == 3.0
    noop = {"move": (0, 0), "aim": 0.0, "fire": False, "swim": False}
    g.step([noop] * 2, [noop] * 2)
    assert g.state.tick == 1
    g.step([noop] * 3, [noop] * 2)  # wrong length for team A: no-op, not a crash
    assert g.state.tick == 2


def test_spawn_jitter_is_seeded_mirrored_and_kept_for_respawns():
    m = load_map("duel")
    g1, g2, g3 = Game(m, seed=1), Game(m, seed=1), Game(m, seed=2)
    assert np.array_equal(g1.spawns, g2.spawns)
    assert not np.array_equal(g1.spawns, g3.spawns)
    assert np.abs(g1.spawns - m.spawns).max() <= g1.rules.spawn_jitter
    w, h = m.width, m.height
    assert g1.spawns[1] == pytest.approx(
        np.stack([w - g1.spawns[0, :, 0], h - g1.spawns[0, :, 1]], 1)
    )
    assert g1.state.pos[0] == pytest.approx(g1.spawns[0, 0])
    assert Game(m, rules=Rules(spawn_jitter=0.0)).spawns == pytest.approx(m.spawns)


def test_shot_spread_is_seeded():
    def shots(seed):
        g = Game(load_map("duel"), seed=seed, rules=Rules(spawn_jitter=0.0))
        fire = [{"move": (0.0, 0.0), "aim": 1.0, "fire": True, "swim": False}] * 2
        g.step(fire, fire)
        return [(p.dx, p.dy) for p in g.state.projectiles]

    assert shots(5) == shots(5)
    assert shots(5) != shots(6)
