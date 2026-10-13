import math

import pytest
from helpers import fill, make_game, park, place

GOOD = {"move": (1.0, 0.0), "aim": math.pi / 2, "fire": True, "swim": False}


@pytest.fixture
def g():
    game = make_game()
    park(game, 4, 5)
    place(game, 0, 6.5, 7.5, facing=0.0)
    place(game, 3, 16.5, 7.5, facing=0.0)
    return game


@pytest.mark.parametrize(
    "bad",
    [
        {"move": (1.0, 0.0), "aim": 0.0, "fire": True},  # missing key
        {"move": (float("nan"), 0.0), "aim": 0.0, "fire": True, "swim": False},
        {"move": (1.0, 0.0), "aim": float("inf"), "fire": True, "swim": False},
        {"move": "fast", "aim": 0.0, "fire": True, "swim": False},
        None,
    ],
)
def test_invalid_agent_action_is_a_noop(g, bad):
    g.step([bad, GOOD, GOOD], [GOOD] * 3)
    s = g.state
    assert tuple(s.pos[0]) == (6.5, 7.5)
    assert s.facing[0] == 0.0
    assert all(p.owner != 0 for p in s.projectiles)
    assert s.pos[1, 0] != g.map.spawns[0, 1, 0]  # the valid agents still act


@pytest.mark.parametrize("bad", [[GOOD, GOOD], [GOOD] * 4, "nope", None])
def test_wrong_list_is_a_noop_for_the_team(g, bad):
    g.step(bad, [GOOD] * 3)
    s = g.state
    assert tuple(s.pos[0]) == (6.5, 7.5)
    assert not any(p.owner < 3 for p in s.projectiles)
    assert any(p.owner == 3 for p in s.projectiles)


def test_noop_keeps_valid_submerge(g):
    fill(g, 5, 5, 8, 9, 1)
    g.step([{**GOOD, "move": (0, 0), "fire": False, "swim": True}] + [GOOD] * 2, [GOOD] * 3)
    assert g.state.submerged[0]
    g.step([{"bad": 1}, GOOD, GOOD], [GOOD] * 3)
    assert g.state.submerged[0]
