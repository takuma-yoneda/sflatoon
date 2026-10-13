import math

import pytest
from helpers import act, make_game, park, place, step


@pytest.fixture
def g():
    game = make_game()
    park(game, 1, 2, 4, 5)
    place(game, 0, 6.5, 7.5, facing=0.0)
    return game


def seen(game, x, y, submerged=False):
    place(game, 3, x, y)
    game.state.submerged[3] = submerged
    game._score_and_visibility()
    return bool(game.state.visible[0, 0])


def at_angle(deg, dist, origin=(6.5, 7.5)):
    a = math.radians(deg)
    return origin[0] + dist * math.cos(a), origin[1] + dist * math.sin(a)


def test_range_is_12_inclusive(g):
    assert seen(g, 18.5, 7.5)
    assert not seen(g, 18.6, 7.5)


def test_cone_is_120_degrees(g):
    assert seen(g, *at_angle(59, 5))
    assert seen(g, *at_angle(-59, 5))
    assert not seen(g, *at_angle(61, 5))
    assert not seen(g, *at_angle(180, 3))


def test_walls_block_sight():
    g = make_game(walls=[(10, 6), (10, 7), (10, 8)])
    park(g, 1, 2, 4, 5)
    place(g, 0, 6.5, 7.5, facing=0.0)
    assert not seen(g, 12.5, 7.5)
    assert seen(g, 12.5, 10.5)


def test_submerged_enemies_need_distance_2(g):
    assert seen(g, 8.5, 7.5, submerged=True)
    assert not seen(g, 8.6, 7.5, submerged=True)
    assert seen(g, 8.6, 7.5, submerged=False)


def test_submerged_still_needs_cone(g):
    assert not seen(g, 5.0, 7.5, submerged=True)


def test_team_vision_is_union_of_living_agents(g):
    place(g, 1, 16.5, 7.5, facing=math.pi)
    assert seen(g, 4.5, 7.5)  # behind agent 0, in front of agent 1
    park(g, 1)
    assert not seen(g, 4.5, 7.5)


def test_respawn_invisible_enemies_are_never_visible(g):
    place(g, 3, 8.5, 7.5)
    g.state.invisible[3] = 1
    g._score_and_visibility()
    assert not g.state.visible[0, 0]


def test_observation_lists_only_visible_enemies(g):
    place(g, 3, 9.5, 7.5)
    place(g, 4, 3.5, 7.5)
    step(g)
    obs = g.observe(0)
    assert [e["pos"] for e in obs["enemies"]] == [(9.5, 7.5)]


def test_enemy_projectiles_follow_visibility(g):
    park(g, 3)
    place(g, 4, 10.5, 2.5, facing=math.pi / 2)  # shooting "down" across agent 0's view
    place(g, 5, 4.5, 12.5, facing=0.0)  # behind agent 0
    step(g, {4: act(fire=True), 5: act(fire=True)})
    obs = g.observe(0)
    theirs = [p for p in obs["projectiles"] if not p["ours"]]
    assert len(theirs) == 1
    assert theirs[0]["pos"] == pytest.approx((10.5, 4.0))
    assert len(g.observe(1)["projectiles"]) == 2  # own projectiles are always known


def test_enemies_have_no_ids(g):
    place(g, 3, 12.5, 9.5)
    place(g, 4, 9.5, 6.5)
    step(g)
    obs = g.observe(0)
    assert set(obs["enemies"][0]) == {"pos", "facing", "swimming"}
    assert [e["pos"] for e in obs["enemies"]] == sorted(e["pos"] for e in obs["enemies"])
