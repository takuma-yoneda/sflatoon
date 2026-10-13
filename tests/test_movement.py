import math

import pytest
from helpers import act, fill, make_game, park, place, step


@pytest.fixture
def g():
    game = make_game()
    park(game, 1, 2, 3, 4, 5)
    return game


def test_turn_is_capped_at_30_degrees(g):
    place(g, 0, 10.5, 7.5, facing=0.0)
    step(g, {0: act(aim=math.pi / 2)})
    assert g.state.facing[0] == pytest.approx(math.radians(30))
    step(g, {0: act(aim=math.pi / 2)})
    step(g, {0: act(aim=math.pi / 2)})
    assert g.state.facing[0] == pytest.approx(math.pi / 2)
    step(g, {0: act(aim=math.pi / 2)})
    assert g.state.facing[0] == pytest.approx(math.pi / 2)


def test_turn_takes_the_short_way_round(g):
    place(g, 0, 10.5, 7.5, facing=math.radians(10))
    step(g, {0: act(aim=math.radians(-10))})
    assert g.state.facing[0] == pytest.approx(math.radians(350))


def test_walk_speed_and_move_clipping(g):
    place(g, 0, 10.5, 7.5)
    step(g, {0: act(move=(3.0, 4.0))})
    assert g.state.pos[0, 0] == pytest.approx(10.5 + 0.25 * 0.6)
    assert g.state.pos[0, 1] == pytest.approx(7.5 + 0.25 * 0.8)
    place(g, 0, 10.5, 7.5)
    step(g, {0: act(move=(0.5, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(10.5 + 0.125)


def test_move_is_independent_of_facing(g):
    place(g, 0, 10.5, 7.5, facing=0.0)
    step(g, {0: act(move=(-1.0, 0.0), aim=0.0)})
    assert g.state.pos[0, 0] == pytest.approx(10.25)
    assert g.state.facing[0] == 0.0


def test_enemy_ink_halves_speed(g):
    place(g, 0, 10.5, 7.5)
    fill(g, 8, 5, 14, 9, -1)
    step(g, {0: act(move=(1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(10.5 + 0.125)


def test_speed_uses_cell_at_start_of_tick(g):
    # Starts on neutral ground, moves onto enemy ink: full speed this tick.
    place(g, 0, 10.9, 7.5)
    fill(g, 11, 5, 14, 9, -1)
    step(g, {0: act(move=(1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(11.15)
    step(g, {0: act(move=(1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(11.275)


def test_swim_on_own_ink(g):
    place(g, 0, 10.5, 7.5)
    fill(g, 5, 5, 18, 9, 1)
    step(g, {0: act(move=(1.0, 0.0), swim=True)})
    assert g.state.submerged[0]
    assert g.state.pos[0, 0] == pytest.approx(11.0)


def test_swim_stationary_stays_submerged(g):
    place(g, 0, 10.5, 7.5)
    fill(g, 5, 5, 18, 9, 1)
    for _ in range(3):
        step(g, {0: act(swim=True)})
        assert g.state.submerged[0]
    assert tuple(g.state.pos[0]) == (10.5, 7.5)


def test_swim_ignored_off_own_ink(g):
    place(g, 0, 10.5, 7.5)
    step(g, {0: act(move=(1.0, 0.0), swim=True)})
    assert not g.state.submerged[0]
    assert g.state.pos[0, 0] == pytest.approx(10.75)
    fill(g, 5, 5, 18, 9, -1)
    step(g, {0: act(swim=True)})
    assert not g.state.submerged[0]


def test_submerged_agent_cannot_leave_own_ink(g):
    place(g, 0, 10.5, 7.5)
    fill(g, 10, 7, 10, 7, 1)  # a single own-ink cell
    for _ in range(4):
        step(g, {0: act(move=(1.0, 0.0), swim=True)})
        assert g.state.submerged[0]
    assert 10.9 < g.state.pos[0, 0] < 11.0


def test_fire_wins_over_swim(g):
    place(g, 0, 10.5, 7.5)
    fill(g, 5, 5, 18, 9, 1)
    step(g, {0: act(swim=True)})
    assert g.state.submerged[0]
    step(g, {0: act(swim=True, fire=True)})
    assert not g.state.submerged[0]
    assert len(g.state.projectiles) == 1


def test_walls_block_movement():
    g = make_game(walls=[(16, y) for y in range(1, 7)])
    park(g, 1, 2, 3, 4, 5)
    place(g, 0, 15.3, 3.5)
    for _ in range(5):
        step(g, {0: act(move=(1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(15.6, abs=1e-3)
    assert g.state.pos[0, 1] == 3.5


def test_wall_sliding_keeps_tangential_motion():
    g = make_game(walls=[(12, y) for y in range(1, 7)])
    park(g, 1, 2, 3, 4, 5)
    place(g, 0, 11.6, 3.5)
    d = 1 / math.sqrt(2)
    step(g, {0: act(move=(d, d))})
    assert g.state.pos[0, 0] == pytest.approx(11.6, abs=1e-3)
    assert g.state.pos[0, 1] == pytest.approx(3.5 + 0.25 * d)


def test_map_border_is_wall(g):
    place(g, 0, 1.5, 7.5)
    for _ in range(5):
        step(g, {0: act(move=(-1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(1.4, abs=1e-3)


def test_agents_pass_through_each_other():
    g = make_game()
    park(g, 2, 3, 4, 5)
    place(g, 0, 10.5, 7.5)
    place(g, 1, 10.7, 7.5)
    step(g, {0: act(move=(1.0, 0.0))})
    assert g.state.pos[0, 0] == pytest.approx(10.75)


def test_dead_agents_do_not_move(g):
    park(g, 0)
    before = tuple(g.state.pos[0])
    step(g, {0: act(move=(1.0, 0.0))})
    assert tuple(g.state.pos[0]) == before
