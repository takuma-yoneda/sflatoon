import math

import pytest
from helpers import act, fill, make_game, park, place, step


@pytest.fixture
def g():
    game = make_game()
    park(game, 1, 2, 3, 4, 5)
    place(game, 0, 6.5, 7.5, facing=0.0)
    return game


def fire(game, k=0):
    return step(game, {k: act(fire=True)})


def test_projectile_speed_and_range(g):
    fire(g)
    (p,) = g.state.projectiles
    assert (p.x, p.y) == pytest.approx((8.0, 7.5))
    for _ in range(2):
        step(g)
    assert g.state.projectiles[0].x == pytest.approx(11.0)
    step(g)  # fourth tick: 6 cells travelled, lands
    assert g.state.projectiles == []


def test_strip_and_splat_painting(g):
    fire(g)
    for _ in range(3):
        step(g)
    grid = g.state.grid
    # Strip: cell centres within 1.0 of the path (6.5, 7.5) -> (12.5, 7.5).
    assert (grid[7, 5:14] == 1).all()
    assert (grid[6, 6:13] == 1).all() and (grid[8, 6:13] == 1).all()
    assert grid[6, 5] == 0 and grid[8, 5] == 0
    # Splat: radius 1.5 around the landing point (12.5, 7.5).
    assert grid[6, 13] == 1 and grid[8, 13] == 1
    assert grid[7, 14] == 0 and grid[5, 12] == 0
    scorable = g.map.scorable
    painted = (grid != 0) & scorable
    assert painted.sum() == painted[5:10].sum()  # nothing painted outside rows 5-9
    assert g.state.score == (int((grid[g.map.scorable] == 1).sum()), 0)


def test_walls_stop_shots_and_splats_do_not_cross_walls():
    g = make_game(walls=[(10, 6), (10, 7), (10, 8)])
    park(g, 1, 2, 3, 4, 5)
    place(g, 0, 6.5, 7.5, facing=0.0)
    fire(g)
    step(g)
    step(g)
    assert g.state.projectiles == []  # landed at the wall on its third tick
    assert g.state.grid[7, 9] == 1
    assert g.state.grid[7, 11] == 0  # behind the wall


def test_hit_deals_one_damage_and_lands_at_target(g):
    place(g, 3, 9.5, 7.5)
    fire(g)
    assert g.state.hp[3] == 3
    ev = step(g)
    assert ev.hits == [(0, 3)]
    assert g.state.hp[3] == 2
    assert g.state.projectiles == []
    assert g.state.grid[7, 9] == 1  # splat around the hit point (9.1, 7.5)
    assert g.state.grid[7, 11] == 0


def test_first_enemy_on_path_is_hit(g):
    place(g, 3, 10.5, 7.5)
    place(g, 4, 9.5, 7.5)
    fire(g)
    step(g)
    assert g.state.hp[4] == 2 and g.state.hp[3] == 3


def test_no_friendly_fire(g):
    place(g, 1, 8.5, 7.5)
    place(g, 3, 10.5, 7.5)
    fire(g)
    step(g)
    step(g)
    assert g.state.hp[1] == 3
    assert g.state.hp[3] == 2


def test_submerged_and_invisible_enemies_can_be_hit(g):
    place(g, 3, 9.5, 7.5)
    fill(g, 9, 7, 9, 7, -1)
    g.state.submerged[3] = True
    place(g, 4, 9.5, 9.5)
    g.state.invisible[4] = 10
    step(g, {3: act(swim=True)})
    assert g.state.submerged[3]
    fire(g)
    step(g, {3: act(swim=True)})
    assert g.state.hp[3] == 2
    step(g)  # cooldown
    g.state.facing[0] = math.atan2(2.0, 3.0)
    fire(g)
    step(g)
    step(g)
    assert g.state.hp[4] == 2


def test_dead_agents_cannot_be_hit(g):
    place(g, 3, 9.5, 7.5)
    g.state.alive[3] = False
    g.state.respawn_in[3] = 100
    fire(g)
    step(g)
    assert len(g.state.projectiles) == 1


def test_fire_cooldown_and_ink_cost(g):
    shots = [len(step(g, {0: act(fire=True)}).shots) for _ in range(7)]
    assert shots == [1, 0, 0, 1, 0, 0, 1]
    assert g.state.ink[0] == pytest.approx(100 - 12)


def test_firing_needs_ink(g):
    g.state.ink[0] = 3.9
    assert fire(g).shots == []
    g.state.ink[0] = 4.0
    assert fire(g).shots == [0]
    assert g.state.ink[0] == 0.0


def test_shot_direction_is_facing_after_turn(g):
    step(g, {0: act(aim=math.pi / 2, fire=True)})
    (p,) = g.state.projectiles
    assert (p.dx, p.dy) == pytest.approx((math.cos(math.pi / 6), math.sin(math.pi / 6)))


def test_shot_keeps_flying_after_shooter_dies(g):
    fire(g)
    park(g, 0)
    step(g)
    assert len(g.state.projectiles) == 1


def test_projectiles_pass_through_each_other(g):
    place(g, 0, 6.5, 6.9)
    place(g, 3, 12.5, 8.1, facing=math.pi)
    step(g, {0: act(fire=True), 3: act(fire=True)})
    step(g)
    assert len(g.state.projectiles) == 2


def test_simultaneous_paint_keeps_previous_value(g):
    place(g, 0, 6.5, 6.5, facing=0.0)
    place(g, 3, 8.0, 8.5, facing=math.pi)
    g.state.grid[7, 6] = 1
    g.state.grid[7, 7] = 0
    step(g, {0: act(fire=True), 3: act(fire=True)})
    assert g.state.grid[7, 6] == 1 and g.state.grid[7, 7] == 0  # both teams painted
    assert g.state.grid[6, 7] == 1  # only team A
    assert g.state.grid[8, 7] == -1  # only team B


def test_base_cells_cannot_be_painted():
    g = make_game()
    park(g, 0, 1, 2, 4, 5)
    place(g, 3, 2.5, 10.5, facing=math.pi / 2)
    step(g, {3: act(fire=True)})
    for _ in range(4):
        step(g)
    assert (g.state.grid[13:15, 1:5] == 1).all()
    assert g.state.grid[12, 2] == -1
    a, b = g.state.score
    assert a == 0 and b == int((g.state.grid[g.map.scorable] == -1).sum()) > 0
