import math

import pytest
from helpers import act, fill, make_game, park, place, step


@pytest.fixture
def g():
    game = make_game()
    park(game, 1, 2, 4, 5)
    place(game, 0, 6.5, 7.5, facing=0.0)
    place(game, 3, 9.5, 7.5, facing=0.0)
    return game


def shoot_once(game, shooter=0):
    """Fire one shot and let the cooldown pass (3 ticks)."""
    events = [step(game, {shooter: act(fire=True)}), step(game), step(game)]
    return events


def test_three_hits_kill_with_burst_and_credit(g):
    shoot_once(g)
    shoot_once(g)
    assert g.state.hp[3] == 1
    step(g, {0: act(fire=True)})
    ev = step(g)
    assert len(ev.kills) == 1
    killer, victim, x, y, dist = ev.kills[0]
    assert (killer, victim) == (0, 3)
    assert (x, y) == (9.5, 7.5) and dist == pytest.approx(3.0)
    s = g.state
    assert not s.alive[3] and s.hp[3] == 0
    assert s.kills[0] == 1 and s.deaths[3] == 1
    assert s.respawn_in[3] == 30
    # Radius-2 burst in the killer's colour around the victim.
    assert s.grid[7, 11] == 1 and s.grid[5, 9] == 1 and s.grid[9, 9] == 1
    assert s.grid[7, 12] == 0


def test_burst_overrides_enemy_ink(g):
    fill(g, 8, 5, 12, 10, -1)
    g.state.hp[3] = 1
    step(g, {0: act(fire=True)})
    step(g)
    assert g.state.grid[9, 9] == 1


def test_respawn_after_30_ticks_with_invisibility(g):
    g.state.hp[3] = 1
    g.state.ink[3] = 10.0
    step(g, {0: act(fire=True)})
    step(g)
    death_tick = g.state.tick - 1
    obs = g.observe(1)
    assert obs["team"][0]["alive"] is False
    assert obs["team"][0]["respawn_in"] == 30
    assert obs["team"][0]["pos"] == (g.map.width - 9.5, g.map.height - 7.5)  # death position
    while g.state.tick < death_tick + 30:
        step(g)
        assert not g.state.alive[3]
    ev = step(g)
    assert ev.respawns == [3]
    s = g.state
    assert s.alive[3] and s.hp[3] == 3 and s.ink[3] == 100
    assert tuple(s.pos[3]) == tuple(g.map.spawns[1, 0])
    cx, cy = g.map.center
    expected = math.atan2(cy - s.pos[3, 1], cx - s.pos[3, 0]) % (2 * math.pi)
    assert s.facing[3] == pytest.approx(expected)
    assert s.invisible[3] == 20


def test_respawn_invisibility_lasts_20_ticks_and_agent_is_hittable(g):
    place(g, 3, 9.5, 7.5)
    g.state.invisible[3] = 20
    g._score_and_visibility()
    for _ in range(19):
        assert not g.state.visible[0, 0]
        step(g)
    assert not g.state.visible[0, 0]
    step(g)
    assert g.state.visible[0, 0]
    g.state.invisible[3] = 5
    shoot_once(g)
    assert g.state.hp[3] == 2


def test_mutual_kill_credits_both(g):
    g.state.facing[3] = math.pi
    g.state.hp[0] = g.state.hp[3] = 1
    step(g, {0: act(fire=True), 3: act(fire=True)})
    ev = step(g)
    assert sorted((k, v) for k, v, *_ in ev.kills) == [(0, 3), (3, 0)]
    assert g.state.kills[0] == g.state.kills[3] == 1
    assert not g.state.alive[0] and not g.state.alive[3]


def test_dead_agents_actions_are_ignored(g):
    park(g, 0)
    step(g, {0: act(fire=True, aim=2.0)})
    assert g.state.projectiles == []
    assert g.last_applied[0].tolist() == [0.0, 0.0, g.state.facing[0], 0.0, 0.0]


def test_hp_regen_timing(g):
    shoot_once(g)
    step(g, {0: act(fire=True)})
    step(g)
    assert g.state.hp[3] == 1
    hit_tick = g.state.tick - 1
    park(g, 0)
    hp = {}
    while g.state.tick <= hit_tick + 71:
        step(g)
        hp[g.state.tick - 1] = int(g.state.hp[3])
    assert hp[hit_tick + 29] == 1
    assert hp[hit_tick + 30] == 2
    assert hp[hit_tick + 49] == 2
    assert hp[hit_tick + 50] == 3
    assert hp[hit_tick + 71] == 3


def test_damage_resets_regen_timer(g):
    g.state.hp[3] = 2
    g.state.last_damage[3] = -25
    step(g, {0: act(fire=True)})
    step(g)  # hit on tick 1 before the regen at tick 5
    for _ in range(10):
        step(g)
    assert g.state.hp[3] == 1


def test_swim_refill(g):
    fill(g, 5, 5, 8, 9, 1)
    g.state.ink[0] = 50.0
    step(g, {0: act(swim=True)})
    assert g.state.ink[0] == 55.0
    g.state.ink[0] = 98.0
    step(g, {0: act(swim=True)})
    assert g.state.ink[0] == 100.0


def test_idle_refill_starts_5_ticks_after_last_shot(g):
    park(g, 3)
    step(g, {0: act(fire=True)})
    assert g.state.ink[0] == 96.0
    for _ in range(4):
        step(g)
        assert g.state.ink[0] == 96.0
    step(g)
    assert g.state.ink[0] == 96.5
    step(g)
    assert g.state.ink[0] == 97.0


def test_idle_refill_works_on_any_ground(g):
    park(g, 3)
    fill(g, 5, 5, 8, 9, -1)
    g.state.ink[0] = 10.0
    step(g)
    assert g.state.ink[0] == 10.5


def test_submerge_lost_when_own_ink_is_painted_over(g):
    fill(g, 5, 5, 8, 9, 1)
    place(g, 3, 9.5, 7.5, facing=math.pi)
    g.state.hp[0] = 3
    step(g, {0: act(swim=True)})
    assert g.state.submerged[0]
    step(g, {0: act(swim=True), 3: act(fire=True)})
    step(g, {0: act(swim=True)})
    assert g.state.grid[7, 6] == -1
    assert not g.state.submerged[0]
