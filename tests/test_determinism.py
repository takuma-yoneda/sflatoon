import numpy as np

from inkarena.bots import noop, random_bot
from inkarena.replay import ActionReplay, load_replay, replay_states, save_replay
from inkarena.sim import Game, load_map, run_match


def digests_of(game_map, actions, seed):
    return [
        g.state.digest()
        for g in replay_states(ActionReplay(seed, game_map.name, actions), game_map)
    ]


def test_same_seed_and_bots_give_identical_matches():
    m = load_map("crossroads")
    r1 = run_match(m, random_bot.Policy(1), random_bot.Policy(2), seed=5)
    r2 = run_match(m, random_bot.Policy(1), random_bot.Policy(2), seed=5)
    assert r1.final_digest == r2.final_digest
    assert np.array_equal(r1.actions, r2.actions)
    assert r1.score == r2.score


def test_replay_reproduces_every_tick(tmp_path):
    m = load_map("zigzag")
    game = Game(m, seed=3)
    pa, pb = random_bot.Policy(11), random_bot.Policy(12)
    live, applied = [game.state.digest()], []
    while not game.done:
        a = game.read_actions(pa.act(game.observe(0)), pb.act(game.observe(1)))
        applied.append(a)
        game.step_applied(a)
        live.append(game.state.digest())
    actions = np.stack(applied)
    path = tmp_path / "m000.npz"
    save_replay(path, ActionReplay(3, m.name, actions))
    loaded = load_replay(path)
    assert loaded.map_name == "zigzag" and loaded.seed == 3
    assert digests_of(m, loaded.actions, loaded.seed) == live


def test_run_match_actions_replay_to_final_state():
    m = load_map("crossroads")
    r = run_match(m, random_bot.Policy(4), noop.Policy(), seed=9)
    assert digests_of(m, r.actions, r.seed)[-1] == r.final_digest
    assert r.actions.shape == (1800, 6, 5)


def test_crashing_policy_is_a_noop_then_forfeits():
    class Crash:
        def act(self, obs):
            raise RuntimeError("boom")

    m = load_map("crossroads")
    r = run_match(m, Crash(), random_bot.Policy(0), seed=0)
    assert r.ticks == 2  # crashes on ticks 0 and 1 are no-ops; the third forfeits
    assert not r.actions[:, :3, [0, 1, 3]].any()  # no moves, no fire
    assert r.forfeit == 0 and r.winner == 1
    assert r.crashes == (3, 0)
