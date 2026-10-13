from pathlib import Path

import pytest

from inkarena.arena.sandbox import Limits, SandboxedPlayer
from inkarena.sim import Game, load_map, run_match

BOTS = Path(__file__).parent / "bots"


@pytest.fixture(scope="module")
def obs():
    return Game(load_map("crossroads")).observe(0)


def tick_obs(obs, t):
    return {**obs, "tick": t}


def test_house_bot_runs_in_sandbox(obs):
    with SandboxedPlayer("baseline") as p:
        acts = p.act(obs)
    assert len(acts) == 3 and p.crashes == 0


def test_bot_dir_with_own_modules_and_prints(obs):
    with SandboxedPlayer(str(BOTS / "printing_bot")) as p:
        acts = p.act(obs)
    assert p.crashes == 0, p.errors
    assert acts[0]["move"] == (1.0, 0.0)


def test_exception_is_a_crash_and_noop(obs):
    with SandboxedPlayer(str(BOTS / "crash_bot.py")) as p:
        assert p.act(obs) is None
        assert p.crashes == 1 and "boom" in p.errors[0]
        assert p.act(obs) is None  # restarted, crashes again
        assert p.crashes == 2


def test_hard_cpu_limit_kills_and_restarts(obs):
    with SandboxedPlayer(str(BOTS / "slow_bot.py")) as p:
        assert p.act(tick_obs(obs, 4)) is not None
        assert p.act(tick_obs(obs, 5)) is None
        assert p.crashes == 1 and p.timeouts == 1
        assert p.act(tick_obs(obs, 6)) is not None  # fresh process
        assert p.crashes == 1


def test_hung_bot_is_killed_by_wall_clock(obs):
    limits = Limits(wall_grace=0.3)
    with SandboxedPlayer(str(BOTS / "hang_bot.py"), limits=limits) as p:
        assert p.act(obs) is None
        assert p.crashes == 1 and p.timeouts == 1


def test_average_cpu_budget_gives_noops_not_crashes(obs):
    limits = Limits(burst_cpu=0.05)
    with SandboxedPlayer(str(BOTS / "greedy_bot.py"), limits=limits) as p:
        results = [p.act(tick_obs(obs, t)) for t in range(10)]
    assert p.crashes == 0
    assert results[0] is not None
    assert p.slow_ticks >= 5
    assert sum(r is None for r in results) == p.slow_ticks


def test_setup_budget(obs):
    with SandboxedPlayer(str(BOTS / "slow_setup_bot.py")) as p:
        assert p.crashes == 1 and "setup" in p.errors[0]


def test_memory_limit_is_a_crash(obs):
    with SandboxedPlayer(str(BOTS / "memory_bot.py"), limits=Limits(memory_bytes=1 << 30)) as p:
        assert p.act(obs) is None
        assert p.crashes == 1 and "MemoryError" in p.errors[0]


def test_network_is_blocked(obs):
    with SandboxedPlayer(str(BOTS / "network_bot.py")) as p:
        assert p.act(obs) is None
        assert p.crashes == 1


def test_three_crashes_forfeit_the_match():
    m = load_map("crossroads")
    with SandboxedPlayer(str(BOTS / "crash_bot.py")) as a, SandboxedPlayer("noop") as b:
        r = run_match(m, a, b)
    assert r.forfeit == 0 and r.winner == 1 and r.ticks == 2


def test_missing_bot_raises():
    with pytest.raises(FileNotFoundError):
        SandboxedPlayer(str(BOTS / "nope"))
