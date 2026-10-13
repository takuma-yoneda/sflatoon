import json

from inkarena.arena.runner import make_jobs, run_jobs, tally, write_summary
from inkarena.replay import load_replay, replay_states
from inkarena.sim import list_maps


def test_jobs_swap_sides_and_cycle_maps():
    jobs = make_jobs("baseline", "random", 6, ["duel", "zigzag"])
    assert [j.swapped for j in jobs] == [False, True] * 3
    assert [j.map_name for j in jobs] == ["duel"] * 2 + ["zigzag"] * 2 + ["duel"] * 2
    assert jobs[0].seed == jobs[1].seed != jobs[2].seed


def test_baseline_beats_random(tmp_path):
    jobs = make_jobs("baseline", "random", 8, list_maps(), sandbox=True, out_dir=str(tmp_path))
    results = run_jobs(jobs, workers=4)
    t = tally(results)
    assert t["wins"] == 8, results
    assert {r["side"] for r in results} == {"A", "B"}
    write_summary(tmp_path / "summary.json", results)
    assert json.loads((tmp_path / "summary.json").read_text())["tally"]["wins"] == 8
    # The saved replay re-simulates to the reported score.
    r = results[1]
    rep = load_replay(tmp_path / f"{r['id']}.replay.npz")
    *_, final = replay_states(rep)
    ours, theirs = final.state.score[::-1] if r["side"] == "B" else final.state.score
    assert [ours, theirs] == r["score"]


def test_tally():
    rs = [{"result": "win"}, {"result": "draw"}, {"result": "loss"}, {"result": "win"}]
    t = tally(rs)
    assert (t["wins"], t["draws"], t["losses"]) == (2, 1, 1)
    assert t["win_rate"] == 0.625
