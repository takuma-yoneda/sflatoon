import json

import numpy as np
import pytest
from helpers import act, make_game, park, place, step

from inkarena.bots import baseline, random_bot
from inkarena.replay import Recorder, load
from inkarena.sim import load_map, run_match


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    out = tmp_path_factory.mktemp("rep")
    r = run_match(load_map("duel"), baseline.Policy(), random_bot.Policy(3), seed=2, record=True)
    r.recorder.save(out / "m000.replay.npz", out / "m000.events.jsonl")
    return r, load(out / "m000.replay.npz"), out


def test_arrays_and_shapes(match):
    r, rep, _ = match
    assert rep.ticks == 600 and rep.n_states == 601
    assert rep.agents.shape == (601, 4, 8)
    assert rep.visible.shape == (601, 2, 2)
    assert rep.actions.shape == (600, 4, 5)
    assert np.array_equal(rep.actions, r.actions)
    assert tuple(rep.score[-1]) == r.score


def test_ink_reconstruction_matches_score(match):
    _, rep, _ = match
    m = rep.map
    for t in (0, 1, 99, 100, 101, 377, 600):
        ink = rep.ink_at(t)
        assert ((ink == 1) & m.scorable).sum() == rep.score[t, 0]
        assert ((ink == -1) & m.scorable).sum() == rep.score[t, 1]


def test_state_at_and_trace(match):
    _, rep, _ = match
    s = rep.state_at(500)
    assert [a["name"] for a in s["agents"]] == ["A0", "A1", "B0", "B1"]
    assert s["score"] == tuple(rep.score[500])
    tr = rep.trace("A1", 10, 20)
    assert tr.shape == (11, 8)
    a1 = rep.state_at(10)["agents"][1]
    assert (float(tr[0, 0]), float(tr[0, 1])) == (a1["x"], a1["y"])
    with pytest.raises(IndexError):
        rep.state_at(1801)


def test_events(match):
    r, rep, out = match
    types = {e["type"] for e in rep.events()}
    assert {"splat", "respawn", "coverage_checkpoint", "enemy_spotted"} <= types
    assert len(rep.events("coverage_checkpoint")) == 6
    assert sum(r.kills) == len(rep.events("splat"))
    for e in rep.events("splat", agent="B1"):
        assert "B1" in (e["killer"], e["victim"])
    assert all(100 <= e["tick"] <= 200 for e in rep.events(t0=100, t1=200))
    lines = (out / "m000.events.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["type"] in types


def test_respawn_and_splat_events_line_up(match):
    _, rep, _ = match
    for e in rep.events("splat")[:5]:
        victim = rep.agent_index(e["victim"])
        assert not rep.state_at(e["tick"] + 1)["agents"][victim]["alive"]
        assert rep.state_at(e["tick"])["agents"][victim]["alive"]


def test_ink_empty_and_victim_saw_killer():
    g = make_game()
    park(g, 1, 2, 4, 5)
    place(g, 0, 6.5, 7.5, facing=0.0)
    place(g, 3, 9.5, 7.5, facing=0.0)  # facing away from the shooter
    g.state.ink[0] = 5.0
    g.state.hp[3] = 1
    rec = Recorder(g)
    for a in ({0: act(fire=True)}, {}):
        prev = rec.pre_step()
        ev = step(g, a)
        rec.record(g.last_applied, ev, prev)
    (empty,) = [e for e in rec.events if e["type"] == "ink_empty"]
    assert empty == {"type": "ink_empty", "tick": 0, "agent": "A0"}
    (splat,) = [e for e in rec.events if e["type"] == "splat"]
    assert splat["killer"] == "A0" and splat["victim"] == "B0"
    assert splat["victim_saw_killer"] is False
    assert splat["dist"] == pytest.approx(3.0)


def test_render_png_and_heatmaps(match, tmp_path):
    _, rep, _ = match
    from PIL import Image

    for t in (0, 300, 600):
        img = Image.open(rep.render(t, tmp_path / f"t{t}.png"))
        assert img.size == (32 * 32, 20 * 32 + 64)
    for kind in ("position", "deaths", "paint"):
        img = Image.open(rep.heatmap(kind, tmp_path / f"{kind}.png", team="A"))
        assert img.size == (32 * 32, 20 * 32)
    with pytest.raises(ValueError):
        rep.heatmap("nope", tmp_path / "x.png")


def test_video_clip(match, tmp_path):
    _, rep, _ = match
    out = rep.video(tmp_path / "clip.mp4", t0=100, t1=110, scale=8)
    assert out.stat().st_size > 1000
