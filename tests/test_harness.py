import json
import os
import subprocess
import sys

import pytest

from inkarena.harness.agent import Reply, run_revision
from inkarena.harness.budget import Budget, BudgetExceeded, charge_matches
from inkarena.harness.pilot import run_pilot
from inkarena.harness.tools import load_image, truncate
from inkarena.harness.workspace import create_workspace


class Scripted:
    """A stand-in model that replays fixed replies and records what it was sent."""

    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def step(self, system, messages, tools):
        self.seen.append(json.loads(json.dumps(messages)))
        return (
            self.replies.pop(0)
            if self.replies
            else Reply([{"type": "text", "text": "done"}], "end_turn")
        )


def bash(cmd, i=0):
    return Reply(
        [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": cmd}}], "tool_use"
    )


def test_budget_charges_and_refuses(tmp_path, monkeypatch):
    path = tmp_path / "b.json"
    Budget.start(path, max_turns=5, max_matches=10, max_seconds=60)
    monkeypatch.setenv("INKARENA_BUDGET", str(path))
    charge_matches(6)
    with pytest.raises(BudgetExceeded, match="only 4"):
        charge_matches(6)
    b = Budget.load(path)
    assert b.matches == 6 and "local matches 6/10" in b.echo()


def test_parallel_charges_all_count(tmp_path):
    bf = tmp_path / "budget.json"
    Budget.start(bf, max_matches=200)
    env = dict(os.environ, INKARENA_BUDGET=str(bf))
    code = "from inkarena.harness.budget import charge_matches as c\nfor _ in range(25): c(1)"
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(4)]
    assert [p.wait() for p in procs] == [0, 0, 0, 0]
    assert Budget.load(bf).matches == 100


def test_truncate_keeps_head_and_tail():
    text = "a" * 20_000 + "END"
    out = truncate(text)
    assert len(out) < 10_100 and out.startswith("a") and out.endswith("END") and "cut" in out


def test_view_image_stays_in_workspace(tmp_path):
    (tmp_path / "x.png").write_bytes(b"\x89PNG fake")
    block, err = load_image(tmp_path, "x.png")
    assert block["source"]["media_type"] == "image/png" and err is None
    assert load_image(tmp_path, "../x.png")[0] is None
    assert load_image(tmp_path, "missing.png")[0] is None


def test_workspace_arena_charges_nothing_for_a_bad_bot(tmp_path):
    ws = create_workspace(tmp_path / "ws")
    budget = tmp_path / "b.json"
    Budget.start(budget, max_matches=50)
    env = {**os.environ, "INKARENA_BUDGET": str(budget)}
    for bad in (["--a", "missing/"], ["--a", "bot/", "--maps", "nowhere"]):
        cmd = [sys.executable, "tools/arena.py", "--b", "noop", "-n", "20", *bad]
        p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True)
        assert p.returncode != 0
    assert Budget.load(budget).matches == 0


def test_workspace_arena_counts_matches(tmp_path):
    ws = create_workspace(tmp_path / "ws")
    budget = tmp_path / "b.json"
    Budget.start(budget, max_matches=3)
    env = {**os.environ, "INKARENA_BUDGET": str(budget)}
    cmd = [
        sys.executable,
        "tools/arena.py",
        "--a",
        "bot/",
        "--b",
        "noop",
        "-n",
        "2",
        "--workers",
        "1",
    ]
    ok = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    refused = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True)
    assert refused.returncode != 0 and "only 1 local matches left" in refused.stderr
    assert Budget.load(budget).matches == 2


def test_revision_runs_tools_and_echoes_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    ws = create_workspace(tmp_path / "ws")
    model = Scripted(
        bash("head -1 NOTES.md; echo key=${ANTHROPIC_API_KEY:-none}", 0),
        Reply(
            [{"type": "tool_use", "id": "t1", "name": "view_image", "input": {"path": "nope.png"}}],
            "tool_use",
        ),
    )
    log = run_revision(model, ws, "go", tmp_path / "b.json", tmp_path / "t.jsonl", {"max_turns": 5})
    assert log.ended == "submitted" and log.turns == 3
    first = model.seen[1][-1]["content"][0]
    out = first["content"][0]["text"]
    assert "# Notes" in out and "key=none" in out
    assert first["content"][-1]["text"].startswith("[budget] turns 1/5")
    assert model.seen[2][-1]["content"][0]["is_error"]
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert kinds[0] == "start" and kinds[-1] == "end"


def test_revision_stops_when_turns_run_out(tmp_path):
    ws = create_workspace(tmp_path / "ws")
    model = Scripted(*[bash("true", i) for i in range(10)])
    log = run_revision(model, ws, "go", tmp_path / "b.json", tmp_path / "t.jsonl", {"max_turns": 2})
    assert log.ended == "turns" and log.turns == 2


def test_pilot_evaluates_and_falls_back_on_a_broken_bot(tmp_path):
    model = Scripted(bash("echo 'raise RuntimeError(1)' > bot/policy.py"))
    rows = run_pilot(
        model, tmp_path / "run", revisions=1, limits={"max_turns": 3}, eval_matches=2, workers=2
    )
    assert [r["revision"] for r in rows] == [0, 1]
    r1 = rows[1]
    assert r1["invalid"] and set(r1["eval"]) == {"baseline", "r0"}
    assert r1["eval"]["baseline"]["matches"] == 2
    run = tmp_path / "run"
    assert (run / "versions" / "r1" / "policy.py").read_text() == (
        run / "versions" / "r0" / "policy.py"
    ).read_text()
    assert (run / "workspace" / "replays" / "round_1" / "baseline" / "summary.json").exists()
    assert (run / "workspace" / "opponents" / "r1" / "policy.py").exists()
    # Resuming a finished run does nothing more.
    assert len(run_pilot(Scripted(), run, revisions=1, eval_matches=2, workers=2)) == 2


def test_pilot_evals_also_play_anchor_versions(tmp_path):
    run = tmp_path / "run"
    rows = run_pilot(Scripted(), run, revisions=3, eval_matches=2, workers=2, anchors=(1,))
    assert [set(r["eval"]) for r in rows[1:]] == [
        {"baseline", "r0"},
        {"baseline", "r1"},
        {"baseline", "r2", "r1"},
    ]
