"""The pilot: one model, several revisions, an official evaluation after each (build plan phase 5).

    run_dir/
      workspace/               the model's workspace (see workspace.py)
      versions/r0 .. rN/       frozen bots; r0 is the starting bot
      transcripts/rN.jsonl     full agent transcript per revision
      budget/rN.json           budget state per revision
      pilot.jsonl              one row per revision: budget used, official results

A run can be resumed: revisions already in pilot.jsonl are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ..arena.runner import make_jobs, play, run_jobs, tally, write_summary
from ..sim import list_maps
from .agent import Model, run_revision
from .workspace import create_workspace, publish_opponent, snapshot_bot

DEFAULT_LIMITS = {"max_turns": 50, "max_matches": 600, "max_seconds": 30 * 60}
# Past versions every official eval also plays, as a fixed yardstick next to the previous one.
DEFAULT_ANCHORS = (1,)
CITATION = re.compile(r"\bm\d{3}\b[^\n]{0,40}?\btick\s*\d+", re.IGNORECASE)


def bot_loads(bot_dir: Path) -> str | None:
    """None if the bot plays a sandboxed match against noop without crashing, else why not."""
    if not (bot_dir / "policy.py").is_file():
        return "bot/policy.py is missing"
    job = make_jobs(str(bot_dir), "noop", 1, [list_maps()[0]])[0]
    r = play(job)
    if r["crashes"][0] or r["forfeit"] == "a":
        return f"crashed {r['crashes'][0]} times: {r['errors'][0]}"
    return None


def official_eval(bot: Path, opponent: str, out_dir: Path, n: int, workers: int) -> dict:
    jobs = make_jobs(str(bot), opponent, n, list_maps(), sandbox=True, out_dir=str(out_dir))
    results = run_jobs(jobs, workers)
    write_summary(out_dir / "summary.json", results)
    return tally(results)


def first_message(rev: int, n_revisions: int, limits: dict, last: dict | None) -> str:
    lines = [
        f"This is revision {rev} of {n_revisions}.",
        f"Budget: {limits['max_turns']} turns, {limits['max_matches']} local matches, "
        f"{int(limits['max_seconds'] // 60)} minutes.",
    ]
    if last is None:
        lines.append(
            "This is the first revision. bot/policy.py starts as a copy of the house baseline."
        )
    else:
        lines.append(
            f"Your last submission is frozen as opponents/r{rev - 1}. Its official results:"
        )
        for opp, t in last["eval"].items():
            lines.append(
                f"- vs {opp}: {t['wins']}W {t['draws']}D {t['losses']}L, win rate "
                f"{t['win_rate']:.0%} ± {t['stderr']:.0%} (replays/round_{rev - 1}/{opp}/)"
            )
        if last.get("invalid"):
            lines.append(f"Note: that submission failed to load ({last['invalid']}).")
    lines.append("Start by reading NOTES.md.")
    return "\n".join(lines)


def _rows(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines() if x] if log.exists() else []


def run_pilot(
    model: Model,
    run_dir: Path,
    revisions: int = 5,
    limits: dict | None = None,
    eval_matches: int = 50,
    workers: int | None = None,
    shell_prefix: list[str] | None = None,
    anchors: tuple[int, ...] = DEFAULT_ANCHORS,
) -> list[dict]:
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    workers = workers or os.cpu_count() or 1
    run_dir = Path(run_dir).resolve()
    ws, versions, log = run_dir / "workspace", run_dir / "versions", run_dir / "pilot.jsonl"
    for d in ("transcripts", "budget"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    rows = _rows(log)
    if not rows:  # revision 0: the starting bot, evaluated the same way
        create_workspace(ws)
        r0 = snapshot_bot(ws, versions / "r0")
        publish_opponent(r0, ws)
        t = official_eval(
            r0, "baseline", ws / "replays" / "round_0" / "baseline", eval_matches, workers
        )
        rows.append({"revision": 0, "eval": {"baseline": t}})
        with open(log, "a") as f:
            f.write(json.dumps(rows[-1]) + "\n")

    for rev in range(len(rows), revisions + 1):
        prev = versions / f"r{rev - 1}"
        result = run_revision(
            model,
            ws,
            first_message(rev, revisions, limits, rows[-1]),
            run_dir / "budget" / f"r{rev}.json",
            run_dir / "transcripts" / f"r{rev}.jsonl",
            limits,
            shell_prefix,
        )
        version = snapshot_bot(ws, versions / f"r{rev}")
        invalid = bot_loads(version)
        if invalid:  # fall back to the previous version so the series stays comparable
            shutil.rmtree(version)
            shutil.copytree(prev, version)
        publish_opponent(version, ws)
        round_dir = ws / "replays" / f"round_{rev}"
        opponents = {"baseline": "baseline", f"r{rev - 1}": str(prev)}
        for a in anchors:
            if a < rev - 1:
                opponents[f"r{a}"] = str(versions / f"r{a}")
        evals = {
            name: official_eval(version, spec, round_dir / name, eval_matches, workers)
            for name, spec in opponents.items()
        }
        analysis = (ws / "ANALYSIS.md").read_text() if (ws / "ANALYSIS.md").exists() else ""
        row = {
            "revision": rev,
            "model": model.name,
            "ended": result.ended,
            "turns": result.turns,
            "local_matches": result.matches,
            "seconds": result.seconds,
            "usage": result.usage,
            "invalid": invalid,
            "analysis_chars": len(analysis),
            "analysis_citations": len(CITATION.findall(analysis)),
            "eval": evals,
        }
        shutil.copy(ws / "ANALYSIS.md", run_dir / "transcripts" / f"r{rev}.ANALYSIS.md")
        rows.append(row)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
    return rows
