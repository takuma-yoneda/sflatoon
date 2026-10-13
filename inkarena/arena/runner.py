"""Play matches between bot specs, with side swapping and multiprocessing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from ..sim import load_map, run_match
from ..sim.match import InProcessPlayer
from .loader import label, load_policy_class, resolve
from .sandbox import SandboxedPlayer


@dataclass(frozen=True)
class Job:
    match_id: str
    bot_a: str  # spec of the bot under test ("A" in the CLI)
    bot_b: str
    map_name: str
    seed: int
    swapped: bool  # True: bot_a plays as team B
    sandbox: bool = True
    out_dir: str | None = None


def make_jobs(bot_a, bot_b, n, maps, sandbox=True, out_dir=None, seed0=0) -> list[Job]:
    """n matches cycling over maps; every map/seed is played from both sides."""
    a, b = resolve(bot_a), resolve(bot_b)
    jobs = []
    for i in range(n):
        pair, swapped = divmod(i, 2)
        jobs.append(
            Job(
                match_id=f"m{i:03d}",
                bot_a=a,
                bot_b=b,
                map_name=maps[pair % len(maps)],
                seed=seed0 + pair,
                swapped=bool(swapped),
                sandbox=sandbox,
                out_dir=out_dir,
            )
        )
    return jobs


def _player(spec: str, seed: int, sandbox: bool):
    if sandbox:
        return SandboxedPlayer(spec, seed)
    return InProcessPlayer(load_policy_class(spec)(seed))


def play(job: Job) -> dict:
    team_specs = (job.bot_b, job.bot_a) if job.swapped else (job.bot_a, job.bot_b)
    # Each side gets its own bot seed so a bot playing itself doesn't mirror its moves.
    players = [_player(spec, 2 * job.seed + t, job.sandbox) for t, spec in enumerate(team_specs)]
    try:
        r = run_match(
            load_map(job.map_name), players[0], players[1], seed=job.seed, record=bool(job.out_dir)
        )
    finally:
        for p in players:
            if hasattr(p, "close"):
                p.close()
    side = 1 if job.swapped else 0
    if r.winner is None:
        result = "draw"
    else:
        result = "win" if r.winner == side else "loss"
    ours, theirs = r.score if side == 0 else r.score[::-1]
    summary = {
        "id": job.match_id,
        "a": label(job.bot_a),
        "b": label(job.bot_b),
        "map": job.map_name,
        "seed": job.seed,
        "side": "AB"[side],
        "result": result,
        "score": [ours, theirs],
        "ticks": r.ticks,
        "forfeit": None if r.forfeit is None else ("a" if r.forfeit == side else "b"),
        "crashes": [r.crashes[side], r.crashes[1 - side]],
        "errors": [getattr(p, "errors", [])[:5] for p in (players[side], players[1 - side])],
    }
    if job.out_dir:
        out = Path(job.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        r.recorder.save(out / f"{job.match_id}.replay.npz", out / f"{job.match_id}.events.jsonl")
        # [seconds, ours, theirs] every 10 s
        summary["coverage"] = [
            [t // 10, *((a, b) if side == 0 else (b, a))] for t, a, b in r.recorder.coverage()
        ]
    return summary


def run_jobs(jobs: list[Job], workers: int = 1) -> list[dict]:
    if workers <= 1:
        return [play(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(play, jobs, chunksize=1)


def tally(results: list[dict]) -> dict:
    n = len(results)
    wins = sum(r["result"] == "win" for r in results)
    draws = sum(r["result"] == "draw" for r in results)
    rate = (wins + 0.5 * draws) / n if n else 0.0
    se = math.sqrt(rate * (1 - rate) / n) if n else 0.0
    return {
        "matches": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
        "win_rate": rate,
        "stderr": se,
    }


def write_summary(path: str | Path, results: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"tally": tally(results), "matches": results}, indent=1))
