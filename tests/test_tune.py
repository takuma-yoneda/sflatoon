import subprocess
import sys
from pathlib import Path

from inkarena.arena.runner import make_jobs, run_jobs

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_self_play_varies_with_seed():
    # Each side gets its own bot seed, so baseline vs itself doesn't mirror into a draw.
    results = run_jobs(make_jobs("baseline", "baseline", 4, ["duel"], sandbox=False), workers=2)
    assert any(r["result"] != "draw" for r in results), results


def test_tune_reports_every_variant(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "tools/tune.py",
            "--only",
            "default,quick_respawn",
            "--seeds",
            "1",
            "--maps",
            "duel",
            "--workers",
            "2",
            "--replays",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "default" in out and "quick_respawn" in out
    assert (tmp_path / "default" / "duel.replay.npz").exists()
