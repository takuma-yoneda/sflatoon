"""Run a bot in its own subprocess with time and memory limits.

Limits (design doc section 3, values from ``Rules``/``Limits``):

- Setup: constructing ``Policy(seed)`` may use 1 s of CPU.
- Per tick: 50 ms CPU hard limit. Going over, or not answering within the wall
  clock grace, kills and restarts the subprocess and counts as a crash.
- Average: 5 ms CPU per tick. A bot over its running budget
  (5 ms x ticks so far + a burst allowance) gets a no-op that tick; not a crash.
- An exception, invalid pickle, or dead process is a crash. Every crash makes
  that tick a no-op for the team; the bot restarts with a fresh ``Policy(seed)``.
- Memory is capped with RLIMIT_AS. Network is disabled best-effort: sockets
  raise inside the worker, and on Linux the worker also tries to enter its own
  empty network namespace.

Time limits use CPU time measured inside the worker, so results are
reproducible across machines; the wall clock only catches hung processes.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass

from .loader import resolve
from .worker import read_msg, write_msg


@dataclass(frozen=True)
class Limits:
    setup_cpu: float = 1.0
    tick_cpu: float = 0.050
    avg_cpu: float = 0.005
    burst_cpu: float = 0.5  # head start on the average budget
    wall_grace: float = 1.0  # extra wall-clock seconds before a silent bot is killed
    memory_bytes: int = 1 << 30


DEFAULT_LIMITS = Limits()


def _preexec(memory_bytes: int):
    def run():
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if sys.platform.startswith("linux"):
            try:
                import ctypes

                libc = ctypes.CDLL(None, use_errno=True)
                clone_newuser, clone_newnet = 0x10000000, 0x40000000
                libc.unshare(clone_newuser | clone_newnet)
            except Exception:
                pass  # namespaces unavailable; the in-process socket block still applies

    return run


class SandboxedPlayer:
    """One bot process. ``act`` returns the bot's actions, or None for a no-op tick."""

    def __init__(self, spec: str, seed: int = 0, limits: Limits = DEFAULT_LIMITS, log=None):
        self.spec = resolve(spec)
        self.seed = seed
        self.limits = limits
        self.log = log  # optional file object for the bot's stderr
        self.crashes = 0
        self.timeouts = 0
        self.slow_ticks = 0
        self.ticks = 0
        self.cpu_used = 0.0
        self.errors: list[str] = []
        self.proc = None
        self._start()

    # ------------------------------------------------------------------ process

    def _start(self) -> None:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "inkarena.arena.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log if self.log is not None else subprocess.DEVNULL,
            env=env,
            preexec_fn=_preexec(self.limits.memory_bytes),
        )
        reply = self._request(("init", self.spec, self.seed), self.limits.setup_cpu)
        if reply is None:
            return
        status, payload, cpu = reply
        if status != "ok":
            self._crash(f"setup: {payload}")
        elif cpu > self.limits.setup_cpu:
            self._crash(f"setup took {cpu:.3f}s CPU (limit {self.limits.setup_cpu}s)")

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def _crash(self, why: str) -> None:
        self.crashes += 1
        self.errors.append(f"tick {self.ticks}: {why}")
        self._kill()

    def _request(self, msg, cpu_limit: float):
        """Send a request and wait for the reply; on failure, record a crash."""
        try:
            write_msg(self.proc.stdin, msg)
        except (BrokenPipeError, OSError):
            self._crash("process died")
            return None
        out = self.proc.stdout
        deadline = time.monotonic() + cpu_limit + self.limits.wall_grace
        if not _wait_readable(out, deadline):
            self.timeouts += 1
            self._crash("no reply before the wall-clock deadline")
            return None
        try:
            reply = read_msg(out)
        except Exception as e:
            self._crash(f"bad reply: {e}")
            return None
        if reply is None:
            self._crash("process died")
            return None
        return reply

    # ---------------------------------------------------------------------- API

    def act(self, obs):
        if self.proc is None:
            self._start()  # restart after a crash; fresh Policy(seed)
            if self.proc is None:
                self.ticks += 1
                return None
        self.ticks += 1
        reply = self._request(("act", obs), self.limits.tick_cpu)
        if reply is None:
            return None
        status, payload, cpu = reply
        self.cpu_used += cpu
        if status != "ok":
            self._crash(payload)
            return None
        if cpu > self.limits.tick_cpu:
            self.timeouts += 1
            self._crash(
                f"tick took {cpu * 1000:.1f} ms CPU (limit {self.limits.tick_cpu * 1000:.0f} ms)"
            )
            return None
        if self.cpu_used > self.limits.avg_cpu * self.ticks + self.limits.burst_cpu:
            self.slow_ticks += 1
            return None
        return payload

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=1)
            except Exception:
                pass
        self._kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _wait_readable(f, deadline: float) -> bool:
    # The worker sends exactly one reply per request, so nothing is ever left
    # buffered in ``f`` between requests and select() on the pipe is accurate.
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        r, _, _ = select.select([f], [], [], remaining)
        if r:
            return True
