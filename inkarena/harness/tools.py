"""The model's two tools: bash in the workspace, and view_image for rendered frames.

Commands are untrusted model output. ``Shell`` runs each one in a fresh ``bash -c`` at the
workspace root with a minimal environment (no API keys) and a timeout. That is not
isolation: run the pilot inside a container or VM, or pass ``prefix`` to run commands
through one (for example ``["docker", "exec", "-i", "-w", "/ws", "pilot"]``).
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

MAX_OUTPUT = 10_000
MAX_IMAGE_BYTES = 4_000_000
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}
VIEW_IMAGE_TOOL = {
    "name": "view_image",
    "description": (
        "Show an image file from the workspace, such as a frame rendered with "
        "tools/replay.py render or a heatmap. Path is relative to the workspace root."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "PNG, JPEG or GIF file"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}
TOOLS = [BASH_TOOL, VIEW_IMAGE_TOOL]


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    """Keep the head and tail of long output, with a note on what was cut."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 6 // 10], text[-(limit * 3 // 10) :]
    return f"{head}\n\n[... {len(text) - len(head) - len(tail)} characters cut ...]\n\n{tail}"


class Shell:
    def __init__(self, ws: Path, budget_file: Path, prefix: list[str] | None = None):
        self.ws = Path(ws).resolve()
        self.prefix = prefix or []
        home = self.ws / ".home"
        home.mkdir(exist_ok=True)
        self.env = {
            # The interpreter running the harness comes first, so `python` has inkarena.
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")]
            ),
            "LANG": "C.UTF-8",
            "HOME": str(home),
            "MPLCONFIGDIR": str(home),
            "INKARENA_BUDGET": str(budget_file),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if os.environ.get("VIRTUAL_ENV"):
            self.env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

    def run(self, command: str, timeout: float) -> tuple[str, bool]:
        """Combined stdout and stderr (truncated), and whether the command failed."""
        try:
            p = subprocess.run(
                [*self.prefix, "bash", "-c", command],
                cwd=self.ws,
                env=self.env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (e.stderr or "")
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            return truncate(f"{out}\n[timed out after {timeout:.0f} s]"), True
        out = p.stdout + p.stderr
        if p.returncode != 0:
            out += f"\n[exit code {p.returncode}]"
        return truncate(out or "[no output]"), p.returncode != 0


def load_image(ws: Path, path) -> tuple[dict | None, str | None]:
    """An image content block for a workspace file, or an error message."""
    if not isinstance(path, str) or not path:
        return None, "view_image needs a 'path' string"
    root = Path(ws).resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        return None, f"{path} is outside the workspace"
    if not p.is_file():
        return None, f"no file at {path}"
    media = IMAGE_TYPES.get(p.suffix.lower())
    if media is None:
        return None, f"{path} is not a PNG, JPEG or GIF"
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        return None, f"{path} is {len(data)} bytes; the limit is {MAX_IMAGE_BYTES}"
    source = {"type": "base64", "media_type": media, "data": base64.b64encode(data).decode()}
    return {"type": "image", "source": source}, None
