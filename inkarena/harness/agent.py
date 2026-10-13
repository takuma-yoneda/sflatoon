"""One revision: a fresh-context agent loop over the workspace (design doc sections 6 and 7).

The loop is append-only: every assistant response goes back into ``messages`` unchanged,
thinking blocks included, and tool results follow as a user message. Each request and
tool result is logged to a JSONL transcript (images by path, not data).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Protocol

from .budget import Budget
from .tools import TOOLS, Shell, load_image

BASH_TIMEOUT = 300.0


def system_prompt() -> str:
    return resources.files("inkarena.harness").joinpath("prompts/system.md").read_text()


@dataclass
class Reply:
    content: list[dict]  # content blocks as plain dicts, ready to send back
    stop_reason: str
    usage: dict = field(default_factory=dict)


class Model(Protocol):
    name: str

    def step(self, system: str, messages: list, tools: list) -> Reply: ...


class AnthropicModel:
    """Claude through the Messages API, streaming, with adaptive thinking and prompt caching."""

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 64000):
        import anthropic  # optional dependency: pip install -e ".[harness]"

        self.client = anthropic.Anthropic()
        self.name, self.effort, self.max_tokens = model, effort, max_tokens

    def step(self, system: str, messages: list, tools: list) -> Reply:
        with self.client.messages.stream(
            model=self.name,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        ) as stream:
            msg = stream.get_final_message()
        content = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
        return Reply(content, msg.stop_reason, msg.usage.model_dump(mode="json", exclude_none=True))


@dataclass
class RevisionLog:
    ended: str  # "submitted", "turns", "wall clock", "refusal", "error: ..."
    turns: int
    matches: int
    seconds: float
    usage: dict


def _log(f, **record) -> None:
    f.write(json.dumps({"t": round(time.time(), 3), **record}) + "\n")
    f.flush()


def _redact(blocks: list) -> list:
    """Tool results for the transcript: image data replaced by its path."""
    out = []
    for b in blocks:
        b = dict(b)
        if isinstance(b.get("content"), list):
            b["content"] = [
                {"type": "image", "path": c.get("_path")} if c.get("type") == "image" else c
                for c in b["content"]
            ]
        out.append(b)
    return out


def run_revision(
    model: Model,
    ws: Path,
    first_message: str,
    budget_file: Path,
    transcript: Path,
    limits: dict,
    shell_prefix: list[str] | None = None,
) -> RevisionLog:
    budget = Budget.start(budget_file, **limits)
    shell = Shell(ws, budget_file, shell_prefix)
    system = system_prompt()
    messages: list = [{"role": "user", "content": first_message}]
    usage: dict[str, int] = {}
    ended = None
    with open(transcript, "w") as f:
        _log(f, kind="start", model=model.name, system=system, first_message=first_message)
        while True:
            budget = Budget.load(budget_file)
            if (why := budget.exhausted) is not None:
                ended = why
                break
            try:
                reply = model.step(system, messages, TOOLS)
            except Exception as e:  # API errors end the revision; the bot is still submitted
                ended = f"error: {type(e).__name__}: {e}"
                _log(f, kind="error", error=ended)
                break
            budget = Budget.load(budget_file)
            budget.turns += 1
            budget.save(budget_file)
            for k, v in reply.usage.items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
            _log(
                f,
                kind="assistant",
                content=reply.content,
                stop=reply.stop_reason,
                usage=reply.usage,
            )
            messages.append({"role": "assistant", "content": reply.content})

            calls = [b for b in reply.content if b.get("type") == "tool_use"]
            if reply.stop_reason == "refusal":
                ended = "refusal"
                break
            if not calls:
                if reply.stop_reason == "max_tokens":
                    messages.append({"role": "user", "content": "Your reply was cut off. Go on."})
                    continue
                ended = "submitted"
                break
            results = [_run_tool(c, ws, shell, budget_file) for c in calls]
            _log(f, kind="tool_results", content=_redact(results))
            for r in results:
                for c in r["content"]:
                    c.pop("_path", None)
            messages.append({"role": "user", "content": results})
        budget = Budget.load(budget_file)
        _log(f, kind="end", ended=ended, turns=budget.turns, matches=budget.matches, usage=usage)
    return RevisionLog(ended, budget.turns, budget.matches, round(budget.elapsed, 1), usage)


def _run_tool(call: dict, ws: Path, shell: Shell, budget_file: Path) -> dict:
    name, args = call.get("name"), call.get("input") or {}
    content: list[dict] = []
    error = False
    if name == "bash":
        if args.get("restart"):
            text = "Shell restarted. (Every command already runs in a fresh shell.)"
        elif isinstance(args.get("command"), str):
            left = Budget.load(budget_file).seconds_left
            text, error = shell.run(args["command"], timeout=max(1.0, min(BASH_TIMEOUT, left)))
        else:
            text, error = "bash needs a 'command' string", True
        content.append({"type": "text", "text": text})
    elif name == "view_image":
        block, err = load_image(ws, args.get("path"))
        if block is None:
            content.append({"type": "text", "text": err})
            error = True
        else:
            content.append({**block, "_path": args["path"]})
    else:
        content.append({"type": "text", "text": f"unknown tool {name!r}"})
        error = True
    content.append({"type": "text", "text": Budget.load(budget_file).echo()})
    result = {"type": "tool_result", "tool_use_id": call["id"], "content": content}
    if error:
        result["is_error"] = True
    return result
