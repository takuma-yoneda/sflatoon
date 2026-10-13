You are competing in Ink Arena, a top-down 2D territory shooter in the style of Splatoon. You don't play it yourself. You write the bot that plays it: `bot/policy.py` in your workspace, a Python `Policy` class that controls a team of two agents for every tick of a 60 second match.

A match is won by having more of the map painted in your team's color at the end. Agents paint by shooting, refill ink by swimming in their own color, and splat enemies by hitting them three times. The full rules are in `docs/rules.md` and the bot API in `docs/api.md`. Read both before changing the bot.

## This revision

Each revision is one session with a fixed budget of agent turns, local matches and wall-clock time. The remaining budget is shown at the end of every tool result. When you end your turn without calling a tool, or when the budget runs out, the current contents of `bot/` are submitted.

After you submit, your bot plays an official evaluation of at least 50 matches against the house baseline and against your previous version, on every map from both sides. Those results and their replays appear in `replays/round_N/` at the start of your next revision.

Each revision starts with no memory of earlier ones. `NOTES.md` is the only thing that carries over, so keep it up to date with what you tried, what worked, and what you plan next.

## What to submit

- `bot/policy.py`, plus any modules you add under `bot/`. It must load and run within the time limits in `docs/rules.md`. A bot that fails to load is replaced by your previous version.
- `ANALYSIS.md`, rewritten each revision. Say what you changed and why. Back every claim about how a bot behaved with a match ID and a tick you looked at, such as `replays/local/m004, tick 312`.

## Your workspace

- `bot/`: your bot. `lib/helpers.py` is the helper library (read-only); import it in your bot as `from inkarena.bots import helpers`.
- `tools/arena.py` plays local matches: `python tools/arena.py --a bot/ --b baseline -n 20 --out replays/local/`. Opponents can be `baseline`, `random`, `noop`, or your own past versions in `opponents/`. Every match counts against your budget.
- `tools/replay.py` queries and renders replays: `python tools/replay.py events replays/local/m000.replay.npz --type splat`, or `render ... --tick 300 -o frame.png`. It also works as a Python library. Use `view_image` to look at a rendered frame or heatmap.
- `replays/round_N/` holds official results, with a `summary.json` per opponent.

## Tools

`bash` runs one command at the workspace root, in a fresh shell each time, so `cd` does not carry over. Long output is cut to about 10,000 characters, so prefer `grep`, `head` and short scripts over printing whole files or replays. `view_image` shows you an image file.

Ground your changes in evidence. Look at replays, not just win rates, before you decide what to change, and check that a change helps before you keep it.
