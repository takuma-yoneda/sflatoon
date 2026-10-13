"""Draw replays: PNG frames, heatmaps and MP4 video (design doc section 8).

The renderer only reads replay arrays; it never touches the simulator.
Viewers see everything: vision cones (clipped by walls), hidden agents drawn
faded, and a coverage bar on top.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..sim.geometry import first_wall_t

BAR = 44  # height of the top bar in pixels

FLOOR = (238, 234, 226)
WALL = (48, 52, 62)
TEAM = {
    0: {"agent": (226, 86, 0), "ink": (255, 192, 132), "base": (238, 150, 80), "name": "A"},
    1: {"agent": (72, 44, 214), "ink": (182, 168, 255), "base": (140, 120, 235), "name": "B"},
}
SHOT_SPEED = 1.5
VIEW_RANGE = 12.0
VIEW_HALF = math.radians(60)
CONE_RAYS = 33


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _palette(m) -> np.ndarray:
    """Colour lookup for ink values -1, 0, +1 (index value + 1), as float."""
    return np.array([TEAM[1]["ink"], FLOOR, TEAM[0]["ink"]], dtype=np.float32)


def cell_colors(m, ink: np.ndarray) -> np.ndarray:
    """float32[H, W, 3] colour per cell for an ink grid."""
    rgb = _palette(m)[ink.astype(np.int64) + 1]
    rgb[m.base == 1] = TEAM[0]["base"]
    rgb[m.base == -1] = TEAM[1]["base"]
    rgb[m.walls] = WALL
    return rgb


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def _with_grid(img: np.ndarray, scale: int) -> np.ndarray:
    """Faint cell grid so distances are readable."""
    out = img.copy()
    out[::scale, :, :] *= 0.93
    out[:, ::scale, :] *= 0.93
    return out


def _cone(m, x: float, y: float, facing: float) -> list[tuple[float, float]]:
    """View cone polygon (cell units) clipped by walls with a fan of rays."""
    pts = [(x, y)]
    for i in range(CONE_RAYS):
        a = facing - VIEW_HALF + 2 * VIEW_HALF * i / (CONE_RAYS - 1)
        ex, ey = x + VIEW_RANGE * math.cos(a), y + VIEW_RANGE * math.sin(a)
        t = first_wall_t(m.walls, x, y, ex, ey)
        t = 1.0 if t is None else t
        pts.append((x + (ex - x) * t, y + (ey - y) * t))
    return pts


class _Frame:
    """Draws one frame from interpolated agent rows and projectiles."""

    def __init__(self, replay, scale: int):
        self.r = replay
        self.m = replay.map
        self.s = scale
        self.w = self.m.width * scale
        self.h = self.m.height * scale + BAR
        self.font = _font(max(11, scale - 2))
        self.big = _font(max(16, int(scale * 1.25)))

    def px(self, x, y):
        return x * self.s, y * self.s + BAR

    def draw(
        self,
        cells: np.ndarray,
        agents: np.ndarray,
        respawn_in,
        projectiles,
        score,
        tick: float,
        vision: bool,
    ) -> Image.Image:
        s = self.s
        board = _with_grid(_upscale(cells, s), s)
        canvas = np.empty((self.h, self.w, 3), dtype=np.float32)
        canvas[:BAR] = (28, 30, 36)
        canvas[BAR:] = board
        img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")

        if vision:
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            for k in range(6):
                x, y, f, *_, alive, _inv = agents[k]
                if alive:
                    team = k // 3
                    poly = [self.px(px_, py_) for px_, py_ in _cone(self.m, x, y, f)]
                    od.polygon(poly, fill=TEAM[team]["agent"] + (38,))
                    od.line(poly + [poly[0]], fill=TEAM[team]["agent"] + (90,), width=1)
            img = Image.alpha_composite(img, overlay)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for x, y, dx, dy, owner in projectiles:
            col = TEAM[int(owner) // 3]["agent"]
            tx, ty = x - dx * 0.8, y - dy * 0.8
            d.line([self.px(tx, ty), self.px(x, y)], fill=col + (140,), width=max(2, s // 5))
            r = 0.22 * s
            cx, cy = self.px(x, y)
            d.ellipse(
                [cx - r, cy - r, cx + r, cy + r], fill=col + (255,), outline=(20, 20, 20, 255)
            )

        for k in range(6):
            self._agent(d, k, agents[k], int(respawn_in[k]))
        img = Image.alpha_composite(img, overlay)
        self._bar(ImageDraw.Draw(img), score, tick)
        return img.convert("RGB")

    def _agent(self, d, k, row, respawn_in):
        s = self.s
        x, y, f, hp, ink, sub, alive, inv = row
        team = k // 3
        col = TEAM[team]["agent"]
        cx, cy = self.px(x, y)
        if not alive:
            r = 0.35 * s
            d.line([cx - r, cy - r, cx + r, cy + r], fill=col + (200,), width=3)
            d.line([cx - r, cy + r, cx + r, cy - r], fill=col + (200,), width=3)
            if respawn_in > 0:
                d.text(
                    (cx + r + 2, cy - r - 2),
                    f"{respawn_in / 10:.1f}s",
                    fill=(20, 20, 20, 220),
                    font=self.font,
                )
            return
        alpha = 110 if (inv or sub) else 255
        r = 0.5 * s
        fx, fy = cx + math.cos(f) * 0.95 * s, cy + math.sin(f) * 0.95 * s
        d.line([cx, cy, fx, fy], fill=(20, 20, 20, alpha), width=max(3, s // 5))
        if sub:
            d.ellipse(
                [cx - r, cy - r * 0.55, cx + r, cy + r * 0.55],
                fill=col + (alpha,),
                outline=(255, 255, 255, 230),
                width=2,
            )
        else:
            d.ellipse([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1], fill=(20, 20, 20, alpha))
            d.ellipse(
                [cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1],
                fill=col + (alpha,),
                outline=(255, 255, 255, alpha),
                width=2,
            )
        # HP pips above, ink bar below.
        for i in range(3):
            px_ = cx - 0.4 * s + i * 0.4 * s
            fill = (230, 40, 40, alpha) if i < hp else (120, 120, 120, alpha // 2)
            d.rectangle([px_, cy - 0.85 * s, px_ + 0.3 * s, cy - 0.65 * s], fill=fill)
        w = 0.9 * s
        d.rectangle([cx - w / 2, cy + 0.6 * s, cx + w / 2, cy + 0.75 * s], fill=(40, 40, 40, alpha))
        d.rectangle(
            [cx - w / 2, cy + 0.6 * s, cx - w / 2 + w * ink / 100, cy + 0.75 * s],
            fill=col + (alpha,),
        )
        label = f"{TEAM[team]['name']}{k % 3}"
        d.text((cx + r + 1, cy - r - 1), label, fill=(15, 15, 15, alpha), font=self.font)

    def _bar(self, d, score, tick):
        a, b = int(score[0]), int(score[1])
        total = int(self.m.scorable.sum())
        x0, x1, y0, y1 = 200, self.w - 200, 12, BAR - 12
        span = x1 - x0
        d.rectangle([x0, y0, x1, y1], fill=FLOOR)
        wa, wb = span * a / total, span * b / total
        d.rectangle([x0, y0, x0 + wa, y1], fill=TEAM[0]["agent"])
        d.rectangle([x1 - wb, y0, x1, y1], fill=TEAM[1]["agent"])
        d.line([(x0 + span / 2, y0 - 4), (x0 + span / 2, y1 + 4)], fill=(255, 255, 255), width=2)
        d.text(
            (16, BAR / 2),
            f"A {a}  {100 * a / total:.0f}%",
            fill=TEAM[0]["ink"],
            font=self.big,
            anchor="lm",
        )
        d.text(
            (self.w - 16, BAR / 2),
            f"{100 * b / total:.0f}%  {b} B",
            fill=TEAM[1]["ink"],
            font=self.big,
            anchor="rm",
        )
        secs = tick / 10
        d.text(
            (self.w / 2, BAR / 2),
            f"{int(secs // 60)}:{secs % 60:04.1f}",
            fill=(20, 20, 20),
            font=self.big,
            anchor="mm",
        )


def _projectiles(replay, i: int) -> list[tuple]:
    owners = replay.proj_owner[i]
    return [(*replay.proj[i, j], owners[j]) for j in range(len(owners)) if owners[j] >= 0]


def render_state(replay, tick: int, vision: bool = True, scale: int = 16) -> Image.Image:
    fr = _Frame(replay, scale)
    cells = cell_colors(replay.map, replay.ink_at(tick))
    return fr.draw(
        cells,
        replay.agents[tick],
        replay.respawn_in[tick],
        _projectiles(replay, tick),
        replay.score[tick],
        tick,
        vision,
    )


def _lerp_agents(a0: np.ndarray, a1: np.ndarray, t: float) -> np.ndarray:
    out = a1.copy()
    for k in range(6):
        if a0[k, 6] and a1[k, 6] and math.dist(a0[k, :2], a1[k, :2]) < 2.0:
            out[k, :2] = a0[k, :2] + (a1[k, :2] - a0[k, :2]) * t
            df = (a1[k, 2] - a0[k, 2] + math.pi) % (2 * math.pi) - math.pi
            out[k, 2] = a0[k, 2] + df * t
            out[k, 4] = a0[k, 4] + (a1[k, 4] - a0[k, 4]) * t
        elif t < 1.0:
            out[k] = a0[k] if t < 0.5 else a1[k]
    return out


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("MP4 output needs ffmpeg on PATH or `pip install imageio-ffmpeg`") from e


def render_video(
    replay,
    out_mp4,
    fps: int = 60,
    speed: float = 1.0,
    t0: int = 0,
    t1: int | None = None,
    scale: int = 16,
    vision: bool = True,
) -> Path:
    """Render ticks t0..t1 to MP4. ``speed`` 2.0 plays twice as fast as real time (10 ticks/s)."""
    t1 = replay.ticks if t1 is None else min(t1, replay.ticks)
    fr = _Frame(replay, scale)
    per_tick = max(1, round(fps / (10 * speed)))
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{fr.w}x{fr.h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        prev_cells = cell_colors(replay.map, replay.ink_at(t0))
        for i in range(t0, t1):
            next_cells = cell_colors(replay.map, replay.ink_at(i + 1))
            a0, a1 = replay.agents[i], replay.agents[i + 1]
            projs = _projectiles(replay, i + 1)
            for f in range(per_tick):
                t = (f + 1) / per_tick
                cells = prev_cells + (next_cells - prev_cells) * min(1.0, t * 1.5)
                back = SHOT_SPEED * (1 - t)
                frame_projs = [
                    (x - dx * back, y - dy * back, dx, dy, o) for x, y, dx, dy, o in projs
                ]
                img = fr.draw(
                    cells,
                    _lerp_agents(a0, a1, t),
                    replay.respawn_in[i + 1],
                    frame_projs,
                    replay.score[i + 1],
                    i + t,
                    vision,
                )
                proc.stdin.write(img.tobytes())
            prev_cells = next_cells
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
    return Path(out_mp4)


def render_heatmap(replay, kind: str, team: str | None = None, scale: int = 16) -> Image.Image:
    m = replay.map
    counts = np.zeros(m.walls.shape, dtype=np.float64)
    agents = range(6) if team is None else range(3) if team == "A" else range(3, 6)
    if kind == "position":
        for k in agents:
            alive = replay.agents[:, k, 6] > 0
            xs = replay.agents[alive, k, 0].astype(int)
            ys = replay.agents[alive, k, 1].astype(int)
            np.add.at(counts, (ys, xs), 1)
    elif kind == "deaths":
        names = {f"{'AB'[k // 3]}{k % 3}" for k in agents}
        for e in replay.events("splat"):
            if e["victim"] in names:
                counts[int(e["pos"][1]), int(e["pos"][0])] += 1
    elif kind == "paint":
        d = replay.ink_diff
        if team is not None:
            d = d[d[:, 3] == (1 if team == "A" else -1)]
        np.add.at(counts, (d[:, 1], d[:, 2]), 1)
    else:
        raise ValueError(f"unknown heatmap kind {kind!r}: use position, deaths or paint")
    v = counts / counts.max() if counts.max() > 0 else counts
    v = np.sqrt(v)  # spread out the low end
    lo, hi = np.array(FLOOR, dtype=np.float32), np.array((200, 20, 60), dtype=np.float32)
    rgb = lo + (hi - lo) * v[..., None]
    rgb[m.walls] = WALL
    img = _with_grid(_upscale(rgb.astype(np.float32), scale), scale)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    title = f"{kind} heatmap" + (f", team {team}" if team else "") + f"  max {int(counts.max())}"
    ImageDraw.Draw(out).text((8, 4), title, fill=(255, 255, 255), font=_font(14))
    return out
