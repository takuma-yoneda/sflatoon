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

BAR = 64  # height of the top bar in pixels

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
SS = 3  # supersampling factor for agents and projectiles


def _dark(c, k=0.6):
    return tuple(int(v * k) for v in c)


def _teardrop(x, y, f, r, nose, to_px, n=28):
    """Polygon (pixels) of a disc of radius r whose front tapers to a point nose*r ahead."""
    ux, uy = math.cos(f), math.sin(f)
    tip = (x + ux * r * nose, y + uy * r * nose)
    half = math.acos(1 / nose)  # tangent points seen from the tip
    pts = [to_px(*tip)]
    for i in range(n + 1):
        a = f + half + (2 * math.pi - 2 * half) * i / n
        pts.append(to_px(x + r * math.cos(a), y + r * math.sin(a)))
    return pts


def _swim_ripple(fr, d, x, y, f, team, tick):
    """A dark puddle in the ink with an expanding ripple ring."""
    S = fr.s * SS
    col = TEAM[team]["agent"]
    cx, cy = fr.pp(x, y)
    phase = (tick / 8) % 1.0  # one ripple every 0.8 s
    r = (0.3 + 0.35 * phase) * S
    d.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=col + (int(200 * (1 - phase)),),
        width=int(0.06 * S),
    )
    fr._body(d, x, y, f, 0.24, _dark(col, 0.8) + (170,), nose=1.6)


def _swim_fin(fr, d, x, y, f, team, tick):
    """A small shark-fin wedge cutting through the ink, with a V wake behind it."""
    S = fr.s * SS
    col = TEAM[team]["agent"]
    ux, uy = math.cos(f), math.sin(f)
    vx, vy = -uy, ux
    for side in (-1, 1):
        a = fr.pp(x - ux * 0.1, y - uy * 0.1)
        b = fr.pp(x - ux * 0.55 + side * vx * 0.35, y - uy * 0.55 + side * vy * 0.35)
        d.line([a, b], fill=(255, 255, 255, 150), width=int(0.05 * S))
    tip = fr.pp(x + ux * 0.32, y + uy * 0.32)
    left = fr.pp(x - ux * 0.2 + vx * 0.16, y - uy * 0.2 + vy * 0.16)
    rr = fr.pp(x - ux * 0.2 - vx * 0.16, y - uy * 0.2 - vy * 0.16)
    d.polygon([tip, left, rr], fill=_dark(col, 0.7) + (235,))


def _swim_eyes(fr, d, x, y, f, team, tick):
    """Only two eyes peek out of the ink, looking where the agent faces."""
    col = TEAM[team]["agent"]
    ux, uy = math.cos(f), math.sin(f)
    vx, vy = -uy, ux
    fr._disc(d, x, y, 0.3, _dark(col, 0.85) + (110,))
    for side in (-1, 1):
        ex, ey = x + ux * 0.08 + side * vx * 0.14, y + uy * 0.08 + side * vy * 0.14
        fr._disc(d, ex, ey, 0.1, (255, 255, 255, 240))
        fr._disc(d, ex + ux * 0.04, ey + uy * 0.04, 0.05, (20, 20, 30, 255))


def _swim_shadow(fr, d, x, y, f, team, tick):
    """The body as a darker, translucent shadow under the ink surface."""
    col = TEAM[team]["agent"]
    fr._body(d, x, y, f, 0.36, _dark(col, 0.55) + (120,))


SWIM_STYLES = {
    "ripple": _swim_ripple,
    "fin": _swim_fin,
    "eyes": _swim_eyes,
    "shadow": _swim_shadow,
}


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

    def __init__(self, replay, scale: int, swim_style: str = "ripple"):
        self.r = replay
        self.m = replay.map
        self.s = scale
        self.w = self.m.width * scale
        self.h = self.m.height * scale + BAR
        self.font = _font(max(11, int(scale * 0.45)))
        self.font_ss = _font(max(11, int(scale * 0.45)) * SS)
        self.big = _font(20)
        self.swim_style = swim_style

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
            for k in range(len(agents)):
                x, y, f, *_, alive, _inv = agents[k]
                if alive:
                    team = k // (len(agents) // 2)
                    poly = [self.px(px_, py_) for px_, py_ in _cone(self.m, x, y, f)]
                    od.polygon(poly, fill=TEAM[team]["agent"] + (38,))
                    od.line(poly + [poly[0]], fill=TEAM[team]["agent"] + (90,), width=1)
            img = Image.alpha_composite(img, overlay)

        # Agents and projectiles are drawn at SS x resolution and downsampled for anti-aliasing.
        big = Image.new("RGBA", (self.w * SS, self.h * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(big)
        m = len(agents) // 2
        for x, y, dx, dy, owner in projectiles:
            col = TEAM[int(owner) // m]["agent"]
            tx, ty = x - dx * 0.8, y - dy * 0.8
            d.line([self.pp(tx, ty), self.pp(x, y)], fill=col + (130,), width=int(0.16 * s * SS))
            self._disc(d, x, y, 0.2, col + (255,))
        for k in range(len(agents)):
            self._agent(d, k, m, agents[k], int(respawn_in[k]), tick)
        overlay = big.resize((self.w, self.h), Image.LANCZOS)
        img = Image.alpha_composite(img, overlay)
        self._bar(ImageDraw.Draw(img), score, tick)
        return img.convert("RGB")

    # Supersampled drawing helpers (cell units in, SS-scaled pixels out).
    def pp(self, x, y):
        return x * self.s * SS, (y * self.s + BAR) * SS

    def _disc(self, d, x, y, r, fill):
        cx, cy = self.pp(x, y)
        r *= self.s * SS
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    def _body(self, d, x, y, f, r, fill, nose=1.75):
        """Teardrop: a disc of radius r with a pointed nose toward facing f."""
        d.polygon(_teardrop(x, y, f, r, nose, self.pp), fill=fill)

    def _agent(self, d, k, m, row, respawn_in, tick):
        S = self.s * SS
        x, y, f, hp, ink, sub, alive, inv = row
        team = k // m
        col = TEAM[team]["agent"]
        cx, cy = self.pp(x, y)
        if not alive:
            r = 0.3 * S
            w = int(0.12 * S)
            d.line([cx - r, cy - r, cx + r, cy + r], fill=col + (200,), width=w)
            d.line([cx - r, cy + r, cx + r, cy - r], fill=col + (200,), width=w)
            if respawn_in > 0:
                d.text(
                    (cx + r + 4, cy - r - 4),
                    f"{respawn_in / 10:.1f}s",
                    fill=(20, 20, 20, 220),
                    font=self.font_ss,
                )
            return
        alpha = 255
        if sub:
            SWIM_STYLES[self.swim_style](self, d, x, y, f, team, tick)
        else:
            if inv:
                alpha = 120
            self._body(d, x, y, f, 0.42, col + (alpha,))
        # HP pips above, ink bar below.
        pip_alpha = 150 if sub else alpha
        for i in range(3):
            px_ = cx + (i - 1) * 0.3 * S - 0.1 * S
            fill = (230, 50, 50, pip_alpha) if i < hp else (120, 120, 120, pip_alpha // 2)
            d.rounded_rectangle(
                [px_, cy - 0.8 * S, px_ + 0.2 * S, cy - 0.66 * S], radius=0.05 * S, fill=fill
            )
        w = 0.8 * S
        y0, y1 = cy + 0.62 * S, cy + 0.72 * S
        d.rounded_rectangle(
            [cx - w / 2, y0, cx + w / 2, y1], radius=0.05 * S, fill=(40, 40, 40, pip_alpha // 2)
        )
        if ink > 0:
            d.rounded_rectangle(
                [cx - w / 2, y0, cx - w / 2 + w * ink / 100, y1],
                radius=0.05 * S,
                fill=col + (pip_alpha,),
            )
        label = f"{TEAM[team]['name']}{k % m}"
        d.text(
            (cx + 0.55 * S, cy - 0.6 * S), label, fill=(15, 15, 15, pip_alpha), font=self.font_ss
        )

    def _bar(self, d, score, tick):
        a, b = int(score[0]), int(score[1])
        total = int(self.m.scorable.sum())
        x0, x1, y0, y1 = int(self.w * 0.22), int(self.w * 0.78), 36, BAR - 10
        span = x1 - x0
        d.rectangle([x0, y0, x1, y1], fill=FLOOR)
        wa, wb = span * a / total, span * b / total
        d.rectangle([x0, y0, x0 + wa, y1], fill=TEAM[0]["agent"])
        d.rectangle([x1 - wb, y0, x1, y1], fill=TEAM[1]["agent"])
        d.line([(x0 + span / 2, y0 - 4), (x0 + span / 2, y1 + 4)], fill=(255, 255, 255), width=2)
        d.text(
            (16, (y0 + y1) / 2),
            f"A {a}  {100 * a / total:.0f}%",
            fill=TEAM[0]["ink"],
            font=self.big,
            anchor="lm",
        )
        d.text(
            (self.w - 16, (y0 + y1) / 2),
            f"{100 * b / total:.0f}%  {b} B",
            fill=TEAM[1]["ink"],
            font=self.big,
            anchor="rm",
        )
        secs = tick / 10
        d.text(
            (self.w / 2, 18),
            f"{int(secs // 60)}:{secs % 60:04.1f}",
            fill=(235, 235, 235),
            font=self.big,
            anchor="mm",
        )


def _projectiles(replay, i: int) -> list[tuple]:
    owners = replay.proj_owner[i]
    return [(*replay.proj[i, j], owners[j]) for j in range(len(owners)) if owners[j] >= 0]


def render_state(
    replay, tick: int, vision: bool = True, scale: int = 32, swim_style: str = "ripple"
) -> Image.Image:
    fr = _Frame(replay, scale, swim_style)
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
    for k in range(len(a1)):
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
    scale: int = 32,
    vision: bool = True,
    swim_style: str = "ripple",
) -> Path:
    """Render ticks t0..t1 to MP4. ``speed`` 2.0 plays twice as fast as real time (10 ticks/s)."""
    t1 = replay.ticks if t1 is None else min(t1, replay.ticks)
    fr = _Frame(replay, scale, swim_style)
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


def render_heatmap(replay, kind: str, team: str | None = None, scale: int = 32) -> Image.Image:
    m = replay.map
    counts = np.zeros(m.walls.shape, dtype=np.float64)
    size = replay.team_size
    agents = (
        range(2 * size) if team is None else range(size) if team == "A" else range(size, 2 * size)
    )
    if kind == "position":
        for k in agents:
            alive = replay.agents[:, k, 6] > 0
            xs = replay.agents[alive, k, 0].astype(int)
            ys = replay.agents[alive, k, 1].astype(int)
            np.add.at(counts, (ys, xs), 1)
    elif kind == "deaths":
        names = {replay.names[k] for k in agents}
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
