"""Storyboards: keyframe stills on one contact sheet, optionally with set-of-mark overlays
(each object's box and id, in a color derived from its id-pass index) so a vision model
can tie what it sees to object ids."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from phasor.core.color import mark_color

PAD = 10
CAPTION = 22
SHEET_BG = (18, 20, 26)
CAPTION_FG = (216, 222, 233)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _hit(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def draw_marks(still: np.ndarray, snapshot: dict[str, Any]) -> np.ndarray:
    """Draw each visible object's bounding box and id onto a still. A tag goes beside its
    box, at whichever spot hides the least of the picture and of the other tags."""
    img = Image.fromarray(still).convert("RGB")
    draw = ImageDraw.Draw(img)
    sx = img.width / snapshot["frame"]["width"]
    sy = img.height / snapshot["frame"]["height"]
    font = _font(max(9, img.height // 32))
    marks = []
    for index, e in enumerate(snapshot["objects"], start=1):
        x0, y0, x1, y1 = e["bbox"]["px"]
        drawn = e["semantic"].get("drawn", 1.0) > 0  # an untraced curve shows nothing
        if (
            e["opacity"] > 0
            and e["on_screen_fraction"] > 0
            and drawn
            and (x1 - x0) * (y1 - y0) >= 1
        ):
            marks.append((index, e["id"], (x0 * sx - 2, y0 * sy - 2, x1 * sx + 2, y1 * sy + 2)))
    # Boxes that span much of the frame (a long diagonal line) are mostly empty; a tag may
    # sit inside them. Everything else is kept clear.
    solid = [b for _, _, b in marks
             if (b[2] - b[0]) * (b[3] - b[1]) < 0.1 * img.width * img.height]  # fmt: skip
    taken: list[tuple[float, float, float, float]] = []
    for index, name, box in sorted(marks, key=lambda m: (m[2][1], m[2][0])):
        color = mark_color(index)
        draw.rectangle(box, outline=color, width=1)
        w, h = draw.textlength(name, font=font) + 6, font.size + 4
        x0, y0, x1, y1 = box
        spots = [(x0, y0 - h), (x0, y1), (x1 - w, y0 - h), (x1 - w, y1), (x1, y0), (x0 - w, y0),
                 (x0, y0 - 2 * h), (x0, y1 + h), (x1 + h, y0 - h), (x0 - w - h, y1)]  # fmt: skip

        def cost(spot, w=w, h=h):
            tag = (spot[0], spot[1], spot[0] + w, spot[1] + h)
            outside = w * h - _hit(tag, (0, 0, img.width, img.height))
            return (2 * outside + sum(_hit(tag, b) for b in solid)
                    + 2 * sum(_hit(tag, t) for t in taken))  # fmt: skip

        tx, ty = min(spots, key=cost)
        taken.append((tx, ty, tx + w, ty + h))
        draw.rectangle((tx, ty, tx + w, ty + h), fill=color)
        draw.text((tx + 3, ty + 1), name, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def contact_sheet(cells: list[tuple[str, np.ndarray]], columns: int = 3) -> np.ndarray:
    """Lay captioned stills out in a grid. Every still must have the same size."""
    if not cells:
        raise ValueError("a storyboard needs at least one frame")
    h, w = cells[0][1].shape[:2]
    columns = max(1, min(columns, len(cells)))
    rows = -(-len(cells) // columns)
    sheet = Image.new(
        "RGB", (PAD + columns * (w + PAD), PAD + rows * (h + CAPTION + PAD)), SHEET_BG
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    for i, (caption, still) in enumerate(cells):
        x = PAD + (i % columns) * (w + PAD)
        y = PAD + (i // columns) * (h + CAPTION + PAD)
        draw.text((x, y + 2), caption, fill=CAPTION_FG, font=font)
        sheet.paste(Image.fromarray(still).convert("RGB"), (x, y + CAPTION))
    return np.asarray(sheet)


LANE_H = 34
LANE_LABEL = 120
LANE_RANGE = (-60.0, -10.0)  # LUFS drawn bottom to top
LANE_COLORS = {"voice": (236, 239, 244), "bed": (129, 161, 193), "cue": (235, 203, 139)}
SPEECH_BG = (34, 40, 52)


def sound_lanes(lanes: list[dict[str, Any]], duration: float, width: int,
                markers: dict[str, float], spans: list[tuple[float, float]],
                frames: list[float]) -> np.ndarray:  # fmt: skip
    """Loudness lanes to sit under a storyboard: one row per lane
    (``{"label", "kind", "times", "lufs", "marks"}``: its loudness in LUFS over time, and
    for cues the times they play, drawn as diamonds), the spoken spans shaded, markers as
    thin lines and the storyboard's frames as ticks, all on the scene's time axis."""
    rows = max(1, len(lanes))
    img = Image.new("RGB", (width, PAD + rows * (LANE_H + 4) + 16), SHEET_BG)
    draw = ImageDraw.Draw(img)
    font = _font(11)
    x0, x1 = LANE_LABEL, width - PAD
    lo, hi = LANE_RANGE

    def x_of(t: float) -> float:
        return x0 + (x1 - x0) * (t / duration if duration > 0 else 0.0)

    for k, lane in enumerate(lanes):
        top = PAD + k * (LANE_H + 4)
        bottom = top + LANE_H
        for a, b in spans:
            draw.rectangle((x_of(a), top, x_of(b), bottom), fill=SPEECH_BG)
        draw.rectangle((x0, top, x1, bottom), outline=(60, 66, 80))
        color = LANE_COLORS.get(lane["kind"], CAPTION_FG)
        draw.text((PAD, top + LANE_H // 2 - 6), str(lane["label"])[:18], fill=color, font=font)
        runs: list[list[tuple[float, float]]] = [[]]
        for t, v in zip(lane["times"], lane["lufs"], strict=True):
            if np.isfinite(v) and 0 <= t <= duration:
                u = (min(max(v, lo), hi) - lo) / (hi - lo)
                runs[-1].append((x_of(t), bottom - u * LANE_H))
            elif runs[-1]:
                runs.append([])
        for run in runs:
            if len(run) > 1:
                draw.line(run, fill=color, width=1)
        for t in lane.get("marks", ()):
            cx, cy = x_of(t), top + LANE_H / 2
            draw.polygon([(cx, cy - 5), (cx + 5, cy), (cx, cy + 5), (cx - 5, cy)], fill=color)
    end = PAD + rows * (LANE_H + 4)
    for t in markers.values():
        draw.line((x_of(t), PAD, x_of(t), end - 4), fill=(76, 86, 106))
    for t in frames:
        draw.line((x_of(t), end, x_of(t), end + 6), fill=CAPTION_FG)
    draw.text((PAD, end), f"loudness {lo:g} to {hi:g} LUFS", fill=(130, 136, 150), font=font)
    return np.asarray(img)
