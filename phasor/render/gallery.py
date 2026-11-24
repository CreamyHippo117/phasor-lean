"""The golden gallery: frames you love, and a way to
hold new frames up against them.

``gallery/`` holds PNG stills (and short clips' poster frames), each with a one-line note
``name.txt`` on why it works. ``nearest`` finds the gallery frames most like a new frame (by
a coarse color layout and brightness distribution, so a dark blue frame meets dark blue
favorites), and ``compare_sheet`` lays each new frame beside its three nearest so a critic
(or you) can ask: does this belong?

An empty gallery is normal at first; it is yours to fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from phasor.render.storyboard import CAPTION, PAD, SHEET_BG, _font

GRID = (16, 9)  # coarse color layout: columns x rows
BINS = 8  # brightness histogram bins
THUMB = (320, 180)


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path
    note: str


def entries(folder: str | Path) -> list[Entry]:
    folder = Path(folder)
    out = []
    for png in sorted(folder.glob("*.png")) if folder.is_dir() else []:
        note_file = png.with_suffix(".txt")
        text = note_file.read_text(encoding="utf-8").strip() if note_file.exists() else ""
        note = text.splitlines()[0] if text else ""
        out.append(Entry(png.stem, png, note))
    return out


def features(img: np.ndarray) -> np.ndarray:
    """A 16x9 color layout plus an 8-bin brightness histogram, as one vector."""
    pil = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).convert("RGB")
    small = np.asarray(pil.resize(GRID, Image.BILINEAR), dtype=np.float64) / 255.0
    lum = np.asarray(pil.convert("L").resize((64, 36)), dtype=np.float64).ravel() / 255.0
    hist = np.histogram(lum, bins=BINS, range=(0, 1))[0] / lum.size
    return np.concatenate([small.ravel() * 0.5, hist * 2.0])


def nearest(frame: np.ndarray, gallery: list[Entry], k: int = 3) -> list[tuple[Entry, float]]:
    """The ``k`` gallery entries most like ``frame``, nearest first, with their distance."""
    if not gallery:
        return []
    f = features(frame)
    scored = []
    for e in gallery:
        g = features(np.asarray(Image.open(e.path).convert("RGB")))
        scored.append((e, float(np.linalg.norm(f - g))))
    return sorted(scored, key=lambda p: (p[1], p[0].name))[:k]


def compare_sheet(
    cells: list[tuple[str, np.ndarray]], gallery: list[Entry], k: int = 3
) -> tuple[np.ndarray, list[dict]]:
    """One row per new frame: the frame, then its ``k`` nearest gallery frames with their
    notes. Returns the sheet and ``[{"frame": caption, "nearest": [{name, note, distance}]}]``."""
    w, h = THUMB
    cols = 1 + k
    rows = len(cells)
    sheet = Image.new("RGB", (PAD + cols * (w + PAD), PAD + rows * (h + CAPTION + PAD)), SHEET_BG)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    report = []
    for r, (caption, still) in enumerate(cells):
        y = PAD + r * (h + CAPTION + PAD)
        matches = nearest(still, gallery, k)
        draw.text((PAD, y + 2), caption, fill=(220, 220, 235), font=font)
        sheet.paste(Image.fromarray(still).convert("RGB").resize(THUMB), (PAD, y + CAPTION))
        row = []
        for c, (e, dist) in enumerate(matches, start=1):
            x = PAD + c * (w + PAD)
            label = e.name + (f": {e.note}" if e.note else "")
            draw.text((x, y + 2), label[:46], fill=(150, 200, 160), font=font)
            sheet.paste(Image.open(e.path).convert("RGB").resize(THUMB), (x, y + CAPTION))
            row.append({"name": e.name, "note": e.note, "distance": round(dist, 4)})
        report.append({"frame": caption, "nearest": row})
    return np.asarray(sheet), report
