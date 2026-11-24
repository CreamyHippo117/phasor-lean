"""The golden gallery and storyboard comparison."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from phasor import Dot, Scene
from phasor.render import gallery
from tests.conftest import gl_available


def _frame(color, size=(160, 90)):
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


@pytest.fixture
def folder(tmp_path):
    g = tmp_path / "gallery"
    g.mkdir()
    for name, color, note in (
        ("deep_blue", (10, 20, 60), "quiet, one warm accent"),
        ("paper", (240, 235, 220), "airy, lots of negative space"),
        ("night_green", (8, 40, 30), "soft glow on dark green"),
        ("bright_red", (220, 30, 30), ""),
    ):
        Image.fromarray(_frame(color)).save(g / f"{name}.png")
        if note:
            (g / f"{name}.txt").write_text(note + "\nmore text ignored\n", encoding="utf-8")
    return g


def test_entries_read_notes_and_ignore_missing_folders(folder, tmp_path):
    es = gallery.entries(folder)
    assert [e.name for e in es] == ["bright_red", "deep_blue", "night_green", "paper"]
    assert {e.name: e.note for e in es}["deep_blue"] == "quiet, one warm accent"
    assert next(e for e in es if e.name == "bright_red").note == ""
    assert gallery.entries(tmp_path / "nothing") == []


def test_nearest_finds_similar_frames(folder):
    es = gallery.entries(folder)
    near = gallery.nearest(_frame((12, 25, 70)), es, k=3)
    assert next(e.name for e, _ in near) == "deep_blue"
    assert near[0][1] < near[1][1] <= near[2][1]
    assert [e.name for e, _ in gallery.nearest(_frame((235, 230, 215)), es, k=1)] == ["paper"]
    assert gallery.nearest(_frame((0, 0, 0)), [], k=3) == []


def test_compare_sheet_lays_each_frame_beside_its_nearest(folder):
    es = gallery.entries(folder)
    cells = [("a", _frame((12, 25, 70))), ("b", _frame((235, 230, 215)))]
    sheet, report = gallery.compare_sheet(cells, es)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert [r["nearest"][0]["name"] for r in report] == ["deep_blue", "paper"]
    assert all(len(r["nearest"]) == 3 for r in report)
    empty, rep = gallery.compare_sheet(cells, [])
    assert [r["nearest"] for r in rep] == [[], []] and empty.shape[0] > 0


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_storyboard_compare_with_the_gallery(folder):
    s = Scene("cmp")
    s.add(Dot(id="d", pos=(0, 0), r=1.0, style={"color": "highlight"}))
    s.marker("beat", 1.0)
    board = s.engine.storyboard([0.0, 1.0], compare=folder)
    assert board["gallery_size"] == 4 and len(board["compare"]) == 2
    assert board["compare_image"].shape[0] > board["image"].shape[0] // 2
    plain = s.engine.storyboard([0.0, 1.0])
    assert "compare" not in plain
    empty = s.engine.storyboard([0.0], compare=folder / "missing")
    assert empty["gallery_size"] == 0 and empty["compare"][0]["nearest"] == []
