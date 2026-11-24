"""``phasor bench --frames``: where a final frame's time goes."""

from __future__ import annotations

import json

import pytest

from phasor import Dot, Scene
from phasor.render.bench import STAGES, run_frame_bench, save_frames

pytestmark = pytest.mark.gl  # it draws


def _moving_scene():
    s = Scene("bench")
    s.add(Dot(id="d", pos=(-2.0, 0.0)))
    s.at(0).to("d.pos", (2.0, 0.0, 0.0), duration=2.0)
    return s.engine


def test_frame_bench_reports_every_stage(tmp_path):
    r = run_frame_bench(_moving_scene(), "bench", count=4, preset="final")
    assert r["frames"] == 4 and r["motion_blur"]
    stages = r["stages_ms_per_frame"]
    assert set(stages) == {*STAGES, "other"}
    for s in ("state", "draw", "readback", "blur sum", "post", "srgb", "encode"):
        assert r["calls"].get(s, 0) > 0, s  # every stage of a blurred frame was seen
    # the color pass is read asynchronously: nothing waits for the card after drawing, so
    # its drawing time lands in "readback" (the wait for the copy) instead of "gpu draw"
    assert stages["gpu draw"] == 0.0
    # blurred samples are widened on the card, inside "blur sum";
    # "widen" (on the host) is left to sharp frames
    assert stages["widen"] == 0.0
    assert all(v >= 0 for v in stages.values())
    total = sum(stages.values())
    assert total == pytest.approx(r["ms_per_frame"], rel=0.05)  # the stages explain the frame
    path = save_frames(r, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["scene"] == "bench"


def test_rendering_without_a_stopwatch_records_nothing(tmp_path):
    """The marks cost nothing in normal renders: no stopwatch, no card synchronization."""
    from phasor.core import timing

    _moving_scene().frame(1.0, (64, 36))
    assert timing._active is None


@pytest.mark.gpu
def test_frame_bench_runs_its_post_stage_on_the_card():
    r = run_frame_bench(_moving_scene(), "bench", count=2, preset="final")
    assert r["backend"] == "cuda" and r["calls"]["post"] > 0
