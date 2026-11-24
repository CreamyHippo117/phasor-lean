"""The Riemann surface of sqrt(z), labels readable through a full
orbit, event log clean (examples/riemann_sqrt.py)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pytest

from phasor import load
from phasor.inspect.events import problems
from tests.conftest import gl_available

DEMO = Path(__file__).parent.parent / "examples" / "riemann_sqrt.py"
GOLDEN = Path(__file__).parent / "golden"

pytestmark = [pytest.mark.gl, pytest.mark.slow,
              pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]  # fmt: skip


@functools.cache
def engine():
    return load(DEMO)


@functools.cache
def estimated():
    """The demo with estimated narration timing, as on a machine without a speech engine.
    The golden storyboard uses it, so the picture does not depend on whether espeak is
    installed (its timing moves every cue and so the whole orbit)."""
    import shutil
    import tempfile

    import phasor.voice as voice

    real = voice.espeak_available
    voice.espeak_available = lambda: False
    try:
        script = Path(tempfile.mkdtemp(prefix="phasor_riemann_")) / DEMO.name
        shutil.copy(DEMO, script)  # no speech cache beside it
        return load(script)
    finally:
        voice.espeak_available = real


@functools.cache
def log():
    return tuple(engine().events())  # 10 Hz plus every marker, real pixels


def test_the_camera_makes_a_full_orbit_chosen_by_goals():
    e = engine()
    m = e.document.timeline.markers
    az = [e.state_at(t).values["camera.azimuth"] for t in (m["walk"], m["home"])]
    assert az[1] - az[0] == pytest.approx(360)
    # elevation, distance and target were never written by the script: the solver chose them
    assert set(e.layout.camera) == {"camera.elevation", "camera.distance", "camera.target"}


def test_event_log_is_clean():
    assert problems(log()) == []
    assert any(ev["event"] == "region" for ev in log())  # the log is alive, just clean


def test_labels_readable_through_the_whole_orbit():
    e = engine()
    m = e.document.timeline.markers
    labels = [o.id for o in e.document.objects if o.type == "text" and o.layer == "world"]
    for t in np.arange(m["walk"], m["home"] + 1e-9, 0.25):
        for e_ in e.inspect(float(t))["objects"]:
            if e_["id"] in labels and e_["opacity"] > 0.5:
                assert e_["visible_fraction"] >= 0.85, (t, e_["id"], e_["hidden_by"])
                assert e_["on_screen_fraction"] == 1.0, (t, e_["id"])


def test_strict_lint_is_clean():
    assert [str(i) for i in engine().lint(strict=True)] == []


def test_golden_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = estimated()
    m = e.document.timeline.markers
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "riemann_sqrt.png"
    sheet = e.storyboard([m["branch"], m["walk"] + 3.0, m["home"]], marks=True)["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.99


def test_without_a_speech_engine_the_orbit_is_clean_too():
    """Without espeak (the owner's Windows PC) narration timing is estimated, every cue
    moves, and the solver plans another orbit. That orbit once crossed the walk in front of
    the branch point (19.1 s) and the lower sheet's point (20.8 s): stage 2 weighed hidden
    shares by priority, so a medium goal 86% hidden looked fine (found 2026-11-21)."""
    e = estimated()
    assert not list(e.base_dir.glob("*.narration/*.wav"))  # estimated: no speech was made
    assert [str(i) for i in e.lint(strict=True)] == []
    assert problems(e.events()) == []
