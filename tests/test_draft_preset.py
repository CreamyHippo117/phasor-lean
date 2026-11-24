"""The ``draft`` render preset: faster and rougher, and never leaking
into ``final``, ``4k`` or ``preview``."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time

import numpy as np
import pytest

from phasor import Dot, Scene
from phasor.domains import cx
from phasor.render import encode
from phasor.render.job import BLUR, FAST_ENCODE, FIELD_SCALE, PRESETS, render_file, render_frame

# --------------------------------------------------------------------- the table


def test_existing_presets_are_exactly_what_they_were():
    assert PRESETS["preview"] == (960, 540, 30.0)
    assert PRESETS["final"] == (None, None, None)
    assert PRESETS["4k"] == (3840, 2160, None)
    assert BLUR["preview"] == (0.0, 1)
    assert BLUR["final"] == (0.5, 4) and BLUR["4k"] == (0.5, 4)


def test_draft_is_smaller_slower_to_move_and_unblurred():
    dw, dh, dfps = PRESETS["draft"]
    pw, ph, pfps = PRESETS["preview"]
    assert BLUR["draft"][0] == 0.0 or BLUR["draft"][1] <= 1  # no motion blur
    assert dw * dh < pw * ph and dfps <= pfps


def test_only_draft_gets_a_quality_cut():
    assert set(FIELD_SCALE) == {"draft"} and FIELD_SCALE["draft"] < 1.0
    assert set(FAST_ENCODE) == {"draft"}
    for p in ("preview", "final", "4k"):
        assert p not in FIELD_SCALE and p not in FAST_ENCODE


def test_draft_is_a_choice_of_render_and_bench():
    out = subprocess.run([sys.executable, "-m", "phasor", "render", "--help"],
                         capture_output=True, text=True, encoding="utf-8").stdout  # fmt: skip
    assert "draft" in out
    from phasor.cli import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    bench = sub.choices["bench"]
    choices = next(a for a in bench._actions if a.dest == "preset").choices
    assert "draft" in choices and set(choices) == set(PRESETS)


# --------------------------------------------------------------------- the encoder


def test_fast_encode_flags():
    assert encode._fast_flags("libx264", False) == []
    assert encode._fast_flags("libx264", True) == ["-crf", "28", "-preset", "veryfast"]
    assert encode._fast_flags("h264_nvenc", True)[:2] == ["-preset", "p1"]
    assert encode._fast_flags("prores_ks", True) == []


def test_open_writer_default_is_not_fast(tmp_path, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, *a):
            seen["args"] = a

    monkeypatch.setattr(encode, "best_encoder", lambda: "pyav:libx264")
    monkeypatch.setattr(encode, "PyAVWriter", Fake)
    encode.open_writer(tmp_path / "a.mp4", 64, 36, 30)
    assert seen["args"][-1] is False
    encode.open_writer(tmp_path / "a.mp4", 64, 36, 30, fast=True)
    assert seen["args"][-1] is True


def test_render_frame_without_a_cut_calls_the_engine_as_before():
    calls = []

    class E:
        def frame(self, *a, **k):
            calls.append((a, k))

        def frame_blurred(self, *a, **k):
            calls.append((a, k))

    render_frame(E(), 1.0, 30.0, (8, 6), 0.0, 1)
    render_frame(E(), 1.0, 30.0, (8, 6), 0.5, 4)
    assert [k for _, k in calls] == [{"output": "srgb8"}, {"shutter": 0.5, "samples": 4,
                                                           "size": (8, 6), "output": "srgb8"}]  # fmt: skip
    render_frame(E(), 1.0, 30.0, (8, 6), 0.0, 1, 0.5)
    assert calls[-1][1] == {"output": "srgb8", "field_scale": 0.5}


# --------------------------------------------------------------------- pictures (OpenGL)


def _scene():
    s = Scene("draft_hash")
    s.add(cx.DomainColoring(id="p", function="1/(z**2+1)"))
    s.add(Dot(id="d", pos=(-2.0, 0.0)))
    s.at(0).to("d.pos", (2.0, 0.0, 0.0), duration=0.5)
    s.marker("end", 0.5)
    return s


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# Taken on the untouched tree (Mesa llvmpipe, 2026-11-23), before field_scale existed.
BEFORE = {
    ((96, 54), "sharp"): "e049f579a6187888",
    ((96, 54), "blur"): "3a15354696829338",
    ((160, 90), "sharp"): "0c34388186dd4c78",
    ((160, 90), "blur"): "f4bdb8a7955ee05f",
}


@pytest.mark.gl
@pytest.mark.parametrize("size", [(96, 54), (160, 90)])
def test_full_quality_frames_are_bit_identical_to_before(size):
    e = _scene().engine
    if e.pass_runner().renderer_class() != "software":
        pytest.skip("hashes were taken on Mesa llvmpipe")
    a = e.frame(0.25, size, output="srgb8")
    b = e.frame_blurred(0.25, 30, shutter=0.5, samples=4, size=size, output="srgb8")
    assert _sha(a) == BEFORE[size, "sharp"] and _sha(b) == BEFORE[size, "blur"]
    # the explicit default is the same call
    assert np.array_equal(a, e.frame(0.25, size, output="srgb8", field_scale=1.0))
    assert np.array_equal(a, render_frame(e, 0.25, 30.0, size, 0.0, 1))


@pytest.mark.gl
def test_a_field_scale_changes_only_kernel_fields():
    e = _scene().engine
    size = (160, 90)
    full = e.frame(0.25, size, output="srgb8")
    half = e.frame(0.25, size, output="srgb8", field_scale=0.5)
    assert half.shape == full.shape and not np.array_equal(half, full)
    # a picture with no kernel field is untouched by it
    s = Scene("dots")
    s.add(Dot(id="d", pos=(0.0, 0.0)))
    s.marker("end", 1.0)
    d = s.engine
    assert np.array_equal(d.frame(0.5, size, output="srgb8"),
                          d.frame(0.5, size, output="srgb8", field_scale=0.5))  # fmt: skip
    # and the half-resolution field is still the same picture to within a smooth stretch
    err = np.abs(half.astype(int) - full.astype(int)).mean()
    assert err < 12, err


@pytest.mark.gl
def test_draft_renders_a_short_mp4_and_final_does_not_get_the_cut(tmp_path):
    e = _scene().engine
    r = render_file(e, tmp_path / "d.mp4", preset="draft", end=0.4)
    assert r["size"] == (640, 360) and r["fps"] == 15.0 and r["frames"] == 6
    assert r["field_scale"] == 0.5 and not r["motion_blur"]
    assert (tmp_path / "d.mp4").stat().st_size > 0
    p = render_file(e, tmp_path / "p.mp4", preset="preview", end=0.1)
    assert p["field_scale"] == 1.0
    f = render_file(e, tmp_path / "f.png", preset="final", frame=0.25)
    assert f["frames"] == 1
    from PIL import Image

    want = e.frame(0.25, (e.document.meta.width, e.document.meta.height), output="srgb8")
    assert np.array_equal(np.asarray(Image.open(tmp_path / "f.png")), want)


@pytest.mark.gl
def test_draft_is_faster_than_preview_for_a_field_scene(tmp_path, capsys):
    from pathlib import Path

    from phasor.scene import load

    path = Path(__file__).resolve().parent.parent / "examples" / "prism_domain.py"
    e = load(str(path))
    secs = {}
    for preset in ("preview", "draft"):
        render_file(e, tmp_path / f"w_{preset}.mp4", preset=preset, end=0.2)  # warm up
        t0 = time.perf_counter()
        render_file(e, tmp_path / f"{preset}.mp4", preset=preset, end=1.0)
        secs[preset] = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nprism_domain 1 s: preview {secs['preview']:.2f} s, draft {secs['draft']:.2f} s")
    assert secs["draft"] < secs["preview"]


def test_a_render_worker_passes_the_field_cut_to_render_frame(monkeypatch):
    """The worker's frame call carries ``field_scale`` (it did not before: drafts drew full
    fields under ``--workers``). Called in-process with the engine and renderer stubbed."""
    import queue

    from phasor import scene
    from phasor.engine import Engine
    from phasor.render import job

    seen = []
    monkeypatch.setattr(Engine, "from_payload", classmethod(lambda cls, payload: object()))
    monkeypatch.setattr(job, "render_frame", lambda *a: seen.append(a[-1]) or b"frame")
    q = queue.Queue()
    scene._render_worker({}, [(0, 2)], [0.0, 0.1], (8, 8), 10.0, 0.0, 1, q, 1.0, 1, None, 0.0, 0.5)
    assert seen == [0.5, 0.5]
    assert [q.get()[0] for _ in range(2)] == ["frame", "frame"]
