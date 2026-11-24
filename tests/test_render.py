"""Kernel-drawn objects, the final look (bloom, tone map), encoding, render and
bench commands, and the exit demo ("domain-colored 1/(z^2+1) with a glowing contour at 4K")."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, cx, load
from phasor.cli import main
from phasor.core.issues import DocumentError
from phasor.render import gpu
from phasor.render.gpu import ComplexFunction
from phasor.render.gpu.bridge import to_host
from phasor.style.theme import load_theme
from tests.conftest import gl_available

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "residues.py"
GOLDEN = Path(__file__).parent / "golden"
THEME = load_theme("midnight_glow")


def _plane(function="1/(z**2 + 1)", passes=None, **kw):
    s = Scene("plane", passes=passes)
    s.set("camera.height", 4.5)
    s.add(cx.DomainColoring(id="plane", function=function, size=(8.0, 4.5), **kw))
    return s


# ---------------------------------------------------------------- kernel-drawn objects


def test_domain_coloring_reports_its_zeros_and_poles():
    e = _plane().engine
    sem = e.inspect(0, id="plane")["objects"][0]["semantic"]
    assert sem["kind"] == "domain_coloring" and sem["backdrop"] is True
    assert sem["latex"] == r"\frac{1}{z^{2} + 1}" and sem["zeros"] == []
    assert sorted(map(tuple, sem["poles"])) == [(0.0, -1.0), (0.0, 1.0)]
    cubic = _plane("z**3 - 1").engine.state_at(0)
    zeros = cubic["plane"].impl.features(cubic)["zeros"]
    np.testing.assert_allclose(np.sort(np.angle(zeros[:, 0] + 1j * zeros[:, 1])),
                               [-2 * np.pi / 3, 0, 2 * np.pi / 3], atol=1e-9)  # fmt: skip
    # Only what lies inside the rectangle is reported.
    small = Scene("s", passes=None)
    small.add(cx.DomainColoring(id="p", function="1/(z**2+1)", size=(1.0, 1.0)))
    assert small.engine.inspect(0, id="p")["objects"][0]["semantic"]["poles"] == []
    # Non-rational functions have no closed-form list: empty, never guessed.
    trig = _plane("sin(z)").engine
    assert trig.inspect(0, id="plane")["objects"][0]["semantic"]["zeros"] == []


def test_bad_functions_and_sizes_are_rejected_at_apply():
    e = _plane().engine
    for path, value in [("plane.function", "import os"), ("plane.function", "w + 1"),
                        ("plane.size", [0, 1])]:  # fmt: skip
        r = e.apply({"op": "set_param", "path": path, "value": value})
        assert not r.ok and r.errors[0].code == "bad_param", (path, value)
    assert e.apply({"op": "set_param", "path": "plane.function", "value": "exp(z)"}).ok


def test_fractal_semantics():
    s = Scene("f", passes=None)
    s.add(cx.Fractal(id="set", kind="julia", c=(-0.8, 0.156)))
    sem = s.engine.inspect(0, id="set")["objects"][0]["semantic"]
    assert sem == {"kind": "fractal", "backdrop": True, "set": "julia", "max_iter": 256,
                   "c": [-0.8, 0.156]}  # fmt: skip


def test_labels_sit_on_backdrops_instead_of_fleeing_them():
    s = _plane()
    s.add(Dot(id="pole", pos=(0, 1), r=0.08))
    s.add(Text(id="pole.label", text="pole"))
    s.constrain("label(pole) near pole")
    e = s.engine
    assert e.layout.slots["pole.label"][0] == "n" and e.lint() == []


@pytest.mark.gl
def test_a_kernel_drawn_plane_is_one_object_to_the_inspector():
    e = _plane(passes="auto").engine
    obj = e.inspect(0, id="plane")["objects"][0]
    assert obj["visible_fraction"] == 1.0 and obj["on_screen_fraction"] == 1.0
    assert e.hit_test(960, 540, 0) == ["plane"]


@pytest.mark.gl
def test_the_color_pass_shows_the_kernel_at_the_right_place():
    e = _plane(passes="auto", contour=0.0).engine
    color = e.pass_runner().color(e.state_at(0), (480, 270))
    # Pixel (row 200, col 300) sits at this point of the plane (4.5 tall, 8 wide, centered).
    x, y = (300 + 0.5) / 480 * 8 - 4, 2.25 - (200 + 0.5) / 270 * 4.5
    lut = np.asarray(e.state_at(0)["plane"].style["colormap"], dtype=np.float32)
    want = gpu.get("domain").reference(gpu.Tile.full(1, 1), ComplexFunction.parse("1/(z**2+1)"),
                                       (x, y), 1e-6, 0.0, lut)[0, 0]  # fmt: skip
    np.testing.assert_allclose(color[200, 300], want, atol=0.03)
    # Around the pole at +i the phase turns a full circle: every hue meets there.
    py = round((2.25 - 1) / 4.5 * 270)
    ring = color[py - 3 : py + 4, 237:244, :3].reshape(-1, 3)
    assert np.ptp(ring, axis=0).min() > 0.1


@pytest.mark.gl
def test_only_the_visible_part_of_a_plane_is_computed(monkeypatch):
    s = _plane(passes="auto")
    s.set("camera.center", (0.5, 1.0, 0.0))
    s.set("camera.height", 0.01)  # deep inside the plane: a tiny window onto it
    calls = []
    real = gpu.render

    def spy(name, w, h, **kw):
        calls.append((name, w, h, kw["center"], kw["height"]))
        return real(name, w, h, **kw)

    monkeypatch.setattr(gpu, "render", spy)
    s.engine.still(0, (320, 180))
    ((name, w, h, center, height),) = calls
    assert (name, w, h) == ("domain", 320, 180)
    assert center == pytest.approx((0.5, 1.0)) and height == pytest.approx(0.01)


@pytest.mark.gl
def test_fractal_color_matches_the_kernels():
    s = Scene("f")
    s.set("camera.center", (-0.6, 0.0, 0.0))
    s.set("camera.height", 1.8)
    s.add(cx.Fractal(id="set"))
    color = s.engine.pass_runner().color(s.engine.state_at(0), (320, 180))
    assert np.abs(color[90, 192, :3] - THEME.background[:3]).max() < 0.01  # inside the set
    field = gpu.render("escape", 320, 180, center=(-0.6, 0.0), height=1.8)
    lut = np.asarray(THEME.values["colormap"]["sequential"], dtype=np.float32)
    want = gpu.run("palette", gpu.Tile.full(320, 180), field=field, lut=lut, density=0.35,
                   inside=THEME.background[:3])  # fmt: skip
    want = to_host(want)  # a CuPy array on an NVIDIA machine
    close = np.abs(color[..., :3] - want[..., :3]).max(axis=-1) < 0.02
    assert close.mean() > 0.97  # (the set's edge differs where float16 and MSAA meet)


# ---------------------------------------------------------------- the final look


def test_paint_colors_pass_through_and_only_light_blooms():
    from phasor.render.post import finish

    img = np.zeros((40, 60, 4), dtype=np.float32)
    img[..., :3] = 0.5
    img[..., 3] = 1
    np.testing.assert_array_equal(finish(img, THEME, backend="reference"), img)  # no glow
    img[20, 30, :3] = 6.0  # one glowing pixel
    out = finish(img, THEME, backend="reference")
    assert out[20, 30, 0] > 0.9 and out.max() <= 1.0  # rolled off, never above white
    assert out[20, 34, 0] > 0.5 + 1e-3 and out[20, 59, 0] < out[20, 34, 0]  # a halo that fades
    assert out[0, 0, 0] == pytest.approx(0.5, abs=1e-3)  # far away, nothing changes


@pytest.mark.gl
def test_glow_role_makes_a_curve_bloom():
    def frame(glow):
        s = Scene("g")
        s.add(Line(id="l", points=[(-3, 0), (3, 0)], style={"color": "highlight", "glow": glow}))
        return s.engine.frame(0, (320, 180))

    plain, strong = frame("none"), frame("strong")
    near, far = (84, 160), (76, 160)  # 6 and 14 pixels above the line
    assert plain[near][:3].max() < 0.01 and plain[far][:3].max() < 0.01  # off the line: dark
    assert strong[near][0] > plain[near][0] + 0.01  # lit by its bloom...
    assert strong[near][0] > strong[far][0] > plain[far][0]  # ...which fades with distance
    assert strong[90, 160, :3].min() > plain[90, 160, :3].min()


# ---------------------------------------------------------------- encoding and commands


def _decode(path):
    av = pytest.importorskip("av")
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def test_video_writers(tmp_path):
    from phasor.render.encode import EncodeError, open_writer

    frames = [np.full((48, 64, 3), 40 * k, np.uint8) for k in range(5)]
    w = open_writer(tmp_path / "v.mp4", 64, 48, 30)
    for f in frames:
        w.write(f)
    w.close()
    got = _decode(tmp_path / "v.mp4")
    assert len(got) == 5 and abs(int(got[3][10, 10, 0]) - 120) <= 3
    seq = open_writer(tmp_path / "frames", 64, 48, 30)
    for f in frames[:2]:
        seq.write(f)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [
        "frame_00000.png", "frame_00001.png"]  # fmt: skip
    with pytest.raises(EncodeError, match="even"):
        open_writer(tmp_path / "odd.mp4", 63, 48, 30)
    with pytest.raises(EncodeError, match="uint8"):
        open_writer(tmp_path / "x.mp4", 64, 48, 30).write(frames[0].astype(np.float32))


def test_bench_times_every_kernel(tmp_path, capsys):
    assert main(["bench", "--width", "64", "--height", "48", "--repeats", "1",
                 "--dir", str(tmp_path)]) == 0  # fmt: skip
    (saved,) = tmp_path.iterdir()
    result = json.loads(saved.read_text(encoding="utf-8"))
    # every kernel is timed; "name:variant" entries time a kernel with other options
    assert {k.split(":")[0] for k in result["kernels"]} == set(gpu.registered())
    assert all(r["ms"] >= 0 for r in result["kernels"].values())
    assert "escape" in capsys.readouterr().out


def test_render_command(tmp_path, capsys):
    scene = tmp_path / "tiny.py"
    scene.write_text(
        "from phasor import Dot, Scene\n"
        "s = Scene('tiny')\n"
        "s.add(Dot(id='d', r=0.5, style={'glow': 'soft'}))\n"
        "s.at(0).to('d.pos', (2, 0), duration=0.2)\n",
        encoding="utf-8",
    )
    still = tmp_path / "still.png"
    code = main(["render", str(scene), "--frame", "0.1", "-o", str(still)])
    if not gl_available():
        assert code == 1 and "tier C" in capsys.readouterr().err
        return
    assert code == 0 and still.exists()
    video = tmp_path / "tiny.mp4"
    assert main(["render", str(scene), "-o", str(video), "--fps", "20", "--quiet"]) == 0
    assert "4 frames, 960x540 at 20 fps" in capsys.readouterr().out
    assert len(_decode(video)) == 4


# ---------------------------------------------------------------- the exit demo


def test_exit_demo_scene_is_clean_and_knows_what_it_shows():
    e = load(DEMO, passes=None)
    assert e.lint() == []
    sem = e.inspect(2.5, id="plane")["objects"][0]["semantic"]
    assert sorted(map(tuple, sem["poles"])) == [(0.0, -1.0), (0.0, 1.0)]
    # The loop encloses exactly one pole (at +i) at every sampled time.
    for t in np.linspace(0, e.duration, 11):
        pts = np.asarray(e.state_at(float(t))["loop"].params.points)[:, :2]
        r = np.hypot(pts[:, 0], pts[:, 1] - 1)
        assert r.max() < 2.0 - 1e-9  # never reaches the pole at -i


@pytest.mark.gl
def test_exit_demo_lint_is_clean_with_real_pixels():
    assert load(DEMO).lint() == []


@pytest.mark.gl
def test_golden_exit_demo_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = load(DEMO)
    path = GOLDEN / e.pass_runner().renderer_class() / "residues_storyboard.png"
    sheet = e.storyboard([0.0, 1.75, 2.5], marks=True)["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.99


@pytest.mark.gl
@pytest.mark.slow
def test_exit_demo_renders_at_4k():
    e = load(DEMO)
    frame = e.frame(2.5, (3840, 2160))
    assert frame.shape == (2160, 3840, 4) and frame.max() <= 1.0 and np.isfinite(frame).all()
    # The loop (radius 0.7 around +i at t = 2.5) glows: its rim is bright (the prism look
    # runs a hue along it, so which channel peaks depends on where), and the plane just
    # outside it is lit by the bloom compared with the same place at t = 0.
    px = 2160 / 4.5  # pixels per unit
    cx_, cy = 1920, round(1080 - 1.0 * px)
    rim = frame[cy, round(cx_ + 0.7 * px)]
    assert rim[:3].max() > 0.95 and rim[:3].sum() > 1.5
    early = e.frame(0.0, (3840, 2160))
    near = (cy, round(cx_ + 0.8 * px))
    assert frame[near][:3].sum() > early[near][:3].sum()


def test_scripts_still_run_as_documented():
    runpy.run_path(str(ROOT / "examples" / "domain_coloring.py"), run_name="not_main")
    with pytest.raises(DocumentError):
        Scene("x", passes=None).add(cx.DomainColoring(id="p", function="1/"))


# ---------------------------------------------------------------- motion blur


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_motion_blur_smears_moving_things_and_leaves_still_moments_alone():
    from phasor import Dot, Scene

    s = Scene("blur")
    s.set("camera.height", 4)
    s.add(Dot(id="d", pos=(-2, 0), r=0.15, style={"color": "highlight"}))
    s.marker("go", 1.0)
    s.at("go").to("d.pos", (2, 0), duration=1.0, ease="linear")
    e = s.engine
    size = (320, 180)
    assert e.is_moving(1.0, 2.0) and not e.is_moving(0.0, 0.9) and not e.is_moving(2.1, 3.0)
    still_a = e.frame_blurred(0.5, 24, size=size)
    assert (still_a == e.frame(0.5, size)).all()  # nothing moves: identical, no extra cost
    sharp = e.frame(1.5, size)
    blur = e.frame_blurred(1.5, 24, shutter=1.0, samples=6, size=size)
    assert np.array_equal(blur, e.frame_blurred(1.5, 24, shutter=1.0, samples=6, size=size))
    _, xs = np.nonzero(sharp[..., :3].max(-1) > 0.05)
    _, bx = np.nonzero(blur[..., :3].max(-1) > 0.05)
    assert (bx.max() - bx.min()) > (xs.max() - xs.min()) + 2  # a streak, wider than the dot
    assert abs(float(blur.sum() - sharp.sum())) < 0.03 * float(
        sharp.sum()
    )  # light is spread, not lost
    assert e.frame_blurred(1.5, 24, shutter=0.0, size=size).tobytes() == sharp.tobytes()


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_motion_blur_follows_expressions_and_keeps_glow_bright():
    """Boundary review M7-M9: ``is_moving`` saw only tracks, so anything driven by
    ``expr("t ...")`` never blurred; and the average was taken after the tone map, so a
    bright glow streak came out dimmer than the same light at rest."""
    from phasor import Dot, Scene, expr

    s = Scene("blur2")
    s.set("camera.height", 4)
    s.add(Dot(id="d", pos=expr("(-4 + 4 * t, 0, 0)"), r=0.15,
              style={"color": "highlight", "glow": "strong"}))  # fmt: skip
    s.marker("end", 2.0)
    e = s.engine
    assert e.is_moving(0.9, 1.1)
    size = (320, 180)
    sharp = e.frame(1.0, size)
    blur = e.frame_blurred(1.0, 24, shutter=1.0, samples=6, size=size)
    _, xs = np.nonzero(sharp[..., :3].max(-1) > 0.6)  # the bright core, not the halo
    _, bx = np.nonzero(blur[..., :3].max(-1) > 0.6)
    assert (bx.max() - bx.min()) > (xs.max() - xs.min()) + 2  # it smears now
    # averaging tone-mapped frames would lose light where the glow saturates; the HDR
    # average keeps at least as much as a post-tonemap average of the same samples
    half = 0.5 / 24
    ts = [1.0 + (k + 0.5) / 6 * 2 * half - half for k in range(6)]
    late = np.mean([e.frame(t, size) for t in ts], axis=0)
    assert blur[..., :3].sum() >= late[..., :3].sum() * 0.999


def test_the_render_job_blurs_finals_not_previews():
    from phasor.render.job import BLUR

    assert BLUR["preview"][0] == 0 and BLUR["final"] == (0.5, 4) == BLUR["4k"]


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_other_threads_can_render_after_this_one_has():
    """The editor draws on the main thread while an AI connection lints from another. A GL
    context stays current in the thread that used it, so other threads used to fail
    ("cannot create buffer"/"texture": found on Windows 2026-11-21, then here). All GL work
    now runs on one GL thread, and every thread gets the same picture."""
    import threading

    from phasor.render.passes import default_renderer

    s = Scene("threads", passes=None)
    s.add(Line(id="l", points=[(-2, 0), (2, 1)]))
    s.add(Text(id="t", text="hello", pos=(0, 1)))
    state = s.engine.state_at(0.0)
    r = default_renderer()
    want = r.color(state, (160, 90)).copy()  # this thread first
    got, errors = [], []

    def work():
        try:
            for _ in range(10):
                got.append(r.color(state, (160, 90)).copy())
                r.id_buffer(state, (160, 90), force=frozenset({"l"}))
                r._cache.clear()  # really draw again
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(got) == 30
    for g in got:
        np.testing.assert_array_equal(g, want)


def test_a_frame_size_nvenc_refuses_falls_back_to_libx264(tmp_path, monkeypatch):
    """NVIDIA's encoder refuses small frames (64x48 on an RTX 4070 Ti, 2026-11-21)."""
    from phasor.render import encode

    monkeypatch.setattr(encode, "best_encoder", lambda: "pyav:h264_nvenc")
    monkeypatch.setattr(encode, "_nvenc_takes", lambda codec, w, h: w >= 256)
    small = encode.open_writer(tmp_path / "small.mp4", 64, 48, 30)
    assert small.encoder == "pyav:libx264"
    small.close()


@pytest.mark.gpu
def test_h264_nvenc_encodes_at_constant_quality(tmp_path, monkeypatch):
    """H.264 on the card is the default; its quality options must be
    ones NVENC accepts, and a fixed bitrate must not be set."""
    from phasor.render import doctor, encode

    if "h264_nvenc" not in doctor.check_pyav().data.get("working", []):
        pytest.skip("no working h264_nvenc")
    monkeypatch.setattr(encode, "best_encoder", lambda: "pyav:h264_nvenc")
    w = encode.open_writer(tmp_path / "q.mp4", 640, 360, 30)
    assert w.encoder == "pyav:h264_nvenc"
    ramp = np.linspace(0, 255, 640, dtype=np.float64).astype(np.uint8)
    frame = np.repeat(np.repeat(ramp[None, :, None], 360, axis=0), 3, axis=2)
    for _ in range(30):
        w.write(frame)
    w.close()
    import av

    with av.open(str(tmp_path / "q.mp4")) as c:
        assert c.streams.video[0].codec_context.name == "h264"
        assert sum(1 for _ in c.decode(video=0)) == 30
