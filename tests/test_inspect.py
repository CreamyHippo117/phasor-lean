"""Inspector: exact logic with a fake pass runner, then ground truth through real GL passes."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, load
from phasor.inspect import inspector
from phasor.inspect.passes import inspect_size
from phasor.time.easing import EASINGS

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "moving_dot.py"
GOLDEN = Path(__file__).parent / "golden"


def _scene(**kw) -> Scene:
    s = Scene("t", **kw)
    s.add(Dot(id="a", pos=(-2, 1), r=0.5, tags=["key"]))
    s.add(Dot(id="b", pos=(2, -1), r=0.5, tags=["key"]))
    return s


# ---------------------------------------------------------------- analytic (tier C)


def test_analytic_mode_gives_geometry_and_says_occlusion_is_unknown():
    e = _scene(passes=None).engine
    snap = e.inspect(0)
    assert snap["occlusion"] == "analytic" and snap["frame"] == {"width": 1920, "height": 1080}
    a = next(o for o in snap["objects"] if o["id"] == "a")
    # camera height 8 -> 135 px per unit; a at (-2, 1) r 0.5
    assert a["center"]["px"] == [690.0, 405.0] and a["bbox"]["px"] == [622.5, 337.5, 757.5, 472.5]
    assert a["center"]["norm"] == [0.359, 0.375] and a["region"] == "center"
    assert a["visible_fraction"] is None and a["hidden_by"] is None
    assert a["on_screen_fraction"] == 1.0
    assert a["semantic"] == {"kind": "point", "position": [-2.0, 1.0, 0.0], "radius": 0.5}
    assert a["relations"] == [
        {"other": "b", "left_of": True, "above": True, "in_front_of": None, "overlaps": False}
    ]


def test_analytic_on_screen_fraction_uses_the_bounding_box():
    s = Scene("t", passes=None)
    s.add(Dot(id="edge", pos=(64 / 9, 0), r=0.5))  # centered on the right edge
    s.add(Dot(id="gone", pos=(20, 0), r=0.5))
    e = s.engine
    assert e.visible("edge", 0)["on_screen_fraction"] == pytest.approx(0.5)
    assert e.visible("gone", 0)["on_screen_fraction"] == 0.0


def test_detail_levels():
    e = _scene(passes=None).engine
    brief = e.inspect(0, detail="brief")
    assert brief["objects"][0].startswith("a [dot] center (0.36,0.38) visible ? on_screen 1.00")
    assert "features" not in e.inspect(0)["objects"][0]
    full = e.inspect(0, detail="full")["objects"][0]
    assert full["features"]["center"] == {"world": [[-2.0, 1.0, 0.0]], "px": [[690.0, 405.0]]}
    assert [o["id"] for o in e.inspect(0, id="b")["objects"]] == ["b"]
    with pytest.raises(ValueError, match="detail"):
        e.inspect(0, detail="verbose")


def test_brief_is_capped_at_40_lines():
    s = Scene("t", passes=None)
    for i in range(60):
        s.add(Dot(id=f"d{i}", pos=(i * 0.1, 0)))
    lines = s.engine.inspect(0, detail="brief")["objects"]
    assert len(lines) == 40 and lines[-1] == "... 21 more objects (ask by id)"


# ---------------------------------------------------------------- fake runner: exact logic


class FakeRunner:
    """Hand-made buffers: a's silhouette is 100 px, of which b (index 2) covers 25."""

    def __init__(self, state):
        w, h = inspect_size(state)
        self.masks = {i: np.zeros((h, w), bool) for i in ("a", "b")}
        self.masks["a"][10:20, 10:20] = True
        self.masks["b"][15:20, 15:30] = True
        self.ids = np.zeros((h, w), np.uint32)
        self.ids[self.masks["a"]] = 1
        self.ids[self.masks["b"]] = 2

    def id_buffer(self, state, size, force=frozenset()):
        return self.ids

    def solo_mask(self, state, size, object_id, zoom_out=1.0):
        return self.masks[object_id]


def test_visibility_and_occluders_from_id_and_solo_buffers():
    state = _scene(passes=None).engine.state_at(0)
    snap = inspector.snapshot(state, FakeRunner(state))
    a, b = snap["objects"]
    assert snap["occlusion"] == "pixel"
    assert a["visible_fraction"] == 0.75 and a["hidden_by"] == [{"id": "b", "share": 0.25}]
    assert b["visible_fraction"] == 1.0 and b["hidden_by"] == []
    assert "hidden_by b 0.25" in inspector.render_detail(snap, "brief")["objects"][0]


# ---------------------------------------------------------------- real GL passes


def _lens_fraction(d: float, r: float, big_r: float) -> float:
    """Share of a disc of radius r covered by a disc of radius big_r at center distance d."""
    if d >= r + big_r:
        return 0.0
    if d <= big_r - r:
        return 1.0
    a1 = r * r * math.acos((d * d + r * r - big_r * big_r) / (2 * d * r))
    a2 = big_r * big_r * math.acos((d * d + big_r * big_r - r * r) / (2 * d * big_r))
    a3 = 0.5 * math.sqrt((-d + r + big_r) * (d + r - big_r) * (d - r + big_r) * (d + r + big_r))
    return (a1 + a2 - a3) / (math.pi * r * r)


def _demo_dot_x(t: float) -> float:
    if t <= 1:
        return -4.0
    if t <= 3:
        return -4 + 8 * EASINGS["smooth"]((t - 1) / 2)
    if t <= 4:
        return 4.0
    return 4 + 5 * min((t - 4) / 2, 1.0)


@pytest.fixture(scope="module")
def demo():
    return load(DEMO)


@pytest.mark.gl
@pytest.mark.parametrize("t", [0, 0.5, 1.3, 1.62, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.35, 2.9, 3.5,
                               4.4, 5.0, 5.2, 5.3, 5.4, 5.5, 6.0, 7.5])  # fmt: skip
def test_exit_demo_where_and_visible_are_correct_at_any_t(demo, t):
    x = _demo_dot_x(t)
    where, vis = demo.where("dot", t), demo.visible("dot", t)
    # where: exact projection (135 px per world unit at 1080p, camera height 8)
    assert where["center"]["px"] == pytest.approx([960 + 135 * x, 540], abs=0.06)
    assert (
        where["region"]
        == ("middle-left", "center", "middle-right")[min(2, max(0, int((x / (128 / 9) + 0.5) * 3)))]
    )
    (rel,) = where["relations"]
    assert rel["other"] == "planet" and rel["left_of"] == (x < -0.04) and not rel["above"]
    assert rel["overlaps"] == (abs(x) < 1.05)
    # on-screen: closed-form share of the disc left of the frame's right edge (x = 64/9)
    edge = (64 / 9 - x) / 0.25
    expected_on = 1.0 if edge >= 1 else 0.0 if edge <= -1 else \
        (math.pi - math.acos(edge) + edge * math.sqrt(1 - edge * edge)) / math.pi  # fmt: skip
    assert vis["on_screen_fraction"] == pytest.approx(expected_on, abs=0.05)
    # visible: closed-form share hidden by the planet (r 0.8 at the origin)
    if expected_on > 0:
        hidden = _lens_fraction(abs(x), 0.25, 0.8)
        assert vis["visible_fraction"] == pytest.approx(1 - hidden, abs=0.06)
        occ = demo.occluders("dot", t)["hidden_by"]
        if hidden > 0.1:
            assert occ[0]["id"] == "planet" and occ[0]["share"] == pytest.approx(hidden, abs=0.06)
        elif hidden == 0:
            assert occ == []
    assert vis["occlusion"] == "pixel"


@pytest.mark.gl
def test_label_follows_the_dot_and_is_hidden_with_it(demo):
    for t in (0, 1.5, 3.2, 5):
        dot, label = demo.where("dot", t), demo.where("dot_label", t)
        assert label["center"]["px"][0] == pytest.approx(dot["center"]["px"][0], abs=0.06)
        assert label["center"]["px"][1] == pytest.approx(540 - 0.7 * 135, abs=0.06)
    assert demo.visible("dot_label", 2.0)["visible_fraction"] < 0.2
    assert demo.visible("dot_label", 0)["visible_fraction"] == 1.0


@pytest.mark.gl
def test_translucent_objects_do_not_occlude_but_are_still_measured():
    s = _scene()
    s.add(Dot(id="veil", pos=(-2, 1), r=1.0, z_index=5, opacity=0.4))
    s.add(Dot(id="lid", pos=(2, -1), r=1.0, z_index=5, opacity=0.6))
    e = s.engine
    assert e.visible("a", 0)["visible_fraction"] == 1.0  # under a translucent veil: not hidden
    assert e.visible("b", 0)["visible_fraction"] == 0.0  # under an opaque-enough lid
    assert e.occluders("b", 0)["hidden_by"] == [{"id": "lid", "share": 1.0}]
    assert e.visible("veil", 0)["visible_fraction"] == 1.0  # measured although it writes no id
    e.apply({"op": "set_param", "path": "lid.opacity", "value": 0.0})
    assert e.visible("lid", 0)["visible_fraction"] == 0.0
    assert e.visible("b", 0)["visible_fraction"] == 1.0


@pytest.mark.gl
def test_z_index_and_document_order_decide_who_is_on_top():
    s = Scene("t")
    s.add(Dot(id="first", r=1.0))
    s.add(Dot(id="second", r=0.5))
    e = s.engine
    assert e.visible("first", 0)["visible_fraction"] == pytest.approx(0.75, abs=0.02)
    assert e.hit_test(960, 540, 0) == ["second", "first"]
    e.apply({"op": "set_param", "path": "first.z_index", "value": 1})
    assert e.visible("second", 0)["visible_fraction"] == 0.0
    assert e.hit_test(960, 540, 0) == ["first", "second"]
    assert e.hit_test(960 + 100, 540, 0) == ["first"] and e.hit_test(5, 5, 0) == []


@pytest.mark.gl
def test_overlay_layer_is_screen_space_and_always_on_top():
    s = Scene("t")
    s.add(Dot(id="world", r=1.0, z_index=99))
    s.add(
        Text(
            id="title", text="Residues", pos=(8 / 9, 0.5), layer="overlay", style={"size": "title"}
        )
    )
    s.add(Dot(id="hud", pos=(0.1, 0.1), r=0.05, layer="overlay"))
    e = s.engine
    assert e.where("hud", 0)["center"]["px"] == [108.0, 108.0]
    assert e.where("hud", 0)["region"] == "upper-left"
    assert e.where("title", 0)["center"]["norm"] == [0.5, 0.5]
    assert e.visible("title", 0)["visible_fraction"] == 1.0
    assert e.occluders("world", 0)["hidden_by"][0]["id"] == "title"
    e.apply({"op": "set_param", "path": "camera.height", "value": 2.0})  # zooming moves the world,
    assert e.where("hud", 0)["center"]["px"] == [108.0, 108.0]  # not the overlay


@pytest.mark.gl
def test_camera_parameters_animate_like_any_other():
    s = _scene()
    s.at(0).to("camera.center", (-2, 1, 0), duration=1, ease="linear")
    s.at(0).to("camera.height", 4, duration=1, ease="linear")
    e = s.engine
    assert e.where("a", 1)["center"]["px"] == [960.0, 540.0]
    assert e.where("a", 0.5)["center"]["px"] == pytest.approx(
        [960 - 180, 540 - 90]
    )  # y is down on screen
    # b spans x 1.5..2.5; the frame now ends at x = -2 + 32/9 = 1.556, so a sliver remains
    assert 0 < e.visible("b", 1)["on_screen_fraction"] < 0.03


@pytest.mark.gl
def test_sub_pixel_objects_are_flagged_not_guessed():
    s = Scene("t")
    s.add(Line(id="hair", points=[(-3, 0), (3, 0)], style={"stroke": "hairline"}))
    entry = s.engine.inspect(0, id="hair")["objects"][0]
    assert entry["visible_fraction"] is None and entry["note"] == inspector.TOO_SMALL


# ---------------------------------------------------------------- render passes + stills


@pytest.mark.gl
def test_three_passes_agree(demo):
    from phasor.render.passes import GLRenderer

    r: GLRenderer = demo.pass_runner()
    state = demo.state_at(0)
    size = inspect_size(state)
    assert size == (480, 270)
    ids = r.id_buffer(state, size)
    assert ids.dtype == np.uint32 and ids.shape == (270, 480)
    index = {o.id: o.index for o in state.objects}
    assert ids[135, 240] == index["planet"] and ids[135, 105] == index["dot"] and ids[5, 5] == 0
    for obj_id in ("dot", "planet", "dot_label"):
        solo = r.solo_mask(state, size, obj_id)
        assert solo.dtype == bool and solo.any()
        assert (ids[solo] == index[obj_id]).all()  # nothing overlaps at t=0: id == solo
        assert ((ids == index[obj_id]) == solo).all()
    area = r.solo_mask(state, size, "planet").sum()
    assert area == pytest.approx(math.pi * (0.8 * 270 / 8) ** 2, rel=0.02)
    zoomed = r.solo_mask(state, size, "planet", zoom_out=2.0).sum()
    assert zoomed == pytest.approx(area / 4, rel=0.05)


@pytest.mark.gl
def test_color_pass_is_linear_and_encoded_to_srgb_exactly_once(demo):
    from phasor.core.color import hex_to_linear
    from phasor.render.output import encode_srgb8

    state = demo.state_at(0)
    linear = demo.pass_runner().color(state, (480, 270))
    assert linear.dtype == np.float32 and linear.shape == (270, 480, 4)
    np.testing.assert_allclose(linear[5, 5], hex_to_linear("#0b0e14"), atol=2e-3)
    np.testing.assert_allclose(linear[135, 240], hex_to_linear("#7aa2f7"), atol=2e-3)
    srgb = encode_srgb8(linear)
    assert srgb.dtype == np.uint8 and srgb.shape == (270, 480, 3)
    assert tuple(srgb[5, 5]) == (0x0B, 0x0E, 0x14) and tuple(srgb[135, 240]) == (0x7A, 0xA2, 0xF7)
    assert tuple(srgb[135, 105]) == (0xFF, 0xB4, 0x54)  # the highlight dot
    # The finished still tone-maps once: channels below the knee (linear 0.9, sRGB 243) come
    # out exactly as the theme wrote them; brighter ones roll off by a few levels.
    still = demo.still(0, (480, 270)).astype(int)
    for y, x in [(5, 5), (135, 240), (135, 105)]:
        want = srgb[y, x].astype(int)
        exact = want <= 243
        assert (still[y, x][exact] == want[exact]).all()
        assert (want[~exact] - still[y, x][~exact] >= 0).all()
        assert (want[~exact] - still[y, x][~exact] <= 5).all()


@pytest.mark.gl
def test_opacity_blends_in_linear_light(demo):
    from phasor.core.color import hex_to_linear

    s = Scene("t")
    s.add(Dot(id="d", r=1.0, opacity=0.5))
    px = s.engine.pass_runner().color(s.engine.state_at(0), (480, 270))[135, 240, :3]
    want = 0.5 * np.array(hex_to_linear("#d8dee9")[:3]) + 0.5 * np.array(
        hex_to_linear("#0b0e14")[:3]
    )
    np.testing.assert_allclose(px, want, atol=3e-3)


@pytest.mark.gl
@pytest.mark.parametrize("t", [0.0, 2.0, 5.0])
def test_golden_stills(demo, t):
    """SSIM >= 0.99 against stored stills. Goldens are kept per renderer class, because a
    software rasterizer and a GPU do not antialias identically."""
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    klass = demo.pass_runner().renderer_class()
    path = GOLDEN / klass / f"moving_dot_{t:.1f}.png"
    still = demo.still(t, (480, 270))
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, still)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(still, np.asarray(Image.open(path).convert("RGB"))) >= 0.99


@pytest.mark.gl
def test_stills_are_deterministic(demo):
    fresh = load(DEMO)
    for t in (0.0, 2.3):
        assert (fresh.still(t, (480, 270)) == demo.still(t, (480, 270))).all()
        assert fresh.inspect(t, detail="full") == demo.inspect(t, detail="full")


# ---------------------------------------------------------------- exit demo script + CLI


def test_exit_demo_script_runs(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    script = tmp_path / "moving_dot.py"
    script.write_text(DEMO.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "dot [dot]" in r.stdout and (tmp_path / "moving_dot.phasor.json").exists()


def test_saved_demo_document_matches_the_script():
    from phasor import Engine
    from phasor.core import io

    saved = Engine.open(ROOT / "examples" / "moving_dot.phasor.json", passes=None)
    assert io.dumps(saved.base) == io.dumps(load(DEMO, passes=None).base)


def test_cli_inspect(capsys):
    from phasor.cli import main

    assert main(["inspect", str(DEMO), "--t", "2", "--detail", "brief"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("t=2.0 occlusion=") and "\ndot [dot] center (0.50,0.50)" in out
    assert main(["inspect", str(DEMO), "--id", "ghost"]) == 1
    assert "no object 'ghost'" in capsys.readouterr().err
    assert main(["inspect", "does_not_exist.phasor.json"]) == 1
