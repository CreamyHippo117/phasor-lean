"""The perspective (orbit) camera, in the one spatial pipeline (contract 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text
from phasor.core.spatial import OrbitCameraParams, perspective_camera
from phasor.style.theme import load_theme
from phasor.time.compile import sway

ASPECT = 16 / 9


def cam(**kw):
    return perspective_camera(OrbitCameraParams(**kw), ASPECT)


def test_target_is_at_the_frame_center_and_z_is_up():
    c = cam(target=(1, 2, 3), azimuth=20, elevation=30, distance=8)
    assert np.allclose(c.project([(1, 2, 3)])[0][:2], 0, atol=1e-12)
    above = c.project([(1, 2, 4)])[0]
    assert above[1] > 0.1 and abs(above[0]) < 1e-9  # world up appears straight up
    assert c.depth((1, 2, 3))[0] == pytest.approx(8)


def test_azimuth_convention():
    # azimuth 0: the camera sits on +x looking back at the target, so +y appears on the right
    c = cam(azimuth=0, elevation=0, distance=10)
    assert c.eye[0] == pytest.approx(10) and c.project([(0, 1, 0)])[0][0] > 0
    c90 = cam(azimuth=90, elevation=0, distance=10)
    assert c90.eye[1] == pytest.approx(10) and c90.project([(-1, 0, 0)])[0][0] > 0


def test_nearer_points_have_smaller_ndc_depth_and_positive_w():
    c = cam(distance=10)
    near, far = c.eye + c.forward * 2, c.eye + c.forward * 20
    assert c.project([near])[0][2] < c.project([far])[0][2]
    assert c.clip_w([near])[0] > 0 and c.clip_w([c.eye - c.forward])[0] < 0


def test_units_per_height_matches_projection_at_any_depth():
    c = cam(distance=10, fov=40)
    for d in (5.0, 10.0, 30.0):
        p = c.eye + c.forward * d
        u = c.units_per_height(p)
        top = c.project([p + c.up * u / 2])[0][1]
        assert top == pytest.approx(1.0, rel=1e-9)
    assert c.units_per_height() == pytest.approx(2 * 10 * math.tan(math.radians(20)))
    pts = np.array([c.eye + c.forward * 5, c.eye + c.forward * 10])
    assert np.allclose(c.units_per_height(pts), [c.units_per_height(p) for p in pts])


def test_drift_is_a_pure_function_of_t_and_can_be_off():
    theme = load_theme("midnight_glow")
    p = OrbitCameraParams()
    assert sway(theme, p, 0.0) == 0.0
    assert sway(theme, p, 6.0) == pytest.approx(theme.camera["drift"])  # quarter period
    assert sway(theme, OrbitCameraParams(drift=0), 6.0) == 0.0


def test_a_document_with_an_orbit_camera_draws_and_inspects():
    s = Scene("orbit3d", passes=None)
    s.use_camera("orbit", distance=8, elevation=25, azimuth=-60, drift=0)
    s.add(Dot(id="a", pos=(0, 0, 1), r=0.2, tags=["key"]))
    s.add(Dot(id="b", pos=(3, 0, 0), r=0.2, tags=["key"]))
    s.add(Line(id="axis", points=[(0, 0, 0), (0, 0, 2)]))
    s.add(Text(id="a.label", text="a"))
    s.constrain("label(a) near a")
    e = s.engine
    st = e.state_at(0)
    assert st.camera.perspective
    snap = e.inspect(0, detail="standard")
    by = {o["id"]: o for o in snap["objects"]}
    assert by["a"]["region"] == "upper-center" or by["a"]["center"]["norm"][1] < 0.5
    # text keeps its size on screen however far it is: label height ~ theme text size
    lab = by["a.label"]
    h = lab["bbox"]["norm"][3] - lab["bbox"]["norm"][1]
    assert h == pytest.approx(0.042, rel=0.25)
    # the label was placed next to its anchor on screen
    ax, ay = by["a"]["center"]["norm"]
    lx, ly = lab["center"]["norm"]
    assert math.hypot((lx - ax) * 16 / 9, ly - ay) < 0.15


def test_line_width_is_constant_on_screen_in_perspective():
    s = Scene("w", passes=None)
    s.use_camera("orbit", distance=6, elevation=0, azimuth=0, drift=0)
    s.add(Line(id="l", points=[(2, -3, 0), (-20, -3, 0)]))
    st = s.engine.state_at(0)
    obj = st["l"]
    pts = np.asarray(obj.params.points)
    w = obj.impl._width(st, pts)
    assert w[1] > 3 * w[0]  # the far end is wider in world units ...
    px = [w[i] / st.camera.units_per_height(pts[i]) for i in range(2)]
    assert px[0] == pytest.approx(px[1])  # ... and the same on screen


# ---------------------------------------------------------------- surfaces (need OpenGL)

from phasor import cx  # noqa: E402
from tests.conftest import gl_available  # noqa: E402


def _surface_scene(material="solid"):
    s = Scene("rs")
    s.use_camera("orbit", distance=9, elevation=20, azimuth=-90, drift=0)
    s.add(cx.RiemannSurface(id="surf", radius=2.2, style={"material": material}))
    # a dot well behind the surface, seen through it from this side
    s.add(Dot(id="behind", pos=(0, 1.5, 0.2), r=0.15, tags=["key"]))
    return s


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_solid_surface_hides_what_is_behind_it():
    e = _surface_scene().engine
    by = {o["id"]: o for o in e.inspect(0)["objects"]}
    assert by["behind"]["visible_fraction"] < 0.5
    assert by["behind"]["hidden_by"][0]["id"] == "surf"
    assert by["surf"]["visible_fraction"] > 0.9


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_glass_does_not_hide_and_is_drawn_see_through():
    e = _surface_scene("glass").engine
    by = {o["id"]: o for o in e.inspect(0)["objects"]}
    assert by["behind"]["visible_fraction"] == 1.0
    solid = _surface_scene().engine.still(0, (320, 180)).astype(float)
    glass = e.still(0, (320, 180)).astype(float)
    assert np.abs(solid - glass).mean() > 2  # it looks different ...
    assert glass.mean() > 5  # ... and is not empty


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_surfaces_are_lit_not_flat():
    e = _surface_scene().engine
    img = e.still(0, (480, 270)).astype(float)
    sheet = img[img.sum(-1) > 60]  # surface pixels
    assert len(sheet) > 5000
    lum = sheet.mean(-1)
    assert lum.std() > 8  # shading varies across the surface


# ---------------------------------------------------------------- 3D labels

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _labelled(passes, azimuth=-90.0):
    s = Scene("lab3d", passes=passes)
    s.use_camera("orbit", distance=9, elevation=25, azimuth=azimuth, drift=0)
    s.add(cx.RiemannSurface(id="surf", radius=2.2))
    s.add(Dot(id="bp", pos=(0, 0, 0), r=0.07, style={"color": "highlight"}, tags=["key"]))
    s.add(Text(id="bp.label", text="branch point"))
    s.constrain("label(bp) near bp")
    return s


@pytest.mark.parametrize("passes", MODES)
def test_a_3d_label_lives_at_its_anchors_depth_and_stays_beside_it(passes):
    e = _labelled(passes).engine
    st = e.state_at(0)
    cam = st.camera
    label_depth = cam.depth(st["bp.label"].params.pos)[0]
    assert label_depth == pytest.approx(0.96 * cam.depth((0, 0, 0))[0], rel=1e-6)
    by = {o["id"]: o for o in e.inspect(0)["objects"]}
    (ax, ay), (lx, ly) = by["bp"]["center"]["norm"], by["bp.label"]["center"]["norm"]
    assert math.hypot((lx - ax) * 16 / 9, ly - ay) < 0.15
    assert by["bp.label"]["on_screen_fraction"] == 1.0


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_3d_labels_avoid_only_what_is_in_front_of_them():
    for az in (-90.0, 0.0, 60.0, 150.0, 240.0):
        e = _labelled("auto", az).engine
        lab = next(o for o in e.inspect(0)["objects"] if o["id"] == "bp.label")
        assert lab["visible_fraction"] >= 0.95, (az, lab["hidden_by"])


def test_solver_projection_matches_the_camera():
    from phasor.layout.camera import _cam_xy

    c = cam(distance=7, azimuth=20, elevation=35)
    pts = np.array([[1.0, 2.0, 0.5], [-1.0, 0.3, -0.2]])
    x, y, behind = _cam_xy(c, pts)
    ndc = c.project(pts)
    assert np.allclose(x, ndc[:, 0]) and np.allclose(y, ndc[:, 1]) and behind == 0


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_label_does_not_cut_a_box_out_of_glass_behind_it():
    def scene(with_label: bool):
        s = Scene("box", passes="auto")
        s.use_camera("orbit", distance=7, elevation=25, azimuth=-90, drift=0)
        s.add(cx.RiemannSurface(id="glass", radius=2.2, style={"material": "glass"}))
        if with_label:
            s.add(Text(id="note", text="i         i", pos=(0, 0.3, 0.3)))
        return s.engine

    plain = scene(False).still(0, (480, 270)).astype(float)
    e = scene(True)
    labelled = e.still(0, (480, 270)).astype(float)
    x0, y0, x1, y1 = next(o for o in e.inspect(0)["objects"] if o["id"] == "note")["bbox"]["px"]
    k = 480 / 1920
    box = (slice(int(y0 * k), int(y1 * k)), slice(int(x0 * k), int(x1 * k)))
    # Light text only brightens what is behind it; a cut-out box would darken it.
    darker = (labelled[box].mean(-1) < plain[box].mean(-1) - 20).mean()
    assert plain[box].mean() > 20  # there is glass behind the label to cut
    assert darker < 0.02


@pytest.mark.parametrize("doc_pos", [(-5, -5, 0), (4, 3, 1), (8, 8, 0)])
def test_baked_3d_label_lands_exactly_where_the_placer_put_it(doc_pos):
    """The bake stores each label's screen offset from its anchor at every key time; reading
    the bake back must reproduce it, whatever the label's own (irrelevant) document pos."""
    from phasor.layout.geometry import screen_box, view_of

    s = Scene("exact", passes=None)
    s.use_camera("orbit", distance=8, elevation=30, azimuth=-50, drift=0)
    s.add(Dot(id="d", pos=(4, 3, 1), r=0.1, tags=["key"]))
    s.add(Text(id="d.label", text="label", pos=doc_pos))
    s.constrain("label(d) near d")
    e = s.engine
    bake = e.layout
    track = next(tr for tr in bake.tracks if tr.path == "d.label.pos")
    for k in (0, len(bake.times) // 2, len(bake.times) - 1):
        st = e.state_at(bake.times[k])
        lab = view_of(st, st["d.label"]).point(st["d.label"].params.pos)
        anchor = screen_box(st, st["d"]).center
        off = (lab[0] - anchor[0], lab[1] - anchor[1])
        assert off == pytest.approx(track.keys[k], abs=1e-6)
