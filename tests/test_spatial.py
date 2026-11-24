"""Camera solver, event log, spatial queries and 3D lint."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Scene, Text, cx
from phasor.core.issues import DocumentError
from phasor.inspect import events
from tests.conftest import gl_available

gl = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]


def _follow(passes=None, **cam):
    s = Scene("follow", passes=passes)
    s.use_camera("orbit", drift=0, **cam)
    s.add(Dot(id="ball", pos=(-6, 0, 0), r=0.3, tags=["key"]))
    s.add(Dot(id="post", pos=(0, 0, 0), r=0.2))
    s.marker("go", 0.0)
    s.at(0).to("ball.pos", (6, 2, 1), duration=4, ease="linear")
    return s


# ---------------------------------------------------------------- camera solver


def test_keep_visible_follows_a_moving_object():
    s = _follow()
    s.constrain("camera keep_visible(ball)")
    e = s.engine
    for t in np.linspace(0, 4, 9):
        ball = next(o for o in e.inspect(t)["objects"] if o["id"] == "ball")
        assert ball["on_screen_fraction"] == 1.0, t
    assert not [i for i in e.lint() if i.code == "unsolved"]
    # smooth: the camera never jumps between neighboring 10 Hz keys
    d = np.array(e.layout.camera["camera.distance"])
    assert np.abs(np.diff(d)).max() < 0.5


def test_the_solver_only_sets_what_the_document_leaves_free():
    s = _follow(azimuth=-30)
    s.constrain("camera keep_visible(ball)")
    s.at(0).to("camera.elevation", 10.0, duration=4, ease="linear")
    e = s.engine
    assert set(e.layout.camera) == {"camera.distance", "camera.target"}
    st = e.state_at(2.0)
    assert st.values["camera.azimuth"] == -30
    assert st.values["camera.elevation"] == pytest.approx(30 + (10 - 30) / 2)


def test_center_and_min_size():
    s = _follow()
    s.constrain("camera center(ball) and camera min_size(ball, 0.3)")
    e = s.engine
    for t in (0.0, 2.0, 4.0):
        ball = next(o for o in e.inspect(t)["objects"] if o["id"] == "ball")
        cx_, cy_ = ball["center"]["norm"]
        assert abs(cx_ - 0.5) < 0.03 and abs(cy_ - 0.5) < 0.03
        _, y0, _, y1 = ball["bbox"]["norm"]
        assert (y1 - y0) >= 0.27


def test_an_impossible_goal_is_reported_with_times_not_hidden():
    s = _follow(azimuth=0, elevation=10, distance=6, target=(0, 0, 0), fov=35)
    s.constrain("camera keep_visible(ball)")  # nothing is free: it can only be checked
    issues = [i for i in s.engine.lint() if i.code == "unsolved"]
    assert issues and "keep_visible" in issues[0].message and issues[0].times


def test_camera_goals_need_the_3d_camera():
    s = Scene("flat", passes=None)
    s.add(Dot(id="a"))
    with pytest.raises(DocumentError, match="3D camera"):
        s.constrain("camera keep_visible(a)")


def test_the_solved_camera_is_deterministic():
    def run():
        s = _follow()
        s.constrain("camera keep_visible(ball, post)")
        return s.engine.layout.camera

    a, b = run(), run()
    assert a == b


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_avoid_occlusion_moves_the_camera_until_it_sees_the_object():
    s = Scene("peek", passes="auto")
    s.use_camera("orbit", drift=0, azimuth=0, target=(0, 0, 0))
    s.add(Dot(id="wall", pos=(2, 0, 0.6), r=1.2))  # a disc between the camera and the gem
    s.add(Dot(id="gem", pos=(0, 0, 0), r=0.25, tags=["key"]))
    s.constrain("camera keep_visible(gem) and camera avoid_occlusion(gem)")
    gem = next(o for o in s.engine.inspect(0)["objects"] if o["id"] == "gem")
    assert gem["visible_fraction"] > 0.5, gem["hidden_by"]


# ---------------------------------------------------------------- event log


def _snap(t, v=1.0, on=1.0, center=(0.5, 0.5), hidden_by=(), kind="label", opacity=1.0):
    return t, {"objects": [{
        "id": "lab", "tags": [], "semantic": {"kind": kind}, "opacity": opacity,
        "on_screen_fraction": on, "visible_fraction": v, "center": {"norm": list(center)},
        "hidden_by": [{"id": i, "share": s} for i, s in hidden_by],
    }]}  # fmt: skip


def test_events_fire_on_threshold_crossings_with_hysteresis():
    vs = [1.0, 0.52, 0.48, 0.47, 0.52, 0.47, 0.30, 0.56, 0.95]  # grazes 0.5 then drops
    samples = [_snap(k / 10, v, hidden_by=[("wall", 1 - v)] if v < 1 else ()) for k, v in
               enumerate(vs)]  # fmt: skip
    log = events.log(samples, {"intro": 0.0, "rays": 0.35})
    kinds = [(e["event"], e["t"]) for e in log]
    assert ("partly_occluded", 0.1) in kinds
    assert [k for k in kinds if k[0] == "occluded"] == [("occluded", 0.2)]  # once, not 3 times
    assert ("visible_again", 0.7) in kinds
    occ = next(e for e in log if e["event"] == "occluded")
    assert occ["marker"] == "intro" and occ["severity"] == "warning"
    assert occ["hidden_by"][0]["id"] == "wall" and occ["fix_hints"]
    assert any(e["event"] == "new_occluder" and e["occluder"] == "wall" for e in log)


def test_frame_and_region_events():
    pts = [(0.5, 0.5), (0.55, 0.5), (0.68, 0.5), (0.9, 0.5)]
    samples = [_snap(k, center=c) for k, c in enumerate(pts)]
    samples.append(_snap(4, on=0.2, center=(1.1, 0.5)))
    samples.append(_snap(5, on=0.53, center=(0.99, 0.5)))  # inside the hysteresis band
    samples.append(_snap(6, on=1.0, center=(0.9, 0.5)))
    log = events.log(samples, {})
    regions = [e for e in log if e["event"] == "region"]
    assert [r["region"] for r in regions] == ["middle-right"]  # 0.68 is inside the band
    assert [e["event"] for e in log if "frame" in e["event"]] == ["leaves_frame", "enters_frame"]
    assert next(e for e in log if e["event"] == "leaves_frame")["t"] == 4


def test_objects_not_shown_make_no_events():
    log = events.log([_snap(0, v=0.0, opacity=0.0), _snap(1, v=1.0)], {})
    assert log == []


def test_engine_events_and_when_hidden_without_opengl():
    s = _follow(azimuth=0, elevation=10, distance=6, target=(0, 0, 0), fov=35)
    e = s.engine
    log = e.events()
    leave = [ev for ev in log if ev["event"] == "leaves_frame" and ev["id"] == "ball"]
    assert leave and leave[0]["severity"] == "warning" and leave[0]["marker"] == "go"
    assert e.when_hidden("ball") == []  # no pixel occlusion without OpenGL


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_when_hidden_spans_with_opengl():
    s = Scene("pass", passes="auto")
    s.use_camera("orbit", drift=0, azimuth=0, elevation=0, distance=8)
    s.add(Dot(id="wall", pos=(1.5, 0, 0), r=0.8))
    s.add(Dot(id="ball", pos=(0, -3, 0), r=0.2, tags=["key"]))
    s.at(0).to("ball.pos", (0, 3, 0), duration=4, ease="linear")
    spans = s.engine.when_hidden("ball")
    assert len(spans) == 1
    t0, t1 = spans[0]
    assert 1.2 < t0 < 2.0 < t1 < 2.8


# ---------------------------------------------------------------- free space, 3D lint


@pytest.mark.parametrize("passes", [None, pytest.param("auto", marks=gl)])
def test_free_space_near(passes):
    s = Scene("free", passes=passes)
    s.add(Dot(id="a", pos=(0, 0), r=0.2))
    s.add(Dot(id="block", pos=(0.6, 0), r=0.3))  # covers the east slot
    slots = s.engine.free_space_near("a", 0.0)
    names = [x["slot"] for x in slots]
    assert "e" not in names and "n" in names and "w" in names
    assert slots == sorted(slots, key=lambda x: x["distance"])
    x0, y0, x1, y1 = slots[0]["box"]
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_occluded_label_and_visible_in_3d():
    s = Scene("hidden", passes="auto")
    s.use_camera("orbit", drift=0, azimuth=0, elevation=0, distance=8)
    s.add(Dot(id="wall", pos=(1.5, 0, 0), r=1.0))
    s.add(Text(id="note", text="behind", pos=(0, 0, 0)))  # placed by hand, behind the wall
    s.constrain("visible_in_3d(note)")
    codes = {i.code for i in s.engine.lint()}
    assert {"occluded_label", "unsolved"} <= codes
    assert "overlap" not in codes  # in 3D a covered label is occluded, not overlapping


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_label_on_a_surface_behind_it_is_not_flagged():
    s = Scene("onsurf", passes="auto")
    s.use_camera("orbit", drift=0, azimuth=-60, elevation=30, distance=9)
    s.add(cx.RiemannSurface(id="surf", radius=2.2))
    s.add(Dot(id="bp", pos=(0, 0, 0), r=0.07, tags=["key"]))
    s.add(Text(id="bp.label", text="branch point"))
    s.constrain("label(bp) near bp")
    codes = {i.code for i in s.engine.lint()}
    assert not codes & {"overlap", "occluded_label"}


# ---------------------------------------------------------------- orbit verbs


def _orbit_scene(**cam):
    s = Scene("orb", passes=None)
    s.use_camera("orbit", **({"drift": 0, "azimuth": -60, "elevation": 25, "distance": 9} | cam))
    s.add(cx.RiemannSurface(id="surf", radius=2.0))
    s.add(Dot(id="bp", pos=(0, 0, 0), r=0.07, tags=["key"]))
    s.marker("spin", 1.0)
    s.marker("stop", 7.0)
    return s


def test_orbit_turns_the_camera_until_a_marker():
    s = _orbit_scene()
    s.at("spin").orbit(360, until="stop")
    e = s.engine
    assert e.state_at(0.5).values["camera.azimuth"] == -60
    assert e.state_at(7.0).values["camera.azimuth"] == pytest.approx(300)
    seg = e.document.timeline.tracks["camera.azimuth"][0]
    assert seg.verb == "orbit" and seg.t1.marker == "stop"


def test_orbit_by_theme_time_and_orbit_reveal():
    s = _orbit_scene()
    s.at("spin").orbit(90)
    e = s.engine
    assert e.state_at(1.0 + 14.0 / 4).values["camera.azimuth"] == pytest.approx(30)
    s2 = _orbit_scene()
    s2.at("spin").orbit_reveal("bp")
    st = s2.engine.state_at(0.5)
    assert st["bp"].params.opacity == 0  # hidden until the reveal
    assert s2.engine.state_at(4.0).values["camera.azimuth"] == pytest.approx(-25)


def test_orbit_needs_the_3d_camera():
    from phasor.time.verbs import VerbError

    s = Scene("flat", passes=None)
    s.add(Dot(id="a"))
    with pytest.raises(VerbError, match="3D camera"):
        s.at(0).orbit()


def test_a_full_orbit_with_goals_keeps_everything_framed():
    s = Scene("orb", passes=None)
    s.use_camera("orbit", drift=0, azimuth=-60)  # elevation, distance, target: the solver's
    s.add(cx.RiemannSurface(id="surf", radius=2.0))
    s.add(Dot(id="bp", pos=(0, 0, 0), r=0.07, tags=["key"]))
    s.marker("spin", 1.0)
    s.marker("stop", 7.0)
    s.at("spin").orbit(360, until="stop")
    s.constrain("camera keep_visible(surf, bp)")
    e = s.engine
    assert set(e.layout.camera) == {"camera.elevation", "camera.distance", "camera.target"}
    assert not [i for i in e.lint() if i.code == "unsolved"]
    probs = events.problems(e.events(hz=4))
    assert not [p for p in probs if p["event"] == "leaves_frame"]


@pytest.mark.parametrize("passes", [None, pytest.param("auto", marks=gl)])
def test_glass_hides_nothing_in_every_mode(passes):
    """The analytic oracle and the camera solver use the same rule as the id pass."""
    from phasor.layout.bake import AnalyticOracle, hidden_share
    from phasor.layout.geometry import occludes

    s = Scene("glass", passes=passes)
    s.use_camera("orbit", drift=0, azimuth=-90, elevation=20, distance=9)
    s.add(cx.RiemannSurface(id="sheets", radius=2.2, style={"material": "glass"}))
    s.add(Dot(id="behind", pos=(0, 1.5, 0.2), r=0.15, tags=["key"]))
    st = s.engine.state_at(0)
    assert not occludes(st["sheets"])
    oracle = s.engine._oracle() or AnalyticOracle()
    assert hidden_share(st, "behind", oracle) == 0.0
    assert hidden_share(st, "behind", AnalyticOracle()) == 0.0


def test_hidden_means_under_ten_percent_and_agrees_with_when_hidden():
    log = events.log([_snap(0, 1.0), _snap(1, 0.07, hidden_by=[("wall", 0.93)])], {})
    assert {"partly_occluded", "occluded", "hidden"} <= {e["event"] for e in log}
    samples = [_snap(0, 1.0), _snap(1, 0.46, hidden_by=[("wall", 0.54)]), _snap(2, 1.0)]
    assert any(e["event"] == "occluded" for e in events.log(samples, {}))
    assert events.hidden_spans(samples, "lab") == [(1, 2)]


def test_from_screen_inverts_to_screen_on_a_plane_in_2d_and_3d():
    from phasor.core.spatial import (
        OrbitCameraParams,
        OrthoCameraParams,
        ortho_camera,
        perspective_camera,
    )

    pts = np.array([[0.3, -1.2, 0.0], [2.0, 1.0, 0.0], [-3.0, 0.5, 0.0]])
    for cam in (
        ortho_camera(OrthoCameraParams(height=6.0), 16 / 9),
        perspective_camera(OrbitCameraParams(azimuth=-40.0, elevation=30.0), 16 / 9),
    ):
        px = cam.to_screen(pts, 1920, 1080)
        assert np.allclose(cam.from_screen(px, 1920, 1080), pts, atol=1e-9)
        lifted = pts + np.array([0.0, 0.0, 1.5])  # a plane above z = 0
        back = cam.from_screen(cam.to_screen(lifted, 1920, 1080), 1920, 1080, z=1.5)
        assert np.allclose(back, lifted, atol=1e-9)
