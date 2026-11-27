"""Camera tracking with the 3D objects (``geo``): camera goals, focus pulls, orbits at
markers and labels on 3D curves, checked through the inspector as an AI would."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Scene, Text, geo


def _helix(**kw) -> geo.ParametricCurve:
    return geo.ParametricCurve(id="helix", fx="cos(t)", fy="sin(t)", fz="0.2*t",
                               t_range=(0, 25), samples=300, **kw)  # fmt: skip


def _ball() -> geo.ParametricSurface:
    return geo.ParametricSurface(id="ball", fx="sin(u)*cos(v)+4", fy="sin(u)*sin(v)",
                                 fz="cos(u)", u_range=(0, 1.57), v_range=(0, 6.28),
                                 res=(16, 24))  # fmt: skip


def _on_screen(e, t, ids):
    return {i: e.inspect(t, id=i)["objects"][0]["on_screen_fraction"] for i in ids}


def _camera(e, t) -> np.ndarray:
    v = e.state_at(t).values
    return np.array([v["camera.azimuth"], v["camera.elevation"], v["camera.distance"],
                     *v["camera.target"]])  # fmt: skip


def _unsolved(e):
    return [i for i in e.lint() if i.code == "unsolved"]


def test_camera_goals_frame_every_kind_of_3d_object():
    s = Scene("goals")
    s.use_camera("orbit")
    s.add(_helix())
    s.add(_ball())
    s.add(geo.FlowLines(id="flow", fp="-y", fq="x", fr="0.2", pos=(-5, 0, 0),
                        extent=(3, 3, 0), seeds=(3, 3, 1)))  # fmt: skip
    s.add(geo.VectorField3D(id="field", fp="x", fq="y", fr="z", pos=(0, -5, 0),
                            extent=(2, 2, 2), counts=(3, 3, 3)))  # fmt: skip
    s.constrain("camera keep_visible(helix, ball, flow, field)")
    e = s.engine
    seen = _on_screen(e, 0.0, ["helix", "ball", "flow", "field"])
    assert all(v > 0.98 for v in seen.values()), seen
    assert not _unsolved(e)
    cam = _camera(e, 0.0)
    assert cam[2] != pytest.approx(10.0)  # the solver chose the distance, not the default


def test_a_focus_pull_at_a_marker_starts_from_the_solved_camera():
    """Before the pull the solver frames the helix; the pull then glides from there to
    the ball. Before this fix the camera sat at its defaults until the pull, out of frame."""
    s = Scene("pull")
    s.use_camera("orbit", elevation=20.0, drift=0.0)
    s.add(_helix())
    s.add(_ball())
    s.constrain("camera keep_visible(ball)")
    s.constrain("camera keep_visible(helix)", priority="weak")
    s.marker("go", 1.0)
    s.marker("end", 4.0)
    s.at("go").focus_pull("ball")
    e = s.engine
    assert _on_screen(e, 0.0, ["helix"])["helix"] > 0.98  # framed before the pull
    assert np.linalg.norm(_camera(e, 0.0)[3:] - 0.0) > 0.3  # target solved, not the origin
    # the pull lands on the ball and nothing jumps on the way
    end = _camera(e, 4.0)
    assert end[3:] == pytest.approx([4.0, 0.0, 0.5], abs=0.05)
    prev = _camera(e, 0.0)
    for t in np.arange(0.05, 4.0, 0.05):
        cur = _camera(e, float(t))
        assert np.abs(cur[3:] - prev[3:]).max() < 0.6, (t, prev, cur)
        assert abs(math.log(cur[2] / prev[2])) < 0.2, (t, prev, cur)  # a zoom moves by factors
        prev = cur
    assert _on_screen(e, 4.0, ["ball"])["ball"] > 0.98


def test_an_orbit_at_a_marker_turns_by_its_angle_from_where_the_solver_was():
    s = Scene("spin")
    s.use_camera("orbit", drift=0.0)
    s.add(_helix())
    s.constrain("camera keep_visible(helix)")
    s.marker("spin", 1.0)
    s.marker("end", 4.5)
    s.at("spin").orbit(90)
    e = s.engine
    seg = e.document.timeline.tracks["camera.azimuth"][0]
    start, end = _camera(e, 1.0)[0], _camera(e, 4.5)[0]
    assert end - start == pytest.approx(90.0, abs=0.5), (seg, start, end)
    for t in (1.0, 2.0, 3.0, 4.5):
        assert _on_screen(e, t, ["helix"])["helix"] > 0.98
    assert not _unsolved(e)


def test_a_label_near_a_3d_curve_follows_it_while_the_camera_turns():
    s = Scene("label")
    s.use_camera("orbit", elevation=25.0, drift=0.0)
    s.add(_helix())
    s.add(Text(id="helix.label", text="helix"))
    s.constrain("label(helix) near helix")
    s.constrain("camera keep_visible(helix)")
    s.marker("spin", 0.5)
    s.marker("end", 3.5)
    s.at("spin").orbit(180)
    e = s.engine
    for t in (0.0, 1.5, 3.5):
        o = e.inspect(t, id="helix.label")["objects"][0]
        assert o["on_screen_fraction"] > 0.99, (t, o)
    assert not [i for i in e.lint() if i.code in ("unsolved", "overlap", "label_far")]


def test_a_growing_curve_stays_framed_as_it_draws():
    from phasor import expr

    s = Scene("grow")
    s.use_camera("orbit", elevation=20.0, drift=0.0)
    s.add(_helix(progress=expr("min(t/3, 1)")))
    s.constrain("camera keep_visible(helix)")
    s.marker("end", 3.0)
    e = s.engine
    for t in (0.5, 1.5, 3.0):
        assert _on_screen(e, t, ["helix"])["helix"] > 0.98
    assert not _unsolved(e)
