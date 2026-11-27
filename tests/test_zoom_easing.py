"""Easing in log space (``ease="log:<curve>"``): a 600x camera dive that moves evenly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Dot, Scene, Text, geo
from phasor.core.issues import DocumentError
from phasor.core.values import geo_lerp
from phasor.time.easing import EASINGS, ease_named
from phasor.time.tracks import ResolvedSegment, ResolvedTrack


def test_geo_lerp_is_the_geometric_path_and_hits_its_ends_exactly():
    assert geo_lerp(10.0, 6000.0, 0.0) == 10.0 and geo_lerp(10.0, 6000.0, 1.0) == 6000.0
    assert geo_lerp(10.0, 6000.0, 0.5) == pytest.approx(math.sqrt(10.0 * 6000.0))
    assert geo_lerp(2.0, 8.0, 1 / 3) == pytest.approx(2.0 * 4.0 ** (1 / 3))
    a, b = (1.0, 100.0), (4.0, 1.0)  # tuples go component by component
    assert geo_lerp(a, b, 0.5) == pytest.approx((2.0, 10.0))
    # an overshooting curve stays positive: a log-space path never crosses zero
    assert geo_lerp(1.0, 0.01, 1.2) > 0


def test_log_names_wrap_every_curve_and_nothing_else():
    for name, curve in EASINGS.items():
        assert ease_named(name) == (curve, False)
        assert ease_named(f"log:{name}") == (curve, True)
    assert ease_named("log:wobble") is None and ease_named("zoom") is None
    assert ease_named("log:log:linear") is None


def test_a_log_track_moves_by_equal_ratios_in_equal_times():
    lin = EASINGS["linear"]
    tr = ResolvedTrack(10.0, (ResolvedSegment(0.0, 1.0, 6000.0, lin, geometric=True),))
    vs = np.array([tr.value_at(t) for t in np.linspace(0, 1, 11)[:-1]] + [tr.value_at(1.0)])
    ratios = vs[1:] / vs[:-1]
    assert np.allclose(ratios, 600**0.1)  # each tenth of the time: the same factor
    smooth = EASINGS["smooth"]
    tr = ResolvedTrack(10.0, (ResolvedSegment(0.0, 2.0, 6000.0, smooth, geometric=True),))
    for t in (0.3, 1.0, 1.7):
        expect = math.log(10.0) + smooth(t / 2) * math.log(600.0)
        assert math.log(tr.value_at(t)) == pytest.approx(expect)  # composed with the curve


def _dive(ease: str) -> list[float]:
    s = Scene("dive", passes=None)
    s.add(Dot(id="d", r=0.01))
    s.set("camera.height", 600.0)
    s.at(0.0).to("camera.height", 1.0, duration=3.0, ease=ease)
    e = s.engine
    return [e.state_at(t).values["camera.height"] for t in np.linspace(0, 3, 31)]


def test_a_600x_dive_in_log_space_zooms_evenly_and_a_plain_one_does_not():
    even = np.array(_dive("log:linear"))
    assert even[0] == pytest.approx(600.0) and even[-1] == pytest.approx(1.0)
    assert np.allclose(even[1:] / even[:-1], 600 ** (-1 / 30))
    plain = np.array(_dive("smooth"))
    # the plain dive leaves most of the zoom (in factors) for the last fifth of the time
    assert math.log(plain[24] / plain[30]) > 0.6 * math.log(600.0)
    smooth = np.array(_dive("log:smooth"))
    assert math.log(smooth[15]) == pytest.approx(0.5 * math.log(600.0))  # half the factor


@pytest.mark.parametrize(
    ("start", "to", "path"),
    [(0.0, 1.0, "d.opacity"), (1.0, -2.0, "camera.center"), ("a", "b", "t.text")],
)
def test_log_easing_needs_both_ends_positive(start, to, path):
    s = Scene("bad", passes=None)
    s.add(Dot(id="d", r=1.0))
    s.add(Text(id="t", text="a"))
    if path == "camera.center":
        start, to = (1.0, 1.0, 0.0), (to, 1.0, 0.0)
    s.set(path, start)
    with pytest.raises(DocumentError) as err:
        s.at(0.0).to(path, to, duration=1.0, ease="log:smooth")
    issue = err.value.issues[0]
    assert issue.code == "bad_segment" and "positive" in issue.message


def test_log_easing_starts_from_the_value_the_previous_segment_left():
    s = Scene("chain", passes=None)
    s.add(Dot(id="d", r=1.0))
    s.at(0.0).to("d.r", 4.0, duration=1.0, ease="log:linear")
    s.at(1.0).to("d.r", 1.0, duration=1.0, ease="log:linear")
    assert s.engine.state_at(1.5)["d"].params.r == pytest.approx(2.0)  # sqrt(4 * 1)
    with pytest.raises(DocumentError, match="positive"):
        s.at(2.0).to("d.opacity", 0.0, duration=1.0)  # fine: a plain ease
        s.at(3.0).to("d.opacity", 1.0, duration=1.0, ease="log:linear")  # from 0: refused


def test_an_unknown_log_curve_names_the_choices():
    s = Scene("unknown", passes=None)
    s.add(Dot(id="d", r=1.0))
    with pytest.raises(DocumentError) as err:
        s.at(0.0).to("d.r", 2.0, duration=1.0, ease="log:wobble")
    assert err.value.issues[0].code == "unknown_ease"
    assert "log:" in err.value.issues[0].message


# ---------------------------------------------------------------- verbs that zoom


def _zoom_scene() -> Scene:
    s = Scene("zoom", passes=None)
    s.add(Dot(id="a", pos=(-2, 0), r=0.3))
    s.add(Dot(id="b", pos=(2, 0), r=0.3))
    s.marker("go", 1.0)
    return s


def test_focus_pull_and_a_camera_morph_zoom_in_log_space():
    s = _zoom_scene()
    s.at("go").focus_pull("a")
    track = s.engine.document.timeline.tracks
    curve = s.engine._snap.compiled.theme.motion["focus_pull"]["ease"]
    assert track["camera.height"][0].ease == f"log:{curve}"
    assert track["camera.center"][0].ease == curve  # only the scale moves by factors
    e = s.engine
    d = e._snap.compiled.theme.motion["focus_pull"]["duration"]
    h0, h1 = e.state_at(1.0).values["camera.height"], e.state_at(1.0 + d).values["camera.height"]
    mid = e.state_at(1.0 + d / 2).values["camera.height"]
    assert mid == pytest.approx(math.sqrt(h0 * h1))  # symmetric curve: halfway in factors

    m = _zoom_scene()
    m.at("go").morph("camera.height", 0.05).morph("a.r", 0.6)
    tracks = m.engine.document.timeline.tracks
    curve = m.engine._snap.compiled.theme.motion["morph"]["ease"]
    assert tracks["camera.height"][0].ease == f"log:{curve}"
    assert tracks["a.r"][0].ease == curve  # any other parameter: as before


def test_zoom_verbs_give_the_same_scene_in_any_order():
    one, two = _zoom_scene(), _zoom_scene()
    one.at("go").focus_pull("a")
    one.at("go").morph("b.r", 0.9)
    two.at("go").morph("b.r", 0.9)
    two.at("go").focus_pull("a")
    for t in (0.0, 1.5, 2.0, 4.5, 6.0):
        assert one.engine.state_at(t).values == two.engine.state_at(t).values


def test_a_3d_focus_pull_glides_in_by_factors_from_the_solved_camera():
    s = Scene("pull3d", passes=None)
    s.use_camera("orbit", elevation=20.0, drift=0.0)
    s.add(geo.ParametricCurve(id="helix", fx="cos(t)", fy="sin(t)", fz="0.2*t",
                              t_range=(0, 25), samples=300))  # fmt: skip
    s.add(geo.ParametricSurface(id="ball", fx="0.05*sin(u)*cos(v)+4", fy="0.05*sin(u)*sin(v)",
                                fz="0.05*cos(u)", u_range=(0, 3.14), v_range=(0, 6.28),
                                res=(8, 12)))  # fmt: skip
    s.constrain("camera keep_visible(helix)", priority="weak")
    s.marker("go", 1.0)
    s.marker("end", 4.0)
    s.at("go").focus_pull("ball")
    e = s.engine
    theme = e._snap.compiled.theme.motion["focus_pull"]
    curve, d = EASINGS[theme["ease"]], theme["duration"]
    times = 1.0 + np.arange(0.0, d + 1e-9, 0.1)  # the bake's solve times: no resampling
    dist = {t: e.state_at(t).values["camera.distance"] for t in times}
    d0, d1 = dist[times[0]], dist[times[-1]]
    assert d0 / d1 > 5  # a real dive
    for t, v in dist.items():
        u = (t - 1.0) / d
        assert math.log(v) == pytest.approx(math.log(d0) + curve(u) * math.log(d1 / d0),
                                            abs=1e-6)  # fmt: skip
