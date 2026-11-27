"""PhasorField: the modes of a filter as turning hands, and their exact motion blur. The maths
first (the hands, the hue, the seats, the swept exposure against a brute-force shutter, its
light conserved and its zero-speed limit), then the object through the engine and its golden."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, filt
from phasor.domains import filters_math as fm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


# ---------------------------------------------------------------- the hands


def test_a_hand_is_the_residue_times_the_pole_to_the_tick():
    poles = filt.polar([0.9, 0.5, 0.99], [0.3, -2.0, math.pi])
    res = ((1.0, 0.5), (-0.2, 0.7), (0.3, 0.0))
    lam = np.array([complex(*p) for p in poles])
    r = np.array([complex(*q) for q in res])
    for n in (0, 1, 2, 7):
        assert fm.mode_hands(poles, res, n) == pytest.approx(r * lam**n, abs=1e-12)
    assert fm.mode_hands(poles, (), 3) == pytest.approx(lam**3, abs=1e-12)  # residues all 1


def test_between_ticks_a_hand_turns_and_shrinks_smoothly():
    """A fractional tick turns by n theta and shrinks by rho^n: a spiral, not a jump."""
    z = fm.mode_hands(filt.polar([0.8], [1.0]), (), 2.5)[0]
    assert abs(z) == pytest.approx(0.8**2.5) and np.angle(z) == pytest.approx(2.5)


def test_polar_gives_the_poles_as_pairs():
    got = filt.polar([1.0, 0.5], [0.0, math.pi / 2])
    assert np.array(got) == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.5]]))
    assert all(isinstance(v, float) for pair in got for v in pair)  # plain data for the document


def test_hue_is_the_angle_round_the_wheel():
    assert fm.phase_hue([0.0, math.pi / 2, math.pi, -math.pi / 2]) == pytest.approx(
        [0.0, 0.25, 0.5, 0.75]
    )
    assert fm.phase_hue(0.0, shift=0.3) == pytest.approx(0.3)
    assert fm.phase_hue(2 * math.pi + 0.1) == pytest.approx(0.1 / (2 * math.pi))


@pytest.mark.parametrize("layout", ["spiral", "grid", "line"])
def test_seats_sit_about_a_spacing_apart(layout):
    xy = fm.seat_offsets(60, layout, 1.5)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(60) * 1e9
    nearest = d.min(axis=1)
    assert np.median(nearest) == pytest.approx(1.5, rel=0.2)
    assert nearest.min() > 0.7 * 1.5  # nobody sits on anybody
    assert np.abs(xy.mean(axis=0)).max() < 1.5  # about the centre


def test_the_order_puts_the_slowest_hand_in_the_middle_seat():
    s = Scene("pf_order", passes=None)
    s.add(filt.PhasorField(id="f", poles=filt.polar([0.9] * 3, [2.0, 0.1, 1.0])))
    st = s.engine.state_at(0.0)
    seats = st["f"].impl.features(st)["seats"]
    r = np.linalg.norm(seats[:, :2], axis=1)
    assert list(np.argsort(r)) == [1, 2, 0]  # 0.1 first, then 1.0, then 2.0


# ---------------------------------------------------------------- the exact smear


def _brute(r, psi, angle0, sweep, length, width, steps=40000):
    """The share of the shutter a point (radius r, angle psi) lies inside the stroke of a hand
    of ``length`` and ``width`` turning from ``angle0`` through ``sweep``: sampled in time."""
    phi = angle0 + sweep * (np.arange(steps) + 0.5) / steps
    across = r * np.sin(psi - phi)
    along = r * np.cos(psi - phi)
    return float(np.mean((np.abs(across) <= width / 2) & (along >= 0) & (along <= length)))


@pytest.mark.parametrize("sweep", [0.6, -0.6, 0.02, 2.5])
def test_the_exposure_is_what_a_shutter_records(sweep):
    w, r = 0.02, 0.5
    for x in np.linspace(-0.03, abs(sweep) * r + 0.03, 23):
        psi = 0.4 + math.copysign(1.0, sweep) * x / r
        want = _brute(r, psi, 0.4, sweep, 1.0, w)
        assert fm.sweep_exposure(x, r, sweep, w) == pytest.approx(want, abs=4e-3), x


def test_the_exposure_falls_as_one_over_r_once_the_arc_is_wider_than_the_stroke():
    w, sweep = 0.01, 1.0
    r = np.array([0.1, 0.2, 0.4])
    e = fm.sweep_exposure(r * sweep / 2, r, sweep, w)  # mid-sweep
    assert e == pytest.approx(w / (r * sweep))
    assert fm.sweep_exposure(0.0, 0.001, sweep, w) == 1.0  # near the seat: always covered


def test_with_no_sweep_the_exposure_is_the_still_stroke():
    x = np.array([-0.02, -0.004, 0.0, 0.0049, 0.006])
    assert fm.sweep_exposure(x, 0.5, 0.0, 0.01).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert fm.sweep_exposure(x, 0.5, 1e-15, 0.01).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("sweep", [0.0, 1e-3, 0.05, 0.5, 2.0, 6.0, -1.0, 12.0])
@pytest.mark.parametrize("lengths", [(1.0, 1.0), (1.0, 0.6)])
def test_a_smear_keeps_the_light_of_the_hand(sweep, lengths):
    """The swept mesh's light integrates to the still hand's: width times its mean length."""
    w = 0.02
    a0, a1 = lengths
    pts, lit, _ = fm.swept_hand(0.3, sweep, a0, a1, w, 96, 24)
    tau = np.linspace(0, 1, 4001)
    mean = float(np.mean(a0 * (a1 / a0) ** tau))
    assert fm.mesh_integral(pts, lit) == pytest.approx(w * mean, rel=0.015)


def test_the_light_of_a_fading_hand_is_its_mean_light():
    w = 0.02
    pts, lit, _ = fm.swept_hand(0.0, 1.0, 1.0, 1.0, w, 96, 24, light0=1.0, light1=0.5)
    tau = np.linspace(0, 1, 4001)
    assert fm.mesh_integral(pts, lit) == pytest.approx(w * np.mean(0.5**tau), rel=0.015)


def test_the_mesh_covers_the_swept_sector_and_takes_its_colour_from_its_angle():
    pts, lit, ang = fm.swept_hand(0.5, 1.2, 1.0, 1.0, 0.01, 64, 16)
    r = np.linalg.norm(pts, axis=-1)
    assert r.max() == pytest.approx(1.0, abs=1e-9)
    at_rim = ang[-1]  # the outer row: from just before 0.5 to just past 1.7
    assert at_rim.min() == pytest.approx(0.5 - 0.005, abs=1e-6)
    assert at_rim.max() == pytest.approx(1.7 + 0.005, abs=1e-6)
    assert np.allclose(np.arctan2(pts[-1, :, 1], pts[-1, :, 0]), at_rim)
    assert lit[1:, 0].max() < 1e-12 and lit[1:, -1].max() < 1e-12  # unlit at both edges
    assert lit[0].min() == 1.0  # the seat itself is always under the hand
    assert lit.max() <= 1.0


# ---------------------------------------------------------------- the object


def _field(**kw):
    s = Scene("pf_test", passes=None)
    poles = filt.polar([0.95, 0.95, 0.95], [0.01, 1.0, 3.0])
    s.add(filt.PhasorField(id="f", poles=poles, **kw))
    return s.engine


def test_semantics_name_the_fast_hands_and_the_sum():
    e = _field(frame_ticks=1.0, tick=2.0)
    sem = e.inspect(0.0, id="f")["objects"][0]["semantic"]
    assert sem["kind"] == "phasor_field" and sem["count"] == 3 and sem["stable"]
    assert sem["fastest"] == 2 and sem["fastest_deg_per_tick"] == pytest.approx(171.8873, 1e-4)
    assert sem["max_turn_per_frame_deg"] == pytest.approx(171.8873, 1e-4)
    assert sem["blurred"] == [1, 2] and sem["blurred_count"] == 2  # the slow one stays sharp
    lam = np.array([complex(*p) for p in filt.polar([0.95] * 3, [0.01, 1.0, 3.0])])
    assert complex(*sem["sum"]) == pytest.approx(complex((lam**2).sum()), abs=1e-8)
    assert sem["aliased"] == []
    fast = _field(frame_ticks=1.2).inspect(0.0, id="f")["objects"][0]["semantic"]
    assert fast["aliased"] == [2]  # past half a turn a frame: the wagon wheel
    still = _field().inspect(0.0, id="f")["objects"][0]["semantic"]
    assert still["blurred"] == [] and still["max_turn_per_frame_deg"] == 0.0


def test_bad_residues_are_refused():
    s = Scene("pf_bad", passes=None)
    with pytest.raises(Exception, match="one"):
        s.add(filt.PhasorField(id="f", poles=((0.5, 0.0), (0.0, 0.5)), residues=((1.0, 0.0),)))
    with pytest.raises(Exception, match="at least 1"):
        s.add(filt.PhasorField(id="g", poles=()))


def test_decay_shows_as_length_or_as_light():
    e = _field(tick=10.0, scale=1.0)
    st = e.state_at(0.0)
    tips = st["f"].impl.features(st)["tips"] - st["f"].impl.features(st)["seats"]
    assert np.linalg.norm(tips, axis=1) == pytest.approx([0.95**10] * 3)
    e = _field(tick=10.0, scale=1.0, decay="glow")
    st = e.state_at(0.0)
    tips = st["f"].impl.features(st)["tips"] - st["f"].impl.features(st)["seats"]
    assert np.linalg.norm(tips, axis=1) == pytest.approx([1.0] * 3)


def test_pure_in_time():
    from phasor import expr

    e = _field(tick=expr("4 * t"), frame_ticks=4 / 60)
    a = e.inspect(1.3)["objects"]
    e.inspect(0.2)
    assert e.inspect(1.3)["objects"] == a


def _light(**kw):
    s = Scene("pf_light")
    s.set("camera.height", 3.0)
    s.add(filt.PhasorField(id="f", layout="line", scale=1.2, rings=0.0, **kw))
    r = s.engine.pass_runner()
    pic, _ = r.color_and_emission(s.engine.state_at(0.0), (640, 360))
    blank = s.engine.fork()
    assert blank.apply({"op": "remove_object", "id": "f"}, origin="script").ok
    bg, _ = r.color_and_emission(blank.state_at(0.0), (640, 360))
    return (pic[..., :3] - bg[..., :3]).astype(np.float64)


@needs_gl
def test_a_slow_hand_draws_exactly_as_with_no_blur():
    one = filt.polar([0.99], [0.002])
    assert np.array_equal(_light(poles=one, frame_ticks=1.0), _light(poles=one))


@needs_gl
def test_a_fast_hand_spreads_the_same_light():
    """On screen, the smear of a fast hand holds the light of the sharp hand (the seat dot is
    the same in both), so it does not flash brighter or darker as it speeds up."""
    one = filt.polar([1.0], [2.0])
    sharp, smear = _light(poles=one), _light(poles=one, frame_ticks=1.0)
    assert not np.array_equal(sharp, smear)
    assert (smear > 1e-4).sum() > 5 * (sharp > 1e-4).sum()  # spread over the sector
    assert smear.sum() == pytest.approx(sharp.sum(), rel=0.08)
    lit = smear.sum(-1) > 1e-4
    assert smear.sum(-1)[lit].mean() < 0.3 * sharp.sum(-1)[sharp.sum(-1) > 1e-4].mean()


@needs_gl
def test_the_field_glows_in_the_prism_look():
    s = Scene("pf_glow", look="prism")
    s.add(filt.PhasorField(id="f", poles=filt.polar([0.9] * 4, [0.2, 1.0, 2.0, 3.0]),
                           frame_ticks=1.0))  # fmt: skip
    _, light = s.engine.pass_runner().color_and_emission(s.engine.state_at(0.0), (320, 180))
    assert light[..., :3].max() > 0.2


def test_the_example_builds_and_lints_clean():
    e = runpy.run_path(str(ROOT / "examples" / "reverb_modes.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sem = e.inspect(0.5, id="modes")["objects"][0]["semantic"]
    assert sem["count"] == 64 and sem["blurred_count"] > 40
    assert abs(sem["sum"][1]) < 1e-9  # conjugate pairs: the impulse response is real


@needs_gl
@pytest.mark.slow
def test_reverb_modes_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "reverb_modes.py"), run_name="x")["s"].engine
    sheet = e.storyboard([0.5, 4.0, 9.0], size=(320, 180))["image"]
    _golden("reverb_modes", sheet, e.pass_runner().renderer_class())
