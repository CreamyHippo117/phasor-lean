"""A lattice of points that flash as a surface sweeps it (``sweep``): the math first
(arrival distances against known geometry, the image-source rule at high order, the
sphere-to-plane blend, the flash envelope, the front's mesh), then the objects, the example
and its golden storyboard."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, expr, sweep
from phasor.domains import ac_math
from phasor.domains import sweep_math as sm
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4), (6.0, 3.4, 1.4)


# ---------------------------------------------------------------- arrival distances


def test_a_sphere_reaches_each_point_at_its_distance_from_the_centre():
    pts = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 2.0, 2.0]])
    s = sm.arrival(pts, (0, 0, 0), (0, 0, 1), 0.0)
    assert np.allclose(s, [5.0, 0.0, 3.0])


def test_a_plane_reaches_each_point_at_its_height_along_the_normal():
    pts = np.array([[3.0, 4.0, 2.0], [1.0, -7.0, -1.0], [0.0, 0.0, 0.0]])
    s = sm.arrival(pts, (1, 1, 1), (0, 0, 2), 1.0)  # the normal need not be a unit vector
    assert np.allclose(s, [1.0, -2.0, -1.0])
    tilted = sm.arrival(pts, (0, 0, 0), (1, 1, 0), 1.0)
    assert np.allclose(tilted, (pts[:, 0] + pts[:, 1]) / np.sqrt(2))


def test_the_blend_is_linear_in_flat_and_keeps_the_apex_fixed():
    gen = np.random.default_rng(1)
    pts = gen.normal(size=(200, 3)) * 5
    c, n = np.array([1.0, -2.0, 0.5]), np.array([0.0, 1.0, 0.0])
    sphere, plane = sm.arrival(pts, c, n, 0.0), sm.arrival(pts, c, n, 1.0)
    for b in (0.25, 0.5, 0.9):
        assert np.allclose(sm.arrival(pts, c, n, b), (1 - b) * sphere + b * plane)
    # straight ahead along the normal, every blend reaches a point at the same distance
    ahead = c + np.outer([0.5, 3.0, 7.0], n)
    for b in (0.0, 0.3, 0.7, 1.0):
        assert np.allclose(sm.arrival(ahead, c, n, b), [0.5, 3.0, 7.0])


def test_the_blend_is_a_conic_with_its_focus_at_the_centre():
    """(1 - b)|x| + b x.n = s is a conic of revolution with focus at the centre and
    eccentricity b / (1 - b): its points' distance from the focus is e times their distance
    from the directrix plane x.n = s / b."""
    for b in (0.2, 0.5, 0.75):
        e = b / (1 - b)
        th = np.linspace(0.0, 1.2, 9)
        rho = 4.0 / ((1 - b) + b * np.cos(th))
        x = np.c_[rho * np.sin(th), np.zeros_like(th), rho * np.cos(th)]
        assert np.allclose(sm.arrival(x, (0, 0, 0), (0, 0, 1), b), 4.0)
        directrix = 4.0 / b
        assert np.allclose(np.linalg.norm(x, axis=1), e * (directrix - x[:, 2]))


def test_arrival_times_of_the_mirror_lattice_are_the_echo_times():
    """The sphere round the listener reaches each image source when its echo arrives:
    exactly ``ac_math.arrivals`` (distance over the speed of sound)."""
    pos, value = sm.mirror_box(SIZE, SRC, 5)
    t = sm.arrival(pos, MIC, (0, 0, 1), 0.0) / ac_math.C
    arr = ac_math.arrivals(SIZE, SRC, MIC, 5, 0.0)
    assert np.allclose(np.sort(t), arr["time"])
    assert sorted(value.tolist()) == sorted(arr["reflections"].tolist())


# ---------------------------------------------------------------- the rules


def test_the_mirror_rule_is_the_image_source_method():
    pos, value = sm.mirror_box(SIZE, SRC, 1)
    assert np.allclose(pos[0], SRC) and value[0] == 0
    ghosts = {tuple(np.round(p, 9)) for p in pos[1:]}
    assert ghosts == {(-2.2, 2.0, 1.4), (13.8, 2.0, 1.4), (2.2, -2.0, 1.4), (2.2, 8.0, 1.4),
                      (2.2, 2.0, -1.4), (2.2, 2.0, 4.6)}  # fmt: skip
    assert (value[1:] == 1).all()


@pytest.mark.parametrize("order", [0, 1, 2, 7, 20])
def test_image_counts_match_the_closed_form(order):
    pos, value = sm.mirror_box(SIZE, SRC, order)
    assert len(pos) == ac_math.image_count(order)
    assert value.max(initial=0) == order
    # every image with k reflections: k = sum over axes of |the mirror index|
    for k in range(order + 1):
        assert (value == k).sum() == (1 if k == 0 else 4 * k * k + 2)


def test_fifty_eight_bounces_is_fast_and_exact():
    import time

    t0 = time.perf_counter()
    pos, value = sm.mirror_box(SIZE, SRC, 58)
    assert time.perf_counter() - t0 < 2.0
    assert len(pos) == ac_math.image_count(58) == 267_033
    # each image's reflections equal the walls crossed by the straight line to the source
    pick = pos[::997]
    cells = np.floor(pick / np.asarray(SIZE)).astype(int)
    assert np.array_equal(np.abs(cells).sum(axis=1), value[::997])


def test_the_grid_rule_counts_steps():
    pos, value = sm.grid(((1, 0, 0), (0, 2, 0), (0, 0, 1)), (3, 2, 1), (1, 1, 1))
    assert len(pos) == 6
    assert np.allclose(pos[value == 0], [[1, 1, 1]])
    assert {tuple(p) for p in pos[value == 3]} == {(3.0, 3.0, 1.0)}
    assert sorted(value.tolist()) == [0, 1, 1, 2, 2, 3]


# ---------------------------------------------------------------- the flash


def test_the_flash_rises_peaks_and_settles_to_the_afterglow():
    age = np.array([-1.0, 0.0, 0.01, 0.02, 0.1, 0.5, 5.0])
    light, flash = sm.envelope(age, attack=0.02, decay=0.2, before=0.1, after=0.3)
    assert light[0] == pytest.approx(0.1) and flash[0] == 0.0  # not yet
    assert light[1] == pytest.approx(0.1) and flash[1] == 0.0  # continuous at arrival
    assert flash[3] == pytest.approx(np.exp(-0.1))  # top of the attack
    assert np.all(np.diff(flash[3:]) < 0)  # then it decays
    assert light[-1] == pytest.approx(0.3, abs=1e-9)  # to the afterglow
    instant, f = sm.envelope(np.array([0.0]), attack=0.0, decay=0.2, before=0.0, after=0.3)
    assert instant[0] == pytest.approx(1.0) and f[0] == pytest.approx(1.0)


def test_rate_counts_arrivals_per_second():
    s = np.array([1.0, 2.0, 2.5, 2.9, 3.5, 9.0])
    assert sm.rate(s, reach=2.0, speed=1.0, window=1.0) == pytest.approx(3.0)  # 2, 2.5, 2.9
    assert sm.rate(s, reach=2.0, speed=2.0, window=0.5) == pytest.approx(6.0)  # same 3, in 0.5 s


def test_echo_rate_grows_as_time_squared():
    """Echo density in a room grows like t^2 (4 pi c^3 t^2 / V per second): the sphere's
    area. The lattice shows it without a formula."""
    pos, _ = sm.mirror_box(SIZE, SRC, 40)
    s = sm.arrival(pos, MIC, (0, 0, 1), 0.0)
    vol = float(np.prod(SIZE))
    for r in (30.0, 60.0, 90.0):  # inside what order 40 covers in every direction
        got = sm.rate(s, reach=r, speed=1.0, window=4.0)
        assert got == pytest.approx(4 * np.pi * (r + 2) ** 2 / vol, rel=0.08)


# ---------------------------------------------------------------- the front's mesh


def test_the_front_mesh_lies_on_the_surface_and_inside_the_extent():
    for b in (0.0, 0.4, 0.5, 0.8, 1.0):
        mesh = sm.front_mesh((1, 2, 3), (0, 1, 0), b, reach=4.0, extent=12.0)
        assert mesh is not None
        pos, normals, tris, edge = mesh
        assert np.allclose(sm.arrival(pos, (1, 2, 3), (0, 1, 0), b), 4.0)
        assert (np.linalg.norm(pos - (1, 2, 3), axis=1) <= 12.0 + 1e-9).all()
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
        assert tris.max() < len(pos) and edge.shape == (len(pos),)
        assert edge.min() >= 0.0 and edge.max() <= 1.0


def test_the_whole_sphere_and_a_plane_disc():
    pos, normals, _, _ = sm.front_mesh((0, 0, 0), (0, 0, 1), 0.0, reach=3.0, extent=10.0)
    assert pos[:, 2].min() == pytest.approx(-3.0) and pos[:, 2].max() == pytest.approx(3.0)
    assert np.allclose(normals, pos / 3.0)  # outward
    pos, normals, _, _ = sm.front_mesh((0, 0, 0), (0, 0, 1), 1.0, reach=3.0, extent=5.0)
    assert np.allclose(pos[:, 2], 3.0) and np.allclose(normals, [0, 0, 1])
    assert np.hypot(pos[:, 0], pos[:, 1]).max() == pytest.approx(4.0)  # sqrt(5^2 - 3^2)


def test_no_front_before_it_starts_or_past_the_extent():
    assert sm.front_mesh((0, 0, 0), (0, 0, 1), 0.0, reach=0.0, extent=5.0) is None
    assert sm.front_mesh((0, 0, 0), (0, 0, 1), 0.0, reach=6.0, extent=5.0) is None
    assert sm.front_mesh((0, 0, 0), (0, 0, 1), 1.0, reach=6.0, extent=5.0) is None


# ---------------------------------------------------------------- the objects


def _scene(**kw):
    s = Scene("sw", passes=None)
    s.use_camera("orbit", azimuth=-60, elevation=25, distance=60, target=MIC)
    params = dict(size=SIZE, source=SRC, order=10, center=MIC, time=expr("t - 1"), speed=10.0)
    params.update(kw)
    s.add(sweep.SweepLattice(id="crystal", **params))
    s.add(sweep.SweepFront(id="front", lattice="crystal"))
    return s.engine


def test_points_flash_when_the_front_reaches_them():
    e = _scene()
    st = e.state_at(3.0)  # 2 s after the start: the front is 20 m out
    sem = st["crystal"].impl.semantics(st)
    pos, _ = sm.mirror_box(SIZE, SRC, 10)
    d = np.linalg.norm(pos - np.asarray(MIC), axis=1)
    assert sem["reach"] == pytest.approx(20.0)
    assert sem["reached"] == int((d <= 20.0).sum())
    assert 0 < sem["flashing"] < sem["reached"]  # the newest arrivals
    assert sem["points"] == len(pos) and sem["kind"] == "sweep_lattice"
    assert sem["next_arrival"] == pytest.approx(float(d[d > 20.0].min() / 10.0 + 1.0), abs=1e-3)
    before = e.state_at(0.5)
    assert before["crystal"].impl.semantics(before)["reached"] == 0


def test_the_lattice_draws_brighter_where_it_flashes():
    e = _scene()
    st = e.state_at(3.0)
    c = _Collector()
    st["crystal"].impl.draw(c, st, "color")
    assert 1 <= len(c.commands) <= 2  # the points, and the halos of those flashing
    core = c.commands[0][1]
    alpha = core["colors"][:, 3]
    assert alpha.max() > 0.9 and alpha.min() < 0.2
    hues = core["colors"][:, :3]
    assert np.ptp(hues, axis=0).max() > 0.1  # colored by bounces


def test_hidden_before_and_culled_to_extent_and_count():
    e = _scene(before=0.0, extent=15.0, max_points=50)
    st = e.state_at(1.5)  # 5 m out
    shown = st["crystal"].impl.shown(st)
    d = np.linalg.norm(shown["pos"] - np.asarray(MIC), axis=1)
    assert (d <= 5.0 + 1e-9).all() and len(d) <= 50
    st = e.state_at(30.0)  # long after: the afterglow of every point within the extent
    shown = st["crystal"].impl.shown(st)
    assert len(shown["pos"]) == 50


def test_the_front_follows_the_lattice_and_glows_at_its_rim():
    e = _scene()
    st = e.state_at(3.0)
    c = _Collector()
    st["front"].impl.draw(c, st, "color")
    ((kind, cmd),) = c.commands
    assert kind == "surface" and cmd["material"] == "emissive" and cmd["opacity"] < 0.999
    d = np.linalg.norm(cmd["positions"] - np.asarray(MIC), axis=1)
    assert np.allclose(d, 20.0)
    cam = st.camera_for(st["front"])
    view = cam.eye - cmd["positions"]
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = np.abs((cmd["normals"] * view).sum(axis=1))
    a = cmd["colors"][:, 3]
    assert a[facing < 0.2].mean() > 3 * a[facing > 0.9].mean()  # brighter at grazing angles
    sem = st["front"].impl.semantics(st)
    assert sem["shape"] == "sphere" and sem["reach"] == pytest.approx(20.0)
    assert np.allclose(sem["apex"], np.add(MIC, (0, 0, 20.0)))


def test_flattening_moves_the_front_into_a_plane():
    e = _scene(flat=1.0, normal=(1, 0, 0))
    st = e.state_at(3.0)
    c = _Collector()
    st["front"].impl.draw(c, st, "color")
    ((_, cmd),) = c.commands
    assert np.allclose(cmd["positions"][:, 0], MIC[0] + 20.0)
    assert st["front"].impl.semantics(st)["shape"] == "plane"
    sem = st["crystal"].impl.semantics(st)
    pos, _ = sm.mirror_box(SIZE, SRC, 10)
    assert sem["reached"] == int(((pos[:, 0] - MIC[0] >= 0) & (pos[:, 0] - MIC[0] <= 20)).sum())


def test_a_front_without_its_lattice_draws_nothing():
    s = Scene("lonely", passes=None)
    s.add(sweep.SweepFront(id="front", lattice="nope"))
    st = s.engine.state_at(0.0)
    c = _Collector()
    st["front"].impl.draw(c, st, "color")
    assert c.commands == []
    assert st["front"].impl.semantics(st)["lattice_missing"] == "nope"


def test_explicit_points_and_values():
    s = Scene("pts", passes=None)
    s.add(sweep.SweepLattice(id="l", rule="points", points=[(1, 0, 0), (2, 0, 0), (3, 0, 0)],
                             values=[0.0, 0.5, 1.0], center=(0, 0, 0), time=2.5, speed=1.0,
                             attack=0.0))  # fmt: skip
    st = s.engine.state_at(0.0)
    sem = st["l"].impl.semantics(st)
    assert sem["points"] == 3 and sem["reached"] == 2
    with pytest.raises(Exception, match="values"):
        s.add(sweep.SweepLattice(id="bad", rule="points", points=[(1, 0, 0)], values=[1.0, 2.0]))


def test_pure_in_time():
    a, b = _scene().state_at(4.2), _scene().state_at(4.2)
    sa, sb = a["crystal"].impl.shown(a), b["crystal"].impl.shown(b)
    assert all(np.array_equal(sa[k], sb[k]) for k in sa)
    e = _scene()
    first = e.state_at(4.2)
    e.state_at(9.0)
    again = e.state_at(4.2)
    assert all(np.array_equal(first["crystal"].impl.shown(first)[k],
                              again["crystal"].impl.shown(again)[k]) for k in sa)  # fmt: skip


def test_the_room_and_listener_sit_inside_the_crystal():
    """The mirror lattice's own room is the order-0 cell, so ``ac`` objects line up with it."""
    s = Scene("both", passes=None)
    s.add(ac.Room3D(id="room", size=SIZE))
    s.add(sweep.SweepLattice(id="crystal", size=SIZE, source=SRC, order=3, center=MIC))
    st = s.engine.state_at(0.0)
    assert np.allclose(st["crystal"].impl.features(st)["source"], SRC)


# ---------------------------------------------------------------- the example


def _example():
    return runpy.run_path(str(ROOT / "examples" / "echo_crystal.py"), run_name="x")["s"].engine


def test_the_example_builds_and_lints_without_errors():
    e = _example()
    assert not [i for i in e.lint() if i.severity == "error"]
    early, late = e.state_at(3.5), e.state_at(e.duration - 0.5)
    assert early["front"].impl.semantics(early)["shape"] == "sphere"
    assert late["front"].impl.semantics(late)["shape"] == "plane"


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_example_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _example()
    path = GOLDEN / e.pass_runner().renderer_class() / "echo_crystal.png"
    sheet = e.storyboard([3.0, 4.5, 6.5, 11.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
