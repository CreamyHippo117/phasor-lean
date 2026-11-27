"""Glowing points that flash on arrival, drawn one way: ``sweep_math.envelope`` (the flash)
and ``strokes.flash_points`` (the drawing), shared by ``sweep.SweepLattice``,
``nd.StarLattice`` and ``ac.ImageSources``; and ``examples/echo_room_to_loops.py``, which
hands the room's sweeping plane over to the loops' grid."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, nd, sweep
from phasor.domains import ac_math
from phasor.domains import ndlattice_math as nlm
from phasor.domains import sweep_math as sm
from phasor.domains.strokes import FLASH_HALO, REST_GLOW, flash_points
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]


def _camera():
    s = Scene("cam", passes=None)
    s.use_camera("orbit", azimuth=-30, elevation=20, distance=10, target=(0, 0, 0))
    s.add(ac.Listener(id="x", pos=(0, 0, 0)))
    st = s.engine.state_at(0.0)
    return st.camera_for(st["x"])


def _discs(cmd, segments):
    """(centres, radii, per-disc RGBA at the centre) of a ``disc_batch`` mesh."""
    pos = cmd["positions"].reshape(-1, segments + 1, 3)
    cols = cmd["colors"].reshape(-1, segments + 1, 4)
    return pos[:, 0], np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1), cols[:, 0], cols[:, 1]


# ---------------------------------------------------------------- the shared pieces


def test_steady_points_are_one_call_with_the_whole_glow():
    cam = _camera()
    c = _Collector()
    colors = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 1.0]])
    flash_points(c, cam, [(0, 0, 0), (1, 0, 0)], 0.1, colors=colors, glow=6.0, segments=8)
    ((kind, cmd),) = c.commands
    assert kind == "mesh" and cmd["glow"] == 6.0
    centres, radii, core, _ = _discs(cmd, 8)
    np.testing.assert_allclose(radii, 0.1)
    for p, col in zip([(0, 0, 0), (1, 0, 0)], colors, strict=True):  # in some depth order
        np.testing.assert_allclose(core[np.flatnonzero((centres == p).all(axis=1))[0]], col)


def test_a_flash_swells_runs_white_hot_and_wears_a_halo():
    cam = _camera()
    c = _Collector()
    colors = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    flash_points(c, cam, [(0, 0, 0), (0, 0, 0.5)], 0.1, colors=colors, flash=[0.0, 1.0],
                 glow=6.0, swell=1.0, white=0.5, segments=8)  # fmt: skip
    (_, core), (_, halo) = c.commands
    assert core["glow"] == pytest.approx(1.0 + REST_GLOW * 5.0) and halo["glow"] == 6.0
    centres, radii, col, _ = _discs(core, 8)
    hot = np.flatnonzero(centres[:, 2] == 0.5)[0]
    assert radii[hot] == pytest.approx(0.2) and radii[1 - hot] == pytest.approx(0.1)
    np.testing.assert_allclose(col[hot, :3], [1.0, 0.5, 0.5])  # half-way to white
    np.testing.assert_allclose(col[1 - hot, :3], [1.0, 0.0, 0.0])
    hc, hr, mid, rim = _discs(halo, 10)
    assert len(hc) == 1 and hc[0][2] == 0.5 and hr[0] == pytest.approx(FLASH_HALO * 0.2)
    assert mid[0, 3] > 0 and rim[0, 3] == 0  # a soft splat


def test_points_are_drawn_far_to_near():
    cam = _camera()
    pts = np.array([np.asarray(cam.forward) * d for d in (-3.0, 4.0, 0.0)])  # round the target
    c = _Collector()
    flash_points(c, cam, pts, 0.1, colors=(1.0, 1.0, 1.0, 1.0), segments=8)
    centres, *_ = _discs(c.commands[0][1], 8)
    assert np.all(np.diff(cam.depth(centres)) < 0)


def test_the_star_flash_is_the_shared_envelope():
    tau = np.linspace(-0.5, 3.0, 701)
    light, burst = nlm.flash(tau, rise=0.05, decay=0.6, afterglow=0.3)
    ref_light, ref_flash = sm.envelope(tau, attack=0.05, decay=0.6, before=0.0, after=0.3)
    np.testing.assert_array_equal(light, ref_light)
    np.testing.assert_array_equal(burst, ref_flash)
    assert nlm.smoothstep is sm.smoothstep


def test_a_star_and_a_swept_lattice_look_the_same():
    """One loop's echoes and the same points swept by a sphere: the same flash, drawn the
    same way, call for call."""
    a = Scene("star", passes=None)
    a.use_camera("orbit", target=(1, 0, 0), distance=6)
    a.add(nd.StarLattice(id="l", delays=(1.0,), directions=((1.0, 0.0, 0.0),), max_time=3.0,
                         time=1.1, rise=0.05, decay=0.4, afterglow=0.2, swell=1.2, white=0.6,
                         arm_opacity=0.0, crowd=1e6, color_by="arrival", hue_span=0.3,
                         style={"glow": "strong"}))  # fmt: skip
    b = Scene("swept", passes=None)
    b.use_camera("orbit", target=(1, 0, 0), distance=6)
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    b.add(sweep.SweepLattice(id="l", rule="points", points=pts, values=(0, 1 / 3, 2 / 3, 1),
                             center=(0, 0, 0), speed=1.0, time=1.1, attack=0.05, decay=0.4,
                             before=0.0, after=0.2, swell=1.2, white=0.6, hue_step=0.3,
                             r=1e-6, min_size=0.005, style={"glow": "strong"}))  # fmt: skip
    out = []
    for s in (a, b):
        st = s.engine.state_at(0.0)
        c = _Collector()
        st["l"].impl.draw(c, st, "color")
        out.append(c.commands)
    star, swept = out
    assert len(star) == len(swept) == 2  # the points, then the halo of the one flashing
    for (_, x), (_, y) in zip(star, swept, strict=True):
        assert x["glow"] == y["glow"]
        np.testing.assert_allclose(x["positions"], y["positions"], atol=1e-9)
        np.testing.assert_allclose(x["colors"], y["colors"], atol=1e-9)


def test_image_sources_draw_as_steady_lights_past_the_old_order_cap():
    s = Scene("ghosts", passes=None)
    s.use_camera("orbit", target=(4, 2.5, 1.5), distance=200)
    s.add(ac.ImageSources(id="g", order=20, material="carpet", max_points=500))
    st = s.engine.state_at(0.0)
    sem = st["g"].impl.semantics(st)
    assert sem["count"] == ac_math.image_count(20) - 1 and sem["shown"] == 500
    c = _Collector()
    st["g"].impl.draw(c, st, "color")
    ((_, cmd),) = c.commands  # one call, however many
    centres, radii, core, _ = _discs(cmd, 16)
    im = ac_math.image_sources((8.0, 5.0, 3.0), (2.0, 2.5, 1.5), 20, 0.30)
    assert len(centres) == 500 and set(map(tuple, centres)) == set(map(tuple, im["pos"][1:501]))
    gain = {tuple(p): g for p, g in zip(im["pos"], im["gain"], strict=True)}
    expect = np.array([gain[tuple(p)] for p in centres]) * st["g"].style["color"][3]
    np.testing.assert_allclose(core[:, 3], expect)
    np.testing.assert_allclose(radii, 0.09)
    with pytest.raises(Exception, match="order"):
        s.add(ac.ImageSources(id="too_deep", order=ac_math.MAX_ORDER + 1))


# ---------------------------------------------------------------- the example


def _example():
    return runpy.run_path(str(ROOT / "examples" / "echo_room_to_loops.py"), run_name="x")


def test_the_plane_lights_the_grid_when_its_echoes_arrive():
    """After the handover the room's plane sweeps on through three loops' grid: every grid
    echo lies on the plane at the moment it arrives, so the two clocks agree."""
    ns = _example()
    e = ns["s"].engine
    assert not [i for i in e.lint() if i.severity == "error"]
    st = e.state_at(10.0)
    assert st["front"].impl.semantics(st)["shape"] == "plane"
    star = st["loops"].impl
    vis = star.visible(st)
    assert st["loops"].impl.semantics(st)["arms_shown"] == 3 and len(vis["points"]) > 20
    s_dist = sm.arrival(vis["points"], ns["MIC"], ns["NORMAL"], 1.0)
    np.testing.assert_allclose(s_dist, ns["SPEED"] * (vis["arrival"] + ns["HANDOVER"]
                                                      - ns["CLAP"]), atol=1e-9)  # fmt: skip
    reach = st["crystal"].impl.reach(st)
    heard = st["loops"].impl.semantics(st)["heard"]
    assert heard == int((s_dist <= reach + 1e-9).sum())  # lit exactly when the plane passes
    end = e.state_at(e.duration)
    sem = end["loops"].impl.semantics(end)
    assert sem["arms_shown"] == 16 and sem["heard"] > 100 * heard


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_example_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _example()["s"].engine
    path = GOLDEN / e.pass_runner().renderer_class() / "echo_room_to_loops.png"
    sheet = e.storyboard([3.0, 6.0, 8.5, 10.5, 13.0, 16.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
