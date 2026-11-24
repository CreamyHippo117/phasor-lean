"""A 3D room with sound bouncing in it: ray tracing, mesh rooms read from
.obj files (and the ``Files`` mechanism that keeps them in the scene's folder), the drawn
rays and fronts, and the example's golden storyboard."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, expr
from phasor.core import files
from phasor.core.issues import DocumentError
from phasor.core.objects import ObjectBase, ParamsBase, RegistrationError, register
from phasor.domains import ac_math
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
L_ROOM = ROOT / "examples" / "assets" / "l_room.obj"
SIZE, SRC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4)
BOX = ac_math.box_mesh(SIZE)


# ---------------------------------------------------------------- the ray tracer


def _reflections(paths, r):
    """(incoming direction, outgoing direction, wall normal) at every reflection of ray r."""
    v, f = BOX
    pts = paths["points"][r]
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    for k in range(1, len(pts) - 1):
        d_in, d_out = pts[k] - pts[k - 1], pts[k + 1] - pts[k]
        yield (d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out),
               n[paths["wall"][r, k]])  # fmt: skip


def test_rays_reflect_with_angle_in_equal_to_angle_out():
    paths = ac_math.rays(BOX, SRC, 64, 6, 0.1, seed=3, key="rays")
    assert paths["alive"].all()  # a closed room loses no ray
    count = 0
    for r in range(64):
        for d_in, d_out, n in _reflections(paths, r):
            assert d_out @ n == pytest.approx(-(d_in @ n), abs=1e-9)  # normal part flips
            assert np.allclose(d_out - (d_out @ n) * n, d_in - (d_in @ n) * n, atol=1e-9)
            count += 1
    assert count == 64 * 5


def test_every_reflection_point_is_on_a_wall_and_the_path_length_adds_up():
    paths = ac_math.rays(BOX, SRC, 200, 8, 0.1, key="rays")
    pts = paths["points"][:, 1:]
    on_wall = np.isclose(pts, 0.0, atol=1e-9) | np.isclose(pts, np.asarray(SIZE), atol=1e-9)
    assert on_wall.any(axis=2).all()
    steps = np.linalg.norm(np.diff(paths["points"], axis=1), axis=2)
    assert np.allclose(np.cumsum(steps, axis=1), paths["length"][:, 1:])


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5])
def test_ray_energy_is_one_minus_alpha_to_the_number_of_bounces(alpha):
    paths = ac_math.rays(BOX, SRC, 50, 7, alpha, key="rays")
    expected = (1.0 - alpha) ** np.arange(8)
    assert np.allclose(paths["energy"], expected[None, :])


def _position_at(paths, r, d):
    """Where ray r is after travelling d meters, and how many reflections it took."""
    length, pts = paths["length"][r], paths["points"][r]
    k = int(np.searchsorted(length, d, side="right")) - 1
    u = (d - length[k]) / (length[k + 1] - length[k])
    return pts[k] + u * (pts[k + 1] - pts[k]), k


def test_first_arrivals_follow_the_image_sources():
    """A ray from the listener aimed at an image source reaches the real source after
    exactly as many reflections as the image has, at the image's distance: the ray tracer
    and the image-source method agree."""
    mic = np.array([6.0, 3.4, 1.4])
    im = ac_math.image_sources(SIZE, SRC, 2, 0.1)
    checked = 0
    for pos, refl in zip(im["pos"], im["reflections"], strict=True):
        dist = float(np.linalg.norm(pos - mic))
        paths = ac_math.trace(*BOX, mic[None], (pos - mic)[None], 3, 0.1)
        hits = paths["points"][0, 1 : refl + 1]
        on_edge = (np.isclose(hits, 0, atol=1e-6) | np.isclose(hits, SIZE, atol=1e-6)).sum(1)
        if (on_edge > 1).any():
            continue  # through an edge or a corner: no single wall, skip
        at, k = _position_at(paths, 0, dist - 1e-9)
        assert np.allclose(at, SRC, atol=1e-6)
        assert k == refl
        checked += 1
    assert checked >= 20


def test_rays_are_deterministic_and_their_randomness_is_the_objects():
    a = ac_math.rays(BOX, SRC, 100, 4, 0.1, seed=0, key="rays")
    b = ac_math.rays(BOX, SRC, 100, 4, 0.1, seed=0, key="rays")
    assert np.array_equal(a["points"], b["points"])
    c = ac_math.rays(BOX, SRC, 100, 4, 0.1, seed=1, key="rays")
    d = ac_math.rays(BOX, SRC, 100, 4, 0.1, seed=0, key="other")
    assert not np.allclose(a["dirs"], c["dirs"]) and not np.allclose(a["dirs"], d["dirs"])
    assert np.allclose(np.linalg.norm(a["dirs"], axis=1), 1.0)
    assert np.abs(a["dirs"].mean(axis=0)).max() < 0.02  # evenly in every direction


def test_drawn_pieces_end_where_the_sound_has_got_to():
    paths = ac_math.rays(BOX, SRC, 300, 8, 0.1, key="rays")
    for d in (0.5, 3.0, 11.0):
        whole = ac_math.ray_segments(paths, d)
        assert whole["along"].max() == pytest.approx(d)
        assert whole["along"].min() == 0.0
        heads = whole["along"][:, 1] >= d - 1e-9
        assert heads.sum() == 300  # every ray is d meters along its path
        head_pts = whole["segments"][heads, 1]
        at = np.array([_position_at(paths, r, d)[0] for r in whole["ray"][heads]])
        assert np.allclose(head_pts, at)
        streak = ac_math.ray_segments(paths, d, trail=1.0)
        assert streak["along"].min() >= d - 1.0 - 1e-9
        assert np.allclose(np.linalg.norm(np.diff(streak["segments"], axis=1), axis=2)[:, 0],
                           streak["along"][:, 1] - streak["along"][:, 0])  # fmt: skip


# ---------------------------------------------------------------- meshes from .obj


def test_the_obj_loader_reads_polygons_slashes_and_negative_indices(tmp_path):
    p = tmp_path / "cube.obj"
    p.write_text(
        "# a unit cube, written every way the format allows\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
        "vt 0 0\nvn 0 0 1\ng cube\nusemtl wall\n"
        "f 1 4 3 2\nf 5/1 6/1 7/1 8/1\nf 1//1 2//1 6//1 5//1\n"
        "f -6/1/1 -5/1/1 -1/1/1 -2/1/1\nf 3 4 8 7\nf 4 1 5 8\n",
        encoding="utf-8",
    )
    v, f = ac_math.load_obj(p)
    assert v.shape == (8, 3) and f.shape == (12, 3)
    assert ac_math.mesh_stats(v, f) == pytest.approx((1.0, 6.0))
    assert len(ac_math.feature_edges(v, f)) == 12  # the cube's edges, not the diagonals
    p.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not in the file"):
        ac_math.load_obj(p)


def test_the_l_shaped_room():
    v, f = ac_math.load_obj(L_ROOM)
    assert ac_math.mesh_stats(v, f) == pytest.approx((108.0, 156.0))
    assert len(ac_math.feature_edges(v, f)) == 22  # 8 floor, 8 ceiling, 6 upright corners
    assert len(ac_math.planes(v, f)) == 8
    assert list(ac_math.inside(v, f, [[2, 1, 1], [6, 5, 1], [2, 5, 1], [6, 1, 1]])) == [
        True, False, True, True]  # fmt: skip
    paths = ac_math.rays((v, f), (2.0, 1.5, 1.5), 500, 10, 0.1, key="rays")
    assert paths["alive"].all()
    # every piece of every path stays in the room (a midpoint test, nudged off the walls)
    mids = (paths["points"][:, 1:] + paths["points"][:, :-1]) / 2
    assert ac_math.inside(v, f, mids.reshape(-1, 3)).all()


def test_paths_stay_in_their_folder(tmp_path):
    assert files.relative_path("assets\\room.obj") == "assets/room.obj"
    for bad in ("/etc/passwd", "C:\\room.obj", "../room.obj", "a/../../b.obj", ""):
        with pytest.raises(ValueError):
            files.relative_path(bad)
    (tmp_path / "a.obj").write_text("", encoding="utf-8")
    assert files.exists(tmp_path, "a.obj") and not files.exists(tmp_path, "b.obj")
    s = Scene("r", passes=None, base_dir=tmp_path)
    with pytest.raises(DocumentError, match="folder"):
        s.add(ac.Room3D(id="room", path="../outside.obj"))


def test_a_type_can_only_list_its_own_parameters_as_files():
    class Bad(ObjectBase):
        type = "bad_files"
        Files = ("nope",)

        class Params(ParamsBase):
            path: str = ""

        def world_bounds(self, state): ...
        def features(self, state): ...
        def semantics(self, state): ...
        def draw(self, ctx, state, pass_): ...

    with pytest.raises(RegistrationError, match="nope"):
        register(Bad)


MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


@pytest.mark.parametrize("passes", MODES)
def test_a_missing_mesh_is_missing_file_and_draws_nothing(tmp_path, passes):
    s = Scene("r", passes=passes, base_dir=tmp_path)
    s.use_camera("orbit", target=(4, 3, 1.5), distance=16)
    s.add(ac.Room3D(id="room", path="rooms/l_room.obj"))
    s.add(ac.SoundRays(id="rays", room="room", source=(2, 1.5, 1.5), time=1.0, count=50))
    e = s.engine
    (issue,) = [i for i in e.lint() if i.code == "missing_file"]
    assert issue.ids == ("room",) and issue.path == "room.path" and issue.times == (0.0,)
    st = e.state_at(0.0)
    c = _Collector()
    st["room"].impl.draw(c, st, "color")
    st["rays"].impl.draw(c, st, "color")
    assert c.commands == []
    assert st["room"].impl.semantics(st)["missing_file"] == "rooms/l_room.obj"
    assert st["rays"].impl.semantics(st)["room_missing"] == "room"
    (tmp_path / "rooms").mkdir()
    (tmp_path / "rooms" / "l_room.obj").write_bytes(L_ROOM.read_bytes())
    e2 = Scene("r", passes=passes, base_dir=tmp_path)
    e2.add(ac.Room3D(id="room", path="rooms/l_room.obj"))
    assert not [i for i in e2.engine.lint() if i.code == "missing_file"]
    st = e2.engine.state_at(0.0)
    assert st["room"].impl.semantics(st)["volume"] == pytest.approx(108.0)


# ---------------------------------------------------------------- the objects


def _scene(t_sound: float, **room):
    s = Scene("r", look="prism", passes=None, base_dir=ROOT / "examples")
    s.use_camera("orbit", target=(4, 2.5, 1.5), distance=15)
    s.add(ac.Room3D(id="room", **room))
    s.add(ac.SoundRays(id="rays", room="room", source=SRC, time=expr(f"t * {t_sound}"),
                       speed=1.0, count=400, bounces=8, trail=0.0))  # fmt: skip
    s.add(ac.Wavefront3D(id="front", room="room", source=SRC, time=expr(f"t * {t_sound}"),
                         speed=1.0))  # fmt: skip
    return s.engine


def test_sound_rays_travel_at_the_speed_of_sound():
    e = _scene(0.01)
    st = e.state_at(1.0)
    rays = st["rays"].impl
    assert rays.distance(st) == pytest.approx(3.43)
    vis = rays.visible(st)
    assert vis["along"].max() == pytest.approx(3.43)
    sem = rays.semantics(st)
    assert sem["distance"] == pytest.approx(3.43) and sem["sound_time"] == pytest.approx(0.01)
    late = e.state_at(4.0)
    assert late["rays"].impl.semantics(late)["reflections_so_far"] > 2
    assert late["rays"].impl.semantics(late)["energy_left"] < 0.9


def test_rays_and_fronts_draw_one_batch_each_colored_by_reflections():
    e = _scene(0.01)
    st = e.state_at(2.0)
    for oid in ("rays", "front"):
        c = _Collector()
        st[oid].impl.draw(c, st, "color")
        assert 1 <= len(c.commands) <= 2  # the walls' circles, the faint shells
        colors = np.concatenate([cmd["colors"] for _, cmd in c.commands])
        assert np.ptp(colors[:, :3], axis=0).max() > 0.1  # more than one hue
        assert colors[:, 3].max() <= 1.0
    before = e.state_at(0.0)
    c = _Collector()
    before["front"].impl.draw(c, before, "color")
    before["rays"].impl.draw(c, before, "color")
    assert c.commands == []  # nothing before the clap


def test_front_circles_lie_on_the_walls_at_the_fronts_radius():
    v, f = BOX
    im = ac_math.image_sources(SIZE, SRC, 2, 0.1)
    segs, who = ac_math.sphere_wall_arcs(v, f, im["pos"], 6.0, samples=256)
    pts = segs.reshape(-1, 3)
    on_wall = np.isclose(pts, 0.0, atol=1e-9) | np.isclose(pts, np.asarray(SIZE), atol=1e-9)
    assert on_wall.any(axis=1).all()
    assert (pts >= -1e-9).all() and (pts <= np.asarray(SIZE) + 1e-9).all()
    r = np.linalg.norm(segs - im["pos"][who][:, None], axis=2)
    assert np.abs(r - 6.0).max() < 6.0 * (1 - np.cos(np.pi / 256)) + 1e-6  # chord sag only


def test_in_an_l_shaped_room_fronts_only_reach_what_the_source_can_see():
    v, f = ac_math.load_obj(L_ROOM)
    walls = ac_math.planes(v, f)
    src = np.array([6.5, 1.5, 1.5])  # in the hall, far end: the wing is round the corner
    segs, _ = ac_math.sphere_wall_arcs(v, f, src[None], 6.5, groups=walls)
    mid = segs.mean(axis=1)
    seen = ac_math.front_visible(v, f, walls, src, np.full(len(mid), -1), mid)
    assert seen.any() and not seen.all()
    hidden = mid[~seen]
    assert (hidden[:, 1] > 3.0 - 1e-6).all()  # only the wing is hidden from the hall
    e = _scene(0.02, path="assets/l_room.obj")
    st = e.state_at(1.0)
    assert st["front"].impl.semantics(st)["fronts"] == 1 + 8  # direct + one image per wall


def test_the_same_frame_is_the_same_scene():
    a, b = _scene(0.01).state_at(3.0), _scene(0.01).state_at(3.0)
    va, vb = a["rays"].impl.visible(a), b["rays"].impl.visible(b)
    assert np.array_equal(va["segments"], vb["segments"])
    assert a["front"].impl.semantics(a) == b["front"].impl.semantics(b)


def test_the_prism_look_colors_rays_from_the_palette():
    s = Scene("r", look="prism", palette="gold", passes=None)
    s.add(ac.Room3D(id="room"))
    s.add(ac.SoundRays(id="rays", room="room"))
    assert s.engine.document.objects[-1].style["colormap"] == "gold"


# ---------------------------------------------------------------- the example


def _example():
    return runpy.run_path(str(ROOT / "examples" / "room_3d.py"), run_name="room_3d")["s"].engine


def test_the_example_builds_and_lints_without_errors():
    e = _example()
    assert not [i for i in e.lint() if i.severity == "error"]
    st = e.state_at(11.0)
    assert st["rays"].impl.semantics(st)["reflections_so_far"] > 8


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_example_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _example()
    path = GOLDEN / e.pass_runner().renderer_class() / "room_3d.png"
    sheet = e.storyboard([2.0, 3.5, 6.0, 10.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    # glass walls differ by a float16 step from run to run on a GPU: a tolerance, not identity
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
