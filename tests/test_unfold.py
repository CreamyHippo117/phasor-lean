"""Mirror unfolding: the ``unfold`` verb, the ``HingedCopy`` object and their maths
(``core/unfold_math.py``), checked against answers worked out by hand and against the
image-source model of ``ac_math``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phasor.core import unfold_math as um
from phasor.domains import ac_math

ROOT = Path(__file__).parents[1]
UNIT = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))  # a unit square at z = 0 (2D)


# ---------------------------------------------------------------- the maths, by hand


def test_a_ray_in_a_unit_square_crosses_the_walls_worked_out_by_hand():
    # From the middle along (2, 1): x walls at t = 0.25, 0.75, 1.25, y walls at 0.5, 1.5
    # (t in units of (2, 1)), so east, north, east, east, north in the unfolded frame.
    u = um.box_unfold(*UNIT, (0.5, 0.5, 0.0), (2.0, 1.0, 0.0), 5)
    unit = np.linalg.norm([2.0, 1.0])
    assert [c.axis for c in u.crossings] == [0, 1, 0, 0, 1]
    assert [c.side for c in u.crossings] == [1, 1, 1, 1, 1]
    assert [c.cell_to[:2] for c in u.crossings] == [(1, 0), (1, 1), (2, 1), (3, 1), (3, 2)]
    np.testing.assert_allclose([c.distance for c in u.crossings],
                               np.array([0.25, 0.5, 0.75, 1.25, 1.5]) * unit)  # fmt: skip
    # In the real room it bounces east, north, west, east, south; the third at (2, 1.25)
    # unfolded, which folds back to (0, 0.75).
    assert [c.wall for c in u.crossings] == ["east", "north", "west", "east", "south"]
    np.testing.assert_allclose(u.crossings[2].wall_point, (0.0, 0.75, 0.0), atol=1e-12)
    np.testing.assert_allclose(u.crossings[4].wall_point, (0.5, 0.0, 0.0), atol=1e-12)
    # It leaves the last copy, (3, 2), through its east wall at x = 4, t = 1.75.
    assert u.end == pytest.approx(1.75 * unit)
    np.testing.assert_allclose(u.end_point, (4.0, 2.25, 0.0), atol=1e-12)


def test_the_real_path_obeys_angle_in_equals_angle_out():
    lo, hi = (0.0, 0.0, 0.0), (3.0, 2.0, 1.5)
    u = um.box_unfold(lo, hi, (0.7, 0.4, 0.3), (1.0, 0.63, 0.41), 9)
    path = um.zigzag(u, lo, hi)
    assert np.all(path >= -1e-9) and np.all(path <= np.asarray(hi) + 1e-9)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    # Folding is length-preserving: the zig-zag is exactly as long as the straight ray.
    assert lengths.sum() == pytest.approx(u.end)
    for k, c in enumerate(u.crossings):
        a, b = path[k + 1] - path[k], path[k + 2] - path[k + 1]
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        n = np.zeros(3)
        n[c.axis] = 1.0
        np.testing.assert_allclose(b, a - 2 * (a @ n) * n, atol=1e-9)  # mirror reflection


def test_the_unfolded_end_is_an_image_source():
    """The straight line from the source to a point of a mirror copy is the echo path to
    that point's image: the copy of the listener is the listener's image source."""
    lo, hi = (0.0, 0.0, 0.0), (8.0, 5.0, 3.0)
    src = (2.0, 2.5, 1.5)
    u = um.box_unfold(lo, hi, src, (1.0, 0.37, 0.0), 5)
    cell = u.crossings[-1].cell_to
    inside = np.asarray(src) + (u.crossings[-1].distance + 0.5) * np.asarray(u.direction)
    listener = um.fold(lo, hi, inside[None])[0]
    np.testing.assert_allclose(um.image_point(lo, hi, listener, cell), inside, atol=1e-9)
    # ... and that image is one of ac_math's image sources of order 5.
    images = ac_math.image_sources(hi, listener, 5, 0.0)["pos"]
    assert np.min(np.linalg.norm(images - inside, axis=1)) < 1e-9


def test_half_a_turn_about_the_wall_edge_is_the_mirror_in_2d():
    u = um.box_unfold(*UNIT, (0.3, 0.6, 0.0), (1.0, -0.45, 0.0), 5)
    for chain, c in zip(u.chains(), u.crossings, strict=True):
        m = um.chain_matrix(chain)
        for q in ((0.3, 0.6, 0.0), (0.9, 0.1, 0.0), (0.0, 1.0, 0.0)):
            np.testing.assert_allclose(um.apply(m, q), um.image_point(*UNIT, q, c.cell_to),
                                       atol=1e-12)  # fmt: skip


def test_half_a_turn_lands_a_box_on_its_mirror_copy_in_3d():
    lo, hi = np.zeros(3), np.array([4.0, 3.0, 2.0])
    u = um.box_unfold(lo, hi, (1.0, 1.0, 1.0), (0.8, 0.5, 0.35), 6)
    assert {c.axis for c in u.crossings} == {0, 1, 2}  # walls, floor and ceiling
    corners = np.array([[x, y, z] for x in (0, 4) for y in (0, 3) for z in (0, 2)], float)
    for chain, c in zip(u.chains(), u.crossings, strict=True):
        moved = um.apply(um.chain_matrix(chain), corners)
        want = np.asarray(c.cell_to) * (hi - lo)
        np.testing.assert_allclose(moved.min(0), lo + want, atol=1e-9)
        np.testing.assert_allclose(moved.max(0), hi + want, atol=1e-9)
        # a turn, never a reflection: the copy keeps its handedness
        assert np.linalg.det(um.chain_matrix(chain)[:3, :3]) == pytest.approx(1.0)


def test_a_page_turns_up_out_of_the_wall():
    u = um.box_unfold(*UNIT, (0.5, 0.5, 0.0), (1.0, 0.2, 0.0), 1)
    (c,) = u.crossings
    half = um.apply(um.chain_matrix([c.hinge], 0.5), (0.5, 0.5, 0.0))
    np.testing.assert_allclose(half, (1.0, 0.5, 0.5), atol=1e-12)  # standing up, +z
    np.testing.assert_allclose(um.apply(um.chain_matrix([c.hinge], 0.0), (0.2, 0.3, 0)),
                               (0.2, 0.3, 0), atol=1e-12)  # fmt: skip
    for wall, d in (("west", (-1, 0.1, 0)), ("north", (0.1, 1, 0)), ("south", (0.1, -1, 0))):
        (c,) = um.box_unfold(*UNIT, (0.5, 0.5, 0.0), d, 1).crossings
        assert c.wall == wall
        assert um.apply(um.chain_matrix([c.hinge], 0.5), (0.5, 0.5, 0.0))[2] > 0.49


def test_a_corner_is_two_crossings_at_once():
    u = um.box_unfold(*UNIT, (0.5, 0.5, 0.0), (1.0, 1.0, 0.0), 3)
    assert [c.axis for c in u.crossings] == [0, 1, 0]
    assert u.crossings[0].distance == pytest.approx(u.crossings[1].distance)


def test_bad_rays_are_refused():
    with pytest.raises(ValueError, match="direction"):
        um.box_unfold(*UNIT, (0.5, 0.5, 0), (0, 0, 0), 2)
    with pytest.raises(ValueError, match="flat room"):
        um.box_unfold(*UNIT, (0.5, 0.5, 0), (1, 0, 0.2), 2)
    with pytest.raises(ValueError, match="inside"):
        um.box_unfold(*UNIT, (1.5, 0.5, 0), (1, 0, 0), 2)


def test_cells_are_named_by_where_they_are():
    assert um.cell_name((2, -1, 0)) == "e2_s1"
    assert um.cell_name((0, 0, -3)) == "d3"
    assert um.cell_name((0, 0, 0)) == "home"


# ---------------------------------------------------------------- the verb and the copies

ROOM = (4.0, 2.5, 2.0)
SRC = (0.8, 0.6, 1.0)
AIM = (1.0, 0.45, 0.0)


def _scene(dims="2d", extra=None, flips=5):
    from phasor import Line, Scene, ac

    s = Scene("unfold", passes=None)
    if dims == "3d":
        s.use_camera("orbit", azimuth=-90, elevation=50, distance=20, target=(8, 4, 1))
        s.add(ac.Room3D(id="room", size=ROOM, preset="outline"))
    else:
        s.add(ac.Room(id="room", size=ROOM))
    s.add(ac.Source(id="src", pos=SRC))
    s.add(Line(id="ray", points=[SRC, tuple(a + b for a, b in zip(SRC, AIM, strict=True))]))
    s.marker("flip", 1.0)
    if extra == "before":
        s.at("flip").dim_others("room", "ray")
    s.at("flip").unfold("room", ray="ray", flips=flips)
    if extra == "after":
        s.at("flip").dim_others("room", "ray")
    return s


def _copies(e):
    return [o.id for o in e.document.objects if o.type == "hinged_copy"]


def _sem(e, i, t):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def _when(e, seg):
    return tuple(e._snap.compiled.time_map.resolve(x) for x in (seg.t0, seg.t1))


@pytest.mark.parametrize("dims", ["2d", "3d"])
def test_each_copy_lands_as_the_ray_arrives_at_its_wall(dims):
    from phasor.style.theme import load_theme

    e = _scene(dims).engine
    u = um.box_unfold((0, 0, 0), ROOM, SRC, AIM, 5)
    want = [f"room.mirror_{um.cell_name(c.cell_to)}" for c in u.crossings]
    assert _copies(e) == want == ["room.mirror_e1", "room.mirror_e1_n1", "room.mirror_e2_n1",
                                  "room.mirror_e2_n2", "room.mirror_e3_n2"]  # fmt: skip
    flip = load_theme(e.document.meta.theme).motion["unfold"]["duration"]
    tracks = e.document.timeline.tracks
    ray = tracks["ray.progress"][0]
    t0, t1 = _when(e, ray)
    assert ray.ease == "linear" and ray.verb == "unfold"
    speed = u.end / (t1 - t0)
    for copy, c in zip(want, u.crossings, strict=True):
        a, b = _when(e, tracks[f"{copy}.turn"][0])
        assert b - a == pytest.approx(flip)  # the theme's time
        assert b == pytest.approx(t0 + c.distance / speed)  # lands as the ray gets there
        st = e.state_at(b)
        assert st["ray"].params.progress * u.end == pytest.approx(c.distance, rel=1e-6)
        assert st[copy].params.turn == pytest.approx(1.0)
        assert e.state_at(a - 0.01)[copy].params.opacity == 0  # unseen before its flip
        mid = _sem(e, copy, (a + b) / 2)
        assert 0 < mid["turn"] < 1 and not mid["landed"]
    end = e.state_at(t1 + 0.1)
    np.testing.assert_allclose(end["ray"].params.points[1], u.end_point, atol=1e-9)
    # once landed, every copy sits on its mirror room
    for copy, c in zip(want, u.crossings, strict=True):
        b = end[copy].impl.world_bounds(end)
        np.testing.assert_allclose(b.lo, np.asarray(c.cell_to) * ROOM, atol=1e-6)
        assert _sem(e, copy, t1 + 0.1)["landed"]


def test_the_copy_draws_the_room_moved():
    from phasor.core.record import Recorder

    e = _scene().engine
    st = e.state_at(30.0)
    cam = st.camera
    src, copy = Recorder(cam.right, cam.up), Recorder(cam.right, cam.up)
    st["room"].impl.draw(src, st, "color")
    st["room.mirror_e1"].impl.draw(copy, st, "color")
    m = st["room.mirror_e1"].impl.matrix(st)
    # the walls and the center line of every edge are the room's, moved by the hinge
    np.testing.assert_allclose(copy.points[0], um.apply(m, src.points[0]), atol=1e-9)
    assert len(copy.strokes) == len(src.strokes) == 12
    for a, b in zip(src.strokes, copy.strokes, strict=True):
        np.testing.assert_allclose(b, um.apply(m, a), atol=1e-9)
    assert np.vstack(copy.strokes)[:, 0].min() == pytest.approx(4.0)  # across the east wall


def test_mid_turn_strokes_are_drawn_for_the_real_camera():
    """Half way over, a copy's edges are the ribbons the real camera would give its moved
    edges: as wide as ever, facing the viewer (never edge-on)."""
    from phasor.core.record import Recorder
    from phasor.domains.strokes import polylines

    e = _scene("3d").engine
    copy = "room.mirror_e1"
    a, b = _when(e, e.document.timeline.tracks[f"{copy}.turn"][0])
    st = e.state_at((a + b) / 2)
    cam = st.camera
    got = Recorder(cam.right, cam.up)
    st[copy].impl.draw(got, st, "color")
    m = st[copy].impl.matrix(st)
    room = st["room"]
    g = room.impl.geometry(st)
    segs = um.apply(m, g.vertices[g.edges])
    width = room.style["stroke"] * cam.units_per_height(segs.reshape(-1, 3)).reshape(-1, 2)
    want = Recorder(cam.right, cam.up)
    polylines(want, cam, segs, width, color=(1, 1, 1, 1))
    np.testing.assert_allclose(np.vstack(got.points), np.vstack(want.points), atol=1e-9)


def test_the_moved_camera_sees_the_copy_as_the_real_one_sees_it_moved():
    from phasor.domains.unfold import moved_camera

    st = _scene("3d").engine.state_at(0.0)
    m = um.chain_matrix([((4.0, 1.25, 1.0), (0.0, 1.0, 0.0), 180.0)], 0.37)
    seen = moved_camera(st.camera, m)
    p = np.array([[0.5, 0.2, 0.3], [3.0, 2.0, 1.0]])
    np.testing.assert_allclose(seen.project(p), st.camera.project(um.apply(m, p)), atol=1e-12)
    np.testing.assert_allclose(seen.depth(p), st.camera.depth(um.apply(m, p)), atol=1e-12)


@pytest.mark.parametrize("extra", ["before", "after"])
def test_the_order_of_verbs_does_not_matter(extra):
    plain = _scene(extra="before").engine
    other = _scene(extra=extra).engine
    for t in (0.5, 1.5, 3.0, 6.0, 12.0):
        assert plain.inspect(t, detail="brief") == other.inspect(t, detail="brief")
        st = other.state_at(t)
        assert all(st[c].params.dim == 1.0 for c in _copies(other))  # copies are the room's


def test_pure_in_time():
    e = _scene("3d").engine
    a, b = e.inspect(3.3), e.inspect(3.3)
    assert a == b
    assert _scene("3d").engine.inspect(3.3) == a


def test_semantics_and_features():
    e = _scene("3d").engine
    sem = _sem(e, "room.mirror_e1", 30.0)
    assert sem["kind"] == "hinged_copy" and sem["of"] == "room" and sem["hinges"] == 1
    assert sem["angle"] == 180.0 and sem["backdrop"]  # a Room3D is a backdrop, so are its copies
    np.testing.assert_allclose(sem["center"], (6.0, 1.25, 1.0), atol=1e-6)
    st = e.state_at(30.0)
    f = st["room.mirror_e1_n1"].impl.features(st)
    np.testing.assert_allclose(f["center"], (6.0, 3.75, 1.0), atol=1e-6)
    assert "hinge" in f


def test_unfold_mistakes_are_explained():
    from phasor import Dot, Line, Scene, ac
    from phasor.time.verbs import VerbError

    s = _scene()
    with pytest.raises(VerbError, match="already in the scene"):
        s.at("flip").unfold("room", ray="ray", flips=2)
    s.add(Dot(id="dot", pos=(1, 1)))
    with pytest.raises(VerbError, match="needs line"):
        s.at("flip").unfold("room", ray="dot")
    s.add(Line(id="up", points=[SRC, (1.0, 1.0, 1.5)]))
    with pytest.raises(VerbError, match="in 2D"):
        s.at("flip").unfold("room", ray="up")
    s.add(Line(id="out", points=[(9, 9, 1), (10, 9, 1)]))
    with pytest.raises(VerbError, match="inside"):
        s.at("flip").unfold("room", ray="out")
    m = Scene("mesh", passes=None)
    m.engine.base_dir = ROOT / "examples"
    m.add(ac.Room3D(id="room", path="assets/l_room.obj"))
    m.add(Line(id="ray", points=[(1, 1, 1), (2, 1, 1)]))
    with pytest.raises(VerbError, match="box room"):
        m.at(0.0).unfold("room", ray="ray")


def test_the_mcp_tool_unfolds(tmp_path):
    from phasor.workspace.tools import Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("demo")
    ws.scene_edit([
        {"op": "add_object", "object": {"id": "room", "type": "room", "params": {"size": ROOM}}},
        {"op": "add_object", "object": {"id": "ray", "type": "line",
                                        "params": {"points": [SRC, [1.8, 1.05, 1.0]]}}},
    ])  # fmt: skip
    r = ws.animate("unfold", 0.5, ["room", "ray"], value=3)
    assert r["end"] > r["start"]
    assert [i for i in ws.summary()["objects"] if i.startswith("room.mirror")] == [
        "room.mirror_e1",
        "room.mirror_e1_n1",
        "room.mirror_e2_n1",
    ]


def test_a_hinged_copy_needs_its_source():
    from phasor import HingedCopy, Scene

    s = Scene("lonely", passes=None)
    s.add(HingedCopy(id="ghost", source="nothing", hinges=[((1, 0, 0), (0, 1, 0), 180.0)]))
    assert _sem(s.engine, "ghost", 0.0)["source_missing"] == "nothing"
    with pytest.raises(Exception, match="direction"):
        s.add(HingedCopy(id="bad", source="ghost", hinges=[((0, 0, 0), (0, 0, 0), 90.0)]))


# ---------------------------------------------------------------- the example


def test_the_example_matches_its_golden():
    import os
    import runpy

    from tests.conftest import gl_available

    e = runpy.run_path(str(ROOT / "examples" / "mirror_rooms.py"), run_name="x")["s"].engine
    assert not e.lint(strict=True)
    assert _sem(e, "room.mirror_e3_n2", 20.0)["landed"]
    if not gl_available():
        pytest.skip("needs OpenGL")
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    sheet = e.storyboard([3.0, 4.9, 6.0, 7.5, 9.5, 12.5], size=(480, 270))["image"]
    path = GOLDEN / e.pass_runner().renderer_class() / "mirror_rooms.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
