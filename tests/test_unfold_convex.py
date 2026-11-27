"""Mirror unfolding in any convex room: polygons (billiards, the kaleidoscope), convex mesh
rooms, rays of ``SoundRays`` and corners (``core/unfold_math.py`` ``room_unfold``, the
``unfold`` verb). Checked against answers worked out by hand, against nearby rays, and
against the true mirror image of the room."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from phasor.core import unfold_math as um

ROOT = Path(__file__).parents[1]
S3 = math.sqrt(3.0)
EQUILATERAL = [(0.0, 0.0), (1.0, 0.0), (0.5, S3 / 2)]  # walls a (bottom), b (right), c (left)
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
HALF_EQUILATERAL = [(0.0, 0.0), (1.0, 0.0), (0.0, S3)]  # 90, 30 and 60 degrees
SCALENE = [(0.0, 0.0), (3.0, 0.0), (0.7, 1.9)]  # no symmetry at all
HEXAGON = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]
# A room with a sloping ceiling: convex, and symmetric about no plane square to a wall.
WEDGE = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0],
                  [0, 0, 2], [4, 0, 3], [4, 3, 3.5], [0, 3, 2.2]], float)  # fmt: skip
WEDGE_TRIS = np.array([
    [0, 2, 1], [0, 3, 2],  # floor
    [0, 1, 5], [0, 5, 4],  # south
    [1, 2, 6], [1, 6, 5],  # east
    [2, 3, 7], [2, 7, 6],  # north
    [3, 0, 4], [3, 4, 7],  # west
    [4, 5, 6], [4, 6, 7],  # the ceiling, two planes
])  # fmt: skip


def _reflect(points, a, b):
    """Points mirrored in the line through a and b (in the plane z = 0), worked out plainly."""
    p = np.asarray(points, dtype=np.float64)
    a, b = np.r_[a, 0.0][:3], np.r_[b, 0.0][:3]
    u = (b - a) / np.linalg.norm(b - a)
    rel = p - a
    along = rel @ u
    return a + 2 * along[:, None] * u - rel


# ---------------------------------------------------------------- the maths, by hand


def test_a_ray_in_an_equilateral_triangle_worked_out_by_hand():
    # From the centre along +x it meets the right wall (b) a third of the way on, at
    # (5/6, sqrt(3)/6); reflected there it goes along (-1/2, -sqrt(3)/2) and meets the
    # bottom (a) after another third, at (2/3, 0). The path closes after six reflections.
    room = um.polygon(EQUILATERAL)
    u = um.room_unfold(room, (0.5, S3 / 6, 0.0), (1.0, 0.0, 0.0), 6)
    assert [c.wall for c in u.crossings] == ["b", "a", "c", "b", "a", "c"]
    np.testing.assert_allclose([c.distance for c in u.crossings],
                               np.array([1, 2, 4, 5, 7, 8]) / 3, atol=1e-12)  # fmt: skip
    np.testing.assert_allclose(u.crossings[0].wall_point, (5 / 6, S3 / 6, 0), atol=1e-12)
    np.testing.assert_allclose(u.crossings[1].wall_point, (2 / 3, 0, 0), atol=1e-12)
    np.testing.assert_allclose(u.real_end, u.crossings[0].wall_point, atol=1e-12)  # closed
    assert [c.name for c in u.crossings[:3]] == ["b", "b_a", "b_a_c"]


def test_a_ray_square_to_a_hexagon_wall_bounces_straight_back():
    room = um.polygon(HEXAGON)
    aim = (math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0)  # at the middle of wall a
    u = um.room_unfold(room, (0.0, 0.0, 0.0), aim, 4)
    assert [c.wall for c in u.crossings] == ["a", "d", "a", "d"]  # there and back
    np.testing.assert_allclose([c.distance for c in u.crossings],
                               np.array([1, 3, 5, 7]) * S3 / 2, atol=1e-12)  # fmt: skip


@pytest.mark.parametrize("corners", [EQUILATERAL, HEXAGON, SCALENE, HALF_EQUILATERAL])
def test_the_straight_line_is_as_long_as_the_path_and_angle_in_equals_angle_out(corners):
    room = um.polygon(corners)
    u = um.room_unfold(
        room, room.vertices.mean(axis=0) + np.array([0.02, 0.01, 0]), (1, 0.37, 0), 12
    )
    path = um.zigzag(u)
    assert all(room.contains(p, 1e-9) for p in path)
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(u.end)
    for k, c in enumerate(u.crossings):
        n = room.normals[c.index]
        assert n @ path[k + 1] == pytest.approx(room.offsets[c.index])  # on that wall
        a, b = path[k + 1] - path[k], path[k + 2] - path[k + 1]
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        np.testing.assert_allclose(b, a - 2 * (a @ n) * n, atol=1e-9)


def test_the_landed_copy_is_the_true_mirror_image_of_an_asymmetric_polygon():
    room = um.polygon(SCALENE)
    v = room.vertices
    u = um.room_unfold(room, (1.0, 0.5, 0.0), (1.0, 0.8, 0.0), 6)
    want = v
    for chain, c in zip(u.chains(), u.crossings, strict=True):
        # the copy before it, mirrored in its wall (worked out plainly, corner by corner)
        k = c.index
        want = _reflect(want, want[k, :2], want[(k + 1) % 3, :2])
        np.testing.assert_allclose(um.apply(um.chain_matrix(chain), v), want, atol=1e-9)
        np.testing.assert_allclose(um.apply(um.copy_matrix(room, c.word), v), want, atol=1e-9)


def test_a_flat_page_lifts_towards_the_viewer():
    room = um.polygon(SCALENE)
    (c,) = um.room_unfold(room, (1.0, 0.5, 0.0), (1.0, 0.8, 0.0), 1).crossings
    half = um.apply(um.chain_matrix([c.hinge], 0.5), room.vertices.mean(axis=0))
    assert half[2] > 0.1  # up out of the plane (+z), as a box's pages do


@pytest.mark.parametrize("corners", [EQUILATERAL, SQUARE, HALF_EQUILATERAL])
def test_copies_of_the_kaleidoscope_triangles_tile_the_plane(corners):
    """Every corner of these is 180/n degrees: the mirror copies along a ray never overlap,
    and every corner of every copy is a corner of the tiling the room's own corners make."""
    room = um.polygon(corners)
    u = um.room_unfold(room, room.vertices.mean(axis=0), (1.0, 0.2913, 0.0), 30)
    copies = [um.apply(um.copy_matrix(room, c.word), room.vertices) for c in u.crossings]
    middles = np.array([c.mean(axis=0) for c in copies])
    for i, mid in enumerate(middles):
        others = np.delete(np.arange(len(copies)), i)
        for j in others:  # no copy's middle lies inside another copy
            inside = um.polygon(copies[j]).contains(mid, -1e-9)
            assert not inside
    # in the unfolded frame the ray runs straight on from copy to copy
    for c in u.crossings:
        assert um.polygon(copies[u.crossings.index(c)]).contains(c.point, 1e-9)


def test_a_ray_into_a_corner_passes_it_as_a_ray_a_hair_to_its_left():
    room = um.polygon(EQUILATERAL)
    start = np.array([0.5, S3 / 6, 0.0])
    aim = np.array([1.0, 0.0, 0.0]) - start  # straight at corner (1, 0): 60 degrees
    u = um.room_unfold(room, start, aim, 4)
    d = np.asarray(u.direction)
    assert [c.name for c in u.crossings[:3]] == ["b", "b_a", "b_a_b"]
    assert u.crossings[0].distance == pytest.approx(u.crossings[2].distance)  # at once
    left, right = np.cross((0, 0, 1), d), -np.cross((0, 0, 1), d)
    corner = um.copy_matrix(room, u.crossings[2].word)
    for side in (left, right):  # 180/3 degrees: both sides end in the same copy
        v = um.room_unfold(room, start + 1e-7 * side, aim, 3)
        np.testing.assert_allclose(um.copy_matrix(room, v.crossings[-1].word), corner, atol=1e-9)
    assert [c.name for c in um.room_unfold(room, start + 1e-7 * left, aim, 3).crossings] == [
        "b", "b_a", "b_a_b"]  # fmt: skip
    # the square's corner (90 degrees) gives the box's two crossings at once
    sq = um.room_unfold(um.polygon(SQUARE), (0.5, 0.5, 0), (1, 1, 0), 2)
    assert sq.crossings[0].distance == pytest.approx(sq.crossings[1].distance)
    np.testing.assert_allclose(um.apply(um.copy_matrix(um.polygon(SQUARE), sq.crossings[1].word),
                                        (0.5, 0.5, 0)), (1.5, 1.5, 0), atol=1e-12)  # fmt: skip


def test_at_other_corners_the_left_side_is_taken():
    room = um.polygon([(0, 0), (2, 0), (0.6, 1.4)])  # corner (2, 0): 45 degrees... not 180/n
    start = np.array([0.8, 0.4, 0.0])
    aim = np.array([2.0, 0.0, 0.0]) - start
    corner = um.room_unfold(room, start, aim, 5)
    left = um.room_unfold(room, start + 1e-8 * np.cross((0, 0, 1), aim / np.linalg.norm(aim)),
                          aim, 5)  # fmt: skip
    assert [c.name for c in corner.crossings] == [c.name for c in left.crossings]


def test_polygon_walls_and_names():
    ccw, cw = um.polygon(SQUARE), um.polygon(SQUARE[::-1])
    np.testing.assert_allclose(ccw.normals[0], (0, -1, 0), atol=1e-12)  # a: the bottom
    np.testing.assert_allclose(cw.normals[0], (0, 1, 0), atol=1e-12)  # cw: (0, 1) to (1, 1)
    closed = um.polygon([*SQUARE, SQUARE[0]])
    np.testing.assert_allclose(closed.normals, ccw.normals)
    straight = um.polygon([(0, 0), (0.5, 0), (1, 0), (1, 1), (0, 1)])  # (0.5, 0) is no corner
    assert len(straight.offsets) == 4
    assert straight.names == ("a", "b", "c", "d")
    assert [um.wall_name(k) for k in (0, 25, 26, 27, 52)] == ["a", "z", "aa", "ab", "ba"]
    assert um.word_name((1, 0, 2)) == "b_a_c" and um.word_name(()) == "home"
    with pytest.raises(ValueError, match="not convex"):
        um.polygon([(0, 0), (2, 0), (2, 2), (1, 0.5), (0, 2)])
    with pytest.raises(ValueError, match="not convex"):  # a star winds round twice
        um.polygon([(math.cos(a), math.sin(a)) for a in np.arange(5) * 4 * math.pi / 5])
    with pytest.raises(ValueError, match="one z"):
        um.polygon([(0, 0, 0), (1, 0, 0), (0, 1, 1)])
    with pytest.raises(ValueError, match="no z"):
        um.room_unfold(ccw, (0.5, 0.5, 0), (1, 0, 0.1), 2)
    with pytest.raises(ValueError, match="inside"):
        um.room_unfold(ccw, (1.5, 0.5, 0), (1, 0, 0), 2)


# ---------------------------------------------------------------- convex polyhedra


def test_a_box_mesh_unfolds_exactly_as_the_box_does():
    from phasor.domains import ac_math

    v, f = ac_math.box_mesh((4.0, 2.5, 2.0))
    room = um.polyhedron(v, f)
    assert room.names == ("a", "b", "c", "d", "e", "f")  # west, east, south, north, floor, up
    args = ((0.8, 0.6, 1.0), (1.0, 0.45, 0.3), 7)
    mesh, box = um.room_unfold(room, *args), um.box_unfold((0, 0, 0), (4, 2.5, 2), *args)
    for a, b in zip(mesh.crossings, box.crossings, strict=True):
        assert a.hinge == b.hinge and a.hinge.mirror is None  # symmetric: plain turns
        assert a.distance == pytest.approx(b.distance)
        np.testing.assert_allclose(a.wall_point, b.wall_point, atol=1e-12)


def test_a_convex_polyhedron_copy_lands_on_its_true_mirror_image():
    room = um.polyhedron(WEDGE, WEDGE_TRIS)
    assert len(room.offsets) == 7  # the ceiling's two halves are two walls
    u = um.room_unfold(room, (1.0, 1.0, 1.0), (1.0, 0.4, 0.5), 8)
    for chain, c in zip(u.chains(), u.crossings, strict=True):
        m = um.chain_matrix(chain)
        want = um.copy_matrix(room, c.word)
        # the copy before it, reflected in the plane of its wall
        prev = um.copy_matrix(room, c.word[:-1])
        n = prev[:3, :3] @ room.normals[c.index]
        d = room.offsets[c.index] + n @ prev[:3, 3]
        np.testing.assert_allclose(want, um.reflection_matrix(d * n, n) @ prev, atol=1e-9)
        np.testing.assert_allclose(m, want, atol=1e-9)  # what is drawn is the true image
        assert np.linalg.det(m[:3, :3]) == pytest.approx((-1) ** len(c.word))
        assert c.hinge.mirror is not None  # a turn alone could not land there
    path = um.zigzag(u)
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(u.end)
    assert all(room.contains(p, 1e-9) for p in path)


def test_the_mirror_comes_first_and_the_turn_lands_it():
    room = um.polyhedron(WEDGE, WEDGE_TRIS)
    (c,) = um.room_unfold(room, (1.0, 1.0, 1.0), (1.0, 0.4, 0.5), 1).crossings
    start = um.chain_matrix([c.hinge], 0.0)
    assert np.linalg.det(start[:3, :3]) == pytest.approx(-1.0)  # mirrored where it appears
    p = np.asarray(c.hinge.point)
    np.testing.assert_allclose(um.apply(start, p), p, atol=1e-12)  # in place, on the wall
    mirror = np.asarray(c.hinge.mirror)
    assert abs(mirror @ np.asarray(c.hinge.axis)) < 1e-12  # the line lies in its plane


def test_a_mesh_that_is_not_convex_is_refused():
    from phasor.domains import ac_math

    v, f = ac_math.load_obj(ROOT / "examples" / "assets" / "l_room.obj")
    with pytest.raises(ValueError, match="not convex"):
        um.polyhedron(v, f)


def test_mirror_chain_makes_a_kaleidoscope_copy_by_copy():
    room = um.polygon(EQUILATERAL)
    for word in ((0,), (0, 1), (2, 0, 1, 2)):
        m = um.chain_matrix(um.mirror_chain(room, word))
        np.testing.assert_allclose(um.apply(m, room.vertices),
                                   um.apply(um.copy_matrix(room, word), room.vertices),
                                   atol=1e-12)  # fmt: skip


# ---------------------------------------------------------------- the verb


def _when(e, seg):
    return tuple(e._snap.compiled.time_map.resolve(x) for x in (seg.t0, seg.t1))


def _sem(e, i, t):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def _copies(e):
    return [o.id for o in e.document.objects if o.type == "hinged_copy"]


def _billiard(room_obj=None, aim=(1.0, 0.37, 0.0), start=(1.0, 0.5, 0.0), flips=5):
    from phasor import Line, Scene

    s = Scene("billiard", passes=None)
    s.add(room_obj or Line(id="table", points=[*[(*p, 0.0) for p in SCALENE], (0, 0, 0)]))
    s.add(Line(id="ray", points=[start, tuple(a + b for a, b in zip(start, aim, strict=True))]))
    s.marker("go", 1.0)
    room = room_obj.id if room_obj is not None else "table"
    s.at("go").unfold(room, ray="ray", flips=flips)
    return s


def test_a_polygon_billiard_unfolds_into_true_mirror_tables():
    from phasor.style.theme import load_theme

    e = _billiard().engine
    room = um.polygon(SCALENE)
    u = um.room_unfold(room, (1.0, 0.5, 0.0), (1.0, 0.37, 0.0), 5)
    want = [f"table.mirror_{c.name}" for c in u.crossings]
    assert _copies(e) == want and want[0] == f"table.mirror_{u.crossings[0].wall}"
    flip = load_theme(e.document.meta.theme).motion["unfold"]["duration"]
    tracks = e.document.timeline.tracks
    t0, t1 = _when(e, tracks["ray.progress"][0])
    speed = u.end / (t1 - t0)
    end = e.state_at(t1 + 0.1)
    for copy, c in zip(want, u.crossings, strict=True):
        a, b = _when(e, tracks[f"{copy}.turn"][0])
        assert b - a == pytest.approx(flip)
        assert b == pytest.approx(t0 + c.distance / speed)
        m = end[copy].impl.matrix(end)
        np.testing.assert_allclose(um.apply(m, room.vertices),
                                   um.apply(um.copy_matrix(room, c.word), room.vertices),
                                   atol=1e-9)  # fmt: skip
        assert _sem(e, copy, t1 + 0.1)["landed"] and "mirrors" not in _sem(e, copy, t1 + 0.1)
    np.testing.assert_allclose(end["ray"].params.points[1], u.end_point, atol=1e-9)


def test_an_algebra_polygon_is_a_kaleidoscope_room():
    from phasor import algebra

    e = _billiard(algebra.Polygon(id="tri", n=3, r=1.0), start=(0.1, -0.1, 0), flips=4).engine
    assert _copies(e)[0].startswith("tri.mirror_") and len(_copies(e)) == 4
    st = e.state_at(30.0)
    v = st["tri"].impl.features(st)["vertices"]
    last = _copies(e)[-1]
    word = [ord(w) - ord("a") for w in last.rsplit("mirror_", 1)[1].split("_")]
    got = st[last].impl.features(st)["vertices"]
    np.testing.assert_allclose(got, um.apply(um.copy_matrix(um.polygon(v), word), v), atol=1e-9)


def test_a_convex_mesh_room_unfolds_with_true_mirror_copies(tmp_path):
    from phasor import Line, Scene, ac

    lines = [f"v {x:g} {y:g} {z:g}" for x, y, z in WEDGE]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in WEDGE_TRIS]
    (tmp_path / "wedge.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    s = Scene("wedge", passes=None)
    s.engine.base_dir = tmp_path
    s.use_camera("orbit", azimuth=-80, elevation=40, distance=24, target=(6, 2, 2))
    s.add(ac.Room3D(id="room", path="wedge.obj", preset="outline"))
    s.add(Line(id="ray", points=[(1.0, 1.0, 1.0), (2.0, 1.4, 1.5)]))
    s.at(0.5).unfold("room", ray="ray", flips=4)
    e = s.engine
    room = um.polyhedron(WEDGE, WEDGE_TRIS)
    u = um.room_unfold(room, (1.0, 1.0, 1.0), (1.0, 0.4, 0.5), 4)
    assert _copies(e) == [f"room.mirror_{c.name}" for c in u.crossings]
    st = e.state_at(60.0)
    for c in u.crossings:
        copy = f"room.mirror_{c.name}"
        want = um.copy_matrix(room, c.word)
        np.testing.assert_allclose(st[copy].impl.matrix(st), want, atol=1e-9)
        center = um.apply(want, (WEDGE.min(axis=0) + WEDGE.max(axis=0)) / 2)
        np.testing.assert_allclose(_sem(e, copy, 60.0)["center"], center, atol=1e-5)
        assert _sem(e, copy, 60.0)["mirrors"] == len(c.word)


def test_one_ray_of_sound_rays_unfolds():
    from phasor import Scene, ac

    s = Scene("rays", passes=None)
    s.use_camera("orbit", azimuth=-90, elevation=50, distance=20, target=(8, 4, 1))
    s.add(ac.Room3D(id="room", size=(4.0, 2.5, 2.0), preset="outline"))
    s.add(ac.SoundRays(id="rays", room="room", source=(0.8, 0.6, 1.0), count=24, bounces=3))
    s.at(0.5).unfold("room", ray="rays", index=3, flips=4)
    e = s.engine
    st0 = e.state_at(0.0)
    d = st0["rays"].impl.paths(st0)["dirs"][3]
    u = um.box_unfold((0, 0, 0), (4.0, 2.5, 2.0), (0.8, 0.6, 1.0), d, 4)
    assert _copies(e) == [f"room.mirror_{c.name}" for c in u.crossings]
    line = e.state_at(60.0)["rays.unfolded_3"]
    np.testing.assert_allclose(line.params.points, [u.start, u.end_point], atol=1e-9)
    assert e.state_at(0.4)["rays.unfolded_3"].params.progress == 0.0
    assert line.params.progress == 1.0
    from phasor.time.verbs import VerbError

    with pytest.raises(VerbError, match="index="):
        s.at(0.5).unfold("room", ray="rays")
    with pytest.raises(VerbError, match="rays 0 to 23"):
        s.at(0.5).unfold("room", ray="rays", index=24)
    with pytest.raises(VerbError, match="already in the scene"):
        s.at(0.5).unfold("room", ray="rays", index=3)
    s.add(ac.Room3D(id="hall", size=(9.0, 6.0, 4.0), origin=(0, 8, 0)))
    with pytest.raises(VerbError, match="bounces in 'room'"):
        s.at(0.5).unfold("hall", ray="rays", index=4)


def test_a_corner_flips_one_after_the_other():
    """A ray into a corner meets two walls at once: the second page turns after the first
    has landed (on top of it), both before the ray gets there."""
    from phasor import Line, Scene, ac
    from phasor.style.theme import load_theme

    s = Scene("corner", passes=None)
    s.add(ac.Room(id="room", size=(4.0, 2.0, 2.0)))
    s.add(Line(id="ray", points=[(1.0, 0.5, 1.0), (2.0, 1.0, 1.0)]))  # to the corner (4, 2)
    s.marker("go", 1.0)
    s.at("go").unfold("room", ray="ray", flips=3)
    e = s.engine
    u = um.box_unfold((0, 0, 0), (4, 2, 2), (1.0, 0.5, 1.0), (2.0, 1.0, 0.0), 3)
    assert u.crossings[0].distance == pytest.approx(u.crossings[1].distance)
    assert _copies(e) == ["room.mirror_e1", "room.mirror_e1_n1", "room.mirror_e2_n1"]
    flip = load_theme(e.document.meta.theme).motion["unfold"]["duration"]
    tracks = e.document.timeline.tracks
    t0, t1 = _when(e, tracks["ray.progress"][0])
    arrive = t0 + u.crossings[0].distance * (t1 - t0) / u.end
    first = _when(e, tracks["room.mirror_e1.turn"][0])
    second = _when(e, tracks["room.mirror_e1_n1.turn"][0])
    assert first[1] == pytest.approx(second[0])  # one after the other
    assert second[1] == pytest.approx(arrive)  # both done as the ray gets there
    assert first[0] >= 1.0 - 1e-9  # not before the anchor
    assert second[1] - first[0] >= flip - 1e-9  # together they take the theme's time at least
    mid = (second[0] + second[1]) / 2
    assert _sem(e, "room.mirror_e1", mid)["landed"]
    assert e.state_at(first[0] - 0.01)["room.mirror_e1_n1"].params.opacity == 0
    assert e.state_at((first[0] + first[1]) / 2)["room.mirror_e1_n1"].params.opacity == 0


def test_convex_unfold_mistakes_are_explained():
    from phasor import Line, Scene
    from phasor.time.verbs import VerbError

    s = Scene("bad", passes=None)
    s.add(Line(id="notch", points=[(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 0.5, 0), (0, 2, 0),
                                   (0, 0, 0)]))  # fmt: skip
    s.add(Line(id="ray", points=[(0.5, 0.3, 0), (1.0, 0.4, 0)]))
    with pytest.raises(VerbError, match="convex"):
        s.at(0.0).unfold("notch", ray="ray")
    with pytest.raises(VerbError, match="index picks"):
        s.at(0.0).unfold("notch", ray="ray", index=2)


def test_pure_and_in_any_order():
    from phasor import Line, Scene

    def make(dim_first):
        s = Scene("order", passes=None)
        s.add(Line(id="table", points=[(0, 0, 0), (1, 0, 0), (0.5, S3 / 2, 0), (0, 0, 0)]))
        s.add(Line(id="ray", points=[(0.4, 0.3, 0), (1.4, 0.55, 0)]))
        s.add(Line(id="other", points=[(-2, -1, 0), (-1, -1, 0)]))
        s.marker("go", 0.5)
        if dim_first:
            s.at("go").dim_others("table", "ray")
        s.at("go").unfold("table", ray="ray", flips=6)
        if not dim_first:
            s.at("go").dim_others("table", "ray")
        return s.engine

    a, b = make(True), make(False)
    for t in (0.7, 2.0, 4.4, 9.0):
        assert a.inspect(t, detail="brief") == b.inspect(t, detail="brief") == make(True).inspect(
            t, detail="brief")  # fmt: skip


# ---------------------------------------------------------------- the example


def test_the_kaleidoscope_example_matches_its_golden():
    import os
    import runpy

    from tests.conftest import gl_available

    ns = runpy.run_path(str(ROOT / "examples" / "kaleidoscope.py"), run_name="x")
    e = ns["s"].engine
    assert not e.lint(strict=True)
    copies = _copies(e)
    assert copies == [f"mirrors.mirror_{c.name}" for c in ns["unfolded"].crossings]
    assert _sem(e, copies[-1], e.duration)["landed"]
    if not gl_available():
        pytest.skip("needs OpenGL")
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    sheet = e.storyboard([2.5, 5.0, 6.5, 8.5, 11.0, 12.5], size=(480, 270))["image"]
    path = GOLDEN / e.pass_runner().renderer_class() / "kaleidoscope.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
