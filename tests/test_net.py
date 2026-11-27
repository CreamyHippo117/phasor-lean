"""Nets and geodesics of convex polyhedra: ``domains/net_math.py`` against answers known by
hand (the spider and the fly, corners of a cube, a tetrahedron) and against a brute-force
shortest path, then the ``Polyhedron`` composite that draws them."""

from __future__ import annotations

import heapq
import itertools
from pathlib import Path

import numpy as np
import pytest

from phasor.core import unfold_math as um
from phasor.domains import net_math as nm

ROOT = Path(__file__).parents[1]
SPIDER_ROOM = (30.0, 12.0, 12.0)
SPIDER, FLY = (0.0, 6.0, 11.0), (30.0, 6.0, 1.0)


def _tetra(edge=1.0):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    v *= edge / np.linalg.norm(v[0] - v[1])
    return v, nm.solid(v, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], ["a", "b", "c", "d"])


# ---------------------------------------------------------------- geodesics, by hand


def test_the_spider_and_the_fly_is_40():
    s = nm.box(SPIDER_ROOM)
    g = nm.geodesic(s, SPIDER, FLY)
    assert g.length == pytest.approx(40.0, abs=1e-9)  # not 42: up, along, down
    assert len(g.faces) == 5  # it crosses five of the six faces
    assert [s.names[f] for f in (g.faces[0], g.faces[-1])] == ["west", "east"]
    path = np.linalg.norm(np.diff(g.points, axis=0), axis=1).sum()
    assert path == pytest.approx(40.0, abs=1e-9)  # the path drawn on the solid is as long
    for q in g.points:  # every point on the surface
        assert s.faces_at(q)


def test_opposite_corners_of_a_unit_cube_are_root_5_apart():
    g = nm.geodesic(nm.box(), (0, 0, 0), (1, 1, 1))
    assert g.length == pytest.approx(np.sqrt(5), abs=1e-12)
    assert len(g.faces) == 2  # across two faces, through the middle of an edge
    np.testing.assert_allclose(np.sort(g.points[1]), [0.0, 0.5, 1.0], atol=1e-12)


def test_more_answers_by_hand():
    cube = nm.box()
    # middles of opposite faces: half, across a whole side, half
    assert nm.geodesic(cube, (0.5, 0.5, 0), (0.5, 0.5, 1)).length == pytest.approx(2.0)
    # the same face: a straight line
    g = nm.geodesic(cube, (0.1, 0.2, 1), (0.9, 0.7, 1))
    assert g.length == pytest.approx(np.hypot(0.8, 0.5)) and len(g.faces) == 1
    # a regular tetrahedron: the middles of opposite edges are 1 apart on the surface
    # (in space 1/sqrt 2): opposite sides of a rhombus of two faces laid flat
    v, t = _tetra()
    g = nm.geodesic(t, (v[0] + v[1]) / 2, (v[2] + v[3]) / 2)
    assert g.length == pytest.approx(1.0, abs=1e-12)


def _brute_force(s: nm.Solid, a, b, n: int) -> float:
    """Dijkstra over points spaced along every edge, joined straight across each face: a
    path on the surface, so never shorter than the geodesic, and close to it for large n."""
    pts = [np.asarray(a, float), np.asarray(b, float)]
    on = [set(s.faces_at(a)), set(s.faces_at(b))]
    done = set()
    for (f, g), (i, j) in s.edges.items():
        if (g, f) in done:
            continue
        done.add((f, g))
        for u in np.linspace(0, 1, n):
            pts.append(s.v[i] + u * (s.v[j] - s.v[i]))
            on.append({f, g})
    p = np.asarray(pts)
    dist = np.full(len(p), np.inf)
    dist[0] = 0.0
    heap, seen = [(0.0, 0)], set()
    while heap:
        d, k = heapq.heappop(heap)
        if k in seen:
            continue
        seen.add(k)
        if k == 1:
            return d
        for m in range(len(p)):
            if m not in seen and on[k] & on[m]:
                nd = d + float(np.linalg.norm(p[m] - p[k]))
                if nd < dist[m]:
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
    return np.inf


def test_geodesics_agree_with_a_brute_force_search():
    rng = np.random.default_rng(7)
    s = nm.box((3.0, 1.5, 1.0))
    for _ in range(4):
        ends = []
        for _ in range(2):
            f = s.faces[rng.integers(6)]
            c = s.v[list(f)]
            u, w = rng.random(2)
            ends.append(c[0] + u * (c[1] - c[0]) + w * (c[3] - c[0]))
        exact = nm.geodesic(s, *ends).length
        rough = _brute_force(s, *ends, 41)
        assert exact <= rough + 1e-9  # nothing on the surface is shorter
        assert rough - exact < 0.01 * exact  # and the fine search comes close
        assert exact >= np.linalg.norm(ends[1] - ends[0]) - 1e-12


def test_a_geodesic_is_straight_when_its_faces_are_laid_flat():
    s = nm.box(SPIDER_ROOM)
    g = nm.geodesic(s, SPIDER, FLY)
    parent = nm.spanning_tree(s, g.faces[0], g.faces)
    m = nm.net_matrices(s, g.faces[0], parent, 1.0)
    flat = [um.apply(m[g.faces[0]], g.points[0])]
    flat += [um.apply(m[f], q) for f, q in zip(g.faces, g.points[1:], strict=True)]
    assert nm.straight_on_net(np.asarray(flat))
    assert np.linalg.norm(flat[-1] - flat[0]) == pytest.approx(40.0)
    # the 24-32-40 triangle: 32 along the strip of faces, 24 across it
    assert sorted(np.round(np.abs(flat[-1] - flat[0]), 9)) == [0.0, 24.0, 32.0]


def test_bad_solids_and_points_are_refused():
    with pytest.raises(ValueError, match="positive"):
        nm.box((1, 0, 1))
    v, _ = _tetra()
    with pytest.raises(ValueError, match="close up"):
        nm.solid(v, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1, 2)])
    cube = nm.box()
    flat_v = [*cube.vertices[:4], (0.5, 0.5, 0.0)]
    with pytest.raises(ValueError):
        nm.solid(flat_v, [(0, 1, 2), (1, 3, 2), (0, 1, 4), (1, 3, 4)])
    with pytest.raises(ValueError, match="surface"):
        nm.geodesic(cube, (0.5, 0.5, 0.5), (1, 1, 1))
    with pytest.raises(ValueError, match="no face"):
        cube.index("front")


# ---------------------------------------------------------------- nets


def test_hinges_are_quarter_turns_on_a_cube_and_dihedral_turns_in_general():
    cube = nm.box()
    for f, g in cube.edges:
        assert nm.hinge(cube, f, g).degrees == pytest.approx(90.0)
    _, t = _tetra()
    dihedral = np.degrees(np.arccos(1 / 3))  # 70.53 degrees
    for f, g in t.edges:
        assert nm.hinge(t, f, g).degrees == pytest.approx(180.0 - dihedral)


@pytest.mark.parametrize("which", ["cube", "tetra", "box"])
def test_a_net_opens_flat_outside_up_and_stays_joined(which):
    s = {"cube": nm.box(), "box": nm.box((3, 2, 1)), "tetra": _tetra()[1]}[which]
    root = 0
    parent = nm.spanning_tree(s, root)
    assert sorted(parent) == [f for f in range(len(s.faces)) if f != root]
    closed = nm.net_matrices(s, root, parent, 0.0)
    np.testing.assert_allclose(closed, np.tile(np.eye(4), (len(s.faces), 1, 1)), atol=1e-12)
    for amount in (0.3, 0.77, 1.0):
        m = nm.net_matrices(s, root, parent, amount)
        for f, g in parent.items():  # a face and its parent meet along their edge
            i, j = s.edges[(f, g)]
            np.testing.assert_allclose(um.apply(m[f], s.v[[i, j]]), um.apply(m[g], s.v[[i, j]]),
                                       atol=1e-9)  # fmt: skip
        for k in range(len(s.faces)):
            assert np.linalg.det(m[k][:3, :3]) == pytest.approx(1.0)  # turned, never mirrored
    flat = nm.net_matrices(s, root, parent, 1.0)
    n0, p0 = s.normals[root], s.v[s.faces[root][0]]
    for k, f in enumerate(s.faces):
        np.testing.assert_allclose(flat[k][:3, :3] @ s.normals[k], n0, atol=1e-9)  # outside up
        np.testing.assert_allclose((um.apply(flat[k], s.v[list(f)]) - p0) @ n0, 0, atol=1e-9)
    # laid flat, no two faces overlap: their centres are at least an inradius apart
    centres = [um.apply(flat[k], s.v[list(f)].mean(0)) for k, f in enumerate(s.faces)]
    near = min(np.linalg.norm(a - b) for a, b in itertools.combinations(centres, 2))
    assert near > 0.5 * min(np.ptp(s.v, axis=0))


def test_a_spanning_tree_keeps_the_run_it_is_given():
    s = nm.box(SPIDER_ROOM)
    run = nm.geodesic(s, SPIDER, FLY).faces
    parent = nm.spanning_tree(s, s.index("top"), run)
    for f, g in itertools.pairwise(run):
        assert parent.get(f) == g or parent.get(g) == f
    nm.check_tree(s, s.index("top"), parent)
    with pytest.raises(ValueError, match="does not hang"):
        nm.check_tree(s, 0, {1: 2, 2: 1})


# ---------------------------------------------------------------- the Polyhedron object


def _scene(**kw):
    from phasor import Scene, net

    s = Scene("net", passes=None)
    s.use_camera("orbit", azimuth=-70, elevation=55, distance=18, target=(2.5, 1, 1.5))
    params = {"size": (5, 2, 2), "root": "top", "geodesic": [(0, 1, 11 / 6), (5, 1, 1 / 6)],
              "unfold": 0.0, **kw}  # fmt: skip
    s.add(net.Polyhedron(id="room", **params))
    s.marker("open", 1.0)
    s.at("open").morph("room.unfold", 1.0)
    return s


def _sem(e, t, i="room"):
    return {o["id"]: o for o in e.inspect(t)["objects"]}[i]["semantic"]


def test_faces_are_parts_named_by_meaning():
    e = _scene().engine
    ids = [o.id for o in e.state_at(0.0).objects]
    assert ids == ["room", "room.face_bottom", "room.face_top", "room.face_south",
                   "room.face_north", "room.face_west", "room.face_east", "room.geodesic",
                   "room.geodesic_start", "room.geodesic_end"]  # fmt: skip
    face = _sem(e, 0.0, "room.face_top")
    assert face["kind"] == "face" and face["sides"] == 4 and face["area"] == pytest.approx(10)
    assert face["normal"] == [0, 0, 1] and face["backdrop"]


def test_the_geodesic_straightens_as_the_net_opens():
    from phasor.style.theme import load_theme

    e = _scene().engine
    seg = e.document.timeline.tracks["room.unfold"][0]
    t0, t1 = (e._snap.compiled.time_map.resolve(x) for x in (seg.t0, seg.t1))
    assert t1 - t0 == pytest.approx(load_theme(e.document.meta.theme).motion["morph"]["duration"])
    closed, half, flat = _sem(e, 0.0), _sem(e, (t0 + t1) / 2), _sem(e, t1 + 0.1)
    assert (closed["shape"], half["shape"], flat["shape"]) == ("solid", "opening", "net")
    assert closed["geodesic"]["length"] == pytest.approx(40 / 6, abs=1e-6)
    assert closed["geodesic"]["faces"] == ["west", "top", "south", "bottom", "east"]
    assert not closed["geodesic"]["straight"] and not half["geodesic"]["straight"]
    assert flat["geodesic"]["straight"]
    assert all(h["angle"] == 90.0 for h in flat["hinges"].values())
    assert 0 < half["hinges"]["west"]["angle"] < 90
    st = e.state_at(t1 + 0.1)
    path = np.asarray(st["room.geodesic"].params.points)
    assert np.linalg.norm(path[-1] - path[0]) == pytest.approx(40 / 6, rel=1e-6)
    # laid flat on the plane of the top, the root, which never moves
    assert np.ptp([o.impl.world_bounds(st).lo[2] for o in st.objects
                   if o.type == "face"]) == pytest.approx(0, abs=1e-9)  # fmt: skip
    np.testing.assert_allclose(st["room.face_top"].params.points,
                               e.state_at(0.0)["room.face_top"].params.points)  # fmt: skip


def test_pure_in_time():
    a = _scene().engine.inspect(1.37)
    assert a == _scene().engine.inspect(1.37)


def test_a_tree_that_cuts_the_geodesic_is_refused_and_a_plain_net_has_none():
    from phasor import Scene, net

    s = Scene("plain", passes=None)
    s.add(net.Polyhedron(id="cube"))
    sem = _sem(s.engine, 0.0, "cube")
    assert sem["root"] == "bottom" and "geodesic" not in sem and len(sem["hinges"]) == 5
    tree = {"top": "north", "north": "bottom", "south": "bottom", "west": "bottom",
            "east": "bottom"}  # fmt: skip
    s.add(net.Polyhedron(id="given", tree=tree, unfold=1.0))
    assert _sem(s.engine, 0.0, "given")["hinges"]["top"]["onto"] == "north"
    with pytest.raises(Exception, match="cuts the geodesic"):
        s.add(net.Polyhedron(id="cut", tree=tree, geodesic=[(0.5, 0.5, 0), (0.5, 0.5, 2)]))
    with pytest.raises(Exception, match="does not hang"):
        s.add(net.Polyhedron(id="loose", tree={"top": "north", "north": "top"}))


def test_any_convex_polyhedron_with_named_faces():
    from phasor import Scene, net

    v, _ = _tetra(2.0)
    s = Scene("tetra", passes=None)
    s.add(net.Polyhedron(id="tet", vertices=[tuple(x) for x in v], names=["a", "b", "c", "d"],
                         faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], unfold=1.0,
                         geodesic=[tuple((v[0] + v[1]) / 2), tuple((v[2] + v[3]) / 2)]))  # fmt: skip
    sem = _sem(s.engine, 0.0, "tet")
    assert sem["root"] == "a" and sem["geodesic"]["length"] == pytest.approx(2.0)
    assert sem["geodesic"]["straight"]
    assert sem["hinges"]["b"]["angle"] == pytest.approx(
        180 - np.degrees(np.arccos(1 / 3)), abs=1e-3
    )
    with pytest.raises(Exception, match="names"):
        s.add(net.Polyhedron(id="bad", vertices=[tuple(x) for x in v], names=["A", "b", "c", "d"],
                             faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))  # fmt: skip


# ---------------------------------------------------------------- the example


def test_the_example_matches_its_golden():
    import os
    import runpy

    from tests.conftest import gl_available

    e = runpy.run_path(str(ROOT / "examples" / "spider_and_fly.py"), run_name="x")["s"].engine
    assert not e.lint(strict=True)
    sem = _sem(e, 9.0)
    assert sem["shape"] == "net" and sem["geodesic"]["straight"]
    assert sem["geodesic"]["length"] * 6 == pytest.approx(40.0, abs=1e-5)
    if not gl_available():
        pytest.skip("needs OpenGL")
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    sheet = e.storyboard([2.5, 4.0, 5.5, 6.5, 9.0, 11.5], size=(480, 270))["image"]
    path = GOLDEN / e.pass_runner().renderer_class() / "spider_and_fly.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
