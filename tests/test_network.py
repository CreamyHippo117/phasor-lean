"""Graphs and networks: reading, making, laying out and drawing them."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Engine, Scene
from phasor.core.issues import DocumentError
from phasor.core.schema import Document, Meta
from phasor.domains import network as nw
from phasor.domains import network_math as nm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _scene(tmp_path, **kw):
    s = Scene("net", passes=None)
    s.engine = Engine(Document(meta=Meta(title="net")), passes=None, base_dir=tmp_path, **kw)
    return s


def test_edge_lists_json_and_graphml_read_the_same_graph(tmp_path):
    (tmp_path / "g.csv").write_text("# a triangle and a tail\na,b\nb,c,2.5\nc,a\nc,d\n",
                                    encoding="utf-8")  # fmt: skip
    (tmp_path / "g.tsv").write_text("a\tb\nb\tc\nc\ta\nc\td\n", encoding="utf-8")
    (tmp_path / "g.json").write_text(json.dumps({
        "nodes": [{"id": "a", "team": "red"}, {"id": "b", "team": "blue"}, {"id": "c"},
                  {"id": "d", "team": "red"}],
        "links": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"},
                  {"source": "c", "target": "a"}, {"source": "c", "target": "d"}]}),
        encoding="utf-8")  # fmt: skip
    (tmp_path / "g.graphml").write_text(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="k0" for="node" '
        'attr.name="size"/><graph edgedefault="undirected">'
        '<node id="a"><data key="k0">1</data></node><node id="b"/><node id="c"/><node id="d"/>'
        '<edge source="a" target="b"/><edge source="b" target="c"/><edge source="c" target="a"/>'
        '<edge source="c" target="d"/></graph></graphml>', encoding="utf-8")  # fmt: skip
    for name in ("g.csv", "g.tsv", "g.json", "g.graphml"):
        g = nm.read(tmp_path / name)
        assert g.names == ("a", "b", "c", "d") and len(g.edges) == 4, name
        assert sorted(g.degree()) == [1, 2, 2, 3], name
    assert nm.read(tmp_path / "g.csv").weights.tolist() == [1.0, 2.5, 1.0, 1.0]
    team = nm.read(tmp_path / "g.json").node_attrs["team"]
    assert (team[0] == team[3] != team[1] and np.isnan(team[2])) or team[2] == -1
    with pytest.raises(ValueError, match="graph files are"):
        nm.read(tmp_path / "g.xyz")


def test_generators_have_the_right_size():
    assert (nm.tree(3, 4).n, len(nm.tree(3, 4).edges)) == (1 + 3 + 9 + 27 + 81, 120)
    assert len(nm.grid(4, 5).edges) == 4 * 4 + 3 * 5
    h = nm.hypercube(5)
    assert h.n == 32 and len(h.edges) == 80 and set(h.degree()) == {5}
    rng = np.random.default_rng(0)
    assert nm.random_graph(200, 0.0, rng).edges.size == 0
    assert len(nm.small_world(100, 4, 0.0, rng).edges) == 200


def test_tictactoe_is_every_position_and_perfect_play_is_a_draw():
    assert nm.tictactoe(symmetric=False).n == 5478  # every reachable board
    g = nm.tictactoe(symmetric=False)
    assert g.node_attrs["outcome"][0] == 0  # the empty board: a draw
    assert g.node_attrs["move"].max() == 9 and g.depth().max() == 9
    # after X takes the centre, O must answer in a corner: the edge squares lose
    centre = g.names.index("....X....")
    assert g.node_attrs["outcome"][centre] == 0
    assert g.node_attrs["outcome"][g.names.index(".O..X....")] == 1  # an edge: X wins
    small = nm.tictactoe()  # up to rotation and reflection
    assert small.n == 765 and small.node_attrs["outcome"][0] == 0
    assert sorted(np.bincount(small.depth())) == sorted([1, 3, 12, 38, 108, 174, 204, 153, 57, 15])


def test_layouts_are_deterministic_centred_and_in_range():
    g = nm.tree(3, 4)
    for kind in ("spring", "layers", "radial", "circle", "random"):
        for dims in (2, 3):
            a = nm.layout(g, kind, dims, np.random.default_rng(1), 50)
            b = nm.layout(g, kind, dims, np.random.default_rng(1), 50)
            np.testing.assert_array_equal(a, b)
            assert np.abs(a).max() == pytest.approx(1.0) and a.shape == (g.n, 3)
            if dims == 2:
                assert np.all(a[:, 2] == 0)


def test_layered_layouts_keep_moves_next_to_the_position_they_came_from():
    g = nm.tictactoe(symmetric=False)
    pos = nm.layout(g, "layers", 3, np.random.default_rng(0))
    e = pos[g.edges[:, 0]] - pos[g.edges[:, 1]]
    ordered = np.linalg.norm(e[:, :2], axis=1).mean()  # sideways length of each move
    rng = np.random.default_rng(0)
    shuffled = pos.copy()
    for d in np.unique(g.depth()):
        idx = np.flatnonzero(g.depth() == d)
        shuffled[idx] = pos[rng.permutation(idx)]
    e2 = shuffled[g.edges[:, 0]] - shuffled[g.edges[:, 1]]
    assert ordered < 0.6 * np.linalg.norm(e2[:, :2], axis=1).mean()


def test_radial_layout_fans_branches_out_and_the_spanning_tree_covers_every_node():
    g = nm.tictactoe()
    tree = nm.tree_edges(g)
    assert tree.sum() == g.n - 1  # a spanning tree: one edge into every node but the root
    pos = nm.layout(g, "radial", 2, np.random.default_rng(0))
    r = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)
    assert np.corrcoef(r, g.depth())[0, 1] > 0.99  # each move a ring further out
    e = pos[g.edges[tree, 0]] - pos[g.edges[tree, 1]]
    cross = pos[g.edges[~tree, 0]] - pos[g.edges[~tree, 1]]
    assert np.linalg.norm(e, axis=1).mean() < np.linalg.norm(cross, axis=1).mean()  # local


def test_a_network_from_a_file_next_to_the_scene(tmp_path):
    (tmp_path / "friends.csv").write_text("ann,bo\nbo,cy\ncy,ann\ncy,dee\n", encoding="utf-8")
    s = _scene(tmp_path)
    s.add(nw.Network(id="f", path="friends.csv", layout="circle", size=2.0))
    sem = s.engine.inspect(0.0, id="f")["objects"][0]["semantic"]
    assert (sem["nodes"], sem["edges"], sem["max_degree"]) == (4, 4, 3)
    b = s.engine.state_at(0.0)["f"].impl.world_bounds(s.engine.state_at(0.0))
    assert max(np.abs(b.lo).max(), np.abs(b.hi).max()) == pytest.approx(2.0)
    s.add(nw.Network(id="g", path="nobody.csv"))  # missing: lint warning, an empty graph
    assert any(i.code == "missing_file" for i in s.engine.lint())
    assert s.engine.inspect(0.0, id="g")["objects"][0]["semantic"]["nodes"] == 0
    with pytest.raises(DocumentError, match="folder|absolute"):
        s.add(nw.Network(id="h", path="../outside.csv"))


def test_a_network_needs_exactly_one_source_and_a_known_colour():
    s = Scene("n", passes=None)
    with pytest.raises(DocumentError, match="exactly one of path, edges or generator"):
        s.add(nw.Network(id="a"))
    with pytest.raises(DocumentError, match="exactly one"):
        s.add(nw.Network(id="b", generator="tree", edges=(("x", "y"),)))
    s.add(nw.Network(id="c", edges=(("x", "y"), ("y", "z")), color_by="nonsense"))
    with pytest.raises(ValueError, match="no node attribute 'nonsense'"):
        s.engine.state_at(0.0)["c"].impl._values(s.engine.state_at(0.0))


def test_grow_shows_the_edges_nearest_the_root_first():
    s = Scene("grow", passes=None)
    s.add(nw.Network(id="t", generator="tree", branching=2, depth=4, grow=0.2))
    st = s.engine.state_at(0.0)
    obj = st["t"].impl
    g = obj.graph(st)
    first = obj._order(st)[: round(0.2 * len(g.edges))]
    d = g.depth()
    assert np.maximum(d[g.edges[first, 0]], d[g.edges[first, 1]]).max() <= 2
    assert s.engine.inspect(0.0, id="t")["objects"][0]["semantic"]["shown_edges"] == 6


@needs_gl
def test_a_prism_network_glows_and_a_dense_one_does_not_burn_out():
    def emit(**kw):
        s = Scene("glow")
        s.set("camera.height", 7.0)
        s.add(nw.Network(id="n", size=3.0, **kw))
        r = s.engine.pass_runner()
        return r.color_and_emission(s.engine.state_at(0.0), (160, 90))

    pic, light = emit(generator="tree", branching=3, depth=3, layout="layers")
    assert light[..., :3].max() == 0.0  # plain: paint only
    pic, light = emit(generator="tree", branching=3, depth=3, layout="layers", preset="prism")
    assert light[..., :3].max() > 0.2
    pic, _ = emit(generator="tictactoe", symmetric=False, layout="layers", preset="prism")
    assert (pic[..., :3] > 4.0).mean() < 0.02  # 16k glowing edges share their light out


@needs_gl
@pytest.mark.slow
def test_game_tree_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "game_tree.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([2.5, 5.5, 11.0], size=(320, 180))["image"]
    _golden("game_tree", sheet, e.pass_runner().renderer_class())


def test_the_ai_can_make_and_load_networks(tmp_path):
    from phasor.workspace import ToolError, Workspace

    root = tmp_path / "proj"
    root.mkdir()
    (root / "g.csv").write_text("a,b\nb,c\n", encoding="utf-8")
    (tmp_path / "outside.csv").write_text("a,b\n", encoding="utf-8")
    ws = Workspace(root)
    ws.scene_new("graphs")
    ok = ws.scene_edit({"op": "add_object", "object": {"id": "g", "type": "network",
                        "params": {"path": "g.csv"}}})  # fmt: skip
    assert ok["ok"]
    # refused by the path rule (core/files.py) or by the workspace check behind it
    try:
        r = ws.scene_edit({"op": "add_object", "object": {"id": "h", "type": "network",
                           "params": {"path": "../outside.csv"}}})  # fmt: skip
        assert not r["ok"] and "folder" in str(r)
    except ToolError as e:
        assert "outside" in str(e)
