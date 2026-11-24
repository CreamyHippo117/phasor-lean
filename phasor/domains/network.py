"""Graphs and networks: imported from a file, given in a script, or made
by a generator, laid out in 2D or 3D and drawn as thousands of thin edges in one draw call.
In the prism look the edges glow, take their colour from the palette by depth or by an
attribute, and pile up light where they are dense.

    from phasor.domains import network as nw
    s.add(nw.Network(id="game", generator="tictactoe", layout="layers", dims=3))
    s.add(nw.Network(id="mine", path="friends.csv"))      # a file next to the scene
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from phasor.core import files
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.rng import rng
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import network_math as nm
from phasor.domains.strokes import disc_batch, segment_batch

SPRING_BUDGET = 2e8  # node pairs x iterations the spring layout may spend (big graphs: fewer)
DENSE = 1500  # above this many edges each edge gets fainter (1 / sqrt), so light piles up
#              where edges crowd instead of the whole graph burning out to white


@functools.lru_cache(maxsize=8)
def _graph(source: str, key: tuple) -> nm.Graph:
    kind, args = key[0], key[1:]
    if source == "file":
        return nm.read(args[0])
    if source == "edges":
        names, pairs = args
        return nm._build(list(names), [tuple(p) for p in pairs])
    if kind == "tree":
        return nm.tree(*args[:2])
    if kind == "grid":
        return nm.grid(*args[:2])
    if kind == "hypercube":
        return nm.hypercube(args[0])
    if kind == "tictactoe":
        return nm.tictactoe(args[0])
    oid, seed, n, p, k = args
    if kind == "random":
        return nm.random_graph(n, p, rng(oid, seed))
    return nm.small_world(n, k, p, rng(oid, seed))


@functools.lru_cache(maxsize=8)
def _positions(source: str, key: tuple, layout: str, dims: int, oid: str, seed: int,
               iterations: int, root: int) -> np.ndarray:  # fmt: skip
    g = _graph(source, key)
    if layout == "spring":
        iterations = int(min(iterations, max(20, SPRING_BUDGET / max(g.n, 1) ** 2)))
    pos = nm.layout(g, layout, dims, rng(oid + ".layout", seed), iterations, root)
    pos.setflags(write=False)
    return pos


@register
class Network(ObjectBase):
    """A graph: nodes joined by edges. Give ``path`` (a file next to the scene: an edge list,
    node-link JSON or GraphML), or ``edges`` (pairs of node names), or a ``generator``
    (``tree``, ``grid``, ``random``, ``small_world``, ``hypercube``, ``tictactoe``: every
    position of the game). ``layout``: ``spring``, ``layers`` (breadth first from ``root``:
    a game tree, a move per layer), ``radial`` (the root in the middle, each move a ring
    further out, branches fanning out as rays), ``circle``, ``random``; ``dims`` 2 or 3
    (3D: give the scene an orbit camera). ``color_by``: ``depth``, ``degree``, a node
    attribute from the file (``outcome``, ``move`` for tic-tac-toe), or empty for one
    colour. ``grow`` (0-1) shows that share of the edges, nearest the root first: animate
    it to grow the graph."""

    type: ClassVar[str] = "network"
    Files: ClassVar = ("path",)
    Style: ClassVar = {"colormap": ("colormap", "spectrum"), "edge": ("emphasis", "muted"),
                       "node": ("emphasis", "primary"), "glow": ("glow", "none"),
                       "stroke": ("stroke", "hairline")}  # fmt: skip
    Presets: ClassVar = {
        "plain": {"params": {"color_by": ""}, "style": {"glow": "none"}},
        # the prism look: glowing edges, colour from the palette by depth, light piling up
        "prism": {"params": {"color_by": "depth", "dim": 0.75}, "style": {"glow": "soft"}},
    }

    class Params(ParamsBase):
        path: str = ""  # a graph file next to the scene
        edges: tuple[tuple[str, str], ...] = ()  # or the edges themselves
        nodes: tuple[str, ...] = ()  # node names in order (optional with ``edges``)
        generator: Literal["", "tree", "grid", "random", "small_world", "hypercube",
                           "tictactoe"] = ""  # fmt: skip
        branching: int = Field(3, ge=1, le=20)  # tree
        depth: int = Field(4, ge=1, le=12)  # tree
        rows: int = Field(10, ge=1, le=300)  # grid
        cols: int = Field(10, ge=1, le=300)  # grid
        n: int = Field(200, ge=2, le=20000)  # random, small_world
        p: float = Field(0.02, ge=0, le=1)  # random: edge chance; small_world: rewiring
        k: int = Field(4, ge=2, le=50)  # small_world neighbours
        cube_dim: int = Field(6, ge=1, le=14)  # hypercube: 2^cube_dim corners
        symmetric: bool = True  # tictactoe: rotated/reflected boards count once (765, not 5,478)
        layout: Literal["spring", "layers", "radial", "circle", "random"] = "spring"
        dims: Literal[2, 3] = 2
        iterations: int = Field(200, ge=1, le=2000)
        root: int = Field(0, ge=0)
        seed: int = 0
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        size: float = Field(3.0, gt=0)  # half the width of the layout, in world units
        color_by: str = ""
        grow: float = Field(1.0, ge=0, le=1)
        node_size: float = Field(0.0, ge=0)  # world radius of the node dots; 0: none
        hue: float = 0.0  # turns the palette
        exposure: float = Field(1.0, gt=0, le=10)  # light per edge; dense graphs share it out
        cross_edges: float = Field(1.0, ge=0, le=1)  # brightness of edges off the spanning
        # tree (in a game: the same position by another order of moves); 0 hides them

        @model_validator(mode="after")
        def _one_source(self):
            given = [bool(self.path), bool(self.edges), bool(self.generator)]
            if sum(given) != 1:
                raise ValueError("give exactly one of path, edges or generator")
            if self.path:  # relative and inside the scene's folder (core/files.py)
                object.__setattr__(self, "path", files.relative_path(self.path))
            return self

    # ------------------------------------------------------------ data

    def _key(self, state: SceneState):
        p = self.me(state).params
        if p.path:  # in the scene's folder; the key changes when the file does (re-read)
            try:
                f = files.resolve(state.base_dir, p.path)
                st = f.stat()
                return "file", ("file", str(f), st.st_mtime_ns, st.st_size)
            except (OSError, ValueError):
                return "edges", ("edges", (), ())  # missing: an empty graph (lint missing_file)
        if p.edges:
            return "edges", ("edges", tuple(p.nodes), tuple(p.edges))
        g = p.generator
        if g == "tree":
            return "gen", (g, p.branching, p.depth)
        if g == "grid":
            return "gen", (g, p.rows, p.cols)
        if g == "hypercube":
            return "gen", (g, p.cube_dim)
        if g == "tictactoe":
            return "gen", (g, p.symmetric)
        return "gen", (g, self.id, p.seed, p.n, p.p, p.k)

    def graph(self, state: SceneState) -> nm.Graph:
        return _graph(*self._key(state))

    def positions(self, state: SceneState) -> np.ndarray:
        """World positions of the nodes, (n, 3)."""
        p = self.me(state).params
        src, key = self._key(state)
        g = _graph(src, key)
        local = _positions(src, key, p.layout, p.dims, self.id, p.seed, p.iterations,
                           min(p.root, max(g.n - 1, 0)))  # fmt: skip
        return np.asarray(p.pos) + local * p.size

    def _order(self, state: SceneState) -> np.ndarray:
        """Edges nearest the root first (by the depth of their deeper end), for ``grow``."""
        g = self.graph(state)
        p = self.me(state).params
        d = g.depth(min(p.root, max(g.n - 1, 0)))
        return np.argsort(np.maximum(d[g.edges[:, 0]], d[g.edges[:, 1]]), kind="stable")

    def _values(self, state: SceneState) -> np.ndarray | None:
        """A 0..1 value per node for ``color_by``, or None for one colour."""
        p = self.me(state).params
        g = self.graph(state)
        if not p.color_by:
            return None
        if p.color_by == "depth":
            v = g.depth(min(p.root, max(g.n - 1, 0))).astype(np.float64)
        elif p.color_by == "degree":
            v = np.log1p(g.degree().astype(np.float64))
        elif p.color_by in g.node_attrs:
            v = np.nan_to_num(g.node_attrs[p.color_by], nan=0.0)
        else:
            raise ValueError(
                f"{self.id}: no node attribute {p.color_by!r}; "
                f"have depth, degree, {', '.join(sorted(g.node_attrs)) or '-'}"
            )
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    # ------------------------------------------------------------ object contract

    def world_bounds(self, state: SceneState) -> Bounds:
        pos = self.positions(state)
        if len(pos) == 0:
            p = np.asarray(self.me(state).params.pos)
            return Bounds.of_points([p, p])
        return Bounds.of_points(pos)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pos = self.positions(state)
        g = self.graph(state)
        root = min(self.me(state).params.root, max(g.n - 1, 0))
        return {"center": pos.mean(axis=0) if len(pos) else np.zeros(3),
                "root": pos[root] if len(pos) else np.zeros(3)}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = self.graph(state)
        deg = g.degree()
        return {
            "kind": "network", "nodes": g.n, "edges": len(g.edges),
            "shown_edges": round(p.grow * len(g.edges)), "layout": p.layout,
            "dims": p.dims, "max_degree": int(deg.max()) if g.n else 0,
            "depth": int(g.depth(min(p.root, max(g.n - 1, 0))).max()) if g.n else 0,
            "attributes": sorted(g.node_attrs),
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        g = self.graph(state)
        if len(g.edges) == 0 and p.node_size <= 0:
            return
        cam = state.camera_for(me)
        pos = self.positions(state)
        order = self._order(state)[: round(p.grow * len(g.edges))]
        e = g.edges[order]
        share = min(1.0, p.exposure * np.sqrt(DENSE / max(len(g.edges), 1)))
        glow = 1.0 + (float(me.style.get("glow", 1.0)) - 1.0) * share
        vals = self._values(state)
        if vals is None:
            edge_col: Any = (*me.style["edge"][:3], 1.0)
            node_col: Any = (*me.style["node"][:3], 1.0)
        else:
            lut = np.asarray(me.style["colormap"], dtype=np.float64)

            def paint(v):
                idx = ((v * 0.85 + p.hue) % 1.0 * (len(lut) - 1)).astype(int)
                return np.c_[lut[idx], np.ones(len(v))]

            edge_col = np.stack([paint(vals[e[:, 0]]), paint(vals[e[:, 1]])], axis=1)
            node_col = paint(vals)
        width = float(me.style["stroke"])
        alpha = p.opacity * max(share, 0.12)
        if p.cross_edges < 1:  # the spanning tree full, the cross links fainter
            tree = nm.tree_edges(g, min(p.root, max(g.n - 1, 0)))[order]
            for keep, a_ in ((tree, alpha * 0.7), (~tree, alpha * p.cross_edges)):
                if keep.any() and a_ > 0:
                    cols = edge_col if np.ndim(edge_col) == 1 else edge_col[keep]
                    segment_batch(ctx, cam, pos[e[keep, 0]], pos[e[keep, 1]], width,
                                  colors=cols, glow=glow, opacity=a_)  # fmt: skip
        else:
            segment_batch(ctx, cam, pos[e[:, 0]], pos[e[:, 1]], width, colors=edge_col,
                          glow=glow, opacity=alpha)  # fmt: skip
        if p.node_size > 0:
            shown = np.unique(e) if p.grow < 1 else np.arange(g.n)
            nc = node_col if np.ndim(node_col) == 1 else node_col[shown]
            disc_batch(ctx, cam, pos[shown], p.node_size, colors=nc, glow=glow,
                       opacity=p.opacity)  # fmt: skip
