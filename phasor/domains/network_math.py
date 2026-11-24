"""Graphs as data (pure NumPy, no drawing): reading them from files, making them, and laying
them out in 2D or 3D. ``domains/network.py`` draws them.

A graph here is ``Graph(names, edges, node_attrs, edge_weights)``: node names in order, edges
as an (E, 2) array of node indices, per-node attributes (a dict of name -> array), and edge
weights.
"""

from __future__ import annotations

import csv
import functools
import json
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

FORMATS = ("edges (.csv .tsv .txt: one 'a,b[,weight]' per line)", "JSON (.json: node-link)",
           "GraphML (.graphml)")  # fmt: skip
MAX_NODES = 50_000
TALL = 1.0  # 3D layers: height to width


@dataclass(frozen=True)
class Graph:
    names: tuple[str, ...]
    edges: np.ndarray  # (E, 2) int
    node_attrs: dict[str, np.ndarray] = field(default_factory=dict, hash=False)
    weights: np.ndarray | None = field(default=None, hash=False)

    @property
    def n(self) -> int:
        return len(self.names)

    def degree(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=np.int64)
        np.add.at(d, self.edges.ravel(), 1)
        return d

    def depth(self, root: int = 0) -> np.ndarray:
        """Hops from ``root`` (breadth first); unreachable nodes get the largest depth + 1."""
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        d = np.full(self.n, -1, dtype=np.int64)
        if self.n == 0:
            return d
        d[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        d[d < 0] = d.max() + 1
        return d


def _build(names: list[str], pairs: list[tuple[str, str]], weights: list[float] | None = None,
           attrs: dict[str, dict[str, Any]] | None = None) -> Graph:  # fmt: skip
    index = {k: i for i, k in enumerate(names)}
    for a, b in pairs:
        for k in (a, b):
            if k not in index:
                index[k] = len(names)
                names.append(k)
    if len(names) > MAX_NODES:
        raise ValueError(f"graph has {len(names)} nodes; at most {MAX_NODES} are drawn")
    edges = np.array([(index[a], index[b]) for a, b in pairs], dtype=np.int64).reshape(-1, 2)
    node_attrs: dict[str, np.ndarray] = {}
    for key in sorted({k for d in (attrs or {}).values() for k in d}):
        vals = [(attrs or {}).get(n, {}).get(key) for n in names]
        try:
            node_attrs[key] = np.array([np.nan if v is None else float(v) for v in vals])
        except (TypeError, ValueError):
            cats = sorted({str(v) for v in vals if v is not None})
            node_attrs[key] = np.array([cats.index(str(v)) if v is not None else -1
                                        for v in vals], dtype=np.float64)  # fmt: skip
    w = np.asarray(weights, dtype=np.float64) if weights else None
    return Graph(tuple(names), edges, node_attrs, w)


def read(path: str | Path) -> Graph:
    """A graph from a file: an edge list (``a,b`` or ``a,b,weight`` per line; tabs or spaces
    work too; ``#`` starts a comment), node-link JSON (``{"nodes": [{"id": ..., ...}],
    "links" or "edges": [{"source": ..., "target": ...}]}``, extra node fields become
    attributes) or GraphML."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        names = [str(nd["id"] if isinstance(nd, dict) else nd) for nd in nodes]
        attrs = {str(nd["id"]): {k: v for k, v in nd.items() if k != "id"}
                 for nd in nodes if isinstance(nd, dict)}  # fmt: skip
        links = data.get("links", data.get("edges", []))
        pairs = [(str(e["source"]), str(e["target"])) for e in links]
        weights = [float(e.get("weight", 1.0)) for e in links]
        return _build(names, pairs, weights, attrs)
    if ext == ".graphml":
        root = ET.parse(p).getroot()
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        keys = {k.get("id"): k.get("attr.name") or k.get("id") for k in root.iter(f"{ns}key")}
        names, attrs, pairs = [], {}, []
        for nd in root.iter(f"{ns}node"):
            names.append(nd.get("id"))
            attrs[nd.get("id")] = {keys.get(d.get("key"), d.get("key")): d.text
                                   for d in nd.iter(f"{ns}data")}  # fmt: skip
        for e in root.iter(f"{ns}edge"):
            pairs.append((e.get("source"), e.get("target")))
        return _build(names, pairs, None, attrs)
    if ext in (".csv", ".tsv", ".txt", ".edges", ".edgelist"):
        pairs, weights = [], []
        with p.open(encoding="utf-8", newline="") as f:
            text = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        delim = "\t" if ext == ".tsv" or (text and "\t" in text[0]) else ","
        rows = csv.reader(text, delimiter=delim) if delim in (text[0] if text else "") else \
            (ln.split() for ln in text)  # fmt: skip
        for row in rows:
            row = [c.strip() for c in row if c.strip()]
            if len(row) < 2 or row[:2] == ["source", "target"]:
                continue
            pairs.append((row[0], row[1]))
            weights.append(float(row[2]) if len(row) > 2 and _is_number(row[2]) else 1.0)
        return _build([], pairs, weights)
    raise ValueError(f"cannot read {p.name}: graph files are {', '.join(FORMATS)}")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- making graphs


def tree(branching: int, depth: int) -> Graph:
    names, pairs, level = ["0"], [], [0]
    for _ in range(depth):
        nxt = []
        for u in level:
            for _ in range(branching):
                v = len(names)
                names.append(str(v))
                pairs.append((str(u), str(v)))
                nxt.append(v)
        level = nxt
    return _build(names, pairs)


def grid(rows: int, cols: int) -> Graph:
    names = [f"{r},{c}" for r in range(rows) for c in range(cols)]
    pairs = [(f"{r},{c}", f"{r},{c + 1}") for r in range(rows) for c in range(cols - 1)]
    pairs += [(f"{r},{c}", f"{r + 1},{c}") for r in range(rows - 1) for c in range(cols)]
    return _build(names, pairs)


def random_graph(n: int, p: float, rng: Generator) -> Graph:
    """Erdos-Renyi: each pair joined with probability ``p``."""
    i, j = np.triu_indices(n, 1)
    keep = rng.random(len(i)) < p
    return _build(
        [str(k) for k in range(n)],
        [(str(a), str(b)) for a, b in zip(i[keep], j[keep], strict=True)],
    )


def small_world(n: int, k: int, p: float, rng: Generator) -> Graph:
    """Watts-Strogatz: a ring where each node joins its ``k`` nearest, each edge rewired to a
    random node with probability ``p``."""
    pairs = []
    for u in range(n):
        for s in range(1, k // 2 + 1):
            v = (u + s) % n
            if rng.random() < p:
                v = int(rng.integers(n))
                if v == u:
                    v = (u + s) % n
            pairs.append((str(u), str(v)))
    return _build([str(k_) for k_ in range(n)], pairs)


def hypercube(dim: int) -> Graph:
    n = 1 << dim
    pairs = [(str(u), str(u ^ (1 << b))) for u in range(n) for b in range(dim) if u < u ^ (1 << b)]
    return _build([str(u) for u in range(n)], pairs)


_SYMMETRIES = (  # the 8 rotations and reflections of the board, as index maps
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
)


def _canonical(b: str) -> str:
    return min("".join(b[i] for i in sym) for sym in _SYMMETRIES)


@functools.cache
def tictactoe(symmetric: bool = True) -> Graph:
    """Every position of tic-tac-toe reachable from the empty board, joined to the positions
    one move later. ``symmetric``: boards that are rotations or reflections of each other
    count as one (765 positions); otherwise all 5,478. Node attributes: ``move`` (marks on
    the board) and ``outcome`` with perfect play from there (1: X wins, 0: draw, -1: O wins)."""
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    key = _canonical if symmetric else (lambda b: b)

    def winner(b: str) -> int:
        for x, y, z in lines:
            if b[x] != "." and b[x] == b[y] == b[z]:
                return 1 if b[x] == "X" else -1
        return 0

    states: dict[str, int] = {}
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    order: list[str] = []
    q = deque(["........."])
    states["........."] = 0
    while q:
        b = q.popleft()
        order.append(b)
        if winner(b) or "." not in b:
            continue
        mark = "X" if b.count("X") == b.count("O") else "O"
        for i, c in enumerate(b):
            if c == ".":
                nb = key(b[:i] + mark + b[i + 1 :])
                if (b, nb) in seen:
                    continue
                seen.add((b, nb))
                pairs.append((b, nb))
                if nb not in states:
                    states[nb] = len(states)
                    q.append(nb)

    @functools.cache
    def value(b: str) -> int:
        w = winner(b)
        if w or "." not in b:
            return w
        mark = "X" if b.count("X") == b.count("O") else "O"
        vals = [value(b[:i] + mark + b[i + 1 :]) for i, c in enumerate(b) if c == "."]
        return max(vals) if mark == "X" else min(vals)

    attrs = {b: {"move": 9 - b.count("."), "outcome": value(b)} for b in order}
    return _build(list(order), pairs, None, attrs)


GENERATORS = ("tree", "grid", "random", "small_world", "hypercube", "tictactoe")


# ---------------------------------------------------------------- layouts


def layout(g: Graph, kind: str, dims: int, rng: Generator, iterations: int = 200,
           root: int = 0) -> np.ndarray:  # fmt: skip
    """Node positions (n, 3), centred, fitting in [-1, 1] on the widest side.

    ``spring``: force-directed (Fruchterman-Reingold), neighbours pulled together, everyone
    pushed apart. ``radial``: the root in the middle, each breadth-first layer a ring further
    out, each node given a wedge of angle in proportion to the nodes below it (a spanning
    tree), so branches fan out as rays (3D: the rings sink into a bowl).
    ``layers``: breadth-first layers from ``root`` (a game tree: one move per
    layer), each layer a ring (3D: a disc, depth along z) or a row. ``circle`` and ``random``.
    """
    n = g.n
    if n == 0:
        return np.zeros((0, 3))
    if kind == "circle":
        a = 2 * np.pi * np.arange(n) / n
        pos = np.c_[np.cos(a), np.sin(a), np.zeros(n)]
    elif kind == "random":
        pos = np.c_[rng.uniform(-1, 1, (n, dims)), np.zeros((n, 3 - dims))]
    elif kind == "layers":
        pos = _layers(g, dims, root)
    elif kind == "radial":
        pos = _radial(g, dims, root)
    elif kind == "spring":
        pos = _spring(g, dims, rng, iterations)
    else:
        raise ValueError(f"layout must be spring, layers, radial, circle or random, got {kind!r}")
    pos = pos - pos.mean(axis=0)
    span = np.abs(pos).max()
    return pos / span if span > 0 else pos


def _layers(g: Graph, dims: int, root: int) -> np.ndarray:
    """Breadth-first layers, each ordered by where its nodes' parents sit (the mean angle
    of their neighbours one layer up), so a position's moves fan out from it instead of
    crossing the whole graph. 3D: each layer a ring, wider as it holds more, stacked down z
    (a cone of play); 2D: each layer a row."""
    depth = g.depth(root)
    top = max(int(depth.max()), 1)
    up: list[list[int]] = [[] for _ in range(g.n)]
    for a, b in g.edges:
        if depth[a] + 1 == depth[b]:
            up[b].append(a)
        elif depth[b] + 1 == depth[a]:
            up[a].append(b)
    turn = np.zeros(g.n)  # each node's place round its layer, 0..1
    pos = np.zeros((g.n, 3))
    biggest = max(np.bincount(depth).max(), 1)
    for d in range(int(depth.max()) + 1):
        idx = np.flatnonzero(depth == d)
        if len(idx) == 0:
            continue
        if d == 0:
            key = np.zeros(len(idx))
        else:  # circular mean of the parents' places
            key = np.array([np.angle(np.exp(2j * np.pi * turn[up[i]]).mean()) / (2 * np.pi) % 1
                            if up[i] else 0.0 for i in idx])  # fmt: skip
        order = idx[np.argsort(key, kind="stable")]
        m = len(order)
        turn[order] = (np.arange(m) + 0.5) / m
        if dims == 3:
            r = 0.1 + 0.9 * np.sqrt(m / biggest) if m > 1 else 0.0
            a = 2 * np.pi * turn[order]
            pos[order] = np.c_[r * np.cos(a), r * np.sin(a),
                               np.full(m, TALL * (1 - 2 * d / top))]  # fmt: skip
        else:
            pos[order, 0] = (turn[order] - 0.5) * 2
            pos[order, 1] = 1 - 2 * d / top
    return pos


def spanning_tree(g: Graph, root: int = 0) -> np.ndarray:
    """Each node's parent in a breadth-first spanning tree from ``root`` (-1: the root, or
    unreachable). Edges not in it are cross links: in a game, the same position reached by
    another order of moves."""
    depth = g.depth(root)
    adj: list[list[int]] = [[] for _ in range(g.n)]
    for a, b in g.edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = np.full(g.n, -1)
    for u in np.argsort(depth, kind="stable"):
        for v in adj[u]:
            if depth[v] == depth[u] + 1 and parent[v] < 0:
                parent[v] = u
    return parent


def tree_edges(g: Graph, root: int = 0) -> np.ndarray:
    """A bool per edge: True where it is in the breadth-first spanning tree."""
    parent = spanning_tree(g, root)
    a, b = g.edges[:, 0], g.edges[:, 1]
    return (parent[b] == a) | (parent[a] == b)


def _radial(g: Graph, dims: int, root: int) -> np.ndarray:
    depth = g.depth(root)
    top = max(int(depth.max()), 1)
    parent = spanning_tree(g, root)
    children: list[list[int]] = [[] for _ in range(g.n)]
    for v in np.argsort(depth, kind="stable"):
        if parent[v] >= 0:
            children[parent[v]].append(int(v))
    weight = np.ones(g.n)  # nodes in each subtree
    for u in np.argsort(-depth, kind="stable"):
        if parent[u] >= 0:
            weight[parent[u]] += weight[u]
    start, span = np.zeros(g.n), np.zeros(g.n)
    span[root] = 1.0
    for u in np.argsort(depth, kind="stable"):
        at = start[u]
        total = sum(weight[c] for c in children[u])
        for c in children[u]:
            span[c] = span[u] * weight[c] / total
            start[c] = at
            at += span[c]
    orphans = np.flatnonzero((parent < 0) & (np.arange(g.n) != root))
    start[orphans] = np.arange(len(orphans)) / max(len(orphans), 1)
    ang = 2 * np.pi * (start + span / 2)
    r = depth / top
    z = -0.6 * r * r if dims == 3 else np.zeros(g.n)
    return np.c_[r * np.cos(ang), r * np.sin(ang), z]


def _spring(g: Graph, dims: int, rng: Generator, iterations: int) -> np.ndarray:
    n = g.n
    pos = rng.uniform(-1, 1, (n, dims))
    k = 1.0 / np.sqrt(n)
    t = 0.1
    a, b = g.edges[:, 0], g.edges[:, 1]
    block = 2048  # repulsion in blocks, so 10k+ nodes fit in memory
    for it in range(iterations):
        disp = np.zeros_like(pos)
        for s in range(0, n, block):
            d = pos[s : s + block, None, :] - pos[None, :, :]
            dist2 = np.maximum((d * d).sum(-1), 1e-9)
            disp[s : s + block] += (d * (k * k / dist2)[..., None]).sum(1)
        e = pos[a] - pos[b]
        dist = np.maximum(np.linalg.norm(e, axis=1), 1e-9)
        f = e * (dist / k)[:, None]
        np.add.at(disp, a, -f)
        np.add.at(disp, b, f)
        length = np.maximum(np.linalg.norm(disp, axis=1), 1e-9)
        pos += disp / length[:, None] * np.minimum(length, t)[:, None]
        t = 0.1 * (1 - it / iterations) + 1e-3
    return np.c_[pos, np.zeros((n, 3 - dims))]
