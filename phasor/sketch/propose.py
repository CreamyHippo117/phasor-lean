"""From strokes to proposals: what each drawn shape should become in the scene.

A proposal is plain data: a one-line summary, the patch ops that make it (applied through
``Engine.apply`` like every other edit, so it can be undone), any verbs to run after them
(``Workspace.animate``) and what was worked out on the way (``data``). Nothing changes until
the person accepts it (``accept``).

What a shape proposes depends on what it is drawn over (the scene knows what is under every
stroke):

- a loop (``circle`` or ``closed``) over a domain coloring: a **contour**: the stroke
  smoothed into a glowing closed Line, traced from the playhead, with its contour integral
  worked out two ways, numerically along the drawn path and exactly as 2 pi i times the
  residues it encloses (rational functions), and a label with the value;
- a loop around other objects: **emphasize** them (the ``highlight`` color role);
- a loop over nothing: a closed curve (``circle``: the fitted circle);
- a ``dot``: a Dot, snapped onto a pole or zero of the function under it when one is near;
- a ``line``: a Line; an ``arrow``: an Arrow, or, when it starts on an object that has a
  position, **move** that object to where the arrow ends;
- a ``curve``: a Line through it (``wavy`` ones say so);
- a ``scribble`` over objects: **remove** them (paint-over: cross it out);
- ``handwriting``: a picture of the strokes for an AI (or the person) to read; ``transcribe``
  turns what it says (LaTeX or text) into a Text where it was written. There is no offline
  handwriting recognizer: the reader is whoever looks at the picture.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from phasor.core.ids import ID_RE
from phasor.engine import Engine
from phasor.sketch.strokes import DOT_PX, Shape, group, resample

SNAP_PX = 14.0  # a dot this near a pole or zero (in pixels) lands on it
POLE_NEAR_SHARE = 0.03  # a contour this near a pole (share of its size) warns


@dataclass
class Proposal:
    kind: str
    summary: str
    ops: list[dict[str, Any]] = field(default_factory=list)
    animate: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    image: bytes | None = None  # handwriting: a PNG of the strokes, to read
    strokes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON data (the picture is left out: ``image`` holds it)."""
        return {"kind": self.kind, "summary": self.summary, "ops": self.ops,
                "animate": self.animate, "data": _plain(self.data),
                "strokes": list(self.strokes), "has_image": self.image is not None}  # fmt: skip


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, complex):
        return [round(v.real, 9), round(v.imag, 9)]
    if isinstance(v, np.floating | np.integer):
        return v.item()
    return v


class Sketch:
    """Strokes drawn over the scene at time ``t``, in output pixels (row 0 at the top)."""

    def __init__(self, engine: Engine, t: float = 0.0) -> None:
        self.engine, self.t = engine, float(t)
        self.state = engine.state_at(self.t)
        self._taken: set[str] = set()

    # ---------------------------------------------------------- coordinates

    def to_world(self, px: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Output pixels -> world points on the plane at height ``z`` (2D: the plane)."""
        st = self.state
        return st.camera.from_screen(px, st.width, st.height, z)

    def to_pixels(self, world: np.ndarray) -> np.ndarray:
        st = self.state
        return st.camera.to_screen(np.atleast_2d(np.asarray(world, dtype=np.float64)),
                                   st.width, st.height)  # fmt: skip

    # ---------------------------------------------------------- what is under a stroke

    def _hits(self, px: np.ndarray) -> list[str]:
        """Objects under any of these pixels, topmost first, backdrops (fields) left out."""
        seen: list[str] = []
        for x, y in px:
            for i in self.engine.hit_test(float(x), float(y), self.t):
                if i not in seen and not self._backdrop(i):
                    seen.append(i)
        return seen

    def _backdrop(self, object_id: str) -> bool:
        o = self.state[object_id]
        return bool(o.impl.semantics(self.state).get("backdrop"))

    def _domain_under(self, world: np.ndarray):
        """The domain coloring most of these world points sit on, or None."""
        for o in reversed(self.state.draw_order()):
            if o.type != "domain_coloring" or o.params.opacity <= 0:
                continue
            b = o.impl.world_bounds(self.state)
            inside = ((world[:, 0] >= b.lo[0]) & (world[:, 0] <= b.hi[0])
                      & (world[:, 1] >= b.lo[1]) & (world[:, 1] <= b.hi[1]))  # fmt: skip
            if inside.mean() > 0.8:
                return o
        return None

    def _objects_inside(self, loop_px: np.ndarray) -> list[str]:
        """Non-backdrop objects whose on-screen center lies inside a drawn loop."""
        out = []
        for o in self.state.objects:
            if o.layer == "overlay" or self._backdrop(o.id) or o.params.opacity <= 0:
                continue
            c = self.to_pixels(np.asarray(o.impl.world_bounds(self.state).center)[None, :])[0]
            if abs(winding(loop_px[:, 0] + 1j * loop_px[:, 1], complex(*c))) > 0.5:
                out.append(o.id)
        return out

    # ---------------------------------------------------------- ids

    def fresh_id(self, base: str) -> str:
        """A new id named for what it is (``contour``, then ``contour_2``...), never one in
        use, retired or already proposed."""
        doc = self.engine.document
        taken = {o.id for o in doc.objects} | set(doc.retired_ids) | self._taken
        base = base if ID_RE.fullmatch(base) else "sketch"
        name, k = base, 2
        while name in taken or any(t.startswith(name + ".") for t in taken):
            name, k = f"{base}_{k}", k + 1
        self._taken.add(name)
        return name

    # ---------------------------------------------------------- proposals

    def propose(self, strokes: list[Any]) -> list[Proposal]:
        """One proposal per drawn shape (strokes that belong together are one shape)."""
        out = []
        for shape in group(strokes):
            p = self._propose(shape)
            if p is not None:
                p.strokes = shape.strokes
                out.append(p)
        return out

    def _propose(self, sh: Shape) -> Proposal | None:
        kind = sh.kind
        if kind in ("circle", "closed"):
            world = self.to_world(sh.points)
            dom = self._domain_under(world)
            if dom is not None:
                return self._contour(sh, world, dom)
            inside = self._objects_inside(sh.points)
            if inside:
                return Proposal(
                    "emphasize", f"emphasize {', '.join(inside)}",
                    [{"op": "set_style", "id": i, "key": "color", "role": "highlight"}
                     for i in inside if "color" in self.state[i].style],
                    data={"ids": inside},
                )  # fmt: skip
            return self._line(sh, closed=True)
        if kind == "dot":
            return self._dot(sh)
        if kind == "line":
            return self._line(sh)
        if kind == "arrow":
            return self._arrow(sh)
        if kind == "curve":
            return self._line(sh, wavy=bool(sh.data.get("wavy")))
        if kind == "scribble":
            hits = self._hits(sh.points[:: max(1, len(sh.points) // 24)])
            if not hits:
                return None
            return Proposal("remove", f"remove {', '.join(hits)} (crossed out)",
                            [{"op": "remove_object", "id": i} for i in hits],
                            data={"ids": hits})  # fmt: skip
        if kind == "handwriting":
            return self._handwriting(sh)
        return None

    def _contour(self, sh: Shape, world: np.ndarray, dom) -> Proposal:
        src = dom.params.function.strip()
        z = smooth_loop(world[:, 0] + 1j * world[:, 1])
        numeric = contour_integral(src, z)
        exact, enclosed = residue_integral(src, z)
        size = float(np.ptp(z.real) + np.ptp(z.imag)) / 2
        near = [p for p in poles_of(src) if np.abs(z - p).min() < POLE_NEAR_SHARE * size]
        cid = self.fresh_id("contour")
        zplane = float(dom.params.pos[2])
        pts = [[round(float(c.real), 5), round(float(c.imag), 5), zplane] for c in z]
        pts.append(pts[0])
        value = exact if exact is not None else numeric
        tex = pretty(value)
        ops: list[dict[str, Any]] = [
            {"op": "add_object", "object": {"id": cid, "type": "line", "params": {"points": pts},
                                            "style": {"glow": "soft", "stroke": "bold"}}},
            {"op": "add_object", "object": {"id": f"{cid}.label", "type": "text",
                                            "params": {"text": f"$\\oint f\\,dz = {tex}$"}}},
            {"op": "add_constraint", "rule": f"label({cid}) near {cid}"},
        ]  # fmt: skip
        data: dict[str, Any] = {"id": cid, "function": src, "integral": numeric,
                                "exact": exact, "latex": tex,
                                "enclosed": [{"pole": p, "winding": w, "residue": r}
                                             for p, w, r in enclosed]}  # fmt: skip
        if near:
            data["warning"] = (f"the contour passes close to a pole at {pretty(near[0])}: the "
                               "integral is sensitive to exactly where it runs")  # fmt: skip
        poles = ", ".join(pretty(p) for p, w, _ in enclosed if w) or "no poles"
        return Proposal(
            "contour", f"contour around {poles}: integral of f dz = {tex}", ops,
            [{"verb": "trace", "at": self.t, "targets": [cid]},
             {"verb": "reveal", "at": self.t, "targets": [f"{cid}.label"]}], data,
        )  # fmt: skip

    def _dot(self, sh: Shape) -> Proposal:
        at = self.to_world(sh.data["at"][None, :])[0]
        dom = self._domain_under(at[None, :])
        what = "point"
        if dom is not None:
            src = dom.params.function.strip()
            marks = [(p, "pole") for p in poles_of(src)] + [(q, "zero") for q in zeros_of(src)]
            best = None
            for c, name in marks:
                d = float(np.linalg.norm(self.to_pixels(np.array([[c.real, c.imag, at[2]]]))[0]
                                         - sh.data["at"]))  # fmt: skip
                if d < SNAP_PX and (best is None or d < best[0]):
                    best = (d, c, name)
            if best is not None:
                at = np.array([best[1].real, best[1].imag, at[2]])
                what = best[2]
        did = self.fresh_id(what if what != "point" else "dot")
        pos = [round(float(v), 5) for v in at]
        label = f" at {pretty(complex(at[0], at[1]))}" if what != "point" else ""
        return Proposal("dot", f"mark the {what}{label}",
                        [{"op": "add_object", "object": {"id": did, "type": "dot",
                                                          "params": {"pos": pos}}}],
                        data={"id": did, "at": pos, "snapped": what})  # fmt: skip

    def _line(self, sh: Shape, closed: bool = False, wavy: bool = False) -> Proposal:
        if sh.kind == "circle":
            c, r = sh.data["center"], sh.data["radius"]
            a = np.linspace(0, 2 * math.pi, 65)
            px = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)
        elif sh.kind == "line":
            px = np.array([sh.data["start"], sh.data["end"]])
        else:
            px = resample(sh.points, 48)
            if closed:
                px = np.concatenate([px, px[:1]])
        world = self.to_world(px)
        name = {"circle": "circle", "closed": "loop", "line": "line"}.get(sh.kind, "curve")
        lid = self.fresh_id("wave" if wavy else name)
        pts = [[round(float(v), 5) for v in p] for p in world]
        return Proposal(sh.kind if not wavy else "wave", f"draw a {'wavy ' if wavy else ''}{name}",
                        [{"op": "add_object", "object": {"id": lid, "type": "line",
                                                          "params": {"points": pts}}}],
                        data={"id": lid})  # fmt: skip

    def _arrow(self, sh: Shape) -> Proposal:
        a, b = sh.data["start"], sh.data["end"]
        wa, wb = self.to_world(np.array([a, b]))
        on = [i for i in self._hits(np.array([a])) if hasattr(self.state[i].params, "pos")]
        if on:
            target = on[0]
            z = float(self.state[target].params.pos[2])
            to = [round(float(v), 5) for v in self.to_world(np.array([b]), z=z)[0]]
            return Proposal("move", f"move {target} to ({to[0]:g}, {to[1]:g})",
                            [{"op": "set_param", "path": f"{target}.pos", "value": to}],
                            data={"id": target, "to": to})  # fmt: skip
        aid = self.fresh_id("arrow")
        return Proposal("arrow", "draw an arrow",
                        [{"op": "add_object", "object": {"id": aid, "type": "arrow", "params": {
                            "start": [round(float(v), 5) for v in wa],
                            "end": [round(float(v), 5) for v in wb]}}}],
                        data={"id": aid})  # fmt: skip

    def _handwriting(self, sh: Shape) -> Proposal:
        x0, y0, x1, y1 = sh.data["box"]
        center = self.to_world(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))[0]
        return Proposal(
            "handwriting",
            "handwriting: read the picture and call transcribe(proposal, text) with what it "
            "says (LaTeX in $...$ for maths)",
            data={"at": [round(float(v), 5) for v in center], "box": [x0, y0, x1, y1]},
            image=render_strokes(sh.data["strokes"]),
        )

    def transcribe(self, proposal: Proposal, text: str) -> Proposal:
        """What the handwriting says, as a Text where it was written."""
        if proposal.kind != "handwriting":
            raise ValueError("only handwriting is transcribed")
        tid = self.fresh_id("note")
        return Proposal("text", f"write {text!r}",
                        [{"op": "add_object", "object": {"id": tid, "type": "text", "params": {
                            "text": text, "pos": proposal.data["at"]}}}],
                        data={"id": tid, "text": text}, strokes=proposal.strokes)  # fmt: skip


def accept(workspace: Any, proposal: Proposal) -> dict[str, Any]:
    """Make a proposal real: its ops in one patch, then its verbs. If a verb fails the
    patch is undone, so a proposal lands whole or not at all."""
    if not proposal.ops:
        raise ValueError(f"nothing to apply for {proposal.kind!r}: {proposal.summary}")
    r = workspace.scene_edit(proposal.ops)
    if not r.get("ok", False):
        return r
    try:
        for a in proposal.animate:
            workspace.animate(a["verb"], a["at"], targets=a.get("targets"))
    except Exception:
        workspace.undo()
        raise
    return r


# ---------------------------------------------------------------- the maths of a contour


def smooth_loop(z: np.ndarray, n: int = 160, passes: int = 2) -> np.ndarray:
    """A hand-drawn loop as ``n`` evenly spaced points, lightly smoothed (circular moving
    average), so the pen's jitter does not show; the curve is closed implicitly."""
    pts = resample(np.stack([z.real, z.imag], axis=1), n + 1)[:-1]
    for _ in range(passes):
        pts = (np.roll(pts, 1, axis=0) + 2 * pts + np.roll(pts, -1, axis=0)) / 4
    return pts[:, 0] + 1j * pts[:, 1]


def winding(loop: np.ndarray, p: complex) -> float:
    """How many times a closed polygon (complex points) winds around ``p``."""
    d = np.append(loop, loop[0]) - p
    if np.abs(d).min() < 1e-12:
        return 0.0
    return float(np.angle(d[1:] / d[:-1]).sum() / (2 * math.pi))


def contour_integral(src: str, loop: np.ndarray) -> complex:
    """The integral of f dz around a closed polygon, by the midpoint rule on each side."""
    from phasor.render.gpu.codegen import ComplexFunction

    f = ComplexFunction.parse(src).numpy()
    a = loop
    b = np.roll(loop, -1)
    with np.errstate(all="ignore"):
        v = np.asarray(f((a + b) / 2), dtype=np.complex128) * (b - a)
    return complex(np.nansum(v))


def _rational(src: str):
    import sympy as sp

    from phasor.render.gpu.codegen import ComplexFunction, Z

    expr = ComplexFunction.parse(src).expr
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    if not (num.is_polynomial(Z) and den.is_polynomial(Z)):
        return None
    return expr, num, den, Z


def _roots(poly, z) -> list[complex]:
    import sympy as sp

    if sp.degree(poly, z) < 1:
        return []
    exact = sp.roots(sp.Poly(poly, z))
    if sum(exact.values()) == sp.degree(poly, z):
        return [complex(r) for r in exact]
    return [complex(r) for r in sp.Poly(poly, z).nroots()]


def poles_of(src: str) -> list[complex]:
    r = _rational(src)
    return [] if r is None else sorted(set(_roots(r[2], r[3])), key=lambda c: (c.real, c.imag))


def zeros_of(src: str) -> list[complex]:
    r = _rational(src)
    return [] if r is None else sorted(set(_roots(r[1], r[3])), key=lambda c: (c.real, c.imag))


def residue_integral(src: str, loop: np.ndarray) -> tuple[complex | None, list]:
    """2 pi i times the sum of (winding number x residue) over the poles a loop encloses,
    for a rational f whose poles sympy finds exactly (None otherwise), and the enclosed
    poles as (pole, winding, residue)."""
    import sympy as sp

    r = _rational(src)
    if r is None:
        return None, []
    expr, _num, den, z = r
    degree = sp.degree(den, z)
    exact = sp.roots(sp.Poly(den, z)) if degree >= 1 else {}
    if degree >= 1 and sum(exact.values()) != degree:
        return None, []
    total = 0j
    found = []
    for p in exact:
        w = round(winding(loop, complex(p)))
        if w == 0:
            continue
        res = complex(sp.N(sp.residue(expr, z, p)))
        total += w * res
        found.append((complex(p), w, res))
    return 2j * math.pi * total, found


def pretty(v: complex, tol: float = 1e-6) -> str:
    """A complex number as LaTeX: exact multiples of pi or of i when it is one
    (``\\pi``, ``-2\\pi i``, ``\\frac{\\pi}{2}``), else decimals."""
    v = complex(v)
    if abs(v) < tol:
        return "0"

    def part(x: float, unit: str) -> str | None:
        for scale, sym in ((math.pi, "\\pi"), (1.0, "")):
            f = Fraction(x / scale).limit_denominator(12)
            if f == 0 or abs(float(f) * scale - x) > tol * max(1.0, abs(x)):
                continue
            n, den = abs(f.numerator), f.denominator
            tail = (" " if sym else "") + unit if unit else ""
            if den == 1:
                core = ("" if n == 1 and (sym or unit) else str(n)) + sym + tail
            else:
                core = f"\\frac{{{'' if n == 1 and sym else n}{sym}}}{{{den}}}{tail}"
            return ("-" if f < 0 else "") + core
        return None

    re_s = part(v.real, "") if abs(v.real) >= tol else ""
    im_s = part(v.imag, "i") if abs(v.imag) >= tol else ""
    if re_s is None or im_s is None:
        a = f"{v.real:.4g}" if abs(v.real) >= tol else ""
        b = f"{abs(v.imag):.4g}i" if abs(v.imag) >= tol else ""
        if a and b:
            return f"{a} {'-' if v.imag < 0 else '+'} {b}"
        return a or ("-" if v.imag < 0 else "") + b
    if re_s and im_s:
        return f"{re_s} {'-' if im_s.startswith('-') else '+'} {im_s.lstrip('-')}"
    return re_s or im_s


def render_strokes(strokes: list[np.ndarray], pad: int = 12, height: int = 96) -> bytes:
    """A PNG of strokes (white ink on black), scaled to ``height`` pixels of writing."""
    from PIL import Image, ImageDraw

    pts = np.concatenate(strokes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    k = height / max(float(hi[1] - lo[1]), DOT_PX)
    w = int((hi[0] - lo[0]) * k) + 2 * pad
    img = Image.new("L", (max(w, 2 * pad + 1), height + 2 * pad), 0)
    draw = ImageDraw.Draw(img)
    for s in strokes:
        xy = [((x - lo[0]) * k + pad, (y - lo[1]) * k + pad) for x, y in s]
        if len(xy) == 1:
            x, y = xy[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=255)
        else:
            draw.line(xy, fill=255, width=max(2, int(height / 24)), joint="curve")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
