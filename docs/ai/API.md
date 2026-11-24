# phasor API (for AI agents)

One canonical example per public name. No deprecated aliases. Every ```python block in this
folder runs in CI (`tests/test_docs.py`), top to bottom in one namespace per file. Mark a
block ```python no-run only when it cannot run in CI, and say why next to it.

This file is the Python API; to drive phasor from an AI client
(MCP) read `MCP.md` first.

## Build a scene

```python
from phasor import Dot, Line, Scene, Text, expr

s = Scene("residues")  # 1920x1080, 60 fps, the default look "prism" (look="none": plain)
s.add(Line(id="axis", points=[(-6, 0), (6, 0)], style={"color": "muted"}))
s.add(Dot(id="pole_i", pos=(0, 1), r=0.15, style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="pole_i_label", text="i", pos=expr("pole_i.pos + vec(0.4, 0, 0)")))
s.set("camera.height", 6)  # world units visible top to bottom
```

- IDs match `[a-z][a-z0-9_]*`, with dots for children (`res.pole_i` needs `res` to exist).
  Name them by meaning (`pole_i`), never by counter (`dot3`).
- Points are world space: x = Re, y = Im, z up. `(x, y)` means z = 0.
- `style` takes role names only. Raw colors and sizes are rejected.
- `tags=["key"]` marks an object other objects are located relative to (see `where`).
- A wrong call raises `DocumentError` at that line and changes nothing.

### Object types

| Type | Parameters | Style keys (role group, default) |
| --- | --- | --- |
| `Dot` | `pos`, `r` (world radius, default 0.1) | `color` (emphasis, primary), `glow` (glow, none) |
| `Line` | `points` (2 or more), `progress`, `hue_along` (colormap cycles along the length; 0: one color), `hue` (shifts them) | `color` (emphasis, primary), `stroke` (stroke, regular), `glow` (glow, none), `colormap` (colormap, spectrum) |
| `cx.DomainColoring` | `function` (e.g. `"1/(z**2 + 1)"`; it may use the constant `a`), `a` ((re, im), the value of `a`: animate it with `.to("id.a", (re, im))` and the zeros and poles move, nothing recompiles), `pos` (center), `size` (w, h), `contour` (0-1), `softness` (0-1, edge feathering), `hue` (turns the wheel); the prism look: `radius` (> 0 turns it on: zeros and poles glow this far), `ambient` (0-1, light elsewhere), `lines` (glowing rays of phase per turn, plus a ring each time \|f\| doubles) | `colormap` (colormap, phase), `glow` (glow, none; used by the prism look) |
| `cx.Fractal` | `kind` (`mandelbrot`/`julia`), `c` (Julia constant), `pos`, `size`, `max_iter`, `density`, `hue`, `center` (exact text `"re im"`), `zoom` (powers of ten, to 280), `steady` (hold the outer color in a dive); the prism look: `radius` (> 0 turns it on: the edge of the set glows this far, in world units of the rectangle), `ambient` (0-1, light far from the edge); the cloud mode `kind="buddhabrot"`: `count`, `seed`, `exposure`, `point_size`, `white` (as `cloud.PointCloud`; `max_iter` at most 5000) | `colormap` (colormap, sequential), `glow` (glow, none; used by the prism look and the cloud mode) |
| `cx.RiemannSurface` | `kind` (`sqrt`/`cbrt`/`log`), `pos` (branch point), `radius`, `height` (`re`/`im`), `scale`, `rings`, `segments` | `colormap` (colormap, phase), `material` (surface, solid) |
| `cloud.PointCloud` | `source` (`roots`/`attractor`/`buddhabrot`/`points`), `degree` and `coefficients` (roots), `kind` (`clifford`/`dejong`/`lorenz`) and `constants` (attractor), `max_iter` (buddhabrot), `points` and `values` (given directly), `count` (at most), `seed`, `pos` (centre and origin of the points), `size` (the rectangle it is seen in), `scale`, `point_size` (splat radius, px), `exposure`, `hue`, `color_by` (`source`/`phase`), `twist` (roots: turns of the last coefficient, animatable), `twinkle` (0-1, life), `white` (0-1, white-hot dense parts) | `colormap` (colormap, spectrum), `glow` (glow, none) |
| `cloud.PointCloud` (`source="bifurcation"`) | `map` (`logistic`/`sine`/a formula in `r` and `x`), `r_range` and `x_range` (the window: r across the rectangle, x up it), `r_focus`, `x_focus` and `focus` (0-1, animatable: a geometric zoom of the window to the focus), `x0`, `n_iter`, `n_keep` (the last iterates drawn), `sweep` (0-1: share drawn from the left), and the cloud's `pos`, `size`, `point_size`, `exposure`, `hue`, `twinkle`, `white`; preset `prism_bifurcation` | `colormap` (colormap, spectrum), `glow` (glow, none) |

| `Value` | `value` (a named number to animate; nothing is drawn: see "Your own types") | none |
| `Image` | `path` (a PNG/JPEG/WebP file, relative to the scene's folder), `pos` (centre), `width`, `height` (0: keep the picture's shape), `frame` (an outline of light) | `glow` (glow, none: bright parts shine), `frame` (emphasis, muted), `stroke` (stroke, hairline). Presets `plain`, `framed`, `luminous`. A missing file is issue `missing_file`; transparent parts show the background; an AI may only name files inside its folder |
| `Text` | `text` (`$...$` is typeset math), `pos`, `anchor` (`center`/`left`/`right`), `leader`, `leader_on` (set by layout) | `color` (emphasis, primary), `size` (text, label), `leader` (stroke, hairline) |
| `Equation` | `parts` (math or text, set on one baseline), `colors` (an emphasis role per part), `pos` | `color` (emphasis, primary), `size` (text, math), `glow` (glow, none) |

Every type also has `opacity` (0-1; below 0.5 it stops hiding things behind it), `dim`
(0-1, owned by `dim_others`; what is drawn is `opacity` x `dim`) and
`z_index` (higher draws on top among objects at the same depth). Every object takes
`layer="overlay"` to live in screen space: units are frame heights, origin top-left, y down.

Roles: emphasis `primary secondary muted highlight danger`; stroke `hairline regular bold`;
text `title label annotation math`; glow `none soft strong`; colormap `phase sequential`;
surface `solid glass wire emissive` (3D materials; glass is see-through and hides nothing).

`cx` objects are rectangles of the complex plane drawn by GPU kernels: world x is Re z and
world y is Im z, so the camera zooms into them like into anything else. They are
*backdrops*: labels may sit on them (layout does not avoid them; lint checks contrast).
Their semantics list the zeros and poles inside them (for rational functions).

```python
from phasor import Line, Scene, cx

s3 = Scene("poles")
s3.set("camera.height", 4.5)
s3.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
s3.add(
    Line(
        id="loop",
        points=[(0.4, 1), (0, 1.4), (-0.4, 1), (0, 0.6), (0.4, 1)],
        style={"color": "highlight", "glow": "strong"},
    )
)
plane = s3.engine.inspect(0, id="plane")["objects"][0]["semantic"]
assert sorted(map(tuple, plane["poles"])) == [(0.0, -1.0), (0.0, 1.0)]
```

**Deep zooms.** `cx.Fractal` shows the plane around `center` magnified `10**zoom` times.
`center` is text, because a float holds 16 digits and a 1e-30 view needs more. Precision
follows the pixel size by itself: float32, float64 under a 1e-5 view, then (Mandelbrot only)
perturbation under 1e-10 per pixel: one orbit computed exactly, every pixel as a float64
difference from it. Semantics report `zoom`, `view_height` and `precision`. Deep views need
`max_iter` in the thousands; `steady=True` keeps a dive's colors from cycling.

```python
from phasor import Scene, cx

dive = Scene("dive")
dive.add(
    cx.Fractal(
        id="set",
        center="-0.77568376800905379746948350393474104 0.13646736829469012473327440961784876",
        pos=(0, 0, 0),
        size=(8, 4.5),
        max_iter=3000,
        steady=True,
    )
)
dive.at(1.0).to("set.zoom", 30.0, duration=10.0)
sem = dive.engine.inspect(11.0, id="set")["objects"][0]["semantic"]
assert sem["precision"] == "perturbation" and sem["view_height"] < 1e-29
```

`glow` makes a curve emit light: the renderer writes that light to a second target and the
final look blooms it. Plain paint (`glow: none`) never blooms, and colors below the theme's
`knee` (linear 0.9) come out exactly as the theme defines them.

### Expressions

`expr("...")` computes a parameter from other parameters (`<id>.<param>`) and time `t`.
Operators `+ - * / ** %`, indexing `dot.pos[0]`, constants `pi tau e`, functions
`sin cos tan exp log sqrt floor ceil abs min max clamp lerp vec length normalize dot midpoint`.
Scalars broadcast over vectors. A cycle is an error.

## Your own types, composites and updaters (ADR 0005)

A type needs two members: `Params` and `draw(ctx, state)`. Its `world_bounds`, `features`
and `semantics` are worked out from what `draw` emits (`core/record.py`), so the inspector,
layout and lint know it like any shipped type:

```python
import numpy as np
from typing import ClassVar

from phasor import Dot, Line, Scene, Text, Value
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3


@register
class Kite(ObjectBase):
    type: ClassVar[str] = "doc_kite"
    Style: ClassVar = {"color": ("emphasis", "highlight")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = 0.4

    def draw(self, ctx, state):
        (x, y, z), r = self.me(state).params.pos, self.me(state).params.r
        pts = np.array([[x, y + r, z], [x + r / 2, y, z], [x, y - r, z], [x - r / 2, y, z]])
        ctx.triangles(pts, np.array([[0, 1, 2], [0, 2, 3]]), color=self.me(state).style["color"])


k = Scene("kites")
k.add(Kite(id="kite", pos=(1, 0)))
seen = k.engine.inspect(0.0, id="kite", detail="full")["objects"][0]
assert seen["semantic"]["type"] == "doc_kite" and "top" in seen["features"]
```

- Derived `world_bounds`: the box of everything drawn (a `field`'s quad corners; its pixels
  are never computed). Derived `features`: `center` and the box's compass points (`top`,
  `top_right`, ...), plus `start`, `end` and `curve` when the object draws one stroke through
  `domains/strokes.py`. Derived `semantics`: `{"type", "params"}`.
- Write any of the three yourself when you know more (a pole, a tip, `"kind": "point"`);
  yours wins. `draw(self, ctx, state, pass_)` still works: the pass is optional.
- A recording is cached per (type, id, parameters, style, camera); `draw` may read other
  objects (`state["other"]`), and the recording is made again when they change.

**Composites.** `build(self, state)` instead of `draw` returns other objects. Each part
gets the id `<composite>.<name>`, is drawn and inspected as itself (`"part_of"` in the
inspector), fades with its composite and may not reuse a retired id. The composite's bounds
are the union of its parts', its features are the box's plus each part's as `<part>.<name>`:

```python
@register
class Signpost(ObjectBase):
    type: ClassVar[str] = "doc_signpost"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def build(self, state):
        x, y, z = self.me(state).params.pos
        return [
            Line(id="post", points=[(x, y, z), (x, y + 1.2, z)]),
            Kite(id="sign", pos=(x, y + 1.2, z), r=0.3),
        ]


k.add(Signpost(id="post", pos=(-2, -1)))
state = k.engine.state_at(0.0)
assert [o.id for o in state.objects if o.parent == "post"] == ["post.post", "post.sign"]
```

Constraints name document objects; a part is placed through its composite.

**Updaters.** `obj.always(fn)` with `fn(t, look) -> {param: value}` keeps parameters set
from time and other objects. `look("id")` gives that object's parameters at `t`
(`look("dot").pos`); each one read is an edge in the parameter graph, so a cycle is an
error, and a read the updater did not make at t = 0 is an error too (read the same things at
every t). It must be pure: `lint(strict=True)` calls every updater again, in another order,
and reports `impure_updater` when answers differ (threshold: the theme's `[lint] impure`).
Something that needs the previous frame is a `Simulation`. `look` sees parameters before
living motion: tag an object `exact` when something must sit on it exactly.

**`Value`** is a named number to animate with no drawing (Manim's `ValueTracker`): read
it in an expression (`expr("speed.value")`), in an updater, or show it with a `Counter`.

```python no-run
# no-run: an updater must be defined in a file in the scene's folder (the document names
# it), and this page's text is not one. examples/open_contract.py runs the same thing, and
# tests/test_open_contract.py checks it.
def above(t, look):
    x, y, z = look("ball").pos
    return {"pos": (x, y + 0.5, z)}


def roll(t, look):
    return {"pos": (look("speed").value * t, 0.0, 0.0)}


k.add(Value(id="speed", value=0.5))
k.add(Dot(id="ball", r=0.15, tags=["exact"]).always(roll))
k.add(Text(id="ball.label", text="ball").always(above))
assert k.engine.state_at(2.0)["ball.label"].params.pos == (1.0, 0.5, 0.0)
assert not [i for i in k.engine.lint(strict=True) if i.code == "impure_updater"]
```

**Where the code lives.** A type or updater defined in the scene script, or in a
`*.types.py` file next to it, is named by the document: `meta.modules` lists the file and
`always` lists `"<file>:<function>"`. Opening the document loads those modules only where
scripts may run (`phasor.load("scene.py")`, `Engine(..., allow_scripts=True)`, the MCP
server's `--allow-scripts`, and `--allow-scripts` on `phasor inspect`, `lint`, `storyboard`,
`render`, `web`, `bench --frames` and `sketch`; render workers inherit it), only from inside the scene's folder. A missing, refused or
broken module is lint `missing_type`: the object draws as a placeholder box and keeps its
parameters and overrides (saving it changes nothing). An updater defined anywhere else (a
notebook cell, a prompt) is refused: the document could not find it again. Types shipped in phasor still get an entry here, an
example and a storyboard test; scene-local types do not need them.

## Narrate: words drive time

```python
from phasor import Dot, Scene

s4 = Scene("narrated", passes=None)
s4.add(Dot(id="pole", pos=(0, 1), r=0.2))
s4.narrate(
    "We start with a function that has a pole [[pole_appears]].\n"
    "[[pause 0.5]]\n"
    "Watch it move [[pole_moves]] to the right.",
    voice="estimate",  # or "auto", "espeak[:lang[:wpm]]", "piper:<model.onnx>"
)
s4.at("pole_appears").bloom_in("pole")
s4.at("pole_moves").settle("pole.pos", (2, 1))
s4.pace()  # breathing room after sentences that change a lot; run last
assert s4.engine.lint(strict=True) == []
```

- One sentence is one block; `[[name]]` marks the moment the next word starts (at the end
  of a sentence: when it ends). `[[pause 1.5]]` is a silent block.
- `narrate` writes the markers. Anchor motion to them with `s.at("name")`; never type seconds.
  Re-narrate (another voice, an edited sentence) and everything moves with the words.
- Voices: `auto` = espeak-ng if installed, else `estimate` (timing from the text, no audio).
  Synthesized voices are spoken piece by piece between markers, so markers are exact; clips
  are cached in `<title>.narration/` beside the script. Your own recordings:
  `phasor.voice.from_recordings(script, wavs, aligner, base_dir, theme)` with
  `WhisperX()` or `TimingsFile()`.
- `phasor render` muxes the narration into the video.

## Animate with verbs

| Verb | Does |
| --- | --- |
| `reveal(*ids)` | fade in; hidden until then; drops earlier dimming (it is the new focus) |
| `exit(*ids)` | fade out |
| `trace(*ids)` | draw a `Line` from start to end (`progress` 0 to 1); nothing drawn before |
| `bloom_in(*ids)` | fade in and grow from a point with a small overshoot (objects with `r`) |
| `dim_others(*keep)` | dim everything but the object (or objects) you keep, to the theme's `dim_level` (backdrops stay) |
| `undim()` | bring dimmed objects back |
| `focus_pull(id, margin=0.15)` | move the camera onto an object |
| `morph(path, value)` | change any parameter smoothly (a whole-number one, such as `terms`, in whole steps) |
| `settle(path, value)` | move like a released spring (overshoot, then rest) |
| `transform(a, b)` | equation `a` becomes `b`: parts that appear in both glide over (and change color); the rest fade |
| `become(a, b)` | curve `a` turns into the shape of curve `b` (a circle into a square), then `b` takes its place |
| `write(*ids)` | write text or an equation in, left to right (`write` 0 to 1) |
| `travel(*ids)` | send `Tracer`s along their curves, trailing light (`progress` 0 to 1) |
| `flash(*ids)` | a quick brightening (every object has `flash`, 0 to 1) |
| `circumscribe(id, shape="circle")` | draw a ring (or `"box"`) around an object, then fade it: the child `<id>.ring` |
| `wiggle(*ids)` | a small side-to-side shake (objects with `pos`) |
| `sweep(path, value, show=True)` | run a number evenly to `value`, with a live readout `<id>.readout` beside it |

Durations, easing and stagger come from the theme's `[motion]` table. Chain with `then()`:
`s.at("loop_draw").trace("loop").then().reveal("loop.label")`. Several ids stagger.
`orbit_reveal` arrives with 3D.

Dimming uses each object's `dim` factor (drawn opacity = `opacity` x `dim`), so dimming and
revealing never fight, whatever order you write them in.

**Objects made for motion.** `Tracer(path="curve_id", progress=, trail=0.15)` is a glowing
point riding another curve (a `Line`, `calc.Graph`, `geo.ParametricCurve` or
`fourier.PartialSum`), its trail fading behind it (preset `comet`); semantics give its
`position`, or `path_missing`. `Counter(value=expr("comet.progress"), prefix="p = ",
digits=2)` shows a number that follows whatever it reads. `Callout(target=, shape=)` is the
ring `circumscribe` draws. `cx.ComplexMap(function="z**2", amount=0)` is a grid pushed
through f (animate `amount` 0 to 1 with `morph`; `center` is the complex point at the grid's
middle, drawn at `pos`; lines break at poles). `geo.FlowParticles(fp=, fq=, fr=, count=,
speed=, lifetime=, tail=)` streams particles along a field (random starts from
`rng(id, seed)`, a pure function of time; presets `streaks`, `sparks`). `geo.ParametricCurve`
and `geo.ParametricSurface` take `to_fx`/`to_fy`/`to_fz` and `blend` (0 to 1) to flow into
another formula. `examples/motion_gallery.py` shows them all.

```python
from phasor import Counter, Equation, Scene, Tracer, cx, expr, geo

mo = Scene("motion", passes=None)
mo.add(Equation(id="before", parts=("$x$", "$=2$")))
mo.add(Equation(id="after", parts=("$x$", "$+1$", "$=3$")))
mo.add(geo.ParametricCurve(id="loop", fx="cos(t)", fy="sin(2*t)", pos=(3, 0, 0)))
mo.add(Tracer(id="comet", path="loop", preset="comet"))
mo.add(Counter(id="comet.count", value=expr("comet.progress"), prefix="p = ", pos=(3, -1.5, 0)))
mo.add(cx.ComplexMap(id="bend", function="z**2", amount=0.0, extent=(1.6, 1.6), pos=(-3, 0, 0)))
mo.marker("go", 0.5)
mo.at("go").transform("before", "after").travel("comet").morph("bend.amount", 1.0)
mo.at("go").then().flash("after").circumscribe("after")
me = mo.engine
assert me.inspect(1.0, id="after")["objects"][0]["semantic"]["matched"] == 1
assert me.inspect(4.0, id="comet.count")["objects"][0]["semantic"]["text"] == "$p = 1.00$"
assert "after.ring" in [o.id for o in me.document.objects]
```

**Smooth transitions** (`examples/prism_transitions.py`).

- *Solving and rearranging.* Give each part of an `Equation` a name with `keys` (`""` for
  none). In a `transform`, the part with the same key in the source becomes that part
  wherever it was, even when its text changes: `+a` on the left becomes `-a` on the right,
  and `0.6 i` turns into `0.7 i` in place. The text changes while the part moves. Parts
  without keys pair by equal text, in order, as before.
- *Carried parts arc.* A part carried past others travels round them by the theme's
  `[motion] transform_arc` line heights: over them going right, under them going left. The
  parts that keep their order go straight.
- *Semantics.* Mid-transform they report `matched`, `changed` (pairs whose text differs)
  and `carried_over` (pairs that arc).
- *A function changing into another.* `cx.DomainColoring(to="...", blend=0..1)` shows
  w = (1 - blend) f + blend g, a straight line between the two functions. Every frame is
  then a true picture of a function: zeros glide, a pole of g grows out of a point, and
  there are no seams. A blend in log space would cut the plane along the branch cuts of
  log f and log g. At 0 and 1 you get exactly f and g. Both `blend` and `a` are kernel
  arguments, so animating them never recompiles.
- *Slow changes.* `morph(path, value, until="marker")` runs a change until a marker instead
  of the theme's `morph` time: a constant drifting slowly while it is talked about.

```python
from phasor import Equation, Scene, cx

tr = Scene("solve", passes=None)
tr.add(
    cx.DomainColoring(
        id="plane",
        function="(z**5 - 1 - a)/(z**2 + a)",
        to="(z**3 - 1 - a)/(z**2 + a)",
        a=(0.0, 0.6),
    )
)
tr.add(Equation(id="step1", parts=("$z^2$", r"$+\,a$", "$=$", "$0$"), keys=("z", "a", "eq", "")))
tr.add(Equation(id="step2", parts=("$z^2$", "$=$", "$-a$"), keys=("z", "eq", "a")))
for name, t in (("solve", 1.0), ("blend", 3.0), ("done", 9.0)):
    tr.marker(name, t)
tr.at("solve").transform("step1", "step2").morph("plane.a", (0.5, 0.7), until="blend")
tr.at("blend").morph("plane.blend", 1.0, until="done")
mid = tr.engine.inspect(1.5, id="step2")["objects"][0]["semantic"]
assert (mid["matched"], mid["changed"], mid["carried_over"]) == (3, 1, 1)
assert len(tr.engine.inspect(9.0, id="plane")["objects"][0]["semantic"]["zeros"]) == 3
```

## Animate by hand

```python
s.marker("appear", 1.0)  # a hand-placed marker, in seconds
s.at("appear").to("pole_i.r", 0.3, duration=0.5)  # ease "smooth" by default
s.at("appear", offset=1.0).to("pole_i.pos", (2, 1), duration=2.0, ease="in_out")
s.at(4.0).set("pole_i_label.text", "moved")  # instant change at 4 s
```

Raw `to()` is allowed but lints as `raw_motion` with `engine.lint(strict=True)`: prefer verbs.
Any parameter is animatable by its path, cameras included (`camera.center`, `camera.height`).
Times anchored to a marker move with it. Segments on one parameter must not overlap.
A parameter set by `expr(...)` cannot also have a track; animate what it reads.
Easings: `linear smooth smoother in out in_out`, and `back spring` (these overshoot: not for
opacity). Custom easing functions are not supported: a document must mean the same
everywhere.

## Engine: state, patches, undo

```python
engine = s.engine
state = engine.state_at(3.0)  # pure function of (document, t); immutable
assert state["pole_i"].params.pos == (1.0, 1.0, 0.0)
assert state["pole_i_label"].params.pos == (1.4, 1.0, 0.0)  # the label followed

result = engine.apply({"op": "set_param", "path": "pole_i.r", "value": 0.2})
assert result.ok and engine.version == result.version

bad = engine.apply({"op": "set_param", "path": "pole_i.r", "value": -1})
assert not bad.ok and bad.errors[0].code == "bad_param"  # nothing changed

engine.undo()
engine.redo()
```

`Engine.apply(patch)` is the only way to change a scene. It takes one op, a list of ops, or
`{"ops": [...]}`; applies all or nothing; returns `PatchResult(ok, version, errors,
lint_added, lint_resolved)`. Each error has `code`, `message`, `ids`, `path`, `fix_hint`.

| Op | Fields |
| --- | --- |
| `add_object` | `object: {id, type, params, style, tags, layer}` |
| `remove_object` | `id` (its tracks go too; fails if another object's expression reads it) |
| `set_param` | `path`, `value` (constant or `{"expr": "..."}`) |
| `set_style` | `id`, `key`, `role` |
| `set_tags` | `id`, `tags` |
| `add_segment` | `path`, `segment: {t0, t1, to, ease}`; a time is seconds or `{"marker", "offset"}` |
| `clear_track` | `path` |
| `set_marker` / `remove_marker` | `name`, `t` |
| `set_meta` | `field`, `value` |
| `add_constraint` | `rule`, `priority` (`required`/`strong`/`medium`/`weak`, default `strong`) |
| `set_narration` | `narration` (from `phasor.voice.narrate`) or `null`; replaces the markers it owns |
| `remove_constraint` | `rule` |

On a scene built by a script, `set_param`, `set_style` and `add_constraint` are stored as
tweaks in `name.overrides.json` and survive re-running the script; other ops are refused
with `scripted_scene` (change the script instead).

`lint_added` and `lint_resolved` are worked out when you first read them (or call
`to_dict()`), so building a scene stays fast.

## Layout constraints: say where things go, not coordinates

```python
from phasor import Dot, Scene, Text

s2 = Scene("layout")
s2.add(Dot(id="pole", pos=(-2, 0), r=0.2, tags=["key"]))
s2.add(Text(id="pole.label", text="pole"))  # the label of X is the child object X.label
s2.add(Text(id="title", text="Residues", pos=(0, 3), style={"size": "title"}))
s2.constrain("label(pole) near pole")  # the placer picks a free spot and keeps it there
s2.constrain("on_screen(title)")
s2.at(0).to("pole.pos", (2, 1), duration=2)

assert s2.engine.layout.slots["pole.label"][0] == "n"  # sits above, and follows the pole
assert s2.engine.lint() == []
```

A rule is `SUBJECT RELATION TARGET [gap N]`, or a guard `NAME(...)`; join several with `and`.
The subject is what moves; it needs a `pos` parameter that is not an expression.

| Rule | Meaning |
| --- | --- |
| `a above b`, `below`, `left_of`, `right_of` (`gap N` optional) | touch that side, centered on it |
| `a near b` (`gap N` optional) | label placer: 8 slots around `b` plus 8 leader-line slots further out |
| `a align_x b`, `a align_y b` | same x (stacked) or same y (side by side) |
| `a inside b` | `a`'s box within `b`'s box |
| `a outside b` (`gap N`) | no overlap: leave by whichever side needs the least movement |
| `a tangent_to b` | touch the outside edge of `b` (boxes), lined up with it |
| `a attach_to b at top` (`gap N`) | sit at a named spot of `b`: `top`, `bottom`, `left`, `right`, `center`, `top_left`, ..., or one of its feature points such as an arrow's `tip` |
| `on_screen(a)` or `on_screen(a, margin)` | keep `a` inside the frame (moves it when it can) |
| `no_overlap(a, b, ...)`, `min_size_px(a, n)`, `readable(a)` | checked by the linter |

- `label(x)` means the object `x.label`. Numbers are in the object's layer units.
- Layout is solved at every marker, every transition end and 10 times a second, then baked.
  `engine.state_at(t)`, the inspector and the renderer all read the same bake.
- A label changes slot only when another slot is over 20% better, and then it jumps.
- `engine.layout` shows the bake: `times`, `slots` (per label, per key time), `issues`.
- A rule that cannot hold is a lint error `unsolved`, with the times. Nothing is dropped.
- `visible_in_3d(a)` (not hidden by anything in front of it) is a guard checked by the linter.
- `camera keep_visible(a)` and the other camera goals are in the 3D section.

## Lint: find problems before rendering

```python
s2.add(Text(id="note", text="note", pos=(-2, 0.1)))  # dropped right on top of the pole
issues = s2.engine.lint()
found = {(i.code, i.ids) for i in issues}
assert ("overlap", ("note", "pole")) in found  # (with OpenGL, also "contrast": it sits on the dot)
assert issues[0].severity == "warning" and issues[0].times and issues[0].fix_hint
assert s2.engine.apply({"op": "set_param", "path": "note.pos", "value": [-4, -2]}).ok
assert s2.engine.lint() == []
```

`engine.lint()` checks every marker, every transition end and every 0.5 s (`step=`).
Each issue has `code`, `severity`, `message`, `ids`, `times`, `fix_hint`, and `fix` (patch ops).

| Code | Severity | Means |
| --- | --- | --- |
| `overlap` | warning | a label, or an object tagged `key`, overlaps something |
| `offscreen` | warning | a label or key object is cut by the frame |
| `contrast` | warning | text too close in brightness to what is behind it (with OpenGL: measured on the real picture) |
| `missing_glyph` | warning | text uses characters the font cannot draw (`²`, `π`, accents): they show as boxes |
| `tiny` | warning | an object or text too small to read |
| `orphan` | warning | an object that never appears |
| `clutter` | warning | too many objects change at once in one block (between two markers) |
| `unsolved` | error | a constraint or guard does not hold |
| `raw_motion` | warning, strict only | a transition written with raw `to()` instead of a verb |
| `orphan_override` | error | a tweak whose object is gone |
| `missing_file` | warning | a file an object reads (a room's `.obj`) is not in the scene's folder; it draws nothing |

`engine.autofix()` applies every `fix` the linter offers (add `near` for a child label that
overlaps, `on_screen` for something cut by the frame), lints again, and returns
`{"applied": [...], "remaining": [...]}`. Each round is one undo step. Thresholds live in
the theme's `[lint]` table.

## Storyboards: look at the whole scene at once

```python no-run
# no-run: needs OpenGL 4.3
board = engine.storyboard(marks=True)  # start, every marker, end; or pass a list of times
save_png("out/board.png", board["image"])  # one contact sheet, captions show time and marker
board["frames"]  # [{"t", "marker", "objects": brief inspector lines}]
```

`marks=True` draws each object's box and id on the stills, so a vision model can tie what it
sees to object ids.

## Inspect: see what is on screen

```python
snap = engine.inspect(3.0, detail="brief")  # one line per object, at most 40
print(*snap["objects"], sep="\n")

where = engine.where("pole_i", 3.0)  # region, bbox, center, depth, relations
assert where["center"]["px"] == [1140.0, 360.0] and where["center"]["norm"] == [0.594, 0.333]

vis = engine.visible("pole_i", 3.0)  # visible_fraction, on_screen_fraction
assert vis["on_screen_fraction"] == 1.0

engine.occluders("pole_i", 3.0)  # {"hidden_by": [{"id", "share"}, ...]}
assert engine.hit_test(1140, 360, 3.0) == ["pole_i"]  # ids under an output pixel, topmost first
engine.inspect(3.0, id="pole_i", detail="full")  # adds projected feature points
```

- `detail`: `brief`, `standard` (default) or `full`. Start with `brief`.
- `visible_fraction`: share of the object's on-screen silhouette that nothing hides.
- `on_screen_fraction`: share of the whole silhouette that is inside the frame.
- `hidden_by`: who covers it, with shares.
- `relations`: `left_of`, `above`, `in_front_of`, `overlaps` against objects tagged `key`,
  all as seen by the camera.
- `"occlusion": "pixel"` means measured from the id and solo render passes. `"analytic"`
  means no OpenGL here: geometry is still exact, `visible_fraction` and `hidden_by` are `null`.
- `note` is set when an object is too thin to measure at inspector resolution (480x270).

## Stills and files

```python no-run
# no-run: needs OpenGL 4.3 and writes files
from phasor import load
from phasor.render.output import save_png

save_png("out/frame.png", engine.still(3.0, (960, 540)))  # (h, w, 3) uint8 sRGB
picture = engine.frame(3.0, (3840, 2160))  # linear, after bloom and tone map; float32
bytes8 = engine.frame(3.0, (3840, 2160), output="srgb8")  # (h, w, 3) uint8, encoded on the card
s.save("residues.phasor.json")  # canonical, diff-friendly
engine = load("residues.phasor.json")  # or load("residues.py")
```

## GPU kernels: per-pixel math

```python
from phasor.render import gpu

f = gpu.ComplexFunction.parse("1/(z**2 + 1)")  # one formula: kernel code, NumPy and LaTeX
assert f.latex() == r"\frac{1}{z^{2} + 1}"

picture = gpu.render("domain", 160, 90, function=f, center=(0, 0), height=4.5)
assert picture.shape == (90, 160, 4)  # linear RGBA float32, row 0 = top

field = gpu.render("escape", 160, 90, center=(-0.5, 0), height=2.5, max_iter=100)
assert field.shape == (90, 160) and (field == -1).any()  # -1 marks points in the set
```

| Kernel | Parameters | Returns |
| --- | --- | --- |
| `domain` | `function`, `center`, `height`, `contour=0.35`, `lut=None`, `precision="f32"`, `a=0j` (the constant `a` in the function) | RGBA: hue = phase, bands = each doubling of the modulus, poles white |
| `domain_glow` | `function`, `center`, `height`, `lut=None`, `radius=0.3`, `ambient=0.08`, `contour=0.15`, `precision="f32"`, `lines=0`, `line_glow=0.6`, `a=0j` | RGBA: hue = phase, lit near zeros and poles (Newton distance \|f/f'\|), rays and rings about a pixel wide; alpha = glow weight |
| `escape` | `center`, `height`, `max_iter=256`, `julia=None`, `precision=None` | smooth escape count; Mandelbrot, or the Julia set of `julia=(re, im)` |
| `fractal_glow` | `field` (from `escape` or `deep`), `lut`, `density=0.35`, `offset=0.0`, `inside=(0,0,0)`, `radius=2.0` (pixels), `ambient=0.08` | RGBA: the palette ramp by sqrt(count), lit along the edge of the set (distance from the count's gradient, 1 / (ln 2 \|grad\|)); inside = `inside`, alpha 0; alpha = glow weight |
| `tonemap` | `image`, `exposure=1.0` | ACES filmic curve, linear in and out (whole image) |
| `gauss` | `image`, `sigma`, `axis` | one pass of a Gaussian blur along x (0) or y (1) (whole image) |
| `srgb8` | `image` | linear display RGBA (0..1) -> (h, w, 3) uint8 sRGB (whole image) |
| `accumulate` | `image`, `weight=1.0`, `acc=None` | float32 `acc + weight * image`; `image` may be float16 (whole image, any channel count; the motion-blur sum) |
| `splat` | `points` ((n, 2) plane coordinates), `colors` ((n, 3) linear RGB, or one), `center`, `height`, `weights=None`, `sigma=1.0` (px), `gain=1.0`, `curve=True`, `white=0.0` | each point adds a Gaussian of light (integer `atomicAdd` in fixed point, so the same bits every run); rgb = mean colour (toward white by white x a^4) x a, alpha a = 1 - exp(-gain x density); `curve=False`: the raw sums |

- `center` and `height` describe the view like the 2D camera: the point at the middle of
  the frame and how many units the frame is tall.
- `gpu.render(name, width, height_px, supersample=1, backend="auto", **params)` runs tile by
  tile within the VRAM budget and returns a NumPy array. `gpu.run(name, tile, **params)`
  runs one tile and returns a CuPy array on CUDA.
- With no CUDA (tier B) the same CUDA source is compiled for the CPU by g++ or clang++
  (found at run time, cached in the user cache) and runs on every core; with no compiler
  either, the NumPy reference runs: same picture, slower. `backend="cuda"` demands the GPU,
  `backend="cpu"` the CPU backend, `backend="reference"` forces NumPy. `gpu.run(...,
  threads=n)`, `phasor.render.cpu.set_threads(n)` or `PHASOR_THREADS` set the CPU threads
  (default: every core); results are the same bits at any thread count.

```python
import numpy as np

from phasor.render import cpu, gpu
from phasor.render.gpu.registry import active_backend

print(active_backend(), cpu.threads(), cpu.find_compiler())  # e.g. "cpu", 4, g++ 13.3
if cpu.find_compiler() is not None:
    lit = np.ones((4, 8, 4), dtype=np.float32)
    one = gpu.run("vignette", gpu.Tile.full(8, 4), image=lit, backend="cpu", threads=1)
    all_cores = gpu.run("vignette", gpu.Tile.full(8, 4), image=lit, backend="cpu")
    assert np.array_equal(one, all_cores)  # the thread count never changes a result
```
- `escape` picks float64 by itself once `height` is under 1e-5.
- Formulas may use `z`, `i`, `pi`, `e`, numbers, `+ - * / **` and
  `exp log sqrt sin cos tan sinh cosh tanh conj abs`. Anything else is a `FunctionError`.
- A new kernel needs `@kernel(name, signature, reference=numpy_fn)`; registration fails
  without a NumPy reference that takes the same parameters.

## Looks, presets, living motion and taste

Start a scene from a **look**: a theme plus a few choices, by name. A new scene gets the
**`prism` look** unless it names another look or a theme; `look="none"`
gives a plain scene on the dark `midnight_glow` theme. Everything stays dark by default;
light is opt-in. (The test suite sets `PHASOR_DEFAULT_LOOK=none`, so its own examples here
run plain.)

| Look | Theme | Life | Character |
| --- | --- | --- | --- |
| `calm` | `midnight_glow` | `subtle` | soft contour edges, gentle glow, slow drift |
| `glow` | `midnight_glow` | `lively` | strong bloom, everything breathes |
| `chalk` | `paper` | `off` | light page, ink strokes, no glow, crisp, perfectly still |

Themes: `midnight_glow` (dark blue-black), `aurora` (dark green-black, teal and violet),
`paper` (light). `Scene(title, theme=..., look=..., life=...)`; explicit `theme` and `life`
beat the look's. The document stores the resolved values, so a look changing later never
changes an old scene.

**Object presets** are named starting points: `Dot(id="d", preset="beacon")`,
`cx.DomainColoring(id="p", function="z", preset="soft_phase")`. Precedence: what you set
yourself, then the preset, then the look, then the type's default. Presets: `dot`: `plain`,
`glowing`, `beacon`; `line`: `ink`, `glowing`, `beam`, `hairline`; `domain_coloring`:
`classic` (hard edges), `soft_phase` (restrained wheel, feathered), `vivid`, `prism` (dark, glowing zeros and poles, rays and rings of light; the `prism` looks use it); `fractal`:
`classic`, `ember`, `ocean`, `prism` (the set dark, its edge glowing; the `prism` looks use it); `riemann_surface`: `glass`, `soft_glass`.

**Palettes** are style roles (`colormap`): `phase` (the full wheel: use it where hue means
phase), `soft_phase` (three-hue wheel), `sequential`, `ember`, `ocean` (two-tone ramps),
`spectrum` (a bright even wheel for color running along a curve).
Fewer rainbows does not mean none: domain coloring keeps the full wheel.

**Edges**: `cx.DomainColoring(softness=0..1)` feathers the modulus bands (0.5 by default).

**Prism domain coloring**: `preset="prism"` (or the `prism` looks) draws the plane dark,
lights each zero and pole with the phase round it as colour, and draws rays of constant
phase and rings of constant size as thin light; the glow blooms and hazes like the strokes.

```python
from phasor import Scene, cx

lit = Scene("lit poles", look="prism", passes=None)
lit.add(cx.DomainColoring(id="plane", function="(z**3 - 1)/(z**2 + 0.5)", size=(8, 4.5)))
assert lit.engine.document.object("plane").params["radius"] > 0
```

A kernel picture glows through its alpha channel: alpha is a per-pixel glow weight, and the
object's `glow` role (above 1) turns it into light and bloom. Kernels that only paint leave
alpha at 1 and the role at `none`.

**Prism fractals**: `cx.Fractal(preset="prism")` (or the `prism` looks) keeps the set dark
(the background), colours the outside from the palette by escape time, and lights the edge
of the set out to `radius` world units, with haze. The glow is sized on screen, so it stays
as fine through a deep zoom (the perturbation path glows too). Palettes still apply.

```python
from phasor import Scene, cx

edge = Scene("glowing edge", look="prism", passes=None)
edge.add(cx.Fractal(id="j", kind="julia", c=(-0.7741, 0.1501), size=(8, 4.5)))
assert edge.engine.document.object("j").params["radius"] > 0
```

**Living motion** (`life`: `off`, `subtle`, `lively`): glow breathes, dots drift a fraction
of their radius, color fields turn a few percent of the wheel, all a deterministic function
of (object id, `seed`, t), so the same frame always renders the same. Tag an object
`"exact"` to keep it perfectly still; the geometry of a field (a pole's place) never moves.
The default for plain `Scene(...)` is `off`; looks turn it on.

```python
from phasor import Dot, Scene, cx

calm = Scene("alive", look="calm", passes=None)
calm.set("camera.height", 4.5)
calm.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", preset="soft_phase"))
calm.add(Dot(id="pole", pos=(0, 1), r=0.08, preset="beacon", tags=["key"]))
calm.add(Dot(id="origin", pos=(0, 0), r=0.08, tags=["exact"]))  # never wiggles

meta = calm.engine.document.meta
assert (meta.theme, meta.life, meta.look) == ("midnight_glow", "subtle", "calm")
assert calm.engine.document.object("plane").params["softness"] == 0.8  # from the preset
assert calm.engine.state_at(0.0)["origin"].params.pos == (0.0, 0.0, 0.0)
assert calm.engine.state_at(2.0)["pole"].params.pos != calm.engine.state_at(0.0)["pole"].params.pos
```

**Math.** Any text written as `$...$` is typeset (TeX math through matplotlib's mathtext,
Computer Modern, no TeX install; for a real TeX set `PHASOR_MATH=latex`, or `auto`, with `latex`
and `dvipng` on the PATH (MiKTeX, TeX Live); `mathtext` stays the default so pictures match
across machines): `Text(text=r"$\psi(x) \approx x - \ln 2\pi$")`. Its box can
be taller than a line (sums, fractions) and layout sees the real size. Bad TeX is refused when
the object is added. For terms in different colors, `Equation(parts=(...), colors=(...))` sets
parts side by side on one baseline (a relation at a part's edge gets TeX's thick space). A
color inside a fraction is not possible: split where a part can end.

**Typeset and LaTeX, both ways.** Write maths in either dialect; it renders on either
engine, because every math string passes through `to_mathtext` (mathtext, the default) or
`to_latex` (real TeX) first. `to_mathtext` maps what mathtext lacks to its nearest form
(`\tfrac`, `\le`, `\lvert`, `\textbf`, `\big(`, `\pmod{n}`, `\xrightarrow{f}`) and drops,
keeping the content, what it cannot show (`\color`, `\boxed`, `\underbrace`; listed by
`texconv.mathtext_losses`); environments (`cases`, `pmatrix`) have no mathtext form and are
refused when added. `to_latex` turns Unicode (`π`, `≤`, `ℝ`) and mathtext-only commands
into LaTeX and escapes the text around the maths. Text and Equation report the LaTeX form
as `semantic["latex"]`.

```python
from phasor import Scene, Text, to_latex, to_mathtext

assert to_mathtext(r"$\tfrac{1}{2} \le \lvert z \rvert$") == r"$\frac{1}{2} \leq | z |$"
assert to_latex("50% of $2π r$") == r"50\% of $2\pi r$"
s = Scene("both")
s.add(Text(id="half", text=r"$\tfrac{1}{2} \le x$"))  # LaTeX-only commands, typeset by mathtext
assert (
    s.engine.inspect(0.0, id="half")["objects"][0]["semantic"]["latex"] == r"$\tfrac{1}{2} \le x$"
)
```

**The prism look** (`look="prism"`, the reference style): near-black stage, curves
glowing with a hue running along them (`Line(hue_along=...)`, the `spectrum` colormap) and
warm/cool fringes on the glow (theme `[bloom] fringe`), a vignette (`[bloom] vignette`),
and typeset equations that float (life `float`).
No axes unless the scene adds them. `examples/prism_primes.py` is the reference scene.

**The prism look on everything** (`look="prism_everywhere"`, an option; the default stays
`midnight_glow`): the same lens (fringes, vignette, drifting equations), and *every* glowing
curve runs the spectrum along its length and lets it drift: graphs, Fourier partial sums and
epicycles, parametric curves, contours, the algebra and PDE drawings. Its theme
`prism_vivid` starts from `prism` (`base = "prism"`) and adds `[strokes] hue_along`, which
any theme may set. Strokes that do not glow (axes, grids) keep their color, and objects that
color by meaning (flow lines by speed, fields by strength) keep theirs.

**The prism look on a field** (`look="prism_field"`, for domain coloring scenes) is `prism`
with two changes, and `prism_domain` and `prism_transitions` use it:

- Equations drift less. `Equation(drift=)` is the share of the theme's `[life] float` that
  an equation drifts; the look sets it to 0.6.
- The field is more see-through, with more light. A shorter haze (`radius`), a darker wash
  (`ambient`), brighter rays and rings (`line_glow`, 0.6 by default) and more emitted light
  (`light`, which scales what the glow role adds; 1 by default).

A look file may start from another with `base = "prism"` and change only what it names.
During a `transform`, an equation's drift passes from the source's to the target's, so
nothing jumps.

**Palettes within the prism looks.** The lens and glow stay; the colors are a choice:
`Scene(..., look="prism_everywhere", palette="moonlight")`. `spectrum` is the prism rainbow
(the default), `moonlight` is white with a cool and warm shimmer, then `gold`, `glacier`,
`borealis` (green to violet) and `rose`. The palette is stored in the document
(`meta.palette`); it colors the hue run along every glowing stroke and the lines the look
runs a hue along. On an open scene, `scene_palette("gold")` (the AI's tool, and the editor's
Looks tab) switches it in one undo step. The palettes are colormap roles in every theme, so
any object with a `colormap` may use them too.

```python
from phasor import Line, Scene

lit = Scene("moon", look="prism_everywhere", palette="moonlight", passes=None)
lit.add(Line(id="arc", points=[(0, 0, 0), (1, 1, 0), (2, 0, 0)]))
assert lit.engine.document.meta.palette == "moonlight"
assert lit.engine.document.object("arc").style["colormap"] == "moonlight"
```

```python
from phasor import Equation, Line, Scene

p = Scene("prism", look="prism", passes=None)
p.add(Line(id="wave", points=[(x / 4, (x % 3) / 3, 0) for x in range(-12, 13)]))
p.add(
    Equation(
        id="eq",
        parts=("$E(x) =$", "$\\psi$", "$(x) - x$"),
        colors=("primary", "secondary", "primary"),
        pos=(0, 2, 0),
    )
)
st = p.engine.state_at(3.0)
assert st["wave"].params.hue_along > 0  # the look's default: color runs along curves
assert st["eq"].params.pos != (0.0, 2.0, 0.0)  # equations drift a little (life)
assert p.engine.inspect(0, id="eq")["objects"][0]["semantic"]["colors"][1] == "secondary"
```

**Taste, the art director and variants.** `lint(aesthetic=True)` adds the taste rules:
`focal_point` (one `highlight` per beat), `color_restraint` (three saturated hues at most;
phase coloring exempt), `glow_clip`, `density`, `motion_rhythm`, `margin`, `stroke_mix`.
Thresholds are the theme's `[lint]`. `engine.critique()` scores them (0 is nothing to
criticise) with the markers and the fixes; `engine.polish(rounds=3)` applies them, each round
one undo step; `engine.variants(beat)` shows calmer and bolder looks of one beat side by side;
`engine.storyboard(compare="gallery")` holds frames beside the three nearest in your
`gallery/`. `phasor style tune` reads `style/choices.jsonl` (every pick, appended by the
`choose` tool) and says what you favor.

```python
from phasor import Dot, Scene

rough = Scene("rough", passes=None)
rough.set("camera.height", 6)
rough.add(Dot(id="a", pos=(5.2, 0), r=0.2, style={"color": "highlight"}))  # at the edge
rough.add(Dot(id="b", pos=(0, 0), r=0.9, style={"color": "highlight"}))  # two heroes
rough.marker("beat", 1.0)
before = rough.engine.critique()["score"]
report = rough.engine.polish(rounds=3)
assert report["after"] < before and len(report["rounds"]) <= 3
```

Finals render with motion blur (`phasor render --preset final`; `--motion-blur 0` turns it
off): `engine.frame_blurred(t, fps)` averages sub-frames only where something moves.

Render presets: `preview` (960x540, 30 fps, no blur), `final` (the scene's own size and rate,
blur), `4k`, and `draft` (640x360, 15 fps, no blur, fast encode, fields such as domain coloring
at half resolution): the quickest look at timing and layout, never for a finished video.
`phasor render --preset draft`, or the MCP `render` tool with `preset="draft"`. Only `draft`
gets the cuts; `engine.frame(t, size, field_scale=0.5)` is the picture it draws.

## Domains: fourier, linalg, acoustics

Domain modules register their own object types: `from phasor import fourier, linalg, ac`.
They are plain objects: styled by roles, labelled by `near` rules, animated by verbs, and
every parameter (a matrix, a frequency, a radius) can be tracked or tied to another with an
expression.

**`fourier`**: `Epicycles(points=..., terms=N, phase=0..1)` (a closed curve drawn by N rotating
circles; `phase` animates the tip), `PartialSum(kind="square"|"sawtooth"|"triangle", terms=N)`
(the Gibbs overshoot is in the semantics), `Spectrum(frequencies=, amplitudes=, window=)` (bars
per bin; `leakage` in the semantics), `Spectrogram(...)` (any signal, backdrop; below). The maths is
`phasor.domains.fourier_math`: `epicycle_coefficients`, `series_partial_sum`, `window`,
`spectrum`, `leakage`, `spectrogram`, `dominant_frequency`, `sampled`, `convolve`.

`fourier.Winding` draws a sum of real sine tones wound clockwise at `freq` (Hz). Its
centroid is the mean of the wound samples, equal to the complex DFT divided by the sample
count at DFT bins. The spectrum stroke plots the centroid magnitude from zero to `freq`.
`max_freq` sets the right edge of the spectrum and must be at or below Nyquist (`fs / 2`).
`spectrum_guide=False` hides the spectrum axes while retaining the winding circle.
`progress` traces the winding curve independently of the frequency sweep. `pos` sets the
circle centre and `size` its radius. Inspector semantics expose the complex centroid as
`(real, imag)`; the `centroid` and `spectrum_tip` features are world positions.

```python
from phasor import Scene, fourier

s = Scene("winding_demo", look="prism", passes=None)
s.add(fourier.Winding(id="wind", frequencies=(3.0,), amplitudes=(1.0,), pos=(-4, 0, 0)))
s.marker("scan", 0.5)
s.marker("end", 5.0)
s.at("scan").sweep("wind.freq", 8.0, show=False)
```

See `examples/fourier_winding.py` for a two-tone sweep.

**`linalg`**: `TransformGrid(matrix=((a, b), (c, d)))` (2x2: the grid, the unit square whose
area is the determinant, the images of the basis vectors; 3x3: the unit cube), `EigenLines`,
`Arrow(start=, end=)`, `Matrix(matrix=expr("grid.matrix"))` (entries stay linked). Semantics
give determinant, eigenvalues, singular values and rank. `linalg_math` has `svd` (its
`stages` are rotate, stretch, rotate: morph through them), `null_space`, `gram_schmidt`,
`project`.

**`ac`** (acoustics): `Room(size=(8, 5, 3), material="curtain")` (glass walls: they never hide
the source or listener), `Source`, `Listener`, `Wavefront(center=, radius=expr("343 * t"))` (a
negative radius draws nothing, so `343 * (t - 0.5)` starts at 0.5 s), `ImageSources(order=)`,
`ImpulseResponse(layer="overlay", pos=, extent=)` (arrival spikes; `reveal` and `show_rt60`
animate; semantics give `rt60`, `rt60_theory_eyring` and `rt60_reliable`), `PressureSlice(time=)`
(a slice of the real wave-equation field). Materials and absorption: `ac_math.MATERIALS`.

```python
import numpy as np
from phasor import Scene, ac, expr, fourier, linalg

dom = Scene("domains", passes=None)
dom.set("camera.height", 8)
dom.add(linalg.TransformGrid(id="grid", matrix=((2.0, 1.0), (1.0, 2.0))))
dom.add(linalg.Matrix(id="mat", matrix=expr("grid.matrix"), pos=(-5.5, 3.2, 0)))  # live entries
dom.add(fourier.PartialSum(id="wave", kind="square", terms=9, pos=(0, -3, 0), extent=(5, 1.6)))
dom.marker("go", 1.0)
dom.at("go").morph("grid.matrix", ((1.0, 0.0), (0.0, 3.0)))

e = dom.engine
assert e.state_at(3.0)["mat"].params.matrix == ((1.0, 0.0), (0.0, 3.0))  # the numbers follow
sem = e.inspect(0.0, id="grid")["objects"][0]["semantic"]
assert sem["determinant"] == 3.0 and sem["singular_values"] == [3.0, 1.0]
assert 0.085 < e.inspect(0.0, id="wave")["objects"][0]["semantic"]["gibbs_overshoot"] < 0.1

room = Scene("room", passes=None)
room.add(ac.Room(id="room", size=(8, 5, 3), material="curtain"))
room.add(
    ac.ImpulseResponse(id="ir", material="curtain", order=10, layer="overlay", pos=(0.1, 0.9, 0))
)
ir = room.engine.inspect(0.0, id="ir")["objects"][0]["semantic"]
assert abs(ir["first_arrival_ms"] - 4.0311 / 343 * 1000) < 1e-3 and ir["arrivals"] > 1000
```

**`ac.Directivity`** draws normalized pressure magnitude as a polar curve. Choose
`kind="array"` (a uniform linear array), `"piston"` (a circular piston in an infinite
baffle), or `"cardioid"`. `steer` is in degrees counterclockwise from +x. The array lies along y,
so +x is broadside. Front/back symmetry can yield two equal peaks.
`elements` and `spacing` (meters) set the array, `radius` (meters) sets the piston, and
`frequency` (Hz) applies to both. `pos` sets the plot center and `size` is the radius of
the unity-gain guide in world units. The prism look gives stronger lobes more glow.

```python
from phasor import Scene, ac

beam = Scene("beam", look="prism", passes=None)
beam.add(ac.Directivity(id="array", elements=8, spacing=0.1715, frequency=1000, steer=-25))
beam.marker("turn", 1.0)
beam.at("turn").sweep("array.steer", 25.0, show=False)
assert beam.engine.inspect(0, id="array")["objects"][0]["semantic"]["peak_gain"] == 1.0
```

**`ac.Doppler(speed=, frequency=, time=, pos=, size=)`** draws wavefronts from a sound
source traveling along +x. `speed` is the constant Mach number for the frame,
`frequency` is in Hz, `time` is seconds from emission start, `pos` is the starting
position, and `size` is the world distance sound travels in one second. Animate `time`
to move the source. For a different constant speed, restart the run. In the prism look
the rings fade with age and their hue encodes local apparent frequency. At Mach 1 the
leading edges meet at the source. Above Mach 1 the lines show the planar section of
the Mach cone. Inspector semantics use `None` for frequency ahead at and above Mach 1
and report the cone half-angle in degrees for supersonic motion.

```python
from phasor import Scene, ac, expr

doppler = Scene("doppler", look="prism", passes=None)
doppler.add(ac.Doppler(id="source", speed=0.5, frequency=12, time=expr("t"), size=2))
doppler.marker("end", 1.0)
wave = doppler.engine.inspect(0.8, id="source")["objects"][0]["semantic"]
assert wave["frequency_ahead"] == 24.0 and wave["frequency_behind"] == 8.0
```

Example: `examples/doppler.py`.

**`ac.Interference(sources=, frequency=, speed=, time=, phases=, amplitudes=, onsets=, mode=,
blend=, pos=, size=)`** is the field of point sources of one frequency, drawn per pixel on
the GPU (kernel `wave_field`, a NumPy fallback without one) in the rectangle `pos`, `size`
of the plane z = 0. `sources` are world (x, y) points; each sends out
`amp cos(k r - w t + phase) / sqrt(r)` (2D spreading). The wavelength is `speed / frequency`
(world units), `time` is in seconds (`expr("t")` sets the waves moving), `phases` are degrees
and `amplitudes` plain numbers, one per source. `onsets` (one `time` per source) switch the
sources on one by one, each sending out a ring. `mode="intensity"` (or `blend` tweened to 1)
shows the time-averaged intensity, where the fringes stand still. The prism look colours it
as a heat map: crests (and loud places) warm, troughs cool, the lines where the waves cancel
dark, the sources as beacons. `gain` sets how strongly the field turns into colour.

**`ac.PhasedArray(count=, spacing=, steer=, center=, layout=, axis=, ...)`** is the same field
from `count` sources `spacing` wavelengths apart on a line (`layout="line"`, along the
direction `axis`, default 90: up) or a ring, centred on `center`, with the phase delays that
point the beam `steer` degrees off broadside, counterclockwise (with the default `axis`
broadside is +x, as for `ac.Directivity`). Tween `steer` to swing the beam. It also draws
the elements as beacons and the beam direction as a ray (`ray=False` to hide it); its
semantics report `beam_direction_deg`, the element `phases_deg` and, for a line,
`beamwidth_deg` (the half-power width).

```python
from phasor import Scene, ac, expr

waves = Scene("waves", look="prism", passes=None)
waves.add(ac.Interference(id="pair", sources=((-1, 0), (1, 0)), time=expr("t"), size=(8, 4.5)))
waves.add(ac.PhasedArray(id="array", center=(-3, 0), count=4, steer=-30, time=expr("t")))
waves.marker("swing", 1.0)
waves.at("swing").morph("array.steer", 30.0)
pair = waves.engine.inspect(0.5, id="pair")["objects"][0]["semantic"]
assert pair["wavelength"] == 1.0 and pair["sources_on"] == 2
beam = waves.engine.inspect(5.0, id="array")["objects"][0]["semantic"]
assert beam["beam_direction_deg"] == 30.0 and beam["beamwidth_deg"] < 30
```

Example: `examples/interference.py`.

**Simulations** (`phasor.core.simulation`): a `Simulation` has a fixed `dt`, `params()` (the
seed included), `initial()` and a pure `step(state, dt)`. `runner_for(sim).state_at(t)` is the
nearest earlier checkpoint plus the remaining steps, cached in memory and under
`.phasor_cache/<hash of the parameters>/` (or `PHASOR_CACHE`), so scrubbing is cheap and the
answer never depends on how you got there. `ac_math.RoomFDTD` is one (it refuses a `dt` above
`dx / (c sqrt(3))`: that is a validation error, not a blow-up).

## Domains, wave 2: pde, algebra, calc

`from phasor import pde, algebra, calc`.

**`pde`**: `Profile(kind="heat"|"wave"|"advection"|"schrodinger", time=...)` (a 1D solution
as a curve, the exact solution drawn beside it), `Field(kind="heat"|"wave"|"reaction")` (a 2D
colour plane, a backdrop) and `Surface(...)` (the same as a lit 3D surface). `time` is
simulation time, usually `expr("t / 10")`; frames are checkpointed simulations, so scrubbing is
cheap. Semantics report the max, `max_error_vs_exact`, the conserved quantity (`total_heat`,
`norm`). An unstable step is a validation error. The numerics are in `domains/pde_math.py`
(`Heat1D`, `Heat2D`, `Wave1D`, `Wave2D`, `Advection1D`, `GrayScott`, `Schrodinger1D`,
`solve_poisson`), all `Simulation`s.

**`algebra`**: `Polygon(n=4, rotation=deg, flip=0..1)` (a dihedral group acting on an n-gon;
the semantics name the group element when it lands on a symmetry), `CayleyTable(group="D4")`
(`D<n>`, `Z<n>`, `S<n>`; coloured by element, a Latin square), `ClockDiagram(n=12, step=4)`
(a cyclic group, a subgroup and its cosets). Groups are in `algebra_math` (`dihedral`, `cyclic`,
`symmetric`: subgroups, cosets, normality, Lagrange).

**`calc`**: `RiemannSum(function="x**2", a=0, b=3, n=8, rule="mid")` (`n` is a number, so
animate it and watch the error fall), `Taylor(function="sin(x)", order=5)` (animate `order`),
`Graph(function=...)` (breaks at poles), `VectorField(p="-y", q="x")` (divergence and curl in the
semantics). Functions are text in `x` (and `y`), read with a whitelist, never evaluated as Python.

```python
from phasor import Scene, algebra, calc, expr, pde

s9 = Scene("wave2", passes=None)
s9.set("camera.height", 10)
s9.add(pde.Profile(id="rod", kind="heat", length=2.0, diffusivity=0.5, modes=(1.0, 0.4),
                   time=expr("t / 10"), pos=(-8, 1, 0)))  # fmt: skip
s9.add(algebra.Polygon(id="square", n=4, pos=(2, 2, 0)))
s9.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=4.0, pos=(-2, -4, 0)))
s9.marker("go", 1.0)
s9.at("go").morph("square.rotation", 90.0)
s9.at("go").morph("area.n", 64.0)

e9 = s9.engine
rod = e9.inspect(3.0, id="rod")["objects"][0]["semantic"]
assert rod["max_error_vs_exact"] < 3e-3  # the solver sits on the exact solution
assert e9.inspect(3.0, id="square")["objects"][0]["semantic"]["element"] == "r"
assert (
    e9.inspect(0.0, id="area")["objects"][0]["semantic"]["error"]
    > 10 * (e9.inspect(3.0, id="area")["objects"][0]["semantic"]["error"])
)  # more pieces, smaller error
```

**More of each module.** `pde.Profile(kind="heat", boundary="neumann", method="spectral")`:
insulated ends (drawn as walls; held ends are drawn as pins) and a spectral solver that is
exact per mode at any step. `pde.Modes(modes=(1, 0, 0.5), count=7, time=...)`: separation of
variables, each mode drawn faintly and decaying at its own rate, their sum bright (semantics:
`amplitudes`, `decay_rates`). `pde.Slice(axis="z", at=0.5, time=...)`: one plane through a
heating cube (`Heat3D`), inside its outline; animate `at` to sweep (orbit camera for 3D).
`algebra.CayleyGraph(group="D4", generators="r s")`, `CayleyTable(subgroup="r", cosets=1)`
(rows in coset order, colored by coset; semantics give the quotient table when the subgroup is
normal), `Homomorphism(n=12, m=4, a=1)` (Z_n -> Z_m, kernel highlighted), `SubgroupLattice`
(Hasse diagram, normal subgroups highlighted), `GaloisDiagram(primes=(2, 3))` (fields of
Q(sqrt 2, sqrt 3) beside the subgroups that fix them), `Permutation(perm=(1, 2, 0), progress=...)`
(tokens travel; semantics: cycles, sign, order). `calc.Limit(function="sin(x)/x", a=0,
epsilon=...)` (the epsilon band and the delta it forces; `holds`), `calc.LoopIntegral(p=, q=,
measure="flux"|"circulation", progress=...)` (the running line integral around a circle, equal at
the end to the integral of the divergence or curl inside: `theorem_holds`).

```python
from phasor import Scene, algebra, calc, expr, pde

more = Scene("more", passes=None)
more.add(pde.Modes(id="modes", modes=(1.0, 0.0, 0.5), count=5, time=expr("t / 4")))
more.add(algebra.CayleyTable(id="cosets", group="D4", subgroup="r2", cosets=1.0, pos=(8, 0, 0)))
more.add(calc.LoopIntegral(id="green", p="-y", q="x", measure="circulation", pos=(0, 6, 0)))


def info(oid, t=0.0):
    return more.engine.inspect(t, id=oid)["objects"][0]["semantic"]


assert info("modes", 2.0)["amplitudes"][2] < info("modes", 0.0)["amplitudes"][2] / 2
assert info("cosets")["normal"] and info("cosets")["quotient"]["order"] == 4
assert info("green")["theorem_holds"]
```


**Heat maps in the prism look.** `pde.Field`, `pde.Slice`, `acoustics.PressureSlice` and
`fourier.Spectrogram` take `smooth` (bicubic between grid values) and `isolines` (that many
contour levels drawn as thin light), and a `glow` style role: above `none`, strong parts
shine and haze and quiet parts melt into the background. A wave glows both ways, warm for up
and cool for down (`positive` and `negative` roles). `preset="prism"` sets all three; the
`prism` looks use it. Plain pictures are unchanged.

The prism preset also colours by the spectrum: `hue_cycles` > 0 reads the `colormap` role
(`spectrum` in the prism looks, or the scene's `palette`) as a wheel, its hue turning that
many times over the value range, so each contour line has the hue of its level; on a
`Spectrogram` the hue runs up the frequency axis, so a sliding tone's trace runs through the
spectrum like a prism stroke. Brightness then comes from the field's strength alone: the
glow weight raised to `contrast` (above 1, only the strongest parts light up), with
`ambient` the share a quiet pixel keeps (0: the dark ground). `fade` melts the edges into the
background over that share of the shorter side, so there is no square border and no square
halo. `hue` turns the colours; a lively look drifts it a little. All default to the plain
picture; they are taste knobs for looks and presets, not numbers for a scene to type.

```python
from phasor import Scene
from phasor.domains import pde

hot = Scene("hot plate", look="prism", passes=None)
hot.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=0.1, scale=0.3))
assert hot.engine.document.object("plate").params["isolines"] == 8
plate = hot.engine.document.object("plate")
assert plate.params["hue_cycles"] > 0 and plate.params["fade"] > 0
assert plate.style["colormap"] == "spectrum"
```

The height mode, `pde.Surface`, has a prism preset too (the prism looks use it): with
`hue_cycles` > 0 the surface is dark where the field is quiet and takes the spectrum's hue of
its value where it is strong (`contrast` as above), under a net of `lines` grid lines each
way drawn as glowing strokes in the same hues (`stroke` and `glow` roles). Give the scene an
orbit camera (`examples/prism_drum_3d.py`).

```python
from phasor import Scene
from phasor.domains import pde

drum = Scene("drum", look="prism", passes=None)
drum.use_camera("orbit")
drum.add(pde.Surface(id="skin", kind="wave", mode=(2, 3), scale=1.2))
assert drum.engine.document.object("skin").params["lines"] > 0
```

**Point clouds.** `from phasor import cloud`. `cloud.PointCloud` splats many points (a
million is fine on the GPU) as tiny Gaussians of light that add up, so dense regions glow and
the structure at their edges shows. Sources: `roots` (the roots of every polynomial of
`degree` with coefficients from `coefficients`, the leading one 1; a random sample when
there are more than `count` roots), `attractor` (`kind` clifford, dejong or lorenz, fitted to
[-1, 1]^2), `buddhabrot` (escaping orbits up to `max_iter`) or `points` given directly. A point
sits at world `pos + scale * point`; the cloud is seen inside the `size` rectangle round `pos`
and only the part on screen is splatted, so zooming in keeps it sharp. `exposure` is light
per point against the mean density at the current zoom (resolution does not change it);
`color_by="phase"` colours by each point's angle round `pos`, `source` by what the source
says (the polynomial's place in the list). Presets `prism` (the `prism` looks use it) and
`plain`. Points are cached per source parameters, so animating `hue`, `exposure` or the
camera does not recompute them. Numerics: `phasor.domains.cloud_math`.

Motion: `twist` (roots) turns the last coefficient of the set round the origin by that share
of a turn (b -> b e^(2 pi i twist); the leading coefficient stays), so tweening it with
`.to("id.twist", 0.06)` makes every root glide and the fractal edge tear and morph; the same
polynomials are used at every value, so colours stay with them. A million roots solve in about
half a second (LAPACK on all cores) per new value. `twinkle` flickers each point's light by up
to that share (the roots of one polynomial together), slowly; it is life: `lively` gives the
full amount, `subtle` half, `off` or an `exact` tag none. `white` turns the colour of the
densest parts toward white (by `white * a^4`, a the tone curve's light), never the light
itself. The `prism` preset sets `white` 0.8 and `twinkle` 0.6.

```python
from phasor import Scene, cloud

sky = Scene("roots", look="prism", passes=None)
sky.add(cloud.PointCloud(id="roots", source="roots", degree=8, size=(4.5, 2.6)))
seen = sky.engine.inspect(0.0, id="roots")["objects"][0]["semantic"]
assert seen["points"] == 2**8 * 8 and seen["backdrop"]  # every +-1 polynomial of degree 8
assert sky.engine.document.object("roots").style["glow"] == "strong"
sky.at(0.5).to("roots.twist", 0.05, duration=3.0)  # the roots glide; the ring tears open
assert 0 < sky.engine.state_at(2.0)["roots"].params.twist < 0.05
```

**A 3D room with sound bouncing in it** (`ac`). Give the scene an orbit
camera.
- `ac.Room3D(size=(8, 5, 3), origin=, material=)`: a shoebox, or any closed room read from a
  Wavefront `.obj` file with `path="assets/l_room.obj"` (relative to the scene's folder, and
  inside it: an absolute path or `..` is refused; `scale` turns file units into meters). Drawn
  as faint glowing lines along its edges (for a mesh, its creases: `edge_angle`, so the
  diagonals of flat walls stay hidden) and faint glass walls (`walls`, 0 for lines only;
  `preset="outline"`). A file that is not there is lint `missing_file` and draws nothing.
  Semantics: volume, surface area, walls, Sabine and Eyring RT60.
- `ac.SoundRays(room="room", source=, time=expr("t - 1"), speed=0.01, count=1200,
  bounces=6, trail=1.5)`: rays leaving the source evenly in every direction, reflecting off
  the walls (angle in = angle out). `time` is scene seconds since the sound left, `speed`
  the slow motion (seconds of sound per scene second): each ray is drawn up to
  `343 * speed * time` meters along its path, as a streak `trail` meters long (0: the whole
  path). Brightness is the energy left, (1 - absorption) ** reflections; color comes from the
  `colormap` (the palette) by reflection count, `hue_step` apart. `seed` turns the ray set.
  Semantics: distance, reflections so far, energy left.
- `ac.Wavefront3D(room="room", source=, time=, speed=, order=2, shell=0.12)`: the expanding
  front drawn where it meets the walls, with the reflected fronts (image sources up to
  `order` reflections in a shoebox; one per wall, only where the sound really reaches, in
  any other room). `shell` draws the fronts themselves inside the room, faintly.
- The numerics are `ac_math`: `box_mesh`, `load_obj`, `feature_edges`, `mesh_stats`, `trace`
  and `rays` (specular ray tracing: points, path lengths, energy per bounce),
  `ray_segments` (what has been reached by a distance), `sphere_wall_arcs`, `front_visible`.

An object type that reads files lists the parameters that name them in `Files = ("path",)`
(`phasor.core.files`): validated to stay in the folder, checked by `missing_file`.

```python
from phasor import Scene, ac, expr

room = Scene("clap", look="prism", passes=None)
room.use_camera("orbit", target=(4, 2.5, 1.5), distance=15)
room.add(ac.Room3D(id="room", size=(8, 5, 3), material="wood"))
clap = dict(room="room", source=(2, 2, 1.5), time=expr("t - 1"), speed=0.01)
room.add(ac.SoundRays(id="rays", count=500, bounces=8, **clap))
room.add(ac.Wavefront3D(id="front", **clap))
rays = room.engine.inspect(3.0, id="rays")["objects"][0]["semantic"]
assert abs(rays["distance"] - 6.86) < 1e-6 and rays["reflections_so_far"] > 1
assert room.engine.inspect(0.0, id="room")["objects"][0]["semantic"]["volume"] == 120.0
```

## Graphs and networks (`nw`)

`nw.Network` draws a graph: nodes joined by edges, thousands of them in one draw call. Give
one source: `path` (a file next to the scene: an edge list `.csv`/`.tsv`/`.txt` with
`a,b[,weight]` per line, node-link `.json` whose extra node fields become attributes, or
`.graphml`), `edges` (pairs of node names, with optional `nodes` for their order), or a
`generator`: `tree` (`branching`, `depth`), `grid` (`rows`, `cols`), `random` (`n`, `p`),
`small_world` (`n`, `k`, `p`), `hypercube` (`cube_dim`), `tictactoe` (every position of the
game: 765 up to rotation and reflection, or all 5,478 with `symmetric=False`; node
attributes `move` and `outcome` under perfect play).

`layout`: `spring` (force-directed), `radial` (the root in the middle, each move a ring
further out, every node a wedge of angle in proportion to what grows from it: branches fan
out as rays; game trees), `layers` (breadth first from `root`, a level per move, ordered by
their parents), `circle`, `random`;
`dims` 2 or 3 (3D wants an orbit camera); `size` (world half-width), `pos`. `color_by`:
`depth`, `degree`, or a node attribute; `hue` turns the palette. `grow` (0-1) shows that share
of the edges, nearest the root first (morph it to grow the graph); `node_size` draws dots.
`exposure` is the light per edge: a dense graph shares its light out so it builds up where
edges crowd instead of burning out. `cross_edges` (0-1) dims the edges off the breadth-first
spanning tree (in a game: the same position reached by another order of moves); 0 hides them. Styles: `colormap` (spectrum), `edge`, `node`, `glow`,
`stroke`. Presets `plain`, `prism` (the prism looks use it). A missing file is
`missing_file`; an AI may only load files inside its folder.

```python
from phasor import Scene
from phasor.domains import network as nw

g = Scene("graphs", passes=None)
g.add(nw.Network(id="tree", generator="tree", branching=2, depth=5, layout="layers"))
g.add(
    nw.Network(
        id="ring",
        edges=(("a", "b"), ("b", "c"), ("c", "a")),
        layout="circle",
        pos=(5, 0, 0),
        size=1.0,
    )
)
sem = g.engine.inspect(0, id="tree")["objects"][0]["semantic"]
assert (sem["nodes"], sem["edges"], sem["depth"]) == (63, 62, 5)
```

## Space: axes, curves, surfaces and fields (`geo`)

`from phasor import geo`. Give the scene an orbit camera (`s.use_camera("orbit")`) to turn
around them: the `orbit` verb, or drag in the editor. Formulas are text in `t` (curves) or `u`,
`v` (surfaces) or `x`, `y`, `z` (fields), read by the whitelist parser, never run as Python.

- `geo.Axes(dims=3, x_range=(-3, 3, 1), z_range=..., labels=("x", "y", "z"))`: axes with ticks
  (ranges are (min, max, tick step); `scale` is world units per unit).
- `geo.ParametricCurve(fx="cos(t)", fy="sin(t)", fz="0.2*t", t_range=(0, 37.7),
  color_by="t")`: a curve; `color_by` is `none`, `t`, `speed`, `x`, `y` or `z` (through the
  `colormap`); `progress` traces it. Semantics: length, closed, color range.
- `geo.ParametricSurface(fx=..., fy=..., fz=..., u_range, v_range, res, color_by="height")`:
  a surface; `color_by` is `height`, `u`, `v`, `radius`, or `expr` with `color_expr` (a formula
  in `u v x y z`). **Mapping input to output:** `with_map=True` draws the flat (u, v) input
  rectangle at `domain_pos` as a faint grid and `morph` (0 to 1, animate it) carries the
  surface from that flat input to its true shape. Semantics give the `area` (a half sphere of
  radius r reports 2 pi r^2).
- `geo.VectorField3D(fp="-y", fq="x", fr="0", extent=(4, 4, 4), counts=(5, 5, 3))`: arrows
  colored by size; semantics give divergence and curl. `calc.VectorField(color_by_size=True)`
  does the same in a plane. `spacing=0.5` (3D) places arrows that far apart instead of
  `counts`.
- **How field arrows look** (both types): `preset="dense"` is many small thin arrows with
  heads in proportion, lengths eased (`length_by="soft"`: tanh of size, so weak arrows still
  show and strong ones never touch) and color saying how strong (the `magnitude` colormap,
  cool to warm; exempt from `color_restraint`, since the hue is read as a quantity).
  `preset="bold"` is few thick glowing arrows. Without a preset: lengths in proportion to
  size (`length_by="size"`). A dense 3D *box* gets crowded: use a plane (extent z = 0) or a
  larger `spacing`.

```python
from phasor import Scene, calc, geo

s = Scene("fields", passes=None)
s.add(calc.VectorField(id="swirl", p="sin(y)", q="-sin(x)", extent=(12.0, 7.0), preset="dense"))
seen = s.engine.inspect(0.0, id="swirl")["objects"][0]
assert seen["semantic"]["arrows"] > 800  # many small arrows
assert s.engine.document.object("swirl").style["colormap"] == "magnitude"
s.add(geo.VectorField3D(id="plane", fp="-y", fq="x", extent=(6.0, 6.0, 0.0), preset="dense"))
assert s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]["arrows"] == 14 * 14
```
- `geo.FlowLines(fp=..., fq=..., fr=..., seeds=(5, 5, 1), length=3.0, progress=...)`: where a
  particle dropped at each seed would travel, colored by the field's size along the line.

```python
import math

from phasor import Scene, expr, geo

s = Scene("space", passes=None)
s.use_camera("orbit", azimuth=40.0, elevation=25.0, distance=14.0, drift=0.0)
s.add(geo.Axes(id="axes", labels=("x", "y", "z")))
s.add(
    geo.ParametricCurve(
        id="helix", fx="cos(t)", fy="sin(t)", fz="0.2*t", t_range=(0.0, 6 * math.pi), color_by="t"
    )
)
s.add(
    geo.ParametricSurface(
        id="ball",
        fx="2*sin(u)*cos(v)",
        fy="2*sin(u)*sin(v)",
        fz="2*cos(u)",
        u_range=(0.0, math.pi / 2),
        v_range=(0.0, 2 * math.pi),
        color_by="height",
        with_map=True,
        morph=expr("t / 4"),
        pos=(4, 0, 0),
        domain_pos=(-4, 0, 0),
    )
)
e = s.engine
assert abs(e.inspect(4.0, id="ball")["objects"][0]["semantic"]["area"] - 2 * math.pi * 4) < 0.1
assert e.inspect(0, id="axes")["objects"][0]["semantic"]["axes"]["x"]["ticks"] == 6
```

## Interactive scenes: sliders, the notebook, a web page

A scene marks numbers a viewer may change: `scene.control("set.c", component=0, min=-1.2,
max=1.2, step=0.05, label="Re c")` (a vector parameter needs its `component`; a parameter
animated by a track cannot be a slider; the document keeps the controls, so they travel with
the scene and show in `diff`). Everything interactive builds on `phasor.interactive.Player`,
which moves sliders on a private copy of the scene (the scene itself never changes):

- the **editor**'s Sliders tab previews them in the viewport ("Apply to scene" makes the
  values part of the scene as one undo step); hovering shows f(z), |f| and arg f over a
  domain coloring, and the escape count over a Julia or Mandelbrot set;
- **Jupyter**: `from phasor.interactive import notebook; notebook(scene.engine)` returns
  widgets (needs `uv sync --extra notebook`);
- **a web page**: `phasor web scene.py -o page.html --steps 7` or `export_web(engine, path)`
  bakes a picture for every combination of slider positions into one self-contained HTML file
  (no server, no network; dragging on the picture moves the first two sliders). It refuses
  more than 600 pictures: use fewer `steps`.

`examples/julia_explorer.py` shows it: a Julia set with sliders for Re c, Im c and the
iteration depth.

```python
from phasor import Scene, cx
from phasor.interactive import Player, export_web

s = Scene("explorer", passes=None)
s.add(cx.Fractal(id="set", kind="julia", c=(-0.8, 0.156), max_iter=60))
s.control("set.c", component=0, min=-1.2, max=1.2, step=0.05, label="Re c")
s.control("set.c", component=1, min=-1.2, max=1.2, step=0.05, label="Im c")

player = Player(s.engine)
assert player.set("set.c[0]", 0.33) == 0.35  # clamped to the range, rounded to the step
assert player.state_at(0.0)["set"].params.c[0] == 0.35
assert s.engine.document.object("set").params["c"] == (-0.8, 0.156)  # the scene is untouched
```

## Manim import

`phasor manim lecture.py -o lecture.phasor.json` (or opening the `.py` in `phasor gui`, or
`scene_open` on a server that allows scripts) runs a Manim Community script *unchanged*
against a stand-in for the `manim` module and records the scene it builds: `self.add` and
`self.play` become objects and transitions at Manim's own times (`run_time`, `wait`), in
Manim's frame (8 units tall, y up). The result is a plain phasor scene you can edit, lint and
render. It runs your code: only import scripts you trust.

| Carries over | How |
| --- | --- |
| `Scene`, `MovingCameraScene`, `ThreeDScene` (`set_camera_orientation`, ambient rotation) | `construct` runs; 3D uses the orbit camera |
| `Dot`, `Circle`, `Square`, `Rectangle`, `Triangle`, `Polygon`, `RegularPolygon`, `Line`, `Arrow`, `Vector`, `ParametricFunction` | `dot`, `line` (closed outlines), `arrow` |
| `Text`, `Tex`, `MathTex` | `text`; math is typeset (`$...$`); `font_size` picks the nearest text role |
| `Axes`, `NumberPlane`, `axes.plot`, `c2p` | `axes` and a sampled `line` |
| `VGroup` and `shift`, `move_to`, `next_to`, `to_edge`, `to_corner`, `scale`, `rotate`, `arrange`, `align_to` | positions are computed as Manim would |
| `Create`, `Uncreate`, `Write`, `FadeIn`, `FadeOut`, `GrowFromCenter`, `Transform`, `ReplacementTransform`, `Rotate`, `Indicate`, `MoveToTarget`, `AnimationGroup`, `LaggedStart`, `Succession`, `.animate` | tracks on `progress`, `opacity`, `pos`, `r`, `points` |
| Colors (`RED`, `"#58C4DD"`) | the nearest style role (`primary`, `secondary`, `muted`, `highlight`, `danger`) |

What does not: fills (drawn as outlines), updaters and `always_redraw` (baked at their first
look), `ValueTracker` animations, arbitrary colors (roles only), font sizes (roles), `Transform`
between different shapes (a crossfade). Each is listed in `ManimImport.warnings` (printed by
the command); a Manim name phasor lacks raises an error that says so.

```python
import tempfile
from pathlib import Path

from phasor.importers import import_manim

path = Path(tempfile.mkdtemp()) / "demo.py"
path.write_text(
    "from manim import *\n"
    "class Demo(Scene):\n"
    "    def construct(self):\n"
    "        c = Circle(radius=1, color=BLUE)\n"
    "        self.play(Create(c), run_time=2)\n"
    "        self.play(c.animate.shift(RIGHT * 2), run_time=1)\n"
    "        self.wait(1)\n"
)
result = import_manim(path)
assert result.duration == 4.0 and len(result.scene.engine.document.objects) == 1
state = result.scene.engine.state_at
assert state(0.0)["circle_1"].params.progress == 0.0
assert state(2.0)["circle_1"].params.progress == 1.0
assert abs(state(3.5)["circle_1"].params.points[0][0] - 3.0) < 1e-9  # moved 2 to the right
```

## Sketch: draw on the scene

Draw on the picture (the editor's **Sketch** tab, or strokes as JSON for `phasor sketch`)
and each stroke becomes a **proposal**: a summary, the patch ops that make it, verbs to run
after them, and what was worked out. Nothing changes until it is accepted (`accept`, one
`Engine.apply` plus the verbs; undo takes it back). Strokes are output pixels (row 0 at the
top), classified by fitting, not by a learned model: `dot`, `line`, `arrow` (a hook at the
end, or a V drawn at the tip), `circle`, `closed`, `scribble`, `curve` (`wavy`), and
`handwriting` (small strokes close together).

| Drawn | Over | Becomes |
| --- | --- | --- |
| a loop | a domain coloring | a glowing contour, traced from the playhead, with its integral of f dz: numeric along the drawn path and exact as 2πi Σ winding × residue (rational f); a label with the value. The direction you draw counts |
| a loop | other objects | they are emphasized (the `highlight` color role) |
| a loop, line, curve | nothing | a Line (a circle is the fitted circle) |
| a dot | a pole or zero (within 14 px) | a Dot snapped onto it |
| an arrow | an object with `pos` | the object moves to the tip; elsewhere, an Arrow |
| a scribble | objects | they are removed (cross it out) |
| handwriting | anywhere | a picture of it (`proposal.image`, PNG) for an AI or the person to read; `transcribe(p, text)` makes the Text. There is no offline handwriting recognizer |

Sketch edits plain scenes. On a scene built by a script, only parameter and style
proposals (move, emphasize) apply; the script's own code can apply the rest with
`engine.apply(p.ops, origin="script")` (see `examples/sketch_contour.py`).

```python
import math

import numpy as np

from phasor import Scene, cx
from phasor.sketch.propose import Sketch

s = Scene("draw")
s.add(cx.DomainColoring(id="plane", function="1/(z**2+1)", size=(8.0, 4.5)))
sk = Sketch(s.engine, t=0.0)
c = sk.to_pixels([[0.0, 1.0, 0.0]])[0]  # the pole at i, in pixels
a = np.linspace(0, 2 * math.pi, 60)
loop = np.stack([c[0] + 80 * np.cos(a), c[1] - 80 * np.sin(a)], axis=1)  # counterclockwise
(p,) = sk.propose([loop])
assert p.kind == "contour" and abs(p.data["exact"] - math.pi) < 1e-9
assert abs(p.data["integral"] - math.pi) < 0.01 and p.data["latex"] == r"\pi"
```

## 3D: orbit camera, surfaces, camera goals

`use_camera("orbit")` makes the scene 3D: a perspective camera on a sphere around
`target`, z up. Parameters: `azimuth` (0 looks from +x, 90 from +y), `elevation`,
`distance`, `fov` (vertical degrees), `drift` (slow sway in degrees; negative = the
theme's, 0 = still). Text and strokes keep their size on screen at any depth; dots and
surfaces get smaller with distance like real things. Surfaces are lit by the theme's
three-point rig and fade into the background with depth (theme `[light]`).

Leave `elevation`, `distance`, `target` (and `azimuth`) unset and give the camera
**goals**; the camera solver picks the rest at every moment, also while you orbit:
`keep_visible(a, b)`, `center(a)`, `thirds(a)`, `avoid_occlusion(a)`, `min_size(a, share)`.

```python
from phasor import Dot, Scene, Text, cx

s3d = Scene("riemann", passes=None)  # passes=None: analytic geometry only, fast
s3d.use_camera("orbit", azimuth=-60)
s3d.add(cx.RiemannSurface(id="sheets", kind="sqrt", radius=2.0, style={"material": "glass"}))
s3d.add(Dot(id="branch", pos=(0, 0, 0), r=0.08, style={"color": "highlight"}, tags=["key"]))
s3d.add(Text(id="branch.label", text="branch point"))
s3d.constrain("label(branch) near branch")  # placed by depth: readable from every side
s3d.constrain("camera keep_visible(sheets)")
s3d.marker("spin", 1.0)
s3d.marker("done", 6.0)
s3d.at("spin").orbit(360, until="done")  # azimuth is animated; the solver works around it

e3 = s3d.engine
assert e3.state_at(6.0).values["camera.azimuth"] == 300
assert set(e3.layout.camera) == {"camera.elevation", "camera.distance", "camera.target"}
sheet = e3.inspect(3.0, id="sheets")["objects"][0]
assert sheet["on_screen_fraction"] == 1.0 and sheet["semantic"]["sheets"] == 2
```

- `orbit(degrees, until=None)` turns the camera (theme `[motion.orbit]` time per turn, or
  until a marker); `orbit_reveal(*ids)` reveals while swinging a little.
- `focus_pull(id)` in 3D aims at the object and comes close enough to fill the frame.
- With camera goals, a camera parameter that is animated but never set is the solver's
  until its first motion: a `focus_pull` or `orbit` at a marker starts from wherever the
  solver had the camera (an orbit turns by its angle from there; a pull glides in), so the
  goals hold before it and nothing jumps when it starts.
- A label (`near` rule) lives at its anchor's depth, nudged toward the camera, and is moved
  only sideways; it avoids only what is **in front** of it, so it may sit on a surface
  behind it. Lint: `occluded_label` when something in front hides it; `visible_in_3d(a)`
  as a rule makes that an error.

### The event log and spatial queries

```python
log = e3.events(hz=2)  # 10 Hz by default: visibility 0.9/0.5/0.1, frame, region, occluders
from phasor.inspect.events import problems

assert problems(log) == []  # nothing important hidden or leaving the frame
assert e3.when_hidden("branch") == []  # spans where less than half of it is visible
free = e3.free_space_near("branch", 3.0)  # empty label-sized boxes, nearest first
assert free and set(free[0]) == {"slot", "box", "distance"}
```

Events carry `t`, the `marker` being spoken, the `id`, `severity` (`warning` = a viewer
would notice) and `fix_hints`.

## Dynamical systems: phase portraits and fixed points

`from phasor import dyn`. A planar system x' = f(x, y), y' = g(x, y) from formulas (read safely,
never run as Python; named numbers go in `constants="mu=1.5, k=2"`). `pos` is the window's
centre and `extent` its size; one unit of (x, y) is one world unit.

- `dyn.PhasePortrait(f=, g=, extent=, seeds=((x, y), ...) or n_seeds=, t_end=, progress=,
  backward=, flow=, nullclines=)`: trajectories from the seeds (RK4), their colour running
  along time; `progress` 0 -> 1 traces them (`s.at("m").sweep("flow.progress", 1.0,
  show=False)`); `flow` adds the small-arrow field and `nullclines` the curves where x' = 0
  and y' = 0. Semantics: the number of trajectories and fixed points.
- `dyn.FixedPoints(f=, g=, extent=, rays=, labels=, reveal=)`: every point where the system rests
  (Newton from a grid), classified from the Jacobian's eigenvalues: `saddle`, `stable node`,
  `unstable node`, `stable spiral`, `unstable spiral`, `centre`, `degenerate` or
  `non-hyperbolic`. Each is a beacon coloured by its type, its real eigen-directions are rays
  (cool where the motion comes in, warm where it goes out), with a label of the eigenvalues.
  Semantics list each point's position, type, stability, eigenvalues, trace and determinant.
  `reveal` 0 -> 1 shows them in order along x.

```python
from phasor import Scene, dyn

s = Scene("pendulum", passes=None)
s.add(dyn.PhasePortrait(id="flow", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6), n_seeds=8))
s.add(dyn.FixedPoints(id="rest", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6)))
kinds = [p["type"] for p in s.engine.inspect(0.0, id="rest")["objects"][0]["semantic"]["points"]]
assert kinds == ["saddle", "stable spiral", "saddle"]
```

Example: `examples/phase_portrait.py`. The prism looks make trajectories glow with the arrow
field and nullclines kept faint.

### Linear systems and their modes

`dyn.LinearSystem(matrix=((a, b), (c, d)), extent=, seeds= or n_seeds=, t_end=, progress=,
backward=, labels=)`: x' = A x with exact trajectories `e^(tA) x0` (`backward` runs each seed
back in time too, so the whole curve shows) and a label of the type and eigenvalues. Semantics:
the type (`saddle`, `stable node`, ...), eigenvalues, real eigenvectors, trace and determinant.
Its modes and the matrix exponential acting on a grid are drawn by the linalg objects beside it:
`linalg.EigenLines(matrix=A)` (the invariant lines) and `linalg.TransformGrid(flow=A,
flow_time=t)`, which shows `e^(tA)` on the grid instead of `matrix` (sweep it with
`s.at("m").sweep("grid.flow_time", 0.6, show=False)`; every cell's area is multiplied by
`e^(trace * t)`). `linalg_math.expm(m, t)` is the matrix exponential.

```python
from phasor import Scene, dyn, linalg

A = ((1, 2), (2, 1))
s = Scene("modes", passes=None)
s.add(dyn.LinearSystem(id="sys", matrix=A))
s.add(linalg.TransformGrid(id="grid", flow=A, flow_time=0.5))
sem = s.engine.inspect(0.0, id="sys")["objects"][0]["semantic"]
assert sem["type"] == "saddle"
det = s.engine.inspect(0.0, id="grid")["objects"][0]["semantic"]["determinant"]
assert abs(det - 2.718282) < 1e-5  # e^(trace * 0.5)
```

Example: `examples/linear_system.py`.
## Signals: live convolution

`from phasor import conv`. `conv.Convolution(f=, g=, progress=, pos=, extent=)` shows two signals
and the output they make. The top panel has `f`, the kernel `g` flipped and slid to the shift
tau (two hues), and under them their product f(s) g(tau - s), filled where they overlap. The
bottom panel shares the time axis: the area of that product is the output (f * g)(tau), traced
point by point with a beacon at the point being made. `progress` 0 -> 1 slides the kernel over
every shift at which the two can overlap (`s.at("m").sweep("c.progress", 1.0, show=False)`).

- A signal is a name (`box`, `triangle`, `decay`, `ramp`, `gauss`, `wave`) or a formula in `t`
  (read safely, never run as Python), zero outside its support. `f_support=(lo, hi)` and
  `g_support` set where it is not zero (`(0, 0)`: the shape's own; a formula without one gets
  -1..1).
- `ghost` draws the whole output faintly ahead of the traced part; `shade` (0..1) is how strongly
  the product is filled; `hue_cycles` runs the palette along the output.
- Semantics: `shift`, `shift_range`, `output` (the area at this shift), `peak` and `overlapping`.

```python
from phasor import Scene, conv

s = Scene("overlap", passes=None)
s.add(conv.Convolution(id="c", f="box", g="box", progress=0.5))  # halfway: shift 0
sem = s.engine.inspect(0.0, id="c")["objects"][0]["semantic"]
assert sem["shift_range"] == [-2.0, 2.0] and abs(sem["shift"]) < 1e-9
assert abs(sem["output"] - 2.0) < 0.01  # a box with itself peaks at its width
```

Example: `examples/convolution.py`. The prism looks make both signals and the output glow, the
product lit where they overlap. Numerics (a sum on a fine grid, checked against box * box = a
triangle, decay * decay = t e^-at and Gaussian widths adding) are in
`phasor/domains/convolution_math.py`.
- `dyn.StrangeAttractor(system="lorenz", sigma=10, rho=28, beta=8/3, seed=0, t_end=36,
  progress=1, pos=(0, 0, 0), size=6)` traces a single three-dimensional trajectory. Use
  `system="rossler"` or `"thomas"` for their classic parameter sets. The three named
  coefficients customize Lorenz. The object ID and seed determine the starting point.
  The solver uses a fixed step of 0.01 and `t_end` is in simulation time, from 0.01 to
  120, rounded to the nearest step. The complete path is centered on `pos` with longest span `size`. Animate
  `progress` from 0 to 1 and use an orbit camera to view its depth. The prism look colors
  the stroke by simulation time. The inspector reports the simulated time at the head.

```python
from phasor import Scene, dyn

s = Scene("lorenz", look="prism")
s.use_camera("orbit", azimuth=-90, elevation=12, distance=11)
s.add(dyn.StrangeAttractor(id="trajectory", progress=0))
s.marker("begin", 0.0)
s.marker("finish", 8.0)
s.at("begin").trace("trajectory")
s.at("begin").orbit(70, until="finish")
```

Example: `examples/strange_attractor.py`.

## Command line

| Command | Status | Purpose |
| --- | --- | --- |
| `phasor doctor [--json] [--require A\|B\|C]` | M0 | Prove CUDA, NVRTC, OpenGL 4.3, encoders and, without CUDA, the C++ compiler for CPU kernels (name, version, threads); print tier |
| `phasor inspect SCENE [--t T] [--id ID] [--detail brief\|standard\|full] [--allow-scripts]` | M1 | What is on screen at time t |
| `phasor lint SCENE [--fix] [--strict] [--json] [--step S] [--allow-scripts]` | M2 | Lint; exit 1 on errors (`--strict`: on warnings too). `--fix` applies suggested fixes and saves them |
| `phasor storyboard SCENE [-o PNG] [--t T]... [--marks] [--json] [--allow-scripts]` | M2 | Keyframe contact sheet plus what the inspector sees |
| `phasor render SCENE [-o OUT] [--preset preview\|final\|4k] [--fps F] [--start S] [--end E] [--frame T] [--workers N] [--threads T] [--allow-scripts]` | M3 | Video (`.mp4`; `.mov` is ProRes; a folder or `%05d.png` is a PNG sequence) or one PNG with `--frame`. `--workers`: render processes (default a quarter of the CPU threads, 1 to 4); the result is the same file as one process makes. `--threads`: CPU kernel threads per process without CUDA (default: the cores shared out, workers x threads <= cores) |
| `phasor kit [FOLDER]` | kit | Set up or update a folder where an AI writes scenes (default `Documents/phasor`): `CLAUDE.md`, the AI docs, the examples sorted by kind, the `/scene` command, `TASTE.md`, and the phasor tools in `.mcp.json`. Your `TASTE.md`, `WISHLIST.md` and `scenes/` are never replaced |
| `phasor bench [--width W] [--height H]` | M3 | ms per frame per kernel and peak VRAM; saved in `benchmarks/` |
| `phasor bench --frames SCENE [--count N] [--preset P] [--allow-scripts]` | speed | ms per final frame per stage (state, simulation, draw, gpu draw, readback, widen, blur sum, post, srgb, encode); saved in `benchmarks/` |
| `phasor mcp [--root DIR] [--allow-scripts] [--attach]` | M5, M10 | MCP server; `--attach` edits the scene open in a running `phasor gui` |
| `phasor gui [SCENE] [--root DIR] [--no-ai] [--install-shortcut]` | M10 | The desktop app (needs `uv sync --extra gui`): a start screen, or the editor on SCENE; also `phasor-gui`; a Manim script opens too |
| `phasor web SCENE [-o PAGE.html] [--steps N] [--allow-scripts]` | M12 | Bake a self-contained interactive web page from the scene's sliders |
| `phasor manim SCRIPT [--scene NAME] [-o OUT] [--code]` | M13 | Import a Manim Community scene as a phasor scene |
| `phasor sketch SCENE STROKES.json [--t T] [--accept [K ...]] [--text K=TEXT]` | M11 | Turn drawn strokes into proposals; `--accept` applies them |

`SCENE` is a `.phasor.json` document or a `.py` script that defines exactly one `Scene`.
`--allow-scripts` (off by default) runs the Python modules a saved document names in
`meta.modules` (its own object types and updaters); without it those objects are
`missing_type` placeholder boxes. A `.py` scene is code you ran, so it needs no flag.

## The desktop app (`phasor-gui`)

`phasor-gui` (or `phasor gui`) with no scene opens a **start screen**: new scene (2D, 3D,
a complex function), open a file, recent scenes, and the examples (double-click copies one
to `Documents/phasor/examples` and opens the copy). Each scene gets its own window with a
menu bar: **File** (New, Open, Open Example, Open Recent, Save, Save As, Export, Close,
Quit), **Edit** (Undo, Redo), **View** (Play, back to the start), **Settings** (math
typesetting: built-in, the default, or LaTeX; add phasor to the applications menu),
**Help** (these docs, a hardware check, about). The title marks unsaved edits with `•`,
and closing such a window asks first. **Export** makes a picture, a video or a web page
with sliders, with a progress bar, in a separate process (it can be cancelled and the
window keeps working). The packaged app is built with PyInstaller (`packaging/README.md`).

The logic is plain Python in `phasor.gui.app` (no Qt): settings, templates, examples,
exports and the shortcut.

```python
import tempfile
from pathlib import Path

from phasor.gui.app import EXPORTS, TEMPLATES, ExportJob, examples

assert {"2D scene", "3D scene"} <= set(TEMPLATES) and set(EXPORTS) == {"picture", "video", "web"}
assert any(e["name"] == "julia explorer" for e in examples())
job = ExportJob(None, "web", Path(tempfile.mkdtemp()) / "page.html", scene_dir=".")
assert job.command("s.phasor.json")[-5:] == ["s.phasor.json", "-o", str(job.out), "--steps", "5"]
```

## The editor and the AI on one scene (`phasor gui`)

`phasor gui scene.py` opens the editor: the viewport (click selects; drag the selected
object to move it), the timeline (markers, one lane per animated parameter, click or drag to
scrub, Space plays), and tabs for the **inspector** (every parameter as a field: type JSON
or text and press Enter; style roles as menus; what the inspector reads), **lint** (click an
issue to jump to its object and time) and **looks** (switch the whole scene to a look; show
the variants of a beat and pick one: recorded in `style/choices.jsonl` as the person's
choice). Ctrl+Z / Ctrl+Shift+Z / Ctrl+S undo, redo, save. Every edit is one
`Engine.apply`, so a script scene keeps them as overrides. Also: drag empty space to turn the 3D
camera (shift-drag or a 2D scene: slide it), the wheel zooms; the **Sliders** tab previews the
scene's controls; the **Code** tab shows the scene as Python (the script itself, or one
generated from the document) with highlighting: edit and Run (Ctrl+Enter) to rebuild the scene;
the timeline shows narration with its waveform, and lets you drag markers and animation spans
(or their ends) to retime them; the **Who changed what** dock lists your edits and an attached
AI's; hovering shows values under the pointer. `phasor-gui` starts the editor as a program.

While it is open, `phasor mcp --attach --root <same folder>` serves the editor's own live
scene to an AI: its edits show up in the window, the window's edits are what it inspects.
The editor writes `<root>/.phasor/gui.json` (host, port, a token; readable only by you)
and removes it on close; `--no-ai` keeps the AI out. The logic is `phasor.gui.controller.Editor`
(no Qt in it), and the shared tools are `phasor.workspace`:

```python
import tempfile

from phasor.gui.controller import Editor
from phasor.workspace import Workspace
from phasor.workspace.remote import RemoteServer, RemoteWorkspace

root = tempfile.mkdtemp()
ws = Workspace(root)
ws.scene_new("together")
ed = Editor(ws)
server = RemoteServer(ws, ed.lock).start()
server.write_handoff()
ai = RemoteWorkspace.from_handoff(root)  # what `phasor mcp --attach` uses
ai.scene_edit({"op": "add_object", "object": {"id": "p", "type": "dot"}})
assert ed.changed()  # the editor sees the AI's edit
assert ed.set_param("p", "r", "0.3")["ok"]  # and the AI sees the person's
assert ai.inspect(0.0, id="p")["objects"][0]["id"] == "p"
ai.close()
server.stop()
```

## Hardware check

```python
from phasor.render.doctor import run_doctor

report = run_doctor()  # never raises; each probe fails on its own
assert report.tier in {"A", "B", "C"}
```

Tiers: **A** CUDA kernels and GL 4.3. **B** GL 4.3 only. **C** no usable GL: the inspector
runs in analytic mode, stills are unavailable.

## Filtering: poles and zeros, response, signal and spectrum

`from phasor import filt`. One digital filter, shown four ways. Every object takes the same
filter parameters: `kind` (`lowpass`, `highpass`, `bandpass`, `bandstop`: Butterworth, with the
half-power frequency `cutoff` in Hz and, for the bands, `cutoff2`; `notch`: zeros on the unit
circle at `cutoff`, poles at `radius` times them; `custom`: `zeros`, `poles` as (re, im) pairs
in conjugate pairs, and `gain`), `order`, and the sample rate `fs`. A cutoff outside 0 to `fs / 2`
is refused when the object is added.

- `filt.transfer_plane(id, pos=, scale=, size=, **filter)`: the `cx.DomainColoring` of H(z),
  placed so the z plane sits at `pos` with `scale` world units per unit of z. In the prism look
  its zeros and poles glow with the phase wrapped round them.
- `filt.PoleZero(pos=, scale=, size=, probe=, **filter)`: the unit circle, zeros (rings) and poles
  (crosses) with their multiplicity. Give it the plane's `pos`, `scale`, `size` and a higher
  `z_index`. `probe` (0 to 1, DC to half the sample rate) puts a bright point on the circle at
  that frequency; sweep it together with the response's `reveal`. Semantics list zeros, poles
  (with multiplicity), `stable` and the largest pole radius.
- `filt.FrequencyResponse(plot="gain" | "phase", scale="db" | "linear", reveal=, pos=,
  extent=, **filter)`: the gain (dB with a floor, or linear) or the unwrapped phase from DC to
  half the sample rate, the cutoffs marked, traced as `reveal` goes 0 -> 1. Semantics give the
  gain and phase at each cutoff (-3.01 dB at a Butterworth cutoff; -n pi / 4 radians for an
  order-n low-pass).
- `filt.FilteredSignal(frequencies=, amplitudes=, duration=, progress=, **filter)`: a sum of
  sinusoids before (faint) and after (bright) the filter, run from rest so the start-up shows.
- `filt.FilteredSpectrum(frequencies=, amplitudes=, window=, samples=, fade=, envelope=,
  **filter)`: the spectrum of the same tones through a window (the one `fourier.Spectrum` draws),
  what the filter keeps solid and what it removes on top in a second color; `fade` 0 -> 1 fades the
  removed part out. Semantics: `removed_energy_share` and each tone's gain.

```python
from phasor import Scene, filt

low = dict(kind="lowpass", order=4, cutoff=1000.0, fs=8000.0)
s = Scene("low_pass", passes=None)
s.add(filt.transfer_plane("plane", **low))
s.add(filt.PoleZero(id="poles", z_index=1, **low))
s.add(filt.FrequencyResponse(id="gain", pos=(0, -4), **low))
s.add(filt.FilteredSpectrum(id="spec", frequencies=(300.0, 2800.0), amplitudes=(1.0, 1.0),
                            pos=(0, 4), **low))  # fmt: skip
e = s.engine
sem = {o["id"]: o["semantic"] for o in e.inspect(0.0)["objects"]}
assert sem["poles"]["zeros"] == [[-1.0, 0.0, 4]] and len(sem["poles"]["poles"]) == 4
assert abs(sem["gain"]["gain_db_at_cutoff"][0] + 3.0103) < 1e-3
assert sem["spec"]["tones"][0]["kept"] > 0.99 > 0.01 > sem["spec"]["tones"][1]["kept"]
```

Example: `examples/filtering.py` (the poles, a probe round the circle with the gain and phase
traced, a signal going in and out, the removed band fading). Numerics are in
`phasor/domains/filters_math.py` (Butterworth by the bilinear transform with the cutoff
pre-warped; checked against `scipy.signal`).

**Spectrogram of any signal.** `fourier.Spectrogram` shows the short-time spectrum of one of
three sources: a `.wav` file next to the scene (`wav="voice.wav"`, mixed down to mono; its length
and sample rate are the file's own; the AI may name only files in its folder), a formula in `t`
seconds (`signal="sin(tau * 440 * t)"` over `duration` at `fs`; at most 400000 samples), or,
with neither, a chirp from `f0` to `f1`. `nwin`, `hop` (0: `nwin / 4`) and `window`
(`rect`, `hann`, `hamming`, `blackman`) shape the frames; `fmax` crops the frequency axis
(0: half the sample rate). `playhead` (0..1 of the duration, below 0 for none) draws a glowing
line, in the `playhead` style role; track it or give it `expr("t / 6")` to sweep. The semantics
report `source`, `duration`, `max_frequency`, `playhead_time` and `dominant_frequency`, the
strongest frequency under the playhead. A missing file is the `missing_file` issue.

```python
from phasor import Scene, expr
from phasor.domains.fourier import Spectrogram

s = Scene("sweep", look="prism", passes=None)
s.add(
    Spectrogram(
        id="sg",
        signal="sin(tau * (200 * t + 400 * t * t))",
        duration=1.0,
        fs=4000.0,
        playhead=expr("t / 4"),
    )
)
s.marker("end", 4.0)
sem = s.engine.inspect(2.0, id="sg")["objects"][0]["semantic"]
assert sem["source"] == "formula" and abs(sem["dominant_frequency"] - 600) < 50
# from a file next to the scene: Spectrogram(id="sg", wav="voice.wav", fmax=4000)
```

**Fractal cloud mode (the Buddhabrot).** `cx.Fractal(kind="buddhabrot")` draws not the escape
time of each pixel but the orbits: `count` points of the orbits z -> z^2 + c (from 0) of
random c that escape within `max_iter` steps (at most 5000), splatted as light by the
`splat` kernel exactly as `cloud.PointCloud` draws. Colour is the `colormap` at
log(escape time) / log(max_iter), turned by `hue`; `exposure`, `point_size` and `white` as in
`PointCloud`; the `glow` role makes it shine (the `prism` looks: soft glow, a white-hot
core). `center` and `zoom` set the view as for the other kinds. The orbits come from
`rng(id, seed)` and are cached per (id, seed, max_iter, count): tween the camera, `hue` or
`exposure`, not `count` or `max_iter`. A few million points take a second or two to make.
The semantics report `points` and `in_rectangle`. See `examples/buddhabrot.py`.

```python
from phasor import Scene, cx

s = Scene("nebula", look="prism", passes=None)
s.add(
    cx.Fractal(
        id="b",
        kind="buddhabrot",
        pos=(0, 0, 0),
        size=(8, 4.5),
        center="-0.45 0",
        zoom=0.2,
        max_iter=200,
        count=50_000,
        style={"colormap": "spectrum"},
    )
)
sem = s.engine.inspect(0.0, id="b")["objects"][0]["semantic"]
assert sem["set"] == "buddhabrot" and sem["points"] == 50_000 and sem["in_rectangle"] > 40_000
```

**Bifurcation diagram** (`cloud.PointCloud(source="bifurcation")`). The
orbit of `x0` under x -> `map`(r, x) (`logistic` r x (1 - x), `sine` r sin(pi x), or any
formula in `r` and `x`, read by the same whitelist parser as functions elsewhere) runs
`n_iter` steps for each r, and its last `n_keep` iterates are splatted as points of light:
cycles burn as bright branches, chaos smears into a haze. Colour is the column's Lyapunov
exponent (cycles one hue, chaos another). The window `r_range` x `x_range` fills the `size`
rectangle. The columns of r are computed for the view, on a lattice fixed in r, about a pixel
apart at any zoom (`count` is not used); new columns fade in between the old ones as a zoom
goes on, and an old column's points never change, so a zoom does not shimmer. Zoomed in on x,
a column shows up to four times `n_keep` iterates. Two ways to zoom: the camera (the same
factor in r and x), or `focus` 0 -> 1, which zooms the window geometrically to
`r_focus` x `x_focus` (different factors in r and x, e.g. into the window of period three).
At `focus` 0 the focus may change unseen, so one object can zoom into several places in turn.
`sweep` 0 -> 1 draws the diagram from the left. Use `preset="prism_bifurcation"` (more light
per point than a cloud, a little white). The maths: `cloud_math.bifurcation(map, r0, r1, n_r,
n_iter, n_keep)` returns the points r + i x and their colour values; `cloud_math.orbits`,
`lyapunov` and `zoom_window` are the parts. Example: `examples/bifurcation.py`.

```python
import math

from phasor import Scene, cloud
from phasor.domains import cloud_math as clm

pts, value = clm.bifurcation("logistic", 3.2, 3.2, 1, 2000, 8)  # one r: a 2-cycle
assert len(set(round(x, 6) for x in pts.imag)) == 2
pts, value = clm.bifurcation("r*x*(1 - x)", 1 + math.sqrt(8) + 0.004, 4.0, 1, 2000, 9)
assert len(set(round(x, 6) for x in pts.imag)) == 3  # the window of period three

s = Scene("logistic", look="prism", passes=None)
s.add(cloud.PointCloud(id="diagram", preset="prism_bifurcation", r_range=(2.8, 4.0),
                       x_range=(0.0, 1.0), size=(7.6, 4.1), sweep=0.0,
                       r_focus=(3.822, 3.862)))  # fmt: skip
s.at(0.3).to("diagram.sweep", 1.0, duration=4.0)  # r runs in from the left
s.at(5.0).to("diagram.focus", 1.0, duration=5.0)  # then the window zooms in on r
sem = s.engine.inspect(10.0, id="diagram")["objects"][0]["semantic"]
assert sem["source"] == "bifurcation" and sem["window"]["r"] == [3.822, 3.862]
assert sem["points"] > 100_000
```
