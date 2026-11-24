# phasor patterns

Idioms for common tasks, one canonical way each.

## Label something

Name the label `<owner>.label` and add a `near` constraint. Never type label coordinates.

```python
from phasor import Dot, Scene, Text, expr

s = Scene("label_follows")
s.add(Dot(id="pole", pos=(-2, 0), style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="pole.label", text="pole"))
s.constrain("label(pole) near pole")
s.at(0).to("pole.pos", (2, 0), duration=2)

mid = s.engine.state_at(1.0)
assert mid["pole.label"].params.pos[0] == 0.0  # it follows the pole
assert s.engine.lint() == []
```

Use `above`/`below`/`left_of`/`right_of` only when the side matters; `near` lets the placer
dodge neighbors.

## Tie any parameter to another

An expression makes one parameter follow others. (For labels, prefer the constraint above.)

```python
s.add(Dot(id="shadow", pos=expr("pole.pos + vec(0, -1, 0)"), r=0.05, style={"color": "muted"}))
assert s.engine.state_at(1.0)["shadow"].params.pos == (0.0, -1.0, 0.0)
```

## The self-review loop

After every edit: check the result, then look at what is on screen at each marker.

```python
engine = s.engine
result = engine.apply({"op": "set_param", "path": "pole.r", "value": 0.2})
assert result.ok, result.errors

for name, t in sorted(engine.document.timeline.markers.items()) or [("start", 0.0)]:
    for line in engine.inspect(t, detail="brief")["objects"]:
        print(name, line)

for issue in engine.lint():  # then let the linter look at every marker and sample
    print(issue.severity, issue, "->", issue.fix_hint)
report = engine.autofix()  # applies the fixes the linter is sure about
assert report["remaining"] == []
```

Errors must be fixed. Warnings need a reason to stay. Look at a storyboard before rendering.

## Screen-space titles

```python
s.add(Text(id="title", text="Residues", pos=(8 / 9, 0.1), layer="overlay", style={"size": "title"}))
assert s.engine.where("title", 0)["region"] == "upper-center"
```

Overlay coordinates are frame heights from the top-left: on a 16:9 frame x runs 0 to 16/9
and y runs 0 to 1. The overlay ignores the camera and always draws on top.

## Check what this machine can do before rendering

```python
from phasor.render.doctor import run_doctor

if run_doctor().tier == "C":
    print("No OpenGL here: inspect works in analytic mode, stills do not.")
```

## A glowing contour on a domain-colored plane

The plane is the backdrop; the contour is a `Line` with `glow`; its label is placed by
constraint and lands on the plane, not away from it.

```python
import numpy as np

from phasor import Line, Scene, Text, cx

s = Scene("contour")
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
ring = [(0.5 * np.cos(a), 1 + 0.5 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 129)]
s.add(
    Line(id="loop", points=ring, style={"color": "highlight", "stroke": "bold", "glow": "strong"})
)
s.add(Text(id="loop.label", text="Res = -i/2"))
s.constrain("label(loop) near loop")
assert s.engine.lint() == []
```

Write math in labels with ASCII (`z^2`, `pi`) until LaTeX arrives: the bundled font has no
superscripts or Greek, and lint reports `missing_glyph` if you use them.

## A short driven by narration

Write the words first, mark the moments, then attach verbs to the marks. Pace last.

```python
from phasor import Dot, Scene, Text

s = Scene("short", passes=None)
s.add(Dot(id="pole", pos=(0, 1), r=0.1, style={"color": "highlight"}))
s.add(Text(id="pole.label", text="i"))
s.constrain("label(pole) near pole")
s.add(Text(id="note", text="only this pole counts", pos=(0, -1)))
s.narrate(
    "There is a pole at i [[pole]].\nOnly this pole counts [[focus]].",
    voice="estimate",
)
s.at("pole").bloom_in("pole").then().reveal("pole.label")
s.at("focus").reveal("note")
s.pace()
assert s.engine.lint(strict=True) == []
```

Check what is on screen at each marker (the self-review loop above) before rendering.

## Orbit a 3D camera around a surface

Say what must stay in view, leave the camera's elevation, distance and target unset, and
orbit by verb. The solver frames it at every angle; labels with `near` stay readable.

```python
from phasor import Dot, Scene, Text, cx

orb = Scene("orbit_pattern", passes=None)
orb.use_camera("orbit", azimuth=-60)
orb.add(cx.RiemannSurface(id="sheets", radius=2.0, style={"material": "glass"}))
orb.add(Dot(id="branch", pos=(0, 0, 0), r=0.08, tags=["key"]))
orb.add(Text(id="branch.label", text="branch point"))
orb.constrain("label(branch) near branch")
orb.constrain("camera keep_visible(sheets)")
orb.marker("turn", 0.5)
orb.marker("still", 5.0)
orb.at("turn").orbit(360, until="still")
assert [i for i in orb.engine.lint() if i.code == "unsolved"] == []
```

- Folded or nested surfaces (the two sheets of sqrt) want `material: glass`: solid, one
  part always hides another from some side, and `events` will say so.
- Check a 3D scene with `engine.events()`: `problems(...)` lists every moment a label or
  key object is hidden or leaves the frame, with the marker being spoken and fixes.

## Make it beautiful: the taste loop

After the scene is correct (`review` clean), run the beauty pass: `lint(aesthetic=True)` or
`critique`, look at the storyboard against `STYLE.md` (and `storyboard(compare=True)` once
the project has a `gallery/`), `polish` for what is certain, then decide the rest yourself.
Unsure between looks? `variants` on a marker and `choose`.

```python
from phasor import Dot, Scene

draft = Scene("taste", passes=None, look="calm")
draft.set("camera.height", 6)
draft.add(Dot(id="hero", pos=(0, 0), r=1.0, preset="beacon", tags=["key"]))
draft.marker("beat", 1.0)
assert draft.engine.critique()["score"] == 0.0  # one hero, clear margin, filled enough
```

- Start from a look (`calm`, `glow`) and presets, not from bare parameters.
- Keep one `highlight` per beat; step everything else back with roles.
- Rainbows where hue carries meaning (domain coloring); elsewhere vary the palette.
- Never hand-tune glow or stroke numbers: roles and presets are the taste knobs.
