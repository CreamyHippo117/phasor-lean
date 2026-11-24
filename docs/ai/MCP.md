# phasor over MCP (for AI agents)

You are connected to `phasor mcp`. You build a scene by sending **patches**, and you check it
with **review**. You never write coordinates for labels, raw colors or pixel sizes: name
objects by meaning, use theme roles, add constraints.

Every ```json patch block (a patch) and ```json calls block (one tool call per line) in
this file runs in order, against one scene, in `tests/test_docs.py`.

## The loop

1. `object_types` (what exists, parameter names and defaults, style roles) and
   `docs_search "<what you want>"`.
2. `scene_new`, then `scene_edit` with patches.
3. `narrate` a script with `[[markers]]`, then `animate` verbs anchored to those markers.
4. `review`: lint + what is on screen at every marker + a storyboard picture.
   Fix each item in `todo` (`scene_edit`, or `lint` with `fix: true` for the certain fixes)
   and `review` again. Stop when `clean` is true or after 5 rounds; say what is left.
5. `scene_save`, then `render` (`preview` is fast).

A failed patch changes nothing and tells you why (`errors[].message`, often with
`fix_hint`). `undo` reverts the last edit; `checkpoint` / `restore` / `diff` let you try
a variant and compare.

## Tools

| Tool | Use it to |
| --- | --- |
| `scene_new`, `scene_open`, `scene_save` | start, load or save `name.phasor.json` (inside the workspace folder only) |
| `scene_palette` | the colors run along strokes: `spectrum` (the prism rainbow), `moonlight`, `gold`, `glacier`, `borealis`, `rose`, or none; in the prism looks every glowing curve takes them |
| `scene_edit` | apply a patch (below); returns `ok`, `errors`, and the lint issues it added or fixed |
| `animate` | a choreography verb at a marker or time: `reveal`, `exit`, `trace`, `bloom_in`, `dim_others`, `undim`, `focus_pull`, `morph`, `settle`, `write`, `travel`, `flash`, `wiggle`, `circumscribe` (`value` "circle" or "box"), `transform` and `become` (`targets` [from, to]), `sweep` (`path`, `value`), and in 3D `orbit` (`value` degrees, `until` a marker) and `orbit_reveal` |
| `narrate` | set the script and voice; markers appear in `markers` |
| `inspect` | what is on screen at `t`: `brief` (a line each), `standard`, `full` |
| `lint` | visual problems; `fix: true` applies the certain fixes |
| `review` | the whole self-review round in one call |
| `storyboard` | keyframe picture with each object's box and id drawn on it |
| `hit_test`, `resolve` | what is at a pixel; which object ids "the label of the pole" means |
| `events`, `when_hidden`, `free_space_near` | 3D: when things get hidden or leave the frame (with fixes), spans an object is hidden, empty places next to an object |
| `critique`, `polish` | the art director: a taste score, aesthetic issues with fixes and the rubric; apply the fixes (3 rounds at most) |
| `variants`, `choose` | alternative looks of one beat side by side (as is, calmer, bolder); pick one, it is recorded in `style/choices.jsonl` and applied |
| `diff`, `checkpoint`, `restore`, `undo`, `redo` | compare and branch |
| `render` | a video (`draft` is the quickest and roughest, then `preview`, `final`, `4k`) or one PNG with `frame` |
| `docs_search` | find the API, a pattern or an example |

## Patch operations

`scene_edit` takes one op, a list of ops, or `{"ops": [...]}`. Ops apply together or not at
all. Type names are lower case: `dot`, `line`, `text`, `domain_coloring`, `fractal`.

```json patch
[
  {"op": "set_param", "path": "camera.height", "value": 6},
  {"op": "add_object", "object": {"id": "pole_i", "type": "dot",
    "params": {"pos": [0, 1], "r": 0.15}, "style": {"color": "highlight"}, "tags": ["key"]}},
  {"op": "add_object", "object": {"id": "pole_i.label", "type": "text",
    "params": {"text": "pole at i"}}},
  {"op": "add_object", "object": {"id": "loop", "type": "line",
    "params": {"points": [[-2, -1], [2, -1], [2, 2], [-2, 2], [-2, -1]]},
    "style": {"color": "secondary"}}}
]
```

- A child id is `<owner>.<name>` and the owner must exist (`pole_i.label`).
- `pos` is world space: x = Re, y = Im, z up; `[x, y]` means z = 0.
- `style` takes role names from `object_types().roles`; a raw color is rejected.
- Never reuse the id of a removed object (`id_retired`); pick a new meaningful name.

Constraints say where things go; the layout solver does the arithmetic:

```json patch
[
  {"op": "add_constraint", "rule": "label(pole_i) near pole_i"},
  {"op": "add_constraint", "rule": "on_screen(loop, 0.05)", "priority": "medium"}
]
```

Rules: `a above b gap N`, `label(x) near x`, `a outside b`, `a tangent_to b`,
`a attach_to b at top` (a named spot: top, bottom, left, right, center, ... or a feature such as
an arrow's `tip`), `align_x(a, b)`, `align_y(a, b)`, `inside(a, b)`, `on_screen(a, margin)`, `no_overlap(a, b)`, `min_size_px(a, N)`,
`readable(a)`; join with `and`; priorities `required`, `strong`, `medium`, `weak`.

Sliders (`add_control` / `remove_control`) make a number something a viewer can move in the
editor, a notebook or a web page (`phasor web`); a vector parameter needs its `component`:

```json patch
[
  {"op": "add_object", "object": {"id": "set", "type": "fractal",
    "params": {"kind": "julia", "c": [-0.8, 0.156]}}},
  {"op": "add_control", "control": {"path": "set.c", "component": 0, "min": -1.2, "max": 1.2,
    "step": 0.05, "label": "Re c"}}
]
```

Change things later with `set_param`, `set_style`, `set_tags`, `remove_object`,
`remove_constraint`:

```json patch
[
  {"op": "set_param", "path": "pole_i.r", "value": 0.2},
  {"op": "set_style", "id": "loop", "key": "stroke", "role": "bold"},
  {"op": "set_tags", "id": "loop", "tags": ["contour"]}
]
```

Domain objects are added the same way (`object_types` lists their parameters and presets):
`equation` (typeset math in colored parts); `axes`, `parametric_curve`, `parametric_surface`, `vector_field_3d`, `flow_lines` (space); `epicycles`, `partial_sum`, `spectrum`, `spectrogram` (fourier); `transform_grid`, `arrow`,
`eigen_lines`, `matrix` (linear algebra); `room`, `source`, `listener`, `wavefront`,
`image_sources`, `impulse_response`, `pressure_slice`, `room3d`, `sound_rays`, `wavefront3d` (acoustics); `pde_profile`, `pde_field`,
`pde_surface`, `pde_modes`, `pde_slice`; `polygon`, `cayley_table`, `clock_diagram`,
`cayley_graph`, `homomorphism`, `subgroup_lattice`, `galois_diagram`, `permutation` (algebra);
`graph`, `riemann_sum`, `taylor`, `vector_field`, `limit`, `loop_integral` (calculus). A parameter may be an
expression, which is how a `matrix` object reads a grid's matrix: `"matrix": {"expr": "grid.matrix"}`.

```json patch
[
  {"op": "add_object", "object": {"id": "grid", "type": "transform_grid",
    "params": {"matrix": [[2, 1], [1, 2]], "pos": [5, -1, 0], "extent": 2}}},
  {"op": "add_object", "object": {"id": "entries", "type": "matrix",
    "params": {"matrix": {"expr": "grid.matrix"}, "pos": [5, 3, 0]}}},
  {"op": "add_object", "object": {"id": "tone", "type": "spectrum",
    "params": {"frequencies": [100.5], "amplitudes": [1.0], "window": "hann", "pos": [-7, -4, 0]}}}
]
```

```json patch
[
  {"op": "add_object", "object": {"id": "rod", "type": "pde_profile",
    "params": {"kind": "heat", "length": 2.0, "diffusivity": 0.5, "modes": [1.0, 0.4],
               "time": {"expr": "t / 10"}, "pos": [-8, 1, 0]}}},
  {"op": "add_object", "object": {"id": "square", "type": "polygon",
    "params": {"n": 4, "pos": [2, 2, 0]}}},
  {"op": "add_object", "object": {"id": "area", "type": "riemann_sum",
    "params": {"function": "x**2", "a": 0, "b": 3, "n": 8, "rule": "mid", "pos": [-2, -4, 0]}}}
]
```

Backgrounds that compute a picture per pixel (`domain_coloring`, `fractal`) are objects too;
labels may sit on them.

```json patch
{"op": "add_object", "object": {"id": "plane", "type": "domain_coloring",
  "params": {"function": "1/(z**2 + 1)", "contour": 0.4}, "style": {"colormap": "phase"}}}
```

## 3D

Switch the camera to `orbit` (perspective, z up) and add surfaces. Leave the camera's
`elevation`, `distance` and `target` unset and give **goals** instead: the camera solver
chooses them at every moment, also while the camera orbits.

```json patch
[
  {"op": "set_camera", "kind": "orbit", "params": {"azimuth": -60}},
  {"op": "add_object", "object": {"id": "sheet", "type": "riemann_surface",
    "params": {"kind": "sqrt", "radius": 2.0}, "style": {"material": "glass"}}},
  {"op": "add_constraint", "rule": "camera keep_visible(sheet)"},
  {"op": "add_constraint", "rule": "camera avoid_occlusion(pole_i)", "priority": "medium"}
]
```

Goals: `keep_visible(a, b)`, `center(a)`, `thirds(a)`, `avoid_occlusion(a)`,
`min_size(a, 0.3)` (share of the frame height), each written after `camera`. Materials
(`surface` roles): `solid`, `glass` (see-through, does not hide things), `wire`,
`emissive`. In 3D a label is hidden only by what is in front of it: lint reports
`occluded_label`, and `events` lists every moment something important is hidden or
leaves the frame. A clean 3D scene has no `problems` in `events`.

## Narration and animation

Write the script first, mark the words things should happen on, then anchor every motion
to a marker with a verb. Do not animate by raw seconds in a narrated scene (lint
`raw_motion` says so).

```json calls
{"tool": "narrate", "script": "Here are the poles. [[poles]] Draw a loop around the first. [[loop]] The result is two pi.", "voice": "estimate"}
{"tool": "animate", "verb": "reveal", "at": "poles", "targets": ["pole_i"]}
{"tool": "animate", "verb": "trace", "at": "loop", "targets": ["loop"]}
```

`animate` returns `start` and `end`: pass `offset: end - start` to start the next step when
the first ends. A verb moves nothing for an object that has no track yet, so `reveal` an
object you want to appear (it is hidden until then).

## What comes back from `review`

`clean`, `todo` (one line per issue: code, ids, what is wrong), `issues` (the same with
`fix` ops you can send to `scene_edit`), `frames` (a line per object at each marker) and
`storyboard` (an image path; the image also arrives inline). Read the picture, not only the
lint: lint cannot tell you a layout is ugly.

## The beauty pass

`review` and `lint` take `aesthetic: true` for the taste rules on top of the correctness
ones: `focal_point` (exactly one `highlight` per beat), `color_restraint` (at most three
saturated hues; a phase-coloring plane is exempt, hue is its meaning), `glow_clip`,
`density`, `motion_rhythm`, `margin`, `stroke_mix`. Then:

1. `critique` returns `score` (0 = nothing to criticise), issues tied to ids and markers,
   `fix` ops (certain) and `suggested` ops (judgement), and the rubric text (`STYLE.md`).
2. Look at the storyboard (`storyboard` with `compare: true` puts each frame beside the three
   nearest frames in the project's `gallery/`) and judge by the rubric. Rules cannot see
   everything.
3. `polish` applies what the critic is sure of, three rounds at most; whatever is left is
   yours to judge (an empty frame, two motions that collide).
4. Unsure which look? `variants` on a marker, then `choose` one.

Start scenes from a look: `scene_new` with `look: "calm"` (soft edges, gentle life), `"glow"`
(lively bloom) or `"chalk"` (light paper page, still, crisp). `object_types` lists each
type's `presets`; in a patch, `{"preset": "soft_phase"}` inside an `add_object`. Explicit
values win, then the preset, then the look. The default theme is dark.

## Working beside a person (`--attach`)

If the person has `phasor gui` open, run `phasor mcp --attach --root <their folder>`: every
tool then works on the scene in their window, and they watch your edits land. Their edits
are parameter and style changes: `inspect(detail="full")` shows every object's `params` as
the document sets them, and `diff` compares with the saved file. Undo is one shared history, so
`undo` takes back the most recent edit, whoever made it: check `diff` before undoing. You
cannot open a `.py` scene through the editor (only the person can); a Manim script
(`from manim import ...`) opens as a plain scene, with `notes` on what could not be carried
over, on a server started with `--allow-scripts`.

## Traps this loop has already hit

- **Overlay positions are in frame heights**, not pixels or world units: `[0, 0]` is the
  top-left corner and the frame is about 1.78 wide, so a centered title is `[0.8889, 0.08]`.
  An overlay added with no `pos` sits at the corner and lint reports `offscreen`.
- **Lint clean does not mean it looks good.** The first draft of the residue scene was lint
  clean with an off-center title, a hand-placed answer and a 24-sided "circle". Look at the
  storyboard image every round.
- **Do not place text by hand.** Hand-picked coordinates are how a result ended up floating in
  the middle of a colored background. Use an overlay position or a constraint.
- **A removed id is gone for good.** Re-adding `result` fails with `id_retired`; use `answer`.
- **A second `reveal` of the same object is refused** (`bad_segment`): its fade already
  exists. Change the first one's marker instead, or `undo`.
- **`.py` scenes are code.** The server refuses them unless started with `--allow-scripts`.
- **A scene may bring its own types** (ADR 0005: `meta.modules`, a scene script or
  `*.types.py` next to it). `object_types` marks those `"scene_local": true`. On a server
  without `--allow-scripts` their code does not run: lint `missing_type`, the object draws as
  a placeholder box, and its parameters and overrides stay as they are. Edit its parameters
  as usual; do not remove it to make the issue go away. A `value` object is a number with
  nothing to draw: animate its `value` and read it with `{"expr": "speed.value"}`.
