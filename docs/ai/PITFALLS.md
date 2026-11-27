# phasor pitfalls

Known traps and how phasor reports them.

## A wrong edit changes nothing, and says why

`Engine.apply` is all-or-nothing. Always check `result.ok`; on failure read
`result.errors[i].code` and `fix_hint`. Common codes: `unknown_id`, `unknown_param`,
`bad_param`, `bad_role` (you passed a raw color or size instead of a role), `bad_expr`,
`unknown_ref`, `cycle`, `track_on_expr`, `bad_segment` (overlapping or backwards),
`unknown_marker`, `scripted_scene`.

## A constraint was accepted but does nothing visible

`apply` only rejects rules that can never work (`unknown_id`, `not_movable`,
`conflicting_constraints`, `bad_constraint`). A rule that cannot hold at some moment is
accepted and reported by `engine.lint()` as `unsolved` with the times. Always lint after
adding constraints.

## The label did not move when I set its `pos`

A constrained object's `pos` is only a starting hint; layout decides where it is. Remove the
constraint (`remove_constraint`) or change the rule. An object whose `pos` is an expression
cannot be constrained (`not_movable`).

## Lint looks at samples, not every frame

Lint checks markers, transition ends and every 0.5 s. A collision that lasts two frames
between samples is missed. Put a marker where it matters, or pass a smaller `step`.

## A `near` label makes a long 3D scene slow to edit

The label placer draws one occlusion buffer (a depth buffer in 3D) at every 10 Hz key time
while something it places is shown. A label left visible for the whole of a 17-minute scene
is 10 000 depth renders, and every verb the script runs after it reads the scene and bakes
to where the verb starts. Show the label only while it is needed (`reveal` ... `exit`):
hidden, it costs nothing. Write the rule after the verbs that hide it (a label is shown
until its `reveal` says otherwise, so a rule written first makes the `reveal` itself bake
everything before it). Edits after the shown span reuse the bake, and so does a `set` on an
object that is hidden until after it (opacity or dim 0), that no rule names and nothing reads;
an edit before it (or another `set`, a new object, a new rule) bakes the shown span again.
Lint `slow_layout` says when a bake is over the theme's `[lint] bake_renders`. A label that never moves needs no rule: give it a `pos`.
An expression that reads a placed object's `pos`, or an object that names it (a
`circumscribe` ring), makes the bake run everywhere, hidden or not.

## Camera goals make every edit of a long 3D scene bake all of it

The camera solver is global: the scale of its search comes from every goal object over the
whole timeline, and it re-checks the whole path. So a bake with `camera ...` rules is never
reused after an edit, and every verb the script runs after such a rule bakes the whole scene,
with a depth buffer per key and per `avoid_occlusion` object (one buffer per camera it
tries; none while nothing else occludes, and none while the object is hidden, outside the
lead-in before it appears). Write the camera rules after the verbs, as the last lines of
the script, and give `avoid_occlusion` only the objects that matter. Lint `slow_layout`
counts these buffers too.

## 3D: the camera moves before my object appears

`camera avoid_occlusion(x)` protects nothing while `x` is hidden, so the camera is free
then, and it swings into a view that shows `x` over a lead-in before `x` appears (theme
`[motion] camera_lead`, longer for a bigger swing so it stays under `camera_speed`). That
early swing is meant: the camera is in place when `x` fades in instead of jerking then.
If `x` appears soon after the start, or is hidden only briefly, there is no room for a
slow swing, and the camera protects `x` throughout, as it did before 2026-11-26. A scene
whose camera has a handoff motion (an `orbit` or `focus_pull` at a marker on a parameter
left unset) also protects throughout.

## `overlap` on a label that sits on a big shape

A label fully inside a shape is not an overlap; it is judged by `contrast`. A label half on
a shape is an overlap.

## Layout can differ slightly between machines

With OpenGL the label placer avoids the pixels objects really cover (the id pass). With no
OpenGL it uses bounding boxes, and a thin stripe for lines. Near-ties may pick a different
slot on a different machine. The linter judges whatever was chosen.

## `visible_fraction` is `null`

Either there is no OpenGL on this machine (`"occlusion": "analytic"`), or the object is
thinner than one inspector pixel (`note` says so; hairlines at 480x270 are). Do not treat
`null` as visible or hidden. Geometry (`bbox`, `center`, `region`) is always exact.

## Translucent objects do not hide things

An object with `opacity` below 0.5 does not count as an occluder, so objects behind it
report `visible_fraction` 1.0. Its own visibility is still measured.

## Tweaks to a scripted scene

`set_param` and `set_style` on a scene built by a `.py` script go to `name.overrides.json`.
If the script later drops that object, lint reports `orphan_override` (an error). Fix the
script or delete the entry; phasor never discards a tweak silently.

## Render workers and state the scene script sets up

`phasor render --workers N` starts fresh processes (spawned, as on Windows). Each builds the
scene from the parent's document and bake; it does not have what the parent's process had.
A scene-local type that draws from module-level state the *script* fills (a registry of
drawings in `progs.py`, filled by `scene.py`) would draw nothing there. So when the scene
came from a script (`phasor render scene.py`, `phasor.load("scene.py")`) and the document
names scene code (`meta.modules`), every worker runs that script first, then uses the
parent's document (`scene._worker_engine`). A script edited after the render started is
refused. `phasor.load("scene.py")` also names the script in `meta.modules`, after the
type's modules, when the scene has code of its own, so the saved `.phasor.json` opened with
scripts allowed runs it too and draws the same in one process. Opening such a document
runs the whole script. A document saved before 2026-11-26 does not name its script: load
the script and save again.

Safety net: before its first frame, each worker reports what the scene's own objects
(scene-local types, objects with updaters, their parts) draw at that frame
(`core.record.census`); if it differs from the parent, the render stops at once, naming
the objects. Keep such state in params, in a file the document names, or fill it in the
module `meta.modules` names; or render with `--workers 1`.

## GL contexts

GL calls go to whichever context is current. Never call `moderngl.create_context`; use
`phasor.render.passes.default_renderer()` (or `Engine`), which shares one context and makes
it current for every render.

## Codec lists lie

PyAV wheels list `h264_nvenc` and `hevc_nvenc` even on machines with no NVIDIA GPU.
`phasor doctor` only reports an encoder as working after encoding real frames with it.
Never pick an encoder from `av.codecs_available`.

## Headless OpenGL on Linux

There is no X display in CI or cloud containers. `create_standalone_context()` falls back
to EGL (Mesa llvmpipe gives GL 4.5 in software). Install `libegl1 libgl1-mesa-dri` there.
Software GL is fine for the inspector's 480×270 passes, too slow for final renders.

## `phasor.time` and `phasor.inspect` share names with the standard library

Always import them by their full name (`from phasor.time import ...`). Never run a
script from inside the `phasor/` folder: Python would put that folder on `sys.path`
and the stdlib `time`/`inspect` modules would be shadowed.

## Importing GPU libraries from a lower layer

`phasor.core`, `phasor.style`, `phasor.time` and `phasor.layout` must not import CuPy,
ModernGL or PyAV, even lazily. `lint-imports` catches static imports and
`tests/test_layers.py` catches runtime ones.

## Kernel tests pass but there is no GPU here

Three different things can be "passing": the NumPy reference, the CUDA source compiled for
the CPU (`tests/cuda_emu.py`), and the kernel on a real GPU (`gpu` marker). Only the last
proves NVRTC and CuPy accept it. Say which ones ran.

## Writing a kernel

Scalars passed to a kernel must be NumPy scalars of the exact C type (`np.int32`,
`np.float32`, `np.float64`). Cast constants with `R(...)` so float32 kernels do not drift
into double math. Guard `if (i >= n) return;` because the last block has spare threads.

## Glow that does not glow

Bloom comes only from `glow` roles (`soft`, `strong`), never from bright paint. A theme
color at full brightness rolls off by a few levels at the knee; that is the tone map, not
a bug.

## Text that shows empty boxes

The bundled font has no `²`, `π` or accented letters. Lint reports `missing_glyph`.

## Rendering on a machine without CUDA

It works (tier B) but every kernel runs on NumPy: a 1080p fractal frame takes about 1.5 s
there, against milliseconds on a GPU. Preview at `--preset preview`; render finals on the
NVIDIA machine.

## The GL context and the bridge

`CopyBridge.upload` must be called inside `with ctx:` and does not enter the context
itself. Leaving a nested `with ctx:` makes no context current, and the next GL call in the
outer block fails with "cannot create program".

## Animating by seconds in a narrated scene

Anchor to markers: `s.at("loop_draw")`, not `s.at(4.2)`. Seconds do not move when the voice
changes, so the picture drifts away from the words.

## A revealed object shows from the start

`reveal` hides an object only if nothing animates its opacity earlier. If you fade an
object yourself before revealing it, it is visible until your fade. Use one verb per
appearance.

## Pace last

`s.pace()` measures how much each sentence changes on screen. Calling it before the verbs
are written gives no extra room. Calling it again is harmless.

## No voice in the video

With `voice="estimate"` (or `auto` on a machine without espeak-ng) there is timing but no
audio. `phasor render` prints "with narration" when a track was muxed.

## Driving phasor over MCP

Positions of overlay text are frame heights (a centered title is `x = 0.8889`), lint clean
does not mean it looks good (look at the storyboard), and `.py` scenes are refused unless
the server runs with `--allow-scripts`. The full list is at the end of `MCP.md`.

## 3D: a point on a solid surface vanishes from some angles

A solid surface hides what is behind it, including a point on its other side or on another
sheet of itself. `events()` reports `occluded`/`hidden` with the time and `hidden_by`.
Fixes, in order: make the surface `glass` (it then hides nothing), add
`camera avoid_occlusion(point)`, or move the point.

## 3D: the camera solver ignores my goal

It only sets camera parameters you left unset. Set `elevation` (or animate it) and the
solver will not touch it; then a goal it cannot meet any other way shows up as `unsolved`
with the times. Leave it out to let the solver choose.

## 3D: the event log has no visibility events

Visibility needs real pixels (OpenGL). Without it (tier C) the log still has frame and
region events, and `when_hidden` returns no spans.

## A camera parameter cannot be `None`

Parameter values are numbers, strings, booleans and tuples. `drift=-1` (the default) means
"the theme's drift"; `drift=0` holds the camera still.

## Everything is still (no life)

A plain `Scene(...)` has `life="off"` (old documents stay still too). Use a look
(`calm`, `glow`) or `life="subtle"`/`"lively"`. A wiggling dot never moves a label's value or
a computed pole; tag objects that must be perfectly still `"exact"`.

## `critique` has issues `polish` will not fix

Only certain fixes (a margin rule, one stroke weight, a glow notch down) and the one
judgement it is willing to make (which of several `highlight`s to keep) are applied. An
empty frame, two colliding motions, or "nothing highlighted yet" are yours to decide.

## `chalk` looks washed out

Light themes are opt-in and glow is nearly absent on them by design (soft halos, never
white blow-outs). For emphasis use `highlight`, heavier strokes and dimming rather than glow.

## `variants` needs OpenGL

It renders pictures. Without OpenGL (tier C) use `critique`, which works from geometry.

## `rt60_reliable` is false

`ImpulseResponse` builds the room's response from image sources up to `order` reflections,
which describes the first `order * shortest wall / c` seconds completely (`complete_until`).
A decay longer than that is cut short, not the room's: raise `order` (to about 40) or use a
more absorbent material. Plaster and concrete ring for seconds, far beyond what the image-source
method can follow in a small room.

## A shoebox rings longer than Eyring says

Room theory (Sabine, Eyring) assumes a diffuse field; the image-source response of a shoebox is
not diffuse, so its fitted RT60 lies between Eyring's value and about twice Sabine's. Do not
treat the formula as the answer the fit must reproduce.

## FDTD refuses my `dt`

The pressure field is a wave-equation simulation: `dt` must be at most `dx / (c sqrt(3))`
(a validation error, not a blow-up). Leave `dt` to its default, or use a larger `dx`.

## `attach_to` needs a spot

`a attach_to b at top`: the spot is a box spot (`top`, `bottom`, `left`, `right`, `center`,
`top_left`, ...) or a feature point of `b` (an arrow's `tip`). An unknown spot is reported as
`unsolved` with the list of spots.

## Simulation cache

Checkpoints live in `.phasor_cache/` (or `PHASOR_CACHE`). Delete it freely: it only costs
time. A changed parameter is a new hash, so stale results never come back.

## A PDE time that never finishes

`time` of a `pde` object is simulation time. Heat on a unit rod with k = 1 is over by 0.1 s;
a 6-metre plate with k = 0.1 takes minutes. Tie it to the scene clock with a factor
(`time=expr("t / 10")`) and look at `steps` in the semantics: a frame more than 400,000 steps in is
refused, and a solver step that would be unstable is a validation error, not a blow-up.

## `CayleyTable` says "at most 24"

Tables are drawn for groups up to 24 elements (`D12`, `Z24`, `S4`). Bigger groups are still
available as numbers in `algebra_math`.

## A `Polygon` has no element name

The semantics name the group element only when the polygon sits on a symmetry: `rotation` a
multiple of `360 / n` and `flip` exactly 0 or 1. Mid-animation it is `None` on purpose.

## `Taylor` and `RiemannSum` numbers are floats

`order` and `n` are numbers so they can be animated; they are rounded when used.

## A deep zoom is all one color

The view is inside the set because `max_iter` is too small for the depth: escape counts grow
with the zoom (about 1700 at 1e-30 on the spiral in `examples/deep_zoom.py`). Raise
`max_iter` to a few thousand. If the colors flash light and dark during a dive, set
`steady=True`.

## A deep Julia zoom turns into blocks

Only the Mandelbrot set has the perturbation path. A Julia set stays float64 and goes blocky
below about 1e-10 per pixel (`precision` in the semantics says `f64`).

## `center` must be text

`cx.Fractal(center=...)` takes `"re im"` as decimal text (`"-0.75 0.1"`, `"1e-30 0"`), not a
tuple: floats would round the point to 16 digits and a deep view would land somewhere else.

## My heat map's hottest spot is dark

Fixed: `ember`, `ocean` and `sequential` are cyclic (made for fractals) and come
back to dark at their end. PDE heat maps and the spectrogram now use only the rising half (`core.color.ramp`). If you
write your own object that maps a value onto a colormap, use `ramp` too.

## `Limit` says "no finite limit"

`calc.Limit` needs the two one-sided limits to agree and be finite (`1/x` at 0 has none). The
limit is found symbolically, so write the function so SymPy can see it (`sin(x)/x`, not a
piecewise trick).

## `CayleyGraph` says "does not generate"

The generators must reach every element, or the graph falls apart into pieces: `"r"` alone
gives only the rotations of D4; use `"r s"`.

## Math shows as plain text with dollar signs

Only strings with a *pair* of unescaped `$` are typeset. In Python write raw strings
(`r"$\psi$"`) or double the backslashes. A lone `\$` is a dollar sign.

## I want one term of an equation in another color

Use `Equation(parts=(...), colors=(...))`, splitting the TeX where a part can stand alone. A
term inside a fraction or a root cannot be colored separately (each part is typeset on its
own).

## The prism look clutters my scene with rainbows

`hue_along` is the look's default for lines. Give a line that should be one color
`hue_along=0` (and a `color` role); keep the rainbow for the curve that is the subject.

## A Manim script imports but looks different

Phasor's import records the scene a script builds; it does not reimplement Manim. Fills are
outlines, colors are the nearest style *role*, updaters and `always_redraw` keep their first
look, and `Transform` between different shapes crossfades. `phasor manim` prints each of these
as a `note:`. If a name is missing the error says `X is not supported ... yet`: replace it in
the script, or add the equivalent phasor object after importing.

## `add_control` says "name a component"

A vector parameter (`c`, `pos`) needs `component`: 0 is the first entry. A parameter that an
animation track drives, or that holds an expression, cannot be a slider.

## The web page is coarse

`phasor web` bakes pictures in advance: each slider has `--steps` positions (5 by default) and
the page shows the nearest. More steps means a bigger file (the limit is 600 pictures in all).
Time can be a slider too: `--time-steps 12`.

## Math looks different on another machine

Only with `PHASOR_MATH=latex` or `auto`: a real TeX draws with its own fonts and spacing. The
default `mathtext` is built in and identical everywhere; keep one engine per project.
