# Contributing to phasor

How phasor is put together, the rules that keep it consistent, and how to check a change.
The reasons behind the bigger choices are in [`docs/decisions/`](docs/decisions).

## Set up and check

```bash
uv sync --all-extras                      # or leave out gpu without an NVIDIA card
uv run ruff check . && uv run ruff format --check .
uv run lint-imports                       # the layer order below
uv run pytest                             # gl and gpu tests skip themselves without hardware
uv run phasor doctor
```

Headless Linux needs `libegl1 libgl1-mesa-dri` for the OpenGL tests. The CPU kernel tests
need `g++` on the PATH (Windows: `winget install BrechtSanders.WinLibs.POSIX.UCRT`).

Opt-in test runs:

```bash
PHASOR_UPDATE_GOLDEN=1 uv run pytest -k golden                 # regenerate reference pictures, then look at them
PHASOR_TEST_PACKAGE=1 uv run pytest tests/test_packaging.py    # build the desktop app (~1 min)
```

Reference pictures live in `tests/golden/<software|gpu>/`. Look at every regenerated one
before committing it.

## The seven contracts

Everything else in phasor can change; these hold the design together. A change to one of
them needs a decision record in `docs/decisions/`.

1. **One engine.** `phasor.engine.Engine` owns the document, the state cache, the layout
   bake and the inspector. The CLI, MCP server and editor are thin clients. Every change is
   `Engine.apply(patch) -> PatchResult` (validate, record undo, return the lint delta).
   There is no other write path.
2. **The document is the truth.** pydantic models in `core/schema.py`, with
   `schema_version` and `core/migrate.py`. A scene saves as `name.phasor.json` (sorted
   keys) plus `name.overrides.json` (tweaks by id and parameter path, and added
   constraints). Re-running a script never loses overrides; one that points at nothing is
   lint `orphan_override`. Ids match `[a-z][a-z0-9_]*`, dots for children, are named by
   meaning and never reused (removed ids go to `retired_ids`).
3. **Time is pure.** `Engine.state_at(t)` depends only on the document, `t`, the occlusion
   oracle and files the document names inside the scene's folder (a missing one is lint
   `missing_file`). It returns frozen data. Parameters form a DAG (constants, tracks,
   expressions); a cycle is a validation error. Updaters (`obj.always`) are pure DAG nodes.
   Narration is a separate `TimeMap`. Simulations use a fixed `dt`, a seed and cached
   checkpoints; randomness only through `rng(object_id, seed)`.
4. **One spatial pipeline.** Everything is 3D; 2D is an orthographic camera looking down
   -z at z = 0. World: right-handed, z up, x = Re, y = Im. Screen: pixels from the top
   left, and 0-1 coordinates. Colors stay linear RGBA16F until the end; tone mapping and
   sRGB encoding happen exactly once.
5. **The object contract.** Every type has `Params` and `draw(ctx, state)`; registration
   fails without either. `world_bounds`, `features` and `semantics` are derived from what
   `draw` records unless the type writes them; `build(state)` makes a composite.
6. **Layers.** Higher layers import lower ones, never the reverse:
   `core ← style ← time ← layout ← inspect ← render ← domains ← engine ← interactive ← voice ← scene ← importers ← workspace ← (cli | gui | mcp | sketch)`.
   `core`, `style`, `time` and `layout` import no GPU, GL or video library. A lower layer
   that needs render output defines a protocol, `render` implements it and `Engine`
   injects it. Enforced by `lint-imports` and `tests/test_layers.py`.
7. **Layout is baked.** The solver runs at markers and sample times and writes tracks;
   renderers and the inspector both read the bake. kiwisolver handles linear relations,
   then the label placer (8 compass slots plus leaders, 20% hysteresis). Unsatisfiable
   constraints become lint issues. A render worker applies its parent engine's bake.

## Where things go

| Layer | What it holds |
| --- | --- |
| `core` | the document schema, ids, patches, migrations, the constraint language (`core/constraints.py`), text and math layout, `Simulation` |
| `style` | themes (`.toml`), roles, looks and presets |
| `time` | expressions (`time/expr.py`), tracks, the parameter graph (`time/compile.py`), easing, narration and the choreography verbs (`time/verbs.py`) |
| `layout` | the bake (`layout/bake.py`): linear constraints (`layout/solver.py`), the label placer (`layout/labels.py`) and the camera solver (`layout/camera.py`) |
| `inspect` | what is on screen, lint rules (`inspect/lint.py`) and taste rules (`inspect/aesthetic.py`) |
| `render` | OpenGL passes, GPU kernels (`render/gpu/kernels/`), the CPU kernel backend, post, encoding |
| `domains` | object types: maths in `domains/<name>_math.py` (pure NumPy), drawing in `domains/<name>.py` |
| `engine.py` | the `Engine` |
| `interactive`, `voice`, `scene.py` | sliders, narration, the Python authoring API |
| `importers` | foreign scripts (Manim) |
| `workspace` | the tools shared by the MCP server and the editor (`workspace/tools.py`) |
| `cli`, `gui`, `mcp`, `sketch` | the clients |

## Conventions

**General**
- One way to do each thing. Extend, never duplicate; search before adding a helper.
- Tests first for the timeline, patches, the inspector and the solvers. Every feature gets a
  storyboard snapshot test.
- When a test and the code disagree, work out which one is wrong before changing either.
- Text files are read and written with `encoding="utf-8"` (Windows defaults to cp1252).
- Every `subprocess` call passes `creationflags=NO_WINDOW` (`core/proc.py`), or the Windows
  app flashes a console window; `tests/test_proc.py` checks it.
- Don't claim something works on a platform (Windows, CUDA, macOS) unless it ran there.

**Object types**
- Every type shipped in phasor has semantics, bounds, inspector output, a lint rule if
  relevant, an entry in `docs/ai/API.md` and an example in `examples/`.
- Kernel-drawn objects call `ctx.field(...)` and render only their visible part; mark them
  `"backdrop": True` in their semantics if labels may sit on them.
- A time-stepping model is a `Simulation` (`core/simulation.py`): fixed `dt`, the seed in
  `params()`, a pure `step`; only `SimRunner.state_at(t)` reads it.
- 3D: the camera solver runs only inside the bake and may set only camera parameters the
  document leaves unset. 3D labels are judged by what is in front of them.

**Rendering**
- GPU code is CuPy `RawKernel` CUDA C in `render/gpu/kernels/`, registered with a NumPy
  reference (registration fails without one). Each kernel is tested three ways: the
  reference against known maths, its CUDA source compiled for the CPU (`tests/cuda_emu.py`)
  and on a real GPU (`gpu` marker). Add new kernels to `CASES` in `tests/test_kernels.py`.
  Launchers use only array calls NumPy and CuPy share.
- Only `render/gpu/bridge.py` moves pixels between CUDA and OpenGL. GL contexts come only
  from `render/glcontext.py`; all GL work goes through `render/passes.py`
  (`default_renderer()`) inside `with ctx:` (queries too) and on the GL thread
  (`glcontext.on_gl_thread(fn)`).
- Bloom comes only from the emission target (glow roles); never threshold the picture.
  Paint below the theme's knee comes out of the tone map unchanged.

**Layout, lint and time**
- Layout runs only inside `layout/bake.py`, and only `Engine` runs or applies the bake.
  Anything that draws or inspects takes `Engine.state_at(t)`.
- A lint rule lives in `inspect/lint.py`, takes its threshold from the theme's `[lint]`
  table (sound rules from `[sound]`), reports `times`, and offers a `fix` only when the fix is certain to be right. Test
  new rules and layout features with and without OpenGL (`MODES` in `tests/test_lint.py`).
- Taste rules live in `inspect/aesthetic.py` and run only with `aesthetic=True`.
- A speed-up may change the `final` and `4k` pictures only where the eye cannot see it: every
  8-bit value within 1 code value of before. A visible change is for the `draft` and `preview`
  presets only.
- Verbs live in `time/verbs.py`, take their timing from the theme's `[motion]`, and give
  the same result whatever order a script writes them in. Narrated scenes anchor every
  motion to a marker; `lint(strict=True)` checks it.
- Life (the living motion) is a pure function of (id, seed, t); `exact` objects never move.

**Looks**
- New scenes get the `prism` look; `look="none"` is the plain dark `midnight_glow` theme.
  Dark by default, light themes opt-in. Hue that carries meaning (domain-coloring phase)
  keeps the whole wheel. Tests run with `PHASOR_DEFAULT_LOOK=none` (`tests/conftest.py`).

**Clients**
- MCP tools are methods of `Workspace` in `workspace/tools.py` (plain data in and out,
  tested without the protocol). `mcp/server.py` only names them. A new tool goes there, in
  `TOOL_NAMES`, in `docs/ai/MCP.md`, and then regenerate `llms.txt`:
  `uv run python -m phasor.mcp.llms`.
- Editor logic lives in `gui/controller.py` (no Qt, tested directly); `gui/window.py` only
  draws it. Window tests run on Qt's `offscreen` platform.
- Importers run foreign scripts only when the person asks; a stand-in module never leaks
  past the import. Interactive features work on `Engine.fork()`, never on the open scene.
- Client commands register through the `phasor.commands` entry point; `cli` never imports
  a sibling client.

**Docs**
- Update `docs/ai/` in the same commit as an API change. Every ```python block there runs
  in CI (`tests/test_docs.py`), and so does every ```json patch block in `docs/ai/MCP.md`.
- Adding an example: put it in `examples/` and give it an entry in `kit/examples.toml`;
  `tests/test_kit.py` fails until it has one.
