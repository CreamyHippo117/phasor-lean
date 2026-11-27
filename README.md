# phasor

Math animations you can check. You write a scene as a short Python script; phasor renders it
on the GPU (or the CPU), and it can also tell you what is on screen at any moment: which
objects, where, what hides what, and which labels collide or leave the frame. That makes it
a good fit for scenes written by an AI, which can look at its own work and fix it.

- **Every object has an id** (`pole`, `pole.label`). Labels are placed by rules, not
  coordinates, so they follow what they label.
- **A linter for pictures.** `phasor lint` finds overlaps, off-screen objects, low contrast,
  tiny text and clutter at every marker, and says how to fix each one.
- **A storyboard**: keyframes on one sheet, with every object's box and id drawn on it.
- **Complex analysis, 3D, fields and more**: domain coloring, fractals with deep zoom,
  Riemann surfaces, heat and wave fields, room acoustics, point clouds, Fourier series,
  phase portraits, filters, graphs. See [`examples/`](examples).
- **The prism look** by default: glowing strokes on a dark stage, color running along curves.
- **Sound**: music and ambience beds and one-shot cues under the narration, ducked and mixed to a
  loudness target, editable in the desktop editor's Sound tab.

## Install

Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/creamyhippo117/phasor-lean.git
cd phasor-lean
uv sync --extra gl --extra encode            # OpenGL drawing and video encoding
uv run phasor doctor                         # what your machine can do
```

Optional extras, added with more `--extra NAME` flags (or `--all-extras`):

| Extra | What it adds |
| --- | --- |
| `gl` | OpenGL drawing (needs OpenGL 4.3). Needed for pictures and videos; without it you still get lint, inspect and the timeline. |
| `encode` | Video files (`phasor render`). Uses NVENC when the driver has it. |
| `gpu` | CUDA for fields, fractals and point clouds on NVIDIA cards. No CUDA Toolkit needed. Without it the same kernels run on the CPU, more slowly. |
| `mcp` | The MCP server an AI client talks to (`phasor mcp`). |
| `gui` | The desktop editor (`phasor gui`). |
| `notebook` | Sliders in Jupyter. |

## Your first scene

Save this as `scenes/orbit.py`:

```python
from phasor import Dot, Line, Scene, Text

s = Scene("orbit")
s.add(Line(id="axis", points=[(-4, 0), (4, 0)], style={"color": "muted"}))
s.add(Dot(id="dot", pos=(-3, 0), r=0.3, style={"color": "highlight", "glow": "soft"}, tags=["key"]))
s.add(Text(id="dot.label", text="$z$", style={"size": "title"}))
s.constrain("label(dot) near dot")  # placed by a rule, so it follows the dot
s.set("camera.height", 5)  # world units visible top to bottom

s.marker("go", 1.0)
s.at("go").to("dot.pos", (3, 0), duration=2.0)
```

Style takes roles (`highlight`, `muted`, `title`) rather than raw colors or pixel sizes, so
a scene keeps working when you change its look. Then check it, look at it and render it:

```bash
uv run phasor lint scenes/orbit.py --aesthetic                     # problems, with fixes
uv run phasor storyboard scenes/orbit.py --marks -o out/orbit.png  # keyframes on one sheet
uv run phasor inspect scenes/orbit.py --t 2 --detail brief         # what is on screen at t = 2 s
uv run phasor render scenes/orbit.py --preset preview -o out/orbit.mp4
```

Presets: `draft` (640x360, 15 fps), `preview` (960x540, 30 fps), `final` (the scene's own
size and rate, 1080p 60 fps by default) and `4k`. `--frame T` renders one PNG instead.

The whole Python API, with a runnable example for each name, is in
[`docs/ai/API.md`](docs/ai/API.md). Read [`docs/ai/PATTERNS.md`](docs/ai/PATTERNS.md) and
[`docs/ai/PITFALLS.md`](docs/ai/PITFALLS.md) before a bigger scene.

## Working with an AI

phasor was built so an AI can write a scene, check it and fix it on its own. Two ways in:

**A ready folder for Claude Code.** `phasor kit` sets up a folder with instructions, the
docs, sorted examples, a `/scene-plan` and a `/scene-builder` command and a taste file:

```bash
uv sync --extra gl --extra encode --extra mcp
uv run phasor kit ~/phasor-scenes
cd ~/phasor-scenes && claude          # then: /scene-plan a Fourier series building a square wave
                                      # and, once you like the plan: /scene-builder plans/NAME.md
```

`/scene-plan` turns an idea into a plan (what to show, in what order, with what math) and writes
`plans/NAME.md`; `/scene-builder` codes the plan as `scenes/NAME.py`, lints it, looks at the storyboard,
fixes what is wrong and renders a preview. When you say what you like or dislike, it adds a line to `TASTE.md` and
reads it before every scene. Run `phasor kit` on the folder again after updating phasor:
your `TASTE.md`, `WISHLIST.md` and `scenes/` are never touched. See [`kit/README.md`](kit/README.md).

**Any MCP client.** `phasor mcp` serves tools to build a scene by patches and check it
(`review` does lint, inspect and a storyboard picture in one call):

```bash
claude mcp add phasor -- uv run --project /path/to/phasor-lean phasor mcp --root ./scenes
```

Scenes stay inside `--root`. Python scenes are code, so the server opens them only with
`--allow-scripts`. The tools are described in [`docs/ai/MCP.md`](docs/ai/MCP.md);
[`llms.txt`](llms.txt) has the same in one page, and [`skills/phasor/SKILL.md`](skills/phasor/SKILL.md)
is a skill that teaches the loop.

## The desktop editor

```bash
uv sync --extra gl --extra encode --extra gui --extra mcp
uv run phasor gui                               # a start screen with the examples
uv run phasor gui examples/julia_explorer.py    # straight into a scene; Sliders tab: move c
```

The editor shows the picture, the timeline, the inspector, lint, looks, sliders and the
scene's code. File > Export makes pictures, videos and web pages. While it is open,
`phasor mcp --attach --root <its folder>` lets an AI edit the same live scene.
A packaged app that needs no Python is built with PyInstaller: see [`packaging/README.md`](packaging/README.md).

## More commands

```bash
uv run phasor web examples/julia_explorer.py -o julia.html --steps 7   # an offline page with sliders
uv run phasor manim my_manim_scene.py -o scene.phasor.json             # import a Manim scene
uv run phasor style tune                                               # what your choices say about your taste
uv run phasor bench                                                    # speed of each kernel
```

`uv run phasor COMMAND --help` lists each command's options.

Math in text (`$...$`) is typeset by a built-in engine. Set `PHASOR_MATH=latex` to use a
real TeX instead (`latex` and `dvipng` on the PATH).

## Examples

Render most of them with `uv run phasor render examples/NAME.py -o out/NAME.mp4`, or open
them in the editor. A few to start with:

| Example | What it shows |
| --- | --- |
| `prism_primes.py` | The reference look: primes heard through the zeta zeros |
| `room_3d.py` | A 3D room with sound bouncing in it, the camera circling |
| `sound_bed.py` | A bed under the narration, with cues on the markers |
| `prism_domain.py`, `prism_transitions.py` | Domain coloring in the prism look, and a function changing |
| `riemann_sqrt.py` | The Riemann surface of the square root, seen through an orbit |
| `deep_zoom.py` | A dive into the Mandelbrot set to 1e-30 |
| `residue_theorem.py` | A narrated scene: every motion anchored to a word |
| `motion_gallery.py` | The motion verbs, one beat each |
| `interference.py` | Two sources interfering, then a steered beam |
| `strange_attractor.py`, `point_cloud.py` | Hundreds of thousands of glowing points |
| `open_contract.py` | Your own object type in about ten lines |

[`kit/examples.toml`](kit/examples.toml) sorts all of them by kind and says what each teaches.

## Repository layout

| Path | What is there |
| --- | --- |
| `phasor/` | The library |
| `examples/` | Example scenes |
| `docs/ai/` | API reference, patterns, pitfalls, MCP tools (shipped inside the package) |
| `docs/decisions/` | Design decisions and why they were made |
| `STYLE.md` | The look phasor aims for; the critic judges scenes by it |
| `kit/`, `skills/` | What `phasor kit` copies into an AI's scene folder |
| `packaging/` | Building the desktop app |
| `tests/` | The test suite |

## Platforms

Tested on Windows 11 with an NVIDIA RTX 4070 Ti, and on Linux with software OpenGL (CI runs
Ubuntu and Windows, Python 3.11 and 3.12). macOS is untested. Without an NVIDIA card the GPU
kernels run on the CPU: compiled with `g++` or `clang++` when one is on the PATH, otherwise
as plain NumPy (slow for final renders). `phasor doctor` says which path your machine takes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): how phasor is put together, the rules that keep it
consistent, and how to run the checks.

## License

MIT, see [LICENSE](LICENSE).
