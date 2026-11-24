"""phasor command line: doctor | render | storyboard | lint | inspect | bench | style | mcp | gui.

A thin client (contract 1): it holds no scene state of its own.
"""

from __future__ import annotations

import argparse
import json
import sys

# command -> (version it is planned for, help); none at the moment
PLANNED: dict[str, tuple[int, str]] = {}

EXIT_NOT_IMPLEMENTED = 2
EXIT_TIER_TOO_LOW = 1
EXIT_LINT_FAILED = 1

ALLOW_SCRIPTS_HELP = (
    "run the Python code a saved .phasor.json names (meta.modules: its own types and "
    "updaters); it runs only on your request, so only your own code"
)


def _open(args: argparse.Namespace, path: str | None = None):
    """Open the scene for a command: the one place the CLI loads a scene. A ``.py`` scene
    is code the person ran; a saved document's modules run only with ``--allow-scripts``."""
    from phasor.scene import load

    kwargs = {"allow_scripts": True} if getattr(args, "allow_scripts", False) else {}
    return load(path or args.scene, **kwargs)


def _cmd_doctor(args: argparse.Namespace) -> int:
    from phasor.render.doctor import format_report, run_doctor

    report = run_doctor()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    if args.require and report.tier > args.require:  # "A" < "B" < "C"
        print(
            f"tier {report.tier} is below the required tier {args.require}",
            file=sys.stderr,
        )
        return EXIT_TIER_TOO_LOW
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from phasor.core.issues import DocumentError

    try:
        engine = _open(args)
        out = engine.inspect(args.t, id=args.id, detail=args.detail)
    except (DocumentError, KeyError, OSError, ValueError) as e:
        print(f"phasor inspect: {e}", file=sys.stderr)
        return 1
    if args.detail == "brief":
        print(f"t={out['t']} occlusion={out['occlusion']}", *out["objects"], sep="\n")
    else:
        print(json.dumps(out, indent=2))
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    from phasor.core.issues import DocumentError

    try:
        engine = _open(args)
        fixed = engine.autofix(aesthetic=args.aesthetic)["applied"] if args.fix else []
        issues = engine.lint(step=args.step, strict=args.strict, aesthetic=args.aesthetic)
        if fixed:
            # A script keeps its fixes in name.overrides.json; a document is rewritten.
            engine.save(_save_target(args.scene))
    except (DocumentError, OSError, ValueError) as e:
        print(f"phasor lint: {e}", file=sys.stderr)
        return EXIT_LINT_FAILED
    errors = [i for i in issues if i.severity == "error"]
    if args.json:
        print(json.dumps({"fixed": fixed, "issues": [i.to_dict() for i in issues]}, indent=2))
    else:
        for op in fixed:
            print(f"fixed: {op['op']} {op.get('rule', '')}".rstrip())
        for i in issues:
            print(f"{i.severity}: {i}")
            if i.fix_hint:
                print(f"    hint: {i.fix_hint}")
        print(f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    return EXIT_LINT_FAILED if errors or (args.strict and issues) else 0


def _save_target(scene: str):
    """Where ``lint --fix`` saves: the document itself, or a script's document beside it
    (which writes only the overrides file's sibling; the script is never edited)."""
    from pathlib import Path

    from phasor.core import io

    p = Path(scene)
    return p if p.suffix != ".py" else p.with_name(p.stem + io.DOC_SUFFIX)


def _cmd_storyboard(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.core.issues import DocumentError

    try:
        engine = _open(args)
        board = engine.storyboard(args.t or None, marks=args.marks, columns=args.columns)
        from phasor.render.output import save_png

        out = Path(args.out or Path(args.scene).name.split(".")[0] + ".storyboard.png")
        save_png(out, board["image"])
    except (DocumentError, OSError, ValueError, RuntimeError) as e:
        print(f"phasor storyboard: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"image": str(out), "frames": board["frames"]}, indent=2))
    else:
        print(f"wrote {out}")
        for fr in board["frames"]:
            name = f" [{fr['marker']}]" if fr["marker"] else ""
            print(f"t={fr['t']:g}{name}", *fr["objects"], sep="\n  ")
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.core.issues import DocumentError
    from phasor.interactive import export_web

    try:
        engine = _open(args)
        if not engine.document.controls:
            print("phasor web: the scene has no sliders (add some with scene.control(...)); "
                  "writing a page with only the picture", file=sys.stderr)  # fmt: skip
        out = Path(args.out or Path(args.scene).name.split(".")[0] + ".html")
        info = export_web(engine, out, size=(args.width, args.height), steps=args.steps,
                          time_steps=args.time_steps)  # fmt: skip
    except (DocumentError, OSError, ValueError, RuntimeError) as e:
        print(f"phasor web: {e}", file=sys.stderr)
        return 1
    axes = ", ".join(f"{a['label']} x{a['steps']}" for a in info["axes"]) or "no sliders"
    print(f"wrote {info['path']}: {info['frames']} pictures ({axes}), {info['bytes'] // 1024} KB")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.core.issues import DocumentError
    from phasor.render.cpu import set_threads
    from phasor.render.encode import EncodeError
    from phasor.render.job import render_file
    from phasor.scene import default_threads, default_workers, parallel_frames

    def progress(k: int, n: int) -> None:
        if not args.quiet:
            print(f"\rframe {k}/{n}", end="", file=sys.stderr, flush=True)

    stem = Path(args.scene).name.split(".")[0]
    try:
        engine = _open(args)
        default = stem + (".png" if args.frame is not None else ".mp4")
        workers = default_workers() if args.workers is None else args.workers
        threads = default_threads(workers) if args.threads is None else args.threads
        set_threads(threads)  # this process's kernels (all of them with --workers 1)
        source = parallel_frames(engine, workers, threads=threads) if workers > 1 else None
        r = render_file(
            engine, args.out or default, preset=args.preset, fps=args.fps, start=args.start,
            end=args.end, frame=args.frame, progress=progress, shutter=args.shutter,
            blur_samples=args.blur_samples, frame_source=source,
        )  # fmt: skip
    except (DocumentError, OSError, ValueError, RuntimeError, EncodeError) as e:
        print(f"\nphasor render: {e}", file=sys.stderr)
        return 1
    for w in r["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    w, h = r["size"]
    if args.frame is not None:
        print(f"wrote {r['path']} ({w}x{h}, t = {args.frame:g} s)")
    else:
        sound = ", with narration" if r["narration"] else ""
        print(f"\nwrote {r['path']}: {r['frames']} frames, {w}x{h} at {r['fps']:g} fps, "
              f"{r['encoder']}{sound}")  # fmt: skip
    return 0


def _cmd_kit(args: argparse.Namespace) -> int:
    from phasor.workspace.kit import KitError, install
    from phasor.workspace.paths import scenes_folder

    try:
        r = install(args.folder or scenes_folder())
    except (KitError, OSError) as e:
        print(f"phasor kit: {e}", file=sys.stderr)
        return 1
    print(f"the AI kit is in {r['dest']}")
    print("  written: " + ", ".join(r["written"]))
    if r["kept"]:
        print("  kept (yours): " + ", ".join(r["kept"]))
    print(f'start an AI there: cd "{r["dest"]}" and run claude; then /scene <your idea>')
    return 0


def _cmd_style(args: argparse.Namespace) -> int:
    from pathlib import Path

    from phasor.style import choices

    path = Path(args.choices)
    try:
        report = choices.tune(path, args.min)
    except (OSError, ValueError) as e:
        print(f"phasor style: {e}", file=sys.stderr)
        return 1
    prefs = path.with_name("preferences.json")
    if report["choices"]:
        prefs.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not report["choices"]:
        print(f"no choices yet in {path}: pick variants (the `variants` and `choose` tools)")
        return 0
    print(f"{report['choices']} choice(s) in {path}")
    for name, r in report["options"].items():
        print(f"  {name:12s} won {r['won']} of {r['offered']} ({r['rate']:.0%})")
    fav = report["favored"]
    print(f"you favor: {fav}" if fav else f"no clear favorite yet (needs {args.min}+ appearances)")
    print(f"wrote {prefs}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.frames:
        return _cmd_bench_frames(args)
    from phasor.render.bench import run_bench, save

    result = run_bench(args.width, args.height, args.repeats)
    path = save(result, args.dir)
    print(f"backend {result['backend']}, {args.width}x{args.height}, median of {args.repeats}")
    for name, r in sorted(result["kernels"].items()):
        print(f"  {name:8s} {r['ms']:10.2f} ms")
    if result["peak_vram_bytes"] is not None:
        print(f"  peak VRAM {result['peak_vram_bytes'] / 2**20:.0f} MB")
    print(f"saved {path}")
    return 0


def _cmd_bench_frames(args: argparse.Namespace) -> int:
    """Where a final frame's time goes, stage by stage."""
    from pathlib import Path

    from phasor.core.issues import DocumentError
    from phasor.render.bench import run_frame_bench, save_frames

    scene = Path(args.frames)
    if not scene.exists() and (Path("examples") / f"{args.frames}.py").exists():
        scene = Path("examples") / f"{args.frames}.py"  # an example by name
    try:
        engine = _open(args, str(scene))
        r = run_frame_bench(engine, scene.name.split(".")[0], args.count, args.preset)
    except (DocumentError, OSError, ValueError, RuntimeError) as e:
        print(f"phasor bench: {e}", file=sys.stderr)
        return 1
    path = save_frames(r, args.dir)
    w, h = r["frame"]
    blur = ", motion blur" if r["motion_blur"] else ""
    print(f"{r['scene']}: {r['frames']} frames, {w}x{h}{blur}, kernels on {r['backend']}, "
          f"GL on {r['gl_renderer']}, {r['encoder']}")  # fmt: skip
    for name, ms in r["stages_ms_per_frame"].items():
        print(f"  {name:10s} {ms:9.1f} ms/frame")
    print(f"  {'total':10s} {r['ms_per_frame']:9.1f} ms/frame  (setup {r['setup_s']:.1f} s, once)")
    print(f"saved {path}")
    return 0


def _planned(name: str, milestone: int):
    def run(_: argparse.Namespace) -> int:
        print(f"phasor {name}: not implemented yet (Milestone {milestone})", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    return run


def build_parser() -> argparse.ArgumentParser:
    from phasor import __version__
    from phasor.render.job import PRESETS

    p = argparse.ArgumentParser(prog="phasor", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"phasor {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    d = sub.add_parser("doctor", help="check CUDA, NVRTC, OpenGL 4.3 and encoders; print tier")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument(
        "--require",
        choices=["A", "B", "C"],
        help="exit 1 if the machine's tier is below this one",
    )
    d.set_defaults(func=_cmd_doctor)

    i = sub.add_parser("inspect", help="what is on screen at time t, as JSON")
    i.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    i.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    i.add_argument("--t", type=float, default=0.0, help="time in seconds (default 0)")
    i.add_argument("--id", help="only this object")
    i.add_argument("--detail", choices=["brief", "standard", "full"], default="standard")
    i.set_defaults(func=_cmd_inspect)

    li = sub.add_parser("lint", help="find visual problems at every marker and sampled times")
    li.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    li.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    li.add_argument("--fix", action="store_true", help="apply the fixes the linter suggests")
    li.add_argument("--aesthetic", action="store_true", help="also check taste (focal point, ...)")
    li.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    li.add_argument("--json", action="store_true", help="machine-readable output")
    li.add_argument("--step", type=float, default=0.5, help="seconds between samples")
    li.set_defaults(func=_cmd_lint)

    sb = sub.add_parser("storyboard", help="keyframe stills on one contact sheet + inspector")
    sb.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    sb.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    sb.add_argument("-o", "--out", help="output PNG (default: <scene>.storyboard.png)")
    sb.add_argument("--t", type=float, action="append", help="a time to show (repeatable)")
    sb.add_argument("--marks", action="store_true", help="draw each object's box and id")
    sb.add_argument("--columns", type=int, default=3)
    sb.add_argument("--json", action="store_true", help="machine-readable output")
    sb.set_defaults(func=_cmd_storyboard)

    r = sub.add_parser("render", help="render a video (or one frame) with the final look")
    r.add_argument("scene", help="a .phasor.json document or a .py script that builds a Scene")
    r.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    r.add_argument("-o", "--out", help="output: .mp4, .mov (ProRes), a folder or a %%05d.png")
    r.add_argument("--preset", choices=sorted(PRESETS), default="preview")
    r.add_argument("--fps", type=float, help="override the preset's frame rate")
    r.add_argument("--start", type=float, help="first time in seconds (default 0)")
    r.add_argument("--end", type=float, help="last time in seconds (default: the end)")
    r.add_argument("--frame", type=float, help="render only the frame at this time, as PNG")
    r.add_argument("--motion-blur", type=float, metavar="SHUTTER", dest="shutter",
                   help="shutter as a share of a frame (0 off, 0.5 film standard); "
                   "default: off for preview and draft, 0.5 for final and 4k")  # fmt: skip
    r.add_argument("--blur-samples", type=int, help="sub-frames per frame when blurring (4)")
    r.add_argument("--workers", type=int,
                   help="render processes (default: a quarter of the CPU threads, 1 to 4; "
                   "1 renders in this process)")  # fmt: skip
    r.add_argument("--threads", type=int,
                   help="CPU kernel threads per render process, without CUDA (default: the "
                   "cores shared out, workers x threads <= cores; env PHASOR_THREADS)")  # fmt: skip
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_render)

    k = sub.add_parser("kit", help="set up (or update) a folder where an AI writes scenes")
    k.add_argument("folder", nargs="?",
                   help="where (default: Documents/phasor, the app's scenes folder)")  # fmt: skip
    k.set_defaults(func=_cmd_kit)

    st = sub.add_parser("style", help="what your recorded choices say about your taste")
    st.add_argument("action", choices=["tune"], help="tune: tally style/choices.jsonl")
    st.add_argument("--choices", default="style/choices.jsonl", help="the choices log")
    st.add_argument("--min", type=int, default=3, help="appearances needed to call a favorite")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    st.set_defaults(func=_cmd_style)

    w = sub.add_parser("web", help="bake an interactive web page (sliders) from a scene")
    w.add_argument("scene", help="a .phasor.json or .py scene with controls")
    w.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    w.add_argument("-o", "--out", help="the HTML file (default: <scene>.html)")
    w.add_argument("--steps", type=int, default=5, help="positions per slider (default 5)")
    w.add_argument("--time-steps", type=int, default=1, help="also bake this many moments of time")
    w.add_argument("--width", type=int, default=320)
    w.add_argument("--height", type=int, default=180)
    w.set_defaults(func=_cmd_web)

    b = sub.add_parser("bench", help="ms per frame per kernel and peak VRAM")
    b.add_argument("--width", type=int, default=1920)
    b.add_argument("--height", type=int, default=1080)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--dir", default="benchmarks", help="where results are saved")
    b.add_argument("--frames", metavar="SCENE", help="time whole frames of SCENE (a file or "
                   "an example name) stage by stage, instead of single kernels")  # fmt: skip
    b.add_argument("--count", type=int, default=60, help="frames to time (with --frames)")
    b.add_argument("--preset", default="final", choices=sorted(PRESETS),
                   help="render preset (with --frames)")  # fmt: skip
    b.add_argument("--allow-scripts", action="store_true", help=ALLOW_SCRIPTS_HELP)
    b.set_defaults(func=_cmd_bench)

    # Client commands (mcp, gui) register themselves through the "phasor.commands"
    # entry-point group, so this module never imports a sibling client (contract 6).
    from importlib.metadata import entry_points

    for ep in entry_points(group="phasor.commands"):
        ep.load()(sub)
    for name, (milestone, help_) in PLANNED.items():
        sp = sub.add_parser(name, help=f"{help_} (Milestone {milestone})")
        sp.add_argument("args", nargs=argparse.REMAINDER)
        sp.set_defaults(func=_planned(name, milestone))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
