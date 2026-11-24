"""``phasor manim``: registers itself with the CLI through the ``phasor.commands`` entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser("manim", help="turn a Manim Community scene script into a phasor scene")
    m.add_argument("script", help="a .py file with a Manim Scene (it runs: only your own code)")
    m.add_argument("--scene", help="which Scene class, if the file defines several")
    m.add_argument("-o", "--out", help="where to save (default: <script>.phasor.json)")
    m.add_argument("--code", action="store_true",
                   help="also write the scene as a phasor Python script beside it")  # fmt: skip
    m.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from phasor.importers.manim import ManimImportError, import_manim

    script = Path(args.script)
    try:
        result = import_manim(script, args.scene)
    except (ManimImportError, AttributeError, NameError, FileNotFoundError) as e:
        print(f"phasor manim: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else script.with_suffix(".phasor.json")
    if not out.name.endswith(".phasor.json"):
        out = out.with_name(out.name + ".phasor.json")
    from phasor.engine import Engine

    doc = result.document()
    Engine(doc, passes=None).save(out)
    print(f"{result.name}: {len(doc.objects)} objects, {result.duration:g} s -> {out}")
    for w in result.warnings:
        print(f"  note: {w}")
    if result.others:
        print(f"  (the file also defines {result.others}: pass --scene to import one)")
    if args.code:
        from phasor.core.pycode import document_to_python

        code = out.with_name(out.name.removesuffix(".phasor.json") + ".scene.py")
        code.write_text(document_to_python(doc), encoding="utf-8")
        print(f"  code: {code}")
    issues = [i for i in Engine(doc, passes=None).lint() if i.severity == "error"]
    for i in issues:
        print(f"  lint error: {i.message}")
    return 1 if issues else 0
