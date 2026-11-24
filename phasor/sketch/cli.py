"""``phasor sketch``: strokes in, proposals out (registers through ``phasor.commands``).

Strokes are a JSON file: a list of strokes, each a list of ``[x, y]`` output pixels (row 0
at the top), or ``{"t": seconds, "strokes": [...]}``. Without ``--accept`` it only lists
what it would do; handwriting is saved as a PNG to read, and ``--text K=...`` says what
proposal K's handwriting reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser("sketch", help="turn drawn strokes into scene edits (proposals)")
    m.add_argument("scene", help="a .phasor.json scene")
    m.add_argument("strokes", help="JSON: a list of strokes, each a list of [x, y] pixels")
    m.add_argument("--t", type=float, help="the scene time the strokes were drawn at")
    m.add_argument("--accept", nargs="*", type=int, metavar="K",
                   help="apply proposals K (all of them when no number follows)")  # fmt: skip
    m.add_argument("--text", action="append", default=[], metavar="K=TEXT",
                   help="what handwriting proposal K says (LaTeX in $...$ for maths)")  # fmt: skip
    m.add_argument("--out", help="folder for handwriting pictures (default: beside the scene)")
    m.add_argument("--allow-scripts", action="store_true",
                   help="run the Python code the scene names (meta.modules); it runs only on "
                   "your request, so only your own code")  # fmt: skip
    m.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    from phasor.sketch.propose import Sketch, accept
    from phasor.workspace.tools import ToolError, Workspace

    scene = Path(args.scene).resolve()
    raw = json.loads(Path(args.strokes).read_text(encoding="utf-8"))
    strokes = raw["strokes"] if isinstance(raw, dict) else raw
    t = args.t if args.t is not None else (raw.get("t", 0.0) if isinstance(raw, dict) else 0.0)
    ws = Workspace(scene.parent, allow_scripts=args.allow_scripts)
    try:
        ws.scene_open(scene.name)
    except ToolError as e:
        print(f"phasor sketch: {e}", file=sys.stderr)
        return 1
    sk = Sketch(ws.engine, t)
    props = sk.propose(strokes)
    texts = dict(_pair(x) for x in args.text)
    out = Path(args.out) if args.out else scene.parent
    for k, p in enumerate(props):
        if p.kind == "handwriting":
            png = out / f"{scene.name.removesuffix('.phasor.json')}.handwriting{k}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(p.image or b"")
            if k in texts:
                props[k] = sk.transcribe(p, texts[k])
                p = props[k]
            else:
                p.data["picture"] = str(png)
        print(json.dumps({"k": k, **p.to_dict()}))
    if args.accept is None:
        return 0
    chosen = range(len(props)) if not args.accept else args.accept
    failed = 0
    for k in chosen:
        p = props[k]
        if not p.ops:
            print(f"  {k}: {p.kind}: nothing to apply ({p.summary})", file=sys.stderr)
            failed += 1
            continue
        r = accept(ws, p)
        print(f"  {k}: {'applied' if r.get('ok') else 'refused'}: {p.summary}")
        failed += 0 if r.get("ok") else 1
    ws.scene_save()
    return 1 if failed else 0


def _pair(s: str) -> tuple[int, str]:
    k, _, text = s.partition("=")
    return int(k), text
