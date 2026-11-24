"""``phasor mcp``: registers itself with the CLI through the ``phasor.commands`` entry point."""

from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser("mcp", help="MCP server so an AI can build, check and fix scenes")
    m.add_argument("--root", help="the only folder scenes may be read from or written to")
    m.add_argument("--out", help="where pictures and renders go (default: <root>/phasor_out)")
    m.add_argument(
        "--allow-scripts",
        action="store_true",
        help="let the AI open .py scenes (they are code: only for scenes you trust)",
    )
    m.add_argument(
        "--attach",
        action="store_true",
        help="edit the scene open in a running `phasor gui` (same --root) instead of a "
        "headless one: the human and the AI work on one live scene",
    )
    m.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    try:
        from phasor.mcp.server import serve
    except ModuleNotFoundError as e:
        print(f"phasor mcp: {e}. Install it with: uv sync --extra mcp", file=sys.stderr)
        return 1
    serve(args.root, args.out, args.allow_scripts, args.attach)
    return 0
