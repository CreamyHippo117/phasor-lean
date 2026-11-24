"""The MCP server: ``phasor mcp``. A thin client of ``Engine`` (contract 1).

Every tool is a method of ``Workspace`` (``tools.py``); this file only names them for the
protocol, turns ``ToolError`` into an error the client can read, and hands pictures to
vision models as images.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as ClientError

from phasor.workspace import docs
from phasor.workspace.tools import ToolError, Workspace

INSTRUCTIONS = """\
phasor makes math animations you can see, check and fix. A scene is a Python script that
builds one Scene; it is the source that you and the person both edit. Work in this loop:
1. Start from the closest example (kit/examples/INDEX.md in a kit folder). docs_search and
   object_types tell you exact names, parameters and style roles: look up, do not guess.
2. Write or edit the script. Name objects by meaning (pole_i, loop, loop.label). Style takes
   role names, never raw colors; labels are placed by constraints, never by position.
3. scene_open the script (again after every edit), then review: lint, what is on screen at
   each marker, and a storyboard picture. Fix in the script and review again until clean,
   and look at the picture: lint cannot tell you a layout is ugly.
4. render a preview. Report what you built and anything you could not fix.
Scenes with no script: scene_new, then scene_edit with patches (narrate, animate);
undo, checkpoint, restore and diff work on those."""


TOOL_NAMES = (
    "scene_new", "scene_open", "scene_save", "scene_edit", "scene_palette", "undo", "redo",
    "animate",
    "narrate", "checkpoint", "restore", "diff", "inspect", "lint", "storyboard", "review",
    "critique", "polish", "variants", "choose",
    "events", "when_hidden", "free_space_near", "hit_test", "resolve", "render", "docs_search",
    "object_types",
)  # fmt: skip


def _wrap(fn):
    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError as e:
            raise ClientError(json.dumps({"error": str(e), "issues": e.issues})) from e

    return run


def build_server(workspace: Workspace | None = None) -> MCPServer:
    ws = workspace or Workspace()
    server = MCPServer("phasor", instructions=INSTRUCTIONS)

    def tool(fn):
        server.tool()(_wrap(fn))
        return fn

    for name in TOOL_NAMES:
        if name not in ("storyboard", "review", "variants"):  # these also return a picture, below
            tool(getattr(ws, name))

    # These two return a picture next to the data, so they are plain content, not structured.
    @server.tool(structured_output=False)
    @_wrap
    def storyboard(
        times: list[float] | None = None,
        marks: bool = True,
        columns: int = 3,
        compare: bool = False,
    ) -> list[Any]:
        """Keyframes on one contact sheet (returned as an image) plus what the inspector sees
        in each. ``marks`` draws each object's box and id on the picture."""
        r = ws.storyboard(times, marks, columns, compare)
        parts: list[Any] = [json.dumps(r), Image(path=r["image"])]
        if "compare_image" in r:
            parts.append(Image(path=r["compare_image"]))
        return parts

    @server.tool(structured_output=False)
    @_wrap
    def variants(beat: str | float, n: int = 3) -> list[Any]:
        """Alternative looks of one beat side by side (as is, calmer, bolder) with taste
        scores, as an image. Nothing changes until you call ``choose``."""
        r = ws.variants(beat, n)
        return [json.dumps(r), Image(path=r["image"])]

    @server.tool(structured_output=False)
    @_wrap
    def review(step: float = 0.5, with_picture: bool = True, aesthetic: bool = False) -> list[Any]:
        """The self-review loop in one call: lint, the inspector at every marker and a
        storyboard image. Fix ``todo`` and call again until ``clean`` is true."""
        r = ws.review(step, with_picture, aesthetic)
        parts: list[Any] = [json.dumps(r)]
        if r["storyboard"]:
            parts.append(Image(path=r["storyboard"]))
        return parts

    @server.resource("phasor://docs/{name}")
    def doc(name: str) -> str:
        """The docs written for AI: API, PATTERNS, PITFALLS."""
        return docs.read(name)

    @server.prompt()
    def build_scene(request: str) -> str:
        """Build a scene from a one-line request and check it before reporting."""
        return (
            f"Build this phasor scene: {request}\n\n"
            "Start with object_types and docs_search (read PATTERNS). scene_new, then add "
            "objects with scene_edit, narrate with markers, animate with verbs anchored to "
            "them. Then loop: review, fix every todo, review again (at most 5 rounds). "
            "Save it, render a preview, and report what changed and any issue you left."
        )

    return server


def serve(root: str | Path | None = None, out: str | Path | None = None,
          allow_scripts: bool = False, attach: bool = False) -> None:  # fmt: skip
    """Serve over stdio. ``attach``: edit the scene a running ``phasor gui`` has open (found
    through ``<root>/.phasor/gui.json``) instead of hosting a headless engine."""
    if attach:
        from phasor.workspace.remote import RemoteWorkspace

        build_server(RemoteWorkspace.from_handoff(root)).run("stdio")  # type: ignore[arg-type]
        return
    build_server(Workspace(root, out, allow_scripts=allow_scripts)).run("stdio")
