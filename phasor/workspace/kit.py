"""The AI kit: a folder where an AI writes phasor scenes, and knows how (``phasor kit``).

It holds only what an AI needs to write scenes: instructions (``CLAUDE.md``, loaded at the
start of every session), the AI docs, the example scenes sorted by kind and tier
(``kit/examples.toml``), the ``/scene`` command, the person's ``TASTE.md`` and the phasor
tools (``.mcp.json``). One source in the repository (``kit/``, ``docs/ai/``,
``examples/``, ``skills/``); installing copies it, so the kit never drifts from the code it
describes (``tests/test_kit.py``).

What the kit owns is replaced on every install: ``kit/``, ``CLAUDE.md``, the skills under
``.claude/skills/``, the ``phasor`` entry of ``.mcp.json``. What the person owns is never
touched: ``TASTE.md``, ``WISHLIST.md``, ``scenes/``, other MCP servers.
"""

from __future__ import annotations

import datetime
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from phasor.workspace.docs import docs_root
from phasor.workspace.paths import cli_command, examples_dir

TIERS = ("model", "technique", "reference", "hidden")
PERSON_FILES = ("TASTE.md", "WISHLIST.md")
SKILLS = {"scene": ("kit", "skills", "scene"), "phasor": ("skills", "phasor")}


class KitError(RuntimeError):
    pass


def _root() -> Path:
    root = docs_root()
    if root is None or not (root / "kit").is_dir():
        raise KitError("the kit is not part of this phasor installation")
    return root


def catalogue() -> dict[str, Any]:
    """``kit/examples.toml``: ``{"kinds": {kind: description}, "examples": {name: {...}}}``,
    checked: every entry has a known tier, every shown one a known kind."""
    data = tomllib.loads((_root() / "kit" / "examples.toml").read_text(encoding="utf-8"))
    kinds, examples = data.get("kinds", {}), data.get("examples", {})
    for name, e in examples.items():
        if e.get("tier") not in TIERS:
            raise KitError(f"example {name}: tier must be one of {TIERS}, got {e.get('tier')!r}")
        if e["tier"] != "hidden" and e.get("kind") not in kinds:
            raise KitError(f"example {name}: unknown kind {e.get('kind')!r}")
    return {"kinds": kinds, "examples": examples}


def phasor_command() -> str:
    """How to run the phasor command line on this machine, as one shell word or two: this
    Python with ``-m phasor``, or the packaged app's ``--cli``. Forward slashes, so it works
    in both Bash and PowerShell (quoted only when the path has spaces)."""
    exe, *rest = cli_command()
    path = Path(exe).as_posix()
    return " ".join([f'"{path}"' if " " in path else path, *rest])


def index_markdown(cat: dict[str, Any]) -> str:
    """``kit/examples/INDEX.md``: the shown examples by kind, model scenes first."""
    order = {t: i for i, t in enumerate(TIERS)}
    lines = [
        "# Example scenes",
        "",
        "Start every scene from the closest one. **model**: the look and pacing to aim for "
        "(lint-clean with the taste checks on). **technique**: how to do something; copy the "
        "technique, not the look. **reference**: which objects exist and how to call them.",
        "",
        "For now these are placeholders (phasor's own test scenes); great examples are on the "
        "way. Until then `TASTE.md` and `kit/docs/STYLE.md` outrank them on how things "
        "should look.",
        "",
    ]
    for kind, about in cat["kinds"].items():
        shown = sorted(((n, e) for n, e in cat["examples"].items()
                        if e["tier"] != "hidden" and e["kind"] == kind),
                       key=lambda ne: (order[ne[1]["tier"]], ne[0]))  # fmt: skip
        if not shown:
            continue
        lines += [f"## {about}", "", "| Scene | Tier | What it teaches |", "| --- | --- | --- |"]
        lines += [f"| `{kind}/{n}.py` | {e['tier']} | {e['teaches']} |" for n, e in shown]
        lines.append("")
    hidden = [n for n, e in cat["examples"].items() if e["tier"] == "hidden"]
    lines += ["Not for you (not copied here): " + ", ".join(sorted(hidden)) + ".", ""]
    return "\n".join(lines)


def install(dest: str | Path) -> dict[str, Any]:
    """Make or update the kit in ``dest``. Returns ``{"dest", "written", "kept"}``."""
    from phasor import __version__

    root, dest = _root(), Path(dest).resolve()
    cat = catalogue()
    ex_dir = examples_dir()
    if ex_dir is None:
        raise KitError("the example scenes are not part of this phasor installation")
    written: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)

    kit = dest / "kit"
    if kit.exists():
        shutil.rmtree(kit)
    docs = kit / "docs"
    docs.mkdir(parents=True)
    for f in sorted((root / "docs" / "ai").glob("*.md")):
        shutil.copyfile(f, docs / f.name)
    if (root / "STYLE.md").is_file():
        shutil.copyfile(root / "STYLE.md", docs / "STYLE.md")
    for name, e in cat["examples"].items():
        if e["tier"] == "hidden":
            continue
        (kit / "examples" / e["kind"]).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ex_dir / f"{name}.py", kit / "examples" / e["kind"] / f"{name}.py")
    (kit / "examples" / "INDEX.md").write_text(index_markdown(cat), encoding="utf-8")
    written.append("kit/")

    stamp = f"phasor {__version__}, {datetime.date.today().isoformat()}"
    text = (root / "kit" / "CLAUDE.md").read_text(encoding="utf-8")
    text = text.replace("{phasor}", phasor_command()).replace("{kit_version}", stamp)
    (dest / "CLAUDE.md").write_text(text, encoding="utf-8")
    written.append("CLAUDE.md")

    for name, parts in SKILLS.items():
        target = dest / ".claude" / "skills" / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root.joinpath(*parts, "SKILL.md"), target / "SKILL.md")
        written.append(f".claude/skills/{name}/SKILL.md")

    kept = []
    for name in PERSON_FILES:
        if (dest / name).exists():
            kept.append(name)
        else:
            shutil.copyfile(root / "kit" / name, dest / name)
            written.append(name)
    for folder in ("scenes", "out"):
        (dest / folder).mkdir(exist_ok=True)

    mcp_path = dest / ".mcp.json"
    config = json.loads(mcp_path.read_text(encoding="utf-8")) if mcp_path.exists() else {}
    exe, *rest = cli_command()
    config.setdefault("mcpServers", {})["phasor"] = {
        "command": exe,
        "args": [*rest, "mcp", "--root", str(dest), "--allow-scripts"],
    }
    mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    written.append(".mcp.json")

    manifest = {"phasor": __version__, "installed": stamp, "command": phasor_command()}
    (dest / ".phasor-kit.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")  # fmt: skip
    return {"dest": str(dest), "written": written, "kept": kept}
