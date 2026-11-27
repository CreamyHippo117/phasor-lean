"""The AI kit (``phasor kit``, ``kit/``) and the docs it relies on stay true to the code.

An AI writing scenes knows phasor only from what the kit puts in front of it: instructions,
docs, examples. The quiet failure is not the AI forgetting but the kit describing a phasor
that no longer exists, so every claim it makes that can be checked is checked here.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from phasor.workspace import kit
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    dest = tmp_path_factory.mktemp("kit")
    kit.install(dest)
    return dest


# ---------------------------------------------------------------- the examples


def test_every_example_is_sorted_into_the_catalogue():
    cat = kit.catalogue()
    on_disk = {p.stem for p in (ROOT / "examples").glob("*.py")}
    assert set(cat["examples"]) == on_disk, "sort new examples into kit/examples.toml"
    shown = [e for e in cat["examples"].values() if e["tier"] != "hidden"]
    assert all(e.get("teaches") for e in shown)
    assert any(e["tier"] == "model" for e in shown)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", [n for n, e in kit.catalogue()["examples"].items() if e["tier"] == "model"]
)
def test_model_examples_pass_lint_with_taste_on_and_strict(name):
    """A model scene is what the AI copies for its look: it must be clean by phasor's own
    standard. A model that fails is either fixed or moved to tier "technique"."""
    from phasor.scene import load

    e = load(ROOT / "examples" / f"{name}.py", passes="auto" if gl_available() else None)
    issues = e.lint(strict=True, aesthetic=True)
    assert not issues, [f"{i.severity}: {i}" for i in issues]


# ---------------------------------------------------------------- installing


def test_the_kit_installs_what_an_ai_needs(installed):
    d = installed
    for f in ("CLAUDE.md", "TASTE.md", "WISHLIST.md", ".mcp.json", ".phasor-kit.json",
              "kit/examples/INDEX.md", "kit/docs/API.md", "kit/docs/PATTERNS.md",
              "kit/docs/PITFALLS.md", ".claude/skills/scene-builder/SKILL.md",
              ".claude/skills/scene-plan/SKILL.md",
              ".claude/skills/phasor/SKILL.md"):  # fmt: skip
        assert (d / f).is_file(), f
    assert (d / "scenes").is_dir() and (d / "out").is_dir()
    claude = (d / "CLAUDE.md").read_text(encoding="utf-8")
    assert "{phasor}" not in claude and "{kit_version}" not in claude
    assert kit.phasor_command() in claude
    assert "@TASTE.md" in claude  # loaded at the start of every session
    index = (d / "kit" / "examples" / "INDEX.md").read_text(encoding="utf-8")
    for name, e in kit.catalogue()["examples"].items():
        if e["tier"] == "hidden":
            assert not list(d.glob(f"kit/examples/*/{name}.py")) and name in index
        else:
            assert (d / "kit" / "examples" / e["kind"] / f"{name}.py").is_file()
            assert f"`{e['kind']}/{name}.py`" in index
    server = json.loads((d / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["phasor"]
    assert server["args"][-4:] == ["mcp", "--root", str(d.resolve()), "--allow-scripts"]


def test_reinstalling_keeps_what_is_yours(tmp_path):
    kit.install(tmp_path)
    (tmp_path / "TASTE.md").write_text("- slower, always\n", encoding="utf-8")
    (tmp_path / "scenes" / "mine.py").write_text("# my scene\n", encoding="utf-8")
    (tmp_path / "kit" / "stale.md").write_text("old\n", encoding="utf-8")
    old_skill = tmp_path / ".claude" / "skills" / "scene"  # renamed scene-builder
    old_skill.mkdir()
    (old_skill / "SKILL.md").write_text("old\n", encoding="utf-8")
    mcp = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    mcp["mcpServers"]["other"] = {"command": "x"}
    (tmp_path / ".mcp.json").write_text(json.dumps(mcp), encoding="utf-8")
    r = kit.install(tmp_path)
    assert "TASTE.md" in r["kept"]
    assert (tmp_path / "TASTE.md").read_text(encoding="utf-8") == "- slower, always\n"
    assert (tmp_path / "scenes" / "mine.py").exists()
    assert not (tmp_path / "kit" / "stale.md").exists()  # the kit's own folder is replaced
    assert not old_skill.exists()
    servers = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert set(servers) == {"phasor", "other"}


# ---------------------------------------------------------------- what the kit tells the AI


def _commands_in(text: str) -> list[list[str]]:
    """Every ``{phasor} ...`` or ``phasor ...`` command line written in ``text``."""
    out = []
    for line in text.splitlines():
        m = re.search(r"(?:\{phasor\}|`phasor)\s+([a-z][^`#]*)", line)
        if m:
            out.append(shlex.split(m.group(1).replace("NAME", "x").replace("SCENE", "x.py")))
    return out


@pytest.mark.parametrize("source", ["kit/CLAUDE.md", "kit/skills/scene-builder/SKILL.md",
                                    "kit/skills/scene-plan/SKILL.md",
                                    "skills/phasor/SKILL.md"])  # fmt: skip
def test_every_command_the_kit_names_exists_with_its_options(source):
    from phasor.cli import build_parser

    text = (ROOT / source).read_text(encoding="utf-8")
    parser = build_parser()
    for words in _commands_in(text):
        try:
            parser.parse_args(words)
        except SystemExit:
            pytest.fail(f"{source}: `phasor {' '.join(words)}` is not a valid command")


# ---------------------------------------------------------------- docs cover the code (C11)


def test_every_public_name_is_in_the_api_docs():
    import phasor

    api = (ROOT / "docs" / "ai" / "API.md").read_text(encoding="utf-8")
    missing = [n for n in phasor.__all__ if not n.startswith("__") and n not in api]
    assert not missing, f"document these in docs/ai/API.md: {missing}"


def test_every_command_is_in_the_api_docs():
    from phasor.cli import build_parser

    api = (ROOT / "docs" / "ai" / "API.md").read_text(encoding="utf-8")
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    missing = [c for c in sub.choices if f"phasor {c}" not in api]
    assert not missing, f"add these to the command table in docs/ai/API.md: {missing}"


def test_every_ai_tool_is_in_the_mcp_docs():
    from phasor.mcp.server import TOOL_NAMES

    mcp = (ROOT / "docs" / "ai" / "MCP.md").read_text(encoding="utf-8")
    missing = [t for t in TOOL_NAMES if f"`{t}`" not in mcp]
    assert not missing, f"document these tools in docs/ai/MCP.md: {missing}"
