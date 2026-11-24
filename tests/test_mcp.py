"""The MCP tools (as plain functions), their safety rules, and the real
protocol over stdio."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from phasor.workspace.tools import ToolError, Workspace
from tests.conftest import gl_available

POLE = {"op": "add_object", "object": {"id": "pole", "type": "dot", "params": {"pos": [1, 1]},
                                       "style": {"color": "highlight"}, "tags": ["key"]}}  # fmt: skip
LABEL = {"op": "add_object", "object": {"id": "pole.label", "type": "text",
                                        "params": {"text": "pole at i", "pos": [1.7, 1.05]}}}  # fmt: skip


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("demo")
    return w


# ---------------------------------------------------------------- editing


def test_edit_adds_objects_and_reports_the_lint_delta(ws):
    r = ws.scene_edit([POLE, LABEL])
    assert r["ok"] and r["objects"] == 2 and "lint_delta" in r
    assert ws.summary()["objects"] == {"pole": "dot", "pole.label": "text"}


def test_bad_patch_changes_nothing_and_says_why(ws):
    ws.scene_edit(POLE)
    bad = {"op": "add_object", "object": {"id": "x", "type": "Dot", "params": {}}}
    r = ws.scene_edit([LABEL, bad])
    assert not r["ok"] and r["errors"][0]["code"] == "unknown_type"
    assert "registered" in r["errors"][0]["message"]
    assert list(ws.summary()["objects"]) == ["pole"]  # the good op in the same patch did not land


def test_style_takes_roles_not_raw_colors(ws):
    r = ws.scene_edit(
        {"op": "add_object", "object": {"id": "d", "type": "dot", "style": {"color": "#ff0000"}}}
    )
    assert not r["ok"]


def test_undo_redo(ws):
    ws.scene_edit(POLE)
    assert ws.undo()["undone"] and ws.summary()["objects"] == {}
    assert ws.redo()["redone"] and "pole" in ws.summary()["objects"]
    ws.undo()
    assert ws.undo() == {"undone": False}  # nothing left to undo


def test_object_types_lists_names_roles_and_defaults(ws):
    t = ws.object_types()
    assert {"dot", "text", "line", "domain_coloring", "fractal"} <= set(t["types"])
    assert "highlight" in t["roles"]["emphasis"] and "midnight_glow" in t["themes"]
    assert t["types"]["dot"]["params"]["r"] == 0.1
    assert t["types"]["line"]["params"]["points"] == "required"


def test_no_scene_open_is_a_clear_error(tmp_path):
    with pytest.raises(ToolError, match="scene_new"):
        Workspace(tmp_path).inspect()


# ---------------------------------------------------------------- narration and verbs


def test_narrate_then_animate_to_a_marker(ws):
    ws.scene_edit([POLE])
    r = ws.narrate("A pole sits here. [[show]] Now look at it.", voice="estimate")
    assert "show" in r["markers"]
    a = ws.animate("reveal", "show", ["pole"])
    assert a["start"] == pytest.approx(r["markers"]["show"]) and a["end"] > a["start"]
    s = ws.animate("morph", "show", path="pole.pos", value=[2, 2], offset=a["end"] - a["start"])
    assert s["start"] >= a["end"] - 1e-9
    # the verb is stored as a verb, anchored to the marker
    seg = ws.engine.document.timeline.tracks["pole.opacity"][0]
    assert seg.verb == "reveal"


def test_animate_mistakes_are_explained(ws):
    ws.scene_edit(POLE)
    with pytest.raises(ToolError, match="unknown verb"):
        ws.animate("explode", 1.0, ["pole"])
    with pytest.raises(ToolError, match="no marker"):
        ws.animate("reveal", "nowhere", ["pole"])
    with pytest.raises(ToolError, match="path and value"):
        ws.animate("morph", 1.0)
    with pytest.raises(ToolError, match="no object"):
        ws.animate("reveal", 1.0, ["ghost"])


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_diff_restore(ws):
    ws.scene_edit(POLE)
    ws.checkpoint("before")
    ws.scene_edit([LABEL, {"op": "set_param", "path": "pole.r", "value": 0.3}])
    d = ws.diff("before", "current")
    kinds = {c["kind"] for c in d["changes"]}
    assert kinds == {"object_added", "param_changed"} and not d["same"]
    ws.restore("before")
    assert list(ws.summary()["objects"]) == ["pole"]
    assert ws.diff("before", "current")["same"]
    with pytest.raises(ToolError, match="no checkpoint"):
        ws.restore("nope")
    with pytest.raises(ToolError, match="names are"):
        ws.checkpoint("bad name!")


def test_diff_against_saved(ws):
    ws.scene_edit(POLE)
    ws.scene_save("demo.phasor.json")
    ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.4})
    assert ws.diff("saved", "current")["changes"][0]["path"] == "pole.r"


# ---------------------------------------------------------------- files and safety


def test_save_and_reopen_round_trip(ws):
    ws.scene_edit([POLE, LABEL])
    out = ws.scene_save("a/demo.phasor.json")
    assert Path(out["saved"]).exists()
    w2 = Workspace(ws.root)
    assert w2.scene_open("a/demo.phasor.json")["objects"] == ws.summary()["objects"]


def test_paths_cannot_leave_the_workspace(ws, tmp_path):
    outside = tmp_path.parent / "elsewhere.phasor.json"
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.scene_save(str(outside))
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.scene_open("../../etc/passwd")
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.render(out="/tmp/x.mp4")


def test_scripts_are_refused_unless_allowed(tmp_path):
    (tmp_path / "evil.py").write_text("raise SystemExit('ran')\n", encoding="utf-8")
    with pytest.raises(ToolError, match="does not run scripts"):
        Workspace(tmp_path).scene_open("evil.py")


def test_a_script_scene_keeps_its_structure_and_takes_tweaks_as_overrides(tmp_path):
    (tmp_path / "s.py").write_text(
        "from phasor import Dot, Scene\ns = Scene('s')\ns.add(Dot(id='d', pos=(0, 0)))\n",
        encoding="utf-8",
    )
    w = Workspace(tmp_path, allow_scripts=True)
    assert w.scene_open("s.py")["source"] == "script"
    assert w.scene_edit({"op": "set_param", "path": "d.r", "value": 0.5})["ok"]
    assert not w.scene_edit({"op": "remove_object", "id": "d"})["ok"]
    with pytest.raises(ToolError, match="built by a script"):
        w.animate("reveal", 1.0, ["d"])
    w.scene_save()
    assert (tmp_path / "s.overrides.json").exists()


# ---------------------------------------------------------------- seeing


def test_inspect_detail_levels_and_hit_test(ws):
    ws.scene_edit([POLE])
    brief = ws.inspect(0.0, detail="brief")
    assert brief["objects"][0].startswith("pole [dot]")
    assert ws.inspect(0.0, id="pole")["objects"][0]["id"] == "pole"
    with pytest.raises(ToolError, match="no object"):
        ws.inspect(0.0, id="ghost")
    with pytest.raises(ToolError, match="detail"):
        ws.inspect(0.0, detail="huge")


def test_resolve_finds_objects_from_words(ws):
    ws.scene_edit([POLE, LABEL])
    ws.scene_edit({"op": "add_object", "object": {"id": "loop", "type": "line",
                                                  "params": {"points": [[-1, -1], [1, -1]]}}})  # fmt: skip
    assert ws.resolve("the label of the pole")["matches"][0]["id"] == "pole.label"
    assert ws.resolve("the pole")["matches"][0]["id"] == "pole"
    assert ws.resolve("the line")["matches"][0]["id"] == "loop"
    assert ws.resolve("the thing that says pole at i")["matches"][0]["id"] == "pole.label"
    assert ws.resolve("banana")["matches"] == []


def test_docs_search_finds_the_verbs_and_the_pitfalls(ws):
    hit = ws.docs_search("dim everything but one object")["results"]
    assert hit and any("dim_others" in h["text"] for h in hit)
    assert ws.docs_search("zzzqqq")["results"] == []


@pytest.mark.parametrize("passes", [None, pytest.param("auto", marks=pytest.mark.gl)])
def test_lint_fix_review_loop_on_a_cluttered_scene(tmp_path, passes):
    w = Workspace(tmp_path)
    w.scene_new("loop")
    w._engine._passes = passes
    ops = [{"op": "set_param", "path": "camera.height", "value": 4}]
    for k, x in enumerate((-0.3, 0.0, 0.3)):
        ops.append({"op": "add_object", "object": {"id": f"p{k}".replace("p", "pole_"), "type": "dot",
                    "params": {"pos": [x, 0], "r": 0.12}, "tags": ["key"]}})  # fmt: skip
        ops.append({"op": "add_object", "object": {"id": f"pole_{k}.label", "type": "text",
                    "params": {"text": f"pole number {k}", "pos": [x + 0.2, 0.05]}}})  # fmt: skip
    assert w.scene_edit(ops)["ok"]
    first = w.lint()
    assert not first["clean"] and {i["code"] for i in first["issues"]} & {"overlap", "clutter"}
    fixed = w.lint(fix=True)
    assert fixed["applied"] and len(fixed["issues"]) < len(first["issues"])
    r = w.review(with_picture=False)
    assert set(r) >= {"clean", "todo", "frames", "next"} and r["frames"][0]["objects"]
    if r["clean"]:
        assert r["next"].startswith("done")
    else:
        assert r["todo"] and r["next"].startswith("fix")


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_storyboard_and_render_write_files(ws):
    ws.scene_edit([POLE, LABEL])
    ws.narrate("A pole. [[go]] Done.", voice="estimate")
    ws.animate("reveal", "go", ["pole"])
    sb = ws.storyboard(marks=True)
    assert Path(sb["image"]).exists() and sb["frames"]
    r = ws.render(frame=0.5)
    assert Path(r["path"]).exists() and r["frames"] == 1
    with pytest.raises(ToolError, match="preset"):
        ws.render(preset="huge")


# ---------------------------------------------------------------- the protocol itself


async def _session(tmp_path, script):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "phasor.cli", "mcp", "--root", str(tmp_path)],
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        return await script(session)


def _data(result):
    for part in result.content:
        if getattr(part, "type", "") == "text":
            return json.loads(part.text)
    raise AssertionError("no text part")


def test_the_real_protocol_lists_the_spec_tools_and_runs_a_session(tmp_path):
    async def script(s):
        tools = {t.name for t in (await s.list_tools()).tools}
        out = {"tools": tools}
        out["new"] = _data(await s.call_tool("scene_new", {"title": "wire"}))
        out["edit"] = _data(await s.call_tool("scene_edit", {"patch": [POLE, LABEL]}))
        out["bad"] = await s.call_tool("inspect", {"t": 0.0, "id": "ghost"})
        out["insp"] = _data(await s.call_tool("inspect", {"t": 0.0, "detail": "brief"}))
        prompts = (await s.list_prompts()).prompts
        out["prompts"] = [p.name for p in prompts]
        return out

    out = asyncio.run(_session(tmp_path, script))
    assert {
        "scene_open", "scene_save", "scene_edit", "inspect", "lint", "storyboard", "hit_test",
        "resolve", "diff", "render", "docs_search",
    } <= out["tools"]  # fmt: skip
    assert out["edit"]["ok"] and out["insp"]["objects"]
    assert out["bad"].is_error and "no object" in out["bad"].content[0].text
    assert "build_scene" in out["prompts"]


# ---------------------------------------------------------------- the packaged knowledge


def test_llms_txt_is_current():
    from phasor.mcp import llms

    assert llms.PATH.read_text(encoding="utf-8") == llms.build(), (
        "llms.txt is stale: uv run python -m phasor.mcp.llms"
    )


def test_every_tool_is_documented_in_mcp_md_and_the_skill():
    from phasor.mcp.server import TOOL_NAMES

    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "ai" / "MCP.md").read_text(encoding="utf-8")
    skill = (root / "skills" / "phasor" / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: phasor\ndescription: ")
    for name in TOOL_NAMES:
        assert f"`{name}`" in doc, f"docs/ai/MCP.md does not mention {name}"


def test_server_exposes_exactly_the_documented_tools():
    from phasor.mcp.server import TOOL_NAMES, build_server

    tools = {t.name for t in asyncio.run(build_server(Workspace()).list_tools())}
    assert tools == set(TOOL_NAMES)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_pictures_arrive_as_images_over_the_protocol(tmp_path):
    async def script(s):
        await s.call_tool("scene_new", {"title": "pic"})
        await s.call_tool("scene_edit", {"patch": [POLE, LABEL]})
        return {
            "review": await s.call_tool("review", {}),
            "board": await s.call_tool("storyboard", {"times": [0.0]}),
        }

    out = asyncio.run(_session(tmp_path, script))
    for key in ("review", "board"):
        res = out[key]
        assert not res.is_error, res.content[0].text
        assert [p.type for p in res.content] == ["text", "image"]
        assert res.content[1].mime_type == "image/png"
    assert json.loads(out["review"].content[0].text)["frames"]


def test_long_time_lists_are_trimmed_for_clients(ws):
    ws.scene_edit({"op": "add_object", "object": {"id": "t", "type": "text", "layer": "overlay",
                                                  "params": {"text": "hello"}}})  # fmt: skip
    ws.engine.apply({"op": "set_marker", "name": "end", "t": 20.0}, origin="script")
    issue = next(i for i in ws.lint()["issues"] if i["code"] == "offscreen")
    assert len(issue["times"]) == 6 and issue["times_total"] > 6


# ---------------------------------------------------------------- the exit demo


def test_exit_demo_an_ai_builds_and_fixes_a_scene_from_a_one_line_prompt(tmp_path):
    """Milestone 5 exit demo. The prompt was one line: "Explain the residue of 1/(z^2+1) at
    i in 15 seconds". `examples/ai_session/session.json` is the tool calls an AI (Claude) made
    to `phasor mcp` to build it, look at the storyboard, spot problems and fix them. This
    replays them over the real protocol and checks what the session had to achieve."""
    import importlib.util

    root = Path(__file__).resolve().parents[1] / "examples" / "ai_session"
    spec = importlib.util.spec_from_file_location("replay", root / "replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    calls = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if not gl_available():  # pictures and renders need OpenGL; the rest of the session does not
        calls = [c for c in calls if c["tool"] != "render"]
    steps = asyncio.run(replay.replay(calls, str(tmp_path)))
    assert not [s for s in steps if s["error"]], [s for s in steps if s["error"]]
    by = [s for s in steps if s["tool"] == "review"]
    first, last = by[0]["data"], by[-1]["data"]
    assert last["clean"] and last["todo"] == [] and last["frames"]
    # the loop of edits is visible in the lint deltas: a mistake appeared, then was fixed
    deltas = [s["data"]["lint_delta"] for s in steps if s["tool"] == "scene_edit"]
    assert any(i["code"] == "offscreen" for d in deltas for i in d["added"])
    assert any(i["code"] == "offscreen" for d in deltas for i in d["resolved"])
    # the repair is a small, structured diff against the checkpoint taken before it
    diff = next(s["data"] for s in steps if s["tool"] == "diff")
    kinds = {c["kind"] for c in diff["changes"]}
    assert {"object_added", "object_removed", "param_changed"} <= kinds
    # it is a narrated scene: every motion anchored to a marker (strict lint is what review ran)
    scene = Workspace(tmp_path)
    summary = scene.scene_open("residue_ai.phasor.json")
    assert set(summary["markers"]) == {"intro", "pole_top", "pole_bottom", "loop_draw", "integral"}
    assert scene.lint(strict=True)["clean"]
    if gl_available():
        assert (tmp_path / "residue_ai.png").exists()
        assert by[-1]["images"] == 1  # the storyboard came back as an image
    assert first["frames"][0]["objects"]


def test_animate_orbits_the_3d_camera(ws):
    ws.scene_edit([{"op": "set_camera", "kind": "orbit", "params": {"azimuth": 0, "drift": 0}},
                   POLE, LABEL])  # fmt: skip
    ws.engine.apply({"op": "set_marker", "name": "stop", "t": 6.0}, origin="script")
    a = ws.animate("orbit", 1.0, value=90, until="stop")
    assert a["start"] == 1.0 and a["end"] == pytest.approx(6.0)
    assert ws.engine.state_at(6.0).values["camera.azimuth"] == pytest.approx(90)
    r = ws.animate("orbit_reveal", 6.0, ["pole"])
    assert r["end"] > 6.0


@pytest.mark.parametrize("title", ["/tmp/escaped", "../../escaped", "..", "a/b"])
def test_a_title_cannot_write_outside_the_workspace(tmp_path, title):
    from phasor.core.ids import file_stem

    stem = file_stem(title)
    assert "/" not in stem and not stem.startswith(".") and stem
    w = Workspace(tmp_path / "root")
    w.scene_new("ok")
    assert w.scene_edit({"op": "set_meta", "field": "title", "value": title})["ok"]
    if gl_available():  # really write one and see where it lands
        path = Path(w.storyboard(times=[0.0])["image"]).resolve()
        assert w.out_dir.resolve() in path.parents


def test_narration_audio_must_stay_in_the_scene_folder(tmp_path):
    from phasor.core.schema import Narration, NarrationBlock
    from phasor.time.narration import mix_track

    n = Narration(script="x", voice="estimate",
                  blocks=(NarrationBlock(text="x", start=0, duration=1, audio="../secret.wav"),))  # fmt: skip
    with pytest.raises(ValueError, match="outside the scene"):
        mix_track(n, tmp_path, 1.0)


def test_restore_is_one_undoable_step_and_versions_only_grow(ws):
    ws.scene_edit(POLE)
    ws.checkpoint("one")
    ws.scene_edit(LABEL)
    before = ws.engine.version
    r = ws.restore("one")
    assert r["version"] > before and list(ws.summary()["objects"]) == ["pole"]
    ws.undo()
    assert "pole.label" in ws.summary()["objects"]


# ---------------------------------------------------------------- Milestone 7: taste tools


def test_scene_new_takes_a_look_and_presets_in_patches(tmp_path):
    w = Workspace(tmp_path)
    assert w.scene_new("lk")["theme"] == "midnight_glow"  # dark by default
    s = w.scene_new("lk", look="calm")
    assert s["theme"] == "midnight_glow"
    assert w.engine.document.meta.life == "subtle" and w.engine.document.meta.look == "calm"
    r = w.scene_edit({"op": "add_object", "object": {"id": "p", "type": "domain_coloring",
                      "preset": "soft_phase", "params": {"function": "1/(z**2 + 1)"}}})  # fmt: skip
    assert r["ok"], r["errors"]
    obj = w.engine.document.object("p")
    assert obj.style["colormap"] == "soft_phase" and obj.params["softness"] == 0.8
    chalk = w.scene_new("c", look="chalk")
    assert chalk["theme"] == "paper"
    kinds = w.object_types()
    assert "soft_phase" in kinds["types"]["domain_coloring"]["presets"]
    assert {"calm", "glow", "chalk"} <= set(kinds["looks"])


def test_an_unknown_preset_is_a_clear_error(tmp_path):
    w = Workspace(tmp_path)
    w.scene_new("x")
    with pytest.raises(ToolError, match="no preset 'nope'"):
        w.scene_edit({"op": "add_object", "object": {"id": "q", "type": "dot", "preset": "nope"}})


def test_critique_polish_and_the_rubric(tmp_path):
    (tmp_path / "STYLE.md").write_text("# my taste\none hero per beat\n", encoding="utf-8")
    w = Workspace(tmp_path)
    w.scene_new("rough")
    w.scene_edit(
        [
            {"op": "set_param", "path": "camera.height", "value": 6},
            {
                "op": "add_object",
                "object": {
                    "id": "a",
                    "type": "dot",
                    "params": {"pos": [5.2, 0]},
                    "style": {"color": "highlight"},
                },
            },
            {
                "op": "add_object",
                "object": {
                    "id": "b",
                    "type": "dot",
                    "params": {"pos": [0, 0]},
                    "style": {"color": "highlight"},
                },
            },
            {"op": "set_marker", "name": "beat", "t": 1.0},
        ]
    )
    c = w.critique()
    assert c["score"] > 0 and "one hero per beat" in c["rubric"] and c["gallery"] == 0
    assert {"focal_point", "margin"} <= {i["code"] for i in c["issues"]}
    p = w.polish()
    assert p["after"] < p["before"] and p["rounds"]


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_variants_choose_and_the_choice_is_recorded(tmp_path):
    from phasor.style import choices

    w = Workspace(tmp_path)
    w.scene_new("vv", look="glow")
    w.scene_edit(
        [
            {
                "op": "add_object",
                "object": {"id": "d", "type": "dot", "params": {"r": 1.0}, "preset": "beacon"},
            },
            {"op": "set_marker", "name": "beat", "t": 1.0},
        ]
    )
    r = w.variants("beat")
    assert Path(r["image"]).exists() and [v["name"] for v in r["variants"]] == [
        "as_is",
        "calmer",
        "bolder",
    ]
    out = w.choose("calmer", note="less shouty")
    assert out["applied"] and w.engine.document.object("d").style["glow"] == "soft"
    log = choices.read(tmp_path / "style" / "choices.jsonl")
    assert log[0]["winner"] == "calmer" and log[0]["options"] == ["as_is", "calmer", "bolder"]
    with pytest.raises(ToolError, match="no variant 'neon'"):
        w.choose("neon")
    with pytest.raises(ToolError, match="call variants first"):
        Workspace(tmp_path).choose("calmer")


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_storyboard_compare_over_the_workspace(tmp_path):
    import numpy as np
    from PIL import Image

    (tmp_path / "gallery").mkdir()
    Image.fromarray(np.full((90, 160, 3), 40, dtype=np.uint8)).save(
        tmp_path / "gallery" / "dark.png"
    )
    (tmp_path / "gallery" / "dark.txt").write_text("calm and dark\n", encoding="utf-8")
    w = Workspace(tmp_path)
    w.scene_new("cmp")
    w.scene_edit({"op": "add_object", "object": {"id": "d", "type": "dot", "params": {"r": 1.0}}})
    r = w.storyboard(times=[0.0], compare=True)
    assert Path(r["compare_image"]).exists() and r["gallery_size"] == 1
    assert r["compare"][0]["nearest"][0]["note"] == "calm and dark"
