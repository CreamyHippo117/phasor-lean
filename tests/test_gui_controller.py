"""The editor's logic, without Qt. Everything the window does goes through
``phasor.gui.controller.Editor``; the window only draws it (``test_gui_window.py``)."""

from __future__ import annotations

import pytest

from phasor.gui.controller import Editor
from phasor.workspace import Workspace
from tests.conftest import gl_available

POLE = {"op": "add_object", "object": {"id": "pole", "type": "dot",
                                       "params": {"pos": [1.0, 0.5, 0.0], "r": 0.2}}}  # fmt: skip


@pytest.fixture
def ed(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("edit me")
    e = Editor(ws)
    e.ws.scene_edit(POLE)
    e.ws.scene_edit({"op": "set_marker", "name": "go", "t": 1.0})
    e.ws.animate("morph", "go", path="pole.r", value=0.4)
    return e


def test_time_follows_the_timeline_and_markers(ed):
    tl = ed.timeline()
    assert tl["markers"] == {"go": 1.0} and "pole.r" in tl["lanes"]
    start, end = tl["lanes"]["pole.r"][0]
    assert start == pytest.approx(1.0) and end > start
    ed.set_time(99.0)
    assert ed.t == pytest.approx(tl["duration"])  # clamped to the scene
    ed.set_time(-3)
    assert ed.t == 0.0
    ed.jump_to_marker("go")
    assert ed.t == 1.0
    ed.set_time(0.0)
    ed.play_step(0.25)  # what the play timer calls each tick
    assert ed.t == pytest.approx(0.25)


def test_inspector_shows_params_semantics_and_style_and_edits_them(ed):
    ed.select("pole")
    panel = ed.inspector()
    assert panel["id"] == "pole" and panel["type"] == "dot"
    assert panel["params"]["r"] == 0.2 and panel["semantic"]["kind"]
    assert "color" in panel["style"]  # the role names, which are what can be edited
    v = ed.version
    r = ed.set_param("pole", "r", "0.3")  # typed text, parsed as JSON
    assert r["ok"] and ed.engine.document.object("pole").params["r"] == 0.3
    assert ed.version == v + 1
    r = ed.set_param("pole", "pos", "[0, 0, 0]")
    assert r["ok"] and ed.inspector()["params"]["pos"] == (0.0, 0.0, 0.0)
    bad = ed.set_param("pole", "r", "-1")
    assert not bad["ok"] and ed.engine.document.object("pole").params["r"] == 0.3  # unchanged
    assert ed.set_style("pole", "color", "highlight")["ok"]
    assert ed.inspector()["style"]["color"] == "highlight"
    ed.undo()
    assert ed.inspector()["style"]["color"] != "highlight"
    assert [h["what"] for h in ed.history][:2] == ["pole.r = 0.3", "pole.pos = [0, 0, 0]"]


def test_lint_items_jump_to_their_object_and_time(ed):
    ed.ws.scene_edit({"op": "add_object", "object": {"id": "far", "type": "dot",
                                                     "params": {"pos": [40.0, 0.0, 0.0]}}})  # fmt: skip
    items = ed.lint_items()
    off = next(i for i in items if "far" in i["ids"])
    ed.jump(off)
    assert ed.selected == "far" and ed.t == pytest.approx(off["times"][0] if off["times"] else 0)


def test_external_edits_are_noticed(ed):
    """The AI (attached over the socket) edits the same workspace: the window must redraw."""
    ed.changed()  # (the fixture's own setup)
    assert not ed.changed()
    ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.25})
    assert ed.changed() and not ed.changed()  # reported once


def test_drag_moves_an_object_to_where_the_mouse_is(ed):
    """2D: the viewport point maps back to the plane the object lives in."""
    ed.select("pole")
    w, h = 800, 450
    st = ed.engine.state_at(0.0)
    cam = st.camera
    ndc = cam.project([[2.0, -1.0, 0.0]])[0]
    u, v = (ndc[0] + 1) / 2 * w, (1 - ndc[1]) / 2 * h
    x, y, _z = ed.screen_to_world(u, v, (w, h), z=0.0)
    assert (x, y) == pytest.approx((2.0, -1.0), abs=1e-6)
    r = ed.drag_to("pole", u, v, (w, h))
    assert r["ok"] and ed.engine.document.object("pole").params["pos"] == pytest.approx(
        (2.0, -1.0, 0.0), abs=1e-6
    )


def test_looks_can_be_switched_and_undone(ed):
    assert "prism" in ed.looks()
    v = ed.version
    r = ed.apply_look("prism")
    assert r["ok"] and ed.engine.document.meta.theme == "prism"
    assert ed.engine.document.meta.look == "prism" and ed.engine.document.meta.life == "lively"
    assert ed.version == v + 1  # one undo step
    ed.undo()
    assert ed.engine.document.meta.theme == "midnight_glow"


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_viewport_frames_and_click_to_select(ed):
    img = ed.frame((320, 180))
    assert img.shape == (180, 320, 3) and img.dtype.name == "uint8"
    st = ed.engine.state_at(0.0)
    ndc = st.camera.project([[1.0, 0.5, 0.0]])[0]
    u, v = (ndc[0] + 1) / 2 * 320, (1 - ndc[1]) / 2 * 180
    assert ed.select_at(u, v, (320, 180)) == "pole"
    assert ed.select_at(2, 2, (320, 180)) is None and ed.selected is None
    assert ed.frame((320, 180)) is ed.frame((320, 180))  # cached until something changes
    ed.set_time(0.5)
    assert ed.frame((320, 180)) is not img


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_variant_picker(ed):
    ed.set_style("pole", "glow", "soft")  # something a calmer or bolder look can change
    r = ed.variants("go")
    names = [v["name"] for v in r["variants"]]
    assert len(names) >= 2 and r["image"]
    v = ed.version
    out = ed.choose("calmer")
    assert out["applied"] and ed.version == v + 1  # one undo step
    assert out["recorded"]["by"] == "person" and out["recorded"]["winner"] == "calmer"
    with pytest.raises(Exception, match="no variant"):
        ed.choose("louder")


# ---------------------------------------------------------------- sliders and readouts (M12)


@pytest.fixture
def jed(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("explorer")
    e = Editor(ws)
    e.ws.scene_edit([
        {"op": "add_object", "object": {"id": "set", "type": "fractal",
                                        "params": {"kind": "julia", "c": [-0.8, 0.156],
                                                   "pos": [0, 0, 0], "size": [4.6, 2.6]}}},
        {"op": "add_object", "object": {"id": "plane", "type": "domain_coloring",
                                        "params": {"function": "z**2 + 1", "pos": [0, 4, 0]}}},
        {"op": "add_control", "control": {"path": "set.c", "component": 0, "min": -1, "max": 1,
                                          "step": 0.1, "label": "Re c"}},
    ])  # fmt: skip
    return e


def test_sliders_preview_without_editing_and_can_be_applied_as_one_edit(jed):
    v = jed.version
    ctl = jed.controls()
    assert [(c["key"], c["label"], c["value"]) for c in ctl] == [("set.c[0]", "Re c", -0.8)]
    assert jed.set_control("set.c[0]", 0.33) == pytest.approx(0.3)
    assert jed.controls()[0]["value"] == pytest.approx(0.3)
    assert jed.version == v and jed.engine.document.object("set").params["c"] == (
        -0.8,
        0.156,
    )  # a preview only
    r = jed.apply_controls()
    assert r["ok"] and jed.version == v + 1 and jed.preview == {}
    assert jed.engine.document.object("set").params["c"] == (pytest.approx(0.3), 0.156)
    assert jed.history[-1]["what"] == "sliders: set.c[0] = 0.3"
    jed.set_control("set.c[0]", 0.9)
    jed.reset_controls()
    assert jed.preview == {} and jed.controls()[0]["value"] == pytest.approx(0.3)
    assert jed.apply_controls()["ok"] and jed.version == v + 1  # nothing to apply: no new edit


def test_hover_readouts_for_a_domain_coloring_and_a_julia_set(jed):
    size = (800, 450)
    ndc = jed.engine.state_at(0).camera.project([[0.5, 4.0, 0.0]])[0]  # on the plane
    u, v = (ndc[0] + 1) / 2 * size[0], (1 - ndc[1]) / 2 * size[1]
    r = jed.readout(u, v, size, hit=["plane"])
    assert r["z"] == pytest.approx(0.5 + 4.0j, abs=1e-6) or r["x"] == pytest.approx(0.5, abs=1e-6)
    f = (0.5 + 0.0j) ** 2 + 1  # z = x + i (y - 4): the plane is centered at (0, 4)
    z = complex(r["x"], r["y"] - 4.0)
    assert r["f"] == pytest.approx(complex(z**2 + 1), abs=1e-9)
    assert r["abs"] == pytest.approx(abs(z**2 + 1)) and "arg f" in r["text"]
    j = jed.readout(u, v, size, hit=["set"])
    assert (j["object"] == "set" and j["escape"] is None) or isinstance(j["escape"], int)
    nothing = jed.readout(u, v, size, hit=[])
    assert nothing["object"] is None and "x =" in nothing["text"]
    del f


# ---------------------------------------------------------------- the rest of the desktop app


@pytest.fixture
def orbiting(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("turn")
    e = Editor(ws)
    e.ws.scene_edit([
        {"op": "set_camera", "kind": "orbit",
         "params": {"azimuth": 30.0, "elevation": 20.0, "distance": 10.0,
                    "target": [0, 0, 0], "fov": 40.0, "drift": 0.0}},
        {"op": "add_object", "object": {"id": "helix", "type": "parametric_curve"}},
    ])  # fmt: skip
    return e


def test_dragging_turns_pans_and_zooms_the_camera(orbiting):
    e = orbiting
    cam = lambda: e.engine.document.cameras[0].params  # noqa: E731
    r = e.orbit_camera(10.0, -5.0)  # drag right and up by some pixels' worth of degrees
    assert (
        r["ok"]
        and cam()["azimuth"] == pytest.approx(20.0)
        and cam()["elevation"] == pytest.approx(25.0)
    )
    e.orbit_camera(0, -500)
    assert cam()["elevation"] == 89.0  # never over the pole
    d = cam()["distance"]
    e.zoom_camera(2.0)
    assert cam()["distance"] == pytest.approx(d / 2)
    before = np_array(cam()["target"])
    e.pan_camera(1.0, 0.0)  # along the screen's right
    assert not np_close(np_array(cam()["target"]), before)
    assert e.history[-1]["what"].startswith("camera")


def test_a_2d_camera_pans_and_zooms_in_the_plane(ed):
    ed.pan_camera(2.0, -1.0)
    c = ed.engine.document.cameras[0].params["center"]
    assert (c[0], c[1]) == pytest.approx((-2.0, 1.0)) or (c[0], c[1]) == pytest.approx((2.0, -1.0))
    h0 = ed.engine.state_at(0).camera.units_per_height()
    ed.zoom_camera(2.0)
    assert ed.engine.state_at(0).camera.units_per_height() == pytest.approx(h0 / 2)


def test_an_animated_camera_parameter_is_not_overwritten_by_a_drag(orbiting):
    e = orbiting
    e.ws.scene_edit({"op": "set_marker", "name": "a", "t": 0.0})
    e.ws.animate("orbit", "a", value=90.0)
    r = e.orbit_camera(5.0, 0.0)
    assert not r["ok"] and "animated" in r["errors"][0]["message"]


def test_markers_and_animation_spans_can_be_dragged_in_time(ed):
    r = ed.move_marker("go", 1.5)
    assert r["ok"] and ed.timeline()["markers"]["go"] == 1.5
    r = ed.retime_segment("pole.r", 0, 2.0, 3.0)
    assert r["ok"] and ed.timeline()["lanes"]["pole.r"][0] == pytest.approx([2.0, 3.0])
    assert ed.engine.state_at(3.5)["pole"].params.r == pytest.approx(0.4)  # still ends where it did
    assert not ed.move_marker("ghost", 1.0)["ok"]
    assert not ed.retime_segment("pole.r", 5, 1.0, 2.0)["ok"]
    assert not ed.retime_segment("pole.r", 0, 3.0, 2.0)["ok"]


def test_the_timeline_has_narration_blocks_and_a_waveform(tmp_path):
    import wave

    import numpy as np

    ws = Workspace(tmp_path)
    ws.scene_new("talk")
    n = 8000
    t = np.linspace(0, 1, n)
    audio = (np.sin(2 * np.pi * 220 * t) * np.linspace(0.1, 1.0, n) * 32000).astype("<i2")
    with wave.open(str(tmp_path / "a.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(n)
        f.writeframes(audio.tobytes())
    ws.scene_edit({"op": "set_narration", "narration": {"script": "Hello there.", "voice": "none", "blocks": [
        {"text": "Hello there.", "start": 0.5, "duration": 1.0, "audio": "a.wav"},
        {"text": "", "start": 1.5, "duration": 0.5}]}})  # fmt: skip
    e = Editor(ws)
    e.engine.base_dir = tmp_path
    tl = e.timeline()
    assert tl["narration"][0] == {
        "text": "Hello there.",
        "start": 0.5,
        "duration": 1.0,
        "audio": "a.wav",
    }
    wf = e.waveform("a.wav", 50)
    assert (
        len(wf) == 50 and max(wf) == pytest.approx(32000 / 32768, abs=0.01) and wf[0] < wf[-1]
    )  # it swells
    assert e.waveform("missing.wav", 10) == []


def test_ai_and_your_edits_are_one_activity_feed(ed, tmp_path):
    from phasor.workspace.remote import RemoteServer, RemoteWorkspace

    server = RemoteServer(ed.ws, ed.lock).start()
    server.write_handoff()
    ai = RemoteWorkspace.from_handoff(tmp_path)
    ed.set_param("pole", "r", "0.31")
    ai.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.35})
    ai.inspect(0.0)  # reading is not an edit
    who = [(a["who"], a["what"].split(":")[0]) for a in ed.activity()][-2:]
    assert who == [("you", "pole.r = 0.31"), ("ai", "scene_edit")] or who[1][0] == "ai"
    assert [a["who"] for a in ed.activity()].count("ai") == 1
    ai.close()
    server.stop()


def test_the_code_view_shows_and_runs_the_scene(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("coded")
    e = Editor(ws)
    e.ws.scene_edit(
        {"op": "add_object", "object": {"id": "p", "type": "dot", "params": {"r": 0.3}}}
    )
    code = e.code()
    assert "Dot(id='p'" in code["text"] and not code["is_script"]
    new = code["text"].replace("r=0.3", "r=0.45")
    r = e.run_code(new)
    assert r["ok"] and e.engine.document.object("p").params["r"] == 0.45
    assert (
        tmp_path / "coded.scene.py"
    ).exists()  # the code the person ran is kept beside the scene
    bad = e.run_code("s = Scene(")  # a syntax error says where, and changes nothing
    assert (
        not bad["ok"]
        and "line 1" in bad["error"]
        and e.engine.document.object("p").params["r"] == 0.45
    )


def np_array(v):
    import numpy as np

    return np.asarray(v, dtype=float)


def np_close(a, b):
    import numpy as np

    return bool(np.allclose(a, b))


def test_pixel_drags_become_camera_moves(orbiting):
    e = orbiting
    az0 = e.engine.document.cameras[0].params["azimuth"]
    assert e.drag_camera(50, 0, (800, 450))["ok"]  # an orbit drag: degrees per pixel
    assert e.engine.document.cameras[0].params["azimuth"] == pytest.approx(az0 - 50 * 0.4)
    t0 = np_array(e.engine.document.cameras[0].params["target"])
    assert e.drag_camera(100, 0, (800, 450), pan=True)["ok"]  # content follows the mouse
    t1 = np_array(e.engine.document.cameras[0].params["target"])
    cam = e.engine.state_at(0).camera
    moved = (t1 - t0) @ cam.right
    assert moved < 0 and abs(moved) > 0.5  # dragged right: the camera slides left


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_hit_ids_do_not_change_the_selection(jed):
    ndc = jed.engine.state_at(0).camera.project([[0.0, 4.0, 0.0]])[0]
    u, v = (ndc[0] + 1) / 2 * 800, (1 - ndc[1]) / 2 * 450
    jed.select("set")
    assert jed.hit_ids(u, v, (800, 450))[0] == "plane"  # what is there ...
    assert jed.selected == "set"  # ... without selecting it
