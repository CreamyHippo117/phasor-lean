"""The Qt window, driven headless (Qt's offscreen platform). The logic is
tested in ``test_gui_controller.py``; here: the widgets show it and forward input to it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from phasor.gui.window import open_editor  # noqa: E402
from phasor.workspace.remote import RemoteWorkspace  # noqa: E402
from tests.conftest import gl_available  # noqa: E402

SCENE = {
    "schema_version": 6,
    "meta": {"title": "window test"},
    "objects": [
        {"id": "pole", "type": "dot", "params": {"pos": [1.0, 0.5, 0.0], "r": 0.25}},
        {"id": "far", "type": "dot", "params": {"pos": [40.0, 0.0, 0.0]}},
    ],
    "timeline": {"markers": {"go": 1.0, "end": 2.0}},
}


@pytest.fixture
def window(tmp_path):
    import json

    path = tmp_path / "w.phasor.json"
    path.write_text(json.dumps(SCENE), encoding="utf-8")
    app, win, server = open_editor(str(path), show=False)
    win.resize(1000, 640)
    yield app, win, server, tmp_path
    win.timer.stop()
    server.stop()
    win.close()


def test_the_panels_show_the_scene(window):
    app, win, _server, _root = window
    app.processEvents()
    assert win.windowTitle() == "phasor: window test"
    assert set(win.timeline.data["markers"]) == {"go", "end"}
    texts = [win.lint.item(i).text() for i in range(win.lint.count())]
    assert any("far" in t or "off_screen" in t or "offscreen" in t for t in texts), texts
    assert "phasor mcp --attach" in win.statusBar().currentMessage()
    assert win.inspector.title.text().startswith("Nothing selected")
    assert win.looks.look.count() >= 4 and win.looks.beat.count() == 2


def test_typing_in_the_inspector_edits_the_scene(window):
    _app, win, _server, _root = window
    win.ed.select("pole")
    win.inspector.refresh()
    field = win.inspector.fields["r"]
    field.setText("0.4")
    field.editingFinished.emit()
    assert win.ed.engine.document.object("pole").params["r"] == 0.4
    win.inspector.fields["r"].setText("-5")  # refused: nothing changes, the reason is shown
    win.inspector.fields["r"].editingFinished.emit()
    assert win.ed.engine.document.object("pole").params["r"] == 0.4
    assert "Not applied" in win.statusBar().currentMessage()
    win.inspector.roles["color"].setCurrentText("highlight")
    assert win.ed.engine.state_at(0.0)["pole"].roles["color"] == "highlight"


def test_clicking_an_issue_jumps_to_it(window):
    _app, win, _server, _root = window
    row = next(i for i in range(win.lint.count())
               if "far" in str(win.lint.item(i).data(Qt.ItemDataRole.UserRole)))  # fmt: skip
    win.lint.itemClicked.emit(win.lint.item(row))
    assert win.ed.selected == "far"
    assert win.inspector.title.text().startswith("far")


def test_scrubbing_and_playing_move_the_playhead(window):
    _app, win, _server, _root = window
    win.timeline.scrubbed.emit(1.5)
    assert win.ed.t == pytest.approx(1.5)
    win.ed.set_time(0.0)
    win.toggle_play()
    for _ in range(6):
        win.tick()
    assert win.ed.t > 0.15 and win.playing


def test_an_attached_ai_edit_shows_up_in_the_open_window(window):
    """The exit demo's other half: the AI edits through the socket; the window redraws."""
    _app, win, _server, root = window
    win.ed.select("pole")
    win.refresh_all()
    remote = RemoteWorkspace.from_handoff(root)
    r = remote.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.33})
    assert r["ok"]
    win.tick()  # the timer notices the new version
    assert win.inspector.fields["r"].text() == "0.33"
    remote.close()


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_viewport_shows_the_picture_and_selects_by_click(window):
    _app, win, _server, _root = window
    win.viewport.resize(640, 360)
    win.viewport.refresh()
    assert win.viewport.error is None and not win.viewport.pixmap().isNull()
    fw, fh = win.viewport._size
    cam = win.ed.engine.state_at(0.0).camera
    ndc = cam.project([[1.0, 0.5, 0.0]])[0]
    u = (ndc[0] + 1) / 2 * fw + (win.viewport.width() - fw) / 2
    v = (1 - ndc[1]) / 2 * fh + (win.viewport.height() - fh) / 2
    for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
        ev = QMouseEvent(kind, QPointF(u, v), QPointF(u, v), Qt.MouseButton.LeftButton,
                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)  # fmt: skip
        (win.viewport.mousePressEvent if kind == QMouseEvent.Type.MouseButtonPress
         else win.viewport.mouseReleaseEvent)(ev)  # fmt: skip
    assert win.ed.selected == "pole"
    assert win.inspector.title.text().startswith("pole")


# ---------------------------------------------------------------- the rest of the desktop app


@pytest.fixture
def julia_window(tmp_path):
    import shutil

    shutil.copy(Path(__file__).parents[1] / "examples" / "julia_explorer.py", tmp_path / "j.py")
    app, win, server = open_editor(str(tmp_path / "j.py"), show=False)
    win.resize(1100, 700)
    yield app, win, server, tmp_path
    win.timer.stop()
    server.stop()
    win.close()


def test_the_sliders_tab_previews_and_applies(julia_window):
    _app, win, _server, _root = julia_window
    assert set(win.controls.sliders) == {"set.c[0]", "set.c[1]", "set.max_iter"}
    v = win.ed.version
    s = win.controls.sliders["set.c[0]"]
    s.setValue(s.maximum())  # all the way right
    assert win.ed.preview["set.c[0]"] == pytest.approx(1.2)
    assert win.controls.reads["set.c[0]"].text() == "1.2"
    assert win.ed.version == v and win._preview_dirty  # a preview: the scene is untouched
    win.controls.applied.emit(win.ed.apply_controls())
    assert win.ed.version == v + 1 and win.ed.engine.document.object("set").params["c"][0] == 1.2
    win.refresh_all()  # rebuilt from the scene: the slider sits where the value now is
    assert win.controls.sliders["set.c[0]"].value() == win.controls.RES


def test_the_code_tab_runs_edited_code(julia_window):
    _app, win, _server, root = julia_window
    assert win.code.text.toPlainText().startswith('"""A Julia-set explorer')
    text = win.code.text.toPlainText().replace("max_iter=120", "max_iter=90")
    win.code.text.setPlainText(text)
    assert win.code.edited
    win.code.run()
    assert "ran:" in win.code.status.text()
    assert win.ed.engine.document.object("set").params["max_iter"] == 90
    assert "max_iter=90" in (root / "j.py").read_text(
        encoding="utf-8"
    )  # the script on disk is the code
    win.code.text.setPlainText("s = Scene(")
    win.code.run()
    assert win.code.status.text().startswith("not run: line 1")
    assert win.ed.engine.document.object("set").params["max_iter"] == 90  # nothing changed


def test_dragging_the_timeline_moves_markers_and_spans(window):
    _app, win, _server, _root = window
    win.ed.ws.scene_edit({"op": "add_object", "object": {"id": "x", "type": "dot"}})
    win.ed.ws.scene_edit({"op": "set_marker", "name": "mid", "t": 1.4})
    win.ed.ws.animate("morph", "mid", path="x.r", value=0.5)
    win.refresh_all()
    tl = win.timeline
    tl.resize(900, 200)
    tl.grab()  # paint once, so the timeline knows where its markers and spans are
    # the marker line "mid": press on it and drag right by 100 px
    x = tl._x(1.4)

    def ev(kind, px):
        return QMouseEvent(
            kind,
            QPointF(px, 40),
            QPointF(px, 40),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    tl.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, x))
    tl.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, x + 100))
    tl.mouseReleaseEvent(ev(QMouseEvent.Type.MouseButtonRelease, x + 100))
    t = win.ed.timeline()["markers"]["mid"]
    assert t == pytest.approx(tl._t(x + 100), abs=1e-6) and t > 1.4


def test_camera_drag_and_hover_readout_in_the_window(window):
    _app, win, _server, _root = window
    win.viewport.resize(640, 360)
    win.viewport.refresh()
    size = win.viewport._size
    if size[0]:
        c0 = win.ed.engine.document.cameras[0].params.get("center", (0.0, 0.0, 0.0))
        win.viewport.turned.emit(30.0, 0.0, size, False)  # in 2D a drag slides the picture
        assert win.ed.engine.document.cameras[0].params["center"] != c0
        win.viewport.zoomed.emit(2.0)
    win._hover = (10.0, 10.0, (640, 360))
    win._show_readout()
    assert win.readout.text().startswith("x = ")


def test_the_activity_dock_lists_your_edits_and_the_ais(window):
    _app, win, _server, root = window
    win.ed.set_param("pole", "r", "0.3")
    remote = RemoteWorkspace.from_handoff(root)
    remote.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.31})
    win.tick()
    items = [win.activity.item(i).text() for i in range(win.activity.count())]
    assert any(t.startswith("you: pole.r") for t in items) and any(
        t.startswith("AI:") for t in items
    )
    remote.close()


# ---------------------------------------------------------------- Milestone 11: the Sketch tab


def test_the_sketch_tab_turns_a_drawn_loop_into_a_traced_contour(tmp_path):
    """Draw on the viewport with the Sketch tab's pen; the loop around i becomes a contour
    with its integral. The tab comes from the sketch client through an entry point."""
    import json
    import math

    import numpy as np
    from PySide6.QtWidgets import QApplication

    scene = {
        "schema_version": 6,
        "meta": {"title": "sketch"},
        "objects": [{"id": "plane", "type": "domain_coloring",
                     "params": {"function": "1/(z**2+1)", "size": [8.0, 4.5]}}],
        "timeline": {"markers": {"end": 2.0}},
    }  # fmt: skip
    (tmp_path / "s.phasor.json").write_text(json.dumps(scene), encoding="utf-8")
    _app, win, server = open_editor(str(tmp_path / "s.phasor.json"), show=False)
    try:
        titles = [t for t, _ in win.panels]
        assert "Sketch" in titles
        panel = dict(win.panels)["Sketch"]
        win.viewport.resize(640, 360)
        win.viewport.refresh()
        st = win.ed.engine.state_at(0.0)
        meta = win.ed.engine.document.meta
        ndc = st.camera.project(np.array([[0.0, 1.0, 0.0], [0.6, 1.0, 0.0]]))
        c = np.array([(ndc[0, 0] + 1) / 2 * meta.width, (1 - ndc[0, 1]) / 2 * meta.height])
        r = (ndc[1, 0] - ndc[0, 0]) / 2 * meta.width
        a = np.linspace(0, 2 * math.pi, 80)
        pts = np.stack([c[0] + r * np.cos(a), c[1] - r * np.sin(a)], axis=1)
        if win.viewport._size[0]:  # with OpenGL: draw through the viewport's mouse events
            panel.draw.setChecked(True)
            fw, fh = win.viewport._size
            ox, oy = (win.viewport.width() - fw) / 2, (win.viewport.height() - fh) / 2
            for k, (x, y) in enumerate(pts):
                q = QPointF(x * fw / meta.width + ox, y * fh / meta.height + oy)
                kind = QMouseEvent.Type.MouseButtonPress if k == 0 else QMouseEvent.Type.MouseMove
                ev = QMouseEvent(kind, q, q, Qt.MouseButton.LeftButton,
                                 Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)  # fmt: skip
                QApplication.sendEvent(win.viewport, ev)
            assert win.ed.selected is None  # the pen drew; it did not select or turn anything
        else:  # no OpenGL: hand the panel the stroke
            panel.strokes = [[tuple(p) for p in pts]]
        (p,) = panel.propose()
        assert p.kind == "contour" and p.data["exact"] == pytest.approx(math.pi)
        (r_,) = panel.apply()
        assert r_["ok"]
        assert win.ed.engine.document.object("contour").type == "line"
    finally:
        win.timer.stop()
        server.stop()
        win.close()
