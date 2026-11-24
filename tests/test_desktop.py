"""The desktop app (``phasor-gui``): settings, recent scenes, unsaved changes, exports, the
start screen, windows and menus, closing, shortcuts and the math setting.

The windows run on Qt's ``offscreen`` platform. Modal dialogs (message boxes, file pickers)
would wait for a person, so each test that reaches one replaces it with a stand-in that
answers as the person would, and records that it was asked."""

from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")

from phasor import Dot, Scene
from phasor.gui import app as appmod
from phasor.gui.app import (
    EXPORTS,
    MAX_RECENT,
    TEMPLATES,
    ExportJob,
    Settings,
    cli_command,
    examples,
    install_shortcut,
)
from tests.conftest import gl_available

gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own home folder and settings folder (nothing touches yours), and
    the math engine is put back to the default afterwards."""
    from phasor.core import text

    h = tmp_path / "home"
    (h / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))  # what Path.home() reads on Windows
    monkeypatch.setenv("PHASOR_CONFIG_DIR", str(h / "settings"))
    monkeypatch.setenv("XDG_DATA_HOME", str(h / "share"))
    yield h
    text.set_math_engine("mathtext")


def plain_scene(path: Path, title: str = "plain") -> Path:
    """A small scene saved as a plain document (one an editor may add objects to)."""
    s = Scene(title, passes=None)
    s.add(Dot(id="pole", pos=(1.0, 0.5)))
    s.marker("end", 2.0)
    doc = s.engine.document.model_dump(mode="json")
    doc["meta"]["source"] = "document"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------- A1: settings


def test_settings_round_trip_and_survive_a_damaged_file(home):
    s = Settings.load()
    assert (s.recent, s.math_engine) == ([], "mathtext")  # nothing saved yet: defaults
    a, b = plain_scene(home / "a.phasor.json"), plain_scene(home / "b.phasor.json")
    s.opened(a)
    s.opened(b)
    s.math_engine = "auto"
    s.save()
    again = Settings.load()
    assert again.recent == [str(b.resolve()), str(a.resolve())]  # newest first
    assert again.last_dir == str(home) and again.math_engine == "auto"
    Settings.path().write_text("{ not json", encoding="utf-8")
    assert Settings.load().recent == []  # damaged: defaults, no crash
    Settings.path().write_text(
        json.dumps({"recent": [1, None, "x"], "math_engine": "bogus"}), encoding="utf-8"
    )
    bad = Settings.load()
    assert bad.recent == ["x"] and bad.math_engine == "mathtext"  # junk values ignored


def test_recent_scenes_have_no_duplicates_keep_a_limit_and_hide_deleted_ones(home):
    s = Settings()
    paths = [plain_scene(home / f"s{k}.phasor.json") for k in range(MAX_RECENT + 3)]
    for p in paths:
        s.opened(p)
    s.opened(paths[-1])  # opened again: still once
    assert len(s.recent) == MAX_RECENT and len(set(s.recent)) == MAX_RECENT
    assert s.recent[0] == str(paths[-1].resolve())
    paths[-1].unlink()
    assert str(paths[-1].resolve()) not in s.existing_recent()


def test_the_settings_folder_is_the_platforms_own(monkeypatch, home):
    monkeypatch.delenv("PHASOR_CONFIG_DIR")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "cfg"))
    assert appmod.config_dir() == home / "cfg" / "phasor"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(home / "AppData"))
    assert appmod.config_dir() == home / "AppData" / "phasor"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert appmod.config_dir() == home / "Library" / "Application Support" / "phasor"


# ---------------------------------------------------------------- A2: unsaved changes


def _editor(path: Path | None):
    from phasor.gui.controller import Editor
    from phasor.gui.window import make_workspace

    return Editor(make_workspace(str(path) if path else None, None if path else str(path)))


def test_edits_by_you_or_an_ai_are_unsaved_until_saved(home):
    ed = _editor(plain_scene(home / "s.phasor.json"))
    assert not ed.dirty
    ed.set_param("pole", "r", "0.3")
    assert ed.dirty
    ed.save()
    assert not ed.dirty
    ed.ws.scene_edit({"op": "set_param", "path": "pole.r", "value": 0.2})  # as an AI would
    assert ed.dirty


def test_save_as_adds_the_suffix_and_moves_the_workspace_folder(home, tmp_path):
    ed = _editor(plain_scene(home / "s.phasor.json"))
    ed.set_param("pole", "r", "0.3")
    inside = ed.save_as(home / "sub" / "copy")  # inside the folder: it stays the folder
    assert inside["saved"] == str((home / "sub" / "copy.phasor.json").resolve())
    assert Path(inside["saved"]).exists() and not ed.dirty
    assert ed.ws.root == home.resolve()
    outside = tmp_path / "elsewhere" / "far.phasor.json"
    r = ed.save_as(outside)  # outside it: an attached AI follows the scene there
    assert r["saved"] == str(outside.resolve()) and ed.ws.root == outside.parent.resolve()
    assert ed.path == Path(r["saved"])


def test_tweaks_to_a_script_scene_are_kept_when_saved(home):
    from phasor import load

    script = home / "scene.py"
    script.write_text("from phasor import Scene, Dot\ns = Scene('script', passes=None)\n"
                      "s.add(Dot(id='p', pos=(0, 0)))\n", encoding="utf-8")  # fmt: skip
    ed = _editor(script)
    ed.set_param("p", "r", "0.4")
    ed.save()
    assert (home / "scene.overrides.json").exists()
    assert load(script, passes=None).document.object("p").params["r"] == 0.4


# ---------------------------------------------------------------- A3: exports


def test_each_export_runs_the_matching_command():
    job = ExportJob(None, "picture", "/out/a.png", scene_dir=".", t=1.5, preset="final")
    assert job.command("s.phasor.json")[-8:] == [
        "render",
        "s.phasor.json",
        "-o",
        str(Path("/out/a.png")),  # backslashes on Windows
        "--frame",
        "1.5",
        "--preset",
        "final",
    ]
    video = ExportJob(None, "video", "/out/a.mp4", scene_dir=".").command("s.phasor.json")
    assert "--frame" not in video and video[-2:] == ["--preset", "preview"]
    web = ExportJob(None, "web", "/out/a.html", scene_dir=".", steps=7).command("s.phasor.json")
    assert web[-6:] == ["web", "s.phasor.json", "-o", str(Path("/out/a.html")), "--steps", "7"]
    with pytest.raises(ValueError, match="export kind"):
        ExportJob(None, "gif", "x", scene_dir=".")
    assert set(EXPORTS) == {"picture", "video", "web"}


def _engine(home):
    from phasor import load

    return load(plain_scene(home / "scene" / "s.phasor.json"), passes=None)


def _wait(job, seconds=300.0):
    t0 = time.time()
    seen = []
    while not job.poll()["done"]:
        seen.append((job.frame, job.frames))
        assert time.time() - t0 < seconds, "export did not finish"
        time.sleep(0.2)
    return seen


def _snapshots(folder: Path) -> list[Path]:
    return list(folder.glob(".phasor-export-*"))


@gl
def test_a_picture_and_a_web_page_export_and_leave_nothing_behind(home):
    e = _engine(home)
    folder = home / "scene"
    pic = ExportJob(e, "picture", home / "out" / "s.png", scene_dir=folder, t=1.0)
    pic.start()
    assert _snapshots(folder)  # the scene as it was, beside the original
    _wait(pic)
    assert pic.ok and (home / "out" / "s.png").read_bytes()[:4] == b"\x89PNG"
    assert not _snapshots(folder)
    page = ExportJob(e, "web", home / "out" / "s.html", scene_dir=folder, steps=2)
    page.start()
    _wait(page)
    assert page.ok and "<html" in (home / "out" / "s.html").read_text(encoding="utf-8")


@gl
def test_a_video_export_reports_progress_and_can_be_cancelled(home):
    e = _engine(home)
    folder = home / "scene"
    job = ExportJob(e, "video", home / "out" / "s.mp4", scene_dir=folder)
    job.start()
    seen = _wait(job)
    assert job.ok and job.frames == 60 and job.frame == 60  # 2 s at 30 fps, read from its output
    counts = [k for k, n in seen if n]
    assert counts == sorted(counts)  # progress only goes forward
    again = ExportJob(e, "video", home / "out" / "c.mp4", scene_dir=folder)
    again.start()
    time.sleep(0.5)
    again.cancel()
    assert again.status()["message"] == "cancelled" and not again.ok
    assert not _snapshots(folder)


def test_a_failed_export_says_why_and_leaves_nothing_behind(home):
    e = _engine(home)
    folder = home / "scene"
    job = ExportJob(e, "video", home / "out" / "s.xyz", scene_dir=folder)  # no such format
    job.start()
    _wait(job)
    assert job.done and not job.ok and job.message and "exit code" not in job.message
    assert not _snapshots(folder)


def test_an_export_to_a_place_that_cannot_exist_leaves_nothing_behind(home):
    e = _engine(home)
    folder = home / "scene"
    (home / "a_file").write_text("not a folder", encoding="utf-8")
    job = ExportJob(e, "picture", home / "a_file" / "s.png", scene_dir=folder)
    with pytest.raises(OSError):
        job.start()
    assert not _snapshots(folder)


@gl
def test_narration_audio_beside_the_scene_is_in_the_exported_video(home):
    import av

    e = _engine(home)
    folder = home / "scene"
    rate = 16000
    tone = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 8000).astype("<i2")
    with wave.open(str(folder / "voice.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(tone.tobytes())
    r = e.apply({"op": "set_narration", "narration": {"script": "Hi.", "voice": "none",
                 "blocks": [{"text": "Hi.", "start": 0.0, "duration": 1.0,
                             "audio": "voice.wav"}]}})  # fmt: skip
    assert r.ok
    job = ExportJob(e, "video", home / "out" / "talk.mp4", scene_dir=folder)
    job.start()
    _wait(job)
    assert job.ok, job.message
    with av.open(str(home / "out" / "talk.mp4")) as f:
        assert any(s.type == "audio" for s in f.streams)  # the relative path resolved


# ---------------------------------------------------------------- A4-A8: the app's windows


class Asked:
    """Stands in for the modal dialogs; records what the app asked, answers as told."""

    def __init__(self, monkeypatch, save_to: Path | None = None, close_answer=None) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        self.warnings: list[tuple[str, str]] = []
        self.save_dialogs = 0
        self.close_prompts = 0
        self.close_answer = close_answer or QMessageBox.StandardButton.Discard

        def warning(_parent, title, text, *a, **k):
            self.warnings.append((title, text))
            return QMessageBox.StandardButton.Ok

        def save_name(*_a, **_k):
            self.save_dialogs += 1
            return (str(save_to) if save_to else "", "")

        def prompt(box):
            self.close_prompts += 1
            return self.close_answer

        monkeypatch.setattr(QMessageBox, "warning", staticmethod(warning))
        monkeypatch.setattr(QMessageBox, "information", staticmethod(warning))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(save_name))
        monkeypatch.setattr(QMessageBox, "exec", prompt)


@pytest.fixture
def desk(home):
    from phasor.gui.desktop import Desktop

    d = Desktop(serve_ai=False)
    yield d
    for w in list(d.windows):
        w.ed.saved_version = w.ed.version  # nothing left to ask about
        w.close()
    if d.start is not None:
        d.start.close()


def test_the_start_screen_lists_recent_scenes_and_the_examples(desk, home):
    a = plain_scene(home / "a.phasor.json")
    desk.settings.opened(a)
    desk.settings.save()
    st = desk.show_start()
    assert st.recent.count() == 1 and st.examples.count() == len(examples()) >= 10
    for i in range(st.examples.count()):
        assert "Milestone" not in st.examples.item(i).text()  # what it is, not project notes


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_each_template_opens_a_window_and_closes_the_start_screen(desk, template):
    desk.show_start()
    w = desk.open(template=template)
    assert w is not None and desk.start is None and w.ed.path is None
    assert w.ed.engine.document.meta.title == template.lower()
    assert "(not saved yet)" in w.windowTitle()
    kind = w.ed.engine.document.cameras[0].kind
    assert kind == ("orbit" if template == "3D scene" else "ortho")
    if template == "Complex function":
        assert w.ed.engine.document.object("plane").type == "domain_coloring"


def test_opening_an_open_scene_again_brings_its_window_forward(desk, home):
    p = plain_scene(home / "a.phasor.json")
    w = desk.open(p)
    assert desk.open(p) is w and len(desk.windows) == 1
    assert desk.settings.recent[0] == str(p.resolve())


def test_a_broken_file_says_why_and_opens_nothing(desk, home, monkeypatch):
    asked = Asked(monkeypatch)
    bad = home / "bad.phasor.json"
    bad.write_text("{ broken", encoding="utf-8")
    assert desk.open(bad) is None and desk.windows == []
    assert asked.warnings and "bad.phasor.json" in asked.warnings[0][1]


def test_an_example_opens_as_a_copy_with_its_narration_and_the_original_untouched(desk, home):
    ex = next(e for e in examples() if e["name"] == "residue theorem")
    original = Path(ex["path"])
    before = original.read_bytes()
    w = desk.open_example(ex)
    copies = home / "Documents" / "phasor" / "examples"
    assert w is not None and (copies / original.name).exists()
    narration = original.with_suffix(".narration")  # a speech cache, made where TTS exists
    if narration.exists():
        assert (copies / narration.name).exists()  # the files that go with it
    w.ed.set_param(w.ed.engine.document.objects[0].id, "opacity", "0.5")
    w.ed.save()
    assert original.read_bytes() == before


def test_the_title_marks_unsaved_edits(desk, home):
    w = desk.open(plain_scene(home / "a.phasor.json"))
    assert w.windowTitle() == "a.phasor.json — phasor"
    w.ed.set_param("pole", "r", "0.3")
    w.tick()
    assert w.windowTitle().startswith("• ")
    w.save_scene()
    w.tick()
    assert not w.windowTitle().startswith("• ")


# ---------------------------------------------------------------- A5: closing and saving


def test_saving_a_new_scene_asks_where(desk, home, monkeypatch):
    target = home / "Documents" / "phasor" / "mine.phasor.json"
    asked = Asked(monkeypatch, save_to=target)
    w = desk.open(template="2D scene")
    w.save_scene()  # Ctrl+S on a scene never saved
    assert asked.save_dialogs == 1 and target.exists() and w.ed.path == target.resolve()
    assert str(target.resolve()) in desk.settings.recent


@pytest.mark.parametrize("answer, closes, saved", [
    ("Save", True, True), ("Discard", True, False), ("Cancel", False, False)])  # fmt: skip
def test_closing_with_unsaved_edits_asks_first(desk, home, monkeypatch, answer, closes, saved):
    from PySide6.QtWidgets import QMessageBox

    asked = Asked(monkeypatch, close_answer=getattr(QMessageBox.StandardButton, answer))
    p = plain_scene(home / "a.phasor.json")
    w = desk.open(p)
    w.ed.set_param("pole", "r", "0.3")
    assert w.close() is closes
    assert asked.close_prompts == 1
    assert (w not in desk.windows) is closes
    assert (
        json.loads(p.read_text(encoding="utf-8"))["objects"][0]["params"].get("r") == 0.3
    ) is saved


def test_a_clean_window_closes_without_asking(desk, home, monkeypatch):
    asked = Asked(monkeypatch)
    w = desk.open(plain_scene(home / "a.phasor.json"))
    assert w.close() and asked.close_prompts == 0


def test_quit_stops_at_the_first_cancel(desk, home, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    asked = Asked(monkeypatch, close_answer=QMessageBox.StandardButton.Cancel)
    a = desk.open(plain_scene(home / "a.phasor.json"))
    b = desk.open(plain_scene(home / "b.phasor.json"))
    a.ed.set_param("pole", "r", "0.3")
    b.ed.set_param("pole", "r", "0.3")
    a.quit_all()
    assert asked.close_prompts == 1 and len(desk.windows) == 2  # stopped at the first


def test_closing_cancels_a_running_export_and_stops_the_ai(home, monkeypatch):
    from phasor.gui.desktop import Desktop

    d = Desktop(serve_ai=True)
    w = d.open(plain_scene(home / "a.phasor.json"))
    assert w.server is not None
    handoff = home / ".phasor" / "gui.json"
    assert handoff.exists()

    class Running:
        done = False
        cancelled = False

        def cancel(self):
            self.cancelled = True
            self.done = True

    job = Running()
    w.export_job = job
    assert w.close()
    assert job.cancelled and w.server is None and not handoff.exists()


# ---------------------------------------------------------------- A6: shortcuts


def test_no_two_actions_share_a_shortcut(desk, home):
    from PySide6.QtGui import QAction

    w = desk.open(plain_scene(home / "a.phasor.json"))
    keys: dict[str, str] = {}
    for a in w.findChildren(QAction):
        for k in a.shortcuts():
            s = k.toString()
            assert s not in keys or keys[s] == a.text(), (s, keys.get(s), a.text())
            keys[s] = a.text()
    assert {"Ctrl+S", "Ctrl+Shift+S", "Ctrl+O", "Ctrl+N", "Ctrl+E", "Ctrl+Z"} <= set(keys)


# ---------------------------------------------------------------- A7: math typesetting


def test_latex_without_tex_is_refused_and_the_choice_is_remembered(desk, home, monkeypatch):
    from phasor.core import text

    asked = Asked(monkeypatch)
    monkeypatch.setattr(text, "latex_available", lambda: False)
    w = desk.open(plain_scene(home / "a.phasor.json"))
    desk.set_math_engine("latex")
    assert asked.warnings and asked.warnings[0][0] == "LaTeX is not available"
    assert text.math_engine() == "mathtext" and desk.settings.math_engine == "mathtext"
    assert w.math_actions["mathtext"].isChecked()
    desk.set_math_engine("auto")  # no TeX: typesets with the built-in engine, remembered
    assert Settings.load().math_engine == "auto" and w.math_actions["auto"].isChecked()


# ---------------------------------------------------------------- A8: menus and the export dialog


def test_recent_menus_update_in_every_window(desk, home):
    a = desk.open(plain_scene(home / "a.phasor.json"))
    desk.open(plain_scene(home / "b.phasor.json"))
    names = [x.text() for x in a.recent_menu.actions()]
    assert names[:2] == ["b.phasor.json", "a.phasor.json"]


def test_the_export_dialog_shows_only_what_each_kind_needs(desk, home):
    from phasor.gui.desktop import ExportDialog

    w = desk.open(plain_scene(home / "a.phasor.json"))
    dlg = ExportDialog(w)
    shown = {}
    for k in EXPORTS:
        dlg.kind.setCurrentIndex(dlg.kind.findData(k))
        shown[k] = (dlg.form.isRowVisible(dlg.preset), dlg.form.isRowVisible(dlg.time),
                    dlg.form.isRowVisible(dlg.steps), dlg.out.text())  # fmt: skip
    assert shown["picture"][:3] == (True, True, False) and shown["picture"][3].endswith("a.png")
    assert shown["video"][:3] == (True, False, False) and shown["video"][3].endswith("a.mp4")
    assert shown["web"][:3] == (False, False, True) and shown["web"][3].endswith("a.html")
    assert "no sliders" in dlg.status.text()  # this scene has none


# ---------------------------------------------------------------- A9: launching


def test_the_command_line_is_found_inside_and_outside_the_packaged_app(monkeypatch):
    assert cli_command("doctor") == [sys.executable, "-m", "phasor", "doctor"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert cli_command("doctor") == [sys.executable, "--cli", "doctor"]
    assert appmod.launch_command() == [sys.executable]


def test_the_python_module_runs_the_command_line():
    import subprocess

    r = subprocess.run(cli_command("--help"), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "COMMAND" in r.stdout


def test_a_menu_shortcut_on_linux_and_honest_answers_elsewhere(home, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    r = install_shortcut()
    entry = (home / "share" / "applications" / "phasor.desktop").read_text(encoding="utf-8")
    assert r["ok"] and "Exec=" in entry and "-m phasor.gui %f" in entry
    assert f"Icon={appmod.ICON}" in entry and appmod.ICON.exists()
    monkeypatch.setattr(sys, "platform", "darwin")
    r = install_shortcut()
    assert not r["ok"] and "macOS" in r["error"]


def test_the_windows_shortcut_is_made_by_powershell(home, monkeypatch):
    """Untested on Windows itself: this checks the command it would run."""
    import subprocess

    seen = {}

    def run(cmd, env=None, **_k):
        seen["cmd"], seen["env"] = cmd, env
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(home / "AppData"))
    monkeypatch.setattr(appmod.subprocess, "run", run)
    r = install_shortcut()
    assert r["ok"] and r["path"].endswith("phasor.lnk")
    assert seen["cmd"][0] == "powershell" and seen["env"]["PHASOR_EXE"] == sys.executable
    assert "phasor.gui" in seen["env"]["PHASOR_ARGS"]


def test_scenes_go_to_the_documents_folder_windows_shows(home, monkeypatch):
    """On Windows, OneDrive often moves Documents (C:/Users/<you>/OneDrive/Documents)."""
    from phasor.gui import desktop
    from phasor.workspace import paths

    moved = home / "OneDrive" / "Documents"
    moved.mkdir(parents=True)
    monkeypatch.setattr(paths, "windows_documents", lambda: moved)
    assert desktop.scenes_folder() == moved / "phasor"
    monkeypatch.setattr(paths, "windows_documents", lambda: home.parent / "elsewhere")
    assert desktop.scenes_folder() == home / "Documents" / "phasor"  # not someone else's
