"""The desktop app (``phasor-gui``): the editor window with what a program needs around it.

- a **start screen** when nothing is open: new scene from a template, open a file, recent
  scenes, the examples (copied to your scenes folder before they open, so saving works);
- a **menu bar**: File (New, Open, Open Example, Open Recent, Save, Save As, Export, Close,
  Quit), Edit (Undo, Redo), View (Play, back to start), Settings (math typesetting, a
  desktop shortcut), Help (docs, hardware check, about);
- **several windows**, one scene each; closing a window with unsaved edits asks first; the
  title says which file and marks unsaved edits;
- **export** of a picture, a video or a web page with a progress bar, in a separate
  process (``app.ExportJob``), cancellable.

The decisions are in ``app.py`` (no Qt) and ``controller.Editor``; this file draws them.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from phasor import __version__
from phasor.gui.app import EXPORTS, TEMPLATES, ExportJob, Settings, examples, install_shortcut
from phasor.gui.controller import Editor
from phasor.gui.window import EditorWindow, make_workspace, qt_app, start_ai_server
from phasor.workspace import ToolError
from phasor.workspace.paths import scenes_folder

SCENE_FILTER = "phasor scenes (*.phasor.json *.py);;All files (*)"


# ---------------------------------------------------------------- the app


class Desktop:
    """The running app: its settings and its open windows."""

    def __init__(self, *, serve_ai: bool = True) -> None:
        from phasor.core import text

        self.app = qt_app()
        self.settings = Settings.load()
        self.serve_ai = serve_ai
        self.windows: list[DesktopWindow] = []
        self.start: StartScreen | None = None
        self._closing: list[QWidget] = []  # closed start screens Qt has yet to delete
        with contextlib.suppress(ValueError):
            text.set_math_engine(self.settings.math_engine)

    # ---------------------------------------------------------- windows

    def open(self, path: str | Path | None = None, template: str | None = None,
             *, show: bool = True) -> DesktopWindow | None:  # fmt: skip
        """A window on ``path``, or on a new scene (from ``template``). On failure the
        reason is shown and None returned."""
        if path is not None:
            p = Path(path).resolve()
            for w in self.windows:  # already open: bring it forward
                if w.ed.path is not None and Path(w.ed.path).resolve() in (p, _doc_of(p)):
                    w.raise_()
                    w.activateWindow()
                    return w
        try:
            ws = make_workspace(str(path) if path else None,
                                None if path else str(scenes_folder()))  # fmt: skip
            if template:
                ops = TEMPLATES[template]["ops"]
                if ops:
                    ws.scene_edit(ops)
                ws.scene_edit({"op": "set_meta", "field": "title", "value": template.lower()})
        except (ToolError, OSError, ValueError, SyntaxError) as e:
            self.error("Cannot open", f"{path or template}: {e}")
            return None
        lock = threading.RLock()
        editor = Editor(ws, lock)
        server, hint = (None, "")
        if self.serve_ai:
            try:
                server, hint = start_ai_server(ws, lock)
            except OSError:
                server, hint = None, "AI attach is off (its port could not be opened)"
        win = DesktopWindow(self, editor, server, hint)
        self.windows.append(win)
        if path is not None:
            self.settings.opened(path)
            self.settings.save()
            self.refresh_menus()
        if show:
            win.show()
        if self.start is not None:
            # Hide it now and let Qt delete it in the event loop: open() usually runs inside
            # one of the start screen's own buttons, and dropping the last Python reference
            # here would delete the widget on the spot (an access violation at exit on
            # Windows, 2026-11-20). The reference lives until Qt has deleted it.
            st, self.start = self.start, None
            st.close()
            self._closing.append(st)
            st.destroyed.connect(lambda *_: self._closing.remove(st))
            st.deleteLater()
        return win

    def closed(self, win: DesktopWindow) -> None:
        if win in self.windows:
            self.windows.remove(win)

    def show_start(self) -> StartScreen:
        self.start = StartScreen(self)
        self.start.show()
        return self.start

    def run(self, paths: list[str] | None = None) -> int:
        for p in paths or []:
            self.open(p)
        if not self.windows:
            self.show_start()
        self.app.setQuitOnLastWindowClosed(True)
        return self.app.exec()

    def refresh_menus(self) -> None:
        for w in self.windows:
            w.fill_recent()

    # ---------------------------------------------------------- shared actions

    def ask_open(self, parent: QWidget | None = None) -> DesktopWindow | None:
        path, _ = QFileDialog.getOpenFileName(parent, "Open a scene",
                                              self.settings.last_dir or str(scenes_folder()),
                                              SCENE_FILTER)  # fmt: skip
        return self.open(path) if path else None

    def open_example(self, example: dict[str, str]) -> DesktopWindow | None:
        """Copy an example (and the files beside it with the same name: narration, audio;
        and the examples' ``assets/`` folder: pictures, meshes, music) to the scenes folder,
        then open the copy: the original stays as it was."""
        src = Path(example["path"])
        dest_dir = scenes_folder() / "examples"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            for f in src.parent.glob(src.stem + ".*"):
                if f.is_file() and not (dest_dir / f.name).exists():
                    shutil.copy2(f, dest_dir / f.name)
        assets = src.parent / "assets"
        if assets.is_dir():
            shutil.copytree(assets, dest_dir / "assets", dirs_exist_ok=True)
        return self.open(dest)

    def set_math_engine(self, name: str) -> None:
        from phasor.core import text

        try:
            text.set_math_engine(name)
            text.math_engine()  # "latex" without TeX installed is refused here
        except ValueError as e:
            text.set_math_engine(self.settings.math_engine)
            self.error("LaTeX is not available", str(e))
            return
        self.settings.math_engine = name
        self.settings.save()
        for w in self.windows:
            w.ed._frame = None  # cached pictures were typeset the old way
            w.refresh_all()
            w.check_math_engine()

    def error(self, title: str, text: str) -> None:
        QMessageBox.warning(None, title, text)


def _doc_of(p: Path) -> Path:
    """The document a script's scene is saved as (``name.py`` -> ``name.phasor.json``)."""
    return p.with_name(p.stem + ".phasor.json") if p.suffix == ".py" else p


# ---------------------------------------------------------------- one window


class DesktopWindow(EditorWindow):
    def __init__(self, desktop: Desktop, editor: Editor, server: Any, hint: str) -> None:
        self.desktop, self.server = desktop, server
        self.export_job: ExportJob | None = None
        super().__init__(editor, hint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._menus()
        self._title_state: tuple | None = None
        self.update_title()

    # ---------------------------------------------------------- menus

    def _menus(self) -> None:
        bar = self.menuBar()
        d = self.desktop

        def act(menu, text, fn, key=None) -> QAction:
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            menu.addAction(a)
            return a

        f = bar.addMenu("&File")
        new = f.addMenu("&New")
        for name, t in TEMPLATES.items():
            a = act(new, f"{name}  ({t['about']})", lambda _=False, n=name: d.open(template=n))
            if name == "2D scene":
                a.setShortcut(QKeySequence("Ctrl+N"))
        act(f, "&Open...", lambda: d.ask_open(self), "Ctrl+O")
        ex = f.addMenu("Open &Example")
        for e in examples():
            a = act(ex, e["name"], lambda _=False, e=e: d.open_example(e))
            a.setToolTip(e["about"])
        self.recent_menu = f.addMenu("Open &Recent")
        self.fill_recent()
        f.addSeparator()
        f.addAction(self.actions["Save"])
        act(f, "Save &As...", self.save_as, "Ctrl+Shift+S")
        act(f, "&Export...", self.export, "Ctrl+E")
        f.addSeparator()
        act(f, "&Close Window", self.close, "Ctrl+W")
        act(f, "&Quit", self.quit_all, "Ctrl+Q")

        e = bar.addMenu("&Edit")
        e.addAction(self.actions["Undo"])
        e.addAction(self.actions["Redo"])

        v = bar.addMenu("&View")
        v.addAction(self.actions["Play"])
        act(v, "Back to the &Start", lambda: self._scrub(0.0), "Home")

        s = bar.addMenu("&Settings")
        math = s.addMenu("&Math Typesetting")
        group = QActionGroup(self)
        self.math_actions: dict[str, QAction] = {}
        for key, label in (("mathtext", "Typeset (built in, the default)"),
                           ("latex", "LaTeX (needs latex and dvipng)"),
                           ("auto", "LaTeX when installed, else typeset")):  # fmt: skip
            a = act(math, label, lambda _=False, k=key: d.set_math_engine(k))
            a.setCheckable(True)
            group.addAction(a)
            self.math_actions[key] = a
        self.check_math_engine()
        act(s, "Add to the Applications &Menu", self.shortcut)

        h = bar.addMenu("&Help")
        act(h, "&Documentation", self.docs, "F1")
        act(h, "&Hardware Check", self.hardware)
        act(h, "&About phasor", self.about)

    def check_math_engine(self) -> None:
        a = self.math_actions.get(self.desktop.settings.math_engine)
        if a is not None:
            a.setChecked(True)

    def fill_recent(self) -> None:
        m = self.recent_menu
        m.clear()
        recent = self.desktop.settings.existing_recent()
        for p in recent:
            a = QAction(Path(p).name, self)
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, p=p: self.desktop.open(p))
            m.addAction(a)
        m.setEnabled(bool(recent))

    # ---------------------------------------------------------- the title

    def update_title(self) -> None:
        state = (self.ed.path, self.ed.dirty, self.ed.engine.document.meta.title)
        if state == self._title_state:
            return
        self._title_state = state
        name = Path(self.ed.path).name if self.ed.path else f"{state[2]} (not saved yet)"
        self.setWindowTitle(f"{'• ' if self.ed.dirty else ''}{name} — phasor")

    def tick(self) -> None:
        super().tick()
        self.update_title()
        if self.export_job is not None and self.export_dialog is not None:
            self.export_dialog.poll()

    # ---------------------------------------------------------- saving

    def save_scene(self) -> None:
        if self.ed.path is None:
            self.save_as()
            return
        super().save_scene()
        self.update_title()

    def save_as(self) -> bool:
        start = self.ed.path or (scenes_folder() / _slug(self.ed.engine.document.meta.title))
        path, _ = QFileDialog.getSaveFileName(self, "Save the scene as", str(start),
                                              "phasor scenes (*.phasor.json)")  # fmt: skip
        if not path:
            return False
        try:
            r = self.ed.save_as(path)
        except ToolError as e:
            self.statusBar().showMessage(str(e), 8000)
            return False
        if self.server is not None:
            self.server.write_handoff()  # the workspace folder may have moved
        self._report(r)
        self.desktop.settings.opened(r["saved"])
        self.desktop.settings.save()
        self.desktop.refresh_menus()
        self.update_title()
        return True

    def maybe_save(self) -> bool:
        """Before closing: True when it is fine to go on (saved, discarded or clean)."""
        if not self.ed.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"Save the changes to {self.windowTitle().lstrip('• ')}?")
        box.setStandardButtons(QMessageBox.StandardButton.Save
                               | QMessageBox.StandardButton.Discard
                               | QMessageBox.StandardButton.Cancel)  # fmt: skip
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            self.save_scene()
            return not self.ed.dirty
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, ev) -> None:
        if not self.maybe_save():
            ev.ignore()
            return
        if self.export_job is not None and not self.export_job.done:
            self.export_job.cancel()
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.desktop.closed(self)
        super().closeEvent(ev)

    def quit_all(self) -> None:
        for w in list(self.desktop.windows):
            if not w.close():
                return
        if self.desktop.start is not None:
            self.desktop.start.close()

    # ---------------------------------------------------------- export

    export_dialog: ExportDialog | None = None

    def export(self) -> None:
        if self.export_job is not None and not self.export_job.done:
            self.export_dialog.show()
            self.export_dialog.raise_()
            return
        self.export_dialog = ExportDialog(self)
        self.export_dialog.show()

    # ---------------------------------------------------------- settings and help

    def shortcut(self) -> None:
        r = install_shortcut()
        if r.get("ok"):
            QMessageBox.information(self, "Added", f"phasor is in the applications menu:\n"
                                                   f"{r['path']}")  # fmt: skip
        else:
            QMessageBox.warning(self, "Not added", r.get("error") or "the shortcut failed")

    def docs(self) -> None:
        here = Path(__file__).resolve()
        for p in (here.parents[2] / "docs" / "ai" / "API.md",
                  here.parents[1] / "_docs" / "docs" / "ai" / "API.md"):  # fmt: skip
            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
                return
        QMessageBox.information(self, "Documentation", "docs/ai/API.md is not installed here.")

    def hardware(self) -> None:
        from phasor.render.doctor import run_doctor

        r = run_doctor()
        mark = {True: "ok", False: "missing", None: "skipped"}
        lines = [f"Tier {r.tier}", ""] + [f"{c.name}: {mark[c.ok]}  {c.detail}"
                                          for c in r.checks]  # fmt: skip
        QMessageBox.information(self, "Hardware check", "\n".join(lines))

    def about(self) -> None:
        QMessageBox.about(self, "About phasor",
                          f"<b>phasor {__version__}</b><p>Math animations an AI can see, check "
                          "and fix.</p><p>MIT licence.</p>")  # fmt: skip


def _slug(title: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_") or "scene"
    return s + ".phasor.json"


# ---------------------------------------------------------------- the start screen


class StartScreen(QWidget):
    """What you see when nothing is open."""

    def __init__(self, desktop: Desktop) -> None:
        super().__init__()
        self.desktop = desktop
        self.setWindowTitle("phasor")
        self.resize(820, 540)
        title = QLabel(f"<h2>phasor</h2><p style='color:#8a93a8'>Math animations an AI can see, "
                       f"check and fix. Version {__version__}.</p>")  # fmt: skip
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Start</b>"))
        for name, t in TEMPLATES.items():
            b = QPushButton(f"New {name[0].lower() + name[1:] if name[1:2].islower() else name}")
            b.setToolTip(t["about"])
            b.clicked.connect(lambda _=False, n=name: desktop.open(template=n))
            left.addWidget(b)
        op = QPushButton("Open a scene...")
        op.clicked.connect(lambda: desktop.ask_open(self))
        left.addWidget(op)
        left.addStretch(1)
        self.recent = QListWidget()
        for p in desktop.settings.existing_recent():
            item = QListWidgetItem(f"{Path(p).name}\n{Path(p).parent}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.recent.addItem(item)
        self.recent.itemActivated.connect(
            lambda it: desktop.open(it.data(Qt.ItemDataRole.UserRole))
        )
        self.examples = QListWidget()
        self.examples.setWordWrap(True)
        self.examples.setSpacing(4)
        self.examples.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for e in examples():
            item = QListWidgetItem(f"{e['name'].capitalize()}\n{e['about']}")
            item.setData(Qt.ItemDataRole.UserRole, e)
            item.setToolTip(e["about"])
            self.examples.addItem(item)
        self.examples.itemActivated.connect(
            lambda it: desktop.open_example(it.data(Qt.ItemDataRole.UserRole))
        )
        right = QVBoxLayout()
        right.addWidget(
            QLabel(
                "<b>Recent</b>"
                if self.recent.count()
                else "<b>Recent</b> <span style='color:#8a93a8'>(none yet)</span>"
            )
        )
        self.recent.setMaximumHeight(140 if self.recent.count() else 60)
        right.addWidget(self.recent)
        right.addWidget(
            QLabel(
                "<b>Examples</b> <span style='color:#8a93a8'>(double-click: "
                "a copy opens in Documents/phasor/examples)</span>"
            )
        )
        right.addWidget(self.examples, 1)
        cols = QHBoxLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 3)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addLayout(cols, 1)


# ---------------------------------------------------------------- export


class ExportDialog(QDialog):
    """Choose what to export and where; watch it go; cancel it."""

    def __init__(self, win: DesktopWindow) -> None:
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Export")
        self.kind = QComboBox()
        for k, e in EXPORTS.items():
            self.kind.addItem(e["label"], k)
        self.out = QLineEdit()
        browse = QPushButton("Choose...")
        browse.clicked.connect(self.choose)
        where = QHBoxLayout()
        where.addWidget(self.out, 1)
        where.addWidget(browse)
        self.preset = QComboBox()
        for p, text in (("preview", "Preview (960 x 540, 30 fps, quick)"),
                        ("draft", "Draft (640 x 360, 15 fps, fastest, rough)"),
                        ("final", "Final (the scene's size and frame rate, motion blur)"),
                        ("4k", "4K (3840 x 2160)")):  # fmt: skip
            self.preset.addItem(text, p)
        self.time = QDoubleSpinBox()
        self.time.setRange(0.0, max(win.ed.engine.duration, 0.0))
        self.time.setDecimals(2)
        self.time.setValue(win.ed.t)
        self.steps = QSpinBox()
        self.steps.setRange(2, 25)
        self.steps.setValue(5)
        form = QFormLayout()
        form.addRow("What", self.kind)
        form.addRow("Save to", where)
        form.addRow("Quality", self.preset)
        form.addRow("At time (s)", self.time)
        form.addRow("Steps per slider", self.steps)
        self.form = form
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.buttons = QDialogButtonBox()
        self.go = self.buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        self.stop = self.buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.show_file = self.buttons.addButton("Show File", QDialogButtonBox.ButtonRole.ActionRole)
        self.show_file.hide()
        self.go.clicked.connect(self.start)
        self.stop.clicked.connect(self.cancel)
        self.show_file.clicked.connect(self.reveal)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.buttons)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()

    def _kind(self) -> str:
        return self.kind.currentData()

    def _kind_changed(self) -> None:
        k = self._kind()
        self.form.setRowVisible(self.preset, k != "web")
        self.form.setRowVisible(self.time, k == "picture")
        self.form.setRowVisible(self.steps, k == "web")
        if k == "web" and not self.win.ed.engine.document.controls:
            self.status.setText("This scene has no sliders: the page will show only the picture.")
        else:
            self.status.setText("")
        base = Path(self.out.text()) if self.out.text() else self._default_dir() / self._stem()
        self.out.setText(str(base.with_name(base.name.split(".")[0] + EXPORTS[k]["suffix"])))

    def _default_dir(self) -> Path:
        p = self.win.ed.path
        return Path(p).parent if p else scenes_folder()

    def _stem(self) -> str:
        p = self.win.ed.path
        return Path(p).name.split(".")[0] if p else _slug(self.win.ed.engine.document.meta.title)

    def choose(self) -> None:
        k = self._kind()
        suffix = EXPORTS[k]["suffix"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to", self.out.text(), f"{EXPORTS[k]['label']} (*{suffix});;All files (*)"
        )
        if path:
            self.out.setText(path)

    def start(self) -> None:
        ed = self.win.ed
        out = Path(self.out.text()).expanduser()
        if not out.name:
            self.status.setText("Choose where to save it.")
            return
        with ed.lock:
            job = ExportJob(ed.engine, self._kind(), out, scene_dir=self._default_dir(),
                            t=self.time.value(), preset=self.preset.currentData(),
                            steps=self.steps.value())  # fmt: skip
            try:
                job.start()
            except OSError as e:
                self.status.setText(f"Could not start: {e}")
                return
        self.win.export_job = job
        self.go.setEnabled(False)
        self.show_file.hide()
        self.bar.setRange(0, 0)
        self.bar.show()
        self.status.setText("Working... (you can keep editing; the export uses the scene as "
                            "it was when you pressed Export)")  # fmt: skip

    def poll(self) -> None:
        job = self.win.export_job
        if job is None:
            return
        st = job.poll()
        if st["frames"]:
            self.bar.setRange(0, st["frames"])
            self.bar.setValue(st["frame"])
        if st["done"]:
            self.bar.hide()
            self.go.setEnabled(True)
            self.win.export_job = None
            if st["ok"]:
                self.status.setText(f"Saved {st['out']}")
                self.show_file.show()
                self.win.statusBar().showMessage(f"Exported {st['out']}", 8000)
            else:
                self.status.setText(f"Export failed: {st['message']}")

    def cancel(self) -> None:
        job = self.win.export_job
        if job is not None and not job.done:
            job.cancel()
            self.win.export_job = None
            self.bar.hide()
            self.go.setEnabled(True)
            self.status.setText("Cancelled.")
        else:
            self.close()

    def reveal(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.out.text()).parent)))


# ---------------------------------------------------------------- entry point


def run(paths: list[str] | None = None, *, serve_ai: bool = True) -> int:
    return Desktop(serve_ai=serve_ai).run(paths)
