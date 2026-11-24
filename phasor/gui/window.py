"""The editor window, Qt (PySide6) widgets over ``controller.Editor``.

Nothing here decides anything: widgets draw what the ``Editor`` holds and forward clicks and
typing to it. A 30 Hz timer advances playback and notices edits made by an attached AI
(``phasor mcp --attach``), so both editors always show the same live scene.

Layout: the viewport (click to select, drag a selected object to move it) above the
timeline (markers, animated spans, the playhead: click or drag to scrub); on the right,
tabs for the inspector (parameters, style roles, what the inspector reads), lint (click an
issue to jump to it) and looks (switch look, pick among variants of a beat).
"""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextOption,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from phasor.gui.controller import Editor
from phasor.style.theme import ROLE_GROUPS
from phasor.workspace import ToolError

TICK_MS = 33
DRAG_PX = 4  # a press that moves less than this is a click


def to_qimage(img: np.ndarray) -> QImage:
    h, w = img.shape[:2]
    data = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    q = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.copy()  # own the pixels: ``data`` goes away


class Viewport(QLabel):
    """The picture at the playhead. Click selects; dragging the selected object moves it."""

    picked = Signal(object)  # the id under a click, or None
    dragged = Signal(str, float, float, tuple)  # id, x, y, viewport size
    turned = Signal(float, float, tuple, bool)  # pixels dx, dy, viewport size, pan
    zoomed = Signal(float)  # factor: > 1 moves in
    hovered = Signal(float, float, tuple)  # viewport point and size

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #050608; color: #8a93a8;")
        self._press: QPointF | None = None
        self._size = (0, 0)
        self.error: str | None = None

    def frame_size(self) -> tuple[int, int]:
        meta = self.ed.engine.document.meta
        w, h = max(self.width(), 2), max(self.height(), 2)
        scale = min(w / meta.width, h / meta.height)
        return max(2, int(meta.width * scale)), max(2, int(meta.height * scale))

    def refresh(self) -> None:
        size = self.frame_size()
        try:
            img = self.ed.frame(size)
        except RuntimeError as e:  # no OpenGL on this machine
            self.error = str(e)
            self.setText(f"No preview: {e}")
            return
        self.error = None
        self._size = size
        pix = QPixmap.fromImage(to_qimage(img))
        if self.ed.selected:
            self._outline(pix)
        self.setPixmap(pix)

    def _outline(self, pix: QPixmap) -> None:
        """A box around the selected object, from the inspector's own bounds."""
        try:
            with self.ed.lock:
                seen = self.ed.engine.inspect(self.ed.t, id=self.ed.selected)["objects"][0]
        except (ToolError, KeyError, IndexError):
            return
        box = (seen.get("bbox") or {}).get("norm")
        if not box:
            return
        x0, y0, x1, y1 = box  # normalized, origin top-left
        w, h = pix.width(), pix.height()
        p = QPainter(pix)
        p.setPen(QPen(QColor(255, 200, 90, 200), 1.5, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h))
        p.end()

    def _local(self, pos: QPointF) -> tuple[float, float]:
        """Widget point -> frame pixel (the frame is centered in the widget)."""
        fw, fh = self._size
        return pos.x() - (self.width() - fw) / 2, pos.y() - (self.height() - fh) / 2

    def mousePressEvent(self, ev) -> None:
        self._press = ev.position()
        self._shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)

    def mouseMoveEvent(self, ev) -> None:
        if self._size[0] and not ev.buttons():
            self.hovered.emit(*self._local(ev.position()), self._size)

    def mouseReleaseEvent(self, ev) -> None:
        if self._press is None or not self._size[0]:
            return
        moved = (ev.position() - self._press).manhattanLength()
        u, v = self._local(ev.position())
        if moved >= DRAG_PX:
            on_selected = False
            if self.ed.selected:
                hit = self.ed.hit_ids(*self._local(self._press), self._size)
                on_selected = bool(hit) and hit[0] == self.ed.selected
            if on_selected:  # a drag that starts on the selected object moves it
                self.dragged.emit(self.ed.selected, u, v, self._size)
            else:  # a drag over anything else moves the camera (shift: slide it)
                d = ev.position() - self._press
                self.turned.emit(d.x(), d.y(), self._size, self._shift)
        else:
            self.picked.emit(self.ed.select_at(*self._local(self._press), self._size))
        self._press = None

    def wheelEvent(self, ev) -> None:
        self.zoomed.emit(1.1 ** (ev.angleDelta().y() / 120))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._size != self.frame_size():
            self.refresh()


def _extra_panels(window) -> list[tuple[str, QWidget]]:
    """Tabs other clients add (the ``phasor.panels`` entry point, e.g. Sketch): each entry
    is ``make(window) -> (title, widget)``. Looked up by name, so the editor never imports
    a sibling client; one that fails to load is reported and left out."""
    from importlib.metadata import entry_points

    out = []
    for ep in sorted(entry_points(group="phasor.panels"), key=lambda e: e.name):
        try:
            out.append(ep.load()(window))
        except Exception as e:  # a broken add-on must not stop the editor
            print(f"phasor gui: the {ep.name!r} tab did not load: {e}")
    return out


class Timeline(QWidget):
    """Narration (with its waveform), markers, the animated spans of every parameter (one
    lane each) and the playhead. Click or drag to scrub; drag a marker to move it; drag a span
    to slide it, or its ends to stretch it."""

    scrubbed = Signal(float)
    edited = Signal(object)  # a patch result

    LANE = 14
    LABELS = 130  # pixels of lane names on the left
    NARRATION = 34  # height of the narration lane, when there is narration
    EDGE = 5  # pixels: grab a span's end within this

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.setMinimumHeight(70)
        self.setMouseTracking(True)
        self.data: dict[str, Any] = {"duration": 0.0, "markers": {}, "lanes": {}, "narration": []}
        self._hits: list[tuple[QRectF, str, tuple]] = []
        self._drag: tuple | None = None  # (kind, key..., original, current)
        self._waves: dict[tuple[str, int], list[float]] = {}

    def refresh(self) -> None:
        self.data = self.ed.timeline()
        lanes = len(self.data["lanes"])
        self.setMinimumHeight(min(40 + self._top() + self.LANE * max(lanes, 1), 340))
        self.update()

    def _top(self) -> int:
        return self.NARRATION if self.data.get("narration") else 0

    def _x(self, t: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return self.LABELS + (self.width() - self.LABELS - 8) * t / d

    def _t(self, x: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return (x - self.LABELS) / max(self.width() - self.LABELS - 8, 1) * d

    def _wave(self, audio: str, bins: int) -> list[float]:
        key = (audio, bins)
        if key not in self._waves:
            self._waves[key] = self.ed.waveform(audio, bins)
        return self._waves[key]

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d12"))
        p.setFont(QFont("Sans", 8))
        self._hits = []
        top = self._top()
        for b in self.data.get("narration", []):  # spoken blocks, with their loudness
            x0, x1 = self._x(b["start"]), self._x(b["start"] + b["duration"])
            box = QRectF(x0, 16, max(x1 - x0, 3), self.NARRATION - 18)
            p.fillRect(box, QColor(80, 120, 90, 90))
            if b.get("audio"):
                bins = max(int(box.width() / 2), 2)
                wave = self._wave(b["audio"], bins)
                p.setPen(QPen(QColor("#7fcf9a"), 1))
                for k, a in enumerate(wave):
                    h = a * box.height() / 2
                    x = box.left() + (k + 0.5) / len(wave) * box.width()
                    p.drawLine(QPointF(x, box.center().y() - h), QPointF(x, box.center().y() + h))
            p.setPen(QColor("#b8e0c4"))
            p.drawText(box.adjusted(2, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(b["text"], Qt.TextElideMode.ElideRight,
                                                  int(box.width()) - 4))  # fmt: skip
        p.fillRect(QRectF(0, 0, self.LABELS - 4, self.height()), QColor("#07080b"))
        for k, (path, spans) in enumerate(self.data["lanes"].items()):
            y = 20 + top + k * self.LANE
            p.setPen(QColor("#6c7891"))
            p.drawText(QRectF(6, y, self.LABELS - 12, self.LANE), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(path, Qt.TextElideMode.ElideMiddle,
                                                  self.LABELS - 12))  # fmt: skip
            for i, (t0, t1) in enumerate(spans):
                if self._drag and self._drag[0] == "span" and self._drag[1:3] == (path, i):
                    t0, t1 = self._drag[-1]
                r = QRectF(self._x(t0), y + 2, max(self._x(t1) - self._x(t0), 2), self.LANE - 4)
                p.fillRect(r, QColor(122, 162, 247, 150))
                self._hits.append((r, "span", (path, i, t0, t1)))
        for name, t in self.data["markers"].items():
            if self._drag and self._drag[0] == "marker" and self._drag[1] == name:
                t = self._drag[-1]
            x = self._x(t)
            p.setPen(QPen(QColor("#5a6478"), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setPen(QColor("#aab3c5"))
            p.drawText(QPointF(x + 3, 12), name)
            self._hits.append((QRectF(x - self.EDGE, 0, 2 * self.EDGE, self.height()), "marker",
                               (name, t)))  # fmt: skip
        x = self._x(self.ed.t)
        p.setPen(QPen(QColor("#ffb454"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def _hit(self, pos: QPointF):
        for rect, kind, data in self._hits:  # markers first (thin, on top of spans)
            if kind == "marker" and rect.contains(pos):
                return kind, data
        for rect, kind, data in self._hits:
            if kind == "span" and rect.adjusted(-self.EDGE // 2, 0, self.EDGE // 2, 0).contains(
                pos
            ):
                return kind, data
        return None

    def mousePressEvent(self, ev) -> None:
        hit = self._hit(ev.position())
        if hit and hit[0] == "marker":
            name, t = hit[1]
            self._drag = ("marker", name, t, t)
        elif hit:
            path, i, t0, t1 = hit[1]
            x = ev.position().x()
            mode = ("start" if abs(x - self._x(t0)) <= self.EDGE
                    else "end" if abs(x - self._x(t1)) <= self.EDGE else "move")  # fmt: skip
            self._drag = ("span", path, i, mode, self._t(x), (t0, t1), (t0, t1))
        else:
            self._drag = None
            self.scrubbed.emit(self._t(ev.position().x()))

    def mouseMoveEvent(self, ev) -> None:
        d = self._drag
        if d is None:
            if ev.buttons():
                self.scrubbed.emit(self._t(ev.position().x()))
            else:
                hit = self._hit(ev.position())
                self.setCursor(Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor)
            return
        t = max(0.0, self._t(ev.position().x()))
        if d[0] == "marker":
            self._drag = ("marker", d[1], d[2], t)
        else:
            _, path, i, mode, grab, (o0, o1), _cur = d
            dt = t - grab
            new = ((max(0.0, o0 + dt), o1 + dt) if mode == "move"
                   else (min(t, o1 - 0.05), o1) if mode == "start"
                   else (o0, max(t, o0 + 0.05)))  # fmt: skip
            self._drag = ("span", path, i, mode, grab, (o0, o1), new)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        if d[0] == "marker" and abs(d[3] - d[2]) > 1e-3:
            self.edited.emit(self.ed.move_marker(d[1], d[3]))
        elif d[0] == "span" and max(abs(d[6][0] - d[5][0]), abs(d[6][1] - d[5][1])) > 1e-3:
            self.edited.emit(self.ed.retime_segment(d[1], d[2], *d[6]))
        self.update()


class Inspector(QWidget):
    """The selected object's parameters (type a value, press Enter), style roles and what
    the inspector reads from the picture."""

    edited = Signal(object)  # a patch result

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.form = QFormLayout()
        self.title = QLabel("Nothing selected")
        self.title.setStyleSheet("font-weight: bold;")
        self.semantic = QPlainTextEdit()
        self.semantic.setReadOnly(True)
        self.semantic.setMaximumHeight(160)
        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        holder = QWidget()
        holder.setLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        lay.addWidget(scroll)
        lay.addWidget(QLabel("What the inspector sees"))
        lay.addWidget(self.semantic)
        self.fields: dict[str, QLineEdit] = {}
        self.roles: dict[str, QComboBox] = {}

    def refresh(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields, self.roles = {}, {}
        panel = self.ed.inspector()
        if panel is None:
            self.title.setText("Nothing selected (click an object)")
            self.semantic.setPlainText("")
            return
        oid = panel["id"]
        self.title.setText(f"{oid}  ({panel['type']})")
        for name, value in panel["params"].items():
            field = QLineEdit(json.dumps(value, default=list))
            if any(p == f"{oid}.{name}" for p in panel["animated"]):
                field.setToolTip("animated: this is its value at the playhead")
            field.editingFinished.connect(
                lambda f=field, n=name: self.edited.emit(self.ed.set_param(oid, n, f.text()))
            )
            self.form.addRow(name, field)
            self.fields[name] = field
        for key, role in panel["style"].items():
            group = self.ed.engine.state_at(self.ed.t)[oid].impl.Style.get(key, (None,))[0]
            box = QComboBox()
            box.addItems(list(ROLE_GROUPS.get(group, (role,))))
            box.setCurrentText(role)
            box.currentTextChanged.connect(
                lambda r, k=key: self.edited.emit(self.ed.set_style(oid, k, r))
            )
            self.form.addRow(f"style: {key}", box)
            self.roles[key] = box
        for rule in panel["constraints"]:
            self.form.addRow("rule", QLabel(rule))
        self.semantic.setPlainText(json.dumps(panel["semantic"], indent=1, default=str))


class LintPanel(QListWidget):
    picked = Signal(dict)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.itemClicked.connect(lambda item: self.picked.emit(item.data(Qt.ItemDataRole.UserRole)))

    def refresh(self) -> None:
        self.clear()
        items = self.ed.lint_items()
        if not items:
            self.addItem("No issues")
        for issue in items:
            when = f" @ {issue['times'][0]:g}s" if issue.get("times") else ""
            it = QListWidgetItem(f"[{issue['severity']}] {issue['code']}{when}: {issue['message']}")
            it.setData(Qt.ItemDataRole.UserRole, issue)
            self.addItem(it)


class LooksPanel(QWidget):
    """Switch the whole scene to a look; see variants of a beat and pick one."""

    changed = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.look = QComboBox()
        self.look.addItems(editor.looks())
        apply = QPushButton("Apply look")
        apply.clicked.connect(
            lambda: self.changed.emit(self.ed.apply_look(self.look.currentText()))
        )
        self.palette = QComboBox()
        self.palette.addItems(["theme's own", *editor.palettes()])
        recolor = QPushButton("Apply palette")
        recolor.clicked.connect(self.apply_palette)
        self.beat = QComboBox()
        show = QPushButton("Show variants")
        show.clicked.connect(self.show_variants)
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.buttons = QHBoxLayout()
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(self.look)
        row.addWidget(apply)
        lay.addLayout(row)
        row_p = QHBoxLayout()
        row_p.addWidget(self.palette)
        row_p.addWidget(recolor)
        lay.addLayout(row_p)
        row2 = QHBoxLayout()
        row2.addWidget(self.beat)
        row2.addWidget(show)
        lay.addLayout(row2)
        lay.addWidget(self.picture)
        lay.addLayout(self.buttons)
        lay.addStretch(1)

    def refresh(self) -> None:
        current = self.beat.currentText()
        self.beat.clear()
        self.beat.addItems(list(self.ed.timeline()["markers"]))
        if current:
            self.beat.setCurrentText(current)
        meta = self.ed.engine.document.meta
        if meta.look:
            self.look.setCurrentText(meta.look)
        self.palette.setCurrentText(meta.palette or "theme's own")

    def apply_palette(self) -> None:
        name = self.palette.currentText()
        self.changed.emit(self.ed.set_palette(None if name == "theme's own" else name))

    def show_variants(self) -> None:
        if not self.beat.currentText():
            return
        r = self.ed.variants(self.beat.currentText())
        self.picture.setPixmap(QPixmap(r["image"]).scaledToWidth(
            max(self.width() - 20, 200), Qt.TransformationMode.SmoothTransformation))  # fmt: skip
        while self.buttons.count():
            w = self.buttons.takeAt(0).widget()
            if w:
                w.deleteLater()
        for v in r["variants"]:
            b = QPushButton(f"{v['name']} ({v['score']:g})")
            b.clicked.connect(lambda _=False, n=v["name"]: self.changed.emit(self.ed.choose(n)))
            self.buttons.addWidget(b)


class ControlsPanel(QWidget):
    """The scene's sliders: moving one previews it in the viewport; Apply makes the values
    part of the scene (one undo step); Reset drops the preview."""

    moved = Signal()
    applied = Signal(object)

    RES = 1000

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.form = QFormLayout()
        holder = QWidget()
        holder.setLayout(self.form)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        apply = QPushButton("Apply to scene")
        apply.clicked.connect(lambda: self.applied.emit(self.ed.apply_controls()))
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        row = QHBoxLayout()
        row.addWidget(apply)
        row.addWidget(reset)
        lay = QVBoxLayout(self)
        lay.addWidget(self.note)
        lay.addWidget(holder)
        lay.addLayout(row)
        lay.addStretch(1)
        self.sliders: dict[str, QSlider] = {}
        self.reads: dict[str, QLabel] = {}

    def refresh(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.sliders, self.reads = {}, {}
        controls = self.ed.controls()
        self.note.setText("" if controls else "This scene has no sliders. Add one in code: "
                          "s.control('object.param', min=0, max=1).")  # fmt: skip
        for c in controls:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, self.RES)
            s.setValue(round((c["value"] - c["min"]) / (c["max"] - c["min"]) * self.RES))
            read = QLabel(f"{c['value']:.4g}")
            s.valueChanged.connect(lambda v, c=c, read=read: self._moved(c, v, read))
            box = QHBoxLayout()
            box.addWidget(s, 1)
            box.addWidget(read)
            self.form.addRow(c["label"], box)
            self.sliders[c["key"]] = s
            self.reads[c["key"]] = read

    def _moved(self, c: dict, v: int, read: QLabel) -> None:
        value = self.ed.set_control(c["key"], c["min"] + v / self.RES * (c["max"] - c["min"]))
        read.setText(f"{value:.4g}")
        self.moved.emit()

    def _reset(self) -> None:
        self.ed.reset_controls()
        self.refresh()
        self.moved.emit()


class PythonHighlighter(QSyntaxHighlighter):
    """Colors for the code view: keywords, strings, numbers, comments, decorators and the
    names a scene script uses (Scene, expr, the object classes)."""

    def __init__(self, document) -> None:
        super().__init__(document)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            f.setFontItalic(italic)
            return f

        kw = "|".join(keyword.kwlist)
        self.rules = [
            (re.compile(rf"\b({kw})\b"), fmt("#c792ea", bold=True)),
            (
                re.compile(r"\b(Scene|expr|geo|cx|pde|calc|algebra|fourier|linalg|ac)\b"),
                fmt("#7aa2f7"),
            ),
            (re.compile(r"\b\d+(\.\d+)?([eE][-+]?\d+)?\b"), fmt("#f78c6c")),
            (re.compile(r"@\w+"), fmt("#ffcb6b")),
            (re.compile(r"(\"[^\"\\]*(\\.[^\"\\]*)*\")|('[^'\\]*(\\.[^'\\]*)*')"), fmt("#c3e88d")),
            (re.compile(r"#[^\n]*"), fmt("#5c6783", italic=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, f in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)


class CodePanel(QWidget):
    """The scene as Python. Edit it and press Run (Ctrl+Enter): it becomes the scene. For a
    script scene this is the script itself (saved on Run); otherwise it is generated from the
    document and kept beside it as ``name.scene.py``."""

    ran = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.text = QPlainTextEdit()
        self.text.setFont(QFont("Monospace", 9))
        self.text.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.highlighter = PythonHighlighter(self.text.document())
        self.status = QLabel("")
        run = QPushButton("Run")
        run.setShortcut("Ctrl+Return")
        run.clicked.connect(self.run)
        self.edited = False
        self._loading = False
        self._note = ""  # the result of the last Run, kept across refreshes
        self.text.textChanged.connect(self._changed)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(run)
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        lay.addWidget(self.text)

    def _changed(self) -> None:
        if not self._loading:
            self.edited = True

    def refresh(self, force: bool = False) -> None:
        """Show the scene's code (unless the person has unsaved edits in the box)."""
        if self.edited and not force:
            self.status.setText("edited: Run to use it")
            return
        code = self.ed.code()
        self._loading = True
        self.text.setPlainText(code["text"])
        self._loading = False
        self.edited = False
        where = ("script: " if code["is_script"] else "generated: ") + code["path"]
        self.status.setText(f"{self._note}   ({where})" if self._note else where)

    def run(self) -> None:
        r = self.ed.run_code(self.text.toPlainText())
        if r["ok"]:
            self._note = "ran: the scene was rebuilt from this code"
            self.refresh(force=True)
        else:
            self._note = ""
            self.status.setText("not run: " + r["error"])
        self.ran.emit(r)


class ActivityDock(QListWidget):
    """Who changed the scene, newest last: your edits and an attached AI's."""

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor

    def refresh(self) -> None:
        self.clear()
        for a in self.ed.activity():
            who = "AI" if a["who"] == "ai" else "you"
            item = QListWidgetItem(f"{who}: {a['what']}")
            item.setForeground(QColor("#7fb8ff" if who == "AI" else "#d8dee9"))
            self.addItem(item)
        self.scrollToBottom()


class EditorWindow(QMainWindow):
    def __init__(self, editor: Editor, attach_hint: str = "") -> None:
        super().__init__()
        self.ed = editor
        self.playing = False
        self.setWindowTitle(f"phasor: {editor.engine.document.meta.title}")
        self.resize(1400, 860)
        self.viewport = Viewport(editor)
        self.timeline = Timeline(editor)
        self.inspector = Inspector(editor)
        self.lint = LintPanel(editor)
        self.looks = LooksPanel(editor)
        self.controls = ControlsPanel(editor)
        self.code = CodePanel(editor)
        self.activity = ActivityDock(editor)
        self.readout = QLabel("")
        self.statusBar().addPermanentWidget(self.readout)
        self._hover: tuple | None = None
        self._hover_done: tuple | None = None
        self._preview_dirty = False
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.viewport)
        left.addWidget(self.timeline)
        left.setStretchFactor(0, 4)
        left.setSizes([620, 180])
        tabs = QTabWidget()
        tabs.addTab(self.inspector, "Inspector")
        tabs.addTab(self.lint, "Lint")
        tabs.addTab(self.looks, "Looks")
        tabs.addTab(self.controls, "Sliders")
        tabs.addTab(self.code, "Code")
        self.panels = _extra_panels(self)
        for title, widget in self.panels:
            tabs.addTab(widget, title)
        dock = QDockWidget("Who changed what", self)
        dock.setWidget(self.activity)
        dock.setObjectName("activity")
        self.activity.setMaximumHeight(110)
        dock.setMaximumHeight(140)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setSizes([1060, 340])
        self.setCentralWidget(split)
        self._actions()
        self.viewport.picked.connect(lambda _: self._after_selection())
        self.viewport.dragged.connect(self._drag)
        self.viewport.turned.connect(self._turn)
        self.viewport.zoomed.connect(lambda f: self._report(self.ed.zoom_camera(f)))
        self.viewport.hovered.connect(lambda u, v, size: setattr(self, "_hover", (u, v, size)))
        self.timeline.scrubbed.connect(self._scrub)
        self.timeline.edited.connect(self._report)
        self.controls.moved.connect(lambda: setattr(self, "_preview_dirty", True))
        self.controls.applied.connect(self._report)
        self.code.ran.connect(self._ran)
        self.inspector.edited.connect(self._report)
        self.lint.picked.connect(self._jump)
        self.looks.changed.connect(self._report)
        self.statusBar().showMessage(attach_hint or "Ready")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.refresh_all()

    def _actions(self) -> None:
        """The toolbar. Its actions are kept in ``self.actions`` so a menu bar can show the
        same ones (one action per shortcut: Qt ignores a shortcut two actions share)."""
        bar = self.addToolBar("edit")
        bar.setObjectName("edit")
        self.actions: dict[str, QAction] = {}
        for text, key, fn in (
            ("Play", "Space", self.toggle_play),
            ("Undo", "Ctrl+Z", lambda: self._report(self.ed.undo())),
            ("Redo", "Ctrl+Shift+Z", lambda: self._report(self.ed.redo())),
            ("Save", "Ctrl+S", lambda: self.save_scene()),
        ):
            act = QAction(text, self)
            act.setShortcut(key)
            act.triggered.connect(fn)
            bar.addAction(act)
            self.actions[text] = act

    def save_scene(self) -> None:
        """Ctrl+S. The desktop app asks where when the scene has never been saved."""
        try:
            self._report(self.ed.save())
        except ToolError as e:
            self.statusBar().showMessage(str(e), 8000)

    # ------------------------------------------------------------ events

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing and self.ed.t >= self.ed.engine.duration:
            self.ed.set_time(0.0)

    def _turn(self, dx: float, dy: float, size: tuple, pan: bool) -> None:
        self._report(self.ed.drag_camera(dx, dy, size, pan))

    def _ran(self, result: dict) -> None:
        if result["ok"]:
            self.setWindowTitle(f"phasor: {self.ed.engine.document.meta.title}")
            self.refresh_all()

    def _show_readout(self) -> None:
        """The value under the pointer (f(z), escape count, coordinates) in the status bar."""
        if self._hover is None or self._hover == self._hover_done:
            return
        self._hover_done = self._hover
        try:
            self.readout.setText(self.ed.readout(*self._hover)["text"])
        except Exception:
            self.readout.setText("")

    def tick(self) -> None:
        self._show_readout()
        if self._preview_dirty:
            self._preview_dirty = False
            self.viewport.refresh()
        if self.playing:
            self.playing = self.ed.play_step(TICK_MS / 1000)
            self.viewport.refresh()
            self.timeline.update()
        if self.ed.changed():  # an edit here or by an attached AI
            self.refresh_all()

    def _scrub(self, t: float) -> None:
        self.ed.set_time(t)
        self.viewport.refresh()
        self.timeline.update()
        self.inspector.refresh()

    def _after_selection(self) -> None:
        self.viewport.refresh()
        self.inspector.refresh()

    def _drag(self, oid: str, u: float, v: float, size: tuple) -> None:
        try:
            self._report(self.ed.drag_to(oid, u, v, size))
        except ToolError as e:
            self.statusBar().showMessage(str(e), 6000)

    def _jump(self, issue: dict) -> None:
        self.ed.jump(issue)
        self.refresh_all()

    def _report(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("ok") is False:
            errs = result.get("errors") or [{"message": "refused"}]
            self.statusBar().showMessage("Not applied: " + errs[0].get("message", ""), 8000)
        elif isinstance(result, dict) and result.get("saved"):
            self.statusBar().showMessage(f"Saved {result['saved']}", 4000)

    def refresh_all(self) -> None:
        self.viewport.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self.lint.refresh()
        self.looks.refresh()
        self.controls.refresh()
        self.code.refresh()
        self.activity.refresh()

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        super().closeEvent(ev)


DARK = """
QWidget { background: #0f1117; color: #d8dee9; }
QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
    background: #161a23; border: 1px solid #262c3a;
}
QPushButton, QToolButton { background: #1b2030; border: 1px solid #2c3346; padding: 3px 8px; }
QPushButton:hover, QToolButton:hover { background: #242b40; }
QTabBar::tab { background: #141822; padding: 5px 12px; }
QTabBar::tab:selected { background: #1f2535; color: #ffb454; }
QStatusBar { color: #8a93a8; }
"""  # the house look is dark, always (STYLE.md); Qt's default is light


def make_workspace(path: str | None, root: str | None = None):
    """A workspace with ``path`` open (a .phasor.json, or a .py builder script: the person
    at the keyboard may open code; an attached AI still may not), or a new scene."""
    from phasor.workspace import Workspace

    ws = Workspace(root or (Path(path).parent if path else Path.cwd()))
    if path:
        ws.allow_scripts = True  # only for this one call, made by the person
        try:
            ws.scene_open(str(Path(path).resolve()))
        finally:
            ws.allow_scripts = False
    else:
        ws.scene_new("untitled")
    return ws


def start_ai_server(ws, lock):
    """Let ``phasor mcp --attach --root <ws.root>`` edit this scene. Returns (server, hint)."""
    from phasor.workspace.remote import RemoteServer

    server = RemoteServer(ws, lock).start()
    server.write_handoff()
    return server, f"An AI can edit this scene: phasor mcp --attach --root {ws.root}"


def qt_app():
    """The one QApplication, in the house look."""
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from phasor.gui.app import ICON

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK)
    app.setApplicationName("phasor")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    return app


def open_editor(path: str | None, root: str | None = None, *, serve_ai: bool = True,
                show: bool = True):  # fmt: skip
    """Open the editor on ``path`` (see ``make_workspace``). Returns (app, window, server)."""
    import threading

    app = qt_app()
    ws = make_workspace(path, root)
    lock = threading.RLock()
    editor = Editor(ws, lock)
    server, hint = None, ""
    if serve_ai:
        server, hint = start_ai_server(ws, lock)
    win = EditorWindow(editor, hint)
    if show:
        win.show()
    return app, win, server
