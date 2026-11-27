"""The editor window, Qt (PySide6) widgets over ``controller.Editor``.

Nothing here decides anything: widgets draw what the ``Editor`` holds and forward clicks and
typing to it. A 30 Hz timer advances playback and notices edits made by an attached AI
(``phasor mcp --attach``), so both editors always show the same live scene.

Layout: the viewport (click to select, drag a selected object to move it) above the
timeline (markers, animated spans, the playhead: click or drag to scrub); on the right,
tabs for the inspector (parameters, style roles, what the inspector reads), lint (click an
issue to jump to it), looks (switch look, pick among variants of a beat), sliders, code and
sound (music, effects, the narration, mastering, listening). Play plays the sound
too, and the playhead follows it.
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
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
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
    QTableWidget,
    QTableWidgetItem,
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
    """Narration (with its waveform), the sound (one row per bed, one for the cues), markers,
    the animated spans of every parameter (one lane each) and the playhead. Click or drag to
    scrub; drag a marker to move it; drag a span to slide it, or its ends to stretch it; drag
    a sentence, a bed or a cue to move it (a cue snaps to the nearest marker)."""

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
        return self._sound_y() - 20 + self.LANE * self._sound_rows()

    def _sound_rows(self) -> int:
        snd = self.data
        return len(snd.get("beds", [])) + (1 if snd.get("cues") else 0)

    def _sound_y(self) -> int:
        """Where the first sound row starts: under the narration, or under the marker names."""
        return self.NARRATION + 2 if self.data.get("narration") else 20

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
            start = b["start"]
            if self._drag and self._drag[0] == "sentence" and self._drag[1] == b["index"]:
                start = self._drag[-1]
            x0, x1 = self._x(start), self._x(start + b["duration"])
            box = QRectF(x0, 16, max(x1 - x0, 3), self.NARRATION - 18)
            self._hits.append((box, "sentence", (b["index"], start)))
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
        self._paint_sound(p)
        p.fillRect(QRectF(0, 0, self.LABELS - 4, self.height()), QColor("#07080b"))
        self._paint_sound_names(p)
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

    def _paint_sound(self, p: QPainter) -> None:
        snd = self.data
        y = self._sound_y()
        for bed in snd.get("beds", []):
            t0, t1 = bed["start"], bed["end"]
            if self._drag and self._drag[0] == "bed" and self._drag[1] == bed["id"]:
                t0, t1 = self._drag[-1], self._drag[-1] + (t1 - t0)
            r = QRectF(self._x(t0), y + 2, max(self._x(t1) - self._x(t0), 3), self.LANE - 4)
            p.fillRect(r, QColor(122, 215, 255, 70))
            p.setPen(QColor("#7ad7ff"))
            p.drawText(r.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(bed["id"], Qt.TextElideMode.ElideRight,
                                                  max(int(r.width()) - 4, 0)))  # fmt: skip
            self._hits.append((r, "bed", (bed["id"], t0)))
            y += self.LANE
        p.setPen(QPen(QColor("#ff8fd0"), 1))
        for cue in snd.get("cues", []):
            t = cue["t"]
            if self._drag and self._drag[0] == "cue" and self._drag[1] == cue["id"]:
                t = self._drag[-1]
            x, cy, h = self._x(t), y + self.LANE / 2, self.LANE / 2 - 2
            p.setBrush(QColor("#ff8fd0"))
            p.drawPolygon([QPointF(x, cy - h), QPointF(x + h, cy), QPointF(x, cy + h),
                           QPointF(x - h, cy)])  # fmt: skip
            p.setBrush(Qt.BrushStyle.NoBrush)
            self._hits.append((QRectF(x - h - 2, cy - h - 2, 2 * h + 4, 2 * h + 4), "cue",
                               (cue["id"], t)))  # fmt: skip

    def _paint_sound_names(self, p: QPainter) -> None:
        snd = self.data
        y = self._sound_y()
        names = [("music: " + b["id"], "#7ad7ff") for b in snd.get("beds", [])]
        if snd.get("cues"):
            names.append(("effects", "#ff8fd0"))
        for text, color in names:
            p.setPen(QColor(color))
            p.drawText(QRectF(6, y, self.LABELS - 12, self.LANE), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight,
                                                  self.LABELS - 12))  # fmt: skip
            y += self.LANE

    def _hit(self, pos: QPointF):
        for want in ("marker", "cue"):  # thin things first, on top of spans and bars
            for rect, kind, data in self._hits:
                if kind == want and rect.contains(pos):
                    return kind, data
        for rect, kind, data in self._hits:
            if kind == "span" and rect.adjusted(-self.EDGE // 2, 0, self.EDGE // 2, 0).contains(
                pos
            ):
                return kind, data
        for rect, kind, data in self._hits:
            if kind in ("bed", "sentence") and rect.contains(pos):
                return kind, data
        return None

    def mousePressEvent(self, ev) -> None:
        hit = self._hit(ev.position())
        if hit and hit[0] == "marker":
            name, t = hit[1]
            self._drag = ("marker", name, t, t)
        elif hit and hit[0] in ("cue", "bed", "sentence"):
            key, t = hit[1]  # (what, which, where it grabbed, where it started, where it is)
            self._drag = (hit[0], key, self._t(ev.position().x()), t, t)
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
        elif d[0] in ("cue", "bed", "sentence"):
            kind, key, grab, start, _cur = d
            self._drag = (kind, key, grab, start, max(0.0, start + t - grab))
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
        elif d[0] in ("cue", "bed", "sentence") and abs(d[4] - d[3]) > 1e-3:
            move = {"cue": self.ed.move_cue, "bed": self.ed.move_bed,
                    "sentence": self.ed.move_sentence}[d[0]]  # fmt: skip
            self.edited.emit(move(d[1], d[4]))
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


class NarrateDialog(QDialog):
    """Write the narration: one sentence per line, ``[[name]]`` where something should
    happen (the timeline gets a marker there), and pick a voice."""

    def __init__(self, parent: QWidget | None, script: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Narration")
        self.text = QPlainTextEdit(script)
        self.text.setPlaceholderText("Here is a pole [[pole_appears]].\n"
                                     "Watch it move [[pole_moves]] to the right.")  # fmt: skip
        self.voice = QComboBox()
        self.voice.addItems(["auto", "espeak", "estimate"])
        self.voice.setToolTip("auto: espeak-ng when installed, otherwise timing only "
                              "(no audio); estimate: timing only")  # fmt: skip
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("One sentence per line; [[name]] marks a moment."))
        lay.addWidget(self.text)
        row = QHBoxLayout()
        row.addWidget(QLabel("Voice"))
        row.addWidget(self.voice, 1)
        lay.addLayout(row)
        lay.addWidget(buttons)
        self.resize(560, 320)


class SoundPanel(QWidget):
    """The scene's sound: the voice, every bed and cue with its level and gain,
    mute and solo for listening, and the ways to add to it: music or ambience from a file,
    an effect at the playhead, the narration. "Master" turns down whatever covers the voice
    or clips. Drag beds, cues and sentences on the timeline to move them."""

    changed = Signal(object)
    COLUMNS = ("Lane", "LUFS", "vs voice", "Mute", "Solo", "")

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.issues = QLabel("")
        self.issues.setWordWrap(True)
        self.issues.setStyleSheet("color: #ffb454;")
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        for col, width in enumerate((130, 48, 72, 40, 40, 28)):
            self.table.setColumnWidth(col, width)
        self.music = QPushButton("Add music...")
        self.music.setToolTip("Your own music or ambience (WAV; OGG, FLAC, MP3 with the sound "
                              "extra), copied next to the scene, looped under it")  # fmt: skip
        self.music.clicked.connect(lambda: self.add_music())
        self.effect = QComboBox()
        from phasor.core.schema import BUILTIN_SOUNDS

        self.effect.addItems([*BUILTIN_SOUNDS, "a file..."])
        add_cue = QPushButton("Add effect at playhead")
        add_cue.clicked.connect(lambda: self.add_effect())
        narrate = QPushButton("Narrate (synthetic)...")
        narrate.setToolTip("Type the script and let a synthetic voice speak it")
        narrate.clicked.connect(lambda: self.narrate())
        self.voice_button = QPushButton("Add voice...")
        self.voice_button.setToolTip("A recording of your voice, at the playhead: levelled "
                                     "to the voice level, music ducks under it")  # fmt: skip
        self.voice_button.clicked.connect(lambda: self.add_voice())
        mark = QPushButton("Marker at playhead")
        mark.setToolTip("Name a moment (a word in your recording) to tie motion and effects "
                        "to; inside a recording it moves with it")  # fmt: skip
        mark.clicked.connect(lambda: self.add_marker())
        self.master_button = QPushButton("Master")
        self.master_button.setToolTip("Turn down whatever covers the voice or clips, then "
                                      "check again (each round can be undone)")  # fmt: skip
        self.master_button.clicked.connect(lambda: self.changed.emit(self.ed.master()))
        levels = QPushButton("Show levels")
        levels.clicked.connect(self.show_levels)
        from phasor.gui.audio import backend

        self.listen = QCheckBox("Play with sound")
        self.listen.setChecked(backend() is not None)
        self.listen.setEnabled(backend() is not None)
        if backend() is None:
            self.listen.setToolTip("No way to play sound found on this system (Linux: install "
                                   "pulseaudio-utils or alsa-utils)")  # fmt: skip
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.status)
        lay.addWidget(self.issues)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.voice_button)
        row.addWidget(self.music)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(mark)
        row.addWidget(narrate)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.effect)
        row.addWidget(add_cue)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.master_button)
        row.addWidget(levels)
        row.addWidget(self.listen)
        lay.addLayout(row)
        lay.addWidget(self.picture)
        self.data: dict[str, Any] = {}
        self.gains: dict[str, float] = {}  # what each bed's and cue's spin box was set to

    def refresh(self) -> None:
        try:
            self.data = d = self.ed.sound()
        except ToolError as e:
            self.status.setText(str(e))
            return
        if d.get("scripted"):
            self.status.setText("This scene is built by a script. Your first sound edit "
                                "makes an editable copy beside it; the script stays.")  # fmt: skip
        elif not d.get("lanes"):
            self.status.setText("No sound yet: add your voice, music or an effect.")
        else:
            under = [u for x in d["lanes"] for u in x.get("under_voice_lu", []) if u is not None]
            parts = [f"mix {d.get('loudness')} LUFS, true peak {d.get('true_peak_dbtp')} dBTP"]
            if under:
                parts.append(f"music at least {min(under):g} dB under the voice")
            self.status.setText("; ".join(parts))
        self.issues.setText("\n".join(i["message"] for i in d.get("issues", [])))
        self.table.setRowCount(0)
        defaults = d.get("defaults", {})
        gains = {x["id"]: (x["gain"] if x["gain"] is not None else defaults.get(kind, 0.0))
                 for kind, items in (("bed", d.get("beds", [])), ("cue", d.get("cues", [])))
                 for x in items}  # fmt: skip
        self.gains = gains
        for lane in d.get("lanes", []):
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = QTableWidgetItem(f"{lane['id']} ({lane['kind']})")
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, name)
            level = QTableWidgetItem("" if lane["loudness"] is None else f"{lane['loudness']:g}")
            level.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 1, level)
            if lane["id"] in gains:
                spin = QDoubleSpinBox()
                spin.setRange(-60.0, 6.0)
                spin.setToolTip("loudness in LU relative to the voice")
                spin.setSingleStep(1.0)
                spin.setValue(gains[lane["id"]])
                spin.editingFinished.connect(
                    lambda s=spin, i=lane["id"]: self._gain(i, s.value()))  # fmt: skip
                self.table.setCellWidget(r, 2, spin)
            switches = ((3, lane["id"] in d.get("muted", []), self.ed.toggle_mute),
                        (4, lane["id"] in d.get("solo", []), self.ed.toggle_solo))  # fmt: skip
            for col, on, flip in switches:
                box = QCheckBox()
                box.setChecked(on)
                box.toggled.connect(lambda _=False, i=lane["id"], f=flip: f(i))
                cell = QWidget()
                centre = QHBoxLayout(cell)
                centre.setContentsMargins(0, 0, 0, 0)
                centre.setAlignment(Qt.AlignmentFlag.AlignCenter)
                centre.addWidget(box)
                self.table.setCellWidget(r, col, cell)
            if lane["id"] in gains:
                drop = QPushButton("x")
                drop.setToolTip(f"remove {lane['id']}")
                drop.clicked.connect(
                    lambda _=False, i=lane["id"]: self.changed.emit(self.ed.remove_sound(i))
                )
                self.table.setCellWidget(r, 5, drop)
        self._voice_rows()
        self.master_button.setEnabled(bool(d.get("beds") or d.get("cues")))

    def _voice_rows(self) -> None:
        """One row per recorded or spoken sentence, under the lanes: what it says, where,
        and a button to take it out."""
        n = self.ed.engine.document.narration
        for k, b in enumerate(n.blocks if n else ()):
            if not b.text:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = QTableWidgetItem(f"  {b.text}")
            item.setToolTip(f"{b.audio or 'no audio'}, {b.start:g} s, {b.duration:g} s long")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, item)
            at = QTableWidgetItem(f"{b.start:.1f}s")
            at.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 1, at)
            drop = QPushButton("x")
            drop.setToolTip("take this sentence out")
            drop.clicked.connect(lambda _=False, i=k: self.changed.emit(self.ed.remove_sentence(i)))
            self.table.setCellWidget(r, 5, drop)

    def add_voice(self, path: str | None = None, text: str | None = None) -> None:
        path = path or self._pick_file("A recording of your voice")
        if not path:
            return
        if text is None:
            from PySide6.QtWidgets import QInputDialog

            text, ok = QInputDialog.getText(self, "Voice", "What it says (optional):")
            if not ok:
                return
        self.changed.emit(self.ed.add_voice(path, text=text))

    def add_marker(self, name: str | None = None) -> None:
        if name is None:
            from PySide6.QtWidgets import QInputDialog

            name, ok = QInputDialog.getText(self, "Marker", "Name (letters, digits, _):")
            if not ok or not name:
                return
        self.changed.emit(self.ed.add_marker(name.strip()))

    def _gain(self, item_id: str, value: float) -> None:
        if abs(value - self.gains.get(item_id, value)) > 1e-9:
            self.changed.emit(self.ed.update_sound(item_id, gain=round(value, 1)))

    def _pick_file(self, title: str) -> str | None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, title, str(self.ed.ws.root),
                                              "Sound (*.wav *.ogg *.flac *.mp3)")  # fmt: skip
        return path or None

    def add_music(self, path: str | None = None) -> None:
        path = path or self._pick_file("Music or ambience")
        if path:
            self.changed.emit(self.ed.add_bed(path))

    def add_effect(self, path: str | None = None) -> None:
        name = self.effect.currentText()
        if name == "a file...":
            name = path or self._pick_file("A sound effect")
            if not name:
                return
        self.changed.emit(self.ed.add_cue(name))

    def narrate(self, script: str | None = None, voice: str = "auto") -> None:
        if script is None:
            n = self.ed.engine.document.narration
            dialog = NarrateDialog(self, n.script if n else "")
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            script, voice = dialog.text.toPlainText(), dialog.voice.currentText()
        self.changed.emit(self.ed.narrate(script, voice=voice))

    def show_levels(self) -> None:
        path = self.ed.sound_picture()
        if path:
            width = max(self.width() - 20, 200)
            smooth = Qt.TransformationMode.SmoothTransformation
            self.picture.setPixmap(QPixmap(path).scaledToWidth(width, smooth))


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
        self.sound = SoundPanel(editor)
        self.activity = ActivityDock(editor)
        from phasor.gui.audio import Player

        self.player = Player()
        self._heard_from: tuple[float, float] | None = None  # (clock, scene time) of Play
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
        tabs.addTab(self.sound, "Sound")
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
        self.sound.changed.connect(self._report)
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
        self._sound_follows()

    def _sound_follows(self) -> None:
        """Start (or restart, after a scrub) the scene's sound from the playhead while
        playing with sound on; stop it otherwise. The playhead then follows the clock, so
        picture and sound stay together."""
        import time

        self.player.stop()
        self._heard_from = None
        if not (self.playing and self.sound.listen.isChecked()):
            return
        try:
            path = self.ed.listen_file()
        except Exception as e:  # a missing narration clip: play the picture without sound
            self.statusBar().showMessage(f"No sound: {e}", 6000)
            return
        if path is not None and self.player.play(path):
            self._heard_from = (time.monotonic(), self.ed.t)

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
            if self._heard_from is not None:
                import time

                clock, t0 = self._heard_from
                self.ed.set_time(t0 + time.monotonic() - clock)
                self.playing = self.ed.t < self.ed.engine.duration
            else:
                self.playing = self.ed.play_step(TICK_MS / 1000)
            if not self.playing:
                self._sound_follows()
            self.viewport.refresh()
            self.timeline.update()
        if self.ed.changed():  # an edit here or by an attached AI
            self.refresh_all()

    def _scrub(self, t: float) -> None:
        self.ed.set_time(t)
        if self.playing:
            self._sound_follows()
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
        if self.ed.copied and self.ed.copied != getattr(self, "_copy_told", None):
            self._copy_told = self.ed.copied
            self.statusBar().showMessage(f"Editing a copy: {self.ed.copied} (the script is "
                                         "unchanged)", 10000)  # fmt: skip
        self.viewport.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self.lint.refresh()
        self.looks.refresh()
        self.controls.refresh()
        self.code.refresh()
        self.sound.refresh()
        self.activity.refresh()

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        self.player.stop()
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
QCheckBox::indicator, QTableWidget QCheckBox::indicator {
    width: 12px; height: 12px; border: 1px solid #5a6478; background: #161a23;
}
QCheckBox::indicator:checked { background: #ffb454; border-color: #ffb454; }
QHeaderView::section { background: #141822; color: #8a93a8; border: 0; padding: 3px; }
QTableWidget { gridline-color: #1d2230; }
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
