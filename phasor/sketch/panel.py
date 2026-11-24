"""The Sketch tab of the desktop editor. The editor finds it through the ``phasor.panels``
entry point (sibling clients never import each other): ``make(window)`` returns the tab's
title and widget. It only needs what any window has: ``viewport`` (with ``_size`` and
``_local``), ``ed`` (the Editor: ``engine``, ``ws``, ``t``, ``lock``) and ``refresh_all``.

Tick "Draw on the picture" and draw on the viewport; "Propose" lists what the strokes would
become; tick the ones you want and "Apply". Handwriting shows its
picture; type what it says and it becomes text.
"""

from __future__ import annotations

import itertools
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phasor.sketch.propose import Proposal, Sketch, accept


def make(window: Any) -> tuple[str, QWidget]:
    return "Sketch", SketchPanel(window)


class SketchPanel(QWidget):
    def __init__(self, window: Any) -> None:
        super().__init__()
        self.win = window
        self.strokes: list[list[tuple[float, float]]] = []  # output pixels
        self.proposals: list[Proposal] = []
        self._sketch: Sketch | None = None
        self._base: QPixmap | None = None
        self.draw = QCheckBox("Draw on the picture")
        self.list = QListWidget()
        self.picture = QLabel()
        self.says = QLineEdit()
        self.says.setPlaceholderText("what the handwriting says ($...$ for maths)")
        self.status = QLabel("Draw a loop over a complex function to integrate around it.")
        self.status.setWordWrap(True)
        propose, clear, apply_ = (QPushButton(t) for t in ("Propose", "Clear", "Apply"))
        propose.clicked.connect(self.propose)
        clear.clicked.connect(self.clear)
        apply_.clicked.connect(self.apply)
        self.list.currentRowChanged.connect(self._show)
        row = QHBoxLayout()
        for b in (propose, clear, apply_):
            row.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addWidget(self.draw)
        lay.addLayout(row)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.picture)
        lay.addWidget(self.says)
        lay.addWidget(self.status)
        self._filter = _Pen(self)
        window.viewport.installEventFilter(self._filter)

    # ---------------------------------------------------------- strokes

    def _to_output(self, pos: QPointF) -> tuple[float, float] | None:
        vp = self.win.viewport
        fw, fh = vp._size
        if not fw:
            return None
        u, v = vp._local(pos)
        meta = self.win.ed.engine.document.meta
        return u * meta.width / fw, v * meta.height / fh

    def pen_down(self, pos: QPointF) -> None:
        p = self._to_output(pos)
        if p is not None:
            self._base = self.win.viewport.pixmap().copy() if self._base is None else self._base
            self.strokes.append([p])

    def pen_move(self, pos: QPointF) -> None:
        p = self._to_output(pos)
        if p is not None and self.strokes:
            self.strokes[-1].append(p)
            self._paint()

    def _paint(self) -> None:
        if self._base is None:
            return
        pix = self._base.copy()
        meta = self.win.ed.engine.document.meta
        sx, sy = pix.width() / meta.width, pix.height() / meta.height
        p = QPainter(pix)
        p.setPen(QPen(QColor(255, 210, 120, 230), 2.0))
        for s in self.strokes:
            for (x0, y0), (x1, y1) in itertools.pairwise(s):
                p.drawLine(QPointF(x0 * sx, y0 * sy), QPointF(x1 * sx, y1 * sy))
        p.end()
        self.win.viewport.setPixmap(pix)

    def clear(self) -> None:
        self.strokes, self.proposals, self._base = [], [], None
        self.list.clear()
        self.picture.clear()
        self.win.viewport.refresh()

    # ---------------------------------------------------------- proposals

    def propose(self) -> list[Proposal]:
        ed = self.win.ed
        with ed.lock:
            self._sketch = Sketch(ed.engine, ed.t)
            self.proposals = self._sketch.propose(self.strokes) if self.strokes else []
        self.list.clear()
        for p in self.proposals:
            item = QListWidgetItem(p.summary)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if p.ops else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.status.setText(f"{len(self.proposals)} proposals from {len(self.strokes)} strokes"
                            if self.strokes else "Nothing drawn yet.")  # fmt: skip
        return self.proposals

    def _show(self, row: int) -> None:
        p = self.proposals[row] if 0 <= row < len(self.proposals) else None
        if p is not None and p.image:
            self.picture.setPixmap(QPixmap.fromImage(QImage.fromData(p.image)))
        else:
            self.picture.clear()
        if p is not None and p.data.get("warning"):
            self.status.setText(p.data["warning"])

    def apply(self) -> list[dict[str, Any]]:
        ed = self.win.ed
        results = []
        for row, p in enumerate(self.proposals):
            if self.list.item(row).checkState() != Qt.CheckState.Checked:
                continue
            if p.kind == "handwriting":
                if not self.says.text().strip() or self._sketch is None:
                    continue
                p = self._sketch.transcribe(p, self.says.text().strip())
            with ed.lock:
                results.append(accept(ed.ws, p))
        done = sum(1 for r in results if r.get("ok"))
        self.status.setText(f"applied {done} of {len(results)}")
        self.strokes, self._base = [], None
        self.win.refresh_all()
        return results


class _Pen(QObject):
    """Turns viewport mouse input into strokes while drawing is on (and only then)."""

    def __init__(self, panel: SketchPanel) -> None:
        super().__init__(panel)
        self.panel = panel

    def eventFilter(self, obj, ev) -> bool:
        if not self.panel.draw.isChecked():
            return False
        kind = ev.type()
        if kind == QEvent.Type.MouseButtonPress:
            self.panel.pen_down(ev.position())
            return True
        if kind == QEvent.Type.MouseMove and ev.buttons():
            self.panel.pen_move(ev.position())
            return True
        return kind == QEvent.Type.MouseButtonRelease
