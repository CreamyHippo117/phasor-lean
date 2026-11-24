"""Interactive scenes: sliders on a scene's controls.

A scene marks numbers a viewer may change (``scene.control("set.c", component=0, min=-1.5,
max=1.5)``). Everything interactive builds on ``Player``:

- the desktop editor's Controls tab (``gui``) drives a player,
- ``notebook(engine)`` returns Jupyter widgets (sliders and a picture) over a player,
- ``export_web(engine, "page.html")`` bakes the picture for a grid of slider values into one
  self-contained HTML file with the same sliders (no server, no network).

A player works on a private fork of the engine (``Engine.fork``): moving a slider never edits
the scene, its history, or the saved file. Every picture is still a pure function of
(document, t), here of the document with the slider values set.
"""

from __future__ import annotations

import html
import io
import itertools
import json
from base64 import b64encode
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core.patch import param_value
from phasor.core.schema import Control
from phasor.engine import Engine

DEFAULT_SIZE = (480, 270)
WEB_SIZE = (320, 180)
MAX_WEB_FRAMES = 600


def snap(c: Control, value: float) -> float:
    """``value`` clamped to the control's range and rounded to its step."""
    v = min(max(float(value), c.min), c.max)
    v = c.min + round((v - c.min) / c.step) * c.step
    return round(min(max(v, c.min), c.max), 10)


class Player:
    """The scene with sliders: ``set`` moves one, ``frame`` draws the scene as it now is."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine.fork()
        self.controls: tuple[Control, ...] = tuple(self._engine.document.controls)
        self._by_key = {c.key: c for c in self.controls}
        self._initial = {c.key: self._read(c) for c in self.controls}

    @property
    def engine(self) -> Engine:
        return self._engine

    def _control(self, key: str | Control) -> Control:
        k = key.key if isinstance(key, Control) else key
        try:
            return self._by_key[k]
        except KeyError:
            raise KeyError(f"no control {k!r}; controls are {sorted(self._by_key)}") from None

    def _read(self, c: Control) -> float:
        raw = param_value(self._engine.document, c.path)
        return float(raw if c.component is None else raw[c.component])

    @property
    def values(self) -> dict[str, float]:
        """The current value of every control, by key (``julia.c[0]``)."""
        return {c.key: self._read(c) for c in self.controls}

    def set(self, key: str | Control, value: float) -> float:
        """Move a slider (clamped to its range, rounded to its step); returns the value used."""
        c = self._control(key)
        v = snap(c, value)
        self._write(c, v)
        return v

    def _write(self, c: Control, v: float) -> None:
        if c.component is None:
            new: Any = v
        else:
            cur = list(param_value(self._engine.document, c.path))
            cur[c.component] = v
            new = cur
        r = self._engine.apply({"op": "set_param", "path": c.path, "value": new})
        if not r.ok:
            raise ValueError("; ".join(i.message for i in r.errors))

    def reset(self) -> None:
        """Every slider back to the scene's own values (exactly: not snapped to the step)."""
        for k, v in self._initial.items():
            self._write(self._by_key[k], v)

    def state_at(self, t: float = 0.0):
        return self._engine.state_at(t)

    def frame(self, t: float = 0.0, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
        """The picture at ``t`` with the sliders where they are: (h, w, 3) uint8."""
        return np.ascontiguousarray(self._engine.still(t, size)[..., :3])


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- the notebook


def notebook(engine: Engine, size: tuple[int, int] = DEFAULT_SIZE):
    """Jupyter widgets over a ``Player``: the picture, a slider per control and, when the
    scene has a duration, one for time. Needs ``ipywidgets`` (``uv sync --extra notebook``).
    The returned box has ``.player`` for scripting."""
    try:
        import ipywidgets as w
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "the notebook widget needs ipywidgets: uv sync --extra notebook"
        ) from e
    player = Player(engine)
    picture = w.Image(format="png")
    sliders: dict[str, Any] = {}
    time = w.FloatSlider(value=0.0, min=0.0, max=max(engine.duration, 0.0), step=1 / 30,
                         description="t", continuous_update=False,
                         readout_format=".2f", layout=w.Layout(width="480px"))  # fmt: skip

    def redraw(*_: Any) -> None:
        picture.value = png_bytes(player.frame(time.value, size))

    def moved(change: dict[str, Any], key: str) -> None:
        player.set(key, change["new"])
        redraw()

    for c in player.controls:
        s = w.FloatSlider(value=player.values[c.key], min=c.min, max=c.max, step=c.step,
                          description=c.label or c.key, continuous_update=False,
                          readout_format=".3f", style={"description_width": "90px"},
                          layout=w.Layout(width="480px"))  # fmt: skip
        s.observe(lambda ch, k=c.key: moved(ch, k), names="value")
        sliders[c.key] = s
    time.observe(redraw, names="value")
    box = w.VBox([picture, *sliders.values(), *([time] if engine.duration > 0 else [])])
    box.player, box.sliders, box.picture, box.time = player, sliders, picture, time  # type: ignore[attr-defined]
    redraw()
    return box


# ---------------------------------------------------------------- the web export


def _axis_values(c: Control, n: int) -> list[float]:
    vals = {snap(c, v) for v in np.linspace(c.min, c.max, n)}
    return sorted(vals)


def export_web(
    engine: Engine,
    path: str | Path,
    *,
    size: tuple[int, int] = WEB_SIZE,
    steps: int | dict[str, int] = 5,
    time_steps: int = 1,
    max_frames: int = MAX_WEB_FRAMES,
) -> dict[str, Any]:
    """Bake the scene into one self-contained HTML page with its sliders.

    The page holds a picture for every combination of slider values (``steps`` samples of
    each control, an int or a dict by control key; plus ``time_steps`` samples of the scene's
    time if more than 1) and shows the nearest as you drag; dragging on the picture moves the
    first two sliders. Baked pictures work everywhere with no
    server, at the cost of a coarse grid (the page says how many steps each slider has).
    """
    player = Player(engine)
    axes: list[dict[str, Any]] = []
    for c in player.controls:
        n = steps.get(c.key, 5) if isinstance(steps, dict) else steps
        vals = _axis_values(c, max(int(n), 2))
        axes.append({"kind": "control", "key": c.key, "label": c.label or c.key, "values": vals})
    if time_steps > 1 and engine.duration > 0:
        ts = [round(float(t), 6) for t in np.linspace(0, engine.duration, time_steps)]
        axes.append({"kind": "time", "key": "t", "label": "time", "values": ts})
    total = int(np.prod([len(a["values"]) for a in axes])) if axes else 1
    if total > max_frames:
        raise ValueError(
            f"{total} pictures would be baked (limit {max_frames}): use fewer steps "
            f"({[(a['key'], len(a['values'])) for a in axes]}) or a smaller scene"
        )
    frames: list[str] = []
    t = 0.0
    for combo in itertools.product(*[range(len(a["values"])) for a in axes]):
        for a, i in zip(axes, combo, strict=True):
            if a["kind"] == "control":
                player.set(a["key"], a["values"][i])
            else:
                t = a["values"][i]
        frames.append(
            "data:image/png;base64," + b64encode(png_bytes(player.frame(t, size))).decode()
        )
    meta = [{"label": a["label"], "values": a["values"]} for a in axes]
    page = _PAGE.replace("__TITLE__", html.escape(engine.document.meta.title))
    page = page.replace(
        "__DATA__", json.dumps({"axes": meta, "frames": frames, "size": list(size)})
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {"path": str(out), "frames": len(frames), "bytes": out.stat().st_size,
            "axes": [{"label": a["label"], "steps": len(a["values"])} for a in axes]}  # fmt: skip


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #07080b; color: #d8dee9; font: 14px system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 16px; gap: 12px; }
  img { max-width: min(96vw, 960px); width: 100%; image-rendering: auto; cursor: crosshair;
        border: 1px solid #1d2330; touch-action: none; }
  .row { display: grid; grid-template-columns: 9em 1fr 6em; gap: 8px; align-items: center;
         width: min(96vw, 960px); }
  input[type=range] { width: 100%; accent-color: #ffb454; }
  small { color: #7d869c; }
</style></head><body>
<h3 style="margin:0">__TITLE__</h3>
<img id="pic" alt="the scene" draggable="false">
<div id="controls"></div>
<small id="hint"></small>
<script>
const D = __DATA__;
const pic = document.getElementById("pic"), box = document.getElementById("controls");
const pos = D.axes.map(a => Math.floor(a.values.length / 2));
const sliders = [], reads = [];
function index() {
  let i = 0;
  D.axes.forEach((a, k) => { i = i * a.values.length + pos[k]; });
  return i;
}
function draw() {
  pic.src = D.frames[index()];
  D.axes.forEach((a, k) => { sliders[k].value = pos[k]; reads[k].textContent = a.values[pos[k]]; });
}
D.axes.forEach((a, k) => {
  const row = document.createElement("div"); row.className = "row";
  const label = document.createElement("span"); label.textContent = a.label;
  const s = document.createElement("input"); s.type = "range"; s.min = 0;
  s.max = a.values.length - 1; s.step = 1;
  s.oninput = () => { pos[k] = +s.value; draw(); };
  const r = document.createElement("span");
  row.append(label, s, r); box.append(row); sliders.push(s); reads.push(r);
});
function drag(ev) {
  if (D.axes.length < 1 || !(ev.buttons || ev.type === "pointerdown")) return;
  const b = pic.getBoundingClientRect();
  const fx = Math.min(Math.max((ev.clientX - b.left) / b.width, 0), 1);
  const fy = 1 - Math.min(Math.max((ev.clientY - b.top) / b.height, 0), 1);
  pos[0] = Math.round(fx * (D.axes[0].values.length - 1));
  if (D.axes.length > 1 && D.axes[1].values.length > 1)
    pos[1] = Math.round(fy * (D.axes[1].values.length - 1));
  draw();
}
pic.addEventListener("pointerdown", drag); pic.addEventListener("pointermove", drag);
document.getElementById("hint").textContent =
  D.frames.length + " pictures, baked in advance: " +
  D.axes.map(a => a.label + " " + a.values.length + " steps").join(", ") +
  (D.axes.length > 1 ? ". Drag on the picture to move the first two sliders." : ".");
draw();
</script></body></html>
"""
