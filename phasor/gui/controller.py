"""The editor's logic, with no GUI toolkit in it.

``Editor`` holds what a person looking at a scene has in mind: the time under the playhead,
the selected object, the panels' contents. Every change goes through the shared
``Workspace`` and so through ``Engine.apply`` (contract 1): one undo step each, recorded as a
parameter or style change (the AI reading the scene sees intent, not pixels). The Qt window
(``window.py``) only draws this and forwards clicks, so all of it is tested without Qt.

The same ``Workspace`` is served to ``phasor mcp --attach`` over ``workspace.remote``; the
editor and the AI share one lock, and ``changed()`` tells the window when the AI has edited.
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from phasor.engine import Engine
from phasor.workspace import ToolError, Workspace


def parse_value(text: str) -> Any:
    """What a person types into a field: JSON (``0.3``, ``[1, 2, 0]``, ``true``,
    ``{"expr": "t / 2"}``), else the text itself (``sin(x)``, ``D4``)."""
    try:
        return json.loads(text)
    except ValueError:
        return text


class Editor:
    def __init__(self, workspace: Workspace, lock: threading.RLock | None = None) -> None:
        self.ws = workspace
        self.lock = lock or threading.RLock()
        self.t = 0.0
        self.selected: str | None = None
        self.history: list[dict[str, Any]] = []  # what the person did here, newest last
        self._seen = self.version
        self.preview: dict[str, float] = {}  # slider values shown but not yet in the document
        self._player: tuple[int, Any] | None = None  # (version, Player) for previews
        self._frame: tuple[tuple, np.ndarray] | None = None
        self._lint: tuple[int, list[dict[str, Any]]] | None = None
        self.saved_version = self.version  # what is on disk (unsaved edits: ``dirty``)

    # ------------------------------------------------------------ the scene

    @property
    def engine(self) -> Engine:
        return self.ws.engine

    @property
    def version(self) -> int:
        try:
            return self.engine.version
        except ToolError:
            return -1

    def changed(self) -> bool:
        """True once after anything (here or an attached AI) changed the scene."""
        v = self.version
        if v != self._seen:
            self._seen = v
            if self.selected and self.selected not in self._ids():
                self.selected = None
            return True
        return False

    def _ids(self) -> set[str]:
        return {o.id for o in self.engine.document.objects}

    def _edit(self, patch: Any, what: str) -> dict[str, Any]:
        with self.lock:
            r = self.ws.scene_edit(patch)
        if r["ok"]:
            entry = {"what": what, "version": self.version, "t": self.t}
            self.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
        return r

    def activity(self) -> list[dict[str, Any]]:
        """Everything that changed the scene, oldest first: ``who`` is "you" or "ai"."""
        return list(self.ws.activity)

    @staticmethod
    def _refused(message: str) -> dict[str, Any]:
        return {"ok": False, "version": -1, "errors": [{"code": "refused", "message": message}]}

    def undo(self) -> dict[str, Any]:
        with self.lock:
            return self.ws.undo()

    def redo(self) -> dict[str, Any]:
        with self.lock:
            return self.ws.redo()

    def save(self, path: str | None = None) -> dict[str, Any]:
        with self.lock:
            r = self.ws.scene_save(path)
            self.saved_version = self.version
            return r

    @property
    def path(self) -> Path | None:
        """Where the scene is saved (None until the first save)."""
        return self.ws.path

    @property
    def dirty(self) -> bool:
        """Whether there are edits not saved yet (yours or an attached AI's)."""
        return self.version != self.saved_version

    def save_as(self, path: str | Path) -> dict[str, Any]:
        """Save to ``path`` anywhere the person chooses (a file dialog): the workspace
        folder moves there, so an attached AI follows the scene. ``.phasor.json`` is added
        when the name lacks it."""
        p = Path(path).resolve()
        if not p.name.endswith(".phasor.json"):
            p = p.with_name(p.name.removesuffix(".json") + ".phasor.json")
        with self.lock:
            if p.parent != self.ws.root and self.ws.root not in p.parents:
                self.ws.root = p.parent
            return self.save(str(p))

    # ------------------------------------------------------------ time

    def timeline(self) -> dict[str, Any]:
        with self.lock:
            return self.engine.timeline()

    def set_time(self, t: float) -> float:
        self.t = float(min(max(t, 0.0), max(self.engine.duration, 0.0)))
        return self.t

    def move_marker(self, name: str, t: float) -> dict[str, Any]:
        """Drag a marker to a new time (animations anchored to it follow)."""
        if name not in self.timeline()["markers"]:
            return self._refused(f"no marker {name!r}")
        return self._edit({"op": "set_marker", "name": name, "t": max(0.0, float(t))},
                          f"marker {name} -> {t:.3g}s")  # fmt: skip

    def retime_segment(self, path: str, index: int, t0: float, t1: float) -> dict[str, Any]:
        """Drag one animation span to new start and end times (the value it moves to stays)."""
        segs = self.engine.document.timeline.tracks.get(path)
        if segs is None or not 0 <= index < len(segs):
            return self._refused(f"{path} has no span number {index}")
        if t1 <= t0 or t0 < 0:
            return self._refused("a span needs 0 <= start < end")
        from phasor.core.schema import Segment

        old = segs[index]
        moved = Segment(t0=float(t0), t1=float(t1), to=old.to, ease=old.ease, verb=old.verb)
        new = [moved if i == index else seg for i, seg in enumerate(segs)]
        ops: list[dict[str, Any]] = [{"op": "clear_track", "path": path}]
        ops += [{"op": "add_segment", "path": path, "segment": seg.model_dump(mode="json")}
                for seg in new]  # fmt: skip
        return self._edit(ops, f"{path} span {index} -> {t0:.3g}..{t1:.3g}s")

    def waveform(self, audio: str, bins: int = 200) -> list[float]:
        """Loudness of a narration audio file in ``bins`` steps (0 to 1, of full scale); an
        empty list when the file is missing or not 16-bit WAV."""
        import wave

        path = Path(self.engine.base_dir) / audio
        try:
            with wave.open(str(path), "rb") as f:
                if f.getsampwidth() != 2:
                    return []
                ch = f.getnchannels()
                data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        except (OSError, wave.Error, EOFError):
            return []
        if ch > 1:
            data = data[::ch]
        if data.size == 0:
            return []
        edges = np.linspace(0, data.size, bins + 1).astype(int)
        return [float(np.abs(data[a:b]).max()) / 32768.0 if b > a else 0.0
                for a, b in itertools.pairwise(edges)]  # fmt: skip

    def jump_to_marker(self, name: str) -> float:
        return self.set_time(self.timeline()["markers"][name])

    def play_step(self, dt: float) -> bool:
        """Advance the playhead; False when it reached the end."""
        end = self.engine.duration
        self.set_time(self.t + dt)
        return self.t < end

    # ------------------------------------------------------------ the viewport

    def frame(self, size: tuple[int, int]) -> np.ndarray:
        """The picture at the playhead, (h, w, 3) uint8, cached until time, size or the
        scene changes. Needs OpenGL."""
        key = (self.version, round(self.t, 6), tuple(size), tuple(sorted(self.preview.items())))
        if self._frame is None or self._frame[0] != key:
            with self.lock:
                if self.preview:
                    img = self._preview_player().frame(self.t, size)
                else:
                    img = np.ascontiguousarray(self.engine.still(self.t, size)[..., :3])
            self._frame = (key, img)
        return self._frame[1]

    # ------------------------------------------------------------ sliders (the scene's controls)

    def _preview_player(self):
        from phasor.interactive import Player

        if self._player is None or self._player[0] != self.version:
            self._player = (self.version, Player(self.engine))
        player = self._player[1]
        for key, v in self.preview.items():
            player.set(key, v)
        return player

    def controls(self) -> list[dict[str, Any]]:
        """The scene's sliders with their current values (a preview, if one is set)."""
        from phasor.core.patch import param_value

        out = []
        for c in self.engine.document.controls:
            cur = param_value(self.engine.document, c.path)
            value = float(cur if c.component is None else cur[c.component])
            out.append({"key": c.key, "label": c.label or c.key, "min": c.min, "max": c.max,
                        "step": c.step, "value": self.preview.get(c.key, value)})  # fmt: skip
        return out

    def set_control(self, key: str, value: float) -> float:
        """Move a slider: the viewport shows it, the scene is not edited (until
        ``apply_controls``). Returns the value used (clamped, on the step)."""
        from phasor.interactive import snap

        ctl = next((c for c in self.engine.document.controls if c.key == key), None)
        if ctl is None:
            raise ToolError(f"no slider {key!r}")
        self.preview[key] = snap(ctl, value)
        return self.preview[key]

    def reset_controls(self) -> None:
        self.preview = {}

    def apply_controls(self) -> dict[str, Any]:
        """Make the slider values part of the scene: one undoable edit."""
        if not self.preview:
            return {"ok": True, "version": self.version, "errors": []}
        from phasor.core.patch import param_value

        doc = self.engine.document
        by_path: dict[str, Any] = {}
        for key, v in self.preview.items():
            ctl = next(c for c in doc.controls if c.key == key)
            cur = by_path.get(ctl.path, param_value(doc, ctl.path))
            if ctl.component is None:
                by_path[ctl.path] = v
            else:
                new = list(cur)
                new[ctl.component] = v
                by_path[ctl.path] = new
        ops = [{"op": "set_param", "path": p, "value": v} for p, v in by_path.items()]
        what = "sliders: " + ", ".join(f"{k} = {v:g}" for k, v in self.preview.items())
        r = self._edit(ops, what)
        if r["ok"]:
            self.preview = {}
        return r

    # ------------------------------------------------------------ readouts (hover)

    def readout(self, u: float, v: float, size: tuple[int, int],
                hit: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
        """What is under the pointer: its plane coordinates and, for a function plot or a
        fractal, the value there (f(z), |f|, arg f; the escape count of a Julia/Mandelbrot
        point). ``hit`` (object ids under the point, topmost first) defaults to a hit test."""
        import cmath

        x, y, _ = self.screen_to_world(u, v, size)
        if hit is None:
            ox, oy = self._to_output(u, v, size)
            with self.lock:
                hit = self.engine.hit_test(ox, oy, self.t)
        out: dict[str, Any] = {"x": x, "y": y, "object": hit[0] if hit else None,
                               "text": f"x = {x:.4g}, y = {y:.4g}"}  # fmt: skip
        if not hit:
            return out
        spec = self.engine.document.object(hit[0])
        st = self.engine.state_at(self.t)[hit[0]]
        if spec is None:
            return out
        p = st.params
        if spec.type == "domain_coloring":
            from phasor.domains.cx import _parse

            z = complex(x - p.pos[0], y - p.pos[1])
            try:
                f = complex(_parse(p.function.strip()).numpy()(np.array([z]))[0])
            except (ValueError, ZeroDivisionError, FloatingPointError):
                f = complex("nan")
            out.update(z=z, f=f, abs=abs(f), arg=cmath.phase(f) if f == f else float("nan"))
            out["text"] = (f"z = {z.real:.4g}{z.imag:+.4g}i   f(z) = {f.real:.4g}{f.imag:+.4g}i   "
                           f"|f| = {abs(f):.4g}   arg f = {out['arg']:.4g}")  # fmt: skip
        elif spec.type == "fractal":
            zc = complex(x - p.pos[0], y - p.pos[1])
            c, z = (complex(*p.c), zc) if p.kind == "julia" else (zc, 0j)
            n = None
            for k in range(p.max_iter):
                z = z * z + c
                if abs(z) > 256:
                    n = k
                    break
            out.update(z=zc, escape=n)
            out["text"] = f"z = {zc.real:.4g}{zc.imag:+.4g}i   " + (
                f"escapes after {n} steps" if n is not None else "stays bounded"
            )
        return out

    def _to_output(self, u: float, v: float, size: tuple[int, int]) -> tuple[float, float]:
        meta = self.engine.document.meta
        return u * meta.width / size[0], v * meta.height / size[1]

    def select_at(self, u: float, v: float, size: tuple[int, int]) -> str | None:
        """Select the topmost object under a viewport point (pixels, origin top-left)."""
        x, y = self._to_output(u, v, size)
        with self.lock:
            ids = self.engine.hit_test(x, y, self.t)
        self.selected = ids[0] if ids else None
        return self.selected

    def hit_ids(self, u: float, v: float, size: tuple[int, int]) -> list[str]:
        """Objects under a viewport point, topmost first (the selection is left alone)."""
        x, y = self._to_output(u, v, size)
        with self.lock:
            return list(self.engine.hit_test(x, y, self.t))

    def drag_camera(self, dx: float, dy: float, size: tuple[int, int],
                    pan: bool = False) -> dict[str, Any]:  # fmt: skip
        """A drag of the empty viewport by ``dx``, ``dy`` pixels: turns the 3D camera (0.4
        degrees a pixel); with ``pan``, or in 2D, slides the camera so the picture follows
        the mouse."""
        spec, cur = self._camera()
        if spec.kind == "orbit" and not pan:
            return self.orbit_camera(dx * 0.4, dy * 0.4)
        import math

        if spec.kind == "orbit":
            per_px = 2 * float(cur["distance"]) * math.tan(math.radians(float(cur["fov"])) / 2)
        else:
            per_px = float(cur["height"])
        per_px /= size[1]
        return self.pan_camera(-dx * per_px, dy * per_px)

    def select(self, object_id: str | None) -> None:
        if object_id is not None and object_id not in self._ids():
            raise ToolError(f"no object {object_id!r}")
        self.selected = object_id

    def screen_to_world(self, u: float, v: float, size: tuple[int, int], z: float = 0.0):
        """The world point under a viewport pixel, on the plane at height ``z`` (2D: z = 0)."""
        cam = self.engine.state_at(self.t).camera
        p = cam.from_screen([(u, v)], size[0], size[1], z)[0]
        return float(p[0]), float(p[1]), float(z)

    def drag_to(self, object_id: str, u: float, v: float, size: tuple[int, int]):
        """Move an object with a ``pos`` to the point under the mouse (one undo step, sent
        when the drag ends)."""
        spec = self.engine.document.object(object_id)
        if spec is None or "pos" not in spec.params:
            raise ToolError(f"{object_id!r} has no position to drag")
        z = float(spec.params["pos"][2]) if len(spec.params["pos"]) > 2 else 0.0
        x, y, _ = self.screen_to_world(u, v, size, z)
        pos = [round(x, 6), round(y, 6), z]
        return self._edit({"op": "set_param", "path": f"{object_id}.pos", "value": pos},
                          f"{object_id}.pos = {pos}")  # fmt: skip

    # ------------------------------------------------------------ the camera

    def _camera(self) -> tuple[Any, dict[str, Any]]:
        from phasor.time.compile import CAMERA_PARAMS

        doc = self.engine.document
        spec = doc.camera(doc.timeline.camera)
        model = CAMERA_PARAMS[spec.kind]
        current = {
            k: f.get_default(call_default_factory=True) for k, f in model.model_fields.items()
        }
        return spec, {**current, **spec.params}

    def _set_camera(self, updates: dict[str, Any], what: str) -> dict[str, Any]:
        spec, _ = self._camera()
        tracks = self.engine.document.timeline.tracks
        busy = [k for k in updates if f"{spec.id}.{k}" in tracks]
        if busy:
            return self._refused(
                f"{spec.id}.{busy[0]} is animated by a track: scrub to a moment without the "
                "animation, or clear it, to move the camera by hand"
            )
        ops = [
            {"op": "set_param", "path": f"{spec.id}.{k}", "value": v} for k, v in updates.items()
        ]
        return self._edit(ops, f"camera {what}")

    def orbit_camera(self, d_azimuth: float, d_elevation: float) -> dict[str, Any]:
        """Turn the 3D camera around its target by degrees (dragging right turns the scene
        to the right). A 2D camera has nothing to turn."""
        spec, cur = self._camera()
        if spec.kind != "orbit":
            return self._refused("a 2D camera does not turn; use the orbit camera for 3D")
        az = float(cur["azimuth"]) - d_azimuth
        el = min(max(float(cur["elevation"]) - d_elevation, -89.0), 89.0)
        return self._set_camera(
            {"azimuth": az, "elevation": el}, f"orbit to {az:.1f}, {el:.1f} degrees"
        )

    def zoom_camera(self, factor: float) -> dict[str, Any]:
        """Move in (factor > 1) or out (< 1): the orbit camera's distance, the 2D camera's
        height."""
        spec, cur = self._camera()
        key = "distance" if spec.kind == "orbit" else "height"
        return self._set_camera({key: float(cur[key]) / max(factor, 1e-3)}, f"zoom x{factor:.2f}")

    def pan_camera(self, dx: float, dy: float) -> dict[str, Any]:
        """Slide the camera ``dx`` along the screen's right and ``dy`` along its up (world
        units)."""
        spec, cur = self._camera()
        cam = self.engine.state_at(self.t).camera
        key = "target" if spec.kind == "orbit" else "center"
        move = cam.right * dx + cam.up * dy
        new = [round(float(a + b), 9) for a, b in zip(cur[key], move, strict=True)]
        return self._set_camera({key: new}, f"pan to {new[0]:.2f}, {new[1]:.2f}")

    # ------------------------------------------------------------ the code view

    def code(self) -> dict[str, Any]:
        """The scene as Python: the script it was built from, or one generated from the
        document. ``path`` is where running edited code keeps it."""
        from phasor.core.ids import file_stem
        from phasor.core.pycode import document_to_python

        p = self.ws.path
        if p is not None and p.suffix == ".py":
            return {"text": p.read_text(encoding="utf-8"), "is_script": True, "path": str(p)}
        stem = (p.name.removesuffix(".phasor.json") if p is not None
                else file_stem(self.engine.document.meta.title))  # fmt: skip
        return {"text": document_to_python(self.engine.document), "is_script": False,
                "path": str(self.ws.root / f"{stem}.scene.py")}  # fmt: skip

    def run_code(self, text: str) -> dict[str, Any]:
        """Run edited code as the scene (the person at the keyboard may: it is their code).
        The code is kept at ``path``; a mistake reports where and changes nothing. Running
        replaces the open scene (its undo history starts again)."""
        import ast

        try:
            ast.parse(text)
        except SyntaxError as e:
            return {"ok": False, "error": f"line {e.lineno}: {e.msg}"}
        path = Path(self.code()["path"])
        old = path.read_text(encoding="utf-8") if path.exists() else None
        path.write_text(text, encoding="utf-8")
        allowed, self.ws.allow_scripts = self.ws.allow_scripts, True
        try:
            with self.lock:
                self.ws.scene_open(str(path))
        except Exception as e:
            tb = getattr(e, "__traceback__", None)
            line = None
            while tb is not None:
                if tb.tb_frame.f_code.co_filename == str(path):
                    line = tb.tb_lineno
                tb = tb.tb_next
            if old is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(old, encoding="utf-8")
            where = f"line {line}: " if line else ""
            return {"ok": False, "error": f"{where}{type(e).__name__}: {e}"}
        finally:
            self.ws.allow_scripts = allowed
        self._frame = None
        self.preview = {}
        self.selected = None
        self.t = 0.0
        return {"ok": True, "version": self.version}

    # ------------------------------------------------------------ the inspector

    def inspector(self, object_id: str | None = None) -> dict[str, Any] | None:
        oid = object_id or self.selected
        if oid is None:
            return None
        with self.lock:
            spec = self.engine.document.object(oid)
            if spec is None:
                return None
            st = self.engine.state_at(self.t)[oid]
            seen = self.engine.inspect(self.t, id=oid)["objects"][0]
        animated = [p for p in self.timeline()["lanes"] if p.startswith(oid + ".")]
        now = st.params.model_dump()
        params = {}
        for k, field in type(st.params).model_fields.items():
            if k in ("opacity", "dim", "z_index") and k not in spec.params:
                continue
            if f"{oid}.{k}" in animated:
                params[k] = now[k]  # a track: its value under the playhead
            elif k in spec.params:
                params[k] = spec.params[k]  # what the document says (an expression stays one)
            else:
                params[k] = field.get_default(call_default_factory=True)
        rules = [c.rule for c in self.engine.document.constraints if oid in c.rule]
        return {"id": oid, "type": spec.type, "params": params, "style": dict(st.roles),
                "semantic": seen.get("semantic", {}), "constraints": rules,
                "animated": animated}  # fmt: skip

    def set_param(self, object_id: str, name: str, text: str) -> dict[str, Any]:
        """Set a parameter from what was typed in its field."""
        value = parse_value(text)
        shown = text.strip()
        return self._edit({"op": "set_param", "path": f"{object_id}.{name}", "value": value},
                          f"{object_id}.{name} = {shown}")  # fmt: skip

    def set_style(self, object_id: str, key: str, role: str) -> dict[str, Any]:
        return self._edit({"op": "set_style", "id": object_id, "key": key, "role": role},
                          f"{object_id} {key}: {role}")  # fmt: skip

    # ------------------------------------------------------------ lint

    def lint_items(self, aesthetic: bool = False) -> list[dict[str, Any]]:
        """The lint panel: live issues (recomputed only after a change)."""
        v = self.version
        if self._lint is None or self._lint[0] != v or aesthetic:
            with self.lock:
                r = self.ws.lint(aesthetic=aesthetic)
            items = r["issues"] if isinstance(r, dict) else r
            if aesthetic:
                return items
            self._lint = (v, items)
        return self._lint[1]

    def jump(self, issue: dict[str, Any]) -> None:
        """Click on an issue: select its first object and move to its first time."""
        ids = [i for i in issue.get("ids", []) if i in self._ids()]
        self.selected = ids[0] if ids else self.selected
        times = issue.get("times") or []
        if times:
            self.set_time(float(times[0]))

    # ------------------------------------------------------------ looks and variants

    def looks(self) -> list[str]:
        from phasor.style.looks import available_looks

        return available_looks()

    def apply_look(self, name: str) -> dict[str, Any]:
        """Switch the scene to a look: its theme and life, and its defaults for the objects
        already there. One undo step."""
        from phasor.style.looks import load_look

        look = load_look(name)
        ops: list[dict[str, Any]] = [
            {"op": "set_meta", "field": "theme", "value": look.theme},
            {"op": "set_meta", "field": "life", "value": look.life},
            {"op": "set_meta", "field": "look", "value": name},
        ]
        for spec in self.engine.document.objects:
            d = look.defaults.get(spec.type, {})
            for k, v in d.get("params", {}).items():
                ops.append({"op": "set_param", "path": f"{spec.id}.{k}", "value": v})
            for k, role in d.get("style", {}).items():
                ops.append({"op": "set_style", "id": spec.id, "key": k, "role": role})
        return self._edit(ops, f"look: {name}")

    def palettes(self) -> list[str]:
        from phasor.style.theme import PALETTES

        return list(PALETTES)

    def set_palette(self, name: str | None) -> dict[str, Any]:
        """Recolor the strokes with a palette (``spectrum``, the prism rainbow, ...): one
        undo step, recorded as your edit."""
        with self.lock:
            r = self.ws.scene_palette(name)
        if r["ok"]:
            entry = {"what": f"palette: {name}", "version": self.version, "t": self.t}
            self.history.append(entry)
            self.ws.activity.append({"who": "you", **entry})
        return r

    def variants(self, beat: str | float, n: int = 3) -> dict[str, Any]:
        with self.lock:
            return self.ws.variants(beat, n)

    def choose(self, winner: str, note: str = "") -> dict[str, Any]:
        """Pick a variant here: recorded as the person's choice (the AI's ``choose`` records
        its own as the AI's) and applied as one undo step."""
        from phasor.style import choices

        offered = getattr(self.ws, "_variants", {})
        if winner not in offered:
            raise ToolError(f"no variant {winner!r}; offered {sorted(offered)}")
        entry = choices.record(
            self.ws.root / choices.DEFAULT_PATH,
            scene=self.engine.document.meta.title,
            beat=getattr(self.ws, "_variant_beat", ""),
            options=list(offered),
            winner=winner,
            by="person",
            note=note,
        )
        ops = offered[winner]
        result = self._edit(ops, f"variant: {winner}") if ops else None
        return {"recorded": entry, "applied": bool(result and result["ok"]), "edit": result}
