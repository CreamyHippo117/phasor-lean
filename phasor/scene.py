"""The Python authoring API: a thin builder over ``Engine`` (contract 1).

A script builds the base document. Every call here is one ``Engine.apply`` with
``origin="script"``; an invalid call raises at the line that made it.

    s = Scene("residues")
    s.narrate(SCRIPT)                                  # markers come from the voice
    s.at("poles_appear").reveal("pole_a", "pole_b")    # animate by verb, at a word
    s.at("loop_draw").trace("loop").then().reveal("loop.label")
    s.pace()                                           # breathing room after busy sentences
"""

from __future__ import annotations

import contextlib
import inspect
import os
import runpy
import sys
from pathlib import Path
from typing import Any

from phasor.core import io
from phasor.core.ids import file_stem
from phasor.core.objects import ObjectBase, scene_module
from phasor.core.patch import (
    AddConstraint,
    AddControl,
    AddObject,
    AddSegment,
    SetCamera,
    SetMarker,
    SetMeta,
    SetNarration,
    SetParam,
)
from phasor.core.schema import (
    Control,
    Document,
    ExprSpec,
    MarkerRef,
    Meta,
    ObjectSpec,
    Segment,
)
from phasor.engine import Engine
from phasor.style.looks import DEFAULT_THEME, default_look, load_look
from phasor.time import verbs as _verbs
from phasor.time.narration import block_of, repace

TimeArg = float | tuple[str, float] | str  # seconds, ("marker", offset) or "marker"


def expr(source: str) -> ExprSpec:
    """A parameter computed from other parameters and ``t``, e.g. ``expr("2 * dot.r + sin(t)")``."""
    return ExprSpec(expr=source)


class At:
    """A point on the timeline, returned by ``Scene.at``. Verbs and ``to`` start here."""

    def __init__(self, scene: Scene, when: str | float, offset: float) -> None:
        self._scene, self._when, self._offset = scene, when, offset
        self._end = 0.0  # where the last verb started from here ended, for ``then``

    def _ref(self, extra: float) -> float | MarkerRef:
        if isinstance(self._when, str):
            return MarkerRef(marker=self._when, offset=self._offset + extra)
        return float(self._when) + self._offset + extra

    def to(self, path: str, value: Any, *, duration: float = 1.0, ease: str = "smooth") -> At:
        """Move parameter ``path`` to ``value`` over ``duration`` seconds, starting here.
        A raw transition: prefer a verb, whose timing comes from the theme."""
        seg = Segment(t0=self._ref(0.0), t1=self._ref(duration), to=value, ease=ease)
        self._scene._apply(AddSegment(path=path, segment=seg))
        self._end = max(self._end, duration)
        return self

    def set(self, path: str, value: Any) -> At:
        """Jump parameter ``path`` to ``value`` at this instant."""
        return self.to(path, value, duration=0.0, ease="linear")

    def then(self, gap: float = 0.0) -> At:
        """The moment everything started here has finished (plus ``gap`` seconds)."""
        return At(self._scene, self._when, self._offset + self._end + gap)

    def _verb(self, name: str, *args: Any, **kwargs: Any) -> At:
        engine = self._scene.engine
        at = self._ref(0.0)
        compiled = engine._snap.compiled
        try:
            t = compiled.time_map.resolve(at)
        except KeyError as e:
            raise _verbs.VerbError(f"no marker {e.args[0]!r}; markers are "
                                   f"{sorted(compiled.time_map.markers)}") from None  # fmt: skip
        motion = getattr(_verbs.Verbs(compiled, engine.state_at(t), at), name)(*args, **kwargs)
        if motion.ops:
            self._scene._apply(list(motion.ops))
        self._end = max(self._end, motion.end)
        return self

    def reveal(self, *ids: str) -> At:
        """Fade objects in (hidden until now), staggered."""
        return self._verb("reveal", *ids)

    def exit(self, *ids: str) -> At:
        """Fade objects out."""
        return self._verb("exit", *ids)

    def trace(self, *ids: str) -> At:
        """Draw curves from start to end (nothing drawn until now)."""
        return self._verb("trace", *ids)

    def bloom_in(self, *ids: str) -> At:
        """Fade in while growing from a point, with a small overshoot."""
        return self._verb("bloom_in", *ids)

    def dim_others(self, *keep: str) -> At:
        """Dim everything but ``keep`` (backdrops stay)."""
        return self._verb("dim_others", *keep)

    def undim(self) -> At:
        """Bring dimmed objects back."""
        return self._verb("undim")

    def focus_pull(self, object_id: str, margin: float = _verbs.FOCUS_MARGIN) -> At:
        """Move the camera onto ``object_id``."""
        return self._verb("focus_pull", object_id, margin)

    def morph(self, path: str, value: Any, until: str | None = None) -> At:
        """Change any parameter smoothly, with the theme's morph timing, or until the
        marker ``until``."""
        return self._verb("morph", path, value, until)

    def settle(self, path: str, value: Any) -> At:
        """Move a parameter like a released spring."""
        return self._verb("settle", path, value)

    def orbit(self, degrees: float = 360.0, until: str | None = None) -> At:
        """Swing the 3D camera ``degrees`` around its target, over the theme's time per turn
        or until the marker ``until``."""
        return self._verb("orbit", degrees, until)

    def orbit_reveal(self, *ids: str) -> At:
        """Reveal objects while the 3D camera swings a little around them."""
        return self._verb("orbit_reveal", *ids)

    def transform(self, source: str, target: str) -> At:
        """Equation ``source`` becomes ``target``: matching parts glide over, others fade."""
        return self._verb("transform", source, target)

    def become(self, source: str, target: str) -> At:
        """Curve ``source`` turns into the shape of ``target``, which then takes its place."""
        return self._verb("become", source, target)

    def write(self, *ids: str) -> At:
        """Write text or equations in, left to right."""
        return self._verb("write", *ids)

    def travel(self, *ids: str) -> At:
        """Send tracers along their curves, trailing light."""
        return self._verb("travel", *ids)

    def flash(self, *ids: str) -> At:
        """A quick brightening of each object."""
        return self._verb("flash", *ids)

    def circumscribe(self, object_id: str, shape: str = "circle") -> At:
        """Draw a ring (or ``shape="box"``) around an object, then fade it."""
        return self._verb("circumscribe", object_id, shape)

    def wiggle(self, *ids: str) -> At:
        """A small side-to-side shake."""
        return self._verb("wiggle", *ids)

    def sweep(self, path: str, value: float, show: bool = True) -> At:
        """Run a number evenly to ``value``, with a live readout beside its object."""
        return self._verb("sweep", path, value, show)


def _caller_dir() -> Path:
    """The folder of the script that called ``Scene(...)``, or the working folder."""
    for frame in inspect.stack()[2:]:
        path = Path(frame.filename)
        if path.suffix == ".py" and path.exists() and "phasor" not in path.parts[-2:-1]:
            return path.parent
    return Path.cwd()


class Scene:
    def __init__(
        self,
        title: str,
        *,
        theme: str | None = None,
        look: str | None = None,
        palette: str | None = None,
        life: str | None = None,
        seed: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        **engine_kwargs: Any,
    ) -> None:
        """``look`` names a preset (``prism``, ``prism_everywhere``, ``calm``, ``glow``,
        ``chalk``): it picks the theme and ``life`` and supplies defaults for objects.
        ``theme`` and ``life`` you give win over the look's. With neither a look nor a theme
        the scene gets the default look, ``prism``; ``look="none"`` gives none (the dark
        ``midnight_glow`` theme, plain objects).
        ``palette`` picks the colors run along strokes (``spectrum``, the prism rainbow,
        or ``moonlight``, ``gold``, ``glacier``, ``borealis``, ``rose``): in the prism looks
        it recolors every glowing curve and line and keeps the rest of the look."""
        look = default_look(look, theme)
        self.look = load_look(look) if look else None
        theme = theme or (self.look.theme if self.look else DEFAULT_THEME)
        life = life or (self.look.life if self.look else "off")
        meta = Meta(title=title, theme=theme, width=width, height=height, fps=fps, life=life,
                    seed=seed, look=look, palette=palette, source="script")  # fmt: skip
        engine_kwargs.setdefault("base_dir", _caller_dir())
        self.engine = Engine(Document(meta=meta), **engine_kwargs)

    @classmethod
    def of(cls, engine: Engine) -> Scene:
        """A builder around an engine that already exists (the MCP server uses this, so a
        client gets the same verbs as a script)."""
        scene = cls.__new__(cls)
        scene.engine = engine
        look = engine.document.meta.look
        scene.look = load_look(look) if look else None
        return scene

    def _apply(self, op: Any) -> None:
        self.engine.apply(op, origin="script").raise_for_errors()

    def add(self, obj: ObjectBase) -> ObjectBase:
        """Add an object: ``s.add(Dot(id="dot", pos=(1, 2)))``. Returns it for chaining.
        The scene's look fills in what the object leaves unset (the values are stored)."""
        base_dir = self.engine.base_dir
        spec = obj.to_spec(base_dir)
        if self.look is not None:
            params, style = self.look.fill(spec.type, dict(spec.params), dict(spec.style),
                                           self.engine.document.meta.palette)  # fmt: skip
            spec = ObjectSpec(id=spec.id, type=spec.type, params=params, style=style,
                              tags=spec.tags, layer=spec.layer, always=spec.always)  # fmt: skip
        # A type or updater defined in the scene's folder (ADR 0005): the document names its
        # module, so the scene opens with it later (where scripts may run).
        have = self.engine.base.meta.modules
        need = [m for m in (scene_module(type(obj), base_dir),
                            *(r.rpartition(":")[0] for r in spec.always)) if m]  # fmt: skip
        new = tuple(dict.fromkeys([*have, *need]))
        ops: list[Any] = [SetMeta(field="modules", value=new)] if new != have else []
        self._apply([*ops, AddObject(object=spec)])
        return obj

    def use_camera(self, kind: str = "orbit", **params: Any) -> None:
        """Choose the camera: ``"orbit"`` (3D perspective around ``target``; ``azimuth``,
        ``elevation``, ``distance``, ``fov``, ``drift``) or ``"ortho"`` (2D, the default).
        Parameters you leave out are free for the camera solver when you give it goals
        (``s.constrain("camera keep_visible(surface)")``)."""
        self._apply(SetCamera(kind=kind, params=params))

    def set(self, path: str, value: Any) -> None:
        """Set a parameter (objects and cameras): ``s.set("camera.height", 6)``."""
        self._apply(SetParam(path=path, value=value))

    def control(self, path: str, *, min: float, max: float, step: float = 0.01, label: str = "",
                component: int | None = None) -> None:  # fmt: skip
        """Make a parameter a slider (``interactive``): the player, the editor, a notebook and
        the web export let a viewer move it between ``min`` and ``max``. For a vector
        parameter (a complex ``c``), ``component`` picks the entry: 0 is Re, 1 is Im."""
        self._apply(AddControl(control=Control(path=path, component=component, label=label,
                                               min=min, max=max, step=step)))  # fmt: skip

    def constrain(self, rule: str, *, priority: str = "strong") -> None:
        """Add a layout rule: ``s.constrain("label(pole) near pole")``. The engine solves it
        at every key time; a rule it cannot satisfy shows up in ``engine.lint()``."""
        self._apply(AddConstraint(rule=rule, priority=priority))

    def narrate(self, script: str, *, voice: str = "auto") -> None:
        """Speak ``script`` and take its ``[[markers]]`` as the timeline.

        ``voice``: ``auto`` (espeak-ng when installed, else estimated timing), ``estimate``,
        ``espeak[:lang[:wpm]]`` or ``piper:<model.onnx>``. Audio clips are cached in
        ``<title>.narration/`` beside the script. Calling it again (another voice, an edited
        script) moves every marker, and everything anchored to them follows.
        """
        from phasor.voice import narrate

        folder = self.engine.base_dir / f"{file_stem(self.engine.document.meta.title)}.narration"
        theme = self.engine._snap.compiled.theme
        self._apply(SetNarration(narration=narrate(script, voice, folder, theme)))

    def pace(self) -> None:
        """Give each sentence breathing room in proportion to how much it changes on screen:
        after a sentence whose transitions touch N objects, wait N x ``pause_per_change``
        seconds more (at most ``pause_max``; theme ``[narration]``). Run it last; running it
        again changes nothing."""
        doc = self.engine.document
        if doc.narration is None:
            raise ValueError("pace() needs narration: call narrate() first")
        compiled = self.engine._snap.compiled
        busy: dict[int, set[str]] = {}
        for path, segs in doc.timeline.tracks.items():
            for seg in segs:
                k = block_of(doc.narration, compiled.time_map.resolve(seg.t0))
                if k is not None:
                    busy.setdefault(k, set()).add(path.rpartition(".")[0])
        n = compiled.theme.narration
        extra = [min(len(busy.get(k, ())) * n["pause_per_change"], n["pause_max"])
                 for k in range(len(doc.narration.blocks))]  # fmt: skip
        self._apply(SetNarration(narration=repace(doc.narration, extra, n["gap"], n["lead"])))

    def segment(self, path: str, t0: TimeArg, t1: TimeArg, to: Any, *,
                ease: str = "smooth") -> None:  # fmt: skip
        """A raw transition between two times, each seconds or a marker (``"go"`` or
        ``("go", 0.5)``: the marker plus an offset): for spans that start and end at
        different markers. Prefer a verb or ``s.at(...).to(...)`` when one marker is enough."""

        def ref(t):
            if isinstance(t, str):
                return MarkerRef(marker=t)
            if isinstance(t, tuple):
                return MarkerRef(marker=t[0], offset=float(t[1]))
            return float(t)

        self._apply(
            AddSegment(path=path, segment=Segment(t0=ref(t0), t1=ref(t1), to=to, ease=ease))
        )

    def marker(self, name: str, t: float) -> None:
        """Place a narration marker at ``t`` seconds."""
        self._apply(SetMarker(name=name, t=t))

    def at(self, when: str | float, offset: float = 0.0) -> At:
        """A time to animate from: a marker name or seconds, plus an optional offset."""
        return At(self, when, offset)

    def save(self, path: str | Path) -> None:
        self.engine.save(path)


def load(path: str | Path, **engine_kwargs: Any) -> Engine:
    """Open a scene for any client: a ``.phasor.json`` document, or a ``.py`` script that
    builds a ``Scene``. Tweaks in ``name.overrides.json`` beside it are applied on top."""
    path = Path(path)
    if path.suffix != ".py":
        return Engine.open(path, **engine_kwargs)
    scenes = [v for v in runpy.run_path(str(path), run_name="__phasor__").values()
              if isinstance(v, Scene)]  # fmt: skip
    if len(scenes) != 1:
        raise ValueError(
            f"{path} must define exactly one Scene at module level, found {len(scenes)}"
        )
    overrides = io.load_overrides(path.with_name(path.stem + io.OVERRIDES_SUFFIX))
    engine_kwargs.setdefault("base_dir", path.parent)
    engine_kwargs.setdefault("allow_scripts", True)  # the person ran this script: its code runs
    return Engine(scenes[0].engine.base, overrides=overrides, **engine_kwargs)


# ---------------------------------------------------------------- render workers (ADR 0003)

# Frames a worker renders in a row. Frames are taken in order, stretch by stretch, so a worker
# must be able to finish its next stretch while the others' are being taken: its queue holds
# about two stretches. Short stretches keep that buffer small (30 frames each stalled every
# worker but one: measured 2026-11-22). Blur samples never leave their frame's worker, and
# simulations keep their checkpoints on disk, so short stretches cost nothing.
WORKER_CHUNK = 4
WORKER_BUFFER_BYTES = 256 * 1024**2  # at most this much finished video waiting per worker


def default_workers() -> int:
    """How many render workers to use by default: a quarter of the CPU threads, 1 to 4 (each
    is a process with its own GL and CUDA context on the same card). Without CUDA the kernels
    use the cores too, and ``default_threads`` gives each worker its share of them."""
    return max(1, min(4, (os.cpu_count() or 1) // 4))


def default_threads(workers: int) -> int | None:
    """Kernel threads per render process without CUDA: the cores shared out, so that
    workers x threads <= cores. None with CUDA (kernels run on the card) or
    when the machine has no C++ compiler (kernels run on NumPy, one core); None too when
    ``PHASOR_THREADS`` is set, which every process then reads for itself."""
    from phasor.render.gpu.registry import active_backend

    if active_backend() != "cpu" or os.environ.get("PHASOR_THREADS", "").strip():
        return None
    from phasor.render.cpu import backend as cpu

    return max(1, cpu.cores() // max(1, workers))


def parallel_frames(engine: Any, workers: int, *, chunk: int = WORKER_CHUNK,
                    threads: int | None = None, _fail_at: int | None = None,
                    _on_start: Any = None, _slow: float = 0.0) -> Any:  # fmt: skip
    """A frame source for ``render.job.render_file`` that renders in ``workers`` processes.

    The engine bakes here, once; each worker gets its document and that bake
    (``Engine.worker_payload``) and never bakes (ADR 0003). Worker w renders stretches w,
    w + N, ... of ``chunk`` frames; frames come back in order, so the video is the one a
    single process makes. A failed worker stops the render with its error; workers stop by
    themselves if this process dies (the desktop app's Cancel kills it). ``threads``: CPU
    kernel threads in each worker (default ``default_threads(workers)``). ``_fail_at``,
    ``_on_start`` and ``_slow`` are for tests."""
    import multiprocessing

    def source(times, size, fps, shutter, samples, field_scale=1.0):
        n = len(times)
        stretches = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
        count = max(1, min(workers, len(stretches)))
        each = default_threads(count) if threads is None else threads
        payload = engine.worker_payload()
        ctx = multiprocessing.get_context("spawn")  # Windows has no fork; the same everywhere
        frame_bytes = size[0] * size[1] * 3
        depth = max(chunk + 1, min(2 * chunk, WORKER_BUFFER_BYTES // frame_bytes))
        queues = [ctx.Queue(maxsize=depth) for _ in range(count)]
        procs = [
            ctx.Process(target=_render_worker, name=f"phasor-render-{w}", daemon=True,
                        args=(payload, stretches[w::count], times, size, fps, shutter, samples,
                              queues[w], 1.0 / count, each, _fail_at, _slow, field_scale))
            for w in range(count)
        ]  # fmt: skip
        with _workers_skip_main():
            for p in procs:
                p.start()
        if _on_start is not None:
            _on_start([p.pid for p in procs])
        try:
            for i, (a, b) in enumerate(stretches):
                w = i % count
                for k in range(a, b):
                    msg = _receive(queues[w], procs[w])
                    if msg[0] == "error":
                        raise RuntimeError(f"render worker {w} failed: {msg[1]}")
                    if msg[1] != k:
                        raise RuntimeError(f"render worker {w} sent frame {msg[1]}, not {k}")
                    yield msg[2]
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=10)
            for q in queues:
                q.cancel_join_thread()
                q.close()

    return source


@contextlib.contextmanager
def _workers_skip_main() -> Any:
    """Start spawned workers without re-running the parent's main script. Spawn re-imports
    a script run as ``python scene.py`` in each child (to recreate its module); a scene script
    without ``if __name__ == "__main__":`` would then run again, renders and all, in every
    worker. Workers need nothing from it (their entry point is in this module), so the
    script's path is hidden while they start. ``python -m ...`` and the packaged app are
    already safe and left alone."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if main is None or getattr(main, "__spec__", None) is not None or path is None:
        yield
        return
    del main.__file__
    try:
        yield
    finally:
        main.__file__ = path


def _receive(q: Any, proc: Any) -> tuple:
    import queue

    while True:
        try:
            return q.get(timeout=1.0)
        except queue.Empty:
            if not proc.is_alive():
                try:
                    return q.get(timeout=1.0)
                except queue.Empty:
                    raise RuntimeError(
                        f"{proc.name} stopped without a word (exit code {proc.exitcode})"
                    ) from None


def _render_worker(payload, stretches, times, size, fps, shutter, samples, q, share, threads,
                   fail_at, slow, field_scale=1.0) -> None:  # fmt: skip
    """A render worker's whole life: build the engine from the payload, render its
    stretches, send each frame. Stops when the parent is gone."""
    import multiprocessing
    import queue
    import time
    import traceback

    parent = multiprocessing.parent_process()

    def orphaned() -> bool:
        if parent is not None and not parent.is_alive():
            q.cancel_join_thread()  # nobody will read what is still buffered
            return True
        return False

    def send(msg) -> bool:
        while True:
            try:
                q.put(msg, timeout=1.0)
                return True
            except queue.Full:
                if orphaned():
                    return False

    try:
        from phasor.engine import Engine
        from phasor.render.cpu import set_threads
        from phasor.render.gpu.budget import set_share
        from phasor.render.job import render_frame

        set_share(share)
        set_threads(threads)  # None: every core (one worker, or kernels on the card)
        engine = Engine.from_payload(payload)
        for a, b in stretches:
            for k in range(a, b):
                if orphaned():
                    return
                if k == fail_at:
                    raise RuntimeError(f"failed on purpose at frame {k}")
                frame = render_frame(engine, times[k], fps, size, shutter, samples, field_scale)
                if slow:
                    time.sleep(slow)
                if not send(("frame", k, frame)):
                    return
    except BaseException:
        send(("error", traceback.format_exc(limit=12)))
