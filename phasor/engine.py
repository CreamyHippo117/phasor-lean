"""Engine: owns the document, the evaluated-state cache and the inspector (contract 1).

The Python API, CLI, MCP server and GUI are thin clients of this class. Every mutation goes
through ``Engine.apply(patch) -> PatchResult``, which validates the whole result, records an
undo step and returns the lint delta. There is no other write path.

Importing this module loads no GPU library; a renderer is created on first use, and when
none can be created the inspector runs in analytic mode (tier C).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import phasor.domains  # noqa: F401  (registers the built-in object types)
from phasor.core import io
from phasor.core.issues import DocumentError, Issue
from phasor.core.patch import TWEAK_OPS, apply_ops, apply_overrides, parse_patch, record_tweaks
from phasor.core.schema import Document, Meta, Overrides
from phasor.core.state import SceneState
from phasor.core.timing import stage
from phasor.inspect import aesthetic as aesthetic_rules
from phasor.inspect import critic, inspector
from phasor.inspect import events as event_log
from phasor.inspect import lint as lint_rules
from phasor.inspect import variants as variant_ops
from phasor.inspect.oracle import PassOracle
from phasor.inspect.passes import PassRunner
from phasor.layout.bake import Bake, bake, baked_state
from phasor.layout.plan import Plan, compile_plan
from phasor.time.compile import Compiled, compile_document
from phasor.time.expr import CompiledExpr
from phasor.time.tracks import ResolvedTrack

STATE_CACHE_SIZE = 256
UNDO_LIMIT = 200


class PatchResult:
    """What ``Engine.apply`` returns. The lint delta is worked out the first time it is read
    (it needs the layout bake and the inspector), so building a scene stays fast."""

    def __init__(
        self,
        ok: bool,
        version: int,
        errors: tuple[Issue, ...] = (),
        delta: Callable[[], tuple[tuple[Issue, ...], tuple[Issue, ...]]] | None = None,
    ) -> None:
        self.ok, self.version, self.errors = ok, version, errors
        self._delta = delta
        self._lint: tuple[tuple[Issue, ...], tuple[Issue, ...]] | None = None

    def _resolve(self) -> tuple[tuple[Issue, ...], tuple[Issue, ...]]:
        if self._lint is None:
            self._lint = self._delta() if self._delta else ((), ())
            self._delta = None
        return self._lint

    @property
    def lint_added(self) -> tuple[Issue, ...]:
        return self._resolve()[0]

    @property
    def lint_resolved(self) -> tuple[Issue, ...]:
        return self._resolve()[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "errors": [i.to_dict() for i in self.errors],
            "lint_delta": {
                "added": [i.to_dict() for i in self.lint_added],
                "resolved": [i.to_dict() for i in self.lint_resolved],
            },
        }

    def raise_for_errors(self) -> PatchResult:
        if not self.ok:
            raise DocumentError(list(self.errors))
        return self

    def __repr__(self) -> str:
        return f"PatchResult(ok={self.ok}, version={self.version}, errors={list(self.errors)})"


class _Snapshot:
    """One point in history: the document, plus what is derived from it. The layout bake
    and the lint result are filled in on first use and kept, so undo gets them back free."""

    def __init__(self, base: Document, overrides: Overrides, base_dir: str = "",
                 allow_scripts: bool = False) -> None:  # fmt: skip
        self.base, self.overrides = base, overrides
        effective, orphans = apply_overrides(base, overrides)
        need = needs_code(effective) or changed_modules(effective, base_dir)
        modules = load_modules(effective, base_dir, allow_scripts) if need else {}
        self.compiled: Compiled = compile_document(effective, base_dir, modules)
        self.plan: Plan = compile_plan(self.compiled)
        self.orphans: tuple[Issue, ...] = tuple(orphans)
        self.bake: Bake | None = None
        self.lint: dict[float, tuple[Issue, ...]] = {}
        self.aesthetic: tuple[Issue, ...] | None = None
        self.states: OrderedDict[float, SceneState] = OrderedDict()
        self.snapshots: OrderedDict[float, dict[str, Any]] = OrderedDict()


_build = _Snapshot


# ---------------------------------------------------------------- scene-local code (ADR 0005)

# resolved path -> (modification time, status, the loaded module or None)
_LOADED: dict[str, tuple[int, str, Any]] = {}
_LOADING: set[str] = set()  # modules being run now (a scene script that builds its own Scene)
REFUSED = ("refused: scripts may not run here (open the scene with allow_scripts=True, or "
           "start the MCP server with --allow-scripts)")  # fmt: skip


def needs_code(doc: Document) -> bool:
    """Whether the document names code that is not here yet: a type nobody registered or an
    updater nobody defined in this process."""
    from phasor.core.objects import get_type, get_updater

    if not doc.meta.modules:
        return False
    return any(get_type(o.type) is None or any(get_updater(r) is None for r in o.always)
               for o in doc.objects)  # fmt: skip


def changed_modules(doc: Document, base_dir: str | Path) -> bool:
    """Whether a module this process loaded for the document has changed on disk since
    (contract 3: files the document names are read again when they change). Modules that
    registered their types some other way (a scene script run by ``load``) are not watched."""
    from phasor.core import files

    for rel in doc.meta.modules:
        try:
            path = files.resolve(base_dir, rel)
            hit = _LOADED.get(str(path))
            if hit is not None and path.is_file() and path.stat().st_mtime_ns != hit[0]:
                return True
        except (ValueError, OSError):
            continue
    return False


def load_modules(doc: Document, base_dir: str | Path, allow: bool) -> dict[str, str]:
    """Load the modules the document lists (``meta.modules``), each once per change of the
    file, and report what became of each: ``"loaded"``, ``"missing"``, a refusal (scripts
    may run only on the person's request, the importers' rule) or ``"failed: ..."``. A
    module is read only from inside the scene's folder (``core/files.py``). One that fails
    leaves nothing behind: not its types, not its updaters, not itself in ``sys.modules``."""
    from phasor.core import files

    out: dict[str, str] = {}
    for rel in doc.meta.modules:
        try:
            path = files.resolve(base_dir, rel)
        except ValueError as e:
            out[rel] = f"refused: {e}"
            continue
        if not path.is_file():
            out[rel] = "missing"
            continue
        if not allow:
            out[rel] = REFUSED
            continue
        key, mtime = str(path), path.stat().st_mtime_ns
        hit = _LOADED.get(key)
        if key in _LOADING or (hit is not None and hit[0] == mtime):
            out[rel] = hit[1] if hit is not None else "loaded"
        else:
            out[rel] = _run_module(path, Path(base_dir or ".").resolve(), mtime)
        if out[rel] == "loaded":
            _adopt_updaters(doc, rel, _LOADED.get(key, (0, "", None))[2])
    return out


def _run_module(path: Path, folder: Path, mtime: int) -> str:
    import hashlib
    import importlib.util
    import sys

    from phasor.core.objects import registry_snapshot, restore_registry

    key = str(path)
    name = "phasor_scene_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _LOADED[key] = (mtime, "failed: not a Python module", None)
        return _LOADED[key][1]
    module = importlib.util.module_from_spec(spec)
    before, snapshot = set(sys.modules), registry_snapshot()
    sys.modules[name] = module  # pydantic resolves a model's annotations through it
    _LOADING.add(key)
    try:
        spec.loader.exec_module(module)
    except BaseException as e:  # a scene's code may raise anything, exit too
        if isinstance(e, KeyboardInterrupt):
            raise
        restore_registry(snapshot)
        for k in set(sys.modules) - before:
            f = getattr(sys.modules[k], "__file__", None) or ""
            if k == name or (f and folder in Path(f).resolve().parents):
                del sys.modules[k]
        _LOADED[key] = (mtime, f"failed: {type(e).__name__}: {e}", None)
        return _LOADED[key][1]
    finally:
        _LOADING.discard(key)
    _LOADED[key] = (mtime, "loaded", module)
    return "loaded"


def _adopt_updaters(doc: Document, rel: str, module: Any) -> None:
    """Updaters the document names in a loaded module that nothing registered yet (a plain
    function in a ``*.types.py``): found by name and registered under their reference."""
    from phasor.core.objects import _UPDATERS, get_updater

    if module is None:
        return
    for o in doc.objects:
        for ref in o.always:
            mod, _, name = ref.rpartition(":")
            if mod != rel or get_updater(ref) is not None or "<" in name:
                continue
            target: Any = module
            for part in name.split("."):
                target = getattr(target, part, None)
            if callable(target):
                _UPDATERS[ref] = target


class Engine:
    def __init__(
        self,
        document: Document | None = None,
        *,
        overrides: Overrides | None = None,
        title: str = "untitled",
        passes: PassRunner | str | None = "auto",
        base_dir: str | Path | None = None,
        allow_scripts: bool = False,
    ) -> None:
        """``allow_scripts``: the modules the document names (``meta.modules``, types and
        updaters defined in the scene's folder) may run. Off by default: like the importers,
        code runs only on the person's request (``phasor.load("scene.py")`` runs the script
        anyway, so it turns this on)."""
        base = document or Document(meta=Meta(title=title))
        # The folder that files the document refers to (narration audio, meshes) are relative to.
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.allow_scripts = allow_scripts
        self._snap = _build(base, overrides or Overrides(), str(self.base_dir), allow_scripts)
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._checkpoints: dict[str, _Snapshot] = {}  # named points to go back to
        self._version = 0
        self._passes = passes

    def fingerprint(self) -> str:
        """A hash of the current document and overrides (canonical JSON): what a bake was
        made for."""
        import hashlib

        text = io.dumps(self._snap.base) + io.dumps(self._snap.overrides)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def worker_payload(self) -> dict[str, Any]:
        """What a render worker needs to draw this scene without baking it again (ADR 0003):
        the document and overrides as canonical JSON, this engine's bake (baked here first
        if need be) and the fingerprint it was made for. Plain data: it pickles."""
        return {
            "document": io.dumps(self._snap.base),
            "overrides": io.dumps(self._snap.overrides),
            "bake": self._bake(self._snap),
            "fingerprint": self.fingerprint(),
            "base_dir": str(self.base_dir),
            # The scene's own code (ADR 0005) runs in a worker when it could run here: the
            # parent was allowed to, or already had every type (the person ran the script).
            "allow_scripts": self.allow_scripts or not self._snap.compiled.missing,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], **kwargs: Any) -> Engine:
        """The engine of a render worker: ``worker_payload``'s document with its parent's
        bake applied as is, never baked here (ADR 0003). A bake made for another document is
        refused."""
        import json

        doc = io.parse_document(json.loads(payload["document"]))
        overrides = Overrides.model_validate_json(payload["overrides"])
        kwargs.setdefault("base_dir", payload["base_dir"])
        kwargs.setdefault("allow_scripts", payload.get("allow_scripts", False))
        engine = cls(doc, overrides=overrides, **kwargs)
        if engine.fingerprint() != payload["fingerprint"]:
            raise ValueError("this bake was made for another document; bake it here instead")
        engine._snap.bake = payload["bake"]
        return engine

    def fork(self) -> Engine:
        """A private copy of this engine's current scene (document and overrides) with its own
        history: edit it freely (a player moving sliders) and this one never changes."""
        return Engine(self._snap.base, overrides=self._snap.overrides, passes=self._passes,
                      base_dir=self.base_dir, allow_scripts=self.allow_scripts)  # fmt: skip

    # ------------------------------------------------------------ files

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> Engine:
        """Load ``name.phasor.json`` and, if present, ``name.overrides.json`` beside it."""
        kwargs.setdefault("base_dir", Path(path).parent)
        return cls(
            io.load_document(path), overrides=io.load_overrides(io.overrides_path(path)), **kwargs
        )

    def save(self, path: str | Path) -> None:
        """Write the base document, and the overrides file when there are any tweaks.
        An emptied overrides file is removed so stale tweaks cannot come back."""
        io.save(path, self._snap.base)
        self._bring_files(Path(path).parent)
        opath = io.overrides_path(path)
        if not self._snap.overrides.is_empty():
            io.save(opath, self._snap.overrides)
        elif opath.exists():
            opath.unlink()

    def _bring_files(self, folder: Path) -> None:
        """Files the scene names by a relative path (pictures) go with it when it is saved
        into another folder, so they are found next to it there."""
        import shutil

        from phasor.core import files
        from phasor.core.objects import get_type

        for spec in self.document.objects:
            for name in getattr(get_type(spec.type), "Files", ()):
                raw = spec.params.get(name)
                if not isinstance(raw, str) or not raw:
                    continue
                try:
                    src, dst = files.resolve(self.base_dir, raw), folder / files.relative_path(raw)
                except ValueError:
                    continue
                if src.is_file() and src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    # ------------------------------------------------------------ read

    @property
    def document(self) -> Document:
        """The effective document: base with overrides applied."""
        return self._snap.compiled.doc

    @property
    def base(self) -> Document:
        return self._snap.base

    @property
    def overrides(self) -> Overrides:
        return self._snap.overrides

    @property
    def version(self) -> int:
        return self._version

    @property
    def scripted(self) -> bool:
        return self._snap.base.meta.source == "script"

    @property
    def duration(self) -> float:
        return self._snap.compiled.duration

    def lint(
        self,
        step: float = lint_rules.DEFAULT_STEP,
        *,
        strict: bool = False,
        aesthetic: bool = False,
    ) -> list[Issue]:
        """Every lint issue: dangling overrides, unsolved constraints, and the visual rules
        checked at each marker and every ``step`` seconds. Cached per document version.
        ``strict`` adds rules that only matter for finished scenes (``raw_motion``);
        ``aesthetic`` adds the taste rules (focal point, color restraint, glow clip, density,
        motion rhythm, margin, stroke mix)."""
        issues = list(self._lint(self._snap, step))
        if strict:
            issues += lint_rules.strict_rules(self._snap.compiled)
        if aesthetic:
            issues += self._aesthetic(self._snap)
        return issues

    def critique(self) -> dict[str, Any]:
        """The art-director's report: the taste score (0 = nothing to criticise) and every
        aesthetic issue with the markers it happens at, certain ``fix`` ops and judgement
        ``suggested`` ops. See ``inspect/critic.py``."""
        issues = [i for i in self._aesthetic(self._snap)]
        return critic.critique(self._snap.compiled, issues, lambda t: self._snapshot(t))

    def polish(self, rounds: int = 3) -> dict[str, Any]:
        """Critique, apply the fixes (certain ones, then judgement suggestions), and repeat,
        at most ``rounds`` times (three by default). Every round is undoable. Returns the score
        before and after and what each round did; stops early when nothing is left to do."""
        history = []
        before = self.critique()
        report = before
        for k in range(1, rounds + 1):
            ops: list[dict[str, Any]] = []
            for i in report["issues"]:
                for op in i["fix"] or i["suggested"]:
                    if op not in ops:
                        ops.append(op)
            if not ops or not self.apply(ops).ok:
                break
            report = self.critique()
            history.append({"round": k, "applied": ops, "score": report["score"]})
        return {"before": before["score"], "after": report["score"], "rounds": history,
                "remaining": report["issues"]}  # fmt: skip

    def _aesthetic(self, snap: _Snapshot) -> tuple[Issue, ...]:
        if snap.aesthetic is None:
            snap.aesthetic = tuple(aesthetic_rules.rules(
                snap.compiled,
                snap.plan,
                lambda t: self._state(snap, t),
                lambda t: self._snapshot_of(snap, t),
                self.pass_runner(),
            ))  # fmt: skip
        return snap.aesthetic

    def audio_track(self):
        """The narration as one mono int16 track covering the scene, and its sample rate;
        None when there is no recorded or synthesized audio."""
        from phasor.time.narration import mix_track

        n = self.document.narration
        return None if n is None else mix_track(n, self.base_dir, self.duration)

    def _lint(self, snap: _Snapshot, step: float) -> tuple[Issue, ...]:
        if step not in snap.lint:
            visual = lint_rules.run(
                snap.compiled,
                snap.plan,
                lambda t: self._state(snap, t),
                lambda t: self._snapshot_of(snap, t),
                self.pass_runner(),
                step=step,
            )
            snap.lint[step] = (*snap.orphans, *self._bake(snap).issues,
                               *lint_rules.control_rules(snap.compiled),
                               *lint_rules.file_rules(snap.compiled),
                               *lint_rules.type_rules(snap.compiled), *visual)  # fmt: skip
        return snap.lint[step]

    def _oracle(self) -> PassOracle | None:
        runner = self.pass_runner()
        return PassOracle(runner) if runner is not None else None

    def _bake(self, snap: _Snapshot) -> Bake:
        if snap.bake is None:
            snap.bake = bake(snap.compiled, snap.plan, self._oracle())
        return snap.bake

    @property
    def layout(self) -> Bake:
        """The layout bake for the current document: key times, tracks, chosen label slots."""
        return self._bake(self._snap)

    def state_at(self, t: float) -> SceneState:
        """Pure function of (document, t), layout included. Cached per document version."""
        with stage("state"):
            return self._state(self._snap, t)

    def _state(self, snap: _Snapshot, t: float) -> SceneState:
        t = float(t)
        if t in snap.states:
            snap.states.move_to_end(t)
            return snap.states[t]
        state = baked_state(snap.compiled, self._bake(snap), t)
        snap.states[t] = state
        if len(snap.states) > STATE_CACHE_SIZE:
            snap.states.popitem(last=False)
        return state

    # ------------------------------------------------------------ write

    def apply(self, patch: Any, *, origin: str = "tweak") -> PatchResult:
        """The only write path. Atomic: on any error nothing changes.

        ``origin="script"`` is used by the Python builder and edits the base document.
        ``origin="tweak"`` (GUI, AI, CLI) edits the base of a plain document, but on a scene
        built by a script it records into the overrides file instead, so re-running the
        script never destroys the tweak.
        """
        if origin not in ("tweak", "script"):
            raise ValueError(f"origin must be 'tweak' or 'script', got {origin!r}")
        old = self._snap
        try:
            parsed = parse_patch(patch)
            if origin == "tweak" and self.scripted:
                overrides = record_tweaks(old.overrides, parsed)
                # Validate the ops against the effective document for precise errors.
                apply_ops(old.compiled.doc, parsed)
                new = _build(old.base, overrides, str(self.base_dir), self.allow_scripts)
            else:
                base = apply_ops(old.base, parsed)
                overrides = old.overrides
                if origin == "tweak":
                    # A direct edit supersedes an older override on the same target.
                    overrides = _drop_superseded(overrides, parsed)
                new = _build(base, overrides, str(self.base_dir), self.allow_scripts)
        except DocumentError as e:
            return PatchResult(False, self._version, errors=tuple(e.issues))
        if self._undo:  # keep evaluated caches only for the current and previous step
            self._undo[-1].states.clear()
            self._undo[-1].snapshots.clear()
        self._undo.append(old)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        return self._switch(new, old)

    def undo(self) -> PatchResult | None:
        if not self._undo:
            return None
        old = self._snap
        self._redo.append(old)
        return self._switch(self._undo.pop(), old)

    def redo(self) -> PatchResult | None:
        if not self._redo:
            return None
        old = self._snap
        self._undo.append(old)
        return self._switch(self._redo.pop(), old)

    # Named checkpoints: like undo, they switch to a document this
    # engine already validated; going back is one step that undo can reverse.

    def checkpoint(self, name: str) -> None:
        """Remember the current document under ``name``."""
        self._checkpoints[name] = self._snap

    def checkpoints(self) -> list[str]:
        return sorted(self._checkpoints)

    def checkpoint_document(self, name: str) -> Document:
        return self._checkpoints[name].compiled.doc

    def restore(self, name: str) -> PatchResult:
        """Go back to checkpoint ``name``: one undoable step; the version keeps counting up."""
        if name not in self._checkpoints:
            raise KeyError(f"no checkpoint {name!r}; have {self.checkpoints()}")
        old = self._snap
        self._undo.append(old)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        return self._switch(self._checkpoints[name], old)

    def _switch(self, new: _Snapshot, old: _Snapshot) -> PatchResult:
        self._snap = new
        self._version += 1

        def delta() -> tuple[tuple[Issue, ...], tuple[Issue, ...]]:
            step = lint_rules.DEFAULT_STEP
            before, after = self._lint(old, step), self._lint(new, step)
            return (
                tuple(i for i in after if i not in before),
                tuple(i for i in before if i not in after),
            )

        return PatchResult(True, self._version, delta=delta)

    def autofix(self, max_rounds: int = 3, *, aesthetic: bool = False) -> dict[str, Any]:
        """Apply the fixes the linter suggests (for example: let the label placer move an
        overlapping label), re-lint, and repeat. Everything goes through ``apply``, so each
        round is one undo step. ``aesthetic`` includes the taste rules' fixes. Returns what
        was applied and what is still wrong."""
        applied: list[dict[str, Any]] = []
        for _ in range(max_rounds):
            ops: list[dict[str, Any]] = []
            for issue in self.lint(aesthetic=aesthetic):
                ops += [op for op in issue.fix if op not in ops and op not in applied]
            if not ops or not self.apply(ops).ok:
                break
            applied += ops
        return {"applied": applied, "remaining": self.lint(aesthetic=aesthetic)}

    # ------------------------------------------------------------ inspect

    def pass_runner(self) -> PassRunner | None:
        """The renderer behind the inspector, or None on a machine with no usable OpenGL."""
        if self._passes == "auto":
            try:
                from phasor.render.passes import default_renderer

                self._passes = default_renderer()
            except Exception:
                self._passes = None
        return self._passes  # type: ignore[return-value]

    def _snapshot(self, t: float) -> dict[str, Any]:
        return self._snapshot_of(self._snap, t)

    def _snapshot_of(self, snap: _Snapshot, t: float) -> dict[str, Any]:
        t = float(t)
        if t not in snap.snapshots:
            snap.snapshots[t] = inspector.snapshot(self._state(snap, t), self.pass_runner())
            if len(snap.snapshots) > STATE_CACHE_SIZE:
                snap.snapshots.popitem(last=False)
        return snap.snapshots[t]

    def inspect(self, t: float, id: str | None = None, detail: str = "standard") -> Any:
        """Everything on screen at ``t`` (or one object), at the requested detail level."""
        self._require(id)
        return inspector.render_detail(self._snapshot(t), detail, only=id)

    def _entry(self, id: str, t: float) -> dict[str, Any]:
        self._require(id)
        return next(e for e in self._snapshot(t)["objects"] if e["id"] == id)

    def where(self, id: str, t: float) -> dict[str, Any]:
        """Region, bbox, depth and relations to objects tagged ``key``."""
        e = self._entry(id, t)
        keys = ("id", "region", "bbox", "center", "depth", "relations")
        return {"t": float(t), **{k: e[k] for k in keys}}

    def visible(self, id: str, t: float) -> dict[str, Any]:
        """Visible and on-screen fractions."""
        e = self._entry(id, t)
        return {
            "t": float(t),
            "id": id,
            "visible_fraction": e["visible_fraction"],
            "on_screen_fraction": e["on_screen_fraction"],
            "occlusion": self._snapshot(t)["occlusion"],
        }

    def occluders(self, id: str, t: float) -> dict[str, Any]:
        """Who hides it and by how much."""
        e = self._entry(id, t)
        return {"t": float(t), "id": id, "hidden_by": e["hidden_by"]}

    # ------------------------------------------------------------ events

    def _samples(self, t0: float, t1: float | None, hz: float):
        ts = event_log.sample_times(self.duration, self.document.timeline.markers, t0, t1, hz)
        return [(t, self._snapshot(t)) for t in ts]

    def events(
        self, t0: float = 0.0, t1: float | None = None, *, hz: float = event_log.HZ
    ) -> list[dict[str, Any]]:
        """What changes on screen between ``t0`` and ``t1``: visibility crossing 0.9 / 0.5 /
        0.1, entering or leaving the frame, region changes, new occluders. Problems a viewer
        would notice have ``"severity": "warning"``; ``event_log.problems`` picks them out."""
        return event_log.log(self._samples(t0, t1, hz), self.document.timeline.markers)

    def when_hidden(self, id: str, *, below: float = 0.5, hz: float = event_log.HZ):
        """Time spans where ``id`` is shown but less than ``below`` of it is visible."""
        self._require(id)
        return [list(s) for s in event_log.hidden_spans(self._samples(0.0, None, hz), id, below)]

    def free_space_near(self, id: str, t: float) -> list[dict[str, Any]]:
        """Empty places on screen around ``id`` at ``t`` (the label placer's slots that
        nothing covers), nearest first: where a label or note could go."""
        self._require(id)
        from phasor.layout.bake import free_slots

        return free_slots(self.state_at(t), id, self._oracle())

    def variants(
        self,
        beat: str | float,
        n: int = 3,
        *,
        size: tuple[int, int] = (480, 270),
        score: bool = True,
    ) -> dict[str, Any]:
        """Alternative looks of one beat (a marker name or a time), side by side: the scene
        as it is, calmer (less glow, softer edges) and bolder (more glow, heavier strokes).
        They are tried on a throwaway copy; this scene does not change. Returns
        ``{"image", "beat", "t", "variants": [{"name", "ops", "score"}]}``; ``score`` is the
        critic's taste score of each (lower is better). Apply the winner's ``ops`` with
        ``apply``. Needs OpenGL."""
        from phasor.render.storyboard import contact_sheet

        t = self.document.timeline.markers[beat] if isinstance(beat, str) else float(beat)
        cells, out = [], []
        for name, ops in variant_ops.make(self.document, n):
            twin = Engine(self.base, overrides=self.overrides, base_dir=self.base_dir,
                          passes=self.pass_runner(), allow_scripts=self.allow_scripts)  # fmt: skip
            if ops and not twin.apply(ops).ok:
                continue
            s = twin.critique()["score"] if score else None
            cells.append((f"{chr(65 + len(out))}: {name}" + ("" if s is None else f"  (taste {s})"),
                          twin.still(t, size)))  # fmt: skip
            out.append({"name": name, "ops": ops, "score": s})
        return {"image": contact_sheet(cells, len(cells)), "beat": beat, "t": t, "variants": out}

    def hit_test(self, x: float, y: float, t: float) -> list[str]:
        """Object ids under an output-pixel point, topmost first."""
        return inspector.hit_test(self.state_at(t), self.pass_runner(), x, y)

    def still(self, t: float, size: tuple[int, int] | None = None):
        """Preview still: (h, w, 3) uint8 sRGB, the same look as the final render.
        Needs OpenGL (tier A or B)."""
        from phasor.render.output import encode_srgb8

        return encode_srgb8(self.frame(t, size))

    def frame(
        self,
        t: float,
        size: tuple[int, int] | None = None,
        *,
        output: str = "linear",
        field_scale: float = 1.0,
    ):
        """The finished picture at ``t`` in linear light: color pass, bloom, tone map.
        (h, w, 4) float32, row 0 = top. ``still`` and ``phasor render`` both use this.
        ``output="srgb8"``: the (h, w, 3) sRGB bytes instead, encoded on the card.
        ``field_scale`` < 1 draws kernel-made fields (domain coloring, fractals) at that
        share of their resolution, stretched by the GPU: for the ``draft`` preset only;
        1.0 (the default) is the unchanged picture."""
        return self._finished([t], size, output, field_scale)

    def _finished(self, times: list[float], size, output: str, field_scale: float = 1.0):
        """The average light of the color pass at ``times`` (one time: the frame itself),
        bloomed, tone-mapped and encoded once. Each sample goes to the card once as float16
        and is widened and added there in float32, in a fixed order;
        with one sample the sum is the widened sample exactly. Without a card the same sum
        runs in NumPy."""
        from phasor.render.gpu import Tile, run
        from phasor.render.passes import GLRenderer
        from phasor.render.post import card_sync, finish

        runner = self.pass_runner()
        if not isinstance(runner, GLRenderer):
            raise RuntimeError("no OpenGL renderer on this machine (run `phasor doctor`)")
        weight = 1.0 / len(times)
        pic = glow = None
        # Start every sample before collecting any: the card copies one sample off while
        # the next is drawn (asynchronous readback). Same samples, same order of the sum.
        pending = [
            runner.start_color_raw(self.state_at(ts), size, field_scale=field_scale) for ts in times
        ]
        try:
            for job in pending:
                p, g = job.collect()
                with stage("blur sum" if len(times) > 1 else "widen", sync=card_sync):
                    tile = Tile.full(p.shape[1], p.shape[0])
                    pic = run("accumulate", tile, image=p, weight=weight, acc=pic)
                    glow = run("accumulate", tile, image=g, weight=weight, acc=glow)
        finally:
            for job in pending:
                job.release()
        # rows bottom-up as GL stores them -> top-down (a view, wherever the sum is)
        return finish(pic[::-1], self._snap.compiled.theme, glow[::-1], output=output)

    def timeline(self) -> dict[str, Any]:
        """What a timeline panel shows: the duration, the markers, for every animated
        parameter its spans in resolved seconds (``{"d.pos": [[1.0, 2.0], ...]}``) and the
        spoken narration blocks (text, start, duration, audio file)."""
        c = self._snap.compiled
        lanes = {
            path: [[seg.t0, seg.t1] for seg in node.segments]
            for path, node in sorted(c.nodes.items())
            if isinstance(node, ResolvedTrack) and node.segments
        }
        markers = dict(sorted(c.time_map.markers.items(), key=lambda kv: kv[1]))
        nar = self._snap.compiled.doc.narration
        blocks = [{"text": b.text, "start": b.start, "duration": b.duration, "audio": b.audio}
                  for b in (nar.blocks if nar else ()) if b.text]  # fmt: skip
        return {"duration": c.duration, "markers": markers, "lanes": lanes, "narration": blocks}

    def is_moving(self, t0: float, t1: float) -> bool:
        """Whether anything animated by the document runs between ``t0`` and ``t1``: a
        transition, or an expression of ``t`` (``expr("t / 10")``; counted as moving whenever
        it exists, so a finished one costs extra samples that come out identical). Life (slow
        wiggle) is not counted: it needs no blur."""
        for node in self._snap.compiled.nodes.values():
            if isinstance(node, CompiledExpr) and node.uses_t:
                return True
            if isinstance(node, ResolvedTrack):
                for seg in node.segments:
                    if seg.t0 < t1 and seg.t1 > t0 and seg.t1 > seg.t0:
                        return True
        return False

    def frame_blurred(
        self,
        t: float,
        fps: float,
        *,
        shutter: float = 0.5,
        samples: int = 4,
        size: tuple[int, int] | None = None,
        output: str = "linear",
        field_scale: float = 1.0,
    ):
        """Like ``frame`` but with motion blur: the average of ``samples`` frames spread over
        a shutter open ``shutter`` of a frame interval (0.5 is the film standard "180
        degrees"), centered on ``t``. Where nothing is moving it is exactly ``frame``, so
        still moments stay sharp and cost nothing extra. Life (slow wiggle) does not count
        as moving (``is_moving``). Linear light, deterministic."""
        half = shutter / fps / 2
        if samples <= 1 or shutter <= 0 or not self.is_moving(t - half, t + half):
            return self.frame(t, size, output=output, field_scale=field_scale)
        # Average the light before the tone map (which then runs once, contract 4): a bright
        # moving glow smears at its real brightness instead of an already squashed one.
        times = [min(max(ts, 0.0), self.duration) for ts in blur_times(t, fps, shutter, samples)]
        return self._finished(times, size, output, field_scale)

    def storyboard_times(self, count: int = 6) -> list[float]:
        """Start, every marker and the end; padded with evenly spaced times up to ``count``."""
        end = self.duration
        ts = {0.0, end, *self.document.timeline.markers.values()}
        if len(ts) < count and end > 0:
            ts.update(round(end * k / (count - 1), 6) for k in range(count))
        return sorted(t for t in ts if 0.0 <= t <= end)

    def storyboard(
        self,
        times: list[float] | None = None,
        *,
        marks: bool = False,
        size: tuple[int, int] = (480, 270),
        columns: int = 3,
        compare: str | Path | None = None,
    ) -> dict[str, Any]:
        """Keyframe stills on one contact sheet, plus what the inspector sees in each.

        Returns ``{"image": (h, w, 3) uint8, "frames": [{"t", "marker", "objects"}]}``.
        ``marks=True`` draws each object's box and id on the stills (set-of-mark), so a
        vision model can tie what it sees to object ids. ``compare`` is a gallery folder:
        the result then also has ``"compare_image"``, each frame beside its three nearest
        gallery frames, and ``"compare"``, who they were and why you like them. Needs OpenGL
        (tier A or B).
        """
        from phasor.render.storyboard import contact_sheet, draw_marks

        named = sorted(self.document.timeline.markers.items(), reverse=True)
        markers = {t: n for n, t in named}
        frames, cells = [], []
        for t in self.storyboard_times() if times is None else times:
            still = self.still(t, size)
            snap = self._snapshot(t)
            if marks:
                still = draw_marks(still, snap)
            label = f"t = {t:g} s" + (f"  [{markers[t]}]" if t in markers else "")
            cells.append((label, still))
            lines = inspector.render_detail(snap, "brief")["objects"]
            frames.append({"t": float(t), "marker": markers.get(t), "objects": lines})
        out = {"image": contact_sheet(cells, columns), "frames": frames}
        if compare is not None:
            from phasor.render import gallery

            plain = [(c, self.still(f["t"], size)) for (c, _), f in zip(cells, frames, strict=True)]
            found = gallery.entries(compare)
            out["compare_image"], out["compare"] = gallery.compare_sheet(plain, found)
            out["gallery_size"] = len(found)
        return out

    def _require(self, id: str | None) -> None:
        if id is not None and self.document.object(id) is None:
            known = sorted(o.id for o in self.document.objects)
            raise KeyError(f"no object {id!r}; objects are {known}")


def blur_times(t: float, fps: float, shutter: float = 0.5, samples: int = 4) -> list[float]:
    """The ``samples`` moments ``Engine.frame_blurred`` averages: evenly spread over a
    shutter open ``shutter`` of a frame interval, centered on ``t``."""
    half = shutter / fps / 2
    return [t + (k + 0.5) / samples * (2 * half) - half for k in range(samples)]


def _drop_superseded(overrides: Overrides, patch: Any) -> Overrides:
    params, styles = dict(overrides.params), dict(overrides.styles)
    for op in patch.ops:
        if isinstance(op, TWEAK_OPS[0]):
            params.pop(op.path, None)
        elif isinstance(op, TWEAK_OPS[1]):
            styles.pop(f"{op.id}.{op.key}", None)
    return overrides.model_copy(update={"params": params, "styles": styles})
