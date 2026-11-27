"""What an AI client can do with phasor, as plain functions (no MCP library in here).

``Workspace`` holds the one open scene. Every method takes and returns JSON-friendly data,
so the same behavior is tested directly and served over MCP by ``server.py``. Nothing here
writes a document except through ``Engine.apply`` (contract 1).

Safety: paths must stay inside ``root``; ``.py`` scenes are executable code and are
refused unless ``allow_scripts`` is on; patches are validated before they apply and a
failed patch changes nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from phasor.core import io
from phasor.core.diff import diff_documents
from phasor.core.ids import file_stem
from phasor.core.issues import DocumentError, Issue
from phasor.core.objects import get_type, is_builtin, registered_types
from phasor.core.schema import Document, Meta
from phasor.engine import Engine
from phasor.scene import Scene, load
from phasor.style.looks import DEFAULT_THEME, available_looks, default_look, load_look
from phasor.style.theme import ROLE_GROUPS, available_themes
from phasor.time import verbs as _verbs
from phasor.workspace import docs

VERBS = (
    "reveal",
    "exit",
    "trace",
    "bloom_in",
    "dim_others",
    "undim",
    "focus_pull",
    "morph",
    "settle",
    "orbit",
    "orbit_reveal",
    "transform",
    "become",
    "write",
    "travel",
    "flash",
    "circumscribe",
    "wiggle",
    "sweep",
    "unfold",
)
MAX_CHECKPOINTS = 32


SHOWN_TIMES = 6


def _slim(issue: dict[str, Any]) -> dict[str, Any]:
    """An issue for a client: a rule that fires at 35 sample times lists the first few and
    the total (the full list is noise to read)."""
    times = issue.get("times", [])
    if len(times) > SHOWN_TIMES:
        issue = {**issue, "times": times[:SHOWN_TIMES], "times_total": len(times)}
    return issue


def _slim_result(r: dict[str, Any]) -> dict[str, Any]:
    d = r.get("lint_delta")
    if d:
        r = {**r, "lint_delta": {k: [_slim(i) for i in v] for k, v in d.items()}}
    return r


class ToolError(Exception):
    """A client mistake with a message the client can act on (not a crash)."""

    def __init__(self, message: str, issues: list[Issue] | None = None) -> None:
        super().__init__(message)
        self.issues = [i.to_dict() for i in issues or []]


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        out_dir: str | Path | None = None,
        *,
        allow_scripts: bool = False,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.out_dir = Path(out_dir).resolve() if out_dir else self.root / "phasor_out"
        self.allow_scripts = allow_scripts
        self._engine: Engine | None = None
        self._variants: dict[str, list[dict[str, Any]]] = {}
        self._variant_beat = ""
        self.path: Path | None = None  # where the open scene was loaded from / last saved
        # Who changed the scene, in order: the editor's person ("you") and an attached AI ("ai").
        self.activity: list[dict[str, Any]] = []
        self.import_notes: list[str] = []  # what the last Manim import could not carry over

    # ------------------------------------------------------------ plumbing

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise ToolError("no scene is open: call scene_new or scene_open first")
        return self._engine

    def _inside(self, path: str | Path, *, must_exist: bool = False) -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{path} is outside the workspace folder {self.root}")
        if must_exist and not p.exists():
            raise ToolError(f"{path} does not exist (workspace folder: {self.root})")
        return p

    def _guard(self, fn):
        try:
            return fn()
        except DocumentError as e:
            raise ToolError(str(e), e.issues) from e
        except (KeyError, ValueError, OSError) as e:
            raise ToolError(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)) from e

    def summary(self) -> dict[str, Any]:
        e = self.engine
        d = e.document
        return {
            "title": d.meta.title,
            "theme": d.meta.theme,
            "size": [d.meta.width, d.meta.height],
            "fps": d.meta.fps,
            "duration": e.duration,
            "version": e.version,
            "path": str(self.path) if self.path else None,
            "source": "script" if e.scripted else "document",
            "objects": {o.id: o.type for o in d.objects},
            "markers": dict(sorted(d.timeline.markers.items(), key=lambda kv: kv[1])),
            "constraints": [f"{c.rule} ({c.priority})" for c in d.constraints],
            "narration": d.narration.script if d.narration else None,
            "beds": [x.id for x in d.sound.beds],
            "cues": [x.id for x in d.sound.cues],
            "checkpoints": e.checkpoints(),
        }

    # ------------------------------------------------------------ scenes

    def scene_new(
        self, title: str, theme: str | None = None, look: str | None = None,
        life: str | None = None, width: int = 1920, height: int = 1080, fps: float = 60.0,
        palette: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Start an empty scene (replaces the open one; save first if it matters). It gets
        the default look, ``prism``, unless a ``look`` ("prism_everywhere", "calm", "glow",
        "chalk", or "none" for plain) or a ``theme`` is given; ``life`` ("off", "subtle",
        "lively") is the living motion; ``palette`` the colors run along strokes
        (``scene_palette``)."""

        def go():
            name = default_look(look, theme)
            chosen = load_look(name) if name else None
            meta = Meta(
                title=title, theme=theme or (chosen.theme if chosen else DEFAULT_THEME),
                life=life or (chosen.life if chosen else "off"), look=name, width=width,
                height=height, fps=fps, palette=palette,
            )  # fmt: skip
            self._engine = Engine(Document(meta=meta), base_dir=self.root)
            self.path = None

        self._guard(go)
        return self.summary()

    def scene_palette(self, palette: str | None) -> dict[str, Any]:
        """Pick the colors run along strokes: ``spectrum`` (the prism rainbow), ``moonlight``
        (white with a cool and warm shimmer), ``gold``, ``glacier``, ``borealis``, ``rose``,
        or None for the theme's own. In the prism looks it recolors every glowing curve and
        the lines the look runs a hue along; fringes, vignette and glow stay. One undo step."""
        from phasor.style.theme import PALETTES

        e = self.engine
        if palette is not None and palette not in PALETTES:
            raise ToolError(f"palette must be one of {PALETTES} (or none), got {palette!r}")
        ops: list[dict[str, Any]] = [{"op": "set_meta", "field": "palette", "value": palette}]
        look = load_look(e.document.meta.look) if e.document.meta.look else None
        targets = look.palette_targets() if look else {"line"}
        role = palette or (look.defaults.get("line", {}).get("style", {}).get("colormap")
                           if look else None) or "spectrum"  # fmt: skip
        for spec in e.document.objects:
            # Without a look, only lines that run a hue along themselves change.
            runs_hue = spec.type != "line" or look is not None or spec.params.get("hue_along", 0)
            if spec.type in targets and "colormap" in _style_keys(spec.type) and runs_hue:
                ops.append({"op": "set_style", "id": spec.id, "key": "colormap", "role": role})
        r = e.apply(ops, origin="tweak")
        return {**_slim_result(r.to_dict()), "palette": palette}

    def scene_open(self, path: str) -> dict[str, Any]:
        """Open ``name.phasor.json`` (or a ``.py`` script when the server allows scripts; a
        script that imports ``manim`` is run through phasor's Manim stand-in and opens as a
        plain scene, with ``notes`` on what could not be carried over)."""
        p = self._inside(path, must_exist=True)
        if p.suffix == ".py" and not self.allow_scripts:
            raise ToolError(
                "this server does not run scripts (a .py scene is code). Open the saved "
                ".phasor.json instead, or start the server with --allow-scripts"
            )
        from phasor.importers import import_manim, looks_like_manim

        if p.suffix == ".py" and looks_like_manim(p):
            # A Manim script: run through the stand-in module; the result is a plain scene
            # (saved beside the script as name.phasor.json), free to edit.
            result = self._guard(lambda: import_manim(p))
            self._engine = Engine(result.document(), base_dir=p.parent)
            self.path = p.with_suffix(io.DOC_SUFFIX)
            self.import_notes = result.warnings
            return {**self.summary(), "imported_from": "manim", "notes": result.warnings}
        self._engine = self._guard(lambda: load(p, allow_scripts=self.allow_scripts))
        self.path = p
        return self.summary()

    def scene_save(self, path: str | None = None) -> dict[str, Any]:
        """Save the document (``name.phasor.json``); tweaks to a script scene go to its
        overrides file beside it."""
        e = self.engine
        if path is None and self.path is None:
            raise ToolError("this scene has no path yet: pass path, e.g. 'residues.phasor.json'")
        target = self._inside(path) if path else self.path
        assert target is not None
        if target.suffix == ".py":
            target = target.with_name(target.stem + io.DOC_SUFFIX)
        if not target.name.endswith(io.DOC_SUFFIX):
            raise ToolError(f"scene files end in {io.DOC_SUFFIX}, got {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._guard(lambda: e.save(target))
        self.path = target
        return {"saved": str(target), "version": e.version}

    def scene_copy(self, path: str | None = None) -> dict[str, Any]:
        """Make an editable copy of a scene built by a script: the scene as it stands
        (script and tweaks) saved as a plain document, ``<name>_edit.phasor.json`` beside the
        script unless ``path`` says otherwise, and opened in its place. Every op works on
        the copy (sound, narration, new objects); the script is never changed. A plain
        document is copied the same way (a new file to experiment on)."""
        e = self.engine
        base = Path(e.base_dir or self.root)
        if path is not None:
            target = self._inside(path)
        else:
            stem = (self.path.name.removesuffix(io.DOC_SUFFIX).removesuffix(".py")
                    if self.path else file_stem(e.document.meta.title))  # fmt: skip
            target, k = base / f"{stem}_edit{io.DOC_SUFFIX}", 2
            while target.exists():
                target, k = base / f"{stem}_edit_{k}{io.DOC_SUFFIX}", k + 1
        source = self.path
        self._engine = self._guard(e.as_document)
        self.path = None
        saved = self.scene_save(str(target))
        return {**self.summary(), "copied_from": str(source) if source else None,
                "saved": saved["saved"]}  # fmt: skip

    # ------------------------------------------------------------ editing

    def scene_edit(self, patch: Any) -> dict[str, Any]:
        """Apply a patch (one op, a list of ops, or ``{"ops": [...]}``). Atomic: if any op
        is invalid nothing changes. Returns the result and the lint issues it added or fixed.
        An ``add_object`` may name a ``preset`` (see ``object_types``); the scene's look fills
        in what is left unset, and explicit values always win."""
        r = self.engine.apply(self._with_defaults(patch), origin="tweak")
        if r.ok:
            try:
                self._files_inside()
            except ToolError:
                self.engine.undo()  # the edit named a file outside the folder: take it back
                raise
        return {**_slim_result(r.to_dict()), "objects": len(self.engine.document.objects)}

    def _files_inside(self) -> None:
        """Every file the scene names (a picture's ``path``) is inside the workspace folder:
        the AI may show (or play) only what is in its folder."""
        from phasor.core import files
        from phasor.core.objects import get_type

        e = self.engine
        for spec in e.document.objects:
            cls = get_type(spec.type)
            for name in getattr(cls, "Files", ()):
                raw = spec.params.get(name)
                if isinstance(raw, str) and raw:
                    try:
                        self._inside(files.resolve(e.base_dir, raw))
                    except ValueError as err:  # outside the scene's folder
                        raise ToolError(str(err)) from None
        named = [b.path for b in e.document.sound.beds]
        named += [c.sound for c in e.document.sound.cues if not c.builtin]
        for path in named:
            try:
                self._inside(files.resolve(e.base_dir, path))
            except ValueError as err:
                raise ToolError(str(err)) from None

    def _with_defaults(self, patch: Any) -> Any:
        """The patch as a list of ops, with ``preset`` and the scene's look resolved in every
        ``add_object`` (the document stores the resolved values). Other ops pass through."""
        ops = patch.get("ops") if isinstance(patch, dict) and "ops" in patch else patch
        listed = ops if isinstance(ops, list | tuple) else [ops]
        look_name = self.engine.document.meta.look
        look = load_look(look_name) if look_name else None
        out = []
        for op in listed:
            if not (isinstance(op, dict) and op.get("op") == "add_object"
                    and isinstance(op.get("object"), dict)):  # fmt: skip
                out.append(op)
                continue
            spec = dict(op["object"])
            cls = get_type(str(spec.get("type")))
            preset = spec.pop("preset", None)
            params, style = dict(spec.get("params", {})), dict(spec.get("style", {}))
            if preset is not None:
                if cls is None or preset not in cls.Presets:
                    known = sorted(cls.Presets) if cls else []
                    raise ToolError(
                        f"no preset {preset!r} for {spec.get('type')!r}; presets: {known}"
                    )
                chosen = cls.Presets[preset]
                params = {**chosen.get("params", {}), **params}
                style = {**chosen.get("style", {}), **style}
            if look is not None:
                params, style = look.fill(str(spec.get("type")), params, style,
                                          self.engine.document.meta.palette)  # fmt: skip
            out.append({**op, "object": {**spec, "params": params, "style": style}})
        return out

    def undo(self) -> dict[str, Any]:
        """Revert the last edit (each patch, verb or narration is one step)."""
        r = self.engine.undo()
        return {"undone": r is not None, **(_slim_result(r.to_dict()) if r else {})}

    def redo(self) -> dict[str, Any]:
        """Re-apply the edit that undo reverted."""
        r = self.engine.redo()
        return {"redone": r is not None, **(_slim_result(r.to_dict()) if r else {})}

    def animate(
        self,
        verb: str,
        at: str | float,
        targets: list[str] | None = None,
        offset: float = 0.0,
        path: str | None = None,
        value: Any = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Animate with a choreography verb starting at a marker (or seconds). Timing comes
        from the theme. Returns when it starts and ends so the next step can follow it.
        ``orbit`` swings the 3D camera ``value`` degrees (default 360), until marker ``until``
        if given; ``morph`` too runs until ``until`` if given. ``transform`` and
        ``become`` take ``targets=[from, to]``; ``sweep`` takes
        ``path`` and ``value``; ``circumscribe`` takes one target and ``value`` "circle" or
        "box"; ``unfold`` takes ``targets=[room, ray]`` and ``value`` flips (default 5)."""
        if verb not in VERBS:
            raise ToolError(f"unknown verb {verb!r}; verbs are {list(VERBS)}")
        e = self.engine
        if e.scripted:
            raise ToolError(
                "this scene is built by a script, so structure and timing belong in the "
                "script; open its .phasor.json to animate here"
            )
        targets = targets or []
        if verb in ("morph", "settle"):
            if path is None or value is None:
                raise ToolError(f"{verb} needs path and value, e.g. path='dot.pos', value=[1, 2]")
            args: tuple = (path, value, until) if verb == "morph" else (path, value)
        elif verb == "orbit":
            args = (360.0 if value is None else float(value), until)
        elif verb == "focus_pull":
            if len(targets) != 1:
                raise ToolError("focus_pull needs exactly one target id")
            args = (targets[0],)
        elif verb in ("transform", "become"):
            if len(targets) != 2:
                raise ToolError(f"{verb} needs two target ids: [from, to]")
            args = (targets[0], targets[1])
        elif verb == "circumscribe":
            if len(targets) != 1:
                raise ToolError("circumscribe needs exactly one target id")
            args = (targets[0], "circle" if value is None else str(value))
        elif verb == "sweep":
            if path is None or value is None:
                raise ToolError("sweep needs path and value, e.g. path='wave.terms', value=40")
            args = (path, float(value))
        elif verb == "unfold":
            if len(targets) != 2:
                raise ToolError("unfold needs two target ids: [room, ray]")
            args = (targets[0], targets[1], 5 if value is None else int(value))
        else:
            args = tuple(targets)
        step = Scene.of(e).at(at, offset)

        def go():
            try:
                getattr(step, verb)(*args)
            except _verbs.VerbError as err:
                raise ToolError(str(err)) from err

        self._guard(go)
        base = e.document.timeline.markers[at] if isinstance(at, str) else float(at)
        start = base + offset
        return {"verb": verb, "start": start, "end": start + step._end, "version": e.version}

    def narrate(self, script: str, voice: str = "auto", pace: bool = True) -> dict[str, Any]:
        """Set the narration. ``[[name]]`` puts a marker at the next word. Every animation
        anchored to a marker follows the voice. ``pace`` adds breathing room after busy
        sentences."""
        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: change the narration there")
        scene = Scene.of(e)

        def go():
            scene.narrate(script, voice=voice)
            if pace:
                scene.pace()

        self._guard(go)
        return {
            "markers": dict(sorted(e.document.timeline.markers.items(), key=lambda kv: kv[1])),
            "duration": e.duration,
            "voice": e.document.narration.voice if e.document.narration else None,
        }

    def cue(
        self, at: str | float, sound: str, offset: float = 0.0, gain: float | None = None,
        pan: float = 0.0, midi: float | None = None, dur: float | None = None,
        voice: str | None = None, id: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Put a one-shot sound on a marker (``at``) plus ``offset`` seconds: a built-in
        (``tick``, ``chime``, ``whoosh``, ``pop``, ``swell``, ``tone``) or an audio file in
        the scene's folder. ``gain``: loudness in LU relative to the voice (default the
        theme's, 10 under); ``pan`` -1..1; ``midi`` tunes chime/pop/swell/tone, ``dur`` sets
        tone/swell/whoosh's length, ``voice`` tone's timbre (sine, soft, pluck, bell).
        Returns its id and time; retime it with scene_edit ``set_cue``, take it out with
        ``remove_sound``."""
        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: add cues there (s.cue), or call "
                            "scene_copy to edit a copy")  # fmt: skip
        scene = Scene.of(e)

        def go():
            cid = scene.cue(at, sound, offset=offset, gain=gain, pan=pan, midi=midi, dur=dur,
                            voice=voice, id=id)  # fmt: skip
            try:
                self._files_inside()
            except ToolError:
                e.undo()  # the cue named a file outside the folder: take it back
                raise
            return cid

        cid = self._guard(go)
        t = next(c["t"] for c in e.timeline()["cues"] if c["id"] == cid)
        return {"id": cid, "t": t, "cues": [c.id for c in e.document.sound.cues],
                "duration": e.duration, "version": e.version}  # fmt: skip

    def bed(
        self, path: str, start: str | float | None = None, end: str | float | None = None,
        gain: float | None = None, fade: float = 2.0, loop: bool = True, id: str | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Put music or ambience (an audio file in the scene's folder: WAV, FLAC, OGG, MP3)
        under the scene from marker ``start`` to marker ``end`` (omit either for the scene's
        start or end), looped, with equal-power fades of ``fade`` seconds. ``gain``: its
        loudness in LU relative to the voice (default the theme's, 18 under); it dips while
        the narration speaks. Returns its id; change it with scene_edit ``set_bed``."""
        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: add beds there (s.bed), or call "
                            "scene_copy to edit a copy")  # fmt: skip
        scene = Scene.of(e)

        def go():
            bid = scene.bed(path, start=start, end=end, gain=gain, fade=fade, loop=loop, id=id)
            try:
                self._files_inside()
            except ToolError:
                e.undo()
                raise
            return bid

        bid = self._guard(go)
        placed = next(b for b in e.timeline()["beds"] if b["id"] == bid)
        return {**placed, "beds": [b.id for b in e.document.sound.beds], "version": e.version}

    def sound_master(self, rounds: int = 3) -> dict[str, Any]:
        """Master the sound: fix what the sound check finds (a bed covering the voice is
        turned down, a mix that would clip is turned down evenly, a cue peaking on a stressed
        word moves just before it), check again, at most ``rounds`` times. Each round is one
        undo step. Returns before and after, what changed, and what is left."""
        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: call scene_copy first to edit "
                            "a copy (the script stays as it is)")  # fmt: skip
        return self._guard(lambda: e.master_sound(rounds))

    def voice_add(self, file: str, at: float | str = 0.0, text: str = "") -> dict[str, Any]:
        """Put a recording of the person's voice (an audio file in the scene's folder) into
        the narration, starting at ``at`` (seconds, or a marker's name). It plays as the
        voice: music ducks under it. ``text`` is what it says (for the script and the
        timeline; the file's name when empty). It may not overlap another sentence; move it
        later with ``set_narration``."""
        from phasor.core import files
        from phasor.core.schema import NarrationBlock
        from phasor.time.narration import add_block
        from phasor.time.sound import file_seconds

        e = self.engine
        if e.scripted:
            raise ToolError("this scene is built by a script: call scene_copy first to edit "
                            "a copy (the script stays as it is)")  # fmt: skip

        def go():
            rel = files.relative_path(file)
            files.resolve(e.base_dir, rel)  # inside the scene's folder, or ValueError
            seconds = file_seconds(e.base_dir, rel)
            if seconds is None:
                raise ValueError(f"{rel!r} is not audio in the scene's folder that can be read")
            t = e.timeline()["markers"][at] if isinstance(at, str) else float(at)
            words = text.strip() or Path(rel).stem.replace("_", " ")
            block = NarrationBlock(text=words, start=round(max(t, 0.0), 4),
                                   duration=round(seconds, 4), audio=rel)  # fmt: skip
            narration = add_block(e.document.narration, block)
            return self.scene_edit({"op": "set_narration",
                                    "narration": narration.model_dump(mode="json")})  # fmt: skip

        r = self._guard(go)
        if not r["ok"]:
            raise ToolError(r["errors"][0]["message"], r["errors"])
        return {**r, "duration": e.duration}

    def sound_report(self) -> dict[str, Any]:
        """What the soundtrack sounds like, in numbers (you cannot hear it): loudness in
        LUFS of the whole mix and of each lane (voice, every bed, every cue) in each spoken
        narration block, how far each bed sits under the voice there and how deep it ducks,
        the final trim and true peak, and the sound lint issues."""
        return self.engine.sound_report()

    # ------------------------------------------------------------ checkpoints

    def checkpoint(self, name: str) -> dict[str, Any]:
        """Remember the current document under ``name`` (to compare or return to later)."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name) or name in ("current", "saved"):
            raise ToolError(
                "checkpoint names are 1-40 letters, digits, _ . -  (not 'current'/'saved')"
            )
        e = self.engine
        if name not in e.checkpoints() and len(e.checkpoints()) >= MAX_CHECKPOINTS:
            raise ToolError(f"at most {MAX_CHECKPOINTS} checkpoints; reuse a name")
        e.checkpoint(name)
        return {"checkpoint": name, "version": e.version, "all": e.checkpoints()}

    def restore(self, name: str) -> dict[str, Any]:
        """Go back to a checkpoint, as one step (undo brings you back again)."""
        r = self._guard(lambda: self.engine.restore(name))
        return {**self.summary(), "restored": name, "version": r.version}

    def _version_of(self, ref: str) -> Document:
        if ref == "current":
            return self.engine.document
        if ref == "saved":
            path = self.path
            if path is not None and path.suffix == ".py":  # a script's saved document
                path = path.with_name(path.stem + io.DOC_SUFFIX)
            if path is None or not path.exists():
                raise ToolError("nothing saved yet")
            # The saved document as it takes effect: base plus its overrides, like "current".
            allow = self.allow_scripts
            return self._guard(lambda: Engine.open(path, passes=None, allow_scripts=allow).document)
        if ref in self.engine.checkpoints():
            return self.engine.checkpoint_document(ref)
        raise ToolError(
            f"no version {ref!r}; use 'current', 'saved' or {self.engine.checkpoints()}"
        )

    def diff(self, a: str = "saved", b: str = "current") -> dict[str, Any]:
        """What changed from version ``a`` to version ``b`` ('current', 'saved' or a checkpoint)."""
        return self._guard(
            lambda: {"a": a, "b": b, **diff_documents(self._version_of(a), self._version_of(b))}
        )

    # ------------------------------------------------------------ seeing

    def inspect(self, t: float = 0.0, id: str | None = None, detail: str = "standard") -> Any:
        """What is on screen at ``t``: ``brief`` (one line each), ``standard`` or ``full``.
        ``full`` also gives each object's ``params`` as the document sets them (expressions
        stay expressions): how you see what a person changed in the editor."""
        if detail not in ("brief", "standard", "full"):
            raise ToolError("detail is brief, standard or full")
        out = self._guard(lambda: self.engine.inspect(t, id=id, detail=detail))
        if detail == "full" and isinstance(out, dict):
            doc = self.engine.document
            for o in out.get("objects", []):
                spec = doc.object(o["id"])
                if spec is not None:
                    o["params"] = _jsonable(dict(spec.params))
        return out

    def lint(
        self, fix: bool = False, strict: bool = False, step: float = 0.5, aesthetic: bool = False
    ) -> dict[str, Any]:
        """Find visual problems (overlap, off-screen, contrast, tiny text, clutter, ...).
        ``fix=True`` applies the fixes that are certain, then reports what is left.
        ``aesthetic=True`` also judges taste: focal point, color restraint, glow clipping,
        density, motion rhythm, margins, stroke weights."""
        e = self.engine
        applied: list[dict[str, Any]] = []
        if fix:
            applied = e.autofix(aesthetic=aesthetic)["applied"]
        issues = e.lint(step, strict=strict, aesthetic=aesthetic)
        return {
            "clean": not issues,
            "errors": sum(i.severity == "error" for i in issues),
            "warnings": sum(i.severity == "warning" for i in issues),
            "issues": [_slim(i.to_dict()) for i in issues],
            "applied": applied,
        }

    def storyboard(
        self,
        times: list[float] | None = None,
        marks: bool = True,
        columns: int = 3,
        compare: bool = False,
    ) -> dict[str, Any]:  # fmt: skip
        """Render keyframes on one contact sheet (a PNG) with what the inspector sees in
        each. ``marks`` draws every object's box and id so you can tie picture to ids.
        ``compare`` puts each frame beside the three nearest frames in the project's
        ``gallery/`` (frames you love), with your notes on why they work."""
        from phasor.render.output import save_png

        e = self.engine
        gal = self.root / "gallery" if compare else None
        board = self._guard(lambda: e.storyboard(times, marks=marks, columns=columns, compare=gal))
        stem = f"{file_stem(e.document.meta.title)}.storyboard.v{e.version}"
        out = self.out_dir / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_png(out, board["image"])
        result = {"image": str(out), "frames": board["frames"], "version": e.version}
        if compare:
            cmp_path = self.out_dir / f"{stem}.compare.png"
            save_png(cmp_path, board["compare_image"])
            result.update(compare_image=str(cmp_path), compare=board["compare"],
                          gallery_size=board["gallery_size"])  # fmt: skip
        return result

    def events(self, t0: float = 0.0, t1: float | None = None, problems_only: bool = True,
               hz: float = 10.0) -> dict[str, Any]:  # fmt: skip
        """The event log: when things get hidden, leave or enter the frame, change region or
        gain an occluder, with the nearest marker and fix hints. ``problems_only`` keeps the
        ones a viewer would notice (a label or key object hidden or leaving the frame)."""
        from phasor.inspect.events import problems

        log = self._guard(lambda: self.engine.events(t0, t1, hz=hz))
        shown = problems(log) if problems_only else log
        return {"events": shown, "total": len(log), "problems": len(problems(log))}

    def when_hidden(self, id: str) -> dict[str, Any]:
        """Time spans where ``id`` is on screen but less than half of it is visible."""
        return {"id": id, "spans": self._guard(lambda: self.engine.when_hidden(id))}

    def free_space_near(self, id: str, t: float = 0.0) -> dict[str, Any]:
        """Empty label-sized places around ``id`` at ``t``, nearest first (boxes are 0-1,
        top-left origin): where a label or note could go."""
        return {"id": id, "t": t, "free": self._guard(lambda: self.engine.free_space_near(id, t))}

    # ------------------------------------------------------------ taste

    def _rubric(self) -> str:
        for folder in (self.root, docs.docs_root()):
            if folder is not None and (folder / "STYLE.md").exists():
                return (folder / "STYLE.md").read_text(encoding="utf-8")
        return ""

    def critique(self) -> dict[str, Any]:
        """The art-director's report: a taste score (0 = nothing to criticise), every
        aesthetic issue with the markers it happens at and its fixes (``fix`` is certain,
        ``suggested`` is judgement), and the rubric (STYLE.md) to judge the storyboard by.
        Look at the picture too: no rule sees everything."""
        from phasor.render import gallery

        report = self.engine.critique()
        report["rubric"] = self._rubric()
        report["gallery"] = len(gallery.entries(self.root / "gallery"))
        return report

    def polish(self, rounds: int = 3) -> dict[str, Any]:
        """Apply the critic's fixes (certain ones, then judgement suggestions), at most
        ``rounds`` (3) times; each round is undoable. Returns the score before and after and
        what is left for you to judge."""
        r = self.engine.polish(min(max(rounds, 1), 3))
        return {**r, "version": self.engine.version}

    def variants(self, beat: str | float, n: int = 3) -> dict[str, Any]:
        """Show alternative looks of one beat (a marker name or a time) side by side: as it
        is, calmer, bolder, each with its taste score. Nothing changes until you ``choose``."""
        from phasor.render.output import save_png

        e = self.engine
        r = self._guard(lambda: e.variants(beat, n))
        out = self.out_dir / f"{file_stem(e.document.meta.title)}.variants.v{e.version}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_png(out, r["image"])
        self._variants = {v["name"]: v["ops"] for v in r["variants"]}
        self._variant_beat = str(beat)
        return {"image": str(out), "beat": r["beat"], "t": r["t"],
                "variants": [{"name": v["name"], "score": v["score"], "changes": len(v["ops"])}
                             for v in r["variants"]]}  # fmt: skip

    def choose(self, winner: str, note: str = "", apply: bool = True) -> dict[str, Any]:
        """Pick the winner of the last ``variants`` call. The choice is appended to
        ``style/choices.jsonl`` (your taste as data) and, with ``apply``, its changes are
        made to the scene."""
        from phasor.style import choices

        if not getattr(self, "_variants", None):
            raise ToolError("call variants first: there is nothing to choose from")
        if winner not in self._variants:
            raise ToolError(
                f"no variant {winner!r}; the last call offered {sorted(self._variants)}"
            )
        e = self.engine
        entry = choices.record(self.root / choices.DEFAULT_PATH, scene=e.document.meta.title,
                               beat=self._variant_beat, options=list(self._variants),
                               winner=winner, by="ai", note=note)  # fmt: skip
        ops = self._variants[winner]
        result = self.scene_edit(ops) if apply and ops else None
        return {"recorded": entry, "applied": bool(result and result["ok"]),
                "edit": result}  # fmt: skip

    def hit_test(self, x: float, y: float, t: float = 0.0) -> dict[str, Any]:
        """Objects under an output-pixel point (origin top-left), topmost first."""
        return {"t": t, "x": x, "y": y, "ids": self.engine.hit_test(x, y, t)}

    def resolve(self, query: str, t: float = 0.0, limit: int = 5) -> dict[str, Any]:
        """Turn words like 'the pole on the left' or 'label of the loop' into object ids."""
        e = self.engine
        snap = e.inspect(t, detail="standard")["objects"]
        return {"query": query, "t": t, "matches": _resolve(e.document, snap, query)[:limit]}

    # ------------------------------------------------------------ outputs

    def review(
        self, step: float = 0.5, with_picture: bool = True, aesthetic: bool = False
    ) -> dict[str, Any]:
        """One call for the self-review loop: lint, what is on screen at every marker, and
        a storyboard. Fix what ``issues`` lists, then call again until ``clean``. Pass
        ``aesthetic=True`` for the beauty pass (taste rules on top of the correctness ones)."""
        e = self.engine
        lint = self.lint(step=step, strict=bool(e.document.narration), aesthetic=aesthetic)
        times = e.storyboard_times()
        marks = {t: n for n, t in e.document.timeline.markers.items()}
        frames = [
            {"t": t, "marker": marks.get(t), "objects": e.inspect(t, detail="brief")["objects"]}
            for t in times
        ]
        board: dict[str, Any] | None = None
        note = None
        if with_picture:
            try:
                board = self.storyboard(times)
            except (ToolError, RuntimeError) as err:
                note = f"no picture ({err}); judge from the inspector lines and lint"
        todo = [f"{i['code']} [{', '.join(i['ids'])}]: {i['message']}" for i in lint["issues"]]
        return {
            "clean": lint["clean"],
            "issues": lint["issues"],
            "todo": todo,
            "frames": frames,
            "storyboard": board["image"] if board else None,
            "note": note,
            "next": "done: report what you built and anything you could not resolve"
            if lint["clean"]
            else "fix the todo items (scene_edit, or lint with fix=true), then review again",
        }

    def render(
        self, preset: str = "preview", out: str | None = None, start: float | None = None,
        end: float | None = None, frame: float | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Render a video (preset: preview 960x540@30, draft 640x360@15 (fastest,
        roughest), final, 4k) or one PNG (``frame``)."""
        from phasor.render.encode import EncodeError
        from phasor.render.job import PRESETS, render_file

        if preset not in PRESETS:
            raise ToolError(f"preset is one of {sorted(PRESETS)}")
        e = self.engine
        stem = file_stem(e.document.meta.title)
        default = self.out_dir / (f"{stem}.png" if frame is not None else f"{stem}.mp4")
        target = self._inside(out) if out else default
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            return render_file(e, target, preset=preset, start=start, end=end, frame=frame)
        except (RuntimeError, EncodeError, OSError, ValueError) as err:
            raise ToolError(str(err)) from err

    # ------------------------------------------------------------ learning

    def docs_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search the API reference, patterns, pitfalls and examples."""
        return {"query": query, "results": docs.search(query, limit)}

    def object_types(self) -> dict[str, Any]:
        """What a patch may add: object types (use these exact names in ``add_object``) with their
        parameters and defaults and style keys (role group, default role), the role names each
        group accepts (style takes roles, never raw colors), and the themes."""
        out: dict[str, Any] = {}
        for name, cls in sorted(registered_types().items()):
            params = {}
            for k, f in cls.Params.model_fields.items():
                params[k] = (
                    "required" if f.is_required() else f.get_default(call_default_factory=True)
                )
            style = {
                k: {"group": g, "default": d} for k, (g, d) in getattr(cls, "Style", {}).items()
            }
            out[name] = {"params": _jsonable(params), "style": style,
                         "presets": sorted(getattr(cls, "Presets", {}))}  # fmt: skip
            if not is_builtin(cls):  # defined by a scene or a package, not shipped (ADR 0005)
                out[name]["scene_local"] = True
        return {
            "types": out,
            "roles": {g: list(r) for g, r in ROLE_GROUPS.items()},
            "themes": available_themes(),
            "looks": available_looks(),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, tuple | list):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return v if isinstance(v, str | int | float | bool | type(None)) else str(v)


# ---------------------------------------------------------------- resolve

_STOP = {"the", "a", "an", "of", "at", "on", "in", "to", "for", "and", "is", "that", "this"}
_REGION_WORDS = {"left", "right", "top", "bottom", "upper", "lower", "center", "middle", "centre"}


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _STOP]


def _resolve(doc: Document, snap: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Score objects against the words: id parts, tags, type, displayed text, screen region.
    "label of X" / "X's label" prefers the child ``X.label``."""
    words = _tokens(query)
    wants_label = "label" in words
    by_id = {e["id"]: e for e in snap}
    scored: list[tuple[float, str, list[str]]] = []
    for o in doc.objects:
        why: list[str] = []
        score = 0.0
        id_parts = set(re.split(r"[._]", o.id))
        text = ""
        tp = o.params.get("text")
        if isinstance(tp, str):
            text = tp
        text_tokens = set(_tokens(text))
        entry = by_id.get(o.id, {})
        region_tokens = set(re.split(r"[-\s]", str(entry.get("region", ""))))
        for w in words:
            if w in id_parts or w == o.id:
                score += 3
                why.append(f"id has '{w}'")
            elif w in {t.lower() for t in o.tags}:
                score += 2
                why.append(f"tagged '{w}'")
            elif w in text_tokens:
                score += 2.5
                why.append(f"text says '{w}'")
            elif w == o.type.split(".")[-1].lower():
                score += 1.5
                why.append(f"is a {w}")
            elif w in _REGION_WORDS and w in region_tokens:
                score += 1
                why.append(f"is on the {w}")
        if wants_label and o.id.endswith(".label"):
            score += 1.5 if score else 0
        elif wants_label and score and not o.id.endswith(".label"):
            score -= 0.5
        if score > 0:
            scored.append((score, o.id, why))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [{"id": i, "score": round(s, 2), "why": w, "region": by_id.get(i, {}).get("region")}
            for s, i, w in scored]  # fmt: skip


def _style_keys(type_name: str) -> set[str]:
    from phasor.core.objects import get_type

    cls = get_type(type_name)
    return set(getattr(cls, "Style", {})) if cls else set()
