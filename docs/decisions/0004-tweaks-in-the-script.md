# ADR 0004: a script scene keeps its tweaks inside the script

Status: **Accepted**, not built yet (2026-11-22). Touches contract 2
("Document is truth"). Nothing here is built yet.

## Context

The owner wants one shared script that the person, the AI and the app all edit
(2026-11-22): open or type a script, or import one, and every tool works on that same file.

Today a scene built by a script has two files. Structure and timing live in `scene.py`.
Tweaks made in the editor or by an AI over MCP live in `scene.overrides.json` (contract 2,
SPEC line 63). The script is not the whole truth, and a person reading it cannot see what
was changed in the app.

## Decision (proposed)

The overrides of a script scene move into a **managed block at the end of the script**,
written by phasor and readable as ordinary code:

```python
# --- phasor tweaks: written by the editor and the AI; edit freely, keep the markers ---
s.tweak("pole.label", "pos", (1.2, 0.4, 0.0))
s.tweak_style("loop", "color", "highlight")
s.tweak_rule("label(pole) near pole")
# --- end phasor tweaks ---
```

- **The same three operations as today, nothing more:** `set_param`, `set_style` and
  `add_constraint`, the ops `core/patch.py` already accepts on a script scene. Adding
  objects and changing timing stay code edits, by the person or the AI; the
  `scripted_scene` refusal stays.
- **A tolerant helper, not bare `s.set` lines.** `Scene.tweak*` records the tweak instead of
  failing when its id no longer exists, so the script still runs and the tweak is lint
  `orphan_override`, as today.
- **Canonical form.** One line per tweak, in sorted order, so rewriting the block never
  reorders or reformats anything. Code outside the markers stays byte for byte as it was.
- **Applied in memory.** A tweak goes through `Engine.apply` as now and is written into the
  block alongside; the script is not re-run for it (a rerun means a new bake, 155 s for
  `riemann_sqrt`). A full rerun happens only when typed code is run or saved.
- **Migration.** An existing `name.overrides.json` is folded into the block the first time
  its script is opened, then removed. `name.phasor.json` scenes (no script) keep their
  overrides file unchanged.

## The invariant that tests it

After any mix of typed edits, editor tweaks, AI edits and undo, **running the script fresh
gives the same `Document` the live engine holds**. A property test drives random sequences
of these through the `Workspace` and checks it after each step.

## SPEC change if accepted

Contract 2 (SPEC line 63): "GUI and AI tweaks to a scripted scene are saved in
`name.overrides.json`" becomes "are saved in a managed block at the end of the script
(ADR 0004); a scene with no script keeps `name.overrides.json`". The rest of the contract
(re-running never loses tweaks, `orphan_override`) is unchanged. The same edit goes into
SPEC line 110 and the `CLAUDE.md` contract summary.

## Alternatives

- **Keep the separate overrides file.** No contract change, but there is no longer one
  script everyone uses: what the app changed is invisible in the code.
- **Rewrite literals in place** (change `pos=(1, 2)` where it was written, with libcst).
  Nicer to read, but a value computed in a loop or a function has no single literal to
  change, so the block is still needed as the fallback. A later improvement, not v1.
- **The document is the truth and the script is generated from it** (`core/pycode.py`).
  Loses the person's own code (loops, functions, comments) on every round trip.
