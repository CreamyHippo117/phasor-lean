# ADR 0005: an open object contract

Status: **Accepted** (owner, 2026-11-23). It touches contracts 5 (object
contract), 3 (time is pure) and 2 (document is truth). Built: see `examples/open_contract.py`
and `tests/test_open_contract.py`.

## Context

The owner, 2026-11-23: "it doesn't make sense to not have freedom like that". This compares
phasor with Manim, where any mobject can be subclassed, drawn with any Python, and moved by
per-frame updaters.

Today (`core/objects.py`), every type must have five members: `Params`, `world_bounds`,
`features`, `semantics`, and `draw(ctx, state, pass_)`. `Dot` takes about 40 lines and `Line`
about 80. Types live in phasor's packages, so a scene cannot define its own. Nothing runs
Python per frame: motion is tracks, expressions and `Simulation`s. The Manim importer bakes
updaters at their first look.

The five members are there because the inspector, layout, lint and camera solver need
bounds, features and facts about every object. That need is real. Requiring each author to
write all five by hand is not the only way to meet it.

## Decision (proposed)

1. **Two required members: `Params` and `draw(ctx, state)`.** `pass_` is optional: a `draw`
   that takes it still gets it. The `DrawContext` is already pass-agnostic, since the
   renderer decides what a command means in `color`, `id` and `solo`. Registration still
   fails without the two required members.
2. **Defaults derived from drawing.** A recording `DrawContext` (`core/record.py`, NumPy only,
   so `core` stays GPU-free) captures what `draw` emits, cached by (id, resolved params).
   From it:
   - `world_bounds` is the box around the recorded geometry;
   - `features` are `center` and the box's compass points, plus `start` and `end` for an
     object that draws one stroke;
   - `semantics` are `{"type", "params"}`.

   A type that knows more (poles, a tip, a period) still defines these members, and its own
   version wins.
3. **Composites.** `build(self, state) -> list[ObjectBase]` defines a type made of other types.
   Children get ids `parent.child`. Bounds and features are their union, and each child is
   inspected as itself.
4. **Types defined in a scene.** A type can be defined in the scene script or in a `*.types.py`
   file in the scene's folder. The document lists the modules it needs (`meta.modules`), read
   only from inside the scene's folder, like `Files` params (contract 3). A module is loaded
   only where scripts may run: the person's request, or `--allow-scripts`, the same rule as
   importers. A missing or refused module is lint `missing_type`, and the object draws as a
   placeholder box. Its parameters and overrides are kept, so nothing is lost (contract 2).
5. **Pure updaters.** `obj.always(fn)`, with `fn(t, look) -> {param: value}`. `look(id)` returns
   another object's resolved state. The engine records each read as an edge in the parameter
   DAG, and a cycle is a validation error, as for expressions. `state_at(t)` still depends
   only on the document, `t` and the files it names.
   - Purity is checked, not assumed: in tests, and in `lint(strict=True)`, each updater is
     evaluated twice and in shuffled order, and lint `impure_updater` fires when the results
     differ.
   - An updater that needs the previous frame (Manim's `dt` updaters) is a `Simulation`
     instead: fixed dt and a seed.
6. **`Value`.** A named, animatable number with no drawing, for Manim's `ValueTracker`. It can
   be read with `look` and shown by a number display.

## What does not change

- `Engine.apply` stays the only write path.
- ID rules stay, and `retired_ids` too.
- Time stays pure.
- The layer order stays: the recorder is in `core`; module loading is in `engine`.
- Kernels stay CUDA strings with NumPy references.
- `exact` objects and `Life` stay as they are.
- Types shipped *inside* phasor still follow the conventions: an API entry, an example, a
  storyboard test and goldens. Scene-local types do not need them.

## Consequences

- **Good:**
  - A new shape takes about 10 lines.
  - Manim subclasses and updaters can be carried over by the importer.
  - Milestones 16-19 get much smaller.
- **Cost:**
  - Each type pays for one recorded draw per new state; the cache keeps that to once per
    change.
  - Default features are coarser than hand-written ones, so labels on a scene-local type are
    placed by its box.
  - Scene-local code is code: it runs only where scripts are allowed, and a document that
    needs a module cannot be fully rendered without it.
- **SPEC edits on acceptance:**
  - contract 5's table: two required members, three derived;
  - contract 3: updaters are DAG nodes; the document may name code modules in its folder;
  - contract 2: `meta.modules` and `missing_type`;
  - CLAUDE.md's contract list, in the same commit.
