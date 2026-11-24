# ADR 0001: layer order details and dependency inversion for GPU feedback

Status: **Accepted** by the owner on 2026-11-18 (touches contract 6)
Date: 2026-11-17, Milestone 0. Milestone 1 was built on it: `inspect` defines `PassRunner`,
`render` implements it, `Engine` injects it, and tier C falls back to analytic mode.

## Context

contract 6 fixes the import direction
`core ← time ← layout ← inspect ← render ← domains ← (gui, mcp, cli)`.
Three things in the spec do not fit that line as written:

1. **`style` and `engine` have no layer.** Choreography verbs in `time` read durations and
   easing from the theme, and lint rules in `inspect` read thresholds from it, so `style`
   must sit below `time`. `Engine` owns the document, state cache, layout bake and
   inspector, and must see object types from `domains`.
2. **`layout` needs render output.** The label placer's cost includes occlusion
   "from the ID buffer", and camera-solver stage 2 verifies with the real `id`/`solo`
   passes. Both live in `render`, which is above `layout`.
3. **`inspect` needs render output.** `visible_fraction` and `hidden_by` are computed from
   the `id` and `solo` passes, which also live in `render`.

## Decision

Layer order enforced by import-linter (`pyproject.toml`):

```
cli | gui | mcp | sketch      (independent of each other)
workspace                     (tools shared by the clients; Milestone 10)
importers                     (foreign scripts, e.g. Manim; Milestone 13)
scene                         (the Python authoring API; added in Milestone 1)
voice                         (narration, Milestone 4)
interactive                   (sliders on Engine.fork; Milestone 12)
engine
domains
render
inspect
layout
time
style
core
```

Points 2 and 3 are solved by **dependency inversion**, not by moving layers:

- `layout` defines a protocol, e.g. `OcclusionOracle.occlusion(ids, t) -> dict[id, float]`.
- `inspect` defines a protocol, e.g. `PassRunner.run(pass_, ids, t) -> ndarray`.
- `render` implements both. `Engine` (top of the stack) wires the implementations in.
- With no GPU (tier C), Engine injects analytic implementations (bounds only, no
  occlusion), and results say so with `"occlusion": "analytic"`.

Import direction from contract 6 is kept exactly; nothing in a lower layer imports `render`.

## Consequences

- Lower layers stay GPU-free and testable on any machine, as contract 6 requires.
- Tests for the label placer and camera solver use fake oracles, so they are fast and exact.
- Tier C needs an analytic inspector mode. The spec says tier C keeps "inspector" working,
  which is only true in this degraded sense. See MILESTONE_0.md, contradiction 3.

## Amendment (owner-approved 2026-11-19)

`interactive`, `importers` and `workspace` sit between `engine` and the clients, in the
order above. They changed no existing import; the owner approved all three on 2026-11-19.
