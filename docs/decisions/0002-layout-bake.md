# ADR 0002: what the layout bake depends on, and constraints in the overrides file

Status: **Accepted** by the owner on 2026-11-18, with both recommended options (touches
contracts 2, 3 and 7; their text was updated in the same commit)
Date: 2026-11-17, Milestone 2. Milestone 2 is built on both points below.

## 1. Overrides may hold constraints (contract 2)

Contract 2 says tweaks to a scripted scene are stored in `name.overrides.json` "keyed by
object id + parameter path". Milestone 2's auto-fix adds layout rules
(`label(pole) near pole`), and on a scripted scene those have to be stored somewhere that
survives re-running the script.

**Decision:** the overrides file gets a third list, `constraints`, holding rules added on
top of the script's own. A rule that names an object the script no longer creates is an
`orphan_override` lint error, like any other dangling tweak. If the script later adopts the
same rule, the override is ignored. Removing a rule that the script itself wrote is still
refused (`scripted_scene`).

**Alternative:** refuse `add_constraint` on scripted scenes, and have auto-fix print rules
for a human or AI to paste into the script. Simpler, but "flag, then auto-fix" stops
working for the most common kind of scene.

## 2. The bake depends on how occlusion is measured (contracts 3 and 7)

Contract 7 says the label placer's cost includes "occlusion (from the ID buffer)".
Contract 3 says `state_at(t)` depends only on (document, t).

The bake is a pure function of (document, occlusion oracle). On a machine with OpenGL the
oracle is the real id pass at 480x270; with no OpenGL (tier C) it is geometry: bounding
boxes, and a thin stripe for anything with a `curve` feature. So two machines can, in a
near-tie, choose different slots for a label, and `state_at(t)` then differs between them.
Hysteresis and the tie tolerance make this rare; the test suite's scenes give identical
results in both modes.

**Decision:** keep the id pass where it exists, as contract 7 says, and
record the oracle kind in `engine.layout`. Within one machine the result is deterministic.

**Alternative:** always use the geometric oracle for layout, so the bake is identical
everywhere, and use the id pass only for linting. That satisfies contract 3 to the letter
and gives up pixel-accurate avoidance (curved shapes would be treated as boxes until they
expose finer footprints).

## Also decided here (no contract change)

- **A label that changes slot jumps.** Sliding it between opposite slots drags it across its
  own anchor. A proper eased move belongs to the choreography verbs.
- **Baked positions are stored on screen, relative to the anchor.** Between key times only
  the offset is interpolated, so a label is glued to a moving anchor exactly, and the gap
  holds while the camera zooms.
- **The lint delta in `PatchResult` is lazy.** It needs the bake and the inspector; working
  it out on every builder call would make scripts slow. Reading it gives the same answer.
