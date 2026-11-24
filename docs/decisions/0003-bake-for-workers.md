# ADR 0003: how parallel render workers get the layout bake

Status: **Accepted** by the owner on 2026-11-22, option A. Touches contract 7 ("Layout
is baked": only `Engine` runs or applies the bake); its text was updated in the same commit
as the build (`Engine.worker_payload`, `Engine.from_payload`, `scene.parallel_frames`).

## Context

`GPU_SPEED.md` step 4 renders a video with several worker processes, each taking every Nth
stretch of 30 frames while the parent writes finished frames to the one encoder, in order.
Only one of the CPU's 16 threads is busy today, so about 3x on wall time is expected.

Each worker builds its own `Engine` from the document. As things stand, each would then
bake the layout again: label placement, the camera solver and the id-pass occlusion
oracle. That is cheap for most scenes and 155 s for `riemann_sqrt`, which would make
workers slower than one process there.

Facts that limit the options:

- `layout.bake.Bake` is plain frozen data (key times, tracks, issues, solved camera, label
  slots, the oracle kind). It pickles.
- The bake is a pure function of (document, occlusion oracle) (ADR 0002). The oracle is
  the real id pass where OpenGL exists, so the result also depends on the machine's GL
  renderer and on any code that draws into the id pass: every object type, text
  rasterization and the theme.
- Windows has no `fork`: workers are spawned and start from nothing.

## Option A (recommended): the parent hands its bake to the workers

The parent `Engine` bakes once (it needs the bake anyway: lint, the inspector and the first
frames). The worker driver passes `(document, overrides, bake)` to each worker, and the
worker's `Engine` is built with that bake instead of computing it:
`Engine.from_document(doc, overrides, bake=...)`, an argument only the worker driver uses.

- Contract 7: still only `Engine` applies a bake, and the bake was run by an `Engine` (the
  parent's) on the same document, with the same oracle, on the same machine. The worker
  checks the document hash that came with the bake and refuses a mismatch.
- Nothing is written to disk and nothing can go stale: the bake lives only for one render.
- Every worker's frames are exactly the parent's frames, because they use the very same
  bake. With independent bakes, a near-tie in the label placer could differ only if the
  occlusion oracle differed, which on one machine it does not, but A makes that a non-question.
- Cost: pickling the bake once per worker (kilobytes to a few MB).

SPEC change: contract 7 gains one sentence, "a worker process may apply the bake its
parent Engine computed for the same document (ADR 0003)".

## Option B: an on-disk bake cache

What `GPU_SPEED.md` first proposed. `Engine` writes the bake to
`.phasor_cache/bake/<key>.pkl` and reads it back when the key matches, like simulation
checkpoints (`core/simulation.py`).

- Also speeds up re-opening and re-rendering the same scene later, not only workers.
- The key must cover everything the bake depends on: the canonical document with its
  overrides, the theme files, the oracle kind and the GL renderer string, and the source of
  every module that draws into the id pass or solves layout (in practice: the whole
  `phasor` package and the versions of matplotlib, NumPy and kiwisolver). Missing any one of
  them gives a stale bake that is wrong silently: a label in the wrong place with no error.
- Writes must be atomic (temporary file, then rename), since workers start together.

SPEC change: contract 7 gains "the bake may be read back from the cache under a key of
everything it depends on (ADR 0003)".

## Option C: every worker bakes again

No contract change. Fine for most scenes and bad for heavy-layout ones (155 s per worker
for `riemann_sqrt`, all workers at once). Would need a rule like "use workers only when the
bake took under N seconds", which is hard to explain.

## Recommendation

Option A now: it is small, it cannot go stale, and it guarantees that workers draw the same
frames as one process. Option B can come later on its own merits (faster re-opening of
heavy scenes) with its own tests for every part of the key.

## Tests (for A)

- `test_parallel_render_equals_serial_render` (`gl`): a short scene to a PNG sequence with
  1 and 3 workers: every PNG is byte-identical; the mp4 has the same frame count and the
  same decoded frames. Lit surfaces stay out of this scene (they have run-to-run noise on
  the RTX 4070 Ti; `GPU_SPEED.md`).
- `test_a_worker_never_bakes`: a worker given a bake computes none (the bake function
  patched to fail inside workers).
- `test_a_bake_for_another_document_is_refused`: a document hash mismatch is an error.
