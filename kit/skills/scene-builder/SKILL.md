---
name: scene-builder
description: Build a phasor scene from a plan made by /scene-plan - code each animation of the plan as faithfully as phasor allows, check it, look at it, render a preview. Use when the person pastes a plan or points at one ("build this", "code this plan", "/scene-builder plans/NAME.md", a pasted block with Point A / Point B / Animations), or says a plan is approved. If there is no plan yet, run /scene-plan first.
argument-hint: "<plans/NAME.md, or the plan pasted in>"
---

# /scene-builder: from a plan to a finished scene

The plan: $ARGUMENTS

The thinking is already done. `/scene-plan` chose what to show, in what order, how long
each animation holds and where the voice is silent, and the person approved it. Your job
is to code that plan as well as phasor can, not to improve, shorten or re-plan it. If you
think the plan is wrong somewhere, build it as written and say so in the report.

The plan is either a file (`plans/NAME.md`) or pasted text. If pasted, save it to
`plans/NAME.md` first so the scene and its plan stay together. If what you were given is
a bare idea and not a plan (no path, no list of animations), stop and tell the person to
run `/scene-plan` on it.

`CLAUDE.md` in this folder has the commands and the rules; `TASTE.md` has the person's
taste, and it outranks everything else here.

## What "faithful" means

These come from the plan and are not yours to trade away for convenience:

- **One animation owns the stage**: full frame on midnight black, alone. Not a small
  figure in an empty frame, and nothing else sharing the frame.
- **The holds and the silences, and no more.** If the plan holds an animation for ninety
  seconds with a forty-second silent stretch, the scene does too. Do not pad: never add a
  replay, another case or a longer hold to fill time. A segment the plan marks as a
  showcase keeps its full length; it is the picture the math after it explains.
- **How it changes while it holds**: the plan's iterations, cases, parameters and big
  moves, in its order.
- **No hard cuts**: each animation becomes the next the way the plan says.
- **What color encodes**, and the same meaning for a hue throughout.
- **The math, as the plan builds it.** Every equation and number in the plan's inventory,
  on the word the plan lands it on, in the place the plan's look gives it. In particular:
  - *Term by term.* An `Equation` whose `parts` are the terms, each part in the colour of
    the object it names (`colors=`), written in with `write` as the word lands. A term the
    plan slides out of an object starts at that object: a one-part `Equation` of that term
    placed at the object, then `transform` it into the line (give the term the same `keys`
    entry in both). `transform` works between equations in the same layer, so this needs
    the line in the world layer beside its object; when the line lives in the overlay
    layer (a 3D or moving camera), flash the object as the term is written instead, and
    put the slide under "can do nearly".
  - *Derivations.* Each line is its own `Equation`; `transform(a, b)` carries the terms
    from one line to the next, and `keys=` names the terms so a term that changes (`+a`
    becoming `-a`) moves as one. Pair each step with the picture change the plan gives it.
  - *The ledger.* A finished result moves to the plan's ledger column: a copy with a
    smaller text role (`style={"size": "annotation"}`), `drift=0` so it holds still,
    reached by a `transform` from the full-size line in the same layer (`layer="overlay"`
    for both when the camera moves: screen units, origin top-left). The terms glide to the
    column, but take the smaller size at the transform's start; say so in the report if it
    shows, and add a size-changing transform to `WISHLIST.md`.
  - *Numbers.* Worked numbers beside their symbols; a value that changes on screen is a
    `Counter` or `sweep`'s readout, linked to the parameter, never typed in by hand.
  - *Matrices.* `linalg.Matrix` (its entries can follow a parameter); mathtext has no
    `pmatrix` or `cases`.
  - Only the text the plan names: no extra labels or captions.

## The steps

1. **Read the plan and list the work.** One line per animation: what fills the stage, how
   it changes, how long, how it becomes the next. Name the scene after the plan.
2. **Find the means.** For each animation, find in `kit/docs/API.md` (or `docs_search`)
   the object types, parameters, verbs and camera moves that make it; and in
   `kit/examples/INDEX.md` the closest example for its technique (copy the technique, not
   the look). Do not guess names. Sort each thing the plan asks for into:
   - **can do**;
   - **can do nearly** (say what differs);
   - **cannot do yet**.

   The plan's own "Wanted additions" section is a start, not the whole answer; phasor may
   have gained some of them since the plan was written.
3. **Say what will fall short, before building.** Show the person the nearly and cannot
   lists in a few lines, with the closest thing you will build for each. Add each cannot
   to `WISHLIST.md` (what was wanted, which scene, what was done instead). Then carry on;
   do not wait unless a missing piece removes the central picture itself.
4. **Write `scenes/NAME.py`, one animation at a time**, in the plan's order, on one
   timeline. Finish and check each before adding the next (steps 5 and 6), so a problem is
   found where it starts. For timing, take the plan's holds and silences; put a
   `[[marker]]` at each change and anchor each motion to its marker with a verb. Where the
   plan gives words to say, use them as placeholder narration, explanations included: the
   person will record their own. Each equation's `write` and each `transform` hangs on the
   marker of the word that names it.
5. **Check:** lint with `--aesthetic` and `--strict`. Fix every error. Fix each warning,
   unless it warns about something the plan asks for on purpose (a frame filled edge to
   edge, the whole color wheel where hue is structure, the planned equations and ledger
   on screen together); then leave it and name it in the report. Check each number on
   screen against the plan's claims list, and with `inspect` that each value the scene
   computes is the one the plan states.
6. **Look:** make the storyboard with `--marks` and look at the picture. For the
   animation you just added, answer:
   - Does it fill the stage, alone?
   - Does it hold as long as the plan says, and is something always changing a little?
   - Picture test: from the frames alone, does the viewer see the evidence the plan says
     this animation shows?
   - The math: is each equation readable at preview size, clear of the picture's subject,
     on screen before the narration leans on it, and in the colour of what it names? Is
     the one being built the only one growing? Does the ledger show every result so far?
   - Does it become the next animation without a cut?
   - Is every color doing what the plan says it encodes?

   Fix and go back to 5. At most 5 rounds per animation; then note what is left and move
   on.
7. **Preview:** render `--preset preview` to `out/NAME.mp4`.
8. **Report** in a few lines: each animation as built against the plan (as planned,
   nearly, or not possible), the files (plan, script, storyboard, video), what went to
   `WISHLIST.md`, warnings left on purpose, and anything in the plan you would question.
