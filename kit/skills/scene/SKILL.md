---
name: scene
description: Turn an idea for a math animation into a finished phasor scene - beats, script, checks, storyboard, preview. Use when the person describes a scene they want ("a scene about ...", "animate ...", "/scene ...").
argument-hint: "<the idea, in a sentence or a paragraph>"
---

# /scene: from an idea to a finished scene

The idea: $ARGUMENTS

Follow these steps in order. `CLAUDE.md` in this folder has the commands and the rules;
`TASTE.md` has the person's taste, and it outranks everything else here.

1. **Beats.** Restate the idea as 3 to 7 beats, one line each: what is on screen, what
   moves, what the viewer should get. Give the scene a short file name. If something would
   change the whole scene (2D or 3D? narrated or silent? how long?), ask once, in one
   message, with your suggested answer; otherwise choose and say what you chose.
2. **Closest example.** Pick it from `kit/examples/INDEX.md` and say which one and why.
   Read it. Build on its structure; copy a model scene's look, a technique scene's
   technique only.
3. **Look up** every object type, parameter and verb you are not sure of in
   `kit/docs/API.md` (or `docs_search`). Do not guess names.
4. **Write** `scenes/NAME.py`. Narrated: write the narration with a `[[marker]]` at each
   beat and anchor each motion to its marker with a verb.
5. **Check:** lint with `--aesthetic` (and `--strict` for narrated scenes). Fix every
   error. Fix each warning, or say in the report why it stays.
6. **Look:** make the storyboard with `--marks` and look at the picture. Does each beat
   read? Is anything cut off, cramped, unreadable, or competing for attention? Fix and go
   back to 5. At most 5 rounds; then stop and report what is left.
7. **Preview:** render `--preset preview` to `out/NAME.mp4`.
8. **Report** in a few lines: the beats as built, the files (script, storyboard, video),
   the example you started from, anything you could not do (also added to `WISHLIST.md`),
   and the one change you would make next.
