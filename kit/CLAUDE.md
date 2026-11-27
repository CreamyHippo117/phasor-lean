# Making phasor scenes

This folder is where scenes are made with **phasor**, a Python library for math animations
that can be checked: every object has an id, a linter finds visual problems, and a
storyboard shows what the frames really look like. You write scenes as Python scripts. The
person you work with has ideas; your job is to turn them into scenes that look good and
are right, checking your own work as you go.

Written by `phasor kit` ({kit_version}). Files under `kit/`, this file and
`.claude/skills/` are replaced on update; `TASTE.md`, `WISHLIST.md` and `scenes/` are the
person's and are never touched.

@TASTE.md

## Running phasor on this machine

```
{phasor} lint scenes/NAME.py --aesthetic          # problems, with fixes; exit 1 on errors
{phasor} storyboard scenes/NAME.py --marks -o out/NAME.storyboard.png
{phasor} inspect scenes/NAME.py --t 3.5 --detail brief   # what is on screen at t = 3.5 s
{phasor} render scenes/NAME.py --preset preview -o out/NAME.mp4
```

Run them from this folder. When the `phasor` tools are connected (`.mcp.json`), `review`
does lint, inspect and storyboard in one call and shows you the picture: open the scene
with `scene_open` (again after every edit to the file), then `review`.

## How to make a scene

The usual way is two commands: `/scene-plan <idea>` decides what to show and why and
writes `plans/NAME.md`; once the person approves it, `/scene-builder plans/NAME.md` codes
it. The steps below are what building comes down to, and serve for a small scene with no
plan.

1. **Understand the idea.** Restate it as a few beats: what appears, what moves, what the
   viewer should understand at each point. Ask at most one or two questions, only when the
   answer changes the scene.
2. **Start from the closest example.** `kit/examples/INDEX.md` lists them by kind. Model
   scenes show the look to aim for; technique scenes show how to do something (copy the
   technique, not the look). Never copy anything marked "not for you". The examples are
   placeholders for now: `TASTE.md` and `kit/docs/STYLE.md` outrank them on looks.
3. **Look things up instead of guessing.** `kit/docs/API.md` is the reference (objects,
   parameters, verbs, themes, looks); `kit/docs/PATTERNS.md` and `kit/docs/PITFALLS.md` are
   short and worth reading before a first scene; `docs_search` searches them all.
4. **Write `scenes/NAME.py`.** One `Scene` at module level, nothing else needed.
5. **Check it, then look at it.** Lint until there are no errors and no warnings you
   cannot justify. Then make the storyboard and **look at the picture yourself**: the
   linter cannot tell you a layout is ugly, a label is unreadable or the idea does not come
   across. Fix and repeat.
6. **Render a preview** and report: what the scene shows, the file, anything you could
   not do (and why), and what you would improve next.

A scene is not finished until lint is clean and you have looked at its storyboard.

## Rules

- Style takes roles (`highlight`, `primary`, `secondary`, `muted`, ...), never raw colors.
  Sizes come from the theme; never type pixel sizes.
- Name objects by meaning (`pole_top`, `loop`, `loop.label`), never `obj1`.
- A label is the child object `x.label` plus the rule `label(x) near x`; never type its
  position.
- One focal point at a time: keep `highlight` on one object; dim the rest.
- Narrated scenes anchor every motion to a `[[marker]]` in the script and use verbs
  (`reveal`, `trace`, `dim_others`, ...); check them with `lint --strict`.
- Dark is the default (`midnight_glow`); light themes only when asked. Hue that carries
  meaning (the phase wheel) keeps its colors; elsewhere, fewer colors.
- Do not edit phasor itself or the files under `kit/`. If phasor cannot do something
  the idea needs, say so, do the closest thing it can, and add a line to `WISHLIST.md`.

## Taste

When the person says what they like or dislike ("too fast", "I like the glow look", "never
put titles at the top"), add one dated line to `TASTE.md` and say you did. Read it before
every scene; it outranks the examples.
