---
name: scene-plan
description: Plan a teaching video, beautiful and full of real mathematics, before any code is written - brainstorm very widely with many subagents, prune to one path, and hand back a plan (the question, the central picture, the spine, an animation list with what to say and the equations on screen). Use whenever the person gives a topic or idea to teach or express ("a video about Lorenz attractors", "something on group theory", "how would you show X beautifully", "plan a scene/video on ...", "/scene-plan ..."), or asks how to make an idea beautiful, intuitive or well taught, even if they do not say "plan". Run it before /scene-builder, which codes the plan.
argument-hint: "<the idea or topic, in a sentence or a paragraph>"
---

# /scene-plan: from an idea to a plan for a beautiful video

The idea: $ARGUMENTS

This skill writes no scene code. It answers two questions, in this order: how is this idea
best taught, and how is that made beautiful? The first decides; beauty serves it. The
result is a plan the person approves; `/scene-builder` codes it afterwards.

`TASTE.md` is the person's taste and outranks everything here. Read it first. The house
rules below are a summary of it as of 2026-11-27; where they differ, `TASTE.md` is right.

## House rules

- **The aim is teaching, made beautiful.** A video is judged first by what the viewer can
  explain or derive at the end. Beauty comes from the mathematics shown well, never as a
  stand-in for it: a gorgeous picture that teaches little is a failure.
- **The spine.** Every plan states two to four sentences the video exists to make obvious
  ("a room is a matrix"; "sines are its eigenvectors"; "so in the Fourier basis it is an
  EQ"). Each part of the video proves one of them; a scene that proves none is cut.
- **The picture owns the stage, and the math shares it.** One animation fills the frame on
  midnight black and keeps it while it has work to do. Equations, numbers and a running
  ledger of results sit with it on screen (see "Math on screen").
- **Explain.** The voice carries the reasoning and the picture carries the evidence. Say
  why, in plain words, not only "watch". Silence is for one named thing to watch (the
  viewer is told what), and it ends once that has been seen. There is no quota of silence.
- **Never rush, never pad.** Hold as long as understanding needs and no longer.
- **A breathtaking scene gets its own segment, if it is worthy.** When a picture is
  breathtaking (the person's example: the opening two minutes of the reverb A1 plan, the
  clap in the room and the room unfolding into its crystal of mirror rooms), give it room
  to be watched: a long hold, little or no math, the music forward. Worthy means it shows
  something true that the video then explains, so the math that follows has a picture the
  viewer already loves. Such a segment is not padding and is never cut to save time; the
  math comes right after it, not instead of it. Mark it in the plan as a showcase segment.
- **Length follows content.** No target length: a video is as long as its spine needs.
  Fewer ideas make a shorter video, not a slower one.
- **Quality, not quantity.** As many animations as the best story needs (ten or three). None
  added to fill time, none dropped to save it.
- **Come back to the same picture** and test it on new cases, instead of a new picture for
  each point.
- **Repeat only to add something.** A picture, case or point comes back only when the
  viewer gains something new from it: a new question, a new sense, an outcome they could
  not predict. The weavers (step 2) decide where returns go and say what each one adds.
- **Open with a question** or a picture the viewer wants explained; never a title or a
  definition. **Let the viewer predict** before each reveal.
- **Usually not concrete before abstract.** Ask which route the person wants (step 4).
- **No hard cuts.** One animation becomes the next.
- **Stay true, and check.** Every number, demo and derivation is checked before the plan is
  handed back (step 9). A simplification may leave things out; it may not say something
  false.
- **Never shrink an idea to fit the tool.** If the best animation is one phasor cannot make
  yet, keep it in the plan, list it under "Wanted additions", and add a line to
  `WISHLIST.md`. The person adds these to phasor; a plan weakened in advance hides what is
  missing.

## Math on screen

The person's rules (2026-11-27, from their own plan "A Room Is a Matrix"):

- **Visual, then words, then symbols.** An idea arrives as a picture or a sound first, is
  named in words, and only then gets its symbol. Each term of an equation is assembled from
  the object it names: a spike becomes δ, a diagonal becomes h[k], a slider becomes H(ω).
- **One equation being built at a time.** Older results stay readable, but only one is
  growing.
- **A derivation step needs a visual change.** Each line on screen pairs with a change in
  the picture. Pure algebra with nothing to see happens off screen ("rearranging").
- **The ledger.** A finished result shrinks into a column at the edge of the frame and stays
  there, so the chain of results is always visible. The ending can reassemble it.
- **Numbers, not only formulas.** Give the worked number beside the symbol (10 ms, 50 Hz,
  about 10,000 echoes a second); the viewer should be able to check it.
- **Callbacks are written.** Where the result is one the viewer met before (for a viewer
  who knows ODEs: the integrating factor, Laplace's shift theorem, guessing e^(rt)), write
  it beside the new result, so the sameness is seen.
- **Colour ties symbol to object.** A term takes the colour of the thing it names, for the
  whole video.
- **The inventory.** Every plan lists each equation on screen, the visual that earns it,
  and any callback beside it. About one every one or two minutes is a fair density for a
  viewer at that level; the consult (step 1) sets it.

The references are the YouTube channel 2swap for the look and the pacing of a picture, and
the person's own plans for the explanation; see "The references" at the end.

## The steps

Work files go in `plans/NAME.work/` (NAME is a short file name for the idea); the plan
itself is `plans/NAME.md`. Have subagents write their output to files and return only a
two-line summary, so your own context stays free for judging.

Step 2 and 3 are meant to be expensive: dozens of subagent runs. That is the point. A plan
can only be as good as the ideas it was chosen from, and the first ideas anyone has are the
ones every existing video already uses. Say in one line at the start that this is a long,
heavy run, then do it in full; do not shorten it to save effort. If you have no way to
spawn subagents, play each lens yourself, one at a time, writing each one's file before
starting the next and without rereading the earlier ones until the merge.

### 1. Consult

Ask in one message, each with your suggested answer; skip what the idea already answers:

1. **Viewer**: who is this for, and what do they already know (which courses, which
   equations they have met)?
2. **Math on screen**: how much: a few named results, or derivations line by line?
3. **Feel**: calm, eerie, playful or grand? (Length follows from the spine; ask only for a
   ceiling, if the person has one.)

If the person does not answer, use your suggestions and say so in the plan.

Do not ask where the video should end, and do not decide it yourself. Point B, what the
viewer understands or feels at the end, is the most important thing the brainstorm finds:
fixed now, it would steer every lens toward one conclusion before any of them had looked.
If the person's idea already names a destination, pass it to the brainstormers as one
possible ending among those they are to find, not as the target.

### 2. Map

From a point A (an opening question) to a point B (what the viewer is left with) there are
thousands of paths: different questions, destinations, pictures, stories, animations and
orders. Neither end is given. See a great many before choosing.

**Cast about ten brainstormers** from the three families below: the idea's home fields,
the craft lenses, and at least three mathematical lenses that seem to have nothing to do
with the topic. The surprising picture usually comes from an outsider (billiards seen by a
hyperbolic geometer). Write the cast and why to `cast.md`.

Mathematical lenses: each asks "what is this, seen from my field?"

| Lens | Asks | Can split into |
| --- | --- | --- |
| Geometer | What shape or space is this? | Euclidean, hyperbolic, projective, differential |
| Algebraist | What is the structure; what is preserved? | groups and symmetry, representations, rings and polynomials |
| Linear algebraist | Which transformation is this; what are its special directions? | eigen-pictures, decompositions, high dimensions |
| Topologist | What survives stretching; what is glued to what? | surfaces, knots, coverings and unfoldings |
| Dynamicist | What happens when it repeats or flows? | orbits, phase space, chaos and bifurcation |
| Analyst | What happens in the limit, or at tiny scales? | approximation, Fourier, complex analysis |
| Combinatorialist | What is the space of all states, drawn as a graph? | counting, graphs, games and puzzles |
| Probabilist | What does a random one look like? | random walks, typical against extreme |
| Number theorist | Where do the integers or primes hide in this? | |
| Physicist | What machine does this; what is conserved? | mechanics, waves, optics |
| Computer scientist | What computes this? | |
| Duality hunter | What other object is secretly the same as this one? | |

Story lenses: skeptic (why would anyone doubt it), newcomer (what looks strange),
puzzle-setter (what game leads here), misconception collector (what people wrongly
believe), archivist (earlier ideas about the concept: how it was first pictured, older
approaches, forgotten constructions; a source of ideas, not a history lesson for the video).

Craft lenses: cinematographer (the unforgettable shot; how each picture moves, holds and
becomes the next; here it works from imagination and has not read phasor's verbs, which
come in step 7), musician (what can be heard), phasor expert (what phasor already draws
beautifully; reads `kit/docs/API.md` and `kit/examples/INDEX.md`), 2swap student (what
would 2swap do with this), textbook writer (which equations, derivations and worked numbers
carry this idea, and which results the viewer already knows that it is secretly the same
as).

**Round one.** Spawn the whole cast in one turn so they run in parallel and cannot see each
other. A brainstormer knows only its brief, so each brief carries: the idea, the viewer,
the feel, its lens and question, the house rules above, and these instructions:

- Fill a mind map with these branches: opening questions, destinations (what could the
  viewer understand or feel at the end that they did not at the start), central pictures,
  ways of seeing each insight, the math (equations, derivations, numbers the viewer can
  compute, results they already know), cases to test it on, connections to other
  mathematics and to the physical world, things that can be heard, transitions, endings.
- Quantity, no judging. At least 8 opening questions, 8 central pictures, 5 destinations
  (the obvious takeaway first, then the ones past it) and 4 in every
  other branch. Write the obvious ones down, then keep going past them.
- For every picture say how it fills the frame, how it moves, and what it proves: which
  claim the viewer can see is true in it. A picture that proves nothing is decoration,
  however long it can be watched.
- Do not ask whether phasor can draw it. (The phasor expert is the exception: it says what
  phasor draws well now.)
- Write to `plans/NAME.work/round1/LENS.md`.

**Merge and cross.** One agent reads all of round one and writes `map.md`: the branches
with repeats removed, each idea tagged with its lens, then a section of crossings (one
lens's motion on another's shape). It also names the two or three most fertile lenses,
and marks the fifteen or so strongest ideas of any kind for the weavers.

**Round two.** Spawn, in parallel: the sub-lenses of the fertile lenses (third column of
the table), plus two fresh brainstormers. Each gets `map.md` and asks: what is missing,
what is the opposite of the obvious choice, what does this look like at another scale, in
another dimension, run backwards? Their additions go into `map.md`.

Aim, across both rounds, for at least 40 opening questions, 40 central pictures and 20 in
every other branch. If the map is short of that, run another round with new lenses.

**Weave.** The lenses find connections; weavers find interconnections: two ideas from
different lenses carried by one object. (A video on differential equations drew pendulums
to show the geometry, then let the viewer hear how a stable pendulum sounds beside an
unstable one: the pendulum that is seen is the one that is heard.) Spawn three or four
weavers in parallel. Each gets the marked ideas from `map.md` and one way of pairing, so
they do not all find the same thing:

- across senses: a picture with a sound, a motion with a rhythm;
- across fields: one lens's object doing another lens's job;
- across scales or states: the small with the whole, a still shape with a process.

A weave is real when the second idea is a consequence of the first inside the same object,
so that removing either weakens the other. Two good ideas placed side by side are a staple,
not a weave; leave those out. "None worth it" is an allowed answer, and no video is obliged
to use a weave.

A weave brings something back: an object, a picture, a case, a sound. That return is
repetition, and it is allowed only when it adds something: a new question asked of the
same thing, a new sense (the picture now heard), a case whose outcome the viewer could
not predict, or another field's view that changes what the thing means. For every return
a weave makes, write in one sentence what the viewer gains from it that they did not
already have. If that sentence restates an earlier point ("again, the hands line up"),
drop the return; a weave whose second idea is its first one said again is a repeat, not
a weave. Weaves go in `weaves.md`, each with what the viewer would see and hear, and that
sentence for each return.

**Paths.** Two or three agents each join branches of the map into whole paths, a short
paragraph each, deliberately unlike one another (different openings, different
destinations, different central pictures, different orders). Each path names its own
point A and point B in its first line, then its spine (the two to four sentences it makes
obvious); the twenty should not all end in the same place. They read `weaves.md` too: a weave often makes the
best return to the central picture (the same object, now heard, or now seen from the other
field), and it comes with what that return adds. A path repeats a picture, case or point
only where it adds something in the same way. At least 20 in all, in `paths.md`.

### 3. Prune

Brainstormers never judge; use fresh agents. Five tests, the first counting double:

- **Content**: what can the viewer explain, derive or compute at the end that they could
  not at the start? The judge writes the list as sentences and equations the viewer could
  reproduce, and counts only reasons, mechanisms and results ("delay is multiplication by
  e^(-sτ), so one echo makes notches every 1/τ"), never impressions ("a room is like a
  crystal"). A short list fails the path however beautiful it is.
- **Beauty**: would a still from it be worth looking at with the sound off?
- **Intuition**: does the viewer see why, not only that? With the voice off, does the
  picture still show the evidence?
- **Cohesiveness**: is it one picture and one thread from A to B, each beat growing out of
  the last?
- **Immersion**: does the viewer stay inside it, with nothing that breaks the spell (a
  hard cut, a crowded frame, a wall of symbols, a change of look)?

1. **20 to 5.** Five judges, one per test, plus a simulated viewer at the level from step
   1, each score every path and give one line on its weakest point. Keep the five best
   that are still unlike one another.
2. **Develop the five.** One agent per path writes it out animation by animation, with the
   spine sentence each animation proves and its equation inventory. Weaknesses hidden in a
   paragraph show here: an animation that proves no spine sentence, or a stretch with
   nothing to explain.

### 4. Ask the person which route

Show the five in two or three lines each, starting each with its point A and point B, with
your recommendation and why. Ask which route they want, or which parts to mix. This is
where the person chooses where the video ends and how it teaches: it comes now, not
earlier, so the choices are real ones found for this idea. If they say "you choose",
choose.

### 5. Choose and attack

- **5 to 1.** Judge again with the person's answer in hand. Content counts double. Among
  paths that teach about as much, one good on every test beats one strong on some and weak
  on others. Parts of losing paths may be grafted on only if cohesiveness survives.
- **Attack the winner.** One agent argues as hard as it can that the winner is the wrong
  choice and proposes a better one. If it is right, go back to "20 to 5".

### 6. Teacher

Now make the chosen path teach as well as it can:

- **The central picture is the idea**, not a decoration of it.
- **Let the picture argue.** The viewer sees that something is true instead of being told.
  (To show two operations are the same, run them side by side on identical objects and let
  them move identically.)
- **Say, show, say.** For each stretch: a setup in words that says what to watch for and
  why it matters, then the animation, then the explanation: what was seen, why it is so,
  and what it means for the spine. A silent stretch names its one thing to watch and ends
  once it has been seen.
- **Write two tests** for each animation. Picture: with the voice off, which evidence does
  it show? Voice: with the picture off, does the narration still give the reason? A scene
  needs both; "the animation does the talking" alone is not teaching.
- **Write the takeaway** for each animation: the sentence or equation the viewer could
  reproduce, and the spine sentence it serves.
- **Build the math** to the rules in "Math on screen": the inventory, each equation's
  visual, the ledger, the worked numbers, the written callbacks. A derivation with a
  picture for each step is the best thing a stretch can hold; do not cut one to keep the
  screen bare.
- **Name what is hard**: the one step where people get lost. Give it the most time.
- **Plan the returns**: cases that run through the same animation, each changing one thing
  and teaching something no earlier case did. The last should surprise.
- **Place the predictions**: points where the picture holds before answering.
- **Earn every symbol.** A word or equation appears only after the viewer has seen the
  thing it names.
- **End on the picture, changed**: the opening picture again, now understood.
- **Stay true.** A simplification may leave things out; it may not say something false.
  Note what was left out.

### 7. Cinematographer

One agent owns everything the eye gets: the stage, color, motion, the camera, the verbs,
the transitions and the text. It may not change what is taught, only how it looks, and it
works on the Teacher's draft, so every choice lands on a hold, a silent stretch or a word
the Teacher already placed. Where a visual choice would make an animation fail its picture
test or hide its math, the teaching wins.

**Stage.** Midnight black. One animation, full frame: the mathematics reaches the edges,
not a small figure in the middle of a slide. It keeps the stage; do not cut away and come
back. Variation comes from inside it. Full, bright color is welcome when the mathematics
is that full (a domain coloring, a tiling).

**Color.** Hue is structure: assign color by something mathematical (phase, position in a
lattice, which tile, time along a path), and then the whole wheel is welcome. Where color
encodes no quantity, use two or three hues, each with a meaning that holds for the whole
video, and write the meanings down. Choose the palette from the feel of the subject.

**Motion.** Draw things on rather than switching them on; how a thing appears should show
how it is made. Slow is the default. During a hold something is always changing a little.
With the voice, each motion belongs to a word; without it, motion follows the music. One
animation becomes the next: an object carries over, morphs, or the camera travels through.

**The big moves** are how an animation renews itself while keeping the stage, mostly in
the silent stretches:

- **One number turns.** The camera is still, one anchor stays fixed, and a single
  parameter changes slowly so the whole picture reorganises around the anchor. The
  workhorse.
- **Sweep.** The camera travels steadily across or around a structure too large or too rich
  for one frame, so the viewer discovers it piece by piece. "Look how much is here."
- **Lift.** The camera pulls up and away from a detail the viewer has been studying until
  the whole is in view, and the detail turns out to be one part of a larger pattern. "What
  you just understood is everywhere." Its reverse, the descent, goes from the whole into
  one part.
- **Passage.** The camera goes into or through the object until its color fills the frame,
  and comes out in the next animation.

Each must show something the viewer could not see before it began, start and end at rest,
end on a composed frame that then holds, and bring into view only things already alive.

**Math and text.** The equations of the inventory are part of the stage, not laid over it.
Give them a place that holds for the whole video: a band (top or bottom) for the equation
being built, and a column at one edge for the ledger, both clear of the picture's subject.
Build each one term by term out of the object it names (the term leaves the picture and
slides into the line), in the colour of that object. Write numbers beside symbols. Keep a
finished result readable until it shrinks into the ledger. Text that is neither an
equation nor a number is a large plain label, used sparingly.

**Verbs.** Only now, with the path chosen and drawn in imagination, read phasor's verbs
and camera moves in `kit/docs/API.md` (the list grows; read it each time, do not work from
memory). The order matters: ideas formed with the tool list in view shrink to fit it.

- Give each motion in the plan the verb that does it best.
- Then read the list once more for inspiration: is there a verb that would show this
  object in a way nobody imagined? Verbs are welcome anywhere, and especially in the long
  holds and in a pause on a great idea, where a well-placed verb keeps the picture alive
  while the viewer thinks.
- Every verb must either reveal something or keep the picture alive. One that does neither
  is decoration; leave it out.
- Mark on each animation where each verb lands (which moment of the hold, which word), so
  Sound can meet it.
- A motion the plan wants that no verb or animation can do stays in the plan as imagined
  and goes under "Wanted additions": what it should do, and where the plan uses it. Do not
  swap it for a weaker motion that exists.

### 8. Sound

Mark cues on the animation list; do not compose. Work on the Cinematographer's draft: an
accent belongs where a verb lands.

- **Music**: one ambient bed, sitting back under the voice and coming forward in the
  silent stretches.
- **Sounded mathematics**: where the subject has a natural sound, note it as MIDI (a
  frequency, a rhythm, a pitch that rises with a value, a note for each element or return).
- **Accents**: a soft sound on a reveal, a landing, a collision. Few, quiet, the same sound
  for the same kind of event.
- **Silence**: a short quiet before the payoff.

### 9. Claims check

A fresh agent, which wrote none of the plan, checks every claim it makes: each number,
each equation in the inventory, each derivation, each demo ("these two clicks sound the
same"), each "the viewer will hear ...". For each it derives the result, computes it (a
short NumPy check is fine), or marks it unverified, and writes `plans/NAME.work/claims.md`.
It looks hardest at the edges of a model: a property true of the infinite case but shown
on a finite one, a limit taken silently, a demo whose physics or hearing does not work as
said.

A false claim is fixed, never softened into vagueness. A claim that needs a listening test
or a simulation nobody has run stays in the plan marked as such, with what to say if it
fails.

### 10. Editor

A fresh agent reads the plan as the viewer from step 1 and answers each; fix what fails:

1. Do the first ten seconds make me want the answer?
2. Do I reach this path's point B, and feel it? Is it a destination worth a whole video?
3. Could I reproduce the spine at the end: say each sentence and write each key result?
   Which animation taught each one?
4. With the voice off, do I still see the evidence? With the picture off, do I still get
   the reason? A scene that passes only one is half a scene.
5. Is the voice only describing what the animation shows? Replace it with the reason.
6. Does every animation prove a spine sentence (a showcase segment counts when the math
   right after it explains what it showed)? Cut any that fills time; restore any cut
   to save it. Is any stretch padded: still running after I have understood it?
7. Is any animation taken off the stage before it has finished its work?
8. Is any animation small, or sharing the frame with anything but its own math?
9. Is every equation earned by a visual, built one at a time, readable, and kept (in the
   ledger) while later steps need it? Is any step a jump I could not follow?
10. Does anything break the spell?
11. Does each color still mean what it meant at the start?
12. Did the claims check pass, and is every unverified claim marked?
13. Held next to the references below, where does this fall short?
14. Was any idea weakened to fit phasor? Restore it and list it under "Wanted additions".
15. Is any verb there only as decoration? Cut it.
16. If the plan uses a weave, do the two ideas truly live in one object, or were they
    stapled together?

If the Editor finds the path itself weak, go back to the map; do not patch.

### 11. Hand back

Write `plans/NAME.md` in this shape, then show it to the person and stop. Do not start
`/scene-builder` until they approve.

```markdown
# NAME: the title

**Point A:** the opening question
**Point B:** what the viewer understands or feels at the end
**Viewer:** ...   **Math on screen:** ...   **Feel:** ...   **Length:** ... (from the spine)

## The spine
The two to four sentences the video makes obvious, each with the animations that prove it.

## The path
Three or four sentences.
Runners-up: the two best paths that lost, one line each, and why.

## Look
The central picture. What color encodes. Where the equations and the ledger sit. The
music bed.

## Animations
### 1. What fills the stage
- **Proves:** which spine sentence; **takeaway:** the sentence or equation the viewer keeps
- **While it holds:** how it changes, in order (iterations, cases, parameters, big moves),
  with the verb for each motion and where it lands
- **Say before:** what to watch for, and why it matters
- **On screen:** each equation and number, the visual that earns it, the word it lands on,
  and when it goes to the ledger
- **Silent stretch:** if any: how long, and the one thing to watch
- **Say after:** the explanation: why it is so, and what it means for the spine
- **Sound:** ...
- **Becomes the next by:** ...

## Equation inventory
| Animation | On screen | Visual that earns it | Callback beside it |

## Claims
Each number, demo and derivation: checked (how), or unverified (what test, and what to say
if it fails).

## Left out
What was simplified away.

## Wanted additions
Each animation or verb the plan wants that phasor does not have yet: what it should do,
which animation uses it, and the closest thing that exists today.
```

Add each wanted addition to `WISHLIST.md` as its own line too. Keep `map.md` and
`weaves.md`; a later video on a nearby idea can start from them. In your reply give the
path in a few sentences, the list of animations by title, the wanted additions, and the
file.

## The references

**For the explanation: the person's own plans** ("A Room Is an EQ" and "A Room Is a
Matrix", 2026-09-30). What to take from them: a spine of three sentences, each proved by
one act; every scene that does not serve one is cut. Every equation is earned by a visual
and assembled from it, with a ledger of results and the viewer's earlier courses written
beside the new results. Numbers worked out on screen. Predictions before the reveals. The
whole video is one mathematical move (there, one diagonalisation), and the ending
reassembles the chain.

**For the look and the pacing of a picture: 2swap.** What the person admires, in their
words: an animation takes the whole stage and keeps it, and the sameness of two things is
seen, not stated. (The person's earlier wish for long silences and brief words is replaced
by "Explain" above, 2026-11-27.) Two clips they named (described from their account and
still frames):

- **"don't flip, algebra inside", 6:55 to 7:10.** The video opens on a full-frame color
  field before any explanation. Two identical rainbow lattices sit side by side, labelled
  only "×i ×j" and "×ij": the sameness is seen, not stated. Within seconds the lattice
  gives way to color filling the frame as the chapter changes.
- **"i'm so bad at billiards that i ended up in the hyperbolic plane", 5:00 to 5:40.** One
  green polygon stays fixed at the centre while the whole tiling around it changes shape
  continuously; every tile its own color, finer and finer detail in the gaps; no text; one
  picture, slowly changing, held for about 40 seconds.
