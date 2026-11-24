# phasor style rubric

Your taste, in your words. The art-director pass critiques storyboards against this file and
`gallery/`. The sections below start from what you have said so far (2026-11-18); edit them
freely, and fill in the ones marked _yours to fill_. `critique` hands this text to the model
that judges the storyboard, so write it as you would brief a person.

## The house look
- **Dark is the default, always.** Dark stage, light ink. Light themes (`paper`, the `chalk`
  look) are opt-in, only when a scene asks.
- Glow should feel like light, not paint: soft halos, never a flat white blob (`glow_clip`).

## The reference look: `prism`
It comes from videos on the explicit formula for primes (Fourier over the zeta zeros):
- Near-black stage; functions glow with **faint different-colored halos** that stand out from
  the background (warm and cool fringes on the glow, a hue running along the curve).
- **A graph is not apparent unless needed**: no axes, ticks or frames by default.
- **Nothing is static**: the equations are LaTeX, but they drift; colors flow along curves.
- Color is welcome: amber and cyan terms in white equations, rainbow-running curves.
- Behind everything: nothing. (The light-blue streaks in the Fourier picture were that
  picture's own artifact, not part of the style.) `look="prism"` is all of this;
  `examples/prism_primes.py` is the reference scene.

## Alive, not rigid
- Nothing sits perfectly still: glow breathes, points drift a hair, color fields turn slowly.
  That is the `life` setting (`off`, `subtle`, `lively`); everything must also work with it
  off, and anything that states a value (a label, a computed pole) stays exact.
- Boundaries should not clash: edges between bands, regions and backdrops are feathered
  unless a hard edge is the point (`softness` on domain coloring).

## Color restraint
- Fewer rainbows, **not no rainbows**. Where hue carries meaning, use the whole wheel:
  domain coloring encodes phase as hue, so it stays a rainbow.
- Everywhere else, vary the palette and mix things up: two or three theme hues, sequential
  ramps (`ember`, `ocean`), a restrained wheel (`soft_phase`), one accent on a quiet field.
  Don't reach for the same rainbow twice (`color_restraint`: at most three saturated hues).

## Hierarchy
- One thing the eye lands on per beat, and it carries `highlight` (`focal_point`). Everything
  else steps back (`primary`, `secondary`, `muted`, dimming).

## Negative space
- _Yours to fill: how much empty frame feels right?_ Until then the linter's guard rails
  are a frame that is neither empty nor crowded (`density`) and a clear outer margin
  (`margin`).

## Motion rhythm
- Motions belong to the words: anchored to markers, by verbs, with room to breathe.
- Unrelated things should not start different motions in the same instant
  (`motion_rhythm`). _Yours to fill: what else should never move together?_
- Fast motion should look smooth, not stuttery: finals render with motion blur.

## Typography
- _Yours to fill: sizes, weights, when to use math vs text._ Until then: titles are `title`,
  labels `label`, math `math`; one stroke weight for plain lines (`stroke_mix`).
