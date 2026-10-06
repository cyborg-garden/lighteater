# lighteater

Turn **video and sound into flowing particle visuals**, in real time — a modular,
code-first take on the kind of generative effects you'd build in TouchDesigner, but driven from a terminal or a simple control panel instead of a GUI node graph.

Point your webcam at yourself and you dissolve into a luminous, flowing cloud of particles. Customisable and genuinely fun.

Dance, play music, and it moves with you.

<img width="1600" height="894" alt="image" src="https://github.com/user-attachments/assets/0f4a036a-ab8a-445a-9d43-00d871481a93" />


## Quick start

```bash
git clone https://github.com/cyborg-garden/lighteater
cd lighteater
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip   # stock macOS pip (21.2) can't do editable installs
pip install -e ".[person]"        # engine + person segmentation
python experiments/05-live-webcam/run.py
```

That `pip` upgrade is not optional on a stock macOS python3: the pip that ships
with 3.9's `venv` predates PEP 660, so `pip install -e .` stops with
`File "setup.py" or "setup.cfg" not found`. One upgrade and it installs.

On macOS you can also just **double-click `start.command`** (it sets up the environment on first
run, then opens the window). Grant your terminal **Camera** (and **Microphone**, for sound) access
in System Settings → Privacy & Security.

The `[person]` extra needs mediapipe, which ships an arm64 macOS wheel only — on an Intel Mac (or
anywhere mediapipe won't resolve) the install continues without it and the two templates that use
the person matte, `portrait` and `sigil`, fall back to another matte.

## The live instrument

One window, and it opens on the **home menu** — with the mode you used last already running
live behind it and already selected, so `Enter` is a one-key resume. A hint says so
(`enter resumes DITHER`), and once you're in, a second one names the three doors:
`m menu - TAB panel - ? keys`. A fourth door isn't in the hint: there's a small chevron in
the top-right corner, and clicking it opens the panel. It's there for as long as the HUD is,
so the mouse is never a dead end.

A launch flag means *boot into*, so `--mode`, `--still`, `--flock` and `--glitch` skip the
menu entirely.

**`TAB` cycles what's drawn over the render:**

- **hidden** — the picture, and after a beat nothing else. A toast you just triggered still
  finishes playing (so does the nudge readout, and an armed blackout keeps its amber corner
  tick) — but those all expire, and once they have, hidden leaves the frame untouched
  pixel-for-pixel. This is what you point OBS at.
- **HUD** — a small status line top-left, momentary toasts so every key you press says what it
  did, and the panel chevron top-right. This is where you start.
- **panel** — the full control sidebar on the right. Everything below lives here.

`Esc` always steps back toward hidden, and never quits.

## Modes

Four instruments share the window, the presets, the keys and the recorder:

- **Particles** (`p`) — camera → matte → a flowing cloud of glowing particles. The original.
- **Dither** (`d`) — the dither pipeline as the picture itself (see below).
- **Physarum** (`o`) — a slime-mold simulation that eats the live video: the matte blends
  every sim parameter between a "field" behavior and a "body" behavior (whoever is in frame
  runs different physics than the room), and the footage's light is food the veins grow
  toward. `X` swaps body/field mid-set; `B` pours a burst of agents onto you; `W` ripples
  the whole organism outward. Inspired by Sage Jenson's *36 Points* and Etienne
  Jacob's *interactive-physarum* (independent implementation of the Jones model).
- **Circuit Bender** (`j`): live JPEG databending. Every frame gets a sensor bend, is
  encoded as a real JPEG, has its bytes bent, and is decoded again. It lands on
  *bent cam*: a pink or green cast, slipped bands and posterised bursts, the way a
  circuit-bent camera looks; *thermal*, next, rings every light in thin rainbow
  contours. `E` steps through the eleven bends, `B` sets how hard,
  `X` split, `C` copy runs, `K` pixel sort, `L` long exposure. Inspired by CyberShot
  Cam by @lixofuturista / @cebolander (independent implementation from the JPEG
  standard). The browser page runs the same bends, byte for byte.

`m` opens the **home menu** — the same one you booted into: your camera behind a scrim,
rendered through 1-bit blue-noise dither, with a card per mode. `,`/`.` move, `Enter`
commits, or just click a card. `Esc` leaves the running mode alone — except at boot, where
there's nothing behind the menu to go back to, so it commits the selection instead of
dropping you into a mode you never chose. You never need the menu mid-set: `p`, `d`, `o` and
`j` switch directly from any state. Switching draws a still boot card rather than a gray flash,
and coming back to a mode you've already used finds it exactly as you left it.

From the command line, `--mode dithergirl` boots into Dither, `--mode physarum` boots
into Physarum, `--mode bender` into Circuit Bender, and `--still photo.jpg`
loads a still image and implies it; `--flock` and `--glitch` boot Particles with those layers
already on.

## The Particles panel

`TAB` to the panel. It's grouped into **collapsible sections** — click a header to open one.
`MOTION` and `SIGNAL` start closed, so the panel opens at the size it always did. When there's
more panel than window, a scrollbar appears in its left gutter with **arrow buttons** at the
top and bottom: a cv2 window's scroll wheel is the least portable input there is (on macOS it
never scrolled up at all), and the mouse is meant to be a fallback that can reach everything.

- **Templates** — one-click looks: `abstract` (glowing cloud), `portrait`/`textured`
  (recognizable, painted with your real colors), `embers`, `aurora`, and **`sigil`** (sharp
  fractal contour lines — move, then hold still and watch them form). Save your own with
  "Save current look" — it captures everything, including Video bg / Vid mix and Sound
  react / Sens. Hover a saved look for `~` (rename — type, Enter saves) and `x` (delete —
  click twice). Built-ins can't be renamed or deleted, and they never touch your live
  video/sound toggles when you switch to them. Click a row's slot badge to put that look on
  a number key (see below).
- **Source** — `matte` (what becomes particles: motion / saliency / person / edges / luma),
  `output` resolution (720p → 4K), and **Video bg** — show the raw camera footage behind the
  particles (screen-blended so the glow stays on top), with a `Vid mix` slider for how present
  it is. Recordings capture it too.
- **Look** — color palette + sliders for Trails, Glow, Spark, Flow, Size, Count, and the sigil
  knobs Glide / Pull / Reseed. Every slider has an `i` tooltip.
- **Motion** — **Flock**: Reynolds boids steering (cohesion / alignment / separation) applied
  to the particle cloud, so it organises into shoals and murmurations while still being shaped
  by the camera. Costs ~4 ms at 200k particles — the neighbourhood is approximated on a spatial
  grid, because true all-pairs boids at this particle count is ~4x10^10 pair terms.
- **Signal** — **Glitch**: the circuit-bent chain (chroma bleed, scan drift and sync tears,
  glitch blocks, bit-crush, CRT scan lines) plus Bayer / blue-noise / Floyd-Steinberg /
  Riemersma **dithering**, with `Pixel` (0–32, the dither's block size in output pixels — 0 is
  auto, 1 is full detail, higher is chunkier), `Bits` (1–4, how many shades survive), `Gamma`
  (dither in linear light so mid-tones keep their brightness — on by default; off is the
  crushed retro look) and `Bias` (which way the ordered dithers round: auto / light / dark).
  Everything under the Glitch switch is only drawn while Glitch is on, because the rack only
  runs while Glitch is on — a row that bends nothing doesn't sit there looking like a control.
  (Same rule elsewhere: the boids gains appear with Flock, `Vid mix` with `Video bg`, `Sens`
  with sound reactivity, and `Bias` only under the ordered dithers that read it. Hidden values
  are still saved and recalled with your looks.) It's applied
  to the finished frame, and it's a layer, not a mode — it's there in every mode. Costs ~6 ms
  at 720p, ~14 ms at 1080p. The control panel is drawn *after* it, so the panel never glitches
  into unreadability, and recordings capture what you see.
- **Audio** — toggle sound reactivity (bass pulses brightness; treble adds spark where Spark > 0)
  and a sensitivity slider.
- **Record** — capture an MP4 of your session.

## Playing it with keys

The mouse is a fallback here, not the instrument. Press **`?`** for the live key map — it's
generated from the commands themselves, so it can't go stale. The ones worth memorising:

| Key | What it does |
|---|---|
| `1`–`9` | recall the look in that bank slot, instantly (the built-ins start there) |
| `[` / `]` | previous / next in your setlist |
| `Space` | blackout — hard black out, with an amber corner tick so you know it's you |
| `0` | panic: the mode's known-good look, blackout off, glitch off |
| `TAB` / `Esc` | show more overlay / step back toward none |
| `p` / `d` / `m` | Particles / Dither / home menu |
| `f` `g` `v` `a` `r` | flock, glitch, video background, sound react, record |
| `s` | save the current look (panel only — saving is an edit, not a move) |
| `,` `.` then `-` `=` | pick a control and nudge it without opening the panel (`_`/`+` = ×5) |
| `q` | quit — press it **twice** within 2 s; the first press just asks. (The panel's Quit button and the window's close box still work.) |

Every key answers with a big momentary toast, so you can play in the dark without reading the
panel. Your saved looks live in `presets.json`, and the mode, bank and current look autosave to
`state.json` — both next to the app, both gitignored, so a restart lands you where you were.

## Dither

<img alt="Dither mode: Riemersma dithering, magenta palette, with the control panel open" src="docs/dither-panel.png" />

The same camera (or a still, via `--still photo.jpg`) run through the dither pipeline as the
*primary* image rather than as an effect on top:

- **Source** — camera or the loaded still, plus an optional matte so only the subject gets
  dithered.
- **Algorithm** — Bayer, blue noise, Floyd-Steinberg, Riemersma, or **ASCII**, with a
  live/slow badge and a swatch strip showing a grey ramp through your current settings.
- **Tone** — Bits, Gamma, Bias, Contrast, and **Scale**: the working height the dither runs at
  (30–720 px, default 72). Big cells are the look; they're also what survives streaming
  compression. Drag Scale up with an error-diffusion algorithm active and it says so in amber
  rather than quietly dropping frames.
- **Palette** — ten two-colour pairs: mono, amber, green phosphor, cyan, magenta, ice,
  blood, gameboy, sepia and hi-vis; plus six multi-colour ramps — aurora, ultraviolet,
  vaporwave, sunset, oil slick and cga — where each shade the dither produces lands on its
  own hue instead of a lighter version of one colour. (They need `Bits` 2 or more to show
  it: at 1 bit there are only two shades, so every palette is just its two ends.) Plus an
  **Invert** switch that flips the ink and the
  background of any of them. Every one was measured for
  contrast between its two ends and none ships under 4.5:1, because this is aimed at a
  projector rather than a monitor. Plus **Hue** and **Tint** — a direction and an amount, so
  you can dial a palette during a set instead of hunting a longer list. At Tint 0 every
  palette is exactly as named; turn Tint up and both ends steer toward Hue together, so a
  duotone stays a duotone and a black ground stays black.

### ASCII

The fifth entry in the algorithm cycle isn't a filter bolted on the side — it's the same
question the four dithers answer (*a smaller alphabet than the picture needs; how do you
spend it?*) answered with glyphs. It costs about the same as an ordered dither, so the badge
says `live`.

- The ramp is **measured, not folklore**. Every candidate character is rasterised by the font
  that will draw it, at the size it will be drawn at, and sorted by its actual ink coverage.
  The familiar `" .:-=+*#%@"` isn't monotonic under measurement — three or four of its nine
  steps go backwards, depending on stroke weight — which is why so much ASCII art has flat,
  muddy mid-tones.
- **Bits** becomes how many characters are in the ramp (2 / 4 / 8 / 16).
- **Scale** becomes character **rows**, not pixels — 45 rows is a 160×45 grid whether you're
  outputting 720p or 4K, so a look you author once stays the same look at any resolution. The
  panel prints the live grid under the slider, and tells you when the slider has run out of
  room rather than pretending it still does something.

The SIGNAL rack is available on top, minus its dither row — this mode owns dithering, and two
dither controls in one panel is how you end up with an incoherent instrument.

## The engine (`dtouch/`)

A small package of composable operators — sources, mattes, a particle-flow simulation, GPU glow
renderer, a 2D stable-fluids solver, audio analysis, and a tiny node-graph spine. Rendering is
headless (moderngl) so everything is scriptable and inspectable. See **[AGENTS.md](AGENTS.md)** for
the architecture and how to extend it, and **[docs/autonomy-pattern.md](docs/autonomy-pattern.md)**
for the self-verifying design philosophy.

## Experiments

Standalone studies, each in `experiments/NN-name/` with its own README and `run.py`:

| #  | Experiment        | What it explores                                        |
|----|-------------------|---------------------------------------------------------|
| 01 | displacement      | luminance displacement, instanced boxes, light/depth    |
| 02 | shadows           | depth-from-light shadow mapping                          |
| 03 | audio-reactive    | sound → displacement                                     |
| 04 | fluid             | stable-fluids advection of a particle field             |
| 05 | live-webcam       | the real-time interactive instrument (above)            |
| 06 | flocking          | Reynolds boids — order grown from local rules            |
| 07 | circuit-bent      | stochastic glitch + dithering: emulating bent hardware   |

Experiments 06 and 07 are also **built into the live instrument** as the `MOTION` and `SIGNAL`
panel sections — the standalone versions remain as focused studies of each idea (06 renders
boids in 3D through the instanced renderer; 07 bends the raw camera feed rather than the
particle render).

## Tests

```bash
pip install -e ".[person,dev]"   # pytest lives in the dev extra
pytest tests/ -q
```

The Quick start above installs `.[person]`, which does **not** include pytest — add `dev` (or
install it on its own) before running the suite.

Verified on Apple Silicon (Metal-backed GL 4.1). The render tests need a GPU/GL context.

## License

[GNU AGPL-3.0](LICENSE). If you run a modified version as a network service, you must offer users
its source. Contributions welcome — see [AGENTS.md](AGENTS.md).
