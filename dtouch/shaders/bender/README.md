# bender: Circuit Bender, shared with the browser

The third browser-shared unit (after `../physarum/` and `../dither/`).
cyborg-garden-site's `/games/lighteater` page vendors this directory
**verbatim** with `scripts/sync-lighteater.mjs` (unit `bender`). The desktop
is the source of truth for everything here.

Circuit Bender is live JPEG databending of the camera: a sensor bend on the
pixels, a real JPEG encode, a bend on the bytes, a decode. The desktop mode
is `dtouch/modes/bender.py`; the browser's is `src/scripts/bender/`.

| file                  | what                                                       |
|-----------------------|------------------------------------------------------------|
| `post.frag`           | COPY (block runs, echoed bands) then SPLIT (aberration)    |
| `stack.frag`          | the long exposure and the settle: a running mean           |
| `view.frag`           | the working picture onto the canvas, sharp bilinear        |
| `bender_looks.json`   | the tables the page bundles (see below)                    |
| `bender_goldens.json` | bend outputs both ports must reproduce, byte for byte      |
| `fixture-*.jpg`       | the goldens' input JPEGs: one each from cv2, ffmpeg, Chrome |

## The shaders

The browser runs all three on the GPU. The desktop runs the same rules in
numpy (`dtouch/bender_post.py`: `post` for `post.frag`, `fold` for
`stack.frag`; NEAREST scaling stands in for `view.frag`), so the mode stays
headless. `tests/test_bender_shared.py` compiles each under 330 core where
a GL context exists and checks `post.frag` against `post` on the GPU.

The files stay inside the intersection of GLSL 3.30 and GLSL ES 3.00 (the
same rules as `../dither/README.md`): no `#version` or `precision` line (the
host prepends them), no integer `%`, `layout(location = 0)` on the output.

Uniforms: `post.frag` `u_src`, `u_size` (ivec2), `u_split` (working
pixels), `u_copy` (0..1), `u_seed`, `u_block` (int, 16); `stack.frag`
`u_cur`, `u_prev`, `u_alpha`, `u_dither` (0 on a float target, else 1/255),
`u_seed`; `view.frag` `u_src`, `u_size`, `u_canvas` (ivec2), `u_blackout`.

## The JSON files

Both are generated from the Python source of truth by

    python -m dtouch.bender_looks

and `tests/test_bender_shared.py` fails when either drifts from it, so
regenerate after touching `dtouch/bender.py`, `dtouch/bender_jpeg.py`,
`dtouch/bender_sensor.py`, `dtouch/bender_post.py` or the mode's card. A
rerun with nothing changed is a no-op.

- `bender_looks.json`: the card (`title`, `key`, `blurb`, `accent` as RGB),
  `effects` in E's order with `effect_titles`, `jpeg_effects`,
  `sensor_effects`, `looks` (one per effect) with `look_names` and
  `safe_look`, `ladders`, `sort_band`, `settle_s`, `cut` (the cut clock and
  `touch_cuts`), `working` (the pixel budget), `governor` (`max_step_ms` is
  the desktop's main-thread cost ceiling; the page may ignore it), `backoff`,
  `sound`, `jpeg` (encode quality 0..1 and the copy block), `bent_mix`,
  `iconic` (how many sensor looks come before the JPEG bends), `thermal`
  (THERMAL's rings, shadow, grain, edge rim, blur radius and palettes),
  `stack_weights`, and `loop` (bend rate, stall window, and the lost-worker
  respawn gap, its doubling ceiling and the losses in a row before giving up).
  `loop.max_bend_hz` is the bend rate, held by a deadline rather than a
  minimum gap: a bend starts when `now >= next_due`, and then `next_due =
  (now if now - next_due > 1 / max_bend_hz else next_due) + 1 / max_bend_hz`,
  so a loop that lands a hair early keeps the pace and a pause restarts it
  from now instead of bursting to catch up. The page's bender.js uses the
  same deadline (looks.js `bendDue`).
- `bender_goldens.json`: sha256 of every JPEG bend at three amounts and two
  seeds on each fixture (plus the byte smear), the MCU walk of each fixture,
  every sensor bend on a rule-defined pixel fixture at two sizes, the pixel
  sort, and the seeded generator's first values. The bends' arithmetic is
  written to match JavaScript's (`Math.round` is half up, `Math.imul`, Uint8
  truncation), so the page's `jpeg.js`, `sensor.js` and `sort.js` reproduce
  every hash.

Changing a uniform name, a table's shape or a bend's output here is an API
change for the browser page: say so in the PR.

Inspired by CyberShot Cam by @lixofuturista / @cebolander
(github.com/cebola4444/cybershot-cam). That repository carries no licence, so
neither its code nor its numbers are used; the bends are designed from the
JPEG standard (ITU-T T.81).
