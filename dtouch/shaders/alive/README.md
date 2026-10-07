# alive shaders: shared with the browser

The "alive" fractal step: what H's third step (depth + fractal veins) carries
on both hosts, on top of the physarum unit's fractal veins, and the molten
calligraphy that draws it. The single source of truth for the Python GPU
engine (`dtouch/physarum_alive.py`, moderngl, GL 3.3 core) and for the WebGL2
page on cyborg-garden-site (`src/scripts/physarum/alive.js`), which vendors
this directory **verbatim** with `scripts/sync-lighteater.mjs` (unit `alive`).

The rules are the physarum unit's (see `../physarum/README.md`): no
`#version` and no `precision` lines (each host prepends its own), the GLSL
3.30 and ES 3.00 intersection only, no integer `%` on a value that could be
negative, `layout(location = 0)` on every fragment output.
`tests/test_alive_shared.py` lints every file, compiles every program under
`#version 330 core`, and holds the rest of this page.

| file | pass |
|------|------|
| `quad.vert` | the alive passes' full-screen triangle, attribute-less (`gl_VertexID`); `v_uv` 0..1 |
| `video.glsl` | camera sampling (`u_video`, `u_vidScale`, `u_vidOff`, `u_mirror`), prepended to `ink.frag` and `carve.frag` |
| `scene.frag` | scene grid (320 wide, the sim's aspect): fresh motion, the matte distance, the depth the veins wrap around, the motion's skirt |
| `update.frag` | the physarum unit's `update.frag` plus marked `alive+` / `alive-` blocks: hunting the fresh motion, contour steering |
| `life.frag` | grid: trail age and the deposit memory pruning reads |
| `blur.frag` | the physarum unit's `blur.frag` plus marked blocks: pruning |
| `events.frag` | blocks of `cell` px: connection events (a thread closing onto a vein) |
| `glow.frag` | grid: the light that runs along a thread when it connects |
| `pulse.frag` | quarter grid: distance along the veins from food, for the shuttle pulse |
| `compose.frag` | the physarum unit's `compose.frag` plus marked blocks: fronts, the pulse, the glow; writes RGBA (r whole, g veins, b accents) |
| `heads.vert`, `heads.frag` | exploring tips: GL_LINES tails and GL_POINTS heads, additive into r and b |
| `ink.frag` | the display pass: molten calligraphy (three stroke tiers, broad nib, chrome) over the palette, or over the camera |
| `carve.frag` | the browser's display over the camera with the ink off; the desktop does not run it yet |
| `alive.json` | the tuning table, `dtouch.alive.ALIVE` key for key, by `python -m dtouch.alive_looks` |

**The copies.** `update.frag`, `blur.frag` and `compose.frag` here are the
physarum unit's files byte for byte once their marked blocks are dropped,
and every addition sits behind `u_alive` (0 runs the stock step). Change the
physarum file and the copy together; the test fails when they drift.

**Pass order** with the step on: scene, update, deposit (stock), life, blur
H and V, events, glow, pulse x2, stats (stock), fractal tonemap (stock),
compose into the alive picture, heads, then the display (`ink.frag`). Flat
and depth-only never run any of it.

**Space.** `ink.frag` and `carve.frag` draw in canvas space and flip the
image-space textures once (`uv.y = 1 - v_uv.y`); a host that reads its
target back with row 0 at the top flips the readback (the desktop does).

**The landing** (`alive.json` `landing`): boot, a first entry and panic (0)
land on the safe look's fractal step in the ink, over this palette, with
the camera's picture hidden. Every other look still lands on depth with the
fractal off, as the physarum unit's `looks.json` says. A host that cannot
build or allocate the passes keeps that shared landing.

**Style options** (`ink.frag` `u_paper`, `u_fold`, `u_foldSide`,
`u_foldAngle`; `alive.json` `fold`): paper (black ink on white, one accent
colour) and a mirror fold of 2, 4 or 6. Both are off by default, and off is
the ink above pixel for pixel (tests/test_ink_style.py holds it against a
frozen copy). Keys K (paper) and Y (fold: off, 2, 4, 6) on both hosts; a
look carries them as `ink_paper` / `ink_fold` (looks.json `inkblot`, the
one look that lands on the fractal step, with the video background on), and
any other look or panic turns them off; the landing never sets them. The
fold's source half, quadrant or wedge follows the performer (the matte's
lit centroid) through `dtouch.alive.fold_side` / alive-core.js `foldSide`:
a hysteresis past the centre lines and a 2 s `hold`, so a side change is one
cut, never a flicker.
