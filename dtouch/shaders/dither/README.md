# dither shaders: shared with the browser

The second browser-shared unit (the first is `../physarum/`). `dither.frag`
is the single source of truth for ordered dithering (Bayer 4x4 and 64x64
blue noise) on **both** the Python GPU rack (`dtouch/rack_gl.py`, moderngl,
GL 3.3 core, loaded through `load_shared_shader("dither", "dither.frag")`)
and the WebGL2 Dither mode on cyborg-garden-site's `/games/lighteater` page,
which vendors this directory **verbatim**. The behavioral contract is
`dtouch.dither._ordered_indices` (plus `_encode_levels` through the
host-uploaded LUT); `tests/test_rack_gl.py` proves the shader bit-exact
against numpy, and `tests/test_dither_shared.py` renders all 48 core
goldens through it with an identity `u_enc_lut`, the browser's recipe.

The file must stay inside the intersection of GLSL 3.30 and GLSL ES 3.00:

- **no `#version` line**: the host prepends it (`#version 330 core` here,
  `#version 300 es` in the browser).
- **no `precision` line**: the browser prepends
  `precision highp float; precision highp int; precision highp sampler2D;`.
- `texelFetch` on `sampler2D` only; no `usampler`/`isampler`, no `double`.
- no integer `%` (undefined in ES 3.00 for negative operands): the matrix
  is tiled as `xy - u_mat_size * (xy / u_mat_size)`, exact for the
  non-negative ints here.
- the fragment output carries `layout(location = 0)`.

| file                  | what                                                  |
|-----------------------|-------------------------------------------------------|
| `dither.frag`         | ordered dither: LUT linearize, threshold, level index, LUT encode |
| `dither_looks.json`   | tables the page bundles (see below)                   |
| `dither_goldens.json` | golden level indices the site's tests and verifier check |

Uniforms: `u_src` (working image, `.rgb`), `u_mat` (R32F threshold matrix in
[0, 1)), `u_lin_lut` (256x1 R32F, `luts.lin[levels]`: sRGB-to-linear of
k/255, times levels), `u_enc_lut` (256x1 R32F, level index to output value),
`u_mat_size` (ivec2), `u_levels` (float, `2^bits - 1`), `u_invert` (0/1, the
resolved bias), `u_gamma` (0/1).

`u_enc_lut` is host-chosen; the browser uploads identity to recover level indices.

## The JSON files

Both are generated from the Python source of truth by

    python -m dtouch.dither_looks

and `tests/test_dither_shared.py` fails when either drifts from it, so
regenerate after touching the Dither mode's `PALETTES`, `AUTHORED_INVERSE`,
`BUILTIN`, `DEFAULTS`, constants, the menu cards (including `AUTO_ID` /
`AUTO_ACCENT`), the `dtouch/auto.py` cadence constants, the blue-noise
asset, or anything in `dtouch/dither.py`. A rerun with nothing changed is a no-op.

- `dither_looks.json`: the mode's `title` / `key` / `blurb` / `accent`
  (RGB), `algos`, `ordered`, `web_algos` (the ordered algorithms the browser
  may claim: ASCII is excluded, it renders through `dtouch.ascii_art`),
  `biases`, `bias_invert`, `palette_names`, `palettes`, `authored_inverse`,
  `legacy_palettes`, `one_bit_pairs` (the stops a 1-bit dither renders,
  `low_depth_stops(palette_stops(name, inv), 2)`), `builtin`, `web_builtin`
  (built-ins whose algorithm is in `web_algos`), `defaults`, `constants`,
  `matrices` (`bayer4` = `_bayer_matrix(4) * 16`, `bluenoise64` = the asset
  times 4096, both exact integers; divide on upload), `luts` (`lin` per
  levels, `enc` per `"levels,gamma"`), `modes` (the home-menu card of
  each card the page shows, accent as RGB: `physarum`, `dithergirl`, and
  `auto`, the AUTO card from `dtouch/menu.py`), and `autopilot`
  (`dtouch/auto.py`'s `DWELL`, `MODE_EVERY` with `mode_every_high:
  "exclusive"` because it feeds numpy's `Generator.integers`,
  `CAST_CHANCE`, `FIRST_DWELL`, `DT_MAX` and `CASTS`).
- `dither_goldens.json`: three 64x64 fixtures given as rules (`gradient`
  holds all 256 codes), the plane rule (`code / 255` in float32), and:
  `core` (48 cases: Bayer and Blue noise x bits 1-4 x gamma on/off x bias
  light/dark on `gradient`, plus bias auto with gamma on over `dark` and
  `light`), each the base64 uint8 level indices (row-major, row 0 = y 0) and
  the resolved `invert`; `fs` (6 Floyd-Steinberg index sets, the oracle for
  a later port); `rgb` (12 `_palette_map` rows).

Changing a uniform name, a texture layout, or a table's shape here is an API
change for the browser page: say so in the PR.
