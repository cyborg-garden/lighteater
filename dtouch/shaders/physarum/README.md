# physarum shaders — shared with the browser

These files are the single source of truth for the physarum simulation on
**both** the Python GPU engine (`dtouch/physarum_gl.py`, moderngl, GL 3.3 core)
and the WebGL2 port on cyborg-garden-site's `/physarum` page, which vendors
them **verbatim**. Every file must stay inside the intersection of GLSL 3.30
and GLSL ES 3.00:

- **no `#version` line** — the host prepends it (`#version 330 core` here,
  `#version 300 es` in the browser).
- **no `precision` line** — ES needs one and desktop GL rejects it, so the
  browser prepends `precision highp float; precision highp int;
  precision highp sampler2D;` (the sampler precision matters: ES defaults
  fragment samplers to lowp and the agent texture is float positions).
- `texelFetch` on `sampler2D` only; no `usampler`/`isampler`, no `double`, no
  geometry/compute shaders, no image load/store, no bindless anything.
- `gl_VertexID` is used (ES 3.00 has it); `gl_PointSize` is written
  explicitly so points are 1 px in both hosts.
- fragment outputs carry `layout(location = 0)`.
- textures are `RGBA32F` / `RGBA16F`-class float textures or plain
  `sampler2D` inputs; the shaders only ever read `.r` of the trail, laid,
  matte and gray textures and write `.r` (in a `vec4`), so the host may back
  them with `R32F` (moderngl does) or `RGBA16F/32F` (WebGL2's color-renderable
  float formats) without touching the GLSL.
- loop bounds may be uniforms (ES 3.00 allows it); `uint` arithmetic wraps.

| file              | pass                                                  |
|-------------------|-------------------------------------------------------|
| `fullscreen.vert` | full-screen triangle for every fragment pass          |
| `update.frag`     | one Jones step per agent (sense, turn, move, respawn); fractal levels when `u_fractal` > 0 |
| `deposit.vert`    | attribute-less GL_POINTS, agent i by `gl_VertexID`    |
| `deposit.frag`    | blended deposit weight, ONE/ONE additive              |
| `blur.frag`       | one axis of the box blur; adds laid, applies decay (scalar `u_decay`, optionally per-pixel via the `u_keep` map) |
| `stats.frag`      | stride-4 subsample of (trail, laid) for p95 / mean; the finer levels in .b/.a when `u_fractal` > 0 |
| `tonemap.frag`    | luminance, optionally lit as a relief (`u_depth`, `u_light`) |
| `impulse.frag`    | burst (gaussian respawn) / wave (radial headings) / gather (local rush) |
| `tonemap_fractal.frag` | the fractal amount's tonemap: per-level FIELDS at grid size, two targets (fields, thread centreline) |
| `compose.frag`    | the fractal amount's picture: the fields' outlines cut at OUTPUT size |
| `looks.json`      | behavior points, built-in looks, palettes + 256-LUTs, depth and fractal tables |

`looks.json` is generated from the Python source of truth by
`python -m dtouch.physarum_looks`; `tests/test_physarum_gl.py` fails when
the two drift, so regenerate it whenever `POINTS`, `BUILTIN`, `DEFAULTS` or the
palette stops change. `tests/test_physarum_gl.py` also lints these files for
the ES-3.00 rules above and compiles each one under `#version 330 core`.

Changing a uniform name, a texture layout, or the agent texture format here
is an API change for the browser page — say so in the PR.

**2026-08 uniform additions** (the browser port must set these):

- `update.frag`: `u_satcap` (sensed-trail soft cap, absolute units; host
  feeds `sat x` last frame's p95, `<= 0` off), `u_jitter` (rad, 0 off),
  `u_hetero` (0-1 sub-population sense split, 0 off).
- `blur.frag`: decay moved out of `u_scale` into a new `u_decay`
  (H pass: `u_scale = 1/k`, `u_decay = 1`; V pass: `u_scale = 1/k`,
  `u_decay = decay`); new `u_keep` sampler + `u_use_keep` flag raise decay
  toward 0.995 per pixel where the keep map is 1 (react's linger).
- `impulse.frag`: `u_mode == 3` (gather) teleports only agents within
  `u_radius` of `u_center` (torus distance) into a 0.15 x radius gaussian.

**2026-09 uniform additions** (depth; a host that sets neither stays flat,
bit for bit):

- `tonemap.frag`: `u_depth` (0 = flat, the original two-tap path; up to 1
  mixes in the lit relief) and `u_light` (vec3 key light, unit, in GRID-TEXEL
  space: +x column, +y row index, +z out of the picture). The host normalises
  `u_light` and keeps its z at or above 0.05 (the shading divides by it).
- The orbit period, the bass rake and the per-look depths are data in
  `looks.json` (the top-level `depth` block and each look's `depth` key), so
  the browser vendors them instead of retyping them. The orbit azimuth there is
  in SCREEN space (+y up); a host whose row 0 is the top of the screen passes
  `-y` into `u_light`. The desktop does (its readback row 0 is the top of the
  window, `test_trail_rows_and_columns_are_grid_coordinates`).

**2026-09 fractal veins** (the browser demo moved here; a host that sets
none of this runs the stock engine, bit for bit —
`tests/test_physarum_fractal.py` holds each file against a frozen
pre-fractal copy for all five looks, and against the browser's demo copies
at amounts above 0):

- `update.frag`, `deposit.vert`, `blur.frag`, `stats.frag` gain `u_fractal`
  (0..1; 0 = stock) and read the rest only when it is above 0:
  - update: `u_lvlScale`, `u_inv_norm`, `u_flank`, `u_flankAt`, `u_shun`,
    `u_dieback`, `u_lvlFood`, `u_roomCull`, `u_fineMin`, `u_fineMax`,
    `u_fineAngle`, `u_trunkAngle`, `u_calm`, `u_strideK`, `u_bloom[2]`,
    `u_bloomFine`, `u_bloomPull`, `u_bodyFine`;
  - deposit: `u_lvlDep`, `u_fineField`, `u_bloom[2]`, `u_bloomDep`;
  - blur: `u_diffC`, `u_sharpC`, `u_decayC` (H pass `u_decayC` = 1);
  - stats: none (writes the veins/threads channels into .b/.a, so the
    stats target must be RGBA, not RG; read back all four).
- `tonemap_fractal.frag` is a separate file, not a branch in `tonemap.frag`:
  it writes four fields plus a second target (`layout(location = 1) f_line`)
  instead of one luminance and reads a 5x5 neighbourhood, so a uniform gate
  would put an MRT output and ~150 never-run lines in front of the flat
  two-tap path while saving the host nothing (it switches targets anyway).
  Uniforms: the stock tonemap's plus `u_inv_nl`, `u_lvlRelief`, `u_lvlExp`,
  `u_grainK`, `u_crest`, `u_lineBeta`, `u_lineGate`.
- `compose.frag` (new) samples `u_fields` LINEAR and `u_line` by texel and
  writes the luminance at output size. Whenever `u_mix` < 1 (every amount
  under `mix_full`) it also reads `u_stock` by texel: the stock picture,
  i.e. `tonemap.frag`'s output at grid size with the relief as set, so a
  small amount morphs out of exactly today's picture. (The browser demo
  rebuilt that side from `u_trail`/`u_laid` as the FLAT sum, and with the
  relief on the first nudge dropped the lighting; this file differs from
  the demo there and only there.) Uniforms: `u_outSize`, `u_grid`, `u_mix`,
  `u_lvlVal`, `u_thr`, `u_body`, `u_hair`, `u_glow`, `u_bloom[2]`,
  `u_bloomThr`, `u_shimmer`.
- Pass order with the amount above 0: update -> deposit -> blur H -> blur V
  -> stats -> (`tonemap.frag` into the stock target, only while the amount
  is under `mix_full`) -> `tonemap_fractal.frag` (grid size, into two
  RGBA16F targets: fields, line) -> `compose.frag` (output size, capped at
  `fractal.table.renderPx` pixels, never under the grid) -> colorize.
- Numbers: `looks.json` `fractal.table` is the tuning table (the browser's
  own key names), `fractal.mix_full` (compose `u_mix` = min(1, amount /
  mix_full)), `fractal.level_norms` (the per-level bright-end rule),
  `fractal.bloom` (the attention zones' schedule), `fractal.px_ref` (the
  length unit), and each look's `fractal` amount (`defaults.fractal` = 0).
  `level_norms.ema` is the per-level smoothing (the stock norm's EMA).
  Python source: `dtouch.physarum` FRACTAL / FRACTAL_MIX_FULL / FRACTAL_NL /
  BLOOM / bloom_zones, `dtouch.modes.physarum` LOOK_FRACTAL.
- The agent pool is allocated at `density` x n and update/deposit run only
  the first n x (1 + (density - 1) x amount) agents (the update pass on the
  rows that hold them). Fill BOTH ping-pong agent textures with the spawned
  pool: a parked row keeps whatever its texture held, and one left empty
  wakes at the origin as level 0 when the amount rises.
- Desktop only: the quality tiers cap the pool (`PhysarumMode.
  FRACTAL_DENSITY_TIER`; `quality` runs 1.0x to hold 60 fps on the GPU,
  measured to leave the fine structure unchanged). A browser that sizes its
  own grid should budget the same way: update + deposit are linear in agents
  and are most of the fractal's cost.
