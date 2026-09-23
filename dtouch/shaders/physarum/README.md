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
| `update.frag`     | one Jones step per agent (sense, turn, move, respawn) |
| `deposit.vert`    | attribute-less GL_POINTS, agent i by `gl_VertexID`    |
| `deposit.frag`    | blended deposit weight, ONE/ONE additive              |
| `blur.frag`       | one axis of the box blur; adds laid, applies decay (scalar `u_decay`, optionally per-pixel via the `u_keep` map) |
| `stats.frag`      | stride-4 subsample of (trail, laid) for p95 / mean    |
| `tonemap.frag`    | luminance, optionally lit as a relief (`u_depth`, `u_light`) |
| `impulse.frag`    | burst (gaussian respawn) / wave (radial headings) / gather (local rush) |
| `looks.json`      | behavior points, built-in looks, palettes + 256-LUTs  |

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
