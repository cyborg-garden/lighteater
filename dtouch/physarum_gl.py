"""Physarum on the GPU — the same Jones-model mold as dtouch.physarum, run as
fragment-shader ping-pong GPGPU on moderngl.

macOS caps OpenGL at 4.1, so there are no compute shaders here and none are
needed: this is the standard WebGL physarum architecture, and it runs on GL
3.3 core.

- **Agents live in a float texture.** RGBA32F, one texel per agent
  (x, y, heading, spare). The *update* pass draws one full-screen triangle
  over the agent texture and, per texel, does exactly what PhysarumField
  does per row: read the matte under the agent (the pen), blend the two
  behavior points, sense the trail+food at three points, turn, step, wrap,
  and — for a trickle of agents — respawn onto the lit subject. It writes the
  next agent texture; the two agent textures ping-pong.
- **Deposit is a point draw.** An attribute-less GL_POINTS draw of N vertices
  whose vertex shader texelFetches its own agent by gl_VertexID and lands a
  1px point on that agent's cell, additively blended into a float "laid"
  texture. That is bincount on the GPU.
- **Diffuse + decay** is a separable box blur (H then V) that adds the laid
  deposits on the way in and multiplies by decay on the way out, ping-ponging
  the trail.
- **The picture is tonemapped on the GPU** and read back as uint8 — the mode
  quantizes luminance to a 256-entry LUT index anyway, so an 8-bit readback
  loses nothing and is 4x smaller than the float trail. The two statistics
  the tonemap needs (the trail's 95th percentile, the deposits' mean) come
  from a 1/4-res subsample pass read back as a small float texture.

The GLSL lives in `dtouch/shaders/physarum/` as standalone files, without a
`#version` line, in the GLSL 3.30 ∩ GLSL ES 3.00 subset: they are the single
source of truth for this engine AND for the WebGL2 port on the public site,
which vendors them verbatim (see that directory's README). `load_shader`
prepends the desktop version line.

Same interface as PhysarumField — `update(matte, gray)`, `luminance()`,
`spawn_burst`, `wave`, the matte-blended POINTS — so PhysarumMode can select
either. Known, deliberate differences from the CPU field: the blur wraps at
the edges (the agents already do; cv2.boxFilter reflects); deposit weight is
read at the agent's post-move cell rather than its pre-move one; the
percentile is estimated on a stride-4 subsample; reseeding is per-agent
Bernoulli(reseed_frac) with rejection sampling against matte*luma instead of
a CDF draw. None of these is visible on the instrument.

**Fractal veins** (`fractal` > 0, GL only): the three trail channels become
three scales of one organism (trunks, veins, threads) in the same update /
deposit / blur / stats passes, each gated on `u_fractal` so amount 0 runs
the stock arithmetic bit for bit. The picture then takes two different
passes: tonemap_fractal.frag writes per-level fields at grid size (two
targets), and compose.frag cuts their outlines at OUTPUT size (`out_size`,
capped at FRACTAL["renderPx"] pixels), so a thread stays a hairline however
far the grid is stretched. `luminance()` then returns that output-size
picture. The pool is allocated at fractal_density x n (FRACTAL["density"]
unless the host caps it); the amount decides how much of it runs.

Clean-room note: the model is Jeff Jones's (sense/rotate/move/deposit/
diffuse/decay) and the semantics are this repo's dtouch/physarum.py. No code
or parameter tables from any CC BY-NC-SA physarum project were used.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .physarum import (BALLISTIC_DECAY, FRACTAL, FRACTAL_MIX_FULL, FRACTAL_NL,
                       NORM_EMA, POINTS, PX_REF, RELIEF_LIGHT, SPATIAL_REGIMES,
                       ZONE_REGIME_COUNT, relief_light, species_matrix)

# Agent texture width. One texel per agent; the height is ceil(n / width).
# 2048 x 16384 (the GL_MAX_TEXTURE_SIZE floor on anything that runs this)
# is 33M agents, well past what the deposit draw can afford anyway.
AGENT_TEX_W = 2048

# Stride of the statistics subsample (percentile + mean readback).
STATS_STRIDE = 4


SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")
SHADER_DIR = os.path.join(SHADERS_ROOT, "physarum")
SHADER_FILES = ("fullscreen.vert", "update.frag", "deposit.vert", "deposit.frag",
                "blur.frag", "stats.frag", "tonemap.frag", "impulse.frag",
                "tonemap_fractal.frag", "compose.frag")

# What moderngl gets in front of every file. The browser prepends its own
# `#version 300 es` + precision lines; the files carry neither.
GLSL_VERSION_LINE = "#version 330 core\n"


def load_shared_shader(unit, name, version_line=GLSL_VERSION_LINE):
    """Source of the browser-shared `dtouch/shaders/<unit>/<name>` with the
    version line prepended. A `#line 1` follows it so compile errors keep
    file-true line numbers. The one loader for every browser-shared unit
    (physarum here, dither via dtouch.rack_gl)."""
    with open(os.path.join(SHADERS_ROOT, unit, name), "r", encoding="utf-8") as fh:
        body = fh.read()
    return version_line + "#line 1\n" + body


def load_shader(name, version_line=GLSL_VERSION_LINE):
    """Source of `dtouch/shaders/physarum/<name>` (see load_shared_shader)."""
    return load_shared_shader("physarum", name, version_line)


NO_BLOOM = ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0))


def fractal_render_size(gw, gh, out_w, out_h, budget=FRACTAL["renderPx"]):
    """Size of the fractal drawing pass: the output, scaled down to the pixel
    budget, never below the grid. Rounds half up, like the browser's
    Math.round."""
    k = min(1.0, math.sqrt(budget / max(1, out_w * out_h)))
    return (max(gw, int(math.floor(out_w * k + 0.5))),
            max(gh, int(math.floor(out_h * k + 0.5))))


def level_norms(stats, norm, prev=None, ema=FRACTAL_NL["ema"]):
    """Per-level bright ends (trunks, veins, threads) from the fractal stats
    subsample (stats.frag with u_fractal > 0: .r total, .b veins, .a
    threads). The trunks take the stock rule (a high percentile over every
    sample), so they burn as bright as today's veins; a finer level is sparse,
    so a plain percentile lands on empty ground: it takes a percentile over
    the samples it actually occupies. Floored against the stock `norm`, and
    smoothed against `prev` like the stock norm is (FRACTAL_NL)."""
    s = np.asarray(stats, np.float32).reshape(-1, 4)
    tot, g, b = s[:, 0], s[:, 2], s[:, 3]
    lv = (np.maximum(tot - g - b, 0.0), g, b)
    floor = max(1e-4, (norm if norm > 0 else 1.0) * FRACTAL_NL["floor"])
    out = []
    for k, a in enumerate(lv):
        # Order statistics by np.partition, not a full sort: the same element
        # of the same array (a sort-then-index, bit for bit), at a fraction
        # of the cost; the sort was 1.4 / 3.4 / 6.4 ms of CPU per frame at
        # the perform / balance / quality stats sizes, inside the frame's
        # GPU wait.
        mx = float(a.max())
        if k > 0:
            # the occupied samples: strictly above occ x the max, compared in
            # double like the searchsorted it replaces
            a = a[a > np.float64(FRACTAL_NL["occ"] * mx)]
        occ = a.size
        pct = FRACTAL_NL["pct"][0 if k == 0 else 1]
        if occ > FRACTAL_NL["min_occ"]:
            i = int(math.floor(occ * pct))
            v = float(np.partition(a, i)[i])
        else:
            v = mx
        want = max(floor, v if math.isfinite(v) else floor)
        out.append(want if prev is None else prev[k] * ema + want * (1.0 - ema))
    return tuple(out)


class PhysarumGLUnavailable(RuntimeError):
    """Raised by PhysarumFieldGL.__init__ when no GL context / float render
    target can be had. The mode catches it and falls back to the CPU field."""


class PhysarumFieldGL:
    """PhysarumField's contract on a moderngl standalone context.

    Raises PhysarumGLUnavailable when the context or the float render
    targets cannot be created; releases whatever it built first.

    One live field at a time (the same rule as GlowRenderer): moderngl
    issues GL calls on whichever standalone context was created last, so a
    second live field silently aliases the first one's textures and
    programs. Release a field before building another.

    Every public GL entry point binds this field's context first (`with
    self.ctx:` — moderngl makes it current and restores the previous one
    on exit). Without the bind, ANY other moderngl context created later
    in the process (GlowRenderer, a matte backend, a window) silently
    receives this field's GL calls: measured, a foreign context drawing
    between steps turned the luminance into saturated garbage (mean abs
    error 0.49 on [0,1] vs an identical solo run). The `with` exits back
    to the other renderer's context, so both sides stay correct."""

    def __init__(self, n=1_000_000, gw=1280, gh=736, seed=0,
                 point_bg="veins", point_fg="fingers",
                 decay=0.94, diffuse=1, food=0.35, exposure=3.5,
                 grain=0.2, reseed_frac=0.004, gain=1.0,
                 sat=0.0, jitter=0.0, hetero=0.0, species=3, cross=0.0,
                 fractal_density=None):
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if gw <= 0 or gh <= 0:
            raise ValueError(f"grid must be positive, got {gw}x{gh}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if not (0.0 <= reseed_frac <= 1.0):
            raise ValueError(f"reseed_frac must be in [0, 1], got {reseed_frac}")
        self.n, self.gw, self.gh = n, gw, gh
        self.point_bg = point_bg
        self.point_fg = point_fg
        self.decay = decay
        self.diffuse = diffuse
        self.food = food
        self.exposure = exposure
        self.grain = grain
        # depth: 0 = the flat picture (the tonemap's original two-tap path,
        # bit for bit); up to 1 lights it as a relief under `light`, a
        # grid-texel-space vector (see tonemap.frag). The mode drives both.
        self.depth = 0.0
        self.light = RELIEF_LIGHT
        self.reseed_frac = reseed_frac
        self.gain = gain
        self.sat = sat            # sensor saturation (x trail bright end); 0 = off
        self.jitter = jitter      # per-step heading wobble, rad
        self.hetero = hetero      # 0..1 sub-population sense split
        self.species = max(1, min(3, int(species)))   # populations, 1..3
        self.cross = float(cross)  # off-diagonal strength of the species matrix
        self.sharpen = 0.0         # lateral inhibition in the diffusion pass
        self.ballistic = 0.0       # steering suppression, decays after a wave
        self.mosaic = 0.0          # 0..1 spatial parameter mosaic strength
        self.zones = 7.0           # zone lattice density across the grid
        # slow structural modulation multipliers (see PhysarumField)
        self.mod_sense = 1.0
        self.mod_turn = 1.0
        self.mod_spread = 1.0
        self.mod_step = 1.0
        self.mod_deposit = 1.0
        # fractal veins: the amount (0 = the stock engine, bit for bit),
        # the output size the compose pass draws at (None = the grid), the
        # length unit (grid px per PX_REF px; the mode sets its own), the
        # two attention zones (dtouch.physarum.bloom_zones; the mode drives
        # them) and the shimmer clock
        self.fractal = 0.0
        self.out_size = None
        self.px_scale = gw / float(PX_REF)
        self.bloom = NO_BLOOM
        self.bloom_t = 0.0
        self.fractal_error = None  # set when the fractal passes cannot build
        self._nl = None            # per-level bright ends (level_norms)
        self._stats_pbo = None     # fractal stats, read a frame late (_stats_late)
        self._stats_pending = False
        self.seed = seed
        self.frame = 0
        self._last_norm = 0.0     # last luminance() percentile (sat cap ref)
        self.ctx = None
        self._rng = np.random.default_rng(seed)
        self._salt = np.random.default_rng(seed + 1)
        # the pool: n agents for the stock engine, up to fractal_density x n
        # as the fractal amount rises (the thin finer levels fill the dark
        # between trunks instead of saturating it). FRACTAL["density"] unless
        # the host caps it (the mode does per quality tier, to hold the
        # frame budget: PhysarumMode.FRACTAL_DENSITY_TIER)
        self.fractal_density = float(FRACTAL["density"] if fractal_density is None
                                     else fractal_density)
        if self.fractal_density < 1.0:
            raise ValueError(f"fractal_density must be >= 1, got {fractal_density}")
        self.n_cap = max(n, int(math.ceil(n * self.fractal_density)))
        self.aw = min(self.n_cap, AGENT_TEX_W)
        self.ah = int(math.ceil(self.n_cap / self.aw))
        try:
            self._build()
        except Exception as e:                          # noqa: BLE001
            self.release()
            raise PhysarumGLUnavailable(f"GPU physarum unavailable: {e}") from e

    # ----- GL setup -----
    def _build(self):
        import moderngl
        self._gl = moderngl
        ctx = self.ctx = moderngl.create_standalone_context()
        gw, gh, aw, ah = self.gw, self.gh, self.aw, self.ah

        fs_vs = load_shader("fullscreen.vert")

        def prog(frag, vert=fs_vs):
            return ctx.program(vertex_shader=vert, fragment_shader=load_shader(frag))
        self.p_update = prog("update.frag")
        self.p_deposit = prog("deposit.frag", load_shader("deposit.vert"))
        self.p_blur = prog("blur.frag")
        self.p_stats = prog("stats.frag")
        self.p_tonemap = prog("tonemap.frag")
        self.p_impulse = prog("impulse.frag")

        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.tri_vbo = ctx.buffer(tri.tobytes())
        self.vao_update = ctx.vertex_array(self.p_update, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_blur = ctx.vertex_array(self.p_blur, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_stats = ctx.vertex_array(self.p_stats, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_tonemap = ctx.vertex_array(self.p_tonemap, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_impulse = ctx.vertex_array(self.p_impulse, [(self.tri_vbo, "2f", "in_vert")])
        # attribute-less point draw: the vertex shader is fed by gl_VertexID
        self.vao_deposit = ctx.vertex_array(self.p_deposit, [])

        def ftex(size, comps, data=None):
            t = ctx.texture(size, comps, data=data, dtype="f4")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = t.repeat_y = False
            return t

        # agents: same initial distribution as the CPU field (same seed)
        rng = self._rng
        init = np.zeros((ah * aw, 4), np.float32)
        init[:self.n, 0] = rng.uniform(0, gw, self.n)
        init[:self.n, 1] = rng.uniform(0, gh, self.n)
        init[:self.n, 2] = rng.uniform(0, 2 * np.pi, self.n)
        # .w is the species, drawn once at spawn and carried for life
        init[:self.n, 3] = rng.integers(0, self.species, self.n)
        # the fractal's extra agents come from their own stream, so the first
        # n (all the stock engine ever runs) are what they always were
        extra = self.n_cap - self.n
        if extra > 0:
            r2 = np.random.default_rng(self.seed + 2)
            init[self.n:self.n_cap, 0] = r2.uniform(0, gw, extra)
            init[self.n:self.n_cap, 1] = r2.uniform(0, gh, extra)
            init[self.n:self.n_cap, 2] = r2.uniform(0, 2 * np.pi, extra)
            init[self.n:self.n_cap, 3] = r2.integers(0, self.species, extra)
        self.tex_agents_a = ftex((aw, ah), 4, init.tobytes())
        # B starts as the same pool, not an empty texture: update() writes
        # only the running rows, so a parked row holds whatever its texture
        # last held, and the fractal amount can wake it on a frame when B is
        # the current side. Left uninitialised, B's parked rows were zeros:
        # the woken pool ran from the origin as species 0 (trunks) for life
        # (measured: 97% of the extra pool, not a third).
        self.tex_agents_b = ftex((aw, ah), 4, init.tobytes())
        self.fbo_agents_a = ctx.framebuffer(color_attachments=[self.tex_agents_a])
        self.fbo_agents_b = ctx.framebuffer(color_attachments=[self.tex_agents_b])

        # Three species channels (RGB) + an unused alpha. Each species lays
        # into its own channel and senses all three through a signed matrix,
        # so one species can be REPELLED by another's trail — the mechanism
        # behind exclusion membranes and travelling fronts, and the one thing
        # a single-channel attract-only field cannot express at any setting.
        # The browser backs the same GLSL with RGBA16F/32F.
        self.tex_trail_a = ftex((gw, gh), 4)
        self.tex_trail_b = ftex((gw, gh), 4)
        self.tex_tmp = ftex((gw, gh), 4)
        self.tex_laid = ftex((gw, gh), 4)
        self.fbo_trail_a = ctx.framebuffer(color_attachments=[self.tex_trail_a])
        self.fbo_trail_b = ctx.framebuffer(color_attachments=[self.tex_trail_b])
        self.fbo_tmp = ctx.framebuffer(color_attachments=[self.tex_tmp])
        self.fbo_laid = ctx.framebuffer(color_attachments=[self.tex_laid])
        for f in (self.fbo_trail_a, self.fbo_trail_b, self.fbo_tmp, self.fbo_laid):
            f.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_matte = ftex((gw, gh), 1)
        self.tex_gray = ftex((gw, gh), 1)
        self.tex_keep = ftex((gw, gh), 1)      # per-pixel decay boost (react)

        self.sw = int(math.ceil(gw / STATS_STRIDE))
        self.sh = int(math.ceil(gh / STATS_STRIDE))
        # RGBA: the fractal stats carry the finer levels in .b/.a; the stock
        # readback takes only .rg of it
        self.tex_stats = ftex((self.sw, self.sh), 4)
        self.fbo_stats = ctx.framebuffer(color_attachments=[self.tex_stats])
        self.tex_lum = ctx.texture((gw, gh), 1)             # uint8 readback target
        self.fbo_lum = ctx.framebuffer(color_attachments=[self.tex_lum])

        # constant uniforms + sampler units
        for p in (self.p_update, self.p_deposit, self.p_impulse):
            p["u_aw"].value = aw
        for p in (self.p_update, self.p_deposit, self.p_blur, self.p_stats,
                  self.p_impulse):
            p["u_grid"].value = (gw, gh)
        # the spatial regime table is constant for the field's life
        reg = [SPATIAL_REGIMES[k] for k in SPATIAL_REGIMES]
        self.p_update["u_zreg"].value = [
            (r["sense"], r["turn"], r["spread"], r["step"]) for r in reg]
        self.p_update["u_zcross"].value = [r["cross"] for r in reg]
        self.p_update["u_agents"].value = 0
        self.p_update["u_trail"].value = 1
        self.p_update["u_matte"].value = 2
        self.p_update["u_gray"].value = 3
        self.p_deposit["u_agents"].value = 0
        self.p_deposit["u_matte"].value = 2
        self.p_deposit["u_nspec"].value = float(self.species)
        self.p_blur["u_src"].value = 0
        self.p_blur["u_add"].value = 1
        self.p_blur["u_keep"].value = 2
        self.p_stats["u_trail"].value = 0
        self.p_stats["u_laid"].value = 1
        self.p_stats["u_stride"].value = STATS_STRIDE
        self.p_tonemap["u_trail"].value = 0
        self.p_tonemap["u_laid"].value = 1
        self.p_impulse["u_agents"].value = 0
        # the stock engine never reads these; set once so a first fractal
        # frame never runs on GL defaults
        for p in (self.p_update, self.p_deposit, self.p_blur, self.p_stats):
            p["u_fractal"].value = 0.0
        self._pf = None           # the fractal-only programs, built on first use
        self.tex_fields = self.fbo_fields = None
        self.tex_lum_hi = self.fbo_lum_hi = None
        self.lum_tex = self.tex_lum     # what luminance_into_tex last drew

        # a render pass to prove float targets + blending actually work here
        # (a context can exist and still refuse them); a failure surfaces as
        # PhysarumGLUnavailable from __init__ rather than a black show later
        self._deposit(1.0, 1.0)
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)

    # ----- fractal veins -----
    def _fractal_amount(self):
        """The amount this frame runs at, 0..1. A context that cannot build
        the fractal-only passes runs the stock engine (amount 0) instead of a
        dead picture; the reason is kept in fractal_error."""
        a = min(max(float(self.fractal), 0.0), 1.0)
        if a <= 0.0 or self.fractal_error is not None:
            return 0.0
        if self._pf is None:
            try:
                self._build_fractal()
            except Exception as e:                  # noqa: BLE001 — §6.4
                self.fractal_error = str(e)
                print("fractal veins unavailable:", e)
                return 0.0
        return a

    def _build_fractal(self):
        ctx, gl = self.ctx, self._gl
        fs_vs = load_shader("fullscreen.vert")
        pt = ctx.program(vertex_shader=fs_vs, fragment_shader=load_shader("tonemap_fractal.frag"))
        pc = ctx.program(vertex_shader=fs_vs, fragment_shader=load_shader("compose.frag"))
        self._pf = {"tonemap": pt, "compose": pc,
                    "vao_tonemap": ctx.vertex_array(pt, [(self.tri_vbo, "2f", "in_vert")]),
                    "vao_compose": ctx.vertex_array(pc, [(self.tri_vbo, "2f", "in_vert")])}
        pt["u_trail"].value, pt["u_laid"].value = 0, 1
        pc["u_fields"].value, pc["u_stock"].value, pc["u_line"].value = 0, 1, 3
        # two grid-size RGBA16F targets: the per-level fields (sampled
        # LINEAR by compose, the way distance-field text is) and the thread
        # centreline (offset, direction; sampled by texel)
        f0 = ctx.texture((self.gw, self.gh), 4, dtype="f2")
        f1 = ctx.texture((self.gw, self.gh), 4, dtype="f2")
        f0.filter = (gl.LINEAR, gl.LINEAR)
        f1.filter = (gl.NEAREST, gl.NEAREST)
        for t in (f0, f1):
            t.repeat_x = t.repeat_y = False
        self.tex_fields = (f0, f1)
        self.fbo_fields = ctx.framebuffer(color_attachments=[f0, f1])

    def render_size(self):
        """(w, h) of the picture luminance()/lum_tex carry next: the grid
        for the stock engine, the output size (within the fractal pixel
        budget) while the fractal amount is up."""
        a = min(max(float(self.fractal), 0.0), 1.0)
        if a <= 0.0 or self.fractal_error is not None:
            return (self.gw, self.gh)
        ow, oh = self.out_size or (self.gw, self.gh)
        return fractal_render_size(self.gw, self.gh, int(ow), int(oh))

    def n_active(self, amount=None):
        """Agents stepped and deposited this frame: n for the stock engine,
        up to fractal_density x n as the fractal amount rises."""
        a = self._fractal_amount() if amount is None else amount
        if a <= 0.0:
            return self.n
        k = 1.0 + (self.fractal_density - 1.0) * a
        return max(1, min(self.n_cap, int(math.floor(self.n * k + 0.5))))

    def _bloom_flat(self):
        """u_bloom as 2 vec4s. Strength already carries the amount
        (bloom_zones, on the mode's side); a field driven directly has none."""
        return [tuple(float(v) for v in z) for z in self.bloom]

    def _salt_value(self):
        return int(self._salt.integers(0, 2**32, dtype=np.uint32))

    def interaction_matrix(self):
        """This field's species matrix — see dtouch.physarum.species_matrix.

        The SPEC, not the transport: update.frag builds one row at a time from
        `u_cross`, because the zone an agent stands in rescales the
        off-diagonals per agent and a single uploaded matrix could not express
        that. Both must agree; tests/test_physarum_mosaic.py pins it.
        """
        return species_matrix(self.species, self.cross)

    def _food_norm(self):
        """Trail-unit scale for `food` and friends: the trail's own bright end
        (last frame's 95th percentile). 1.0 before the first stats pass, when
        the trail is empty and food is the only signal there is anyway."""
        return self._last_norm if self._last_norm > 0.0 else 1.0

    # ----- parameter blending -----
    def _points(self):
        return POINTS[self.point_bg], POINTS[self.point_fg]

    def swap_points(self):
        self.point_bg, self.point_fg = self.point_fg, self.point_bg

    # ----- passes -----
    def _deposit(self, dep_bg, dep_fg, fa=0.0):
        ctx, gl = self.ctx, self._gl
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.enable(gl.BLEND)
        ctx.blend_func = (gl.ONE, gl.ONE)
        self.tex_agents_a.use(0)
        self.tex_matte.use(2)
        p = self.p_deposit
        p["u_deposit"].value = (dep_bg, dep_fg)
        p["u_nspec"].value = float(self.species)
        p["u_fractal"].value = float(fa)
        if fa > 0.0:
            def m(v):
                return 1.0 + (v - 1.0) * fa
            p["u_lvlDep"].value = tuple(m(v) for v in FRACTAL["dep"])
            p["u_fineField"].value = tuple(m(v) for v in FRACTAL["fineField"])
            p["u_bloom"].value = self._bloom_flat()
            p["u_bloomDep"].value = float(FRACTAL["bloomDep"])
        self.vao_deposit.render(gl.POINTS, vertices=self.n_active(fa))
        ctx.disable(gl.BLEND)

    def _blur_decay(self, use_keep=False, fa=0.0):
        ctx, gl = self.ctx, self._gl
        r = int(self.diffuse) if self.diffuse > 0 else 0
        k = 2 * r + 1
        p = self.p_blur
        p["u_radius"].value = r
        p["u_wide"].value = max(3 * r, r + 2)
        p["u_fractal"].value = float(fa)
        if fa > 0.0:
            # per-level diffusion, inhibition and loss, eased in by the
            # amount (all ones is the stock arithmetic); the trunks remember
            # at least FRACTAL["trunkKeep"] per frame (a short memory leaves
            # a fast point's trunk stippled, and it frays at its outline)
            def m3(key):
                return [1.0 + (v - 1.0) * fa for v in FRACTAL[key]]
            p["u_diffC"].value = tuple(m3("diff"))
            p["u_sharpC"].value = tuple(m3("sharp"))
            dc = m3("decay")
            keep = min(1.0, (1.0 - FRACTAL["trunkKeep"]) / max(1e-3, 1.0 - float(self.decay)))
            dc[0] *= 1.0 + (keep - 1.0) * fa
            decay_c = tuple(dc)
        # H: trail + laid -> tmp
        self.fbo_tmp.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        p["u_use_add"].value = 1
        p["u_use_keep"].value = 0
        p["u_decay"].value = 1.0
        # Inhibition is SPLIT across the two axes, half on each. On one axis
        # only it is a directional operator, and the picture laminates along
        # it — visibly so on the narrower CPU grid, where the same integer
        # blur radius is 2.2x coarser relative to the frame. Half and half is
        # isotropic. (Full strength on BOTH axes is not: that doubles the
        # operator and collapses the image into a pixel-scale Turing dot
        # pattern. Ask me how I know.)
        p["u_sharpen"].value = max(float(self.sharpen), 0.0) * 0.5
        p["u_dir"].value = (1, 0)
        p["u_scale"].value = 1.0 / k
        if fa > 0.0:
            p["u_decayC"].value = (1.0, 1.0, 1.0)     # the H pass does not decay
        self.vao_blur.render(gl.TRIANGLES, vertices=3)
        # V: tmp -> trail_b, times decay (per-pixel when a keep map rode in)
        self.fbo_trail_b.use()
        self.tex_tmp.use(0)
        self.tex_keep.use(2)
        p["u_use_add"].value = 0
        p["u_use_keep"].value = 1 if use_keep else 0
        p["u_sharpen"].value = max(float(self.sharpen), 0.0) * 0.5
        p["u_decay"].value = float(self.decay)
        p["u_dir"].value = (0, 1)
        p["u_scale"].value = 1.0 / k
        if fa > 0.0:
            p["u_decayC"].value = decay_c
        self.vao_blur.render(gl.TRIANGLES, vertices=3)
        self.tex_trail_a, self.tex_trail_b = self.tex_trail_b, self.tex_trail_a
        self.fbo_trail_a, self.fbo_trail_b = self.fbo_trail_b, self.fbo_trail_a

    def _swap_agents(self):
        self.tex_agents_a, self.tex_agents_b = self.tex_agents_b, self.tex_agents_a
        self.fbo_agents_a, self.fbo_agents_b = self.fbo_agents_b, self.fbo_agents_a

    # ----- one simulation frame -----
    def update(self, matte, gray, keep=None):
        """matte, gray: float32 (gh, gw) in [0,1]. Advances agents one frame
        and rebuilds the trail map — three GPU passes, no readback.

        `keep` (optional, same shape, [0,1]): per-pixel decay boost — where
        keep is 1 the trail decays at 0.995 instead of `decay`, so swept
        paths linger (the mode's react machinery paints it from motion)."""
        gl = self._gl
        gh, gw = self.gh, self.gw
        if matte.shape != (gh, gw) or gray.shape != (gh, gw):
            raise ValueError(f"matte/gray must be {(gh, gw)}, got "
                             f"{matte.shape} / {gray.shape}")
        if keep is not None and keep.shape != (gh, gw):
            raise ValueError(f"keep must be {(gh, gw)}, got {keep.shape}")
        matte = np.ascontiguousarray(matte, dtype=np.float32)
        gray = np.ascontiguousarray(gray, dtype=np.float32)
        with self.ctx:
            self.tex_matte.write(matte)
            self.tex_gray.write(gray)
            if keep is not None:
                self.tex_keep.write(np.ascontiguousarray(keep, np.float32))

            a, b = self._points()
            p = self.p_update
            ms, mt, msp = self.mod_sense, self.mod_turn, self.mod_spread
            mst = self.mod_step
            p["u_sense"].value = (a["sense"] * ms, b["sense"] * ms)
            p["u_spread"].value = (a["spread"] * msp, b["spread"] * msp)
            p["u_turn"].value = (a["turn"] * mt, b["turn"] * mt)
            p["u_step"].value = (a["step"] * mst, b["step"] * mst)
            p["u_gain"].value = float(self.gain)
            # food is in TRAIL units, not [0,1] units. The sensors read raw
            # trail (mean ~33, p95 ~120 at equilibrium), so an unscaled
            # `food * gray` with gray in [0,1] is a ~1% perturbation and the
            # camera stops steering the mold the moment any trail exists.
            # Scale it by the trail's own bright end, exactly as `sat` is
            # below, so food=0.35 means "light is worth 35% of the bright end".
            p["u_food"].value = max(float(self.food), 0.0) * self._food_norm()
            p["u_reseed"].value = float(self.reseed_frac)
            # rejection-sampling bound for the respawn weight matte*clip(gray):
            # an upper bound keeps the draw exact; <= 0 means "nothing lit" and
            # the CPU field's uniform fallback
            mmax = float(matte.max())
            wmax = mmax * min(1.0, max(float(gray.max()), 0.05)) if mmax > 0 else 0.0
            p["u_wmax"].value = wmax
            p["u_salt"].value = self._salt_value()
            # sat's cap is in absolute trail units: sat x the trail's own
            # bright end (last frame's 95th percentile — one frame stale,
            # invisible on a value that moves slowly at equilibrium)
            p["u_satcap"].value = float(self.sat) * self._last_norm
            p["u_jitter"].value = max(float(self.jitter), 0.0)
            p["u_hetero"].value = min(max(float(self.hetero), 0.0), 1.0)
            p["u_nspec"].value = float(self.species)
            p["u_cross"].value = max(float(self.cross), 0.0)
            p["u_mosaic"].value = min(max(float(self.mosaic), 0.0), 1.0)
            p["u_zones"].value = max(float(self.zones), 1.0)
            p["u_sense_max"].value = 0.10 * float(min(self.gw, self.gh))
            p["u_ballistic"].value = min(max(float(self.ballistic), 0.0), 1.0)
            # ~0.55 s at 60 fps, then steering is fully back
            self.ballistic *= BALLISTIC_DECAY
            p["u_time"].value = self.frame * (1.0 / 60.0)
            fa = self._fractal_amount()
            p["u_fractal"].value = fa
            if fa > 0.0:
                self._update_fractal_uniforms(p, fa)
            else:
                self._nl = None           # measured afresh when it returns

            # only the rows that hold running agents (the rest of the pool
            # waits for a higher fractal amount); the stock engine's rows are
            # exactly the texture it always had
            rows = int(math.ceil(self.n_active(fa) / self.aw))
            self.fbo_agents_b.viewport = (0, 0, self.aw, rows)
            self.fbo_agents_b.use()
            self.tex_agents_a.use(0)
            self.tex_trail_a.use(1)
            self.tex_matte.use(2)
            self.tex_gray.use(3)
            self.vao_update.render(gl.TRIANGLES, vertices=3)
            self._swap_agents()

            md = self.mod_deposit
            self._deposit(a["deposit"] * md, b["deposit"] * md, fa)
            self._blur_decay(use_keep=keep is not None, fa=fa)
        self.frame += 1

    def _update_fractal_uniforms(self, p, fa):
        F = FRACTAL
        norm = max(self._food_norm(), 1e-4)
        px = float(self.px_scale)
        p["u_lvlScale"].value = (F["scale"], F["angle"], 1.0)
        p["u_inv_norm"].value = 1.0 / norm
        p["u_flank"].value = F["flank"]
        p["u_flankAt"].value = F["flankAt"] * norm
        p["u_shun"].value = F["shun"] * fa
        p["u_dieback"].value = F["dieback"] * fa
        p["u_bodyFine"].value = 1.0 + (F["bodyFine"] - 1.0) * fa
        p["u_roomCull"].value = F["roomCull"]
        p["u_fineMin"].value = (F["fineMin"][0] * px * fa, F["fineMin"][1] * px * fa)
        p["u_fineMax"].value = (F["fineMax"][0] * px, F["fineMax"][1] * px)
        p["u_strideK"].value = tuple(F["stride"])
        p["u_calm"].value = F["calm"]
        p["u_fineAngle"].value = tuple(F["fineAngle"])
        p["u_trunkAngle"].value = tuple(F["trunkAngle"])
        p["u_bloom"].value = self._bloom_flat()
        p["u_bloomFine"].value = F["bloomFine"]
        p["u_bloomPull"].value = F["bloomPull"] * fa
        p["u_lvlFood"].value = tuple(1.0 + (v - 1.0) * fa for v in F["food"])

    # ----- interactions -----
    def _impulse(self, mode, x, y, frac=0.0, radius=0.0):
        gl = self._gl
        with self.ctx:
            p = self.p_impulse
            p["u_mode"].value = mode
            p["u_center"].value = (float(x), float(y))
            p["u_frac"].value = float(frac)
            p["u_radius"].value = float(radius)
            p["u_salt"].value = self._salt_value()
            # the whole pool (update() narrows the viewport to the running
            # rows; an impulse moves every agent, parked ones included)
            self.fbo_agents_b.viewport = (0, 0, self.aw, self.ah)
            self.fbo_agents_b.use()
            self.tex_agents_a.use(0)
            self.vao_impulse.render(gl.TRIANGLES, vertices=3)
            self._swap_agents()

    def spawn_burst(self, x, y, frac=0.08, radius=6.0):
        """Teleport a fraction of the pool into a tight gaussian at (x, y)
        with fresh random headings — 'pour more mold HERE'."""
        if frac <= 0:
            return
        self._impulse(1, x, y, frac, radius)

    def wave(self, x, y):
        """Point every agent's heading away from (x, y), and hold it there.

        The impulse alone is one frame of new headings, and steering takes
        most of them back before the front has gone anywhere — visible for
        about a frame, which is not a gesture. Arming the ballistic phase
        suppresses steering while it decays, so the ring actually travels.
        """
        self._impulse(2, x, y)
        self.ballistic = 1.0

    def gather(self, x, y, frac=0.5, radius=60.0):
        """Rush agents already within `radius` of (x, y) into a tight knot
        there — a LOCAL impulse that leaves the rest of the organism alone
        (the react 'spell' move; burst teleports from the whole pool)."""
        if frac <= 0 or radius <= 0:
            return
        self._impulse(3, x, y, frac, radius)

    # ----- picture -----
    def _stats(self, fa=0.0):
        """(95th percentile of the trail, mean of this frame's deposits),
        estimated on a stride-STATS_STRIDE subsample read back as floats.
        With the fractal amount up the readback also carries the finer
        levels (.b, .a) for level_norms, kept in self._last_stats4, and is
        the previous frame's (_stats_late)."""
        gl = self._gl
        self.p_stats["u_fractal"].value = float(fa)
        self.fbo_stats.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_stats.render(gl.TRIANGLES, vertices=3)
        comps = 4 if fa > 0.0 else 2
        if fa > 0.0:
            raw = self._stats_late()
        else:
            self._stats_pending = False
            raw = self.fbo_stats.read(components=comps, dtype="f4")
        s4 = np.frombuffer(raw, np.float32).reshape(self.sh, self.sw, comps)
        self._last_stats4 = s4 if fa > 0.0 else None
        s = s4[..., :2]
        self._last_stats = s        # free spatial subsample (lum_sample)
        return float(np.percentile(s[..., 0], 95.0)), float(s[..., 1].mean())

    def _stats_late(self):
        """The fractal stats, read a frame late. The stats pass just drawn
        is copied into a pixel-pack buffer without waiting for it (the GPU
        finishes it in the background), and the copy the previous frame
        queued is what this frame uses: that frame's picture readback has
        already drained the queue, so it is ready and costs no stall. A
        synchronous glReadPixels here was the fractal frame's first GPU
        sync. The browser reads its stats every 6th frame; one frame of lag
        under the 0.9 EMA on every bright end is not visible. The first
        fractal frame (from the stock engine, or after a stock frame) has
        nothing queued and reads synchronously. The stock engine keeps its
        same-frame read (amount 0 is the pre-fractal engine, bit for bit)."""
        prev = self._stats_pbo.read() if self._stats_pending else None
        if self._stats_pbo is None:
            self._stats_pbo = self.ctx.buffer(reserve=self.sw * self.sh * 16)
        self.fbo_stats.read_into(self._stats_pbo, components=4, dtype="f4")
        self._stats_pending = True
        if prev is None:
            prev = self.fbo_stats.read(components=4, dtype="f4")
        return prev

    def lum_sample(self):
        """Tonemapped luminance on the stats subsample grid, (sh, sw)
        float32 in [0,1], or None before the first stats pass / on an empty
        trail. Zero extra GL work: the stats readback already happens every
        frame — this is the staleness tracker's input on the GPU-rack path,
        where the full picture never leaves the GPU (no grain term: the
        tracker blurs dust away anyway)."""
        s = getattr(self, "_last_stats", None)
        if s is None or self._last_norm <= 0:
            return None
        x = s[..., 0] * np.float32(1.0 / self._last_norm)
        return (1.0 - np.exp(-self.exposure * x)).astype(np.float32)

    def luminance_into_tex(self):
        """Tonemap the trail into self.tex_lum on the GPU — no readback.

        Returns False when the trail is empty (tex_lum is cleared to black
        instead, matching luminance()'s zeros). The GL output path
        (dtouch.rack_gl) composes tex_lum onward without the picture ever
        leaving the GPU; luminance() is this plus the uint8 readback.

        The CALLER holds the context binding (like _deposit / _blur_decay):
        moderngl's save/restore of the previously-current context is a
        single slot, so a nested `with self.ctx:` here would clobber the
        outer scope's restore and hand a later foreign context our GL
        calls (the test_field_survives_a_foreign_context failure mode)."""
        gl = self._gl
        fa = self._fractal_amount()
        prev_norm = self._last_norm
        norm, lmean = self._stats(fa)
        # Smooth the exposure reference. A raw per-frame p95 renormalizes the
        # picture against its own noise every frame, which reads as a slow
        # pump and costs the image its crispness (the browser port already
        # EMAs this; the desktop did not). It also feeds `sat` and `food`, so
        # a stable reference keeps the sensing scale steady too.
        if self._last_norm > 0.0 and norm > 0.0:
            norm = self._last_norm * NORM_EMA + norm * (1.0 - NORM_EMA)
        self._last_norm = norm      # feedback for the `sat` + `food` scales
        if norm <= 0:
            # black, at the size render_size() promised the caller
            if fa > 0.0:
                self._lum_hi_target()
                self.lum_tex = self.tex_lum_hi
                self.fbo_lum_hi.use()
            else:
                self.lum_tex = self.tex_lum
                self.fbo_lum.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            return False
        gnorm = lmean * 4.0
        if fa > 0.0:
            if min(1.0, fa / FRACTAL_MIX_FULL) < 1.0:
                # compose morphs out of the stock picture at small amounts:
                # draw it first, relief and all, into tex_lum
                self._stock_picture(norm, gnorm)
            self._fractal_picture(fa, norm, gnorm, prev_norm)
            return True
        self.lum_tex = self.tex_lum
        self._stock_picture(norm, gnorm)
        return True

    def _stock_picture(self, norm, gnorm):
        """tonemap.frag into tex_lum (grid size). The caller holds the
        context."""
        gl = self._gl
        p = self.p_tonemap
        p["u_inv_norm"].value = 1.0 / norm
        p["u_grain"].value = float(self.grain) if gnorm > 0 else 0.0
        p["u_inv_gnorm"].value = (1.0 / gnorm) if gnorm > 0 else 0.0
        p["u_exposure"].value = float(self.exposure)
        # relief: the host normalises the light and keeps z off the floor
        # (relief_light) because the shader divides by it unguarded
        p["u_depth"].value = float(self.depth)
        p["u_light"].value = relief_light(self.light)
        self.fbo_lum.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_tonemap.render(gl.TRIANGLES, vertices=3)

    def _fractal_picture(self, fa, norm, gnorm, prev_norm):
        """The fractal amount's picture: per-level fields at grid size
        (tonemap_fractal.frag, two targets), then outlines cut at the output
        size (compose.frag) into tex_lum_hi, which becomes lum_tex. The
        caller holds the context."""
        gl, F = self._gl, FRACTAL
        if self._last_stats4 is not None:
            self._nl = level_norms(self._last_stats4, prev_norm, self._nl)
        if self._nl is None:
            # no readback yet: split the total bright end by deposit share
            self._nl = tuple(max(1e-4, norm * d * FRACTAL_NL["split"]) for d in F["dep"])
        pt = self._pf["tonemap"]
        grain = float(self.grain) if gnorm > 0 else 0.0
        pt["u_inv_norm"].value = 1.0 / norm
        pt["u_grain"].value = grain
        pt["u_inv_gnorm"].value = (1.0 / gnorm) if gnorm > 0 else 0.0
        pt["u_exposure"].value = float(self.exposure)
        pt["u_depth"].value = float(self.depth)
        pt["u_light"].value = relief_light(self.light)
        pt["u_inv_nl"].value = tuple(1.0 / v for v in self._nl)
        pt["u_lvlRelief"].value = tuple(F["relief"])
        pt["u_lvlExp"].value = tuple(F["exp"])
        pt["u_grainK"].value = 1.0 + (F["grain"] - 1.0) * fa
        pt["u_crest"].value = tuple(F["crest"])
        pt["u_lineBeta"].value = F["lineBeta"]
        pt["u_lineGate"].value = F["lineGate"]
        self.fbo_fields.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self._pf["vao_tonemap"].render(gl.TRIANGLES, vertices=3)

        rw, rh = self._lum_hi_target()
        pc = self._pf["compose"]
        pc["u_outSize"].value = (float(rw), float(rh))
        pc["u_grid"].value = (self.gw, self.gh)
        pc["u_mix"].value = min(1.0, fa / FRACTAL_MIX_FULL)
        pc["u_lvlVal"].value = tuple(F["val"])
        pc["u_thr"].value = tuple(F["thr"])
        pc["u_body"].value = tuple(F["body"])
        pc["u_hair"].value = tuple(F["hair"])
        pc["u_glow"].value = F["glow"]
        pc["u_bloom"].value = self._bloom_flat()
        pc["u_bloomThr"].value = F["bloomThr"]
        pc["u_shimmer"].value = (F["shimmer"][0] * fa, F["shimmer"][1] * float(self.px_scale),
                                 float(self.bloom_t))
        self.fbo_lum_hi.use()
        self.tex_fields[0].use(0)
        self.tex_lum.use(1)       # the stock picture (read when u_mix < 1)
        self.tex_fields[1].use(3)
        self._pf["vao_compose"].render(gl.TRIANGLES, vertices=3)
        self.lum_tex = self.tex_lum_hi

    def _lum_hi_target(self):
        """The output-size R8 target of the compose pass, (re)allocated when
        render_size() changes; returns that size."""
        rw, rh = self.render_size()
        if self.tex_lum_hi is None or self.tex_lum_hi.size != (rw, rh):
            if self.tex_lum_hi is not None:
                self.fbo_lum_hi.release()
                self.tex_lum_hi.release()
            # R8 like tex_lum: the picture is a [0,1] luminance, and the
            # desktop quantizes it to a LUT index anyway
            self.tex_lum_hi = self.ctx.texture((rw, rh), 1)
            self.fbo_lum_hi = self.ctx.framebuffer(color_attachments=[self.tex_lum_hi])
        return rw, rh

    def luminance_u8(self):
        """luminance() as the 8-bit values the GPU wrote, (h, w) uint8, no
        float round trip. The mode's colorize wants exactly these bytes
        (uint8 -> x/255 float32 -> x*255 -> uint8 is the identity on all 256
        values), and at the fractal's output size the round trip alone was
        ~2.5 ms of CPU per frame."""
        with self.ctx:
            ok = self.luminance_into_tex()
            t = self.lum_tex
            w, h = t.size
            out = np.zeros((h, w), np.uint8)
            if ok:
                fbo = self.fbo_lum if t is self.tex_lum else self.fbo_lum_hi
                fbo.read_into(out, components=1)
        return out

    def luminance(self):
        """Tonemapped trail in [0,1] float32 — same curve as the CPU field
        (trail normalized by its 95th percentile, `grain` mixing this frame's
        raw deposits over it, 1 - exp(-exposure * x)), evaluated on the GPU
        and read back as 8-bit. (gh, gw) for the stock engine; with the
        fractal amount up it is the compose pass's output-size picture,
        render_size() reversed as (h, w)."""
        return self.luminance_u8().astype(np.float32) * np.float32(1.0 / 255.0)

    # ----- readbacks (slow; tests and diagnostics) -----
    def _read_f4(self, fbo, components):
        """fbo's contents as float32, binding it with use() first. Apple's GL
        needs the bind: once the tonemap has rendered into the uint8 target,
        the second and every later glReadPixels of a float target returns
        the uint8 target's contents (all ones / [0,1] values) with no GL
        error, until some framebuffer is bound for drawing again. The
        per-frame path never hit it — luminance() reads each target right
        after rendering into it — but trail / agents() after luminance()
        did."""
        with self.ctx:
            fbo.use()
            raw = fbo.read(components=components, dtype="f4")
        return np.frombuffer(raw, np.float32).copy()

    @property
    def trail(self):
        """The full float trail, read back from the GPU (gh, gw) float32 —
        the whole organism, i.e. the sum of the species channels, which is
        what the picture and the statistics are taken over."""
        rgba = self._read_f4(self.fbo_trail_a, 4).reshape(self.gh, self.gw, 4)
        return rgba[..., :3].sum(axis=2)

    @trail.setter
    def trail(self, value):
        """Write the whole trail — TESTS AND DIAGNOSTICS ONLY.

        Accepts (gh, gw) to fill every species channel equally, or
        (gh, gw, 3) to set them independently. Exists because the species
        chirality — which population repels which — is otherwise only
        observable through many frames of emergent behaviour, and a test that
        cannot place a known trail cannot ask "which way did this agent turn".
        """
        a = np.asarray(value, dtype=np.float32)
        if a.shape == (self.gh, self.gw):
            a = np.repeat(a[..., None], 3, axis=2)
        if a.shape != (self.gh, self.gw, 3):
            raise ValueError(f"trail must be {(self.gh, self.gw)} or "
                             f"{(self.gh, self.gw, 3)}, got {a.shape}")
        rgba = np.zeros((self.gh, self.gw, 4), np.float32)
        rgba[..., :3] = a
        with self.ctx:
            self.tex_trail_a.write(np.ascontiguousarray(rgba))

    @property
    def trail_species(self):
        """Per-species trail, (gh, gw, 3) float32 — diagnostics and tests."""
        rgba = self._read_f4(self.fbo_trail_a, 4).reshape(self.gh, self.gw, 4)
        return rgba[..., :3].copy()

    def species_of(self):
        """Per-agent species index, int32 length n.

        Diagnostics only — nothing in the render path calls this. Kept because
        "which species is where" is the first question when the mosaic or the
        cross-terms misbehave, and reconstructing it from the trail is lossy.
        """
        a = self._read_f4(self.fbo_agents_a, 4).reshape(self.ah * self.aw, 4)[:self.n]
        return a[:, 3].astype(np.int32)

    def agents(self):
        """(px, py, heading) float32 arrays of length n, read back."""
        a = self._read_f4(self.fbo_agents_a, 4).reshape(self.ah * self.aw, 4)[:self.n]
        return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()

    @property
    def px(self):
        return self.agents()[0]

    @property
    def py(self):
        return self.agents()[1]

    @property
    def heading(self):
        return self.agents()[2]

    # ----- lifecycle -----
    def release(self):
        """Release the context; idempotent (the mode's stop() contract)."""
        ctx, self.ctx = self.ctx, None
        if ctx is not None:
            ctx.release()
