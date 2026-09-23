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

Clean-room note: the model is Jeff Jones's (sense/rotate/move/deposit/
diffuse/decay) and the semantics are this repo's dtouch/physarum.py. No code
or parameter tables from any CC BY-NC-SA physarum project were used.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .physarum import (BALLISTIC_DECAY, NORM_EMA, POINTS, SPATIAL_REGIMES,
                       ZONE_REGIME_COUNT, species_matrix)

# Agent texture width. One texel per agent; the height is ceil(n / width).
# 2048 x 16384 (the GL_MAX_TEXTURE_SIZE floor on anything that runs this)
# is 33M agents, well past what the deposit draw can afford anyway.
AGENT_TEX_W = 2048

# Stride of the statistics subsample (percentile + mean readback).
STATS_STRIDE = 4


SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")
SHADER_DIR = os.path.join(SHADERS_ROOT, "physarum")
SHADER_FILES = ("fullscreen.vert", "update.frag", "deposit.vert", "deposit.frag",
                "blur.frag", "stats.frag", "tonemap.frag", "impulse.frag")

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
                 sat=0.0, jitter=0.0, hetero=0.0, species=3, cross=0.0):
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
        self.seed = seed
        self.frame = 0
        self._last_norm = 0.0     # last luminance() percentile (sat cap ref)
        self.ctx = None
        self._rng = np.random.default_rng(seed)
        self._salt = np.random.default_rng(seed + 1)
        self.aw = min(n, AGENT_TEX_W)
        self.ah = int(math.ceil(n / self.aw))
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
        self.tex_agents_a = ftex((aw, ah), 4, init.tobytes())
        self.tex_agents_b = ftex((aw, ah), 4)
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
        self.tex_stats = ftex((self.sw, self.sh), 2)
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

        # a render pass to prove float targets + blending actually work here
        # (a context can exist and still refuse them); a failure surfaces as
        # PhysarumGLUnavailable from __init__ rather than a black show later
        self._deposit(1.0, 1.0)
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)

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
    def _deposit(self, dep_bg, dep_fg):
        ctx, gl = self.ctx, self._gl
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.enable(gl.BLEND)
        ctx.blend_func = (gl.ONE, gl.ONE)
        self.tex_agents_a.use(0)
        self.tex_matte.use(2)
        self.p_deposit["u_deposit"].value = (dep_bg, dep_fg)
        self.p_deposit["u_nspec"].value = float(self.species)
        self.vao_deposit.render(gl.POINTS, vertices=self.n)
        ctx.disable(gl.BLEND)

    def _blur_decay(self, use_keep=False):
        ctx, gl = self.ctx, self._gl
        r = int(self.diffuse) if self.diffuse > 0 else 0
        k = 2 * r + 1
        p = self.p_blur
        p["u_radius"].value = r
        p["u_wide"].value = max(3 * r, r + 2)
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

            self.fbo_agents_b.use()
            self.tex_agents_a.use(0)
            self.tex_trail_a.use(1)
            self.tex_matte.use(2)
            self.tex_gray.use(3)
            self.vao_update.render(gl.TRIANGLES, vertices=3)
            self._swap_agents()

            md = self.mod_deposit
            self._deposit(a["deposit"] * md, b["deposit"] * md)
            self._blur_decay(use_keep=keep is not None)
        self.frame += 1

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
    def _stats(self):
        """(95th percentile of the trail, mean of this frame's deposits),
        estimated on a stride-STATS_STRIDE subsample read back as floats."""
        gl = self._gl
        self.fbo_stats.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_stats.render(gl.TRIANGLES, vertices=3)
        raw = self.fbo_stats.read(components=2, dtype="f4")
        s = np.frombuffer(raw, np.float32).reshape(self.sh, self.sw, 2)
        self._last_stats = s        # free spatial subsample (lum_sample)
        return float(np.percentile(s[..., 0], 95.0)), float(s[..., 1].mean())

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
        norm, lmean = self._stats()
        # Smooth the exposure reference. A raw per-frame p95 renormalizes the
        # picture against its own noise every frame, which reads as a slow
        # pump and costs the image its crispness (the browser port already
        # EMAs this; the desktop did not). It also feeds `sat` and `food`, so
        # a stable reference keeps the sensing scale steady too.
        if self._last_norm > 0.0 and norm > 0.0:
            norm = self._last_norm * NORM_EMA + norm * (1.0 - NORM_EMA)
        self._last_norm = norm      # feedback for the `sat` + `food` scales
        if norm <= 0:
            self.fbo_lum.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            return False
        gnorm = lmean * 4.0
        p = self.p_tonemap
        p["u_inv_norm"].value = 1.0 / norm
        p["u_grain"].value = float(self.grain) if gnorm > 0 else 0.0
        p["u_inv_gnorm"].value = (1.0 / gnorm) if gnorm > 0 else 0.0
        p["u_exposure"].value = float(self.exposure)
        self.fbo_lum.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_tonemap.render(gl.TRIANGLES, vertices=3)
        return True

    def luminance(self):
        """Tonemapped trail in [0,1] float32 (gh, gw) — same curve as the CPU
        field (trail normalized by its 95th percentile, `grain` mixing this
        frame's raw deposits over it, 1 - exp(-exposure * x)), evaluated on
        the GPU and read back as 8-bit."""
        with self.ctx:
            if not self.luminance_into_tex():
                return np.zeros((self.gh, self.gw), np.float32)
            raw = self.fbo_lum.read(components=1)
        lum = np.frombuffer(raw, np.uint8).reshape(self.gh, self.gw)
        return lum.astype(np.float32) * np.float32(1.0 / 255.0)

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
