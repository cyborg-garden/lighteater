"""The "alive" fractal step on the GPU engine: the passes around the stock
fractal veins, and the molten-ink display. The desktop twin of the browser's
alive.js; both run the shared shaders in dtouch/shaders/alive/ with the table
in dtouch.alive.ALIVE.

What H's third step (depth + fractal veins) carries, on top of the fractal
veins. Flat and depth-only never touch any of it: PhysarumFieldGL asks
`alive_on()` once per pass and, when it says no, runs the stock programs
exactly as before.

  A. hunting: motion is fast food (a fast-decaying map from the frame
     difference, weighted to the fine levels, plus a wider skirt the threads
     lean toward), exploring tips drawn as small bright heads, fresh trail
     drawn brighter than settled trail, a short light that travels along a
     thread when it closes onto another vein, pruning of threads nobody
     walks, and a slow shuttle pulse travelling along the veins from food;
  B. bending in: veins steer along lines of equal depth (a dome over the
     matte, or a luminance pseudo-depth) so they wrap around shapes;
  C. the molten calligraphy (ink.frag), the display pass.

Pass order with the step on (the field drives it):
  update(): scene -> update (alive copy) -> deposit -> life -> blur (alive
  copy) -> events, glow, pulse;  luminance_into_tex(): stats -> fractal
  tonemap -> compose (alive copy, into this module's RGBA16F picture) ->
  heads -> (.r copied into the field's luminance target);  ink_draw():
  ink (capped at INK_MAX_PX) -> ink_blit / ink_read: flip + scale on the GPU.

Failures are fail-soft: the field catches any exception from an alive pass,
and checks the GL error flag after each alive half, the compose and the ink
(an out-of-memory allocation only sets the flag), then records alive_error
and runs the stock fractal step that same frame (the browser's aliveFail).

Not ported from the browser (yet): the carved composite (carve.frag, the
look with ALIVE["ink"] < 0 over the camera; here that case colourises the
alive picture's .r like any luminance) and the cast accents on the ink.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .alive import (ALIVE, advance_pulse, count_events, depth_quantile,
                    head_stride, keep_for)
from .physarum import FRACTAL, FRACTAL_NL, PX_REF

SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")
ALIVE_DIR = os.path.join(SHADERS_ROOT, "alive")
GLSL_VERSION_LINE = "#version 330 core\n"

# Width of the scene pass's grid (the browser's matte resolution): the dome
# radius and the pseudo-depth blur are counted in its texels, so both hosts
# see shapes at the same scale whatever the sim grid.
SCENE_W = 320

# fx accents are absent on the desktop: an age past every lifetime
_NO_FX = (0.0, 0.0, 99.0)

# (vertex, fragments...) per program; "fs" is the physarum unit's attribute
# full-screen triangle, "quad" the alive unit's attribute-less one
PROGRAMS = {
    "update": ("fs", ("update.frag",)),
    "blur": ("fs", ("blur.frag",)),
    "compose": ("fs", ("compose.frag",)),
    "life": ("fs", ("life.frag",)),
    "events": ("fs", ("events.frag",)),
    "glow": ("fs", ("glow.frag",)),
    "scene": ("quad", ("scene.frag",)),
    "pulse": ("quad", ("pulse.frag",)),
    "heads": ("heads.vert", ("heads.frag",)),
    "ink": ("quad", ("video.glsl", "ink.frag")),
}

# the ink pass's pixel budget (ink_size): above it the ink draws smaller and
# the blit upscales it (measured at 4K: 2733x1537 -> ~1 MP saves ~9 ms)
INK_MAX_PX = 1_000_000

# the ink's flip to image space + bilinear scale (canvas -> rows top-down),
# desktop only
_BLIT_FRAG = """#version 330 core
uniform sampler2D u_src;
in vec2 v_uv;
layout(location = 0) out vec4 o;
void main() { o = vec4(texture(u_src, vec2(v_uv.x, 1.0 - v_uv.y)).rgb, 1.0); }
"""

# the luminance copy (alive picture .r -> the field's R8 target), desktop only
_COPY_FRAG = """#version 330 core
uniform sampler2D u_src;
layout(location = 0) out vec4 o;
void main() { o = vec4(texelFetch(u_src, ivec2(gl_FragCoord.xy), 0).r, 0.0, 0.0, 1.0); }
"""


def alive_source(*names):
    """The shared alive files concatenated behind the desktop version line,
    each under its own `#line 1` so compile errors keep file-true lines."""
    parts = [GLSL_VERSION_LINE]
    for name in names:
        with open(os.path.join(ALIVE_DIR, name), "r", encoding="utf-8") as fh:
            parts.append("#line 1\n" + fh.read())
    return "".join(parts)


def ink_size(w, h, budget=INK_MAX_PX):
    """The ink pass's size: (w, h), scaled down to `budget` pixels."""
    k = min(1.0, math.sqrt(budget / max(1.0, float(w) * h)))
    return max(1, int(round(w * k))), max(1, int(round(h * k)))


def scene_size(gw, gh):
    """The scene pass's grid: SCENE_W wide at the sim grid's aspect."""
    return SCENE_W, max(16, int(round(SCENE_W * gh / max(1, gw))))


class AliveGL:
    """The alive passes on a PhysarumFieldGL's context. Build inside the
    field's context binding; raises on any program that will not build (and
    leaves nothing behind)."""

    def __init__(self, field):
        import moderngl
        self.mgl = moderngl
        self.f = field
        ctx = self.ctx = field.ctx
        from .physarum_gl import load_shader
        fs = load_shader("fullscreen.vert")
        quad = alive_source("quad.vert")
        self.prog, self.vao = {}, {}
        try:
            for key, (vert, frags) in PROGRAMS.items():
                vs = fs if vert == "fs" else quad if vert == "quad" else alive_source(vert)
                p = ctx.program(vertex_shader=vs, fragment_shader=alive_source(*frags))
                self.prog[key] = p
                self.vao[key] = (ctx.vertex_array(p, [(field.tri_vbo, "2f", "in_vert")])
                                 if vert == "fs" else ctx.vertex_array(p, []))
            cp = ctx.program(vertex_shader=quad, fragment_shader=_COPY_FRAG)
            self.prog["copy"], self.vao["copy"] = cp, ctx.vertex_array(cp, [])
            bp = ctx.program(vertex_shader=quad, fragment_shader=_BLIT_FRAG)
            self.prog["blit"], self.vao["blit"] = bp, ctx.vertex_array(bp, [])
        except Exception:
            self._release_programs()
            raise
        lin, near = (moderngl.LINEAR, moderngl.LINEAR), (moderngl.NEAREST, moderngl.NEAREST)
        self.s_lin = ctx.sampler(filter=lin, repeat_x=False, repeat_y=False)
        self.s_near = ctx.sampler(filter=near, repeat_x=False, repeat_y=False)
        self.s_mip = ctx.sampler(filter=(moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR),
                                 repeat_x=False, repeat_y=False)
        self._units = set()
        self.pulse_phase = 0.0
        self.event_count = 0
        self.event_log = []       # (sim_t, n) per readback batch
        self.last_read = 0.0
        self.heads = 0
        self.sim_t = 0.0
        self.frame = 0
        self.src_depth = None
        self.depth_failed = False
        self.read_failed = False
        self.pairs = {}           # key -> [(tex, fbo), (tex, fbo)]
        self.cur = {}
        self.fresh = {k: True for k in ("scene", "life", "events", "glow", "pulse")}
        self.out = None           # (tex, fbo): the alive picture, output size
        self.ink_out = None       # (tex, fbo): the ink pass's RGBA8 picture
        self.blit_out = None      # (tex, fbo): ink_read's output-size RGBA8
        self.tex_motion = self.tex_matte = None
        self.tex_video = None
        self.tex_lut = None
        self._lut_key = None

    # ----- small GL helpers --------------------------------------------------
    def _u(self, p, name, value):
        """Set a uniform when the program kept it (the GLSL compiler drops
        what a pass never reads)."""
        m = p.get(name, None)
        if m is not None:
            m.value = value

    def _bind(self, p, unit, tex, samp, name):
        tex.use(location=unit)
        samp.use(location=unit)
        self._units.add((unit, samp))
        self._u(p, name, unit)

    def _unbind(self):
        """Sampler objects override a texture's own filter on their unit:
        clear them after each pass so the field's stock passes read their
        textures exactly as before."""
        for unit, samp in self._units:
            samp.clear(location=unit)
        self._units.clear()

    def _target(self, w, h, comps=4, dtype="f2"):
        t = self.ctx.texture((w, h), comps, dtype=dtype)
        t.repeat_x = t.repeat_y = False
        try:
            fbo = self.ctx.framebuffer(color_attachments=[t])
        except Exception:
            t.release()
            raise
        return (t, fbo)

    def _free(self, tf):
        if tf is None:
            return
        for o in reversed(tf):
            try:
                o.release()
            except Exception:                     # noqa: BLE001: already gone
                pass

    def _pair(self, key, w, h):
        cur = self.pairs.get(key)
        if cur and cur[0][0].size == (w, h):
            return cur
        if cur:
            for tf in cur:
                self._free(tf)
        self.pairs.pop(key, None)
        built = []
        try:
            for _ in range(2):
                built.append(self._target(w, h))
        except Exception:
            for tf in built:
                self._free(tf)
            raise
        self.pairs[key] = built
        self.cur[key] = 0
        self.fresh[key] = True
        return built

    def tex(self, key):
        return self.pairs[key][self.cur[key]][0]

    def size(self, key):
        return self.pairs[key][0][0].size

    def targets(self):
        f = self.f
        sw, sh = scene_size(f.gw, f.gh)
        self._pair("scene", sw, sh)
        self._pair("life", f.gw, f.gh)
        c = ALIVE["cell"]
        self._pair("events", int(math.ceil(f.gw / c)), int(math.ceil(f.gh / c)))
        self._pair("glow", f.gw, f.gh)
        self._pair("pulse", max(8, int(math.ceil(f.gw / 4))), max(8, int(math.ceil(f.gh / 4))))

    def out_for(self, w, h):
        """The alive picture (r whole, g veins, b accents), output size."""
        if self.out is None or self.out[0].size != (w, h):
            self._free(self.out)
            self.out = None
            self.out = self._target(w, h)
        return self.out

    def _draw(self, key, fbo, setup):
        gl = self.mgl
        fbo.use()
        p = self.prog[key]
        setup(p)
        self.vao[key].render(gl.TRIANGLES, vertices=3)
        self._unbind()

    def _run(self, key, prog, setup):
        """Step one ping-pong pair through `prog`."""
        pair, cur = self.pairs[key], self.cur[key]

        def go(p):
            self._bind(p, 0, pair[cur][0], self.s_near, "u_prev")
            self._u(p, "u_reset", 1.0 if self.fresh[key] else 0.0)
            setup(p)
        self._draw(prog, pair[1 - cur][1], go)
        self.cur[key] = 1 - cur
        self.fresh[key] = False

    def norms(self):
        """The fractal's per-level bright ends (the field's level_norms), or
        the field's split of the total before its first readback."""
        f = self.f
        if f._nl is not None:
            return f._nl
        norm = f._food_norm()
        return tuple(max(1e-4, norm * d * FRACTAL_NL["split"]) for d in FRACTAL["dep"])

    # ----- the passes ---------------------------------------------------------
    def _upload_scene_inputs(self):
        """The mode's motion map (r history, g luma, b instantaneous motion)
        and matte at the scene size; zeros when the host gave none."""
        f = self.f
        sw, sh = scene_size(f.gw, f.gh)
        motion, matte = getattr(f, "alive_scene_in", (None, None))
        if motion is None or motion.shape != (sh, sw, 4):
            motion = np.zeros((sh, sw, 4), np.float32)
        if matte is None or matte.shape != (sh, sw):
            matte = np.zeros((sh, sw), np.float32)
        if self.tex_motion is None or self.tex_motion.size != (sw, sh):
            for t in (self.tex_motion, self.tex_matte):
                if t is not None:
                    t.release()
            self.tex_motion = self.ctx.texture((sw, sh), 4, dtype="f4")
            self.tex_matte = self.ctx.texture((sw, sh), 1, dtype="f4")
        self.tex_motion.write(np.ascontiguousarray(motion, np.float32))
        self.tex_matte.write(np.ascontiguousarray(matte, np.float32))

    def step_scene(self, dt):
        """Before the update: fresh motion, the matte distance and the depth."""
        f = self.f
        self.targets()
        self.sim_t += dt
        self._upload_scene_inputs()
        sw, sh = self.size("scene")
        person = bool(getattr(f, "alive_person", False))
        react = float(getattr(f, "alive_react", 0.0))

        def setup(p):
            self._bind(p, 1, self.tex_motion, self.s_lin, "u_motion")
            self._bind(p, 2, self.tex_matte, self.s_lin, "u_matte")
            self._u(p, "u_texel", (1.0 / sw, 1.0 / sh))
            self._u(p, "u_freshKeep", keep_for(ALIVE["freshHalfLife"], dt))
            self._u(p, "u_skirtKeep", keep_for(ALIVE["skirtHalfLife"], dt))
            self._u(p, "u_freshIn", ALIVE["freshIn"] * max(react, 0.25))
            self._u(p, "u_domeR", float(ALIVE["domeR"]))
            self._u(p, "u_domeW", ALIVE["domeW"][1 if person else 0])
            self._u(p, "u_ease", ALIVE["depthEase"])
        self._run("scene", "scene", setup)

    def update_uniforms(self, p, fa):
        """The alive update's extra uniforms (the field sets every stock one)."""
        norm = max(self.f._food_norm(), 1e-4)
        sw, sh = self.size("scene")
        self._bind(p, 4, self.tex("scene"), self.s_lin, "u_scene")
        self._u(p, "u_alive", fa)
        self._u(p, "u_sceneTexel", (1.0 / sw, 1.0 / sh))
        self._u(p, "u_freshFood", tuple(v * fa * norm for v in ALIVE["freshFood"]))
        self._u(p, "u_freshTurn", tuple(v * fa for v in ALIVE["freshTurn"]))
        self._u(p, "u_freshGain", float(ALIVE["freshGain"]))
        self._u(p, "u_freshLand", float(ALIVE["freshLand"]))
        self._u(p, "u_freshPull", ALIVE["freshPull"] * fa)
        self._u(p, "u_contour", tuple(v * fa for v in ALIVE["contour"]))
        self._u(p, "u_contourGain", float(ALIVE["contourGain"]))
        self._u(p, "u_slope", ALIVE["slope"] * fa)

    def step_life(self, dt):
        """After the deposit: trail age and the deposit memory. The fields
        are last frame's drawing: none yet on the very first frame."""
        f = self.f
        if f.tex_fields is None or f.tex_fields[0].size != (f.gw, f.gh):
            return

        def setup(p):
            self._bind(p, 1, f.tex_fields[0], self.s_near, "u_fields")
            self._bind(p, 2, f.tex_laid, self.s_near, "u_laid")
            self._u(p, "u_thr", tuple(FRACTAL["thr"]))
            self._u(p, "u_onOff", tuple(ALIVE["onOff"]))
            self._u(p, "u_dt", float(dt))
            self._u(p, "u_fluxKeep", keep_for(ALIVE["fluxHalfLife"], dt))
        self._run("life", "life", setup)

    def blur_uniforms(self, p, fa):
        """The alive blur's extra uniforms, both passes (pruning bites only in
        the V pass, where the decay is below 1). No deposit memory yet (the
        first alive frame): no pruning either."""
        self._bind(p, 3, self.tex("life"), self.s_near, "u_life")
        self._u(p, "u_alive", 0.0 if self.fresh["life"] else fa)
        self._u(p, "u_prune", tuple(ALIVE["prune"]))
        self._u(p, "u_pruneAt", tuple(ALIVE["pruneAt"]))

    def step_network(self, dt):
        """After the blur: connection events, their glow, the pulse's
        distance; the readbacks on their cadence."""
        f = self.f
        life = self.tex("life")
        c = ALIVE["cell"]

        def ev(p):
            self._bind(p, 1, life, self.s_near, "u_life")
            self._u(p, "u_grid", (f.gw, f.gh))
            self._u(p, "u_cell", int(c))
            self._u(p, "u_dt", float(dt))
            self._u(p, "u_cool", float(ALIVE["cool"]))
            self._u(p, "u_young", float(ALIVE["young"]))
            self._u(p, "u_settled", float(ALIVE["settled"]))
            self._u(p, "u_reach", float(ALIVE["reach"]))
            self._u(p, "u_minAge", float(ALIVE["minAge"]))
        self._run("events", "events", ev)
        events = self.tex("events")

        def glow(p):
            self._bind(p, 1, life, self.s_near, "u_life")
            self._bind(p, 2, events, self.s_near, "u_events")
            self._u(p, "u_grid", (f.gw, f.gh))
            self._u(p, "u_cell", int(c))
            self._u(p, "u_dt", float(dt))
            self._u(p, "u_step", tuple(ALIVE["glowStep"]))
            self._u(p, "u_young", float(ALIVE["young"]))
            self._u(p, "u_fade", float(ALIVE["glowFade"]))
            self._u(p, "u_refr", float(ALIVE["glowRefr"]))
            self._bind(p, 3, self.tex("scene"), self.s_lin, "u_scene")
            self._u(p, "u_calm", float(ALIVE["glowCalm"]))
            self._u(p, "u_time", float(self.sim_t))
        self._run("glow", "glow", glow)
        if self.frame % 30 == 0 or self.src_depth is None:
            self.read_depth()
        nl = self.norms()
        pw, ph = self.size("pulse")

        def pulse(p):
            self._bind(p, 1, f.tex_trail_a, self.s_near, "u_trail")
            self._bind(p, 2, self.tex("scene"), self.s_lin, "u_scene")
            self._u(p, "u_inv_nl", tuple(1.0 / v for v in nl))
            self._u(p, "u_texel", (1.0 / pw, 1.0 / ph))
            self._u(p, "u_src", float(self.src_depth))
            self._u(p, "u_creep", float(ALIVE["pulseCreep"]))
            self._u(p, "u_far", float(ALIVE["pulseFar"]))
            self._u(p, "u_offCost", float(ALIVE["pulseOffCost"]))
        for _ in range(2):
            self._run("pulse", "pulse", pulse)
        snd = float(getattr(f, "alive_sound", 0.0))
        self.pulse_phase = advance_pulse(self.pulse_phase, dt, ALIVE["pulsePeriod"],
                                         snd, ALIVE["pulseSound"])
        if self.frame % 15 == 0:
            self.read_events()
        self.frame += 1

    def read_depth(self):
        """Where the pulse starts: the depth at a high quantile of the scene
        (the nearest few percent), read back small and seldom. Fail-soft:
        without the readback the pulse starts at a fixed depth."""
        if self.src_depth is None:
            self.src_depth = ALIVE["pulseSrcFallback"]
        if self.depth_failed:
            return
        try:
            fbo = self.pairs["scene"][self.cur["scene"]][1]
            fbo.use()
            buf = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32)
            self.src_depth = depth_quantile(buf, ALIVE["pulseSrcQ"])
        except Exception:                         # noqa: BLE001: fail-soft
            self.depth_failed = True

    def read_events(self):
        """Count connection events (fail-soft: a stack that refuses the float
        readback only loses the count, never the glow)."""
        since = self.sim_t - self.last_read
        self.last_read = self.sim_t
        if self.read_failed or since <= 0:
            return
        try:
            fbo = self.pairs["events"][self.cur["events"]][1]
            fbo.use()
            buf = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32)
            k = count_events(buf, since)
            self.event_count += k
            self.event_log.append((round(self.sim_t, 2), k))
            if len(self.event_log) > 400:
                self.event_log.pop(0)
        except Exception:                         # noqa: BLE001: fail-soft
            self.read_failed = True

    def compose_uniforms(self, p, fa):
        """The alive compose: the field sets the stock uniforms; these are
        the additions."""
        self._bind(p, 4, self.tex("life"), self.s_lin, "u_life")
        self._bind(p, 5, self.tex("glow"), self.s_lin, "u_glowTex")
        self._bind(p, 6, self.tex("pulse"), self.s_lin, "u_pulseD")
        self._u(p, "u_alive", fa)
        self._u(p, "u_front", tuple(ALIVE["front"]))
        self._u(p, "u_frontAge", float(ALIVE["frontAge"]))
        self._u(p, "u_pulse", (float(ALIVE["pulseAmp"]), float(ALIVE["pulseWave"]),
                               float(self.pulse_phase), float(ALIVE["pulseSwell"])))
        self._u(p, "u_pulseFar", float(ALIVE["pulseFar"]))
        self._u(p, "u_glowK", float(ALIVE["glowK"]))

    def draw_heads(self):
        """Exploring tips, drawn into the alive picture's r and b channels."""
        gl, f = self.mgl, self.f
        tex, fbo = self.out
        p = self.prog["heads"]
        n_sim = f.n_active()
        stride = head_stride(n_sim, ALIVE["headsTarget"])
        n = int(math.ceil(n_sim / stride))
        px = f.gw / float(PX_REF)
        nl = self.norms()
        fbo.use()
        f.tex_agents_a.use(location=0)
        self.s_near.use(location=0)
        f.tex_trail_a.use(location=1)
        self.s_near.use(location=1)
        self._units.update({(0, self.s_near), (1, self.s_near)})
        self._u(p, "u_agents", 0)
        self._u(p, "u_trail", 1)
        self._u(p, "u_grid", (f.gw, f.gh))
        self._u(p, "u_aw", int(f.aw))
        self._u(p, "u_n", int(n_sim))
        self._u(p, "u_stride", int(stride))
        self._u(p, "u_inv_nl", tuple(1.0 / v for v in nl))
        self._u(p, "u_look", ALIVE["headLook"] * px)
        self._u(p, "u_tail", ALIVE["headTail"] * px)
        self._u(p, "u_headPx", ALIVE["headPx"] * math.sqrt(max(1.0, tex.size[0] / f.gw)))
        self._u(p, "u_k", ALIVE["headK"] * min(1.0, max(float(f.fractal), 0.0)))
        ctx = self.ctx
        ctx.enable(gl.BLEND | gl.PROGRAM_POINT_SIZE)
        ctx.blend_func = (gl.ONE, gl.ONE)
        fbo.color_mask = (True, False, True, False)
        try:
            self._u(p, "u_points", 0.0)
            self.vao["heads"].render(gl.LINES, vertices=2 * n)
            self._u(p, "u_points", 1.0)
            self.vao["heads"].render(gl.POINTS, vertices=n)
        finally:
            fbo.color_mask = (True, True, True, True)
            ctx.disable(gl.BLEND | gl.PROGRAM_POINT_SIZE)
            self._unbind()
        self.heads = n

    def copy_luminance(self, dst_fbo):
        """The alive picture's .r into the field's R8 luminance target, so
        luminance() and the rack path read the alive picture."""
        def setup(p):
            self._bind(p, 0, self.out[0], self.s_near, "u_src")
        self._draw("copy", dst_fbo, setup)

    # ----- the display ----------------------------------------------------------
    def _lut(self, lut_rows):
        """The palettes as one RGBA8 texture, a row per palette (the
        browser's buildLut)."""
        key = id(lut_rows)
        if self.tex_lut is None or self._lut_key != key:
            if self.tex_lut is not None:
                self.tex_lut.release()
            rows = np.asarray(lut_rows, np.uint8)        # (n, 256, 3)
            rgba = np.concatenate([rows, np.full(rows.shape[:2] + (1,), 255, np.uint8)], axis=2)
            self.tex_lut = self.ctx.texture((256, rows.shape[0]), 4, data=np.ascontiguousarray(rgba))
            self._lut_key = key
        return self.tex_lut

    def _video(self, rgb):
        """The camera frame as the ink's ground (RGB uint8, image space)."""
        if rgb is None:
            rgb = np.zeros((2, 2, 3), np.uint8)
        h, w = rgb.shape[:2]
        if self.tex_video is None or self.tex_video.size != (w, h):
            if self.tex_video is not None:
                self.tex_video.release()
            self.tex_video = self.ctx.texture((w, h), 3)
        self.tex_video.write(np.ascontiguousarray(rgb, np.uint8))
        return self.tex_video

    def ink_draw(self, lut_rows, pal_row, video_rgb=None, bg="off", video_mix=0.5,
                 blackout=False):
        """The molten calligraphy into ink_out (RGBA8, canvas space: row 0 at
        the bottom), at the alive picture's size capped at INK_MAX_PX pixels
        (the strokes are anti-aliased per pixel, so a smaller target is the
        same picture, softer; ink_blit upscales it). `pal_row` is the
        palette's row centre in lut_rows ((index + 0.5) / rows)."""
        gl, f = self.mgl, self.f
        w, h = ink_size(*self.out[0].size)
        if self.ink_out is None or self.ink_out[0].size != (w, h):
            self._free(self.ink_out)
            self.ink_out = None
            self.ink_out = self._target(w, h, comps=4, dtype="f1")
        lut = self._lut(lut_rows)
        video = self._video(video_rgb)
        fields = f.tex_fields[0]
        # the molten heat reads a blur of the fields: their mip chain,
        # rebuilt each frame (level 0 is all the rest of the pipeline reads;
        # build_mipmaps switches the texture to a mip filter, so put the
        # field's own filter back and let the mip sampler own unit 5)
        fields.build_mipmaps()
        fields.filter = (gl.LINEAR, gl.LINEAR)

        def setup(p):
            self._bind(p, 0, self.out[0], self.s_lin, "u_lum")
            self._bind(p, 1, lut, self.s_lin, "u_lut")
            self._bind(p, 2, video, self.s_lin, "u_video")
            self._bind(p, 4, fields, self.s_lin, "u_fields")
            self._bind(p, 5, fields, self.s_mip, "u_fieldsMip")
            self._u(p, "u_thr", tuple(FRACTAL["thr"]))
            self._u(p, "u_inkThr", tuple(ALIVE["inkThr"]))
            self._u(p, "u_gridTexel", (1.0 / f.gw, 1.0 / f.gh))
            self._u(p, "u_gridSize", (float(f.gw), float(f.gh)))
            # the shell already mirrored and sized the frame: identity fit
            self._u(p, "u_vidScale", (1.0, 1.0))
            self._u(p, "u_vidOff", (0.0, 0.0))
            self._u(p, "u_mirror", 0.0)
            self._u(p, "u_palRow", float(pal_row))
            self._u(p, "u_palRowFrom", float(pal_row))
            self._u(p, "u_palMix", 1.0)
            self._u(p, "u_bgOn", 0.0 if bg == "off" or video_rgb is None else 1.0)
            self._u(p, "u_bgLuma", 1.0 if bg == "veil" else 0.0)
            self._u(p, "u_bgMix", 0.35 + video_mix if bg == "veil" else 1.0)
            self._u(p, "u_blackout", 1.0 if blackout else 0.0)
            self._u(p, "u_light", tuple(float(v) for v in f.light))
            self._u(p, "u_variant", float(ALIVE["ink"]))
            self._u(p, "u_press", tuple(ALIVE["press"]))
            self._u(p, "u_dry", tuple(ALIVE["dry"]))
            self._u(p, "u_sheen", tuple(ALIVE["sheen"]))
            self._u(p, "u_ground", tuple(ALIVE["ground"]))
            self._u(p, "u_edge", tuple(ALIVE["edge"]))
            self._u(p, "u_nib", tuple(ALIVE["nib"]))
            self._u(p, "u_thorn", tuple(ALIVE["thorn"]))
            self._u(p, "u_tierBump", tuple(ALIVE["tierBump"]))
            self._u(p, "u_body", tuple(ALIVE["inkBody"]))
            self._u(p, "u_chrome", tuple(ALIVE["chrome"]))
            self._u(p, "u_thornRim", float(ALIVE["thornRim"]))
            self._u(p, "u_time", float(self.sim_t))
            self._u(p, "u_fxBurst", _NO_FX)
            self._u(p, "u_fxWave", _NO_FX)
            self._u(p, "u_fxAspect", w / max(1.0, float(h)))
        self._draw("ink", self.ink_out[1], setup)

    def ink_blit(self, fbo, w, h):
        """ink_out flipped to image space (row 0 at the top, as the desktop's
        frames and the GPU rack's source are) and bilinearly scaled into
        `fbo` (w x h), on the GPU."""
        fbo.viewport = (0, 0, w, h)

        def setup(p):
            self._bind(p, 0, self.ink_out[0], self.s_lin, "u_src")
        self._draw("blit", fbo, setup)

    def ink_read(self, w, h, timed=None):
        """The ink at w x h as (h, w, 3) uint8 RGB, row 0 at the top: one
        GPU blit, one RGB readback, no CPU flip or resize. `timed` wraps the
        blit (the field's alive clock); the readback stays outside it, since
        it also waits on the rest of the frame."""
        if self.blit_out is None or self.blit_out[0].size != (w, h):
            self._free(self.blit_out)
            self.blit_out = None
            self.blit_out = self._target(w, h, comps=4, dtype="f1")
        blit = lambda: self.ink_blit(self.blit_out[1], w, h)   # noqa: E731
        timed(blit) if timed is not None else blit()
        out = np.empty((h, w, 3), np.uint8)
        self.blit_out[1].read_into(out, components=3, alignment=1)
        return out

    def stats(self):
        t = self.sim_t
        recent = [(at, n) for at, n in self.event_log if at > t - 60]
        first = self.event_log[0][0] - 0.25 if self.event_log else t
        span = min(60.0, max(1e-3, t - first))
        return {
            "events": self.event_count,
            "eventsPerMin": round(sum(n for _, n in recent) * (60.0 / span), 1),
            "heads": self.heads, "pulsePhase": round(self.pulse_phase, 3),
            "readFailed": bool(self.read_failed),
        }

    # ----- lifecycle ------------------------------------------------------------
    def free_readback(self):
        """ink_read's output-size target, not needed while the ink feeds the
        GPU rack directly."""
        self._free(self.blit_out)
        self.blit_out = None

    def free_targets(self):
        """The render targets only (the programs stay built): freed when the
        step turns off and allocated again on the next alive frame."""
        for key in list(self.pairs):
            for tf in self.pairs[key]:
                self._free(tf)
        self.pairs.clear()
        for tf in (self.out, self.ink_out, self.blit_out):
            self._free(tf)
        self.out = self.ink_out = self.blit_out = None
        for k in self.fresh:
            self.fresh[k] = True

    def _release_programs(self):
        for v in list(self.vao.values()) + list(self.prog.values()):
            try:
                v.release()
            except Exception:                     # noqa: BLE001
                pass
        self.vao, self.prog = {}, {}

    def release(self):
        self.free_targets()
        for t in (self.tex_motion, self.tex_matte, self.tex_video, self.tex_lut,
                  self.s_lin, self.s_near, self.s_mip):
            try:
                if t is not None:
                    t.release()
            except Exception:                     # noqa: BLE001
                pass
        self._release_programs()
