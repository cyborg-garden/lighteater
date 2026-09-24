"""SIGNAL rack + physarum output composition on the GPU.

The CPU rack (dtouch.circuit_bent) costs ~36-45 ms at 4K — scanlines +
quantisation 22.6 ms, scan-drift 10.7, chroma 8.3, dither 2.5 (PR #26's
measured port list). This module runs the same stages as fragment passes on
the *simulation's own* moderngl context, so in physarum's GL engine the
picture never round-trips through numpy: tonemap -> colorize -> upscale ->
video composite -> the whole rack, all on-GPU, then ONE uint8 RGB readback
of the composed frame at output resolution.

Split of responsibilities (the parity contract):

- ``CircuitBent.plan()`` owns ALL randomness — per-row drift, chroma IIR,
  glitch events. Both backends consume the same plan, so a fixed seed bends
  the picture identically on either path.
- ``SignalRackGL`` mirrors ``CircuitBent.apply_plan`` pass for pass:
  drift+chroma (one wrapped fetch per channel), glitch tile capture/replay
  (a held GL texture), bit crush (roundEven == np.round), dither at the
  rack's working resolution, scanlines + the final truncating u8 encode.
- Ordered dither (bayer / blue-noise) is a pure threshold test — fully
  in-shader against the same matrices. Error diffusion (fs / riemersma) is
  inherently serial: the working-res image (small — max(96, h//6) rows for
  physarum) is read back, run through the *same* dtouch.dither functions on
  the CPU, and uploaded again. Same cost as the CPU rack pays for those
  modes, honest and identical.
- ``PhysarumOutGL`` adds the mode's output stages in front of the rack:
  palette-LUT / video-lit colorize at grid res, bilinear upscale to output
  res (cv2 INTER_LINEAR convention), and the video-background screen blend.

Shaders live in ``dtouch/shaders/rack/`` (NOT browser-shared — see that
directory's README), except ``dither.frag``, which lives in the
browser-shared unit ``dtouch/shaders/dither/`` and is loaded through
``load_shared_shader``. ``tests/test_rack_gl.py`` holds the backends together
with per-stage parity tests; the caller binds the context (``with ctx:``,
the physarum_gl pattern — a foreign moderngl context created later would
otherwise silently receive these GL calls).
"""
from __future__ import annotations

import os

import numpy as np

from .circuit_bent import SCANLINE_ROWS
# The one loader for browser-shared units (version line + `#line 1`).
from .physarum_gl import load_shared_shader  # noqa: F401  (re-exported)
from .dither import (_MID_GREY_LINEAR, _bayer_matrix, _blue_noise_matrix,
                     _lut_apply, _srgb2lin_lut, _to_u8, linear_to_srgb)

SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")
SHADER_DIR = os.path.join(SHADERS_ROOT, "rack")


def load_rack_shader(name):
    """Verbatim source of dtouch/shaders/rack/<name> (they carry their own
    #version line — unlike the browser-shared units)."""
    with open(os.path.join(SHADER_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def _nearest(tex):
    import moderngl
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = tex.repeat_y = False
    return tex


class SignalRackGL:
    """CircuitBent's pixel backend as fragment passes on an existing context.

    Built once per output size; ``run(cb, plan, src_tex)`` renders the final
    racked frame into ``self.fbo_out`` and ``read()`` returns it as uint8
    RGB (h, w, 3). The caller owns the context binding and the plan draw —
    exactly one ``cb.plan(h, w)`` per frame, whichever backend applies it.
    """

    def __init__(self, ctx, w, h):
        import moderngl
        self._gl = moderngl
        self.ctx = ctx
        self.w, self.h = w, h

        vs = load_rack_shader("fullscreen.vert")

        def prog(frag):
            return ctx.program(vertex_shader=vs,
                               fragment_shader=load_rack_shader(frag))
        self.p_driftchroma = prog("driftchroma.frag")
        self.p_blit = prog("blit.frag")
        self.p_crush = prog("crush.frag")
        self.p_down = prog("downsample.frag")
        self.p_dither = ctx.program(
            vertex_shader=vs,
            fragment_shader=load_shared_shader("dither", "dither.frag"))
        self.p_compose = prog("compose.frag")

        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.tri_vbo = ctx.buffer(tri.tobytes())

        def vao(p):
            return ctx.vertex_array(p, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_driftchroma = vao(self.p_driftchroma)
        self.vao_blit = vao(self.p_blit)
        self.vao_crush = vao(self.p_crush)
        self.vao_down = vao(self.p_down)
        self.vao_dither = vao(self.p_dither)
        self.vao_compose = vao(self.p_compose)

        # full-res ping (RGBA8: pre-crush stages only move pixels, so the
        # u8 lattice is preserved exactly) + the final output target
        self.tex_a = _nearest(ctx.texture((w, h), 4, dtype="f1"))
        self.tex_out = _nearest(ctx.texture((w, h), 4, dtype="f1"))
        self.fbo_a = ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_out = ctx.framebuffer(color_attachments=[self.tex_out])

        # glitch tile: rack blocks are < h//4 x w//2 (rng.integers upper
        # bounds are exclusive); held across frames like the numpy tile
        tw, th = w // 2 + 8, h // 4 + 8
        self.tex_tile = _nearest(ctx.texture((tw, th), 4, dtype="f1"))
        self.fbo_tile = ctx.framebuffer(color_attachments=[self.tex_tile])
        self._tile_seq = -1          # which glitch event the tile belongs to

        self.tex_drift = _nearest(ctx.texture((1, h), 1, dtype="f4"))

        # full-res float target for the crush stage (lazy: crush defaults 0)
        self.tex_crush = None
        self.fbo_crush = None

        # dither working buffers, sized on first use (dither_size can change)
        self._work_size = None
        self.tex_work = None         # RGBA32F downsample target
        self.fbo_work = None
        self.tex_workd = None        # RGBA32F ordered-dither target
        self.fbo_workd = None
        self.tex_workup = None       # RGB32F upload for CPU error diffusion
        self._mats = {}              # threshold textures by dither mode
        self._luts = {}              # (levels, gamma) -> (lin, enc) textures

        # constant sampler units
        self.p_driftchroma["u_src"].value = 0
        self.p_driftchroma["u_drift"].value = 1
        self.p_blit["u_src"].value = 0
        self.p_crush["u_src"].value = 0
        self.p_down["u_src"].value = 0
        self.p_dither["u_src"].value = 0
        self.p_dither["u_mat"].value = 1
        self.p_dither["u_lin_lut"].value = 2
        self.p_dither["u_enc_lut"].value = 3
        self.p_compose["u_src"].value = 0

    # ----- helpers -----
    def _mat_tex(self, mode):
        tex = self._mats.get(mode)
        if tex is None:
            mat = (_bayer_matrix(4) if mode == "bayer"
                   else _blue_noise_matrix()).astype(np.float32)
            mh, mw = mat.shape
            tex = _nearest(self.ctx.texture((mw, mh), 1,
                                            np.ascontiguousarray(mat),
                                            dtype="f4"))
            self._mats[mode] = tex
        return tex

    def _ensure_crush(self):
        if self.tex_crush is None:
            self.tex_crush = _nearest(
                self.ctx.texture((self.w, self.h), 4, dtype="f4"))
            self.fbo_crush = self.ctx.framebuffer(
                color_attachments=[self.tex_crush])

    def _ensure_work(self, dw, dh):
        if self._work_size == (dw, dh):
            return
        for obj in (self.tex_work, self.fbo_work, self.tex_workd,
                    self.fbo_workd, self.tex_workup):
            if obj is not None:
                obj.release()
        ctx = self.ctx
        self.tex_work = _nearest(ctx.texture((dw, dh), 4, dtype="f4"))
        self.fbo_work = ctx.framebuffer(color_attachments=[self.tex_work])
        self.tex_workd = _nearest(ctx.texture((dw, dh), 4, dtype="f4"))
        self.fbo_workd = ctx.framebuffer(color_attachments=[self.tex_workd])
        self.tex_workup = _nearest(ctx.texture((dw, dh), 3, dtype="f4"))
        self._work_size = (dw, dh)

    def _dither_luts(self, levels, gamma):
        """The linearize and level-encode LUTs of dtouch.dither, as 256x1
        R32F textures — uploading the CPU's own float32 tables makes the
        ordered pass bit-exact against the numpy path (in-shader pow
        differed in the last ulp and flipped rare threshold boundaries)."""
        key = (levels, gamma)
        pair = self._luts.get(key)
        if pair is None:
            lin = _srgb2lin_lut(float(levels))            # k -> lin(k/255)*levels
            enc = np.zeros(256, dtype=np.float32)         # _encode_levels' table
            enc[:levels + 1] = (np.arange(levels + 1, dtype=np.float32)
                                / float(levels))
            if gamma:
                enc[:levels + 1] = linear_to_srgb(enc[:levels + 1])
                enc[levels] = 1.0
            pair = tuple(
                _nearest(self.ctx.texture((256, 1), 1,
                                          np.ascontiguousarray(a), dtype="f4"))
                for a in (lin, enc))
            self._luts[key] = pair
        return pair

    # ----- the rack -----
    def run(self, cb, plan, src_tex):
        """Apply one frame's plan + cb settings to src_tex (RGBA8, out res).

        Mirrors CircuitBent.apply_plan stage for stage; leaves the final
        frame in self.fbo_out. Caller must hold the context binding."""
        gl = self._gl
        ctx = self.ctx
        w, h = self.w, self.h

        # stages 1+2: scan drift + chroma shift, one pass src -> A
        p = self.p_driftchroma
        if plan.drift_px is not None:
            self.tex_drift.write(
                np.ascontiguousarray(plan.drift_px.astype(np.float32)))
            p["u_use_drift"].value = 1
        else:
            p["u_use_drift"].value = 0
        p["u_shift_r"].value = int(plan.shift_r)
        p["u_shift_b"].value = int(plan.shift_b)
        p["u_w"].value = w
        self.fbo_a.use()
        src_tex.use(0)
        self.tex_drift.use(1)
        self.vao_driftchroma.render(gl.TRIANGLES, vertices=3)

        # stage 3: glitch tile capture (freeze from the post-chroma frame)
        # and replay. The tile lives in a texture, keyed to its glitch event
        # (plan.glitch_seq) so a stale tile from before an engine switch is
        # never replayed — the numpy backend keeps its own array the same way.
        if plan.glitch_capture is not None:
            sy, sx, th, tw = plan.glitch_capture
            pb = self.p_blit
            pb["u_src_off"].value = (sx, sy)
            pb["u_dst_off"].value = (0, 0)
            self.fbo_tile.use()
            self.fbo_tile.viewport = (0, 0, tw, th)
            self.tex_a.use(0)
            self.vao_blit.render(gl.TRIANGLES, vertices=3)
            self.fbo_tile.viewport = (0, 0, self.tex_tile.width,
                                      self.tex_tile.height)
            self._tile_seq = plan.glitch_seq
        if plan.glitch_rect is not None and self._tile_seq == plan.glitch_seq:
            y0, x0, ah, aw = plan.glitch_rect
            pb = self.p_blit
            pb["u_src_off"].value = (0, 0)
            pb["u_dst_off"].value = (x0, y0)
            self.fbo_a.use()
            self.fbo_a.viewport = (x0, y0, aw, ah)
            self.tex_tile.use(0)
            self.vao_blit.render(gl.TRIANGLES, vertices=3)
            self.fbo_a.viewport = (0, 0, w, h)

        cur_tex, cur_size = self.tex_a, (w, h)

        # stage 4: bit crush, into a float target (k/levels is off-lattice)
        if cb.bit_crush > 0:
            self._ensure_crush()
            levels = float((1 << int(cb.bit_crush)) - 1)
            self.p_crush["u_levels"].value = levels
            self.fbo_crush.use()
            cur_tex.use(0)
            self.vao_crush.render(gl.TRIANGLES, vertices=3)
            cur_tex = self.tex_crush

        # stage 5: dither at the rack's working res
        dithered_tex, dithered_size = None, None
        if cb.dither_mode:
            ds = cb.dither_size
            if ds and ds < h:
                dh = int(ds)
                dw = max(1, int(w * dh / h))
            else:
                dw, dh = w, h
            self._ensure_work(dw, dh)
            self.p_down["u_src_size"].value = cur_size
            self.p_down["u_scale"].value = (cur_size[0] / dw,
                                            cur_size[1] / dh)
            self.fbo_work.use()
            cur_tex.use(0)
            self.vao_down.render(gl.TRIANGLES, vertices=3)

            bits = int(cb.dither_bits)
            levels = (1 << bits) - 1
            gamma = bool(cb.dither_gamma)
            if cb.dither_mode in ("bayer", "blue"):
                invert = cb.dither_invert
                if invert == "auto":
                    # the CPU decides on the working image's mean in the
                    # working domain — same helpers, same image, small read
                    img = self._read_work(dw, dh)
                    if gamma:
                        work = _lut_apply(_to_u8(img), _srgb2lin_lut(float(levels)))
                        invert = float(work.mean()) / levels < _MID_GREY_LINEAR
                    else:
                        invert = float(np.clip(img, 0.0, 1.0).mean()) < 0.5
                pd = self.p_dither
                mat = self._mat_tex(cb.dither_mode)
                lin_lut, enc_lut = self._dither_luts(levels, gamma)
                pd["u_mat_size"].value = (mat.width, mat.height)
                pd["u_levels"].value = float(levels)
                pd["u_invert"].value = 1 if invert else 0
                pd["u_gamma"].value = 1 if gamma else 0
                self.fbo_workd.use()
                self.tex_work.use(0)
                mat.use(1)
                lin_lut.use(2)
                enc_lut.use(3)
                self.vao_dither.render(gl.TRIANGLES, vertices=3)
                dithered_tex, dithered_size = self.tex_workd, (dw, dh)
            elif cb.dither_mode in ("fs", "riemersma"):
                # error diffusion is serial — same dtouch.dither code, on
                # the CPU, at the (small) working res, then uploaded back
                img = self._read_work(dw, dh)
                out = cb._dither_array(img)
                self.tex_workup.write(np.ascontiguousarray(out, np.float32))
                dithered_tex, dithered_size = self.tex_workup, (dw, dh)
            # unknown modes fall through undithered, like _dither_array

        # stage 6: nearest upscale + scanlines + clip + truncating u8 encode
        pc = self.p_compose
        if dithered_tex is not None:
            src, ssize = dithered_tex, dithered_size
        else:
            src, ssize = cur_tex, cur_size
        pc["u_src_size"].value = ssize
        pc["u_scale"].value = (ssize[0] / w, ssize[1] / h)
        if cb.scanlines:
            sp = max(2, round(h / SCANLINE_ROWS))
            pc["u_scan_p"].value = sp
            pc["u_scan_half"].value = sp // 2
            pc["u_scan_dim"].value = 1.0 - float(cb.scanline_strength)
        else:
            pc["u_scan_p"].value = 0
            pc["u_scan_half"].value = 0
            pc["u_scan_dim"].value = 1.0
        self.fbo_out.use()
        src.use(0)
        self.vao_compose.render(gl.TRIANGLES, vertices=3)

    def _read_work(self, dw, dh):
        """The working-res image as float32 (dh, dw, 3) — the one small
        mid-frame readback (auto-bias decision / error-diffusion input)."""
        raw = self.fbo_work.read(components=3, dtype="f4")
        return np.frombuffer(raw, np.float32).reshape(dh, dw, 3)

    def read(self):
        """The composed frame as uint8 RGB (h, w, 3) — THE readback."""
        raw = self.fbo_out.read(components=3)
        return np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3).copy()

    def release(self):
        """Release GL objects (context stays the caller's); idempotent."""
        objs = [self.p_driftchroma, self.p_blit, self.p_crush, self.p_down,
                self.p_dither, self.p_compose, self.tri_vbo,
                self.vao_driftchroma, self.vao_blit, self.vao_crush,
                self.vao_down, self.vao_dither, self.vao_compose,
                self.tex_a, self.tex_out, self.fbo_a, self.fbo_out,
                self.tex_tile, self.fbo_tile, self.tex_drift,
                self.tex_crush, self.fbo_crush, self.tex_work, self.fbo_work,
                self.tex_workd, self.fbo_workd, self.tex_workup,
                *self._mats.values(),
                *(t for pair in self._luts.values() for t in pair)]
        self._mats = {}
        self._luts = {}
        for o in objs:
            if o is not None:
                try:
                    o.release()
                except Exception:      # noqa: BLE001 — release is best-effort
                    pass


class PhysarumOutGL:
    """Physarum's output stages on the field's context, feeding SignalRackGL.

    colorize (palette LUT / video-lit) at grid res -> bilinear upscale to
    output res -> optional video-background screen blend. Mirrors the CPU
    tail of PhysarumMode.step (dtouch.modes.physarum._colorize + cv2.resize
    + composite_video_bg) so the two engines render the same picture."""

    def __init__(self, ctx, gw, gh, out_w, out_h):
        import moderngl
        self._gl = moderngl
        self.ctx = ctx
        self.gw, self.gh = gw, gh
        self.out_w, self.out_h = out_w, out_h

        vs = load_rack_shader("fullscreen.vert")
        self.p_colorize = ctx.program(
            vertex_shader=vs, fragment_shader=load_rack_shader("colorize.frag"))
        self.p_upscale = ctx.program(
            vertex_shader=vs, fragment_shader=load_rack_shader("upscale.frag"))
        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.tri_vbo = ctx.buffer(tri.tobytes())
        self.vao_colorize = ctx.vertex_array(
            self.p_colorize, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_upscale = ctx.vertex_array(
            self.p_upscale, [(self.tri_vbo, "2f", "in_vert")])

        self.tex_color = _nearest(ctx.texture((gw, gh), 4, dtype="f1"))
        self.fbo_color = ctx.framebuffer(color_attachments=[self.tex_color])
        self.tex_src = _nearest(ctx.texture((out_w, out_h), 4, dtype="f1"))
        self.fbo_src = ctx.framebuffer(color_attachments=[self.tex_src])

        self.tex_lut = _nearest(ctx.texture((256, 1), 3, dtype="f1"))
        self._lut_id = None
        self.tex_video = None        # small camera (video palette), per shape
        self.tex_cam = None          # camera frame (video bg), per shape

        self.p_colorize["u_lum"].value = 0
        self.p_colorize["u_lut"].value = 1
        self.p_colorize["u_video"].value = 2
        self.p_upscale["u_src"].value = 0
        self.p_upscale["u_cam"].value = 1

        self.rack = SignalRackGL(ctx, out_w, out_h)

    def _upload(self, attr, arr):
        """(Re)allocate + fill a u8 RGB texture from an (H, W, 3) array."""
        h, w = arr.shape[:2]
        tex = getattr(self, attr)
        if tex is None or (tex.width, tex.height) != (w, h):
            if tex is not None:
                tex.release()
            tex = _nearest(self.ctx.texture((w, h), 3, dtype="f1"))
            setattr(self, attr, tex)
        tex.write(np.ascontiguousarray(arr))
        return tex

    def compose(self, lum_tex, lut, video_small, cam, mix):
        """Render the mode's composed output frame into self.tex_src.

        lum_tex: the field's R8 luminance texture (grid res). lut: 256x3
        uint8 RGB palette, or None for the "video" palette with video_small
        as the small camera RGB frame. cam + mix: the video-background
        screen blend (mix <= 0 skips it). Caller binds the context."""
        gl = self._gl
        pc = self.p_colorize
        if lut is not None:
            if self._lut_id != id(lut):
                self.tex_lut.write(np.ascontiguousarray(lut, np.uint8))
                self._lut_id = id(lut)
            pc["u_mode"].value = 0
            vid = None
        else:
            vid = self._upload("tex_video", video_small)
            pc["u_mode"].value = 1
            pc["u_vsize"].value = (vid.width, vid.height)
            pc["u_vscale"].value = (vid.width / self.gw, vid.height / self.gh)
        self.fbo_color.use()
        lum_tex.use(0)
        self.tex_lut.use(1)
        (vid if vid is not None else self.tex_lut).use(2)
        self.vao_colorize.render(gl.TRIANGLES, vertices=3)

        pu = self.p_upscale
        pu["u_src_size"].value = (self.gw, self.gh)
        pu["u_src_scale"].value = (self.gw / self.out_w, self.gh / self.out_h)
        if cam is not None and mix > 0.0:
            ct = self._upload("tex_cam", cam)
            pu["u_cam_size"].value = (ct.width, ct.height)
            pu["u_cam_scale"].value = (ct.width / self.out_w,
                                       ct.height / self.out_h)
            pu["u_mix"].value = float(mix)
            cam_tex = ct
        else:
            pu["u_mix"].value = 0.0
            cam_tex = self.tex_color
        self.fbo_src.use()
        self.tex_color.use(0)
        cam_tex.use(1)
        self.vao_upscale.render(gl.TRIANGLES, vertices=3)
        return self.tex_src

    def release(self):
        """Release GL objects (not the shared context); idempotent."""
        self.rack.release()
        objs = [self.p_colorize, self.p_upscale, self.tri_vbo,
                self.vao_colorize, self.vao_upscale, self.tex_color,
                self.fbo_color, self.tex_src, self.fbo_src, self.tex_lut,
                self.tex_video, self.tex_cam]
        for o in objs:
            if o is not None:
                try:
                    o.release()
                except Exception:      # noqa: BLE001 — release is best-effort
                    pass
