"""Fractal veins — the browser demo moved into the shared physarum unit.

The three trail channels become three scales of one organism (trunks, veins,
threads) when `u_fractal` > 0. Pinned here:

- **Amount 0 is the stock engine, bit for bit.** update.frag, deposit.vert,
  blur.frag and stats.frag each gate every fractal term on `u_fractal`; run
  on the same inputs for every look, each must match a frozen copy of the
  pre-fractal file (tests/fixtures/physarum/pre_fractal/) exactly, both with
  u_fractal set to 0 and never set (a host that never learns about it).
  tonemap.frag is untouched: the fractal picture is its own pass.
- **The shared files ARE the browser demo.** At amounts above 0 each merged
  pass matches the browser's demo copy (tests/fixtures/physarum/web_fractal/,
  frozen from cyborg-garden-site fractal-veins 1814856) bit for bit, and
  tonemap_fractal.frag / compose.frag are its code line for line, but for
  compose's stock side (COMPOSE_DELTA: it morphs out of the lit stock
  picture, where the demo dropped the relief at the first nudge).
- **Perceptible, in the right direction.** At each look's shipped amount and
  at the slider's 0.3, over six seeds, the fractal picture carries more fine
  structure than stock on every seed (normalised against the same seed's
  stock arm).
- The tables reach looks.json; the bloom schedule places the zones exactly
  where the browser does; H steps flat, relief, relief + fractal on the GPU
  and skips the third step on the CPU fallback, saying so; 4K cost on the GL
  path, by GPU timer queries.
"""
import functools
import json
import os
import re

import cv2
import numpy as np
import pytest

from dtouch.modes.physarum import (FRACTAL_CPU, FRACTAL_DEFAULT, LOOK_FRACTAL,
                                   PALETTES_PH, PhysarumMode)
from dtouch.panelspec import apply_look, visible
from dtouch.physarum import (BLOOM, FRACTAL, FRACTAL_MIX_FULL, FRACTAL_NL,
                             POINT_NAMES, POINTS, PX_REF, SPATIAL_REGIMES,
                             bloom_zones)
from dtouch.physarum_gl import (SHADER_DIR, PhysarumFieldGL, PhysarumGLUnavailable,
                                fractal_render_size, level_norms, load_shader)
from dtouch.physarum_looks import LOOKS_PATH
from dtouch.shell import Host

from test_physarum_gl import _grown, _StubHost
from test_shell import SyntheticSource

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "physarum")
PRE = os.path.join(FIX, "pre_fractal")
WEB = os.path.join(FIX, "web_fractal")
SEEDS = range(6)                 # n >= 6 before a first green is trusted
LOOKS = list(PhysarumMode.BUILTIN)


def _src(path):
    with open(path, encoding="utf-8") as fh:
        return "#version 330 core\n#line 1\n" + fh.read()


def _code(path):
    """A shader's code with comments and blank lines dropped."""
    with open(path, encoding="utf-8") as fh:
        lines = [l.split("//", 1)[0].rstrip() for l in fh.read().splitlines()]
    return [l for l in lines if l.strip()]


# ---------- a standalone pass harness ----------

class _Passes:
    """Runs single physarum passes from arbitrary sources on one standalone
    context, into RGBA32F targets, and reads the float bits back."""

    def __init__(self):
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception as e:                    # noqa: BLE001
            pytest.skip(f"no GL context available (CI): {e}")
        self._gl = moderngl
        self.tri = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.fs = load_shader("fullscreen.vert")

    def tex(self, a):
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            a = a[..., None]
        h, w, c = a.shape
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[..., :c] = a
        t = self.ctx.texture((w, h), 4, rgba.tobytes(), dtype="f4")
        t.filter = (self._gl.NEAREST, self._gl.NEAREST)
        return t

    def _uniforms(self, p, uniforms, textures):
        for i, (name, t) in enumerate(textures.items()):
            if p.get(name, None) is not None:
                p[name].value = i
            t.use(i)
        for name, v in uniforms.items():
            if p.get(name, None) is not None:
                p[name].value = v

    def frag(self, src, size, textures, uniforms):
        p = self.ctx.program(vertex_shader=self.fs, fragment_shader=src)
        vao = self.ctx.vertex_array(p, [(self.tri, "2f", "in_vert")])
        out = self.ctx.texture(size, 4, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[out])
        try:
            fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self._uniforms(p, uniforms, textures)
            vao.render(self._gl.TRIANGLES, vertices=3)
            raw = fbo.read(components=4, dtype="f4")
        finally:
            for o in (vao, p, out, fbo):
                o.release()
        return np.frombuffer(raw, np.float32).reshape(size[1], size[0], 4).copy()

    def points(self, vsrc, n, size, textures, uniforms):
        p = self.ctx.program(vertex_shader=vsrc, fragment_shader=load_shader("deposit.frag"))
        vao = self.ctx.vertex_array(p, [])
        out = self.ctx.texture(size, 4, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[out])
        gl = self._gl
        try:
            fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.enable(gl.BLEND)
            self.ctx.blend_func = (gl.ONE, gl.ONE)
            self._uniforms(p, uniforms, textures)
            vao.render(gl.POINTS, vertices=n)
            self.ctx.disable(gl.BLEND)
            raw = fbo.read(components=4, dtype="f4")
        finally:
            for o in (vao, p, out, fbo):
                o.release()
        return np.frombuffer(raw, np.float32).reshape(size[1], size[0], 4).copy()

    def release(self):
        self.ctx.release()


@pytest.fixture(scope="module")
def passes():
    p = _Passes()
    yield p
    p.release()


AW, NAG = 2048, 8192


@functools.lru_cache(maxsize=None)
def _inputs(look, seed):
    """Grown trail + laid (the CPU field, per look and seed), a blob matte and
    luma, a keep map with both signs, and an agent texture of NAG agents
    over all three species/levels."""
    tr, la, ex = _grown(look, seed)
    gh, gw = tr.shape[:2]
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    gray = np.exp(-((xx - gw * 0.4) ** 2 + (yy - gh * 0.5) ** 2) / (2 * 25.0 ** 2))
    matte = np.clip(gray * 1.4 - 0.2, 0.0, 1.0)
    keep = np.clip(np.sin(xx / 9.0) * np.cos(yy / 7.0) * 1.3, -1.0, 1.0)
    rng = np.random.default_rng(100 + seed)
    ag = np.zeros((NAG // AW, AW, 4), np.float32)
    ag[..., 0] = rng.uniform(0, gw, ag.shape[:2])
    ag[..., 1] = rng.uniform(0, gh, ag.shape[:2])
    ag[..., 2] = rng.uniform(0, 2 * np.pi, ag.shape[:2])
    ag[..., 3] = rng.integers(0, 3, ag.shape[:2])
    norm = float(np.percentile(tr.sum(2), 95.0))
    return tr, la, ex, gray.astype(np.float32), matte.astype(np.float32), \
        keep.astype(np.float32), ag, norm


def _stock_uniforms(look, seed, wmax=True):
    """Every stock uniform of every sim pass, with each optional branch on
    (hetero, satcap, jitter, mosaic, ballistic, reseed, keep, sharpen)."""
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    gh, gw = tr.shape[:2]
    L = PhysarumMode.BUILTIN[look]
    a, b = POINTS[L["point_bg"]], POINTS[L["point_fg"]]
    reg = [SPATIAL_REGIMES[k] for k in SPATIAL_REGIMES]
    return {
        "u_grid": (gw, gh), "u_aw": AW,
        "u_sense": (a["sense"], b["sense"]), "u_spread": (a["spread"], b["spread"]),
        "u_turn": (a["turn"], b["turn"]), "u_step": (a["step"], b["step"]),
        "u_gain": 1.1, "u_food": L["food"] * norm, "u_reseed": 0.2,
        "u_wmax": float(matte.max()) if wmax else 0.0, "u_salt": 12345 + seed,
        "u_satcap": 1.2 * norm, "u_jitter": 0.12, "u_hetero": 0.6,
        "u_cross": 0.5, "u_nspec": 3.0, "u_mosaic": 0.6, "u_zones": 7.0,
        "u_sense_max": 0.1 * min(gw, gh), "u_ballistic": 0.2, "u_time": 3.25,
        "u_zreg": [(r["sense"], r["turn"], r["spread"], r["step"]) for r in reg],
        "u_zcross": [r["cross"] for r in reg],
        "u_deposit": (a["deposit"], b["deposit"]),
    }


def _fractal_uniforms(look, seed, fa):
    """The fractal uniforms exactly as PhysarumFieldGL sets them."""
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    gh, gw = tr.shape[:2]
    F, px = FRACTAL, gw / PX_REF
    m = lambda v: 1.0 + (v - 1.0) * fa        # noqa: E731
    return {
        "u_fractal": fa, "u_lvlScale": (F["scale"], F["angle"], 1.0),
        "u_inv_norm": 1.0 / norm, "u_flank": F["flank"], "u_flankAt": F["flankAt"] * norm,
        "u_shun": F["shun"] * fa, "u_dieback": F["dieback"] * fa * 20,  # x20: exercise die-back
        "u_bodyFine": m(F["bodyFine"]), "u_roomCull": F["roomCull"],
        "u_fineMin": (F["fineMin"][0] * px * fa, F["fineMin"][1] * px * fa),
        "u_fineMax": (F["fineMax"][0] * px, F["fineMax"][1] * px),
        "u_strideK": tuple(F["stride"]), "u_calm": F["calm"],
        "u_fineAngle": tuple(F["fineAngle"]), "u_trunkAngle": tuple(F["trunkAngle"]),
        "u_bloom": bloom_zones(40.0 + seed, gw, gh, fa),
        "u_bloomFine": F["bloomFine"], "u_bloomPull": 0.05,   # exercise the pull
        "u_lvlFood": tuple(m(v) for v in F["food"]),
        "u_lvlDep": tuple(m(v) for v in F["dep"]),
        "u_fineField": tuple(m(v) for v in F["fineField"]), "u_bloomDep": F["bloomDep"],
        "u_diffC": tuple(m(v) for v in F["diff"]), "u_sharpC": tuple(m(v) for v in F["sharp"]),
        "u_decayC": (m(F["decay"][0]) * 0.8, m(F["decay"][1]), m(F["decay"][2])),
    }


def _run_update(ps, src, look, seed, extra, wmax=True):
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    u = dict(_stock_uniforms(look, seed, wmax), **extra)
    return ps.frag(src, (AW, NAG // AW),
                   {"u_agents": ps.tex(ag), "u_trail": ps.tex(tr),
                    "u_matte": ps.tex(matte), "u_gray": ps.tex(gray)}, u)


def _run_deposit(ps, src, look, seed, extra):
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    gh, gw = tr.shape[:2]
    u = dict(_stock_uniforms(look, seed), **extra)
    return ps.points(src, NAG, (gw, gh), {"u_agents": ps.tex(ag), "u_matte": ps.tex(matte)}, u)


def _run_blur(ps, src, look, seed, extra, axis):
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    gh, gw = tr.shape[:2]
    u = dict(_stock_uniforms(look, seed), u_radius=2, u_wide=6, u_scale=1 / 5.0,
             u_sharpen=0.09, **extra)
    if axis == "H":
        u.update(u_use_add=1, u_use_keep=0, u_dir=(1, 0), u_decay=1.0)
    else:
        u.update(u_use_add=0, u_use_keep=1, u_dir=(0, 1), u_decay=PhysarumMode.BUILTIN[look]["decay"])
    return ps.frag(src, (gw, gh), {"u_src": ps.tex(tr), "u_add": ps.tex(la), "u_keep": ps.tex(keep)}, u)


def _run_stats(ps, src, look, seed, extra):
    tr, la, ex, gray, matte, keep, ag, norm = _inputs(look, seed)
    gh, gw = tr.shape[:2]
    u = dict(_stock_uniforms(look, seed), u_stride=4, **extra)
    return ps.frag(src, ((gw + 3) // 4, (gh + 3) // 4), {"u_trail": ps.tex(tr), "u_laid": ps.tex(la)}, u)


def _bits(a):
    return a.view(np.uint32)


# ---------- the frozen references ----------

def test_frozen_references_are_what_they_claim():
    """The bit-exact tests are only as good as their references: the
    pre-fractal copies carry no fractal term, and the browser copies are the
    demo files (each fractal-only, each marked as the demo)."""
    for name in ("update.frag", "deposit.vert", "blur.frag", "stats.frag", "tonemap.frag"):
        with open(os.path.join(PRE, name), encoding="utf-8") as fh:
            assert "fractal" not in fh.read().lower(), name
    with open(os.path.join(PRE, "tonemap.frag"), encoding="utf-8") as fh:
        pre_tm = fh.read()
    with open(os.path.join(SHADER_DIR, "tonemap.frag"), encoding="utf-8") as fh:
        assert fh.read() == pre_tm, "tonemap.frag changed; the fractal picture is its own pass"
    for name in sorted(os.listdir(WEB)):
        with open(os.path.join(WEB, name), encoding="utf-8") as fh:
            assert fh.read().startswith("// DEMO, WEB ONLY"), name
    assert sorted(os.listdir(WEB)) == ["blur_fractal.frag", "compose_fractal.frag",
                                       "deposit_fractal.vert", "stats_fractal.frag",
                                       "tonemap_fractal.frag", "update_fractal.frag"]


# ---------- amount 0: bit for bit ----------

@pytest.mark.parametrize("look", LOOKS)
def test_amount_zero_is_the_pre_fractal_engine_bit_for_bit(passes, look):
    """For every look (two seeds, both respawn paths, both blur axes): the
    merged update / deposit / blur / stats at u_fractal 0, and with
    u_fractal never set, equal the frozen pre-fractal files bit for bit."""
    merged = {n: load_shader(n) for n in ("update.frag", "deposit.vert", "blur.frag", "stats.frag")}
    pre = {n: _src(os.path.join(PRE, n)) for n in merged}
    zero = {"u_fractal": 0.0}
    for seed in (0, 1):
        for wmax in (True, False):
            ref = _run_update(passes, pre["update.frag"], look, seed, {}, wmax)
            assert np.isfinite(ref).all()
            moved = np.abs(ref[..., :2] - _inputs(look, seed)[6][..., :2]).max()
            assert moved > 0.5, "the update did not move anyone: nothing is tested"
            for extra in (zero, {}):
                got = _run_update(passes, merged["update.frag"], look, seed, extra, wmax)
                assert np.array_equal(_bits(ref), _bits(got)), ("update", look, seed, wmax, extra)
        ref = _run_deposit(passes, pre["deposit.vert"], look, seed, {})
        assert ref[..., :3].sum() > 0
        for extra in (zero, {}):
            got = _run_deposit(passes, merged["deposit.vert"], look, seed, extra)
            assert np.array_equal(_bits(ref), _bits(got)), ("deposit", look, seed, extra)
        for axis in ("H", "V"):
            ref = _run_blur(passes, pre["blur.frag"], look, seed, {}, axis)
            for extra in (zero, {}):
                got = _run_blur(passes, merged["blur.frag"], look, seed, extra, axis)
                assert np.array_equal(_bits(ref), _bits(got)), ("blur", axis, look, seed, extra)
        ref = _run_stats(passes, pre["stats.frag"], look, seed, {})
        for extra in (zero, {}):
            got = _run_stats(passes, merged["stats.frag"], look, seed, extra)
            assert np.array_equal(_bits(ref), _bits(got)), ("stats", look, seed, extra)


def test_every_fractal_term_sits_behind_the_amount_gate():
    """Other drivers (ANGLE, WebGL) promise less than this one about x * 1.0
    or mix(a, b, 0), so the stock arithmetic must not merely come out equal
    here: each merged file keeps the stock expression verbatim on the
    u_fractal <= 0 side. Pinned in the source."""
    def body(name):
        return "\n".join(_code(os.path.join(SHADER_DIR, name)))
    up = body("update.frag")
    assert "if (u_fractal > 0.0) ladder(sp, t, p, sense, spread, turn, stp);" in up
    assert "sense *= 1.0 + (m - 1.0) * u_hetero;" in up
    assert re.search(r"if \(u_fractal > 0\.0\) \{\s*fc = foodFractal", up)
    assert re.search(r"if \(u_fractal > 0\.0\) \{\s*f_agent = vec4\(fractalReseed", up)
    assert "v_dep = mix(u_deposit.x, u_deposit.y, texelFetch(u_matte, c, 0).r);" in body("deposit.vert")
    bl = body("blur.frag")
    assert bl.index("if (u_fractal > 0.0) {") < bl.index("vec3 out3 = acc * u_scale;")
    assert "f_color = vec4(t.r + t.g + t.b, l.r + l.g + l.b, 0.0, 1.0);" in body("stats.frag")


# ---------- amount > 0: the browser demo, bit for bit ----------

@pytest.mark.parametrize("fa", [0.3, 1.0])
def test_merged_passes_equal_the_browser_demo_copies(passes, fa):
    """The shared files at amount fa run exactly the browser demo's copies:
    the web can switch to them without a visible change."""
    pairs = {"update.frag": "update_fractal.frag", "deposit.vert": "deposit_fractal.vert",
             "blur.frag": "blur_fractal.frag", "stats.frag": "stats_fractal.frag"}
    for look in ("veinwork", "ghost"):
        seed = 2
        fx = _fractal_uniforms(look, seed, fa)
        m = {n: load_shader(n) for n in pairs}
        w = {n: _src(os.path.join(WEB, f)) for n, f in pairs.items()}
        for wmax in (True, False):
            a = _run_update(passes, m["update.frag"], look, seed, fx, wmax)
            b = _run_update(passes, w["update.frag"], look, seed, fx, wmax)
            assert np.array_equal(_bits(a), _bits(b)), ("update", look, wmax)
        a = _run_deposit(passes, m["deposit.vert"], look, seed, fx)
        b = _run_deposit(passes, w["deposit.vert"], look, seed, fx)
        assert np.array_equal(_bits(a), _bits(b)), ("deposit", look)
        for axis in ("H", "V"):
            a = _run_blur(passes, m["blur.frag"], look, seed, fx, axis)
            b = _run_blur(passes, w["blur.frag"], look, seed, fx, axis)
            assert np.array_equal(_bits(a), _bits(b)), ("blur", axis, look)
        a = _run_stats(passes, m["stats.frag"], look, seed, fx)
        b = _run_stats(passes, w["stats.frag"], look, seed, fx)
        assert np.array_equal(_bits(a), _bits(b)), ("stats", look)
        # and the gate is live: the fractal step is not the stock step
        stock = _run_update(passes, m["update.frag"], look, seed, {"u_fractal": 0.0})
        assert not np.array_equal(_bits(stock), _bits(_run_update(
            passes, m["update.frag"], look, seed, fx)))


# compose.frag's one departure from the demo: the u_mix < 1 side reads the
# stock picture itself (u_stock, tonemap.frag's output, relief included)
# instead of rebuilding the FLAT sum from the trail, which dropped the
# relief at the first nudge of the amount (see
# test_small_amounts_morph_out_of_the_lit_stock_picture). Web -> desktop.
COMPOSE_DELTA = {
    "uniform sampler2D u_trail;": "uniform sampler2D u_stock;",
    "uniform sampler2D u_laid;": None,
    "uniform float u_inv_norm;": None,
    "uniform float u_inv_gnorm;": None,
    "uniform float u_grain;": None,
    "uniform float u_exposure;": None,
    "vec3 trailLin(vec2 gp) {": "float stockLin(vec2 gp) {",
    "    vec3 a = texelFetch(u_trail, c0, 0).rgb + u_grain * u_inv_gnorm / u_inv_norm"
    " * texelFetch(u_laid, c0, 0).rgb;": "    float a = texelFetch(u_stock, c0, 0).r;",
    "    vec3 b = texelFetch(u_trail, ivec2(c1.x, c0.y), 0).rgb + u_grain * u_inv_gnorm"
    " / u_inv_norm * texelFetch(u_laid, ivec2(c1.x, c0.y), 0).rgb;":
        "    float b = texelFetch(u_stock, ivec2(c1.x, c0.y), 0).r;",
    "    vec3 c = texelFetch(u_trail, ivec2(c0.x, c1.y), 0).rgb + u_grain * u_inv_gnorm"
    " / u_inv_norm * texelFetch(u_laid, ivec2(c0.x, c1.y), 0).rgb;":
        "    float c = texelFetch(u_stock, ivec2(c0.x, c1.y), 0).r;",
    "    vec3 d = texelFetch(u_trail, c1, 0).rgb + u_grain * u_inv_gnorm / u_inv_norm"
    " * texelFetch(u_laid, c1, 0).rgb;": "    float d = texelFetch(u_stock, c1, 0).r;",
    "        vec3 t = trailLin(gp);": None,
    "        float lumS = 1.0 - exp(-u_exposure * (t.r + t.g + t.b) * u_inv_norm);": None,
    "        outL = mix(lumS, outL, u_mix);": "        outL = mix(stockLin(gp), outL, u_mix);",
}


def test_picture_passes_are_the_browser_demo_code():
    """tonemap_fractal.frag and compose.frag are the demo's code line for
    line (headers aside), but for two recorded departures: tonemap_fractal
    drops u_fractal, which the shader never read, and compose's stock side
    is COMPOSE_DELTA, every line of it."""
    dead = "uniform float u_fractal;"
    web = [l for l in _code(os.path.join(WEB, "tonemap_fractal.frag")) if not l.startswith(dead)]
    assert _code(os.path.join(SHADER_DIR, "tonemap_fractal.frag")) == web
    web = _code(os.path.join(WEB, "compose_fractal.frag"))
    assert set(COMPOSE_DELTA) <= set(web), "the recorded delta no longer matches the demo"
    want = [COMPOSE_DELTA.get(l, l) for l in web]
    assert _code(os.path.join(SHADER_DIR, "compose.frag")) == [l for l in want if l is not None]


def test_small_amounts_morph_out_of_the_lit_stock_picture():
    """compose.frag at u_mix < 1 blends toward the stock picture. It rebuilt
    that side from the trail as the FLAT sum, so with the relief on (every
    look ships depth 0.5-0.9) the first nudge of the amount dropped the
    lighting: measured |d| 0.070 at amount 0.001, the picture ~26% brighter,
    against 0.000 with depth 0. Same trail, same size, now continuous."""
    gw, gh = 192, 108
    yy, xx = np.mgrid[0:gh, 0:gw]
    gray = np.exp(-((xx - gw * 0.4) ** 2 + (yy - gh * 0.5) ** 2)
                  / (2 * (gw / 8) ** 2)).astype(np.float32)
    matte = (gray > 0.4).astype(np.float32)
    f = _gl_field(n=12000, gw=gw, gh=gh, seed=3)
    try:
        f.depth, f.light, f.out_size = 0.9, (-0.5, 0.5, 0.7), (gw, gh)
        for _ in range(60):
            f.update(matte, gray)
        stock = f.luminance()
        assert stock.mean() > 0.05
        got = []
        for a in (0.001, 0.02):
            f.fractal = a
            f.bloom = bloom_zones(10.0, gw, gh, a)
            with f.ctx:
                f._stats(a)                    # this trail's levels, no step
            got.append(float(np.abs(f.luminance() - stock).mean()))
        assert got[0] < 0.003 and got[1] < 0.02, got
    finally:
        f.release()


# ---------- tables ----------

def test_looks_json_exports_the_fractal_tables():
    """The browser reads the tuning table, the host rules, the bloom schedule
    and every look's amount from looks.json instead of its own literals."""
    with open(LOOKS_PATH, encoding="utf-8") as fh:
        d = json.load(fh)
    fr = d["fractal"]
    assert fr["table"] == FRACTAL
    assert list(fr["table"]) == list(FRACTAL)          # key order kept for review
    assert fr["mix_full"] == FRACTAL_MIX_FULL
    assert fr["level_norms"] == FRACTAL_NL
    # every number the level rule runs on, its frame-to-frame smoothing too
    import inspect
    from dtouch.physarum import NORM_EMA
    assert fr["level_norms"]["ema"] == NORM_EMA == \
        inspect.signature(level_norms).parameters["ema"].default
    assert fr["bloom"] == BLOOM
    assert fr["px_ref"] == PX_REF == PhysarumMode.CPU_GRID[0]
    for name, look in d["builtin"].items():
        assert look["fractal"] == LOOK_FRACTAL[name] == PhysarumMode.BUILTIN[name]["fractal"]
        assert 0.0 < look["fractal"] <= 1.0
    assert d["defaults"]["fractal"] == FRACTAL_DEFAULT == 0.0


def test_bloom_zones_land_where_the_browser_puts_them():
    """bloom_zones against the browser's bloomU (physarum.js, computed with
    node from the same clocks): same doubles, same zones."""
    web = [[277.38418426137156, 549.2573266860733, 145.61780818849394, 0.4999999999999999,
            849.7955825171412, 126.28816338770059, 117.80956498150947, 0.5000000000000001],
           [242.864203484342, 528.1994064484934, 145.61780818849394, 0.6811953007101154,
            645.5018743930908, 235.5340052282797, 96.13338240171316, 0.04748738713499457],
           [792.659914731958, 797.8403508537134, 191.29738373490514, 0.0715258171054495,
            374.03785520952215, 414.4231230658411, 173.99362066942965, 0.1430663882707208],
           [185.76907873301235, 102.24948882813298, 52.5359741126548, 0.0035051004872410745,
            80.14336537163209, 148.3541142429248, 59.19965891344844, 0.7351563513052407]]
    cases = [(0, 1280, 736, 1, 0, 1), (17.3, 1280, 736, 0.7, 0.2, 2),
             (123.456, 1920, 1104, 1, 0.9, 3), (1000.1, 576, 324, 0.85, 0, 1)]
    for c, want in zip(cases, web):
        got = [v for z in bloom_zones(*c) for v in z]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9), c
    # each zone blooms fully and fades to nothing once a cycle (it moves
    # while invisible), and the amount scales it
    t = np.linspace(0.0, 400.0, 4001)
    s = np.array([[z[3] for z in bloom_zones(float(x), 640, 360, 1.0)] for x in t])
    assert (s.max(axis=0) > 0.99).all() and (s.min(axis=0) < 0.01).all()
    half = np.array([[z[3] for z in bloom_zones(float(x), 640, 360, 0.5)] for x in t])
    assert np.allclose(half, 0.5 * s)


def test_render_size_and_level_norms():
    """The drawing pass is the output within the pixel budget, never under
    the grid; level bright ends follow the occupied samples."""
    assert fractal_render_size(1280, 736, 1920, 1080) == (1920, 1080)
    w, h = fractal_render_size(1280, 736, 3840, 2160)
    assert w * h <= FRACTAL["renderPx"] * 1.001 and abs(w / h - 16 / 9) < 0.01
    assert fractal_render_size(1280, 736, 640, 360) == (1280, 736)
    rng = np.random.default_rng(0)
    s = np.zeros((40, 50, 4), np.float32)
    s[..., 2] = np.where(rng.random((40, 50)) < 0.1, 5.0, 0.0)   # sparse veins
    s[..., 0] = 20.0 + s[..., 2]
    nl = level_norms(s, 20.0)
    assert nl[0] == pytest.approx(20.0) and nl[1] == pytest.approx(5.0)
    assert nl[2] == pytest.approx(20.0 * FRACTAL_NL["floor"])    # empty: the floor


def _level_norms_by_sort(stats, norm, prev=None, ema=0.9):
    """The rule as first written (full sort + searchsorted): the reference
    level_norms' partition must reproduce exactly."""
    import math
    s = np.asarray(stats, np.float32).reshape(-1, 4)
    tot, g, b = s[:, 0], s[:, 2], s[:, 3]
    lv = (np.maximum(tot - g - b, 0.0), g, b)
    floor = max(1e-4, (norm if norm > 0 else 1.0) * FRACTAL_NL["floor"])
    out = []
    for k, a in enumerate(lv):
        a = np.sort(a)
        mx = float(a[-1])
        lo = 0
        if k > 0:
            lo = int(np.searchsorted(a, FRACTAL_NL["occ"] * mx, side="right"))
        occ = a.size - lo
        pct = FRACTAL_NL["pct"][0 if k == 0 else 1]
        v = float(a[lo + int(math.floor(occ * pct))]) if occ > FRACTAL_NL["min_occ"] else mx
        want = max(floor, v if math.isfinite(v) else floor)
        out.append(want if prev is None else prev[k] * ema + want * (1.0 - ema))
    return tuple(out)


def test_level_norms_partition_equals_the_sorted_rule():
    """level_norms takes its order statistics by np.partition (a full sort
    was 1.4-6.4 ms of CPU per frame); it must pick the very same elements,
    including samples that sit exactly on the occupancy threshold (compared
    in double, as searchsorted did), sparse levels under min_occ, and NaN."""
    from dtouch.physarum import NORM_EMA
    rng = np.random.default_rng(7)
    cases = []
    for occ_p in (0.0, 0.001, 0.05, 0.3, 1.0):
        s = rng.random((60, 70, 4)).astype(np.float32) * np.float32(30.0)
        for c in (2, 3):
            s[..., c] *= rng.random((60, 70)) < occ_p
        cases.append(s)
    edge = cases[2].copy()
    mx = float(edge[..., 2].max())
    # values exactly at, and one float32 ulp either side of, occ x max
    t32 = np.float32(FRACTAL_NL["occ"] * mx)
    edge[:5, :5, 2] = t32
    edge[5:10, :5, 2] = np.nextafter(t32, np.float32(0))
    edge[10:15, :5, 2] = np.nextafter(t32, np.float32(1e9))
    cases.append(edge)
    nan = cases[3].copy()
    nan[0, 0, 3] = np.nan
    nan[1, 1, 0] = np.nan
    cases.append(nan)
    for s in cases:
        for prev in (None, (3.0, 1.0, 0.5)):
            got = level_norms(s, 20.0, prev)
            want = _level_norms_by_sort(s, 20.0, prev, NORM_EMA)
            assert got == want or all(
                (g == w) or (np.isnan(g) and np.isnan(w)) for g, w in zip(got, want)), (got, want)


# ---------- the engine and the mode ----------

def _gl_field(**kw):
    kw.setdefault("n", 20000)
    kw.setdefault("gw", 192)
    kw.setdefault("gh", 108)
    try:
        return PhysarumFieldGL(**kw)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")


def test_engine_pool_and_picture_follow_the_amount():
    f = _gl_field(n=20000)
    try:
        assert f.n_cap == int(np.ceil(20000 * FRACTAL["density"]))
        assert f.n_active() == 20000 and f.render_size() == (192, 108)
        gray = np.full((108, 192), 0.5, np.float32)
        matte = np.zeros_like(gray)
        f.out_size = (640, 360)
        for _ in range(20):
            f.update(matte, gray)
        assert f.luminance().shape == (108, 192)
        f.fractal = 0.5
        f.bloom = bloom_zones(10.0, 192, 108, 0.5)
        assert f.n_active() == round(20000 * (1 + (FRACTAL["density"] - 1) * 0.5))
        for _ in range(20):
            f.update(matte, gray)
        lum = f.luminance()
        assert lum.shape == (360, 640) and np.isfinite(lum).all() and lum.max() > 0.2
        assert f.lum_tex.size == (640, 360) and f.fractal_error is None
        assert np.isfinite(f.trail_species).all()
    finally:
        f.release()


def test_the_woken_pool_is_the_pool_on_either_ping_pong_side():
    """update() writes only the running rows, so the fractal's parked agents
    wake from whichever agent texture is current. Turned on after an odd
    number of stock frames that was the second texture, which was never
    filled: 97% of the extra pool woke at the origin as trunks, for life.
    Both sides must hold the same spawned pool (a third per level, spread
    over the grid)."""
    for stock_frames in (1, 2):
        f = _gl_field(n=20000)
        try:
            gray = np.full((108, 192), 0.5, np.float32)
            matte = np.zeros_like(gray)
            for _ in range(stock_frames):
                f.update(matte, gray)
            f.fractal = 1.0
            f.update(matte, gray)
            a = f._read_f4(f.fbo_agents_a, 4).reshape(-1, 4)[f.n:f.n_active()]
            share = np.bincount(a[:, 3].astype(int), minlength=3) / len(a)
            assert np.all(np.abs(share - 1 / 3) < 0.03), (stock_frames, share)
            assert (np.hypot(a[:, 0], a[:, 1]) > 2.0).mean() > 0.99, stock_frames
        finally:
            f.release()


def test_engine_runs_stock_when_the_fractal_passes_cannot_build(monkeypatch):
    """A context that cannot build the fractal picture passes keeps the
    stock mold (amount 0) and says why, rather than a dead picture; the mode
    then stops offering the control."""
    f = _gl_field()
    try:
        def boom():
            raise RuntimeError("no MRT here")
        monkeypatch.setattr(f, "_build_fractal", boom)
        f.fractal = 1.0
        f.update(np.zeros((108, 192), np.float32), np.full((108, 192), 0.5, np.float32))
        assert f.fractal_error == "no MRT here"
        assert f.n_active() == f.n and f.render_size() == (192, 108)
        assert f.luminance().shape == (108, 192)
        m = PhysarumMode(engine="gl")
        m.engine, m.pf = "gl", f
        assert not m.fractal_available()
    finally:
        f.release()


def test_mode_says_so_when_the_fractal_cannot_build():
    """The Fractal row hides when this GPU cannot build the fractal passes;
    a visible control does not vanish without a word: the mode toasts the
    reason once, and H's two-step toast names the real cause (not
    'GPU-only' on the GPU)."""
    m = PhysarumMode(matte="luma", seed=1, engine="gl", grid=(192, 108), n=20000)
    host = _StubHost((320, 180))
    said = []
    host.hud.toasts.flash = lambda msg, *a, **k: said.append(msg)
    m.start(host)
    try:
        if m.engine != "gl":
            pytest.skip("no GL context available (CI)")
        m.configure_ui(host.ui)

        def boom():
            raise RuntimeError("no MRT here")
        m.pf._build_fractal = boom
        host.ui.ph_fractal, host.ui.ph_depth = 1.0, 0.9
        frame = np.full((36, 64, 3), 128, np.uint8)
        for _ in range(3):
            m.step(frame, None, 1 / 30)
        assert said == ["fractal veins unavailable: no MRT here"]
        assert host.ui.ph_fractal_ok is False
        m.commands()["physarum.depth"].run()
        assert said[-1] == "FLAT  (fractal unavailable here)"
    finally:
        m.stop()


def _scene_frames(n, w=320, h=180):
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = np.linspace(20, 50, w, dtype=np.uint8)[None, :, None]
    out = []
    for i in range(n):
        fr = base.copy()
        cx = int(w * (0.3 + 0.4 * i / n))
        cy = int(h * (0.5 + 0.2 * np.sin(i * 0.05)))
        cv2.circle(fr, (cx, cy), 34, (235, 235, 235), -1)
        out.append(cv2.GaussianBlur(fr, (15, 15), 0))
    return out


PERCEPT_FRAMES = 120
PERCEPT_RES = (640, 368)


@functools.lru_cache(maxsize=None)
def _picture(look, seed, amount):
    """The mode's own output (grey) after PERCEPT_FRAMES on the GL engine,
    at `look` with its fractal amount overridden to `amount`."""
    m = PhysarumMode(matte="luma", seed=seed, engine="gl", grid=(384, 216), n=45000)
    host = _StubHost(PERCEPT_RES)
    m.start(host)
    try:
        if m.engine != "gl":
            pytest.skip("no GL context available (CI)")
        m.configure_ui(host.ui)
        L, ui = PhysarumMode.BUILTIN[look], host.ui
        ui.ph_point_bg_idx = POINT_NAMES.index(L["point_bg"])
        ui.ph_point_fg_idx = POINT_NAMES.index(L["point_fg"])
        ui.ph_palette_idx = PALETTES_PH.index("mono")
        for k in ("food", "gain", "decay", "exposure", "weave", "evolve", "react", "depth"):
            setattr(ui, "ph_" + k, L[k])
        ui.ph_matte_idx = 5                    # luma: deterministic on synth frames
        ui.ph_fractal = amount
        for fr in _scene_frames(PERCEPT_FRAMES):
            out = m.step(fr, None, 1 / 30)
    finally:
        m.stop()
    return cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _fine(L):
    """(edge, fine): crisp-boundary density (the 3x3-smoothed luma's central
    gradient over 24/255 per 2 px) and thin-structure density (luma above
    its own 9x9 mean by 10/255): the browser verifier's measures."""
    S = cv2.blur(L, (3, 3))
    gx = np.zeros_like(S)
    gy = np.zeros_like(S)
    gx[:, 1:-1] = S[:, 2:] - S[:, :-2]
    gy[1:-1] = S[2:] - S[:-2]
    return float((np.hypot(gx, gy) > 24).mean()), float(((L - cv2.blur(L, (9, 9))) > 10).mean())


# Bars at about half the weakest look/seed measured on 2026-09-25 (M-series
# GL 4.1, 120 frames, grid 384x216): shipped amounts gave edge x3.3-7.4 and
# fine x2.3-3.4 over the same seed's stock; the slider at 0.3 gave edge
# x2.2-5.5, fine x1.5-2.3 and a mean |delta| of 0.11-0.21 on [0, 1].
EDGE_BAR, FINE_BAR = 1.6, 1.3
EDGE_BAR_03, FINE_BAR_03, DELTA_FLOOR = 1.3, 1.15, 0.02


@pytest.mark.parametrize("look", LOOKS)
def test_fractal_is_perceptible_at_each_looks_amount(look):
    """Normalised against the same seed's stock arm, on every seed: more
    crisp boundaries and more thin structure, never less (direction), by
    the bars above (magnitude)."""
    got = []
    for s in SEEDS:
        e0, f0 = _fine(_picture(look, s, 0.0))
        e1, f1 = _fine(_picture(look, s, LOOK_FRACTAL[look]))
        assert e0 > 0.01 and f0 > 0.005, (look, s, "the stock arm grew nothing to compare")
        got.append((e1 / e0, f1 / f0))
    got = np.array(got)
    assert (got > 1.0).all(), (look, got)                      # direction
    assert got[:, 0].min() >= EDGE_BAR and got[:, 1].min() >= FINE_BAR, (look, got)


@pytest.mark.parametrize("look", LOOKS)
def test_fractal_slider_is_honest_at_0_3(look):
    """The MOLD-slider contract, for Fractal: the bottom third of the slider
    is already clearly there, in the right direction, on every seed."""
    got = []
    for s in SEEDS:
        a, b = _picture(look, s, 0.0), _picture(look, s, 0.3)
        (e0, f0), (e1, f1) = _fine(a), _fine(b)
        got.append((e1 / e0, f1 / f0, float(np.abs(a - b).mean() / 255.0)))
    got = np.array(got)
    assert (got[:, :2] > 1.0).all(), (look, got)
    assert got[:, 0].min() >= EDGE_BAR_03 and got[:, 1].min() >= FINE_BAR_03, (look, got)
    assert got[:, 2].min() >= DELTA_FLOOR, (look, got)


def test_mode_drives_the_fractal_into_the_engine():
    """The mode hands the GL engine the amount, the output size, the length
    unit and the two zones on its own clock; the picture comes back at the
    output size (the video palette too); a stub host never sets ph_fractal
    and gets the stock engine."""
    m = PhysarumMode(matte="luma", seed=1, engine="gl", grid=(192, 108), n=20000)
    host = _StubHost((320, 180))
    m.start(host)
    try:
        if m.engine != "gl":
            pytest.skip("no GL context available (CI)")
        frame = np.full((36, 64, 3), 128, np.uint8)
        m.step(frame, None, 1 / 30)
        assert m.pf.fractal == 0.0 and m.pf.render_size() == (192, 108)
        host.ui.ph_fractal = 0.8
        host.ui.ph_palette_idx = PALETTES_PH.index("video")
        for _ in range(10):
            out = m.step(frame, {"amp": 1.0, "bass": 0.0, "treble": 0.0}, 1 / 30)
        assert out.shape == (180, 320, 3)
        pf = m.pf
        assert pf.fractal == 0.8 and pf.out_size == (320, 180)
        assert pf.px_scale == pytest.approx(192 / 576)
        assert pf.bloom == bloom_zones(m._t, 192, 108, 0.8, m._bloom_snd, 1.0)
        assert 0.0 < m._bloom_snd < 1.0 and pf.bloom_t == m._t
        assert pf.luminance().shape == (180, 320)
    finally:
        m.stop()


# ---------- H, the panel control, the CPU fallback ----------

def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _booted(tmp_path, engine):
    host = Host(PhysarumMode(engine=engine), source=SyntheticSource(), res=(192, 108),
                show=False, preset=None, max_frames=1, **_paths(tmp_path))
    host.run()
    return host


def _fractal_widget(ui):
    look = [s for s in ui.spec if getattr(s, "title", None) == "LOOK"][0]
    return [w for w in look.widgets if getattr(w, "label", None) == "Fractal"][0]


def test_h_steps_flat_relief_fractal_on_the_gpu(tmp_path):
    host = _booted(tmp_path, "auto")
    ui, m = host.ui, host.mode
    if m.engine != "gl":
        pytest.skip("no GL context available (CI)")
    run = m.commands()["physarum.depth"].run
    toast = lambda: host.hud.toasts._center.text          # noqa: E731
    assert ui.ph_fractal == LOOK_FRACTAL["veinwork"] and ui.ph_depth == 0.9
    run()
    assert (ui.ph_depth, ui.ph_fractal, toast()) == (0.0, 0.0, "FLAT")
    run()
    assert (ui.ph_depth, ui.ph_fractal, toast()) == (0.9, 0.0, "DEPTH 0.9")
    run()
    assert (ui.ph_depth, ui.ph_fractal, toast()) == (0.9, 1.0, "DEPTH 0.9 + FRACTAL 1.0")
    run()
    assert (ui.ph_depth, ui.ph_fractal, toast()) == (0.0, 0.0, "FLAT")
    # a look (or the slider) setting another amount is what the third step
    # brings back; an amount of 0 makes H two steps again
    ui.ph_depth, ui.ph_fractal = 0.5, 0.7
    run()
    run()
    run()
    assert (ui.ph_depth, ui.ph_fractal) == (0.5, 0.7)
    ui.ph_fractal = 0.0
    run()
    run()
    assert (ui.ph_depth, ui.ph_fractal, toast()) == (0.5, 0.0, "DEPTH 0.5")
    run()
    assert toast() == "FLAT"
    # the slider shows on the GPU engine, in LOOK after Depth, saved as fractal
    w = _fractal_widget(ui)
    assert visible(ui, w) and w.store_key == "fractal" and (w.lo, w.hi) == (0.0, 1.0)
    look = [s for s in ui.spec if getattr(s, "title", None) == "LOOK"][0]
    labels = [getattr(x, "label", None) for x in look.widgets]
    assert labels[labels.index("Depth") + 1] == "Fractal"
    # a look without an amount (one saved before the fractal): the stock
    # organism. (Panic recalls veinwork, and so its 1.0.)
    ui.ph_fractal = 0.9
    apply_look(ui, ui.spec, {}, defaults=PhysarumMode.DEFAULTS)
    assert ui.ph_fractal == FRACTAL_DEFAULT
    m.stop()


def test_cpu_fallback_hides_the_control_and_h_skips_the_fractal_step(tmp_path):
    """FRACTAL_CPU = False: no numpy port. The slider is hidden, not shown
    doing nothing, and H cycles flat / relief, saying the third step is
    GPU-only."""
    assert FRACTAL_CPU is False
    host = _booted(tmp_path, "cpu")
    ui, m = host.ui, host.mode
    assert m.engine == "cpu" and not m.fractal_available()
    assert not visible(ui, _fractal_widget(ui))
    run = m.commands()["physarum.depth"].run
    toast = lambda: host.hud.toasts._center.text          # noqa: E731
    assert ui.ph_fractal == LOOK_FRACTAL["veinwork"]       # the look carries it
    seen = []
    for _ in range(4):
        run()
        seen.append((ui.ph_depth, ui.ph_fractal, toast()))
    assert seen == [(0.0, 0.0, "FLAT  (fractal is GPU-only)"), (0.9, 0.0, "DEPTH 0.9"),
                    (0.0, 0.0, "FLAT  (fractal is GPU-only)"), (0.9, 0.0, "DEPTH 0.9")]
    m.stop()


def test_cpu_picture_ignores_the_fractal_amount():
    """Whatever a look says, the CPU engine draws the stock picture."""
    frame = np.full((36, 64, 3), 128, np.uint8)
    outs = []
    for amount in (0.0, 1.0):
        m = PhysarumMode(matte="luma", seed=3, engine="cpu", grid=(96, 54), n=2000)
        host = _StubHost((192, 108))
        m.start(host)
        host.ui.ph_fractal = amount
        for _ in range(10):
            out = m.step(frame, None, 1 / 30)
        outs.append(out)
        assert not hasattr(m.pf, "fractal")
        m.stop()
    assert np.array_equal(outs[0], outs[1])


# ---------- cost at 4K on the GL path ----------

# Measured 2026-09-25 (M4 Max, GL 4.1, perform tier 1280x736, 500k agents,
# 4K output), GPU per frame by these two queries: stock 1.9 ms (sim 1.32 +
# picture 0.59), fractal 3.6 ms (sim 2.23 with 900k agents + picture 1.34:
# fields at grid size, outlines at 2733x1537). The picture query also holds
# the CPU work between its GL calls (stats readback, level_norms): with
# level_norms on a full sort it read 3.33 ms, and the ratio 3.1.
FRACTAL_GPU_RATIO = 1.9


def test_quality_tier_governs_the_fractal_pool():
    """The tier governor (PhysarumMode.FRACTAL_DENSITY_TIER): at `quality`
    the full 1.8x pool cost 20.6 ms of GPU against 16.7, so that tier's
    fractal adds no agents (13.5 ms; measured to leave the suite's fine-
    structure measures inside their seed spread). The other tiers keep the
    browser's density. A switch to the governed tier says so on screen."""
    assert PhysarumMode.FRACTAL_DENSITY_TIER == {
        "perform": FRACTAL["density"], "balance": FRACTAL["density"], "quality": 1.0}
    m = PhysarumMode(matte="luma", seed=1, engine="gl")
    host = _StubHost((640, 360))
    said = []
    host.hud.toasts.flash = lambda msg, *a, **k: said.append(msg)
    m.start(host)
    try:
        if m.engine != "gl":
            pytest.skip("no GL context available (CI)")
        m.configure_ui(host.ui)
        assert m.pf.fractal_density == FRACTAL["density"]
        host.ui.ph_quality_idx = PhysarumMode.QUALITY_NAMES.index("quality")
        host.ui.ph_fractal = 1.0
        m.step(np.full((36, 64, 3), 128, np.uint8), None, 1 / 30)
        pf = m.pf
        assert (pf.gw, pf.gh) == PhysarumMode.QUALITY["quality"][0]
        assert pf.fractal_density == 1.0 and pf.n_cap == pf.n == pf.n_active()
        assert said and said[-1].endswith("fractal pool 1.0x"), said
    finally:
        m.stop()


def test_fractal_gpu_cost_at_4k(capsys):
    """GL timer queries around the sim and the picture, stock vs fractal, at
    the perform tier with a 4K output. The budget is the conftest rule:
    2x the measured ratio over the stock frame on the same machine."""
    from conftest import assert_within, skip_if_contended
    skip_if_contended()
    gw, gh = 1280, 736
    f = _gl_field(n=500_000, gw=gw, gh=gh, seed=1)
    try:
        yy, xx = np.mgrid[0:gh, 0:gw]
        gray = np.exp(-((xx - gw * 0.4) ** 2 + (yy - gh * 0.5) ** 2)
                      / (2 * (gw / 8) ** 2)).astype(np.float32)
        matte = (gray > 0.4).astype(np.float32)
        f.out_size, f.depth, f.gain = (3840, 2160), 0.9, gw / PX_REF
        f.sat, f.jitter, f.hetero, f.cross, f.sharpen, f.mosaic = 1.2, 0.1, 0.7, 0.5, 0.15, 0.5
        ms = {}
        for amt in (0.0, 1.0):
            f.fractal = amt
            qs, qp = f.ctx.query(time=True), f.ctx.query(time=True)
            sim, pic = [], []
            for i in range(90):
                f.bloom = bloom_zones(i / 60, gw, gh, amt)
                with f.ctx:
                    with qs:
                        f.update(matte, gray)
                    with qp:
                        f.luminance_into_tex()
                    f.ctx.finish()
                if i >= 30:
                    sim.append(qs.elapsed / 1e6)
                    pic.append(qp.elapsed / 1e6)
            ms[amt] = (float(np.median(sim)), float(np.median(pic)))
        assert f.lum_tex.size == fractal_render_size(gw, gh, 3840, 2160)
    finally:
        f.release()
    stock, frac = sum(ms[0.0]), sum(ms[1.0])
    with capsys.disabled():
        print(f"\n[fractal 4K perform] GPU ms stock sim {ms[0.0][0]:.2f} + picture "
              f"{ms[0.0][1]:.2f} = {stock:.2f}; fractal sim {ms[1.0][0]:.2f} + picture "
              f"{ms[1.0][1]:.2f} = {frac:.2f}")
    assert_within(frac, FRACTAL_GPU_RATIO, stock, "fractal GPU frame at 4K (perform)")
