"""PhysarumFieldGL — the GPU field behind PhysarumMode, plus engine selection.

The GPU tests skip (not fail) wherever a standalone GL context cannot be
created — CI has no GPU — through the `gl_field` fixture. The mode-level
fallback test needs no GL at all: it mocks the GPU field's constructor into
raising and asserts the CPU field boots with an amber toast, which is the
path CI actually exercises for every other physarum mode test.

The engine tests mirror tests/test_physarum.py's CPU assertions so the two
fields are held to the same contract: deposit + bounded luminance, the matte
as the pen, luminance as food, determinism, burst, wave.
"""
import re

import cv2
import numpy as np
import pytest

from dtouch.modes.particles import MATTES
from dtouch.modes.physarum import (PALETTES_PH, PhysarumMode, _colorize,
                                   _fmt_agents, _palette_lut)
from dtouch.physarum import POINT_NAMES, POINTS, PhysarumField
from dtouch.physarum_gl import (SHADER_FILES, PhysarumFieldGL, PhysarumGLUnavailable,
                                load_shader)
from dtouch.physarum_looks import LOOKS_PATH, dumps
from dtouch.shell import Host

from test_shell import SyntheticSource

RES = (192, 108)


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode, source=SyntheticSource(), **_paths(tmp_path), **kw)


@pytest.fixture
def gl_field():
    """Factory for small GPU fields; skips the test when GL is absent and
    releases every field it built."""
    built = []

    def make(**kw):
        kw.setdefault("n", 4000)
        kw.setdefault("gw", 96)
        kw.setdefault("gh", 54)
        kw.setdefault("seed", 7)
        try:
            f = PhysarumFieldGL(**kw)
        except PhysarumGLUnavailable as e:
            pytest.skip(f"no GL context available (CI): {e}")
        built.append(f)
        return f
    yield make
    for f in built:
        f.release()


def _flat(f, v=0.0):
    return np.full((f.gh, f.gw), v, np.float32)


# ---------- engine ----------

def test_update_deposits_trail_and_luminance_is_a_picture(gl_field):
    f = gl_field()
    for _ in range(5):
        f.update(_flat(f), _flat(f))
    trail = f.trail
    assert trail.shape == (f.gh, f.gw) and trail.sum() > 0
    assert np.isfinite(trail).all()
    lum = f.luminance()
    assert lum.shape == (f.gh, f.gw) and lum.dtype == np.float32
    assert float(lum.min()) >= 0.0 and float(lum.max()) <= 1.0
    # non-black with structure, not a correctly shaped nothing
    assert float(lum.mean()) > 0.05
    assert float(lum.std()) > 0.01


def test_matte_is_the_pen_blends_toward_body_point(gl_field):
    """Same assertion as the CPU field: matte=0 runs the field point, matte=1
    the body point, and mean per-frame displacement follows the step."""
    def mean_step(matte_value):
        # reseed off: teleports are displacement noise on top of the stride
        # being measured (haze's 0.35 px stride would drown in them)
        f = gl_field(point_bg="haze", point_fg="fingers", reseed_frac=0.0)
        m, g = _flat(f, matte_value), _flat(f)
        f.update(m, g)
        px, py, _ = f.agents()
        f.update(m, g)
        qx, qy, _ = f.agents()
        f.release()                       # one live field at a time
        dx = np.minimum(np.abs(qx - px), f.gw - np.abs(qx - px))
        dy = np.minimum(np.abs(qy - py), f.gh - np.abs(qy - py))
        return float(np.hypot(dx, dy).mean())

    slow, fast = mean_step(0.0), mean_step(1.0)
    assert slow == pytest.approx(POINTS["haze"]["step"], rel=0.15)
    assert fast == pytest.approx(POINTS["fingers"]["step"], rel=0.15)


def test_luminance_is_food_draws_the_mold(gl_field):
    f = gl_field(food=1.0, reseed_frac=0.05)
    gray = _flat(f)
    band = slice(f.gw // 3, 2 * f.gw // 3)
    gray[:, band] = 1.0
    for _ in range(40):
        f.update(_flat(f), gray)
    trail = f.trail
    inside = float(trail[:, band].mean())
    outside = float(np.delete(trail, np.s_[band], axis=1).mean())
    assert inside > outside * 1.3


def test_reseed_lands_on_the_lit_subject(gl_field):
    """With every agent recycled each frame and food off, positions must
    follow matte*luma — the rejection sampler's whole job."""
    f = gl_field(reseed_frac=1.0, food=0.0)
    matte, gray = _flat(f), _flat(f, 1.0)
    matte[:, :f.gw // 2] = 1.0
    f.update(matte, gray)
    px, _, _ = f.agents()
    assert (px < f.gw // 2).mean() > 0.98


def test_trail_rows_and_columns_are_grid_coordinates(gl_field):
    """The readback must not be flipped: park the pool on one cell (burst
    with a hair of radius, tempo ~0 so nobody walks off it, no blur) and the
    trail's peak must sit at that (row=y, col=x)."""
    f = gl_field(diffuse=0, gain=1e-4, reseed_frac=0.0)
    f.spawn_burst(70.0, 12.0, frac=1.0, radius=0.05)
    f.update(_flat(f), _flat(f))
    y, x = np.unravel_index(int(f.trail.argmax()), (f.gh, f.gw))
    assert (y, x) == (12, 70)


def test_same_seed_same_trail(gl_field):
    """Fields are built one at a time, each released before the next: with
    two live standalone contexts moderngl issues every GL call on the last
    one created, so two live fields alias one set of textures and this
    test compared a field with itself. A different seed must differ, so the
    equality is a real read and not an empty one."""
    def trail_for(seed):
        f = gl_field(seed=seed)
        for _ in range(3):
            f.update(_flat(f), _flat(f))
        t = f.trail
        f.release()
        return t

    a, b, c = trail_for(7), trail_for(7), trail_for(8)
    assert a.sum() > 0 and np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_readbacks_stay_true_after_luminance(gl_field):
    """Regression: on Apple GL, after the tonemap rendered into the uint8
    target the second and later reads of the float targets returned the
    uint8 target's contents (agents ~1.0, trail all ones) with no GL error.
    trail / agents() must keep reading true after luminance(), repeatedly."""
    f = gl_field()
    for _ in range(3):
        f.update(_flat(f), _flat(f))
    t0 = f.trail
    px0, py0, h0 = f.agents()
    assert t0.sum() > 0 and float(px0.max()) > 1.0
    for _ in range(2):
        f.luminance()
        for _ in range(2):
            assert np.array_equal(f.trail, t0)
            px, py, h = f.agents()
            assert np.array_equal(px, px0) and np.array_equal(py, py0)
            assert np.array_equal(h, h0)


def test_spawn_burst_concentrates_agents(gl_field):
    f = gl_field()
    f.spawn_burst(20.0, 20.0, frac=0.5, radius=2.0)
    px, py, _ = f.agents()
    d = np.hypot(px - 20.0, py - 20.0)
    assert (d < 8.0).mean() > 0.4


def test_wave_points_everyone_outward(gl_field):
    f = gl_field()
    f.wave(f.gw / 2, f.gh / 2)
    px, py, h = f.agents()
    dx, dy = px - f.gw / 2, py - f.gh / 2
    outward = np.cos(h) * dx + np.sin(h) * dy
    assert (outward >= 0).mean() > 0.99


def test_gpu_and_cpu_settle_to_the_same_trail_mass(gl_field):
    """Same agents, same deposit, same decay: the two engines' total trail
    must converge to the same equilibrium (n * deposit * decay / (1 - decay))
    — the check that this is the same model, not a look-alike."""
    g = gl_field(n=8000, diffuse=1, decay=0.9)
    c = PhysarumField(n=8000, gw=g.gw, gh=g.gh, seed=7, diffuse=1, decay=0.9)
    m, y = _flat(g), _flat(g)
    for _ in range(80):
        g.update(m, y)
        c.update(m, y)
    assert float(g.trail.sum()) == pytest.approx(float(c.trail.sum()), rel=0.1)


def test_rejects_bad_sizes_before_touching_gl():
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=0)
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=10, gw=0, gh=5)
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=10, decay=1.5)


def test_update_rejects_wrong_shaped_input(gl_field):
    f = gl_field()
    with pytest.raises(ValueError):
        f.update(np.zeros((3, 3), np.float32), _flat(f))


def test_release_is_idempotent(gl_field):
    f = gl_field()
    f.release()
    f.release()
    assert f.ctx is None


# ---------- mode: engine selection + fallback ----------

def test_mode_boots_the_gpu_field_when_gl_is_there(tmp_path):
    try:
        probe = PhysarumFieldGL(n=16, gw=8, gh=8)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    probe.release()
    m = PhysarumMode()
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "gl" and isinstance(m.pf, PhysarumFieldGL)
        assert (m.grid, m.n) == (PhysarumMode.GL_GRID, PhysarumMode.GL_N)
        out = m.step(np.full((36, 64, 3), 255, np.uint8), None, 1 / 30)
        assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
        assert m.status_tail("synthetic").startswith(
            _fmt_agents(PhysarumMode.GL_N).join(("gl ", "")))
    finally:
        m.stop()
    assert m.pf is None
    m.stop()                                  # idempotent after release


def test_mode_engine_cpu_is_an_explicit_opt_out(tmp_path):
    m = PhysarumMode(engine="cpu")
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "cpu" and isinstance(m.pf, PhysarumField)
        assert (m.grid, m.n) == (PhysarumMode.CPU_GRID, PhysarumMode.CPU_N)
        assert m.status_tail("synthetic") == (
            f"cpu {_fmt_agents(PhysarumMode.CPU_N)}  cam synthetic")
    finally:
        m.stop()


def test_mode_falls_back_to_cpu_with_a_toast_when_gl_init_fails(tmp_path, monkeypatch):
    """No GL needed: the GPU constructor is mocked into failing the way a
    missing driver does, and the show must still start — on the CPU field,
    at the CPU sizing, with the reason toasted in amber (DESIGN.md §6.4)."""
    def boom(*a, **k):
        raise PhysarumGLUnavailable("GPU physarum unavailable: no context")
    monkeypatch.setattr("dtouch.modes.physarum.PhysarumFieldGL", boom)
    m = PhysarumMode()                        # engine="auto"
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "cpu" and isinstance(m.pf, PhysarumField)
        assert (m.grid, m.n) == (PhysarumMode.CPU_GRID, PhysarumMode.CPU_N)
        toast = host.hud.toasts._center
        assert toast is not None and "CPU" in toast.text
        out = m.step(np.full((36, 64, 3), 255, np.uint8), None, 1 / 30)
        assert out.shape == (RES[1], RES[0], 3)
    finally:
        m.stop()


def test_explicit_grid_and_agents_win_on_either_engine(tmp_path, monkeypatch):
    monkeypatch.setattr("dtouch.modes.physarum.PhysarumFieldGL",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    m = PhysarumMode(grid=(64, 36), n=1234)
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert (m.grid, m.n) == ((64, 36), 1234)
    finally:
        m.stop()


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError):
        PhysarumMode(engine="metal")


def test_fmt_agents():
    assert _fmt_agents(400_000) == "400k"
    assert _fmt_agents(2_000_000) == "2.0M"
    assert _fmt_agents(1_234) == "1k"


def test_colorize_matches_the_lut_index():
    """applyColorMap replaced numpy fancy indexing for speed; it must stay
    pixel-identical, RGB order intact."""
    lum = np.random.default_rng(3).random((20, 30)).astype(np.float32)
    idx = (lum * 255.0).astype(np.uint8)
    for pal in ("arctic", "fire", "mono"):
        assert np.array_equal(_colorize(lum, pal), _palette_lut(pal)[idx])


# ---------- the shared-shader contract (dtouch/shaders/physarum/README.md) ----------

def _shader_sources():
    return {name: load_shader(name, version_line="") for name in SHADER_FILES}


def test_every_shader_file_exists_and_carries_no_host_lines():
    """The host prepends `#version` (330 core here, 300 es in the browser)
    and the browser prepends `precision`; neither may live in the files."""
    for name, src in _shader_sources().items():
        body = src.split("#line 1\n", 1)[1]
        assert body.strip(), f"{name} is empty"
        assert "#version" not in body, f"{name} carries a #version line"
        assert "precision " not in body, f"{name} carries a precision line"


def _lint_es300(name, body):
    """Static lint for the GLSL 3.30 ∩ ES 3.00 subset the site vendors, on a
    shared file's body (no host lines). Returns the comment-stripped code for
    any per-file checks. Shared by every browser-shared unit's suite
    (tests/test_dither_shared.py imports it)."""
    code = "\n".join(l.split("//", 1)[0] for l in body.splitlines())
    banned = ("double", "dvec", "usampler", "isampler", "sampler1D", "sampler3D",
              "image2D", "imageStore", "imageLoad", "gl_FragColor", "texture2D(",
              "#include", "#extension", "layout(binding", "subroutine",
              "gl_PrimitiveID", "gl_Layer")
    for tok in banned:
        assert tok not in code, f"{name} uses {tok!r}, outside ES 3.00"
    # integer % is undefined in ES 3.00 when an operand is negative; the one
    # allowed use is on gl_VertexID (never negative). Wrap with floor().
    for m in re.finditer(r"%", code):
        assert code[:m.start()].rstrip().endswith("gl_VertexID"), (
            f"{name}: integer % on a possibly negative operand is undefined in "
            "ES 3.00 — wrap with floor() as blur.frag does")
    if name.endswith(".frag"):
        assert "layout(location = 0) out vec4" in code, \
            f"{name}: fragment output needs an explicit layout(location = 0)"
    return code


@pytest.mark.parametrize("name", SHADER_FILES)
def test_shader_stays_inside_glsl_es_300(name):
    """Static lint for the GLSL 3.30 ∩ ES 3.00 subset the site vendors."""
    body = load_shader(name, version_line="").split("#line 1\n", 1)[1]
    code = _lint_es300(name, body)
    if name == "deposit.vert":
        assert "gl_PointSize = 1.0" in code   # ES has no glPointSize


@pytest.mark.parametrize("name", SHADER_FILES)
def test_shader_compiles_under_330_core(name):
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        src = load_shader(name)
        assert src.startswith("#version 330 core\n#line 1\n")
        # deposit.vert/.frag are a pair (v_dep); everything else rides the
        # full-screen triangle
        if name.startswith("deposit"):
            ctx.program(vertex_shader=load_shader("deposit.vert"),
                        fragment_shader=load_shader("deposit.frag"))
        elif name.endswith(".vert"):
            ctx.program(vertex_shader=src, fragment_shader=(
                "#version 330 core\nout vec4 f; void main(){ f = vec4(1.0); }"))
        else:
            ctx.program(vertex_shader=load_shader("fullscreen.vert"), fragment_shader=src)
    finally:
        ctx.release()


def test_looks_json_matches_the_python_source():
    """dtouch/shaders/physarum/looks.json is vendored by the site; it must
    equal what the Python tables generate. Regenerate with
    `python -m dtouch.physarum_looks` after touching POINTS / BUILTIN /
    DEFAULTS / palette stops."""
    import json
    with open(LOOKS_PATH, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == json.loads(dumps()), (
        "looks.json is stale: run `python -m dtouch.physarum_looks`")
    assert set(on_disk["points"]) == set(POINTS)
    assert set(on_disk["builtin"]) == set(PhysarumMode.BUILTIN)
    for name, pal in on_disk["palettes"].items():
        assert len(pal["lut"]) == 256 and all(len(c) == 3 for c in pal["lut"])
        assert pal["lut"] == _palette_lut(name).tolist()


# ---------- mode-level regressions (the real-app symptoms of issue: every
# look rendered the same wire-mesh and the video background disappeared) ----

def _skip_without_gl():
    try:
        probe = PhysarumFieldGL(n=16, gw=8, gh=8)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    probe.release()


def _blob_frames(n, w=320, h=180):
    """Synthetic camera: a bright blob wandering over a dim gradient."""
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = np.linspace(20, 50, w, dtype=np.uint8)[None, :, None]
    frames = []
    for i in range(n):
        f = base.copy()
        cx = int(w * (0.2 + 0.6 * i / n))
        cy = int(h * (0.5 + 0.25 * np.sin(i * 0.15)))
        cv2.circle(f, (cx, cy), 28, (235, 235, 235), -1)
        frames.append(cv2.GaussianBlur(f, (15, 15), 0))
    return frames


def _apply_look(ui, look, **over):
    ui.ph_point_bg_idx = POINT_NAMES.index(look["point_bg"])
    ui.ph_point_fg_idx = POINT_NAMES.index(look["point_fg"])
    ui.ph_palette_idx = PALETTES_PH.index(look["palette"])
    ui.ph_food = look["food"]; ui.ph_gain = look["gain"]
    ui.ph_decay = look["decay"]; ui.ph_exposure = look["exposure"]
    ui.ph_grain = 0.5
    ui.ph_matte_idx = MATTES.index("luma")     # deterministic headless matte
    ui.ph_video_bg = False; ui.ph_video_mix = 0.5
    for k, v in over.items():
        setattr(ui, k, v)


def _median_vein_width(out):
    """Median thickness (px) of the bright structure, via distance transform."""
    g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    d = cv2.distanceTransform((g > 96).astype(np.uint8), cv2.DIST_L2, 3)
    core = d[d > 0.5]
    return float(2.0 * np.median(core)) if core.size else 0.0


class _StubToasts:
    def flash(self, *a, **k):
        pass


class _StubHost:
    """The three attributes step() actually reads — ui, hud.toasts, res —
    without booting the shell (Host only builds its UI inside run())."""

    def __init__(self, res):
        self.res = res
        self.ui = type("Ui", (), {})()
        self.hud = type("Hud", (), {})()
        self.hud.toasts = _StubToasts()


def _booted_mode(engine, res=(640, 368)):
    m = PhysarumMode(matte="luma", seed=1, engine=engine)
    host = _StubHost(res)
    m.start(host)
    return m, host


def test_gl_default_sizing_keeps_the_cpu_looks_scale():
    """The look parameters (sense/step, and the blur that sets vein width)
    are calibrated in CPU-grid pixels. Driving them unscaled on the GL grid
    halved the mold's relative scale: veins fell below what the projector /
    SIGNAL dither stage resolves, and every look collapsed into the same
    fine wire-mesh (measured 0.52x the CPU vein width before the fix)."""
    _skip_without_gl()
    frames = _blob_frames(50)
    widths = {}
    for engine in ("gl", "cpu"):
        m, host = _booted_mode(engine)
        try:
            _apply_look(host.ui, PhysarumMode.BUILTIN["veinwork"])
            for i in range(45):
                out = m.step(frames[i % len(frames)], None, 1 / 30)
            widths[engine] = _median_vein_width(out)
        finally:
            m.stop()
    assert m.engine == "cpu" and widths["cpu"] > 0
    ratio = widths["gl"] / widths["cpu"]
    assert ratio >= 0.7, (
        f"GL veins are {ratio:.2f}x the CPU ground truth's width "
        f"({widths['gl']:.1f}px vs {widths['cpu']:.1f}px) — the GL engine is "
        "rendering the mold at the wrong relative scale")


def test_looks_differ_under_gl():
    """Recalling different BUILTIN looks must change the rendered picture —
    the panel/look parameters have to reach the GL field every frame."""
    _skip_without_gl()
    frames = _blob_frames(50)
    m, host = _booted_mode("gl", res=(320, 184))
    outs = {}
    try:
        for name, look in PhysarumMode.BUILTIN.items():
            _apply_look(host.ui, look)
            for i in range(40):
                out = m.step(frames[i % len(frames)], None, 1 / 30)
            outs[name] = out.astype(np.float32)
    finally:
        m.stop()
    names = list(outs)
    dists = [np.abs(outs[a] - outs[b]).mean()
             for i, a in enumerate(names) for b in names[i + 1:]]
    assert min(dists) > 5.0, (
        f"some GL looks render nearly identically (min pairwise distance "
        f"{min(dists):.2f}) — look parameters are not reaching the field")


def test_video_bg_composites_under_gl():
    """Video bg must screen-blend the live frame under the GL mold: turning
    it on changes the picture a lot, and the result carries the frame."""
    _skip_without_gl()
    frames = _blob_frames(50)
    m, host = _booted_mode("gl", res=(320, 184))
    try:
        _apply_look(host.ui, PhysarumMode.BUILTIN["veinwork"])
        for i in range(40):
            out_off = m.step(frames[i % len(frames)], None, 1 / 30)
        host.ui.ph_video_bg = True
        host.ui.ph_video_mix = 0.8
        out_on = m.step(frames[40 % len(frames)], None, 1 / 30)
    finally:
        m.stop()
    diff = np.abs(out_on.astype(np.float32) - out_off.astype(np.float32)).mean()
    assert diff > 5.0, f"video bg changed the picture by only {diff:.2f}"
    fin = cv2.cvtColor(cv2.resize(frames[40 % len(frames)], (320, 184)),
                       cv2.COLOR_BGR2GRAY).astype(np.float32)
    og = cv2.cvtColor(out_on, cv2.COLOR_RGB2GRAY).astype(np.float32)
    corr = float(np.corrcoef(fin.ravel(), og.ravel())[0, 1])
    assert corr > 0.1, f"bg-on output does not carry the frame (corr {corr:.3f})"


def test_field_survives_a_foreign_context(gl_field):
    """In the app other moderngl contexts exist (GlowRenderer, and whatever
    created one last is current). The field must bind its OWN context per
    call: before the fix, a foreign context drawing between steps silently
    received the field's GL calls and the luminance came back as saturated
    garbage (mean abs error 0.49 on [0,1] vs an identical solo run)."""
    import moderngl
    m = np.zeros((54, 96), np.float32); m[20:34, 40:60] = 1.0
    g = m.copy()

    def run(interleave):
        f = gl_field(n=8000, seed=7)
        ctx2 = None
        try:
            for i in range(40):
                if interleave and i >= 20:
                    if ctx2 is None:
                        ctx2 = moderngl.create_standalone_context()
                        prog = ctx2.program(
                            vertex_shader="#version 330\nin vec2 v;"
                                          "\nvoid main(){gl_Position=vec4(v,0,1);}",
                            fragment_shader="#version 330\nout vec4 c;"
                                            "\nvoid main(){c=vec4(1,0,0,1);}")
                        vbo = ctx2.buffer(
                            np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
                        vao = ctx2.vertex_array(prog, [(vbo, "2f", "v")])
                        fbo = ctx2.framebuffer(
                            color_attachments=[ctx2.texture((64, 64), 4)])
                    fbo.use()
                    ctx2.clear(0, 0, 0, 1)
                    vao.render(moderngl.TRIANGLES, vertices=3)
                f.update(m, g)
                lum = f.luminance()
            return lum
        finally:
            if ctx2 is not None:
                ctx2.release()

    solo = run(False)
    interleaved = run(True)
    assert np.abs(solo - interleaved).mean() < 1e-6, (
        "a foreign GL context drawing between steps corrupted the field")
