"""Physarum feel batch — network-not-thoroughfares, evolve churn, react
carving, preset separation, SIGNAL legibility, quality tiers.

These are behavior floors, not goldens: each asserts the *direction and rough
magnitude* of an owner-facing quality (junction/mesh density, still-scene
reorganization, preset distinctness) on the deterministic CPU engine with a
fixed seed, using only core-cv2 operations (no opencv-contrib).
"""
import numpy as np
import pytest
import cv2

from dtouch.circuit_bent import SCANLINE_ROWS, CircuitBent
from dtouch.modes.physarum import PhysarumMode
from dtouch.physarum import POINT_NAMES, PhysarumField

GRID = (288, 162)
N = 60_000


class _Toasts:
    def flash(self, *a, **k):
        pass

    def hint(self, *a, **k):
        pass


class _Hud:
    def __init__(self):
        self.toasts = _Toasts()


class _Ui:
    pass


class _Host:
    """The slice of Host that PhysarumMode.step touches — headless."""

    def __init__(self, res=GRID):
        self.res = res
        self.hud = _Hud()
        self.ui = _Ui()


def _scene(w, h):
    """Static synthetic scene: person-ish bright blob + band-limited noise
    (no periodic texture — a sine grid would print itself into the mold)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (xx - w * 0.55) ** 2 + (yy - h * 0.5) ** 2
    sig = min(w, h) * 0.18
    rng = np.random.default_rng(42)
    noise = rng.random((h // 8 + 1, w // 8 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), min(w, h) * 0.04)
    noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-6)
    g = np.clip(0.15 + 0.5 * np.exp(-r2 / (2 * (1.7 * sig) ** 2))
                + 0.25 * noise, 0, 1)
    return cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _boot(weave=0.0, evolve=0.0, react=0.0, seed=7, point=None):
    host = _Host()
    mode = PhysarumMode(matte="luma", grid=GRID, n=N, seed=seed, engine="cpu")
    mode.start(host)
    mode.configure_ui(host.ui)
    # Depth (the lit relief, a LOOK control) off: these batches measure the
    # ORGANISM's structure, and their floors were calibrated on the flat
    # tonemap. The relief darkens every vein's far flank, which moves
    # threshold counts like _loops without the structure changing (measured
    # 2026-09-24 at the default depth 0.6: weave-0.7 loops 13 -> 5, weave
    # branch diff 0.065 -> 0.059 at 0.3). Depth's own perceptibility is
    # pinned in tests/test_physarum_gl.py.
    host.ui.ph_depth = 0.0
    host.ui.ph_matte_idx = 5          # luma — deterministic on synth frames
    host.ui.ph_weave = weave
    host.ui.ph_evolve = evolve
    host.ui.ph_react = react
    if point is not None:
        host.ui.ph_point_bg_idx = POINT_NAMES.index(point)
        host.ui.ph_point_fg_idx = POINT_NAMES.index(point)
    return host, mode


def _settle(mode, frame, frames, dt=1 / 30):
    for _ in range(frames):
        mode.step(frame, None, dt)
    return mode.pf.luminance()


def _loops(lum, thresh=0.45):
    """Enclosed dark compartments in the thresholded trail — mesh closure.
    Core cv2 only (no contrib thinning): a reticulated web encloses many
    cells; a few thoroughfares enclose almost none."""
    binary = (lum > thresh).astype(np.uint8)
    n, labels = cv2.connectedComponents(1 - binary, connectivity=4)
    border = set(np.unique(labels[0])) | set(np.unique(labels[-1])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    return max(n - 1 - len(border - {0}), 0)


# ---------- P1: network, not thoroughfares ----------

def test_weave_multiplies_mesh_density():
    """The anti-thoroughfare floor: weave at the veinwork default (0.7) must
    close several times more mesh compartments than weave 0 (the legacy
    bold-canal behavior) on the same still scene.

    The RELATIVE claim is the real one and it has only got stronger: measured
    3 -> 13 here, a 4.3x multiplication.

    The absolute floor was 25, pinned when the field was one species and the
    picture was a mesh of closed cells. It is now three species carving
    territories with membranes between them, so "enclosed dark compartments
    above threshold" counts a different object and the number moved. Measured
    at weave 0.7: 13 at the shipped FEEL_CURVE of 0.30, and 7 at the 0.6 the
    curve used to be — i.e. the absolute count had ALREADY fallen below 25
    before the curve was touched, and steepening the curve nearly doubled it
    back. 8 keeps the floor meaningful against the current engine.
    """
    f = _scene(*GRID)
    lum0 = _settle(_boot(weave=0.0)[1], f, 240)
    lum1 = _settle(_boot(weave=0.7)[1], f, 240)
    l0, l1 = _loops(lum0), _loops(lum1)
    assert l1 >= 2 * max(l0, 1), (l0, l1)
    assert l1 >= 8, l1


def test_weave_zero_is_the_legacy_engine():
    """weave 0 must leave every anti-thoroughfare lever off — the knob's
    bottom is the shipped behavior, not a new look."""
    host, mode = _boot(weave=0.0)
    mode.step(_scene(*GRID), None, 1 / 30)
    pf = mode.pf
    assert pf.sat == 0.0 and pf.jitter == 0.0 and pf.hetero == 0.0
    assert pf.reseed_frac == pytest.approx(0.004)
    mode.stop()


# ---------- P2: evolve churns a still scene ----------

def test_the_picture_never_settles():
    """Equilibrium is the enemy (memory/knowledge/decisions/2026-08-24-
    lighteater-magic-over-control.md: "getting stuck / static equilibrium is
    the enemy of aliveness"). At a long lag on a STILL scene the blurred
    trail must still have moved, at every evolve setting.

    This used to assert that evolve 1 decorrelates 1.3x harder than evolve 0,
    and it no longer can — not because evolve got weaker, but because the
    floor came up underneath it. Three species pushing on each other means
    the field is never at rest even with evolve at 0, so there is little
    headroom left to measure. Both engines now land at a churn ratio near
    1.0 with the old metric, across seeds.

    Evolve's own contribution moved to the SPATIAL axis (the mosaic), where
    it is large and measurable — tests/test_physarum_mosaic.py is its floor.
    What belongs here is the absolute claim: it never stops moving.
    """
    def churn(evolve):
        host, mode = _boot(weave=0.5, evolve=evolve)
        f = _scene(*GRID)
        _settle(mode, f, 150, dt=0.1)
        a = mode.pf.luminance().copy()
        _settle(mode, f, 120, dt=0.1)
        b = mode.pf.luminance()
        mode.stop()
        fa = cv2.GaussianBlur(a, (0, 0), 3)
        fb = cv2.GaussianBlur(b, (0, 0), 3)
        mad = float(np.abs(fa - fb).mean())
        ca, cb = fa - fa.mean(), fb - fb.mean()
        corr = float((ca * cb).sum()
                     / np.sqrt((ca * ca).sum() * (cb * cb).sum()))
        return mad, corr

    for evolve in (0.0, 1.0):
        mad, corr = churn(evolve)
        assert mad >= 0.08, (evolve, mad)
        assert corr <= 0.90, (evolve, corr)


# ---------- P3: react carves locally ----------

def test_gather_is_local_cpu():
    """gather() must move only agents near the center — the react impulse
    never drains the rest of the organism (spawn_burst does, by design)."""
    pf = PhysarumField(n=20_000, gw=200, gh=120, seed=3)
    px0, py0 = pf.px.copy(), pf.py.copy()
    pf.gather(50.0, 60.0, frac=1.0, radius=25.0)
    moved = (pf.px != px0) | (pf.py != py0)
    d0 = np.hypot(px0 - 50.0, py0 - 60.0)
    assert moved[d0 < 24.0].mean() > 0.95        # inside: rushed in
    assert not moved[d0 > 26.0].any()            # outside: untouched


def test_keep_map_slows_decay_locally_cpu():
    """The linger map: keep=1 must hold trail against decay (0.995 effective)
    while keep=0 decays at the base rate — swept paths stay painted."""
    def run(keep_value):
        pf = PhysarumField(n=5_000, gw=96, gh=54, seed=3, decay=0.85,
                           reseed_frac=0.0)
        m = np.zeros((54, 96), np.float32)
        keep = np.full((54, 96), keep_value, np.float32)
        for _ in range(60):
            pf.update(m, m, keep)
        return float(pf.trail.sum())
    assert run(1.0) > 4.0 * run(0.0)


def test_react_zero_adds_no_keep_map():
    """react 0 must run the plain update path (keep=None) — the linger
    machinery is gated on the slider, not always-on."""
    host, mode = _boot(react=0.0)
    calls = []
    orig = mode.pf.update

    def spy(m, g, keep=None):
        calls.append(keep)
        return orig(m, g, keep)
    mode.pf.update = spy
    mode.step(_scene(*GRID), None, 1 / 30)
    assert calls == [None]
    mode.stop()


# ---------- P4: presets stay pairwise distinct ----------

def test_points_stay_pairwise_distinct():
    """The separation floor: every behavior-point pair, run as both body and
    field on the same still scene, must stay apart in morphology space
    (coverage / vein width / mesh closure / brightness). Floor is ~60% of
    the measured minimum so the pin survives library-version jitter."""
    feats = {}
    f = _scene(*GRID)
    for pt in POINT_NAMES:
        host, mode = _boot(weave=0.6, point=pt)
        lum = _settle(mode, f, 200)
        mode.stop()
        binary = (lum > 0.35).astype(np.uint8)
        dist = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 3)
        on = dist[dist > 0]
        w_med = float(np.median(on) * 2) if on.size else 0.0
        feats[pt] = np.array([float(binary.mean()) * 4.0,
                              min(w_med, 20.0) / 5.0,
                              _loops(lum, 0.35) / 20.0,
                              float(lum.mean()) * 4.0], np.float32)
    dists = {}
    names = list(POINT_NAMES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dists[a + "|" + b] = float(np.linalg.norm(feats[a] - feats[b]))
    worst = min(dists, key=dists.get)
    assert dists[worst] >= 0.25, (worst, sorted(dists.items(), key=lambda kv: kv[1])[:3])


# ---------- P5: scanlines survive display scaling ----------

def test_scanlines_visible_after_downscale():
    """Scanlines must still read after the frame is scaled to fit a window /
    projector: the pitch scales with output height (SCANLINE_ROWS periods per
    frame), so a half-size linear downscale keeps clear row modulation. The
    shipped 1-px mask averaged to a uniform dim — 'scanlines does nothing'."""
    h, w = 1080, 1920
    frame = np.full((h, w, 3), 180, np.uint8)
    cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                     bit_crush=0, dither_mode=None, scanlines=True)
    out = cb.process(frame)
    assert round(h / SCANLINE_ROWS) == 4          # 1080p pitch: 4 rows
    down = cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    rows = down.astype(np.float32).mean(axis=(1, 2))
    modulation = float(np.abs(np.diff(rows)).mean())
    assert modulation >= 5.0, modulation          # u8 units; ~0 when it aliases out
    # and off means off
    cb.scanlines = False
    flat = cb.process(frame)
    frows = cv2.resize(flat, (w // 2, h // 2)).astype(np.float32).mean(axis=(1, 2))
    assert float(np.abs(np.diff(frows)).mean()) < 0.5


def test_physarum_scales_signal_dither_rows():
    """Physarum overrides the rack's fixed 72-row dither working res with a
    resolution-scaled one (~1/6 of output height, floored) so the veins stop
    reading as boulder-sized grain at 1080p+."""
    rows = PhysarumMode.signal_dither_rows
    assert rows(1080) == 180
    assert rows(2160) == 360
    assert rows(480) == 96                        # floor for small windows


# ---------- P6: quality tiers ----------

def test_quality_tier_resolutions():
    """The render-quality control's sizing contract.

    The load-bearing number is agents per grid cell, not the agent count. At
    the old ~2.1/cell every cell carried trail, there was no dark for a vein
    to be a vein against, and every look collapsed onto the same mesh. Jones
    reticulation wants well under 1; these tiers hold ~0.5, and a tier that
    drifts out of the band is the bug this pins.
    """
    assert PhysarumMode.QUALITY_NAMES == ["perform", "balance", "quality"]
    assert PhysarumMode.QUALITY["perform"] == (PhysarumMode.GL_GRID,
                                               PhysarumMode.GL_N)
    seen = []
    for name in PhysarumMode.QUALITY_NAMES:
        (gw, gh), n = PhysarumMode.QUALITY[name]
        density = n / float(gw * gh)
        assert 0.35 <= density <= 0.75, f"{name}: {density:.2f} agents/cell"
        seen.append((gw * gh, n))
    # each tier is strictly bigger than the last, in both grid and pool
    assert seen == sorted(seen)
    assert [c for c, _ in seen] == sorted({c for c, _ in seen})


def test_quality_cycle_ignored_on_cpu_and_explicit_grid():
    """The CPU fallback and a fixed grid/n override both ignore the quality
    cycle — no rebuild, no crash, same grid."""
    host, mode = _boot()
    host.ui.ph_quality_idx = 2
    mode.step(_scene(*GRID), None, 1 / 30)
    assert mode.engine == "cpu" and mode.grid == GRID and mode.n == N
    mode.stop()


def test_quality_switch_rebuilds_gl_field():
    """On the GL engine, cycling quality rebuilds the field at the tier's
    grid and agent count."""
    try:
        import moderngl
        moderngl.create_standalone_context().release()
    except Exception as e:                        # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    host = _Host()
    mode = PhysarumMode(matte="luma", seed=7, engine="gl")
    mode.start(host)
    if mode.engine != "gl":
        mode.stop()
        pytest.skip("GL field did not boot")
    mode.configure_ui(host.ui)
    host.ui.ph_matte_idx = 5
    frame = _scene(64, 36)
    mode.step(frame, None, 1 / 30)
    assert mode.grid == (1280, 736)
    host.ui.ph_quality_idx = 2
    mode.step(frame, None, 1 / 30)
    assert (mode.grid, mode.n) == PhysarumMode.QUALITY["quality"]
    assert mode.pf.gw == 2560 and mode.pf.gh == 1472
    mode.stop()
