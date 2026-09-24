"""Physarum aliveness batch — regime hops, staleness melt, perceptual slider
mapping, and the Z random cast (2026-08 owner playtest: "still focusing on
thoroughfares... maybe even less alive than before; Weave/Evolve/React all
sound cool but it's kind of unclear what they do").

Two contracts pinned here:

- **Perceptible within 5 seconds.** Every MOLD feel slider (weave / evolve /
  react), branched from an identical settled state, must change the rendered
  picture above a floor within 5 simulated seconds — at 0.3, not just at 1.0
  (the sliders ride a value**0.6 perceptual curve for exactly this reason).
- **Never settle.** Evolve's regime scheduler hops between distinct growth
  characters with decisive crossfades and bounded random dwells, and the
  staleness melt dissolves whatever stops changing — with evolve at 0 both
  are fully off (zero still holds one structure).

Deterministic CPU engine, fixed seeds, behavior floors not goldens.
"""
import numpy as np
import pytest
import cv2

from dtouch.modes.physarum import (REGIME_DWELL, REGIME_KEYS, REGIME_NAMES,
                                   REGIMES, PhysarumMode)
from dtouch.physarum import KEEP_HOLD, MELT_DROP, MELT_FLOOR, PhysarumField
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable

from test_physarum_feel import _Host, _boot, _scene, GRID, N


# ---------- P1: every feel slider is perceptible within 5 seconds ----------

def _swept(base, i, w=GRID[0], h=GRID[1]):
    """The scene with a bright dot sweeping a diagonal — react's motion."""
    f = base.copy()
    x = int((0.15 + 0.6 * (i / 150.0)) * w)
    y = int((0.25 + 0.5 * (i / 150.0)) * h)
    cv2.circle(f, (x, y), max(3, h // 30), (255, 255, 255), -1)
    return f


def _branch_diff(slider, value, moving=False, settle=180, frames=150):
    """Mean abs luminance difference after `frames` steps (5 s at 30 fps)
    between slider-at-0 and slider-at-value, branched from one settled
    state. Both branches see identical input frames."""
    host0, m0 = _boot()
    host1, m1 = _boot()
    f = _scene(*GRID)
    for _ in range(settle):
        m0.step(f, None, 1 / 30)
        m1.step(f, None, 1 / 30)
    setattr(host1.ui, "ph_" + slider, value)
    for i in range(frames):
        fr = _swept(f, i) if moving else f
        m0.step(fr, None, 1 / 30)
        m1.step(fr, None, 1 / 30)
    a, b = m0.pf.luminance(), m1.pf.luminance()
    m0.stop()
    m1.stop()
    return float(np.abs(a - b).mean())


# Floors sit at ~30-40% of the values measured at pinning time (weave
# 0.19/0.22/0.26, evolve 0.26/0.26/0.27, react 0.041/0.038/0.047) so the
# contract survives library-version jitter while still failing loudly if a
# slider ever goes back to imperceptible.

@pytest.mark.parametrize("value,floor", [(0.3, 0.06), (0.6, 0.08), (1.0, 0.08)])
def test_weave_perceptible_in_5s(value, floor):
    assert _branch_diff("weave", value) >= floor


@pytest.mark.parametrize("value,floor", [(0.3, 0.08), (0.6, 0.08), (1.0, 0.08)])
def test_evolve_perceptible_in_5s(value, floor):
    assert _branch_diff("evolve", value) >= floor


@pytest.mark.parametrize("value,floor", [(0.3, 0.012), (0.6, 0.012), (1.0, 0.015)])
def test_react_perceptible_in_5s(value, floor):
    assert _branch_diff("react", value, moving=True) >= floor


# ---------- P2: regime scheduler ----------

def test_regime_table_is_sane():
    """Every regime is a bundle of finite multipliers around 1 (jitter
    additive >= 0), cruise is the identity, and there are enough regimes
    for randomized order to matter."""
    assert len(REGIME_NAMES) >= 4
    assert REGIMES["cruise"] == {}
    for name in REGIME_NAMES:
        for k, v in REGIMES[name].items():
            assert k in REGIME_KEYS + ("jitter",), (name, k)
            assert 0.1 <= v <= 4.0 or (k == "jitter" and 0.0 <= v <= 1.0)


def test_evolve_zero_holds_one_structure():
    """evolve 0: no regime modulation, no melt — mods stay identity and the
    keep map stays None (react 0 too)."""
    host, mode = _boot(weave=0.5, evolve=0.0)
    calls = []
    orig = mode.pf.update

    def spy(m, g, keep=None):
        calls.append(keep)
        return orig(m, g, keep)
    mode.pf.update = spy
    f = _scene(*GRID)
    for _ in range(30):
        mode.step(f, None, 1 / 30)
    pf = mode.pf
    assert pf.mod_sense == pf.mod_turn == pf.mod_spread == 1.0
    assert pf.mod_step == pf.mod_deposit == 1.0
    assert all(k is None for k in calls)
    mode.stop()


def test_regime_hops_have_bounded_random_dwell():
    """Scheduler advances through distinct regimes; every sampled dwell sits
    inside REGIME_DWELL; the sequence never repeats a regime back-to-back."""
    host, mode = _boot(evolve=1.0)
    f = _scene(*GRID)
    seen = []
    # big dt strides through several dwells quickly (clock is dt-summed)
    for _ in range(200):
        mode.step(f, None, 1.0)
        assert REGIME_DWELL[0] <= mode._reg_dwell <= REGIME_DWELL[1]
        if not seen or seen[-1] != mode._reg_to:
            seen.append(mode._reg_to)
    assert len(seen) >= 4                        # it actually hops
    for prev, cur in zip(seen, seen[1:]):
        assert prev != cur                       # decisive, not a rename
    mode.stop()


def test_regime_modulation_is_applied():
    """With evolve up, the engine's structural multipliers leave identity —
    the regimes actually reach the sim."""
    host, mode = _boot(evolve=1.0)
    f = _scene(*GRID)
    left_identity = False
    for _ in range(120):
        mode.step(f, None, 0.5)
        pf = mode.pf
        if any(abs(m - 1.0) > 0.05 for m in
               (pf.mod_sense, pf.mod_turn, pf.mod_spread,
                pf.mod_step, pf.mod_deposit)):
            left_identity = True
            break
    assert left_identity
    mode.stop()


# ---------- P3: staleness melt (signed keep map) ----------

def test_negative_keep_melts_cpu():
    """keep=-1 must decay trail FASTER than keep=0 (the melt side of the
    signed map), bounded by MELT_DROP / MELT_FLOOR."""
    def run(keep_value):
        pf = PhysarumField(n=5_000, gw=96, gh=54, seed=3, decay=0.94,
                           reseed_frac=0.0)
        m = np.zeros((54, 96), np.float32)
        keep = np.full((54, 96), keep_value, np.float32)
        for _ in range(60):
            pf.update(m, m, keep)
        return float(pf.trail.sum())
    base, melted = run(0.0), run(-1.0)
    # deposits continue during the melt, so the bound is the equilibrium
    # ratio (1-decay)/(1-decay+MELT_DROP) ~= 1/3, not a bare decay race
    assert melted < 0.4 * base, (base, melted)
    # constants stay in the sane band the shader mirrors
    assert 0.0 < MELT_DROP < 0.3 and 0.5 < MELT_FLOOR < 0.9 < KEEP_HOLD < 1.0


def test_melt_engages_on_a_stale_scene():
    """On a still scene with evolve up, the mode's keep map goes negative
    somewhere within ~15 s — locked structure starts re-fluidizing."""
    host, mode = _boot(weave=0.5, evolve=0.8)
    mins = []
    orig = mode.pf.update

    def spy(m, g, keep=None):
        mins.append(0.0 if keep is None else float(keep.min()))
        return orig(m, g, keep)
    mode.pf.update = spy
    f = _scene(*GRID)
    for _ in range(450):
        mode.step(f, None, 1 / 30)
    mode.stop()
    assert min(mins) < -0.05, min(mins)


def test_depth_does_not_move_the_staleness_tracker():
    """Depth is lighting on the picture, not a property of the organism, so
    it must stay out of the simulation's feedback. On one FROZEN trail with
    the key light orbiting, the tracker's stale map is identical at depth 0
    and depth 0.9. (Review 2026-09-24: the tracker read the relief-lit
    picture, whose dark flanks and orbiting light cut the mean stale mass
    by ~37% at depth 0.9 with the organism unchanged.)"""
    from dtouch.modes.physarum import MELT_VAR

    def stale_map(depth):
        host, mode = _boot(weave=0.5, evolve=0.8)
        f = _scene(*GRID)
        for _ in range(120):
            mode.step(f, None, 1 / 30)
        mode.pf.update = lambda *a, **k: None     # freeze the organism
        host.ui.ph_depth = depth
        for _ in range(150):                      # 5 s of orbit
            mode.step(f, None, 1 / 30)
        assert mode._orbit_t > 4.0                # the light did move
        ema, var = mode._ema.copy(), mode._var.copy()
        mode.stop()
        return (np.clip(1.0 - var / MELT_VAR, 0.0, 1.0)
                * np.clip((ema - 0.25) * 5.0, 0.0, 1.0))

    flat, deep = stale_map(0.0), stale_map(0.9)
    assert flat.mean() > 0.01, flat.mean()        # non-vacuous: something is stale
    np.testing.assert_array_equal(flat, deep)


def _gl_field(**kw):
    kw.setdefault("n", 4000)
    kw.setdefault("gw", 96)
    kw.setdefault("gh", 54)
    kw.setdefault("seed", 7)
    try:
        return PhysarumFieldGL(**kw)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")


def test_negative_keep_melts_gl_parity():
    """The melt side of the signed keep map behaves the same on the GPU:
    keep=-1 holds the settled trail mass well below keep=0's, and the
    melted/base ratio matches the CPU engine's (same character, the
    constants are mirrored in blur.frag)."""
    def run(engine_cls, keep_value):
        pf = engine_cls(n=4000, gw=96, gh=54, seed=3, decay=0.94,
                        reseed_frac=0.0)
        try:
            m = np.zeros((54, 96), np.float32)
            keep = np.full((54, 96), keep_value, np.float32)
            for _ in range(60):
                pf.update(m, m, keep)
            return float(pf.trail.sum())
        finally:
            pf.release()

    def ratio(engine_cls):
        if engine_cls is PhysarumFieldGL:
            base = run(_gl_field_cls(), 0.0)
            melted = run(_gl_field_cls(), -1.0)
        else:
            base, melted = run(engine_cls, 0.0), run(engine_cls, -1.0)
        return melted / max(base, 1e-9)

    def _gl_field_cls():
        # constructing probes availability once; skip cleanly when absent
        f = _gl_field(n=16, gw=8, gh=8)
        f.release()
        return PhysarumFieldGL
    r_gl = ratio(PhysarumFieldGL)
    r_cpu = ratio(PhysarumField)
    assert r_gl < 0.4, r_gl
    assert abs(r_gl - r_cpu) < 0.12, (r_gl, r_cpu)


def test_mod_step_zero_freezes_gl_agents():
    """mod_step reaches the GPU update pass: zero stride leaves every agent
    exactly where it was (turning in place), with reseed off."""
    f = _gl_field(reseed_frac=0.0)
    try:
        px0, py0 = f.px.copy(), f.py.copy()
        f.mod_step = 0.0
        m = np.zeros((f.gh, f.gw), np.float32)
        for _ in range(3):
            f.update(m, m)
        assert np.allclose(f.px, px0) and np.allclose(f.py, py0)
    finally:
        f.release()


def test_mod_deposit_scales_gl_deposits():
    """mod_deposit reaches the GPU deposit pass: tripling it triples the
    first frame's laid mass (fresh fields, same seed)."""
    def one_frame(mod):
        f = _gl_field(reseed_frac=0.0, decay=1.0, diffuse=0)
        try:
            f.mod_deposit = mod
            m = np.zeros((f.gh, f.gw), np.float32)
            f.update(m, m)
            return float(f.trail.sum())
        finally:
            f.release()
    a, b = one_frame(1.0), one_frame(3.0)
    assert b == pytest.approx(3.0 * a, rel=1e-3)


def test_gpu_rack_path_feeds_staleness_tracker():
    """With the GL engine and the SIGNAL rack ON (the no-readback fast
    path), the staleness tracker still gets its input — from the stats
    subsample, not a full luminance readback — so melt keeps working."""
    from test_rack_gl import _blob_frames, _boot_gl_mode, _signal_ui
    from dtouch.circuit_bent import CircuitBent
    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=True)
    host.ui.ph_evolve = 0.8
    host.cb = CircuitBent(seed=42)
    try:
        for f in _blob_frames(10):
            m.step(f, None, 1 / 30)
        assert m.signal_done, "GPU rack path did not engage"
        assert m._last_lum is not None
        assert m._last_lum.shape == (m.pf.sh, m.pf.sw)   # the stats grid
        assert m._ema is not None                        # tracker running
    finally:
        m.stop()


# ---------- P4: structural modulators reach both blend paths ----------

def test_mod_step_and_deposit_scale_cpu_blend():
    """mod_step / mod_deposit multiply the blended point parameters."""
    pf = PhysarumField(n=1_000, gw=64, gh=36, seed=1)
    t = np.zeros(1_000, np.float32)
    base = pf._blend_params(t)
    pf.mod_step = 1.5
    pf.mod_deposit = 0.5
    mod = pf._blend_params(t)
    assert np.allclose(mod["step"], base["step"] * 1.5)
    assert np.allclose(mod["deposit"], base["deposit"] * 0.5)


# ---------- P5: the Z random cast ----------

def _settle_feats(mode, f, frames=200):
    for _ in range(frames):
        mode.step(f, None, 1 / 30)
    lum = mode.pf.luminance()
    binary = (lum > 0.35).astype(np.uint8)
    dist = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 3)
    on = dist[dist > 0]
    w50 = float(np.median(on) * 2) if on.size else 0.0
    return lum, np.array([float(binary.mean()) * 4.0,
                          min(w50, 20.0) / 5.0,
                          float(lum.mean()) * 4.0], np.float32)


def test_random_cast_always_pays():
    """N successive Z casts: every landing is a live, non-degenerate picture
    (variance and coverage floors) and successive landings stay apart in
    morphology space — a slot machine that always pays, never a lemon."""
    host, mode = _boot(weave=0.6, evolve=0.5)
    f = _scene(*GRID)
    for _ in range(150):
        mode.step(f, None, 1 / 30)
    prev = None
    dists = []
    for _ in range(5):
        mode.cast_random()
        lum, feats = _settle_feats(mode, f)
        assert float(lum.std()) > 0.08, "degenerate: flat picture"
        assert 0.03 < float((lum > 0.35).mean()) < 0.92, "degenerate coverage"
        if prev is not None:
            dists.append(float(np.linalg.norm(feats - prev)))
        prev = feats
    assert np.mean(dists) >= 0.15, dists          # successive casts differ
    mode.stop()


def test_random_cast_draws_from_validated_bands():
    """Casts land inside the validated bands and force a decisive regime
    transition (fresh hop clock + melt surge), with a real point pairing."""
    host, mode = _boot()
    for _ in range(3):
        bg, fg = mode.cast_random()
        ui = host.ui
        assert bg != fg
        assert (bg, fg) == (ui.ph_point_bg_idx, ui.ph_point_fg_idx)
        assert 0.35 <= ui.ph_weave <= 0.90
        assert 0.45 <= ui.ph_evolve <= 0.90
        assert 0.40 <= ui.ph_react <= 0.90
        assert 0.88 <= ui.ph_decay <= 0.96
        assert 0.80 <= ui.ph_gain <= 1.60
        assert mode._reg_t == 0.0
        assert mode._melt_pulse == 1.0
        assert mode._reg_from != mode._reg_to
    mode.stop()


def test_random_cast_command_registered():
    """Z is wired as a mode command and toasts what it landed on."""
    host, mode = _boot()
    mode.host.ui.ph_point_bg_idx = 0
    mode.host.ui.ph_point_fg_idx = 2
    cmds = mode.commands()
    assert "physarum.random" in cmds
    assert cmds["physarum.random"].key == "z"
    cmds["physarum.random"].run()
    mode.stop()


# ---------- P6: quality-tier diffuse trim is rounding-consistent ----------

def test_diffuse_trim_rounds_once():
    """The weave diffuse trim computes in float from the unrounded scaled
    base and rounds ONCE — at 'quality' px_scale the trim used to bite a
    full extra pixel (relatively thinner veins than 'perform')."""
    host, mode = _boot(weave=0.31)
    f = _scene(*GRID)
    for scale, gw in ((2.2222, 1280), (4.4444, 2560)):
        mode._base_diffuse = scale               # GL bases stay float now
        mode.step(f, None, 1 / 30)
        expect = max(1, round(scale * (1.0 - 0.45 * 0.31 ** 0.6)))
        assert mode.pf.diffuse == expect, (scale, mode.pf.diffuse, expect)
    mode.stop()


# ---------- P7: the veins stay veins ----------

def _vein_contrast(seed, weave=0.6, evolve=0.5, frames=300):
    """p99/median of the rendered luminance at the shipped default feel.

    docs/ALIVENESS.md defines the original bug partly as "measured vein/floor
    contrast lands near 2-4x where a network normally reaches 20-50x. Veins
    need dark to be veins."  Nothing measured it, so it was free to drift —
    and it did: an audit caught a 7.5x collapse that arrived as a side effect
    of satisfying the slider floors above, with no test and no mention.
    """
    host, mode = _boot(weave=weave, evolve=evolve, seed=seed)
    f = _scene(*GRID)
    for _ in range(frames):
        mode.step(f, None, 1 / 30)
    lum = mode.pf.luminance()
    mode.stop()
    return float(np.percentile(lum, 99) / max(float(np.median(lum)), 1e-6))


def test_the_default_look_keeps_its_veins():
    """A floor, not a target. Measured 10.6 at the shipped mapping on this
    rig; it read 16.1 before the feel curve was steepened to keep the sliders
    perceptible at 0.3, and that trade was made deliberately and recorded.
    What this stops is the NEXT change quietly flattening it further: the
    broken regime this project started from measured 2-4.
    """
    got = float(np.mean([_vein_contrast(s) for s in (7, 11, 23)]))
    assert got >= 6.0, f"vein contrast collapsed to {got:.1f} (floor 6.0)"
