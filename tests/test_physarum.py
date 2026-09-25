"""Physarum mode + engine — protocol conformance, video coupling, key safety.

The mode tests mirror test_dithergirl.py (the full-frame-image exemplar). The
key-uniqueness test exists because mode-switch keys are only registered when a
window opens (show=True): a collision passes the whole headless suite and
kills the real app at boot, so it is asserted here at the derivation level.
"""
import numpy as np
import pytest

from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.physarum import (ACCENT, PALETTES_PH, PhysarumMode, _fmt_agents,
                                   _palette_lut)
from dtouch.physarum import POINT_NAMES, POINTS, PhysarumField
from dtouch.shell import Host

from test_shell import SyntheticSource

RES = (192, 108)


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(PhysarumMode(), source=SyntheticSource(), **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _white(h=36, w=64):
    return np.full((h, w, 3), 255, np.uint8)


# ---------- mode protocol conformance (DESIGN.md §2.2) ----------

def test_mode_protocol_attrs():
    m = PhysarumMode()
    assert m.id == "physarum" and m.title == "Physarum"
    assert m.accepts_still is False
    assert m.accent == ACCENT
    assert m.safe_look() in m.BUILTIN


def test_registered_in_registry():
    assert PhysarumMode in REGISTRY
    assert mode_by_id("physarum") is PhysarumMode


def test_mode_switch_keys_unique_and_off_the_global_row():
    """id[:1] / the `key` class attr must not collide — the registry raises
    only when a window opens, which no headless test exercises."""
    keys = [getattr(m, "key", m.id[:1]) for m in REGISTRY]
    assert len(set(keys)) == len(keys), f"mode-switch keys collide: {keys}"
    globals_taken = set("sgariqm")          # preset.save, glitch, audio,
    assert not set(keys) & globals_taken    # record, debug, quit, menu


def test_start_stop_idempotent_and_headless(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    m.stop()
    m.stop()                                # second stop must be a no-op
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_step_without_ui_uses_defaults(tmp_path):
    """The soak path: step before the shell builds the shared UI state."""
    host = _host(tmp_path)
    m = PhysarumMode()
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_status_line_is_spec_derived_and_ascii(tmp_path):
    """The tail names the engine that actually booted (gl on a GPU box, cpu
    on CI after the fallback) and its agent count."""
    host = _booted(tmp_path)
    s = host._status_line()
    assert s == s.encode("ascii", "replace").decode()
    m = host.mode
    assert m.engine in ("gl", "cpu")
    assert s == ("PHYSARUM  matte auto  body fingers  field veins  arctic"
                 f"  {m.engine} {_fmt_agents(m.n)}  cam synthetic")


def test_panel_sections():
    spec = PhysarumMode().panel_spec()
    assert [sec.title for sec in spec] == ["TEMPLATES", "SOURCE", "MOLD", "LOOK"]


def test_swap_command_flips_the_points(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.ph_point_fg_idx, ui.ph_point_bg_idx = 2, 0
    host.mode.commands()["physarum.swap"].run()
    assert (ui.ph_point_fg_idx, ui.ph_point_bg_idx) == (0, 2)


def test_audio_pulses_the_picture(tmp_path):
    """Bass raises exposure for the frame — the engine must see it."""
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)                           # run()'s finally stopped it
    m.step(_white(), {"bass": 1.0, "treble": 0.0}, 1 / 30)
    loud = m.pf.exposure
    m.step(_white(), {"bass": 0.0, "treble": 0.0}, 1 / 30)
    assert loud > m.pf.exposure


# ---------- engine (dtouch.physarum) ----------

def _field(**kw):
    kw.setdefault("n", 4000)
    kw.setdefault("gw", 96)
    kw.setdefault("gh", 54)
    kw.setdefault("seed", 7)
    return PhysarumField(**kw)


def _flat(f, v=0.0):
    return np.full((f.gh, f.gw), v, np.float32)


def test_update_deposits_trail_and_luminance_is_bounded():
    f = _field()
    for _ in range(5):
        f.update(_flat(f), _flat(f))
    assert f.trail.sum() > 0
    lum = f.luminance()
    assert lum.shape == (f.gh, f.gw) and lum.dtype == np.float32
    assert float(lum.min()) >= 0.0 and float(lum.max()) <= 1.0
    assert np.isfinite(f.trail).all()


def test_matte_is_the_pen_blends_toward_body_point():
    """matte=0 runs the field point, matte=1 the body point — stride follows.
    haze steps 0.7/frame, fingers 2.0, so mean per-frame displacement must
    roughly double between the two (wrap-aware distance)."""
    def mean_step(matte_value):
        # reseed off: teleports are displacement noise on top of the stride
        # being measured (haze's 0.35 px stride would drown in them)
        f = _field(point_bg="haze", point_fg="fingers", reseed_frac=0.0)
        m = _flat(f, matte_value)
        g = _flat(f)
        f.update(m, g)                       # warm-up (headings settle)
        px, py = f.px.copy(), f.py.copy()
        f.update(m, g)
        dx = np.minimum(np.abs(f.px - px), f.gw - np.abs(f.px - px))
        dy = np.minimum(np.abs(f.py - py), f.gh - np.abs(f.py - py))
        return float(np.hypot(dx, dy).mean())

    slow, fast = mean_step(0.0), mean_step(1.0)
    assert slow == pytest.approx(POINTS["haze"]["step"], rel=0.15)
    assert fast == pytest.approx(POINTS["fingers"]["step"], rel=0.15)


def test_luminance_is_food_draws_the_mold():
    """A bright band in the footage must accumulate more trail than the dark
    rest of the frame once food coupling is on."""
    f = _field(food=1.0, reseed_frac=0.05)
    gray = _flat(f)
    band = slice(f.gw // 3, 2 * f.gw // 3)
    gray[:, band] = 1.0
    for _ in range(40):
        f.update(_flat(f), gray)
    inside = float(f.trail[:, band].mean())
    outside = float(np.delete(f.trail, np.s_[band], axis=1).mean())
    assert inside > outside * 1.3


def test_agents_stay_on_grid_even_from_the_edge():
    """float32 wrap can land exactly ON the bound — indices must never
    escape. Park every agent at the last representable spot and step."""
    f = _field()
    f.px[:] = np.nextafter(np.float32(f.gw), np.float32(0))
    f.py[:] = np.nextafter(np.float32(f.gh), np.float32(0))
    f.update(_flat(f), _flat(f))             # must not raise
    assert (f.px >= 0).all() and (f.py >= 0).all()


def test_same_seed_same_trail():
    a, b = _field(), _field()
    for _ in range(3):
        a.update(_flat(a), _flat(a))
        b.update(_flat(b), _flat(b))
    assert np.array_equal(a.trail, b.trail)


def test_builtin_points_and_palettes_are_wired():
    for name in PhysarumMode.BUILTIN:
        look = PhysarumMode.BUILTIN[name]
        assert look["point_bg"] in POINT_NAMES
        assert look["point_fg"] in POINT_NAMES
        assert look["palette"] in PALETTES_PH
    for pal in PALETTES_PH:
        if pal != "video":
            lut = _palette_lut(pal)
            assert lut.shape == (256, 3) and lut.dtype == np.uint8


# ---------- cheap-wins round: burst / wave / palettes ----------

def test_spawn_burst_concentrates_agents():
    f = _field()
    f.spawn_burst(20.0, 20.0, frac=0.5, radius=2.0)
    d = np.hypot(f.px - 20.0, f.py - 20.0)
    assert (d < 8.0).mean() > 0.4          # half the pool landed on the point


def test_wave_points_everyone_outward():
    f = _field()
    f.wave(f.gw / 2, f.gh / 2)
    dx, dy = f.px - f.gw / 2, f.py - f.gh / 2
    outward = np.cos(f.heading) * dx + np.sin(f.heading) * dy
    assert (outward >= 0).mean() > 0.99


def test_burst_and_wave_commands_land_on_next_step(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    cmds = m.commands()
    cmds["physarum.burst"].run()
    cmds["physarum.wave"].run()
    assert m._burst_pending and m._wave_pending
    m.step(_white(), None, 1 / 30)
    assert not m._burst_pending and not m._wave_pending
    m.stop()


def test_depth_key_h_is_free_everywhere_and_not_a_scene_change(tmp_path):
    """h ('height') is claimed only by physarum.depth. Building the real
    command registry for this mode raises on any collision (shell globals,
    mode-switch letters, the mode's own commands); the menu's card keys are a
    separate router and are checked directly. It is a toggle, not a scene
    change, so it must not release autopilot."""
    from dtouch.menu import registry_cards
    from dtouch.shell import AUTO_RELEASE_KEYS
    host = _booted(tmp_path)
    host._wire_keys()                       # raises if any key is bound twice
    cmd = host.reg._by_key.get(ord("h"))
    assert cmd is not None and cmd.name == "physarum.depth"
    assert host.reg._by_key.get(ord("H")) is cmd
    assert "h" not in {c.key for c in registry_cards() if c.key}
    assert ord("h") not in AUTO_RELEASE_KEYS and ord("H") not in AUTO_RELEASE_KEYS


def test_depth_key_toggles_flat_and_back_to_the_last_depth(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    run = m.commands()["physarum.depth"].run
    ui.ph_depth = 0.9
    # a look without a fractal amount: two steps on either engine (the
    # three-step ladder is tests/test_physarum_fractal.py's)
    ui.ph_fractal = 0.0
    run()
    assert ui.ph_depth == 0.0
    assert host.hud.toasts._center.text == "FLAT"
    run()
    assert ui.ph_depth == 0.9
    assert host.hud.toasts._center.text == "DEPTH 0.9"


def test_depth_slider_defaults_and_panic(tmp_path):
    """Depth sits in LOOK after Grain, saves as `depth`, and every built-in
    look plus DEFAULTS (what panic and custom looks fall back to) carries
    one, so recalling a look or panicking sets it."""
    from dtouch.modes.physarum import DEPTH_DEFAULT, LOOK_DEPTH
    from dtouch.panelspec import apply_look
    look = [s for s in PhysarumMode().panel_spec() if s.title == "LOOK"][0]
    labels = [getattr(w, "label", None) for w in look.widgets]
    assert labels[labels.index("Grain") + 1] == "Depth"
    depth_w = look.widgets[labels.index("Depth")]
    assert depth_w.store_key == "depth" and (depth_w.lo, depth_w.hi) == (0.0, 1.0)
    assert PhysarumMode.DEFAULTS["depth"] == DEPTH_DEFAULT == 0.6
    assert PhysarumMode._UI_DEFAULTS["ph_depth"] == DEPTH_DEFAULT
    for name, cfg in PhysarumMode.BUILTIN.items():
        assert cfg["depth"] == LOOK_DEPTH[name]
    host = _booted(tmp_path)
    host.ui.ph_depth = 0.1
    apply_look(host.ui, host.ui.spec, {}, defaults=PhysarumMode.DEFAULTS)
    assert host.ui.ph_depth == DEPTH_DEFAULT


def test_random_cast_never_lands_flat(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    for _ in range(12):
        m.cast_random()
        assert 0.4 <= host.ui.ph_depth <= 1.0


def test_depth_reaches_the_cpu_engine_and_changes_the_picture(tmp_path):
    """The CPU fallback honours Depth too: the same grown field renders
    differently at depth 0 and at the default depth."""
    f = _field()
    for _ in range(40):
        f.update(np.zeros((f.gh, f.gw), np.float32),
                 np.full((f.gh, f.gw), 0.6, np.float32))
    trail, norm = f.trail.copy(), f._norm
    flat = f.luminance()
    f.trail, f._norm, f.depth = trail, norm, 0.6
    lit = f.luminance()
    vein = flat > 0.3
    assert vein.any()
    assert float(np.abs(lit - flat)[vein].mean()) >= 0.02


def test_mode_keys_do_not_collide_with_new_commands():
    host_keys = {"x", "b", "w", "h"}
    mode_switch = {getattr(m, "key", m.id[:1]) for m in REGISTRY}
    assert not host_keys & mode_switch
    assert not host_keys & set("sgariqm")
