"""Flash guards: no key, held or spammed, strobes the picture.

The owner's hard floor is 3 flashes a second (WCAG 2.3.1; a flash is a pair
of opposing >10% luminance changes). Each test holds a key the way waitKey
repeats it (a press every other frame, 30 a second) for 2 s, through the
shell's real routing (`_route_key`), and reads the composed frame's mean
luminance (blackout and the menu included, as the window shows it):
- blackout turns ON at once (a safety cut) and OFF at most once per
  SCENE_COOLDOWN_S after the last on, so at most 2 flashes a second;
- the menu (m, and m / Esc closing it), physarum's swap (x) and Circuit
  Bender's cycle keys (e, b, x, c, k, l) take one change per cooldown.
Each guard test fails with its guard removed (checked by hand when written:
held space jumped 17 to 59 times in 2 s, held m 59). GPU tests skip where
no GL context can be made (CI).
"""
import numpy as np
import pytest

from dtouch.modes.bender import BenderMode
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.modes.physarum import PhysarumMode
from dtouch.shell import FREE_COMMANDS, SCENE_COOLDOWN_S, Host

from test_physarum_ink import _booted, _paths
from test_shell import SyntheticSource

FLOOR_JUMPS_PER_S = 2 * 2      # 2 flashes a second, under the floor of 3


def _lum(bgr):
    o = bgr.astype(np.float32)
    return float((o[..., 2] * 0.2126 + o[..., 1] * 0.7152 + o[..., 0] * 0.0722).mean()) / 255.0


def _drive(host, key, frames=120, every=2, read=None):
    """Hold `key` (a press every `every` frames at 60 fps) on the shell's
    clock; return (luminance per composed frame, `read()` per frame)."""
    m = host.mode
    # one clock per host that only runs forward (a gate stamped in one run
    # must not look like the future to the next)
    t = host.__dict__.setdefault("_test_clock", [1000.0])
    t[0] += 1.0
    host.scene_clock = lambda: t[0]
    if hasattr(m, "_clock"):
        m._clock = lambda: t[0]
    cam = np.full((108, 192, 3), 128, np.uint8)

    def one():
        out = m.step(cam, None, 1 / 60).copy()
        if host.ps.blackout:
            out[:] = 0
        bgr = host._compose_frame(out, cam)
        t[0] += 1 / 60
        return _lum(bgr)

    for _ in range(30):
        if host.ui.pending_preset:
            host._apply_pending_preset()
        one()
    lums, seen = [], []
    for i in range(frames):
        if i % every == 0:
            host._route_key(key)
        if host.ui.pending_preset:
            host._apply_pending_preset()
        lums.append(one())
        seen.append(read() if read else None)
    return lums, seen


def _jumps(lums):
    return sum(1 for a, b in zip(lums, lums[1:]) if abs(b - a) > 0.1)


def _changes(xs):
    return sum(1 for a, b in zip(xs, xs[1:]) if a != b)


def _reset(host):
    host.ps.blackout = False
    if host.menu.open:
        host.menu.close()


EXPECTED_FREE = {
    "output.blackout", "preset.panic",
    "param.prev", "param.next", "param.down", "param.up",
    "param.down.big", "param.up.big",
    "preset.save", "record.toggle", "audio.toggle", "debug.toggle", "app.quit",
    "physarum.burst", "physarum.wave", "layer.flock",
}


def test_the_allowlist_is_the_only_exemption(tmp_path):
    """Gated by default: in every mode, every registered command is either
    on the reviewed allowlist or wrapped by the shared flash budget. A new
    binding that nobody argued free is gated without anyone remembering."""
    assert SCENE_COOLDOWN_S == 0.5
    assert set(FREE_COMMANDS) == EXPECTED_FREE
    from dtouch.modes import REGISTRY
    seen = set()
    for cls in REGISTRY:
        host = Host(cls(), source=SyntheticSource(), res=(192, 108), show=False,
                    preset=None, max_frames=2, **_paths(tmp_path))
        host.run()
        try:
            host._wire_keys()
            for cmd in host.reg.commands():
                seen.add(cmd.name)
                gated = getattr(cmd.run, "_scene_gated", False)
                assert gated != (cmd.name in FREE_COMMANDS), (cls.id, cmd.name)
        finally:
            stop = getattr(host.mode, "really_stop", None) or host.mode.stop
            stop()
    # the allowlist names real commands (a typo would exempt nothing and
    # leave the intended command gated, which is safe, but stale)
    assert EXPECTED_FREE <= seen, EXPECTED_FREE - seen


def test_held_space_turns_blackout_on_at_once_and_flashes_at_most_twice_a_second(tmp_path):
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host._apply_look("amoeba", PhysarumMode.BUILTIN["amoeba"])
        # ON is instant, whatever came before
        host.scene_clock = lambda: 5.0
        host._route_key(ord(" "))
        assert host.ps.blackout is True
        host._route_key(ord(" "))                 # OFF inside the hold: held
        assert host.ps.blackout is True
        _reset(host)
        lums, bo = _drive(host, ord(" "), read=lambda: host.ps.blackout)
        assert _changes(bo) >= 2, "it really toggles"
        assert _changes(bo) <= FLOOR_JUMPS_PER_S * 2
        if m.engine == "gl":
            assert _jumps(lums) <= FLOOR_JUMPS_PER_S * 2, _jumps(lums)
    finally:
        _reset(host)
        m.really_stop()


def test_held_m_cannot_flicker_the_menu(tmp_path):
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        lums, opened = _drive(host, ord("m"), read=lambda: host.menu.open)
        assert _changes(opened) >= 2, "the menu really opens and closes"
        assert _changes(opened) <= FLOOR_JUMPS_PER_S * 2
        assert _jumps(lums) <= FLOOR_JUMPS_PER_S * 2
    finally:
        _reset(host)
        m.really_stop()


def test_held_x_swaps_at_most_twice_a_second(tmp_path):
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host._apply_look("amoeba", PhysarumMode.BUILTIN["amoeba"])
        lums, pts = _drive(host, ord("x"), read=lambda: (host.ui.ph_point_bg_idx,
                                                         host.ui.ph_point_fg_idx))
        assert 2 <= _changes(pts) <= FLOOR_JUMPS_PER_S * 2
        if m.engine == "gl":
            assert _jumps(lums) <= FLOOR_JUMPS_PER_S * 2
    finally:
        m.really_stop()


def test_dither_held_space_and_m(tmp_path):
    host = Host(DitherGirlMode(), source=SyntheticSource(), res=(192, 108), show=False,
                preset=None, max_frames=2, **_paths(tmp_path))
    host.run()
    try:
        host._wire_keys()
        for key, read in ((ord(" "), lambda: host.ps.blackout),
                          (ord("m"), lambda: host.menu.open)):
            lums, seen = _drive(host, key, read=read)
            assert 2 <= _changes(seen) <= FLOOR_JUMPS_PER_S * 2, chr(key)
            assert _jumps(lums) <= FLOOR_JUMPS_PER_S * 2, chr(key)
            _reset(host)
    finally:
        host.mode.stop()


def test_circuit_bender_cycle_keys_hold_to_twice_a_second(tmp_path):
    host = Host(BenderMode(), source=SyntheticSource(), res=(192, 108), show=False,
                preset=None, max_frames=2, **_paths(tmp_path))
    host.run()
    m = host.mode
    try:
        host._wire_keys()
        reads = {"e": "bd_effect_idx", "b": "bd_amount", "x": "bd_split_idx",
                 "c": "bd_copy_idx", "k": "bd_sort_idx", "l": "bd_long_idx"}
        for ch, attr in reads.items():
            lums, seen = _drive(host, ord(ch), frames=60, read=lambda a=attr: getattr(host.ui, a))
            assert 1 <= _changes(seen) <= FLOOR_JUMPS_PER_S, (ch, _changes(seen))
            assert _jumps(lums) <= FLOOR_JUMPS_PER_S, (ch, _jumps(lums))
    finally:
        _reset(host)
        stop = getattr(m, "really_stop", None) or m.stop
        stop()


def _windows(lums, fps=60):
    """Most >10% jumps in any 1 s window."""
    flags = [1 if abs(b - a) > 0.1 else 0 for a, b in zip(lums, lums[1:])]
    return max((sum(flags[i:i + fps]) for i in range(max(1, len(flags) - fps + 1))),
               default=0)


def _drive_seq(host, keys, frames=180, every=2):
    """Press `keys` round-robin, one every `every` frames, on the shell's
    clock; return per-frame composed luminance."""
    m = host.mode
    t = host.__dict__.setdefault("_test_clock", [1000.0])
    t[0] += 1.0
    host.scene_clock = lambda: t[0]
    if hasattr(m, "_clock"):
        m._clock = lambda: t[0]
    cam = np.full((108, 192, 3), 128, np.uint8)
    lums, k = [], 0
    for i in range(frames):
        if i % every == 0:
            host._route_key(keys[k % len(keys)])
            k += 1
        if host.ui.pending_preset:
            host._apply_pending_preset()
        out = m.step(cam, None, 1 / 60).copy()
        if host.ps.blackout:
            out[:] = 0
        if host.ps.help_open:
            out[:] = 255          # stand-in for the modal (a full-frame change)
        lums.append(_lum(host._compose_frame(out, cam)))
        t[0] += 1 / 60
    return lums


def test_space_and_zero_alternating_respects_the_blackout_hold(tmp_path):
    """0 (panic) used to turn blackout off at once, so space/0 at 30 presses
    a second strobed 12-15 times a second. Panic still resets at once, but
    a blackout inside its hold stays on."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host._apply_look("amoeba", PhysarumMode.BUILTIN["amoeba"])
        host.scene_clock = lambda: 10.0
        host._route_key(ord(" "))
        assert host.ps.blackout is True
        host._route_key(ord("0"))                 # inside the hold
        assert host.ps.blackout is True
        assert host.ui.pending_preset == m.safe_look()   # the reset still ran
        host.scene_clock = lambda: 10.6
        host._route_key(ord("0"))                 # after the hold: free
        assert host.ps.blackout is False
        _reset(host)
        seen = []
        t = [20.0]
        host.scene_clock = lambda: t[0]
        for i in range(60):                       # 30 presses / s, 2 s
            t[0] += 1 / 30
            host._route_key(ord(" ") if i % 2 == 0 else ord("0"))
            seen.append(host.ps.blackout)
        assert 2 <= _changes(seen) <= 4 * 2, _changes(seen)
    finally:
        _reset(host)
        m.really_stop()


def test_a_mode_switch_does_not_reopen_the_blackout_hold(tmp_path):
    """The hold lives on the Host: the registry rebuilt on a switch keeps
    it (it used to reset with every _wire_keys)."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host.scene_clock = lambda: 30.0
        host._route_key(ord(" "))
        assert host.ps.blackout is True
        host._wire_keys()                         # what a mode switch does
        host.scene_clock = lambda: 30.2
        host._route_key(ord(" "))
        assert host.ps.blackout is True           # still held
    finally:
        _reset(host)
        m.really_stop()


def test_held_question_mark_cannot_flicker_help(tmp_path):
    """? opened help and any key closed it, so a held ? toggled it on every
    repeat (15 flashes a second). Open and close now share the budget."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        t = [40.0]
        host.scene_clock = lambda: t[0]
        seen = []
        for _ in range(60):                       # 30 presses / s, 2 s
            t[0] += 1 / 30
            host._route_key(ord("?"))
            seen.append(host.ps.help_open)
        assert 2 <= _changes(seen) <= 4, _changes(seen)
    finally:
        host.ps.help_open = False
        m.really_stop()


def test_held_cycle_row_nudges_take_the_budget_and_sliders_stay_free(tmp_path):
    host = Host(DitherGirlMode(), source=SyntheticSource(), res=(192, 108), show=False,
                preset=None, max_frames=2, **_paths(tmp_path))
    host.run()
    try:
        host._wire_keys()
        t = [60.0]
        host.scene_clock = lambda: t[0]
        host.ui.nudge_attr = "dg_algo_idx"
        seen = []
        for _ in range(60):                       # held '=', 30 / s, 2 s
            t[0] += 1 / 30
            host.reg.dispatch(ord("="))
            seen.append(host.ui.dg_algo_idx)
        assert 2 <= _changes(seen) <= 4, _changes(seen)
        # a slider on the same keys is not a cut: every press moves it
        slider = next(w for w in host.ui.iter_widgets()
                      if hasattr(w, "lo") and getattr(w, "attr", None)
                      and not getattr(w, "options", None) and not getattr(w, "step", None))
        host.ui.nudge_attr = slider.attr
        setattr(host.ui, slider.attr, slider.lo)
        host.reg.dispatch(ord("="))               # re-anchor press may only show
        moved = []
        for _ in range(5):
            t[0] += 1 / 30
            host.reg.dispatch(ord("="))
            moved.append(float(getattr(host.ui, slider.attr)))
        assert _changes(moved) >= 4, (slider.attr, moved)
    finally:
        host.mode.stop()


def test_the_menu_close_gate_holds_on_its_own(tmp_path):
    """The close is gated by itself, not only by the open: with the menu
    already open, m / Esc spammed closes it once and then waits."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host.scene_clock = lambda: 70.0
        host.menu.toggle(m.id)                    # opened outside the budget
        host._scene_key_t = 70.0                  # ...but a change just landed
        assert host.menu.open is True
        host.scene_clock = lambda: 70.1
        host._route_key(27)                       # Esc inside the budget: held
        assert host.menu.open is True
        host._route_key(ord("m"))
        assert host.menu.open is True
        host.scene_clock = lambda: 70.6
        host._route_key(27)
        assert host.menu.open is False
    finally:
        _reset(host)
        m.really_stop()


def test_combinations_share_one_budget(tmp_path):
    """Separate per-key gates added up (h + k reached 2/s from two gates,
    4/s in theory). One budget for every path: any mix of keys held at
    repeat rate stays within 3 flashes (6 jumps) in every 1 s window, and
    the frame-changing presses that land are at most 2 a second."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        host._wire_keys()
        host._apply_look("veinwork", PhysarumMode.BUILTIN["veinwork"])
        host._apply_pending_preset()
        combos = ("hk", "xc", " m", " x", "hy", "x0", "06", "?h", " 0h")
        for combo in combos:
            keys = [ord(c) for c in combo]
            lums = _drive_seq(host, keys)
            if m.engine == "gl" or "?" in combo or " " in combo:
                assert _windows(lums) <= 6, (combo, _windows(lums))
            _reset(host)
            host.ps.help_open = False
        # counts that do not need a GPU: h and k together land 2 a second
        t = [90.0]
        host.scene_clock = lambda: t[0]
        m._clock = lambda: t[0]
        seen = []
        for i in range(60):
            t[0] += 1 / 30
            host._route_key(ord("hk"[i % 2]))
            seen.append((host.ui.ph_depth, host.ui.ph_fractal, m.ink_paper))
        assert _changes(seen) <= 4, _changes(seen)
    finally:
        _reset(host)
        host.ps.help_open = False
        m.really_stop()
