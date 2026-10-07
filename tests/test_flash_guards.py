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
from dtouch.shell import SCENE_COOLDOWN_S, SCENE_KEY_COMMANDS, Host

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


def test_the_new_scene_keys_are_gated():
    assert SCENE_COOLDOWN_S == 0.5
    for name in ("menu.open", "physarum.swap", "bender.effect", "bender.amount",
                 "bender.split", "bender.copy", "bender.sort", "bender.long"):
        assert name in SCENE_KEY_COMMANDS, name


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
