"""Shell-level regression pins for the verified UI-overhaul bug list.

Everything here drives the real Host loop headless with DitherGirlMode (pure
numpy/cv2 — no GL, so these run everywhere test_dithergirl.py does). GUI-path
tests monkeypatch cv2's window calls so `show=True` code paths run headless.
"""
import json
import os
import time

import numpy as np
import pytest

import cv2

from dtouch import matte as matte_mod
from dtouch import presets
from dtouch import shell as shell_mod
from dtouch.hud import OverlayState
from dtouch.modes.dithergirl import MATTES_DG, DitherGirlMode
from dtouch.shell import PANEL_OPEN, CameraSource, Host

RES = (192, 108)


class SyntheticSource:
    """Deterministic frame source (seeded per read) with an optional per-read
    hook and a fail-from mode (fail_after=0 = never yields a frame)."""

    def __init__(self, w=64, h=36, on_read=None, fail_after=None):
        self.w, self.h = w, h
        self.reads = 0
        self.on_read = on_read
        self.fail_after = fail_after
        self.released = False
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        rng = np.random.default_rng(self.reads)
        return True, rng.integers(0, 256, (self.h, self.w, 3), np.uint8)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, f):
        assert not self.closed
        self.frames.append(np.asarray(f).copy())

    def close(self):
        self.closed = True


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode=None, src=None, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode or DitherGirlMode(), source=src or SyntheticSource(),
                **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _free_a_bank_slot(host):
    """Clear one built-in off the bank, for tests that need a KNOWN slot free.

    This used to be load-bearing: Dither's nine built-ins once filled all nine
    slots, so a fresh instance had nowhere to save a first look and
    `_assign_slot` refused with "bank full (1-9)". The seed now stops at
    BANK_SEED_MAX and holds two slots open, so nothing depends on this any
    more — it stays because a test that wants to pin WHICH slot a look lands
    in still has to control the bank rather than hope.
    """
    ui = host.ui
    slot = max(ui.bank, key=int)
    ui.pending_slot = ui.bank[slot]
    host._pump_preset_mailboxes()
    assert slot not in ui.bank
    return slot


def _hints(host):
    return [t.text for t in host.hud.toasts._hints]


def _hud_strings(hud, monkeypatch, **kw):
    """Every string one HUD draw actually puts on the frame."""
    from dtouch import hud as hud_mod
    seen = []
    real = hud_mod.put_outlined
    monkeypatch.setattr(hud_mod, "put_outlined",
                        lambda img, text, *a, **k:
                        seen.append(text) or real(img, text, *a, **k))
    hud.draw(np.zeros((360, 640, 3), np.uint8), OverlayState.HUD, **kw)
    return seen


def _patch_gui(monkeypatch, keys=(), shown=None):
    """Run show=True paths headless: no-op the cv2 window calls; waitKey pops
    from `keys` (then 255); imshow appends to `shown` when given."""
    seq = list(keys)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow",
                        (lambda win, img: shown.append(img.copy()))
                        if shown is not None else (lambda *a: None))
    monkeypatch.setattr(cv2, "waitKey",
                        lambda ms=0: seq.pop(0) if seq else 255)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


# ---------- res change while recording: stop cleanly, never crash ----------

def test_res_change_stops_recording_before_resize(tmp_path):
    """imageio's ffmpeg writer needs a constant frame size — a res cycle while
    recording must close the writer first (same path as 'r' off), toast why,
    then resize."""
    resized = []

    class Spy(DitherGirlMode):
        def on_resize(self, w, h):
            resized.append((w, h, self.host.writer is None))

    writer = FakeWriter()
    host = _host(tmp_path, mode=Spy(), max_frames=6)

    def on_read(n):
        if n == 2:
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n == 4:
            host.ui.res_idx = 1              # cycle output res mid-recording

    host._source.on_read = on_read
    host.run()
    assert resized and resized[0][2] is True     # writer closed BEFORE resize
    assert host.ui.record is False               # recording stopped cleanly
    hints = _hints(host)
    assert any("recording stopped - resolution changed" in t for t in hints)
    assert any(t.startswith("saved") for t in hints)


# ---------- blackout defeats the bright boot card ----------

def test_blackout_switch_records_black_not_the_bright_card(tmp_path,
                                                           monkeypatch):
    """DESIGN.md §3: while blackout is armed the switch happens under black —
    a black frame with the amber tick is shown/recorded, never the bright
    card."""
    import dtouch.modes as modes

    class OtherMode(DitherGirlMode):
        id = "other"
        title = "Other"

    monkeypatch.setattr(modes, "REGISTRY", modes.REGISTRY + [OtherMode])
    writer = FakeWriter()
    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 2:
            host.ps.blackout = True
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n == 3:
            host.request_mode("other")

    host._source.on_read = on_read
    host.run()
    assert host.mode.id == "other"
    assert writer.frames
    for f in writer.frames:
        assert float(f.mean()) < 1.0     # no bright frame ever hit the recorder
    # the switch frame still carries the amber corner tick on black (§3/§5)
    assert any(f[:8, -8:].max() > 0 for f in writer.frames)


# ---------- recorder content: never panel/HUD/toast pixels ----------

def _record_run(tmp_path, decorate, monkeypatch):
    writer = FakeWriter()
    src = SyntheticSource()
    host = _host(tmp_path, src=src, max_frames=8, show=decorate, panel=True)

    def on_read(n):
        if n == 1:
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n >= 6:
            host.ps.blackout = True
        if decorate:                         # panel open + toasts firing
            host.overlay = OverlayState.PANEL
            host.hud.toasts.flash("TESTING")
            host.hud.toasts.hint("hint hint")

    src.on_read = on_read
    host.run()
    return writer.frames


def test_recordings_contain_no_panel_hud_or_toast_pixels(tmp_path,
                                                         monkeypatch):
    """Recording captures what the audience sees, minus panel/HUD (DESIGN.md
    principle 2): a run with the panel open and toasts firing records frames
    identical to a UI-free run of the same source; blackout frames record
    black."""
    _patch_gui(monkeypatch)
    clean = _record_run(tmp_path, decorate=False, monkeypatch=monkeypatch)
    ui_run = _record_run(tmp_path, decorate=True, monkeypatch=monkeypatch)
    assert len(clean) == len(ui_run) == 8
    for a, b in zip(clean, ui_run):
        assert np.array_equal(a, b), "UI pixels leaked into the recording"
    assert any(float(f.mean()) > 1.0 for f in ui_run[:5])    # really rendered
    for f in ui_run[5:]:                                     # blackout frames
        assert not f.any()


# ---------- help modal swallows the mouse ----------

def test_open_help_swallows_mouse_and_click_closes_it(tmp_path):
    host = _booted(tmp_path)
    host.ps.help_open = True
    host.ui._hot = [((0, 0, 200, 200), "quit", None)]
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 50, 0)
    assert host.ui.quit is False             # the click never reached the panel
    assert host.ps.help_open is False        # ...and it closed the help
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 50, 0)
    assert host.ui.quit is True              # normal routing resumed


# ---------- menu key routing through the shell ----------

def test_menu_q_closes_and_arms_quit_confirm(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    host._route_key(ord("q"))
    assert host.menu.open is False
    assert any("q again to quit" in t for t in _hints(host))
    assert host.ps.quit is False
    host._route_key(ord("q"))                # second press, menu closed
    assert host.ps.quit is True


def test_open_menu_consumes_comma_period_before_param_nudge(tmp_path):
    """','/'.' inside the open menu move card selection — the menu keeps
    priority over the param-nudge selection (DESIGN.md §3 vs §6.2)."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    sel = host.menu.sel
    host._route_key(ord(","))
    assert host.menu.sel != sel              # the menu moved...
    assert host.ui.nudge_idx == 0            # ...the nudge selection did not
    assert host.hud.osd._show is None


def test_menu_unknown_key_hints(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    host._route_key(ord("z"))
    assert host.menu.open is True
    assert any("? for keys" in t for t in _hints(host))


def test_no_key_in_the_menu_is_silent(tmp_path):
    """The menu is the first screen anyone meets. Every digit, every arrow and
    Enter must leave something behind — a hint, a moved selection, or a
    committed card. `0` and `3`-`9` used to leave nothing at all."""
    for ch in "0123456789":
        host = _booted(tmp_path)
        host._wire_keys()
        host.menu.show("dithergirl")
        host.hud.toasts._hints.clear()       # the boot hint is not an answer
        before = (host.menu.sel, host.menu.open)
        host._route_key(ord(ch))
        moved = (host.menu.sel, host.menu.open) != before
        assert moved or _hints(host), f"digit {ch} did nothing and said nothing"


def test_zero_in_the_menu_lands_on_the_for_keys_hint(tmp_path):
    """`0` (panic everywhere else, so the likeliest stray digit) names no
    card. The no-silence sweep above only asserts something happened; this
    pins the actual answer through the real key route: the same `? for keys`
    hint every unknown printable gets."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    host.hud.toasts._hints.clear()
    host._route_key(ord("0"))
    assert host.menu.open is True
    assert any("? for keys" in t for t in _hints(host))


def test_the_reserved_card_names_itself_in_the_menu(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    i = next(i for i, c in enumerate(host.menu.cards) if not c.enabled)
    host._route_key(ord(str(i + 1)))
    assert host.menu.open is True
    assert any("coming soon" in t for t in _hints(host))


def test_arrows_navigate_the_menu_from_the_real_key_route(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    sel = host.menu.sel
    host._route_key(1)                       # down arrow (macOS 63233 & 0xFF)
    assert host.menu.sel != sel
    assert host.menu.open is True
    host._route_key(0)                       # up arrow, back again
    assert host.menu.sel == sel


# ---------- camera that never yields: responsive, quittable ----------

def test_camera_that_never_yields_keeps_keys_alive_and_quits(tmp_path,
                                                             monkeypatch):
    """DESIGN.md §6.4: no keyboard-reachable state requires a restart — a
    camera that never yields shows an intentional black frame with the human
    fix and keeps pumping keys, so q/quit works."""
    shown = []
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")], shown=shown)
    host = _host(tmp_path, src=SyntheticSource(fail_after=0), show=True)
    count, out = host.run()
    assert count == 0
    assert host.ps.quit is True              # q reached the quit confirm
    assert any("q again to quit" in t for t in _hints(host))
    assert shown                             # an intentional image was shown
    f = shown[0]
    assert f.shape == (RES[1], RES[0], 3)
    assert float(f.mean()) < 20 and f.max() > 0   # black + on-canvas note


def test_camera_that_never_yields_bounded_run_exits(tmp_path, monkeypatch):
    _patch_gui(monkeypatch)
    host = _host(tmp_path, src=SyntheticSource(fail_after=0), show=True,
                 max_frames=3)
    count, out = host.run()
    assert count == 0 and out is None


# ---------- panic through the shell wiring ----------

def test_panic_key_via_shell_wiring_lands_on_safe_look(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.ui.preset_idx = host.ui.presets.index("phosphor")
    host.ui.glitch = True
    host.ps.blackout = True
    host.reg.dispatch(ord("0"))
    assert host.ps.blackout is False and host.ui.glitch is False
    assert host.ui.pending_preset == "menu"          # the mode's safe look


# ---------- switch-away-and-back contract (amended DESIGN.md §6.2) ----------

def test_first_entry_applies_safe_look_reentry_preserves_settings(
        tmp_path, monkeypatch):
    """FIRST entry to a mode applies safe_look(); RE-entry to an
    already-visited mode preserves its current settings (rehearsal parity —
    presets are an instrument, not a reset trap)."""
    import dtouch.modes as modes
    from dtouch.modes.dithergirl import ALGOS

    class OtherMode(DitherGirlMode):
        id = "other"
        title = "Other"

    monkeypatch.setattr(modes, "REGISTRY", modes.REGISTRY + [OtherMode])
    host = _host(tmp_path, max_frames=10)
    seen = {}

    def on_read(n):
        ui = host.ui
        if n == 2:
            host.request_mode("other")       # FIRST entry
        if n == 4:
            seen["after_first_entry"] = (ui.preset_name, ui.dg_algo_idx)
            ui.dg_algo_idx = ALGOS.index("Bayer")    # operator reshapes look
            ui.preset_idx = ui.presets.index("phosphor")
        if n == 6:
            host.request_mode("dithergirl")  # back to boot mode (re-entry)
        if n == 8:
            host.request_mode("other")       # re-entry to 'other'

    host._source.on_read = on_read
    host.run()
    # first entry landed on the safe look
    assert seen["after_first_entry"] == ("menu", ALGOS.index("Blue noise"))
    # re-entry preserved the operator's settings — no safe_look re-post
    assert host.mode.id == "other"
    assert host.ui.dg_algo_idx == ALGOS.index("Bayer")
    assert host.ui.preset_name == "phosphor"
    assert host.ui.pending_preset is None


# ---------- save flow (amended DESIGN.md §3: naming is one flow) ----------

def test_s_in_panel_saves_and_opens_rename_with_toast(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.PANEL
    host.reg.dispatch(ord("s"))
    assert host.ui.pending_save is True
    host._pump_preset_mailboxes()
    assert host.ui.user_presets                      # the look landed
    name = next(iter(host.ui.user_presets))
    assert host.ui.renaming == name                  # rename box opened
    assert host.ui.rename_buf == name
    assert host.ui.preset_name == name
    assert any("saved" in t for t in _hints(host))   # toast, not just stdout


def test_s_outside_panel_does_not_save_but_hints(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.HUD
    host.reg.dispatch(ord("s"))
    assert host.ui.pending_save is False
    assert host.hud.toasts.active()                  # gated, with feedback


def test_two_saves_in_the_same_second_get_distinct_names(tmp_path,
                                                         monkeypatch):
    host = _booted(tmp_path)
    import dtouch.shell as shell_mod
    monkeypatch.setattr(shell_mod.time, "strftime", lambda fmt: "120000")
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    host.ui.renaming = None
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    assert host.ui.user_presets == {"mine_120000", "mine_120000_2"}


def test_rename_and_delete_confirm_with_toasts(tmp_path):
    host = _booted(tmp_path)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    name = host.ui.renaming
    host.ui.renaming = None
    host.ui.pending_rename = (name, "neon")
    host._pump_preset_mailboxes()
    assert any("renamed" in t for t in _hints(host))
    host.ui.pending_delete = "neon"
    host._pump_preset_mailboxes()
    assert any("deleted" in t for t in _hints(host))
    assert "neon" not in host.ui.presets


# ---------- rename mailbox pump ----------

def test_rename_mailbox_reloads_names_follows_selection_and_bank(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.pending_save = True
    host._pump_preset_mailboxes()
    name = next(iter(ui.user_presets))
    ui.renaming = None                       # close any auto-opened rename box
    ui.preset_idx = ui.presets.index(name)
    _free_a_bank_slot(host)
    ui.pending_slot = name                   # assign a bank slot
    host._pump_preset_mailboxes()
    slot = next(s for s, n in ui.bank.items() if n == name)
    ui.pending_rename = (name, "neon dancer")
    host._pump_preset_mailboxes()
    assert "neon dancer" in ui.presets and name not in ui.presets
    assert ui.preset_name == "neon dancer"           # selection followed
    assert ui.bank[slot] == "neon dancer"            # bank followed
    assert presets.bank(host.presets_path,
                        mode="dithergirl")[slot] == "neon dancer"


# ---------- SIGNAL rack dither-quality controls (DESIGN.md §2.4/§4.1) ----------

class NoClaimsMode(DitherGirlMode):
    """A GL-free mode that claims nothing — the full rack stays visible."""
    id = "noclaims"
    title = "NoClaims"
    claims = frozenset()


def test_rack_quality_controls_drive_circuitbent(tmp_path):
    """Bits/Gamma/Bias from the rack reach the CircuitBent dither call."""
    host = _host(tmp_path, mode=NoClaimsMode(), max_frames=3)

    def on_read(n):
        if n == 2:
            ui = host.ui
            ui.glitch = True
            ui.dither_idx = 0                # bayer
            ui.sig_bits = 1.6                # int-snaps to 2
            ui.sig_gamma = False
            ui.sig_bias_idx = 2              # dark -> invert True

    host._source.on_read = on_read
    host.run()
    assert host.cb is not None
    assert host.cb.dither_bits == 2
    assert host.cb.dither_gamma is False
    assert host.cb.dither_invert is True


def test_rack_quality_controls_round_trip_inside_signal(tmp_path):
    """The rack's Bits/Gamma/Bias persist under the look's "signal" block and
    apply back (DESIGN.md §7)."""
    host = _booted(tmp_path, mode=NoClaimsMode())
    ui = host.ui
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 2.0, False, 2
    cfg = host._capture_cfg()
    assert cfg["signal"]["bits"] == 2.0
    assert cfg["signal"]["gamma"] is False
    assert cfg["signal"]["bias"] == "dark"
    presets.save("bent", cfg, path=host.presets_path, mode="noclaims")
    host._reload_presets()
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 3.0, True, 0   # scramble
    ui.pending_preset = "bent"
    host._apply_pending_preset()
    assert (ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx) == (2.0, False, 2)


# ---------- global Menu (M) row + panel title + 'i' in HIDDEN ----------

def test_menu_row_action_opens_the_menu(tmp_path):
    """DESIGN.md §4.1: the global 'Menu (M)' Action posts menu.open through
    pending_commands; the shell routes it to the same command 'm' runs."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.ui.pending_commands.append("menu.open")
    host._pump_preset_mailboxes()
    assert host.menu.open is True
    assert host.ui.pending_commands == []


def test_panel_title_carries_the_mode(tmp_path):
    """DESIGN.md §4.1/§4.2 sketch: panel title is 'lighteater - MODE TITLE'."""
    host = _booted(tmp_path)
    assert host.ui.panel_title == "lighteater - DITHER"


def test_i_in_hidden_posts_the_debug_line_as_a_toast(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.HIDDEN
    host.reg.dispatch(ord("i"))
    assert any("fps" in t for t in _hints(host))
    n = sum("fps" in t for t in _hints(host))
    host.overlay = OverlayState.HUD          # HUD: the status line IS feedback
    host.reg.dispatch(ord("i"))
    assert sum("fps" in t for t in _hints(host)) == n


# ---------- boot: hint + last-used-mode resume (DESIGN.md §3) ----------

def test_boot_posts_the_4s_hud_hint(tmp_path):
    """DESIGN.md §3: a HUD hint fades in for 4 s on boot — the three doors."""
    host = _booted(tmp_path)
    hint = next((t for t in host.hud.toasts._hints
                 if t.text == "m menu - TAB panel - ? keys"), None)
    assert hint is not None
    assert hint.ttl == 4.0


def test_no_explicit_mode_resumes_last_used_from_state(tmp_path):
    """DESIGN.md §3: launch goes straight into the last-used mode — a Host
    built with mode=None reads state.json and boots into it."""
    p = _paths(tmp_path)
    presets.save_state({"mode": "dithergirl", "preset": None, "bank": {}},
                       path=p["state_path"])
    host = Host(None, source=SyntheticSource(), res=RES, show=False,
                preset=None, max_frames=1, **p)
    host.run()
    assert host.mode.id == "dithergirl"


def test_no_state_file_resolves_boot_mode_to_particles(tmp_path):
    from dtouch.modes.particles import ParticlesMode
    host = _host(tmp_path)
    assert isinstance(host._resolve_boot_mode(), ParticlesMode)


def test_unknown_state_mode_resolves_to_particles(tmp_path):
    from dtouch.modes.particles import ParticlesMode
    p = _paths(tmp_path)
    presets.save_state({"mode": "gone-forever"}, path=p["state_path"])
    host = _host(tmp_path)
    assert isinstance(host._resolve_boot_mode(), ParticlesMode)


def test_the_shipped_key_map_fits_every_shipped_resolution(tmp_path):
    """The real registry, not a synthetic row list: Particles wires the most
    keys (33 rows with TAB/Esc appended), and the fixed-column map put the
    last two of them — the two that walk you back OUT of an overlay state —
    below the bottom of the frame at 720p and 1080p, three rows off at 4K."""
    from dtouch.modes.particles import ParticlesMode
    from dtouch.hud import help_layout, text_size, TITLE_SAFE

    host = _host(tmp_path, mode=ParticlesMode(), max_frames=1)
    host.run()
    host._wire_keys()
    rows = host.help_rows
    assert ("TAB", "Cycle overlay") in rows and rows[-1][0] == "Esc"

    for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
        (_, ty), _, px, placed = help_layout(w, h, rows)
        iy = int(h * TITLE_SAFE)
        assert len(placed) == len(rows)
        assert ty >= iy
        for key, label, kx, lx, y in placed:
            assert y <= h - iy, f"{key} at {w}x{h} is off the bottom ({y})"
            assert lx + text_size(label, px)[0] <= w - int(w * TITLE_SAFE)


# ---------- corrupt presets file: the note reaches the toasts ----------

def test_corrupt_presets_note_reaches_the_toasts(tmp_path):
    p = _paths(tmp_path)
    with open(p["presets_path"], "w") as f:
        f.write("{broken")
    host = _booted(tmp_path)
    assert any("backup" in t for t in _hints(host))
    assert list(tmp_path.glob("presets.corrupt.*.bak.json"))


# ---------- a failed store write never takes the show down (§6.4) ----------

def _full_disk(monkeypatch, *names):
    def boom(*a, **kw):
        raise OSError(28, "No space left on device")
    for name in names:
        monkeypatch.setattr(presets, name, boom)


def test_save_click_on_a_full_disk_toasts_and_keeps_running(tmp_path,
                                                            monkeypatch):
    """json.dump's OSError used to propagate out of the mailbox pump and exit
    run() mid-performance — a save click could kill the show."""
    host = _host(tmp_path, max_frames=4,
                 src=SyntheticSource(
                     on_read=lambda n: setattr(host.ui, "pending_save", True)
                     if n == 2 and host.ui else None))
    _full_disk(monkeypatch, "save")
    count, out = host.run()
    assert count == 4                                # the loop survived
    assert host.hud.toasts._center.text == "save failed - disk?"
    assert not host.ui.user_presets                  # nothing pretends to exist
    assert host.ui.renaming is None                  # no rename box for a ghost


@pytest.mark.parametrize("mailbox,call,value", [
    ("pending_delete", "delete", lambda name: name),
    ("pending_rename", "rename", lambda name: (name, "neon")),
    ("pending_slot", "set_bank", lambda name: name),
])
def test_every_store_mailbox_survives_a_full_disk(tmp_path, monkeypatch,
                                                  mailbox, call, value):
    host = _booted(tmp_path)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()                    # a real look to act on
    host.ui.renaming = None
    name = next(iter(host.ui.user_presets))
    if mailbox == "pending_slot":
        _free_a_bank_slot(host)              # pin which slot the look lands in
    _full_disk(monkeypatch, call)
    setattr(host.ui, mailbox, value(name))
    host._pump_preset_mailboxes()                    # must not raise
    assert host.hud.toasts._center.text == "save failed - disk?"


# ---------- a REFUSED write is not a successful one (DESIGN.md §9) ----------

def test_refused_save_never_says_saved_and_never_opens_the_rename_box(
        tmp_path, monkeypatch):
    """The store refuses to overwrite a file it could not back up. That
    refusal raises nothing, so the shell used to report 'saved - <name>',
    print 'saved preset', and drop the operator into a rename box on a preset
    that does not exist — a text field that then eats every keypress (q, m,
    panic included) with nothing on screen to explain it."""
    host = _booted(tmp_path)
    monkeypatch.setattr(presets, "write_file", lambda data, path=None: False)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()

    assert host.ui.renaming is None                  # no text field on a ghost
    assert not host.ui.user_presets                  # nothing pretends to exist
    assert not any("saved -" in t for t in _hints(host))
    assert host.hud.toasts._center.text == "save refused"


def test_refused_store_write_toasts_the_stores_own_reason(tmp_path,
                                                          monkeypatch):
    """`_store_write` drains the store's queued notes onto the toasts, so the
    operator gets the REASON, not just 'refused' (silence on a data-loss event
    is a bug — DESIGN.md §9)."""
    host = _booted(tmp_path)

    def refuse(data, path=None):
        presets._note("presets.json not saved - unreadable and could not be "
                      "backed up")
        return False
    monkeypatch.setattr(presets, "write_file", refuse)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    assert any("not saved" in t for t in _hints(host))
    presets.take_notes()


# ---------- the frame loop is contained: no traceback ends the show ----------
# DESIGN.md §6.4 + §8 step 10 ("any traceback = release blocker"). Nothing in
# the per-frame loop used to be wrapped, so ONE exception out of source.read,
# cv2.flip, mode.step or writer.append_data ended the performance with a stack
# trace on the projector.


class BoomMode(DitherGirlMode):
    """A mode whose step() raises from frame `fail_from` onward."""

    def __init__(self, fail_from=2, exc=None, **kw):
        super().__init__(**kw)
        self.fail_from = fail_from
        self.exc = exc or ModuleNotFoundError("No module named 'mediapipe'")
        self.steps = 0

    def step(self, frame_bgr, audio_levels, dt):
        self.steps += 1
        if self.steps >= self.fail_from:
            raise self.exc
        return super().step(frame_bgr, audio_levels, dt)


def test_mode_step_exception_holds_the_last_frame_instead_of_ending_the_show(
        tmp_path):
    """The most reachable path: a start.command install with no mediapipe,
    one click on the shipped `portrait` template, ModuleNotFoundError out of
    mode.step(). The loop must survive it and keep holding the last good
    picture."""
    mode = BoomMode(fail_from=3)
    host = _host(tmp_path, mode=mode, max_frames=8)
    count, out = host.run()

    assert count == 8                                # the loop never exited
    assert mode.steps == 8                           # ...and kept stepping
    assert out is not None and out.any()             # last good picture held
    assert host.hud.toasts._center.text == "something went wrong - show continues"
    assert any("ModuleNotFoundError" in t for t in _hints(host))


def test_blackout_still_blacks_the_output_while_the_produce_half_is_failing(
        tmp_path):
    """DESIGN.md §6.2: SPACE kills the output, and the amber tick says so.

    The blackout write lived INSIDE the try, so on the failure path (`out =
    last_out`) it was never re-applied: while a mode raised every frame — the
    most reachable failure there is — SPACE lit the tick, flashed BLACKOUT,
    and left the projector fully live (measured nonblack_px=51607). That is
    the one key a performer hits when something is already wrong on screen."""
    mode = BoomMode(fail_from=3)
    host = _host(tmp_path, mode=mode, max_frames=6)
    host._source.on_read = lambda n: (setattr(host.ps, "blackout", True)
                                      if n == 3 else None)
    _, out = host.run()
    assert mode.steps == 6                       # the produce half kept failing
    assert not out.any(), "armed blackout, live output - the tick was lying"


def test_the_held_picture_survives_a_blackout_taken_while_broken(tmp_path):
    """...and the fix must not black the HELD frame in place: `out` is
    `last_out` on that path, so `out[:] = 0` there would destroy the last good
    picture permanently and keep the projector black long after blackout was
    disarmed."""
    good = []

    class Watched(BoomMode):
        def step(self, frame_bgr, audio_levels, dt):
            out = super().step(frame_bgr, audio_levels, dt)
            good.append(out.copy())
            return out

    host = _host(tmp_path, mode=Watched(fail_from=3), max_frames=8)
    host._source.on_read = lambda n: (setattr(host.ps, "blackout", n < 6)
                                      if n >= 3 else None)
    _, out = host.run()
    assert good and good[-1].any()               # a real picture was held
    assert out.any(), "the held picture never came back after blackout"
    assert np.array_equal(out, good[-1])


def test_a_frame_error_on_the_very_first_frame_is_survivable(tmp_path):
    """Nothing good has been rendered yet, so there is no picture to hold —
    an intentional black frame, not an unbound-variable crash."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1), max_frames=3)
    count, out = host.run()
    assert count == 3
    assert out is not None and not out.any()


def test_repeating_frame_error_does_not_spam_the_toasts(tmp_path):
    """A per-frame exception at 60 fps would refill the toast stack sixty
    times a second and bury the status line under its own error.

    This covers ONE window only (40 frames land inside ERR_TOAST_S however
    fast the loop runs) — the other half of the contract, that the same error
    re-toasts once the window passes, is
    test_a_permanently_broken_show_keeps_saying_so."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1), max_frames=40)
    flashes = []
    host.hud.toasts.flash = lambda text, color=None: flashes.append(text)
    host.run()
    assert flashes.count("something went wrong - show continues") == 1


def test_a_permanently_broken_show_keeps_saying_so(tmp_path):
    """The rate limit is a sliding WINDOW, and the entry has to be allowed to
    age out of it. Re-stamping the key on every arrival — suppressed ones
    included — refreshed it 30 times a second, so the 5 s prune never dropped
    it: a show broken for good (mode.step raising ModuleNotFoundError after
    one click on `portrait` with no mediapipe) toasted once, three seconds in,
    then went silent for the rest of the night while the projector stayed
    black and the status line read completely normally.

    16 s at 30 fps of the identical error: 1 flash before, ~4 after."""
    clock = [0.0]
    host = _host(tmp_path, max_frames=1, now=lambda: clock[0])
    at = []
    host.hud.toasts.flash = lambda text, color=None: at.append(clock[0])
    for _ in range(16 * 30):
        host._frame_error(ModuleNotFoundError("No module named 'mediapipe'"))
        clock[0] += 1.0 / 30.0

    assert len(at) >= 4, "a permanently broken show went silent"
    gaps = [b - a for a, b in zip(at, at[1:])]
    assert all(g >= shell_mod.ERR_TOAST_S - 1e-6 for g in gaps), \
        "re-toasting faster than the window is spam"
    assert all(g < shell_mod.ERR_TOAST_S + 0.2 for g in gaps), \
        "the window is meant to be ERR_TOAST_S, not longer"


def test_a_different_frame_error_toasts_immediately(tmp_path):
    """Rate-limiting is per (type, message) — a NEW failure is news."""
    host = _host(tmp_path, max_frames=1)
    flashes = []
    host.hud.toasts.flash = lambda text, color=None: flashes.append(text)
    host._frame_error(ValueError("first"))
    host._frame_error(TypeError("second"))
    assert len(flashes) == 2
    assert any("TypeError: second" in t for t in _hints(host))


def test_two_alternating_errors_cannot_defeat_the_rate_limit(tmp_path):
    """The limit remembered exactly ONE (type, message), so A→B→A→B passed
    its equality check every single time and sprayed toasts and stdout twice
    per frame forever — on the most reachable repeating failure there is (a
    mode that raises one error while the overlay raises another). The window
    now remembers every key inside it."""
    host = _host(tmp_path, max_frames=1)
    flashes, prints = [], []
    host.hud.toasts.flash = lambda text, color=None: flashes.append(text)
    for i in range(120):                             # two seconds at 60 fps
        host._frame_error(ValueError("a") if i % 2 else TypeError("b"))
    assert len(flashes) == 2                         # one per DISTINCT error
    assert len(_hints(host)) <= 3                    # the stack never refills


def test_errors_that_vary_their_text_are_a_flood_too(tmp_path):
    """A message that carries a frame number or a coordinate is a different
    key every frame — per-key limiting alone would still spray."""
    host = _host(tmp_path, max_frames=1)
    flashes = []
    host.hud.toasts.flash = lambda text, color=None: flashes.append(text)
    for i in range(120):
        host._frame_error(ValueError("bad pixel at %d" % i))
    assert len(flashes) <= shell_mod.ERR_TOAST_MAX
    assert any("more errors" in t for t in _hints(host))


def test_a_flood_is_bounded_on_stdout_and_in_memory_too(tmp_path, capsys):
    """The cap bounded the TOASTS and nothing else: 300 varying-text errors
    printed 300 stdout lines — the flood simply moved to the terminal the
    hint points at — while `_err_seen` grew to 300 entries that the sliding
    window rebuilt by comprehension on every single call."""
    host = _host(tmp_path, max_frames=1)
    host.hud.toasts.flash = lambda *a, **k: None
    capsys.readouterr()
    for i in range(300):
        host._frame_error(ValueError("bad pixel at %d" % i))
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]

    assert len(lines) <= shell_mod.ERR_PRINT_MAX + 1, "stdout was not bounded"
    assert any("suppressed" in l for l in lines), "silence is not an answer"
    assert len(host._err_seen) <= shell_mod.ERR_SEEN_MAX


def test_an_error_from_our_own_reporting_never_hides_the_real_cause(tmp_path):
    """The present half is OUR code. When it fails while reporting a produce
    failure, the operator was left reading our bug ('UnboundLocalError: frame')
    with the real cause — the camera — permanently unreachable."""
    host = _host(tmp_path, max_frames=1)
    host._frame_ok = False                           # produce failed this frame
    host._frame_error(OSError(5, "Input/output error"))
    host._frame_error(UnboundLocalError("frame"), internal=True)
    assert any("OSError" in t for t in _hints(host))  # the cause still stands
    assert not any("UnboundLocalError" in t for t in _hints(host))


def test_keys_still_work_while_every_single_frame_fails(tmp_path, monkeypatch):
    """The contract that matters (DESIGN.md §6.4: no keyboard-reachable state
    requires a restart) — the produce half is contained, so the present half
    still draws and still pumps `q`."""
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")])
    host = _host(tmp_path, mode=BoomMode(fail_from=1), show=True)
    host.run()
    assert host.ps.quit is True


# ---------- a contained failure must idle, not spin ----------
# Containment turned a fatal produce error into a caught one, which is right —
# but a caught error that repeats every frame with nothing to wait on is a busy
# loop. Shown, that pegged a core at 983 iterations/sec behind a frozen
# picture. Headless with no frame budget it was worse: no window, no key pump,
# no output, and only Ctrl-C to end it — a state the pre-containment code
# could not reach, so containment created it.


def test_a_headless_permanent_failure_ends_with_the_real_cause(tmp_path,
                                                               monkeypatch):
    """Nothing to look at and nothing to press: an infinite silent busy loop
    is worse than the traceback containment replaced. It ends, and it says
    what was actually wrong."""
    monkeypatch.setattr(shell_mod, "FRAME_FAIL_SLEEP_S", 0.0)
    src = SyntheticSource()
    host = _host(tmp_path, mode=BoomMode(fail_from=1), src=src)
    assert host.max_frames is None and host.show is False

    with pytest.raises(RuntimeError) as caught:
        host.run()
    assert "mediapipe" in str(caught.value)       # the cause, not just 'failed'
    assert src.released                           # teardown still ran


def test_a_headless_camera_that_never_arrives_ends_the_run(tmp_path,
                                                           monkeypatch):
    """The same shape from the other direction: reads that return (False,
    None) forever, with no window to draw the waiting card on."""
    monkeypatch.setattr(shell_mod, "FRAME_FAIL_SLEEP_S", 0.0)
    host = _host(tmp_path, src=SyntheticSource(fail_after=0))
    with pytest.raises(RuntimeError) as caught:
        host.run()
    assert "no frame ever arrived" in str(caught.value)


def test_a_shown_permanent_failure_runs_at_a_sane_rate(tmp_path, monkeypatch):
    """Shown, the loop must NOT end — a window that explains itself and
    answers `q` is the whole point. It must not burn a core to do it: measured
    983 iterations/sec, against ~30/s for a healthy loop."""
    _patch_gui(monkeypatch, keys=[255] * 30 + [ord("q"), ord("q")])
    host = _host(tmp_path, mode=BoomMode(fail_from=1), show=True)

    t0 = time.perf_counter()
    count, _ = host.run()
    rate = count / max(time.perf_counter() - t0, 1e-6)

    assert host.ps.quit is True                   # it kept answering keys
    assert count >= 30
    assert rate < 200, f"the failing loop spun at {rate:.0f} frames/sec"


def test_a_bounded_headless_render_does_not_idle_through_its_budget(
        tmp_path, monkeypatch):
    """The throttle is for a loop with nothing to wait on and no end in sight.
    A headless run WITH a frame budget has an end in sight and nobody watching
    it: one frame-time per failing frame turned a 300-frame render against a
    broken source into 11.2 s of sleeping for 0.4 s of work."""
    slept = []
    monkeypatch.setattr(shell_mod.time, "sleep", lambda s: slept.append(s))
    host = _host(tmp_path, mode=BoomMode(fail_from=1), max_frames=300)

    t0 = time.perf_counter()
    count, _ = host.run()
    elapsed = time.perf_counter() - t0

    assert count == 300                          # the budget still ran in full
    assert slept == [], "a bounded headless render paid the show's throttle"
    assert elapsed < 2.0, f"300 broken frames took {elapsed:.1f}s"


def test_an_unbounded_headless_failure_still_idles_before_it_gives_up(
        tmp_path, monkeypatch):
    """...and the case the throttle exists for keeps it: no window, no budget,
    so it must not spin at 983 iterations/sec on its way to the limit."""
    slept = []
    monkeypatch.setattr(shell_mod.time, "sleep", lambda s: slept.append(s))
    host = _host(tmp_path, mode=BoomMode(fail_from=1))
    with pytest.raises(RuntimeError):
        host.run()
    assert len(slept) >= shell_mod.FRAME_FAIL_LIMIT - 1
    assert all(s == shell_mod.FRAME_FAIL_SLEEP_S for s in slept)


def test_a_shown_bounded_run_still_idles(tmp_path, monkeypatch):
    """A window is a show whether or not it has a frame budget: it holds ~30 Hz
    so the key pump stays responsive instead of pegging a core."""
    slept = []
    _patch_gui(monkeypatch)
    monkeypatch.setattr(shell_mod.time, "sleep", lambda s: slept.append(s))
    host = _host(tmp_path, mode=BoomMode(fail_from=1), show=True, max_frames=5)
    host.run()
    assert len(slept) == 5


def test_fps_reports_rendered_frames_not_loop_spins(tmp_path):
    """`count` incremented on frames that were never rendered, so the HUD and
    the `i` readout showed 953.5 fps while the picture was frozen."""
    healthy = _host(tmp_path, max_frames=12)
    healthy.run()
    assert healthy.fps > 0.0                      # a working loop still reports

    broken = _host(tmp_path, mode=BoomMode(fail_from=2), max_frames=40)
    broken.run()
    assert broken.fps < 1.0, "fps reported the spin rate, not the picture"


def test_a_first_read_that_raises_shows_the_real_cause(tmp_path, monkeypatch):
    """A first `source.read()` that RAISES (rather than returning False) left
    `frame` unbound inside the contained produce half, so the present half
    raised too — caught, and then standing in the operator's error line in
    place of the camera error, forever, over a black rectangle instead of the
    §6.4 waiting card."""
    shown = []
    _patch_gui(monkeypatch, keys=[255] * 3 + [ord("q"), ord("q")], shown=shown)
    src = SyntheticSource()

    def boom(n):
        raise OSError(5, "Input/output error")

    src.on_read = boom
    host = _host(tmp_path, src=src, show=True)
    cards = []
    real = host._draw_waiting_note
    host._draw_waiting_note = lambda img: cards.append(1) or real(img)
    host.run()

    # the real cause, and only it
    assert any(k[0] == "OSError" for k in host._err_seen)
    assert any("OSError" in t for t in _hints(host))
    # the present half never raised at all — not raised-and-swallowed
    assert not any(k[0] == "UnboundLocalError" for k in host._err_seen)
    assert cards, "the §6.4 waiting card never drew - black fallback instead"
    assert shown and shown[-1].any()


def test_a_stall_never_hands_a_mode_an_unbounded_dt(tmp_path, monkeypatch):
    """`last_t` only advanced after the mailboxes, so anything failing earlier
    kept accumulating: a measured 7-frame stall handed mode.step a single
    0.409 s dt, and a long one would hand a mode a multi-second integration
    step — particles teleport, fades blow out, and the recovery frame looks
    worse than the stall did."""
    monkeypatch.setattr(shell_mod, "FRAME_FAIL_SLEEP_S", 0.0)
    dts = []

    class Watch(DitherGirlMode):
        def step(self, frame_bgr, audio_levels, dt):
            dts.append(dt)
            return super().step(frame_bgr, audio_levels, dt)

    src = SyntheticSource()

    def stall(n):
        # the failure lands BEFORE the clock line, which is the whole bug:
        # a camera that raises mid-read, three frames of it
        if 2 <= n <= 4:
            time.sleep(0.15)
            raise OSError(5, "Input/output error")

    src.on_read = stall
    host = _host(tmp_path, mode=Watch(), src=src, max_frames=6)
    t0 = time.perf_counter()
    host.run()
    stalled_for = time.perf_counter() - t0

    assert len(dts) == 3                          # frames 1, 5, 6 got through
    assert stalled_for > shell_mod.DT_MAX         # a real stall really happened
    assert max(dts) <= shell_mod.DT_MAX, f"a mode was handed {max(dts):.3f}s"


def test_keyboard_interrupt_still_propagates(tmp_path):
    """Ctrl-C means stop. The guard is for bugs, not for the operator."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1, exc=KeyboardInterrupt()),
                 max_frames=4)
    with pytest.raises(KeyboardInterrupt):
        host.run()


def test_source_read_exception_is_contained(tmp_path):
    """A camera that raises rather than returning (False, None) — a USB yank
    mid-read — is the same event to the operator."""
    src = SyntheticSource()

    def boom(n):
        if n >= 2:
            raise OSError(5, "Input/output error")
    src.on_read = boom
    host = _host(tmp_path, src=src, max_frames=5)
    count, _ = host.run()
    assert count == 5
    assert any("OSError" in t for t in _hints(host))


# ---------- a failing recorder stops recording, not the show ----------

def test_recorder_write_failure_stops_the_take_and_keeps_running(tmp_path):
    class BadWriter(FakeWriter):
        def append_data(self, f):
            raise OSError(28, "No space left on device")

    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 2:
            host.ui.record = True
            host.writer, host.rec_path = BadWriter(), "fake.mp4"

    host._source.on_read = on_read
    count, out = host.run()

    assert count == 6                                # the show carried on
    assert host.writer is None                       # closed cleanly
    assert host.ui.record is False                   # and disarmed, not retried
    assert host.hud.toasts._center.text == "recording stopped - write failed"


# ---------- recordings: out/ on first take, and never a silent overwrite ----------

def test_out_dir_is_not_created_at_boot(tmp_path, monkeypatch):
    """`os.makedirs("out")` at boot raised PermissionError on a read-only
    launch dir (a run from a mounted DMG) — a traceback before any window
    existed, for a directory most sessions never use."""
    monkeypatch.chdir(tmp_path)
    _booted(tmp_path)
    assert not (tmp_path / "out").exists()


def test_first_record_creates_out_and_names_the_take(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host = _booted(tmp_path)
    made = []
    monkeypatch.setattr(shell_mod.imageio, "get_writer",
                        lambda p, **kw: made.append(p) or FakeWriter())
    host.ui.record = True
    host._sync_host_state()
    assert (tmp_path / "out").is_dir()
    assert made and made[0] == host.rec_path
    assert os.path.basename(host.rec_path).startswith("rec_")


def test_a_launch_dir_that_cannot_take_recordings_toasts_and_disarms(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").write_text("not a directory")   # makedirs raises here
    host = _booted(tmp_path)
    host.ui.record = True
    host._sync_host_state()                            # must not raise
    assert host.writer is None
    assert host.ui.record is False                     # disarmed, not retried
    assert host.hud.toasts._center.text == "cannot write recordings here"


def test_two_takes_in_the_same_second_get_distinct_filenames(tmp_path,
                                                             monkeypatch):
    """Names are second-resolution. Stopping and restarting a take inside one
    second reused the name and silently truncated the first file — the take
    you just made, gone."""
    monkeypatch.chdir(tmp_path)
    host = _booted(tmp_path)
    monkeypatch.setattr(shell_mod.time, "strftime",
                        lambda fmt: "20260101_120000")
    os.makedirs("out", exist_ok=True)

    first = host._rec_path()
    open(first, "wb").close()                          # the take now on disk
    second = host._rec_path()
    open(second, "wb").close()
    third = host._rec_path()

    assert os.path.basename(first) == "rec_20260101_120000.mp4"
    assert os.path.basename(second) == "rec_20260101_120000_2.mp4"
    assert os.path.basename(third) == "rec_20260101_120000_3.mp4"


# ---------- a camera that will not open never reaches a traceback ----------

def test_camera_that_cannot_be_opened_still_opens_the_window(tmp_path,
                                                             monkeypatch):
    """`open_camera` raises whenever isOpened() is False — no camera, TCC
    denied, or the device held by Zoom/OBS. That traceback landed before any
    window existed, so the plain-language permission card DESIGN.md §6.4
    promises could never be shown."""
    def denied(device):
        raise RuntimeError("could not open camera 'builtin'")
    monkeypatch.setattr(shell_mod, "open_camera", denied)
    shown = []
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")], shown=shown)

    host = Host(DitherGirlMode(), source=None, res=RES, show=True, preset=None,
                **_paths(tmp_path))
    count, out = host.run()                            # must not raise

    assert count == 0
    assert host.ps.quit is True                        # still quittable
    assert shown and float(shown[0].mean()) < 20 and shown[0].max() > 0
    assert any("could not open camera" in t for t in _hints(host))


def test_camera_source_recovers_when_the_camera_appears(monkeypatch):
    """Recovery is automatic: the open is retried, so the show starts by
    itself when the device is freed."""
    class FakeCap:
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    tries = []
    clock = [0.0]

    def flaky(device):
        tries.append(device)
        if len(tries) < 3:
            raise RuntimeError("busy")
        return FakeCap(), "FaceTime HD"
    monkeypatch.setattr(shell_mod, "open_camera", flaky)

    src = CameraSource("builtin", now=lambda: clock[0])
    assert src.read() == (False, None)
    assert src.name == "no camera" and src.error is not None

    clock[0] += CameraSource.RETRY_S
    assert src.read() == (False, None)                 # retried, still busy
    clock[0] += CameraSource.RETRY_S
    ok, frame = src.read()
    assert ok and frame is not None
    assert src.name == "FaceTime HD" and src.error is None
    src.release()


def test_camera_source_does_not_retry_faster_than_the_backoff(monkeypatch):
    calls = []

    def busy(device):
        calls.append(device)
        raise RuntimeError("nope")
    monkeypatch.setattr(shell_mod, "open_camera", busy)
    src = CameraSource("builtin", now=lambda: 0.0)
    for _ in range(20):
        src.read()
    assert len(calls) == 1                             # one open, not twenty


def test_a_late_camera_gets_its_name_onto_the_status_line(tmp_path):
    """A CameraSource that only opened on its third retry boots named 'no
    camera'; the status line has to pick up the real name when frames start."""
    src = SyntheticSource()
    src.name = "no camera"
    host = _host(tmp_path, src=src, max_frames=4)

    def on_read(n):
        if n >= 3:
            src.name = "FaceTime HD"                   # the camera turns up

    src.on_read = on_read
    host.run()
    assert host.cam_name == "FaceTime HD"


# ---------- a missing optional matte reverts the cycle, it never raises ----------

def _no_mediapipe(monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'mediapipe'")
    monkeypatch.setitem(matte_mod.KINDS, "person", missing)


def test_missing_person_matte_reverts_the_cycle_and_toasts_the_extra(
        tmp_path, monkeypatch):
    """The shipped `portrait` and `sigil` templates both select `person`, so
    on any install without the extra this is one click from the frame loop."""
    _no_mediapipe(monkeypatch)
    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 3:
            host.ui.dg_matte_idx = MATTES_DG.index("person")

    host._source.on_read = on_read
    count, out = host.run()

    assert count == 6
    assert host.ui.dg_matte_idx == MATTES_DG.index("off")   # reverted
    # a graceful step-back with the fix in it, NOT the generic frame-error path
    assert host.hud.toasts._center.text == "person matte needs the [person] extra"


def test_select_matte_keeps_the_callers_operator_on_refusal(monkeypatch):
    _no_mediapipe(monkeypatch)

    class UI:
        idx = 1
    ui = UI()
    mat, kind = matte_mod.select_matte(ui, "idx", ["luma", "person"], "luma")
    assert mat is None and kind == "luma"
    assert ui.idx == 0                                 # the cycle stepped back


def test_make_matte_raises_matte_unavailable_not_the_raw_import_error(
        monkeypatch):
    _no_mediapipe(monkeypatch)
    with pytest.raises(matte_mod.MatteUnavailable) as e:
        matte_mod.make_matte("person")
    assert "[person] extra" in str(e.value)
    with pytest.raises(KeyError):                      # a bad kind is still a bug
        matte_mod.make_matte("nonsense")


# ---------- mouse/key parity at boot (DESIGN.md §6.1 + §6.3) ----------
# Boot state is HUD, which draws no sidebar, and ui._hot is emptied whenever
# the panel is hidden — so after boot NOTHING on the frame answered a click.
# An audit fired 576 clicks across the whole frame and got no response at all.


def _run_frames(tmp_path, monkeypatch, overlay, n=2, keys=()):
    shown = []
    _patch_gui(monkeypatch, keys=keys, shown=shown)
    host = _host(tmp_path, show=True, panel=True, max_frames=n)
    host._source.on_read = lambda i: setattr(host, "overlay", overlay)
    host.run()
    return host, shown


def test_hud_state_leaves_one_clickable_way_into_the_panel(tmp_path,
                                                           monkeypatch):
    host, shown = _run_frames(tmp_path, monkeypatch, OverlayState.HUD)
    assert host.ui._hot, "HUD had no hit rect at all — the mouse is a dead end"
    rects = [(r, k) for r, k, _ in host.ui._hot]
    assert [k for _, k in rects] == ["panel.open"]     # exactly one, no clutter
    (x0, y0, x1, y1), _ = rects[0]
    w, h = RES
    assert x1 <= w and x0 > w * 0.7 and y0 < h * 0.3   # shipped top-right spot


def test_clicking_the_hud_chevron_opens_the_panel(tmp_path, monkeypatch):
    """It routes through the same named command TAB reaches — one
    implementation for both paths (DESIGN.md principle 7)."""
    host, _ = _run_frames(tmp_path, monkeypatch, OverlayState.HUD)
    host._wire_keys()
    (x0, y0, x1, y1), _, _ = host.ui._hot[0]
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0)
    assert host.ui.pending_commands == ["panel.open"]
    host._pump_preset_mailboxes()
    assert host.overlay is OverlayState.PANEL
    # the overlay state alone is NOT the panel being open — see the pixel test
    assert host.ui.open is True


def test_the_chevron_opens_a_sidebar_the_operator_collapsed(tmp_path,
                                                            monkeypatch):
    """The state assertion above passed while the operator saw NOTHING.

    `overlay is PANEL` is not the sidebar being on screen: the sidebar keeps
    its own collapsed flag, and a collapsed sidebar draws its expand button at
    exactly the coordinates the chevron uses. Collapse the panel (one click,
    the shipped '>' button), step back to HUD, and the chevron click then
    swapped one small box for a near-identical small box in the same place —
    click 1 nothing, click 2 sidebar. That IS the 'my click did nothing'
    symptom the chevron was added to kill, so this asserts PIXELS."""
    shown = []
    _patch_gui(monkeypatch, shown=shown)
    host = _host(tmp_path, show=True, panel=True, max_frames=5)

    def click(rect):
        (x0, y0, x1, y1) = rect
        host._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0)

    def on_read(n):
        if n == 1:
            host.overlay = OverlayState.PANEL        # TAB to the panel…
        elif n == 2:                                 # …collapse it, as shipped
            click(next(r for r, k, _ in host.ui._hot if k == "collapse"))
            host.overlay = OverlayState.HUD          # …and Esc back to HUD
        elif n == 3:
            assert host.ui.open is False             # the state we are fixing
            click(next(r for r, k, _ in host.ui._hot if k == PANEL_OPEN))

    host._source.on_read = on_read
    _, out = host.run()

    assert host.overlay is OverlayState.PANEL
    picture = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    covered = np.any(shown[-1] != picture, axis=2).mean()
    assert covered > 0.5, ("the click drew another small box, not a sidebar "
                           f"(only {covered:.0%} of the frame changed)")


def test_the_chevron_never_draws_in_hidden(tmp_path, monkeypatch):
    """HIDDEN's whole contract is provably clean output — the OBS capture
    contract. Output is sacred: nothing of ours may be in that picture."""
    host, shown = _run_frames(tmp_path, monkeypatch, OverlayState.HIDDEN)
    assert host.ui._hot == []           # nothing drawn means nothing to hit


def test_hidden_output_is_pixel_identical_to_a_ui_free_render(tmp_path,
                                                              monkeypatch):
    """Directly: the same source through the same mode, once with the window
    in HIDDEN and once headless, must produce the same picture."""
    shown = []
    _patch_gui(monkeypatch, shown=shown)
    lit = _host(tmp_path, show=True, panel=True, max_frames=3)

    def hidden_and_quiet(i):
        lit.overlay = OverlayState.HIDDEN
        # the boot hint is a toast, and toasts legitimately draw in HIDDEN
        # until they fade (DESIGN.md §6.1) — this test is about the chrome
        lit.hud.toasts._hints, lit.hud.toasts._center = [], None

    lit._source.on_read = hidden_and_quiet
    lit.run()

    bare = _host(tmp_path, show=False, max_frames=3)
    _, out = bare.run()
    assert np.array_equal(shown[-1], cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def test_the_chevron_does_not_draw_with_the_panel_disabled(tmp_path,
                                                           monkeypatch):
    """`--ui keys` means no panel, so there is no panel to open."""
    shown = []
    _patch_gui(monkeypatch, shown=shown)
    host = _host(tmp_path, show=True, panel=False, max_frames=2)
    host._source.on_read = lambda i: setattr(host, "overlay", OverlayState.HUD)
    host.run()
    assert host.ui._hot == []


# ---------- pins for behaviour the suite could not see ----------
# Each of these survived the whole suite unchanged: the line could be broken
# and every test still passed. They are shipped promises, so they get pins.


def test_state_json_records_the_active_mode(tmp_path):
    """DESIGN.md §6.4: crash restart resumes the same look, and state.json is
    the single last-mode authority (§7). Nothing checked the mode it wrote."""
    import dtouch.modes as modes

    class OtherMode(DitherGirlMode):
        id = "other"
        title = "Other"

    host = _booted(tmp_path)
    written = presets.load_state(host.state_path)
    assert written["mode"] == "dithergirl"
    assert written["preset"] == host.ui.preset_name

    modes.REGISTRY.append(OtherMode)
    try:
        host._switch_mode("other")                  # a live switch re-autosaves
    finally:
        modes.REGISTRY.remove(OtherMode)
    assert presets.load_state(host.state_path)["mode"] == "other"


def test_state_json_records_the_active_modes_bank(tmp_path):
    host = _booted(tmp_path)
    host.ui.bank = {"4": "phosphor"}
    host._autosave_state()
    assert presets.load_state(host.state_path)["bank"] == {
        "dithergirl": {"4": "phosphor"}}


def test_the_panel_mirror_toggle_reaches_the_frame(tmp_path):
    """The panel's Mirror row sets ui.mirror; the shell copies it every frame
    and flips the picture. Nothing pinned the copy, so the row could stop
    doing anything at all."""
    seen = []

    class Spy(DitherGirlMode):
        def step(self, frame_bgr, audio_levels, dt):
            seen.append(frame_bgr.copy())
            return super().step(frame_bgr, audio_levels, dt)

    class ConstSource(SyntheticSource):
        """Same picture every read, so a difference is the mirror and only
        the mirror."""
        base = np.random.default_rng(3).integers(0, 256, (36, 64, 3), np.uint8)

        def read(self):
            self.reads += 1
            if self.on_read:
                self.on_read(self.reads)
            return True, self.base.copy()

    src = ConstSource()
    host = _host(tmp_path, mode=Spy(), src=src, max_frames=4, mirror=True)

    def on_read(n):
        if n == 3:
            host.ui.mirror = False                  # the panel row, in effect

    src.on_read = on_read
    host.run()

    assert host.mirror is False                     # the host followed the UI
    assert np.array_equal(seen[0], src.base[:, ::-1])   # frame 1 WAS flipped…
    assert np.array_equal(seen[2], src.base)            # …and frame 3 is not


def test_a_mirror_toggle_is_actually_on_the_composed_panel(tmp_path):
    """Mouse parity for the same state (DESIGN.md §6.3): exactly one Mirror
    control on the panel — the mode's own, or the shell's global row when the
    mode does not declare one (the duplicate-control rule, §4.2)."""
    from dtouch.panelspec import Section, Toggle

    host = _booted(tmp_path)
    rows = [w for w in host.ui.iter_widgets()
            if isinstance(w, Toggle) and w.attr == "mirror"]
    assert len(rows) == 1
    before = host.ui.mirror
    host.ui._activate(rows[0].attr, None, 0)        # what a click does
    assert host.ui.mirror is not before


def test_the_mic_starts_and_stops_with_the_sound_toggle(tmp_path,
                                                        monkeypatch):
    """`a` / the Sound react row flips ui.audio; the shell owns the LiveMic
    lifecycle off it. A leaked mic is a held OS device."""
    class FakeMic:
        instances = []

        def __init__(self):
            self.started = self.stopped = False
            self.available = False
            FakeMic.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def levels(self):
            return None

    monkeypatch.setattr(shell_mod, "LiveMic", FakeMic)
    host = _host(tmp_path, max_frames=7)
    seen = {}

    def on_read(n):
        # _sync_host_state runs after this hook, so frame N observes the
        # lifecycle decision frame N-1 made
        if n == 2:
            host.ui.audio = True
        if n == 4:
            seen["armed"] = host.mic
        if n == 5:
            host.ui.audio = False
        if n == 7:
            seen["disarmed"] = host.mic

    host._source.on_read = on_read
    host.run()

    assert len(FakeMic.instances) == 1              # one mic, opened once
    mic = FakeMic.instances[0]
    assert seen["armed"] is mic and mic.started
    # released the moment Sound react went off — NOT merely at teardown, which
    # would leave a live OS mic held for the rest of the performance
    assert seen["disarmed"] is None
    assert mic.stopped
    assert host.mic is None


def test_the_mic_is_released_when_the_run_ends(tmp_path, monkeypatch):
    class FakeMic:
        def __init__(self):
            self.stopped = False
            self.available = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def levels(self):
            return None

    monkeypatch.setattr(shell_mod, "LiveMic", FakeMic)
    host = _host(tmp_path, max_frames=3)
    host._source.on_read = lambda n: setattr(host.ui, "audio", True)
    host.run()
    assert host.mic is None


def test_camera_loss_reaches_the_hud(tmp_path, monkeypatch):
    """DESIGN.md §6.4: hold the last good frame AND say so. The saying-so half
    is a kwarg the HUD receives — nothing checked it ever arrived True."""
    _patch_gui(monkeypatch)
    host = _host(tmp_path, src=SyntheticSource(fail_after=2), show=True,
                 max_frames=6)
    seen = []
    real = host.hud.draw
    host.hud.draw = (lambda img, state, **kw:
                     seen.append(kw.get("camera_lost")) or real(img, state, **kw))
    host.run()
    assert seen[:2] == [False, False]               # frames still arriving
    assert all(seen[2:]), "camera loss never reached the HUD"


class BlackSource(SyntheticSource):
    """Frames keep arriving, and every one of them is black."""

    def read(self):
        self.reads += 1
        return True, np.zeros((self.h, self.w, 3), np.uint8)


def test_a_long_black_streak_reaches_the_hud(tmp_path, monkeypatch):
    """The other half of the same note: frames arrive, but they are black
    (the iPhone Continuity Camera case the message names)."""
    _patch_gui(monkeypatch)
    host = _host(tmp_path, src=BlackSource(), show=True, max_frames=20)
    seen = []
    real = host.hud.draw
    host.hud.draw = (lambda img, state, **kw:
                     seen.append(kw.get("camera_black")) or real(img, state, **kw))
    host.run()
    assert seen[:15] == [False] * 15                # the streak has to build
    assert seen[-1] is True


def test_a_read_failure_is_not_told_to_turn_off_continuity_camera(
        tmp_path, monkeypatch):
    """ANY failed read printed `CAMERA IS BLACK - disable iPhone Continuity
    Camera`, so an unplugged camera — or one Zoom had already taken — sent the
    operator into iPhone settings after a phone that was never involved. The
    two faults have two different fixes and now say so."""
    _patch_gui(monkeypatch)
    host = _host(tmp_path, src=SyntheticSource(fail_after=2), show=True,
                 max_frames=6)
    seen = []
    real = host.hud.draw
    host.hud.draw = (lambda img, state, **kw:
                     seen.append((kw.get("camera_lost"), kw.get("camera_black")))
                     or real(img, state, **kw))
    host.run()
    assert seen[-1][0] is True, "the read failure never reached the HUD"
    assert not seen[-1][1], "a failed read is not the black-frame symptom"

    drawn = " | ".join(_hud_strings(host.hud, monkeypatch, camera_lost=True))
    assert "Continuity" not in drawn
    assert "STOPPED SENDING FRAMES" in drawn


def test_the_black_frame_streak_keeps_the_continuity_camera_wording(
        tmp_path, monkeypatch):
    """The wording is right for the fault it names — frames ARRIVING but black
    — and DESIGN.md §6.4 keeps it verbatim for that case."""
    host = _host(tmp_path, max_frames=1)
    drawn = " | ".join(_hud_strings(host.hud, monkeypatch, camera_black=True))
    assert "CAMERA IS BLACK - disable iPhone Continuity Camera" in drawn


def test_an_unbanked_look_takes_slot_one_first(tmp_path):
    """DESIGN.md §6.3: a slot-badge click takes the NEXT FREE slot, and on an
    empty bank that is 1 — the slot a performer reaches for first."""
    host = _booted(tmp_path)
    host.ui.bank = {}
    host._assign_slot("phosphor")
    assert host.ui.bank == {"1": "phosphor"}
    host._assign_slot("classic")
    assert host.ui.bank == {"1": "phosphor", "2": "classic"}
    host._assign_slot("phosphor")                   # clicking again clears it
    assert host.ui.bank == {"2": "classic"}
    host._assign_slot("newsprint")                  # 1 is free again
    assert host.ui.bank == {"1": "newsprint", "2": "classic"}


def test_a_full_bank_says_so_instead_of_dropping_the_click(tmp_path):
    host = _booted(tmp_path)
    host.ui.bank = {str(i): "x%d" % i for i in range(1, 10)}
    host._assign_slot("phosphor")
    assert "phosphor" not in host.ui.bank.values()
    assert any("bank full" in t for t in _hints(host))


# ---------- a look's VALUES are not validated by the store either ----------
# The earlier hardening covered the preset file's SHAPE. Its values still went
# straight into `float()`, and the boot application sat outside every guard —
# so a hand-edited (or half-merged) presets.json ended the launch with a
# traceback and no window, the one failure a performer cannot work around.


def _write_look(tmp_path, name="broken", **over):
    """A well-SHAPED v2 file whose look carries a bad value."""
    cfg = dict(DitherGirlMode.BUILTIN["classic"])
    cfg.update(over)
    with open(str(tmp_path / "presets.json"), "w") as f:
        json.dump({"version": 2, "meta": {},
                   "modes": {"dithergirl": {"looks": {name: cfg}}}}, f)
    return name


def test_a_look_with_an_unusable_value_still_boots(tmp_path):
    """`"contrast": "high"` — well-shaped, unusable, and fatal at boot."""
    name = _write_look(tmp_path, contrast="high")
    host = _host(tmp_path, max_frames=1, preset=name)
    count, out = host.run()

    assert count == 1 and out is not None            # the show started
    assert host.ui.dg_contrast == 1.0                # bad key skipped…
    assert host.ui.dg_scale == 72.0                  # …the rest of it loaded
    assert any("unusable values" in t and "contrast" in t for t in _hints(host))


def test_a_resumed_look_with_an_unusable_value_still_boots(tmp_path):
    """The same file reached without naming it: state.json resume (§6.4) is
    the path a crashed session comes back through, so it must not be the path
    that keeps it dead."""
    name = _write_look(tmp_path, scale="wide")
    presets.save_state({"mode": "dithergirl", "preset": name, "bank": {}},
                       path=str(tmp_path / "state.json"))
    host = _host(tmp_path, max_frames=1, preset=None)
    count, _ = host.run()
    assert count == 1
    assert any("unusable values" in t for t in _hints(host))


def test_a_store_note_reaches_the_operator_on_its_own_frame(tmp_path):
    """The store queues warnings on a module-global list that only two places
    drain, both of them after an operation they then blame the note on. Every
    other store call — the boot state.json read, the autosave, the bank and
    setlist reads after a reload — queued and drained nothing, so a note could
    wait there and surface later as the reason an unrelated save refused: a
    data-loss message about the wrong file, which is the same bug as silence
    about the right one (DESIGN.md §9)."""
    host = _booted(tmp_path)                         # boot notes already drained
    presets._note("state.json unreadable - backup saved to state.corrupt.json")

    host._pump_preset_mailboxes()                    # one frame of the loop

    assert any("state.json unreadable" in t for t in _hints(host))
    assert presets.take_notes() == []                # nothing left to misattribute


def test_a_look_that_raises_cannot_re_raise_every_frame(tmp_path, monkeypatch):
    """The mailbox was cleared AFTER the apply, so a look that raised left it
    armed: the next frame applied the same bad look and raised again, forever,
    at frame rate, with no way to select a different one."""
    host = _booted(tmp_path)
    calls = []

    def boom(*a, **kw):
        calls.append(a)
        raise ValueError("could not convert string to float: 'high'")

    monkeypatch.setattr(shell_mod, "apply_look", boom)
    host.ui.pending_preset = "classic"
    for _ in range(5):                               # five frames of pumping
        host._pump_preset_mailboxes()

    assert len(calls) == 1                           # applied once, not forever
    assert host.ui.pending_preset is None            # mailbox disarmed
    assert host.hud.toasts._center.text == "look not loaded - classic"


# ---------- F1: an invisible rename box must not eat the keyboard ----------

def _present(host, key=255):
    """One loop iteration's present half: compose the frame, then route one key.

    Exactly the order run() uses — _compose_frame, then the waitKey code — so
    what these tests exercise is the shipped per-frame contract, not a helper.
    """
    out = np.zeros((RES[1], RES[0], 3), np.uint8)
    frame = np.zeros((36, 64, 3), np.uint8)
    host._compose_frame(out, frame)
    host._route_key(key)


def _click(host, kind):
    """Click the centre of the panel's published hit rect for `kind`."""
    rect = next(r for r, k, _p in host.ui._hot if k == kind)
    x0, y0, x1, y1 = rect
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0)


def _renaming_host(tmp_path):
    """A booted host in PANEL with a saved look's rename box open."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.PANEL
    host.reg.dispatch(ord("s"))                      # save opens the rename box
    host._pump_preset_mailboxes()
    assert host.ui.renaming is not None
    return host


def test_collapsing_the_panel_cancels_the_rename_that_would_eat_every_key(tmp_path):
    """Two clicks reached a stuck state: rename pencil, then collapse chevron.

    The sidebar collapsed, `renaming` stayed set, and `OverlayUI.on_key`
    consumes EVERY key while renaming (correctly — `q` must not quit
    mid-typing, DESIGN.md §6.2). So the screen showed a completely normal
    instrument, bottom hint and all, while TAB, `m`, `?`, space, `0` and both
    presses of `q` were typed into a box nobody could see. The measured buffer
    for that exact sequence was `"classicm? 0qq"` with quit still False; only
    Esc got out, and nothing on screen said so.
    """
    host = _renaming_host(tmp_path)
    _present(host)                                   # a frame that DRAWS the box
    assert host.ui.renaming is not None, "a visible box keeps the keyboard"

    _click(host, "collapse")                         # the second of the two clicks
    assert host.ui.open is False
    _present(host)                                   # the box is no longer drawn

    assert host.ui.renaming is None
    assert host.ui.rename_buf == ""
    # ...and the keyboard is answering again: the three keys the on-screen
    # hint names, then the quit it refused.
    _present(host, 9)                                # TAB
    assert host.overlay is not OverlayState.PANEL
    t = [100.0]
    host.scene_clock = lambda: t[0]
    _present(host, ord("m"))
    assert host.menu.open is True
    t[0] += 0.6          # a person's pause: m and Esc share the menu's gate
    _present(host, 27)                               # Esc closes the menu
    _present(host, ord("?"))
    assert host.ps.help_open is True
    _present(host, ord("?"))                         # any key closes help
    _present(host, ord("q"))
    _present(host, ord("q"))
    assert host.ps.quit is True, "q must quit once the box is gone"


def test_hiding_the_overlay_cancels_the_rename(tmp_path):
    """Same cause, second route: a provably clean output frame with a hidden
    text field swallowing everything (measured buffer `"m?qqq"`, quit False)."""
    host = _renaming_host(tmp_path)
    _present(host)
    assert host.ui.renaming is not None

    host.overlay = OverlayState.HIDDEN               # the panel stops drawing
    _present(host)

    assert host.ui.renaming is None
    _present(host, ord("q"))
    _present(host, ord("q"))
    assert host.ps.quit is True


def test_opening_the_menu_cancels_the_rename(tmp_path):
    """Third route: the panel's own `Menu (M)` row. The menu routes keys first
    so the menu itself worked — but `renaming` stayed set underneath, and the
    keyboard went deaf again the moment a card was picked."""
    host = _renaming_host(tmp_path)
    _present(host)
    assert host.ui.renaming is not None

    host.menu.show(host.mode.id)
    _present(host)                                   # menu frame: no panel drawn

    assert host.ui.renaming is None
    host.menu.close()
    _present(host, ord("q"))
    _present(host, ord("q"))
    assert host.ps.quit is True


def test_a_visible_rename_box_still_owns_the_keyboard(tmp_path):
    """The fix must not weaken the contract it protects: while the box IS on
    screen, every key is still typed into it — `q` included (DESIGN.md §6.2)."""
    host = _renaming_host(tmp_path)
    start = host.ui.rename_buf
    for _ in range(6):
        _present(host, ord("q"))

    assert host.ui.renaming is not None
    assert host.ui.rename_buf == start + "qqqqqq"
    assert host.ps.quit is False
    assert host.overlay is OverlayState.PANEL        # TAB never fired either


def test_scrolling_the_rename_box_off_the_top_cancels_it(tmp_path):
    """Fourth route, and the only one that needs no second control: the panel
    column is several windows tall, so a few wheel notches put the open rename
    box above the frame. cv2 clips off-frame draws silently, so the walk still
    "drew" the box every frame — reaching the row is not painting it — and the
    measured buffer for TAB s <wheel> m ? 0 q q was `mine_112710m?qq` with
    quit still False and zero box pixels on screen. The flag must be earned by
    the drawn rect actually intersecting the frame."""
    host = _renaming_host(tmp_path)
    _present(host)                                   # reveal queued if needed
    _present(host)                                   # box painted
    assert host.ui.renaming is not None

    # while the box IS on screen the contract is unweakened: typing still
    # captures, and `q` lands in the buffer instead of quitting
    _present(host, ord("q"))
    assert host.ui.rename_buf.endswith("q")
    assert host.ps.quit is False

    # real wheel events through the real mouse route (Cocoa encoding: the
    # delta rides in (x, y); negative = toward the user = scroll down)
    for _ in range(24):
        host._on_mouse(cv2.EVENT_MOUSEWHEEL, 0, -3, 0)
        _present(host)
    assert host.ui.scroll > 0
    assert host.ui.renaming is None, "an off-screen box kept the keyboard"

    _present(host, ord("q"))
    _present(host, ord("q"))
    assert host.ps.quit is True, "q must quit once the box is gone"


def test_a_rename_box_born_below_the_fold_is_scrolled_into_view(tmp_path):
    """The visibility gate must not break the flow it guards: save appends the
    new look's row, and with enough looks (or a small window) that row opens
    BELOW the fold. Silently expiring a rename the operator just asked for
    would send the name they type to the global keys — `q q` would quit the
    show mid-naming. Instead the next draw scrolls the row into view, and the
    box keeps the keyboard because it is genuinely painted."""
    host = _renaming_host(tmp_path)
    # the harness window is 108 px against a ~1200 px column, so the fresh
    # save's row starts off-screen — the exact case
    _present(host)
    assert host.ui.renaming is not None              # grace: reveal queued
    _present(host)                                   # reveal applied: painted
    assert host.ui.scroll > 0, "the panel did not scroll to the box"
    assert host.ui.renaming is not None
    for ch in "x!":
        _present(host, ord(ch))
    assert host.ui.rename_buf.endswith("x!")         # typing lands in the box


# ---------- a named look must be a loaded look ----------

def test_a_boot_preset_the_mode_does_not_own_falls_back_to_its_safe_look(tmp_path):
    """The panel must never name a look it did not apply.

    `Host`'s own default preset is a Particles look, and `--preset` resolution
    can hand a name from the other mode, so a boot preset the active mode does
    not own is reachable. It used to fall straight through the apply and leave
    the look UNAPPLIED: `preset_name` showed the name, and every parameter was
    whatever `_UI_DEFAULTS` said.

    It hid for as long as it did because Dither's first built-in was `classic`,
    whose values are identical to its `_UI_DEFAULTS` — the no-op and the
    correct result rendered the same frame. Changing the first look is what
    made it visible.
    """
    import numpy as np
    from dtouch.modes.dithergirl import ALGOS, DitherGirlMode

    class Syn:
        name = "synthetic"
        def read(self):
            return True, np.full((54, 96, 3), 128, np.uint8)
        def release(self):
            pass

    mode = DitherGirlMode()
    host = Host(mode, source=Syn(), res=(96, 54), show=False, max_frames=2,
                preset="abstract",                  # a Particles look
                presets_path=str(tmp_path / "p.json"),
                state_path=str(tmp_path / "s.json"))
    host.run()
    ui = host.ui
    safe = mode.safe_look()
    assert ui.preset_name == safe
    # ...and the parameters are that look's, not the UI defaults
    want = mode.BUILTIN[safe]
    assert ALGOS[ui.dg_algo_idx] == want["algorithm"]
    assert ui.dg_scale == want["scale"]
    # the guard has to bite on a name that exists in NO mode, too
    host2 = Host(DitherGirlMode(), source=Syn(), res=(96, 54), show=False,
                 max_frames=2, preset="no-such-look-anywhere",
                 presets_path=str(tmp_path / "p2.json"),
                 state_path=str(tmp_path / "s2.json"))
    host2.run()
    assert host2.ui.preset_name == safe
    assert ALGOS[host2.ui.dg_algo_idx] == want["algorithm"]
