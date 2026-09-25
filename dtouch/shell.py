"""The shell (Host) — owns the show; modes are plugins racked into it.

DESIGN.md §2.1 split: the shell owns the window (WINDOW_AUTOSIZE, refit on res
change), camera capture + loss degradation (hold last good frame + amber HUD
note — modes never see None), mouse routing, key routing through the command
registry, audio (LiveMic, lazy), the recorder (post-FX, pre-panel — the shipped
invariant), preset store CRUD, the overlay state machine + HUD, blackout and
panic, the shared SIGNAL post-FX rack (CircuitBent applied after mode output +
video composite — §2.4: Glitch is not a mode), fps/status, mode lifecycle
(idempotent stop → start; a start() exception toasts and reinstates the
previous mode), and quit (double-q confirm).

Testability: Host accepts an injectable frame source (any object with
cv2-style ``read() -> (ok, frame_bgr)`` and optional ``release()``/``name``),
so tests drive the full loop headless with synthetic frames; ``show=False`` /
``max_frames`` semantics are live_flow's, kept.
"""
from __future__ import annotations

import os
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from .audio import LiveMic
from .camera import open_camera
from .circuit_bent import CircuitBent
from .commands import Command, CommandRegistry
from .hud import (AMBER, RED, Hud, OverlayState, cycle_overlay,
                  draw_corner_tick, draw_help, esc_overlay, put_outlined,
                  u as _u)
from .imgui import in_rect, panel_handle, panel_handle_rect
from .auto import Autopilot
from .menu import Menu, draw_menu, render_boot_card
from .modes import REGISTRY, mode_by_id
from .overlay_ui import (BASE_H, OverlayUI, build_signal_section,
                         build_global_rows, sync_signal)
from .panelspec import (Cycle, Section, Slider, apply_look, capture_look,
                        display_fmt, nudge_to, nudgeable, visible)
from . import presets as _presets

# Keys that release the autopilot. Anything that changes WHICH SCENE is on
# screen: look recall and stepping, parameter selection and nudges, the
# physarum point swap, mode switches, panic.
#
# Deliberately NOT the casts and toggles — burst, wave, Z's random cast,
# glitch, audio, record, menu, help, debug, save. You should be able to throw
# a burst over a running autopilot without ending it, the same way you can
# lean on an instrument someone else is playing; Z in particular is a cast,
# not a takeover. The rule has to be learnable by accident, so it is one set
# in one place rather than a per-command flag.
# Bank slots (keys 1-9) seeded from a mode's built-ins at first boot. Short of
# nine on purpose — see _seed_bank_setlist.
BANK_SLOTS = 9
BANK_SEED_MAX = 7

AUTO_RELEASE_KEYS = frozenset(
    [ord(c) for c in "0123456789[],.-=_+xpdo"]
)

REC_DIR = "out"        # recordings land beside the launch dir; created on first take
RESUME_HINT = "enter resumes"    # boot-menu hint prefix; retired when Enter lands
ERR_TOAST_S = 5.0      # a repeating per-frame error re-toasts at most this often
ERR_TOAST_MAX = 3      # ...and at most this many DISTINCT errors per window
ERR_PRINT_MAX = 20     # ...and at most this many stdout lines per window
ERR_SEEN_MAX = 64      # ...and the remembered-key map never grows past this
PANEL_OPEN = "panel.open"   # the HUD chevron and TAB reach the same named command

# A produce half that fails every frame must idle, not spin. Containment
# removed the exception that used to end the run, so a permanent failure became
# a busy loop: measured 983 iterations/sec against a frozen picture, pegging a
# core for as long as the show lasts. One frame-time of sleep on the failure
# path costs a healthy loop nothing and holds a broken one at ~30 Hz — the rate
# it would run at anyway.
FRAME_FAIL_SLEEP_S = 1.0 / 30.0
# ...and headless there is no window to explain itself and no `q` to stop it,
# so a permanent failure ends the run with the real cause instead of spinning
# forever behind no output at all. ~4 s at the throttled rate.
FRAME_FAIL_LIMIT = 120
# The largest dt a mode is ever handed (DESIGN.md §2.2). A stall accumulates
# wall-clock between two step() calls; a mode that integrates it moves a
# multi-second jump in one frame (particles teleport, fades blow out). A
# quarter second = 4 fps: below that the show is already visibly broken, and
# capping is the honest lie.
DT_MAX = 0.25


class PerformState:
    """Mutable perform-layer flags (DESIGN.md §6.2), owned by the shell."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self.blackout = False     # output.blackout — hard black, recorded
        self.help_open = False    # '?' overlay; any key closes it
        self.quit = False
        self._q_at = None         # first-press time of the quit confirm

    def q_pressed(self):
        """Quit confirm: 'q' twice within 2 s quits. True when this press quits."""
        t = self._now()
        if self._q_at is not None and t - self._q_at <= 2.0:
            self.quit = True
            return True
        self._q_at = t
        return False


def ui_toggle_command(ui, toasts, name, label, key, attr, flash_label):
    """A named command that flips a UI-state attr with momentary feedback.
    Shared by the shell's global toggles and the modes' local ones, so both
    paths stay one implementation."""
    def run():
        v = not getattr(ui, attr)
        setattr(ui, attr, v)
        toasts.flash(f"{flash_label} ON" if v else f"{flash_label.lower()} off")
    return Command(name, label, key, run)


def _register_quit(reg, ps, toasts):
    def run():
        if not ps.q_pressed():
            toasts.hint("q again to quit")
    reg.add("app.quit", "Quit (press twice)", "q", run)


def _wire_perform_keys(reg, ui, hud, ps, recall, mode_commands=None,
                       safe_look=None, get_overlay=None, debug_line=None):
    """Register the perform layer (DESIGN.md §6.2) on `reg`.

    `recall(name)` must route a preset apply through the same path a panel click
    takes (the shell's pending_preset mailbox). Commands read `ui.presets`,
    `ui.bank`, and `ui.setlist` at press time, so saves/deletes/assignments are
    picked up live. Digits 1–9 recall from the ACTIVE mode's bank — explicit
    slot assignments, not list positions (DESIGN.md §7); `[`/`]` walk the
    mode's setlist (empty = all looks in load order). The panel's own toggles
    remain and stay in sync: both paths set the same ui attributes.

    `mode_commands` is the active mode's `commands()` dict; when None the
    Particles-local toggles (F flock, V video bg) are registered directly —
    same commands, same keys, so direct callers see the full shipped key set.
    """
    toasts = hud.toasts

    def _slot_of(name):
        return next((s for s, n in (ui.bank or {}).items() if n == name), None)

    def _recall(name):
        ui.preset_idx = ui.presets.index(name)
        recall(name)

    def apply_slot(i):
        name = (ui.bank or {}).get(str(i))
        if not name or name not in ui.presets:
            toasts.hint(f"no preset {i}")
            return
        _recall(name)
        toasts.flash(f"{i} - {name}")

    def setlist_step(d):
        order = [n for n in (ui.setlist or list(ui.presets)) if n in ui.presets]
        if not order:
            toasts.hint("setlist empty")
            return
        cur = ui.preset_name if ui.preset_idx < len(ui.presets) else None
        if cur in order:
            i = (order.index(cur) + d) % len(order)
        else:
            i = 0 if d > 0 else len(order) - 1
        name = order[i]
        _recall(name)
        slot = _slot_of(name)
        toasts.flash(f"{slot} - {name}" if slot else name)

    def blackout():
        ps.blackout = not ps.blackout
        if ps.blackout:
            toasts.flash("BLACKOUT", AMBER)
        else:
            toasts.hint("blackout off")

    def panic():
        # Amended DESIGN.md §6.2 '0': panic restores a known-good PICTURE —
        # the mode's safe_look, blackout disarmed, SIGNAL rack (glitch) off.
        ps.blackout = False
        ui.glitch = False
        name = safe_look() if safe_look is not None else ui.preset_name
        if isinstance(name, str):
            if name in ui.presets:
                _recall(name)
            else:
                recall(name)
        toasts.flash("RESET", AMBER)

    def record():
        ui.record = not ui.record
        if ui.record:
            toasts.flash("REC", RED)
        # the stop toast (filename) comes from the loop when the file closes

    def save_preset():
        # 's' = preset.save (amended DESIGN.md §3/§6.2): PANEL state only —
        # saving is an edit action; elsewhere it hints instead of silence.
        state = (get_overlay() if get_overlay is not None
                 else OverlayState.PANEL)
        if state is not OverlayState.PANEL:
            toasts.hint("save is a panel action - TAB to open the panel")
            return
        ui.pending_save = True

    # ----- param nudging without the panel (DESIGN.md §6.2) -----
    # ','/'.' select prev/next nudgeable control (Sliders + Cycles from the
    # composed spec, spec order); '-'/'=' nudge by 1/40 of range ('_'/'+' =
    # x5 on a continuous slider, one whole step where the quantum is coarser
    # than that — Bits, Crush; cycles rotate by one option). The OSD (name +
    # value + bar) is the
    # feedback, and it works in every overlay state. Inside the open menu
    # ','/'.' move card selection instead — the menu consumes keys before the
    # registry (Host._route_key), so priority is already right.
    def _nudgeables(gated=False):
        # `nudgeable`, not "every Slider and Cycle": output resolution opts out
        # (see its docstring — one key must not resize the show). Gated-off
        # rows (panelspec.visible — SIGNAL under Glitch-off, boids gains under
        # Flock-off) are skipped too: nudging a control the panel does not
        # show and the engine does not read is the same dead surface on keys.
        rows = [w for w in ui.iter_widgets() if nudgeable(w)]
        return rows if gated else [w for w in rows if visible(ui, w)]

    def _selected(ws):
        """(index, adrift) — where the selection sits in the CURRENT list.

        The list is filtered by `visible`, so pressing G or F changes its
        length and membership under a selection that is only a NUMBER. The
        selection is therefore anchored to the chosen control's `attr`
        (ui.nudge_attr) and the index re-derived from it every press:
        selecting Crush and turning Glitch off used to leave nudge_idx
        pointing at whatever row had slid into slot 20, and the next '-'
        edited THAT control (Particles: Crush -> Size) before the OSD named
        it. `adrift` says the anchored control is gated off now, so the
        caller re-anchors and shows the OSD INSTEAD of writing a value —
        one press to see where you are, never a silent edit elsewhere.
        """
        attr = getattr(ui, "nudge_attr", None)
        if attr is None:
            return ui.nudge_idx % len(ws), False
        for i, w in enumerate(ws):
            if getattr(w, "attr", None) == attr:
                return i, False
        # gated off: land on the nearest visible row in SPEC order, not on
        # whatever now occupies the old index
        rows = _nudgeables(gated=True)
        pos = next((i for i, w in enumerate(rows)
                    if getattr(w, "attr", None) == attr), 0)
        before = sum(1 for w in rows[:pos] if visible(ui, w))
        return min(before, len(ws) - 1), True

    def _osd_show(w):
        val = getattr(ui, w.attr)
        if isinstance(w, Cycle):
            opts = list(w.options)
            hud.osd.show(w.label, str(opts[int(val) % len(opts)]))
        else:
            # display_fmt: a stored legacy fraction on an engine-continuous
            # slider shows one honest decimal instead of a rounded-up lie
            hud.osd.show(w.label, float(val), w.lo, w.hi,
                         fmt="{:%s}" % display_fmt(w, val))

    def nudge_select(d):
        ws = _nudgeables()
        if not ws:
            return
        i, adrift = _selected(ws)
        # an adrift press re-anchors where the selection actually is; the
        # step comes on the next one
        ui.nudge_idx = i if adrift else (i + d) % len(ws)
        ui.nudge_attr = getattr(ws[ui.nudge_idx], "attr", None)
        _osd_show(ws[ui.nudge_idx])

    def nudge(d, big=False):
        ws = _nudgeables()
        if not ws:
            return
        i, adrift = _selected(ws)
        ui.nudge_idx = i
        w = ws[i]
        ui.nudge_attr = getattr(w, "attr", None)
        if adrift:
            # the selected control was gated off between presses — show what
            # the keys now hold and change NOTHING this press
            _osd_show(w)
            return
        if isinstance(w, Cycle):
            opts = list(w.options)
            setattr(ui, w.attr, (int(getattr(ui, w.attr)) + d) % len(opts))
        else:
            cur = float(getattr(ui, w.attr))
            step = (w.hi - w.lo) / 40.0 * (5.0 if big else 1.0)
            val = min(max(cur + d * step, w.lo), w.hi)
            if w.step:
                # the nudge is an input surface, so it snaps here explicitly:
                # OverlayUI.__setattr__ only does it for engine-snapped
                # sliders, and Scale/Hue (engine-continuous, quantised for
                # control feel) must not drift onto fractions under the keys.
                # Idempotent for the engine-snapped ones.
                val = nudge_to(w, val, cur)
            setattr(ui, w.attr, val)
        _osd_show(w)

    reg.add("param.prev", "Select prev param", ",", lambda: nudge_select(-1))
    reg.add("param.next", "Select next param", ".", lambda: nudge_select(+1))
    reg.add("param.down", "Nudge param down", "-", lambda: nudge(-1))
    reg.add("param.up", "Nudge param up", "=", lambda: nudge(+1))
    # "big", not "x5": on the whole-number sliders whose step is coarser than
    # five presses (Bits, Crush) a big nudge moves one whole step, so a label
    # promising x5 would lie on exactly the controls it moves least.
    reg.add("param.down.big", "Big nudge param down", "_",
            lambda: nudge(-1, big=True))
    reg.add("param.up.big", "Big nudge param up", "+",
            lambda: nudge(+1, big=True))

    reg.add("output.blackout", "Blackout", " ", blackout)
    reg.add("preset.panic", "Panic reset", "0", panic)
    reg.add("preset.save", "Save current look", "s", save_preset)
    for i in range(1, 10):
        reg.add(f"preset.recall.{i}", f"Recall bank slot {i}", str(i),
                lambda i=i: apply_slot(i))
    reg.add("preset.prev", "Previous in setlist", "[", lambda: setlist_step(-1))
    reg.add("preset.next", "Next in setlist", "]", lambda: setlist_step(+1))
    if mode_commands is None:
        mode_commands = {
            "layer.flock": ui_toggle_command(ui, toasts, "layer.flock", "Flock",
                                             "f", "flock", "FLOCK"),
            "video_bg.toggle": ui_toggle_command(ui, toasts, "video_bg.toggle",
                                                 "Video background", "v",
                                                 "video_bg", "VIDEO BG"),
        }
    for cmd in mode_commands.values():
        reg.register(cmd)
    reg.register(ui_toggle_command(ui, toasts, "layer.glitch", "Glitch",
                                   "g", "glitch", "GLITCH"))
    reg.register(ui_toggle_command(ui, toasts, "audio.toggle", "Sound react",
                                   "a", "audio", "SOUND"))
    reg.add("record.toggle", "Record", "r", record)

    def debug_toggle():
        # feedback: the HUD status-line variant. In HIDDEN there is no status
        # line, so 'i' posts the debug line as a transient toast instead —
        # silence-on-input is a bug (DESIGN.md principle 4).
        hud.debug = not hud.debug
        state = get_overlay() if get_overlay is not None else None
        if state is OverlayState.HIDDEN and debug_line is not None:
            toasts.hint(debug_line())
    reg.add("debug.toggle", "Debug readout", "i", debug_toggle)
    reg.add("help.overlay", "Key map", "?",
            lambda: setattr(ps, "help_open", True))
    _register_quit(reg, ps, toasts)
    reg.on_unknown = lambda code: toasts.hint("? for keys")


def _perform_key(key, overlay, ps, reg, hud):
    """One unconsumed keypress through the perform layer. Returns the overlay state.

    Order matters: an open help overlay eats the key (any key closes it), then
    TAB/Esc step the overlay state, then the command registry dispatches —
    unknown printable keys fall through to the gentle '? for keys' toast.
    """
    if ps.help_open:
        ps.help_open = False
        return overlay
    overlay, handled = _overlay_key(key, overlay, hud.toasts)
    if not handled:
        reg.dispatch(key)
    return overlay


def _overlay_key(key, state, toasts):
    """TAB/Esc overlay-state stepping (DESIGN.md §6.1). Returns (state, handled).

    TAB cycles HIDDEN -> HUD -> PANEL -> HIDDEN; Esc steps one toward HIDDEN and
    in HIDDEN does nothing — Esc never quits. Entering HIDDEN emits one final
    toast, then the output is provably clean once it fades.
    """
    if key == 9:       # TAB
        new = cycle_overlay(state)
    elif key == 27:    # Esc
        new = esc_overlay(state)
    else:
        return state, False
    if new is OverlayState.HIDDEN and state is not OverlayState.HIDDEN:
        toasts.hint("overlay hidden - TAB to show")
    return new, True


class CameraSource:
    """The default frame source — wraps camera selection (dtouch.camera).

    A camera that will not open is NOT fatal (DESIGN.md §6.4). `open_camera`
    raises whenever `isOpened()` is False — no camera attached, macOS TCC
    denied, or the device already held by Zoom/OBS — and that traceback used
    to land before any window existed, so the plain-language permission card
    the design promises could never be shown.

    Instead the source starts empty: every `read()` reports failure (the shell
    draws its 'waiting for camera' card and stays quittable) and the open is
    retried about once a second, so the show starts by itself the moment the
    camera appears or the other app lets go of it.
    """

    RETRY_S = 1.0

    def __init__(self, device="builtin", now=time.monotonic):
        self._device = device
        self._now = now
        self.cap = None
        self.name = "no camera"
        self.error = None          # the last open failure, for the boot toast
        self._retry_at = 0.0
        self._open()

    def _open(self):
        self._retry_at = self._now() + self.RETRY_S
        try:
            self.cap, self.name = open_camera(self._device)
            self.error = None
        except Exception as e:                       # noqa: BLE001 — §6.4
            self.cap, self.error = None, e

    def read(self):
        if self.cap is None:
            if self._now() >= self._retry_at:
                self._open()                          # automatic recovery
            if self.cap is None:
                return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


class StillSource:
    """A loaded still image as a frame source (DESIGN.md §2.1: the shell owns
    still sources; modes never special-case stills). read() returns the same
    frame every tick."""

    def __init__(self, path):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not read still image: {path}")
        self.frame = frame
        self.name = os.path.basename(path)

    def read(self):
        return True, self.frame

    def release(self):
        pass


class Host:
    """The shell. Construct with a mode (and optionally an injected frame
    source), then ``run()`` — returns ``(frame_count, last_rgb_frame)`` exactly
    like live_flow did."""

    WIN = "lighteater - flow"

    def __init__(self, mode, source=None, device="builtin", res=(1920, 1080),
                 mirror=True, seed=1, preset="abstract", audio=False,
                 panel=True, show=True, max_frames=None, still=None,
                 presets_path="presets.json", state_path="state.json",
                 now=time.monotonic):
        self.mode = None
        # injectable clock, same pattern as PerformState: the error rate limit
        # is a TIME window, and a window nothing can advance is untestable
        self._now = now
        self._boot_mode = mode
        # No explicit mode = launch opens on the HOME MENU (DESIGN.md §3,
        # amended 2026-08-15 — the user's call). The last-used mode still boots
        # and runs live BEHIND the menu, so the menu is not a dead screen and
        # Enter is a one-key resume. An explicit mode (--mode/--still/an engine
        # flag) still goes straight in: a flag means "boot into", and making a
        # flag wait through a menu would be a worse launch than we had.
        self._boot_menu = mode is None
        self._source = source
        self._device = device
        self._still_path = still if isinstance(still, str) else None
        self.still = still if not isinstance(still, str) else None
        self.res = tuple(res)
        self.mirror = mirror
        self.seed = seed
        self._boot_preset = preset
        self._boot_audio = audio
        self.panel = panel
        self.show = show
        self.max_frames = max_frames
        self.presets_path = presets_path
        self.state_path = state_path

        self.ui = None
        self.hud = Hud()
        self.overlay = OverlayState.HUD   # boot state: HUD (DESIGN.md §6.1)
        self.ps = PerformState()
        self.reg = CommandRegistry()
        self.menu = Menu()                # home menu — a shell overlay state (§3)
        self.auto = Autopilot()           # the AUTO card — dtouch.auto
        self.pending_mode = None          # mode id posted by a key/menu commit
        self._mode_instances = {}         # id -> constructed Mode (reused on switch)
        self._mode_preset = {}            # id -> last selected look (re-entry)
        self.help_rows = []
        self.cb = None                    # SIGNAL rack post-FX, built on first use
        self.mic = None
        self.writer = None
        self.rec_path = None
        self.all_presets = {}
        self.cam_name = "?"
        self.fps = 0.0
        self._err_seen = {}       # every (type, message) seen inside the window
        self._err_flood = False   # ...more distinct errors than fit in a window
        self._err_prints = 0      # stdout lines spent on errors this window…
        self._err_print_t0 = None  # …and when that window opened
        self._frame_ok = True     # did THIS frame's produce half succeed?

    # ----- mode lifecycle (DESIGN.md §2.1) -----
    def _resolve_boot_mode(self):
        """Boot with no explicit mode (DESIGN.md §3, amended: launch opens on
        the home menu with the LAST-USED mode running behind it and selected).
        state.json's autosaved mode wins; an absent or unknown mode falls back
        to particles."""
        st = _presets.load_state(self.state_path)
        cls = mode_by_id(st.get("mode")) or mode_by_id("particles")
        return cls()

    def set_mode(self, mode):
        """Idempotent stop of the current mode, then start the new one. A
        start() exception toasts the human summary and reinstates the previous
        mode; with no previous mode it propagates (boot failure is fatal)."""
        prev = self.mode
        if prev is not None:
            prev.stop()
        try:
            mode.start(self)
        except Exception as e:                       # noqa: BLE001 — §6.4
            if prev is None:
                raise
            self.hud.toasts.flash(f"{mode.title} failed to start", AMBER)
            self.hud.toasts.hint(str(e)[:80])
            prev.start(self)
            self.mode = prev
            return False
        self.mode = mode
        if self.ui is not None:
            # live switch (step 8): rebind the option lists + accent, recompose
            # the panel, seed the mode's own UI attrs, and reload the new
            # mode's looks/bank/setlist
            ui = self.ui
            ui.palettes = list(getattr(mode, "palettes", ui.palettes))
            ui.mattes = list(getattr(mode, "mattes", ui.mattes))
            pal = getattr(getattr(mode, "pf", None), "palette", None)
            ui.palette_idx = (ui.palettes.index(pal)
                              if pal in ui.palettes else 0)
            mk = getattr(mode, "matte_kind", None)
            ui.matte_idx = ui.mattes.index(mk) if mk in ui.mattes else 0
            ui.accent = mode.accent
            ui.panel_title = "lighteater - " + mode.title.upper()
            ui.set_spec(self.compose_spec(mode))
            mode.configure_ui(ui)
            self._reload_presets()
            if ui.preset_idx >= len(ui.presets):
                ui.preset_idx = 0
            self._seed_bank_setlist()
            self._autosave_state()
        return True

    def compose_spec(self, mode):
        """The mode's declared sections + the shell's SIGNAL rack + global rows
        (DESIGN.md §2.4: the rack is a shell-owned section on every panel).

        Suppression rule (DESIGN.md §2.4, judge finding): the rack hides any
        control the active mode claims — a mode declares `claims` (a set of
        store keys, e.g. Dither claims all the dither quality controls
        because it owns dithering as the primary image; two visible dither
        subsystems in one panel is the bolted-features incoherence the
        overhaul exists to kill).

        Duplicate-control rule (DESIGN.md §4.2, same principle, generic): a
        global row whose `attr` the mode's own sections already declare is
        omitted — e.g. Dither's SOURCE has its own Mirror row, so the
        global Mirror would be a second face on the same state. Derived from
        the spec itself (no per-mode claims needed): both faces would set the
        same ui attr, so the attr IS the identity."""
        spec = mode.panel_spec()
        rack = build_signal_section()
        claims = frozenset(getattr(mode, "claims", ()))
        if claims:
            rack.widgets = [w for w in rack.widgets
                            if getattr(w, "store_key", None) not in claims]
        mode_attrs = {getattr(w, "attr", None)
                      for s in spec if isinstance(s, Section)
                      for w in s.widgets}
        mode_attrs.discard(None)
        rows = [r for r in build_global_rows()
                if getattr(r, "attr", None) not in mode_attrs]
        return spec + [rack] + rows

    # ----- mode switching (DESIGN.md §3 / §8 step 8) -----
    def request_mode(self, mode_id):
        """Post a switch — the loop performs it at the top of the next frame
        (boot card, stop → start, toast). Same-mode requests are a hint."""
        if self.mode is not None and mode_id == self.mode.id:
            self.hud.toasts.hint(f"already in {self.mode.title}")
            return
        self.pending_mode = mode_id

    def _menu_commit(self, mode_id, from_boot):
        """One card committed from the menu (key, Enter, Esc-at-boot, click).

        Releases the autopilot. The keyboard release check in `_route_key`
        cannot see this: the open menu consumes every key and returns before
        it, so picking a mode card — which is about as deliberate an act as
        the instrument has — used to leave AUTO running, and it would recall
        a look over the top of your choice a minute later.

        A commit from the BOOT menu that lands on the mode already running is
        not a switch — that mode was started behind the menu so the screen
        would be live, and it is what the selection defaulted to. Routing it
        through `request_mode` scolded the operator with 'already in
        Particles' for pressing Enter on the one card the menu pre-selected.
        It gets the mode-title flash a real entry gets, and the three-doors
        hint is posted HERE rather than at boot, so it lands on the mode
        instead of on top of the menu's own hint line."""
        if self.auto.interrupt():
            self.hud.toasts.hint("auto off - you took over")
        if from_boot:
            # the resume hint is a pointer at Enter, and Enter has just been
            # pressed. Its 4 s ttl outlives that by ~3 s, so without this it
            # stacks above the doors hint and the operator reads two
            # sentences, one of them about a menu they have already left.
            self.hud.toasts.retire(RESUME_HINT)
            self.hud.toasts.hint("m menu - TAB panel - ? keys", ttl=4.0)
            if self.mode is not None and mode_id == self.mode.id:
                self.hud.toasts.flash(self.mode.title, self.mode.accent)
                return
        self.request_mode(mode_id)

    def _switch_mode(self, mode_id):
        """One live mode switch: static boot card (shown + recorded — the
        recorder captures the card, not a gray flash), old.stop() →
        new.start(host) via set_mode (which toasts + reinstates the previous
        mode on failure), then the mode-title center toast in the new mode's
        accent. Overlay state, blackout, recording, and the audio toggle are
        host/UI state and survive untouched."""
        cls = mode_by_id(mode_id)
        if cls is None:
            self.hud.toasts.hint(f"unknown mode {mode_id}")
            return False
        first_entry = mode_id not in self._mode_instances
        # remember the outgoing mode's selected look so re-entry can restore
        # the selection highlight without re-applying anything
        if (self.mode is not None and self.ui is not None
                and self.ui.preset_idx < len(self.ui.presets)):
            self._mode_preset[self.mode.id] = self.ui.preset_name
        new = self._mode_instances.get(mode_id) or cls()
        if self.ps.blackout:
            # DESIGN.md §3: while blackout is armed the switch happens under
            # black — never flash the bright card to screen or recorder; only
            # the amber corner tick stays.
            card = np.zeros((self.res[1], self.res[0], 3), np.uint8)
            draw_corner_tick(card, 2.0 * _u(self.res[1]))
        else:
            card = render_boot_card(self.res, new.title, new.accent)
        if self.writer is not None:
            self.writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        if self.show:
            cv2.imshow(self.WIN, card)
            cv2.waitKey(1)
        if not self.set_mode(new):
            return False                       # toasted + previous reinstated
        self._mode_instances[mode_id] = new
        if first_entry:
            # FIRST entry lands on the mode's known-good look (predictable
            # from 2 m away). RE-entry to an already-visited mode preserves
            # its current settings (amended DESIGN.md §6.2 — presets are an
            # instrument; switching away and back must not reset the look).
            safe = new.safe_look()
            if isinstance(safe, str) and safe in self.ui.presets:
                self.ui.preset_idx = self.ui.presets.index(safe)
                self.ui.pending_preset = safe
        else:
            prev = self._mode_preset.get(mode_id)
            if prev in self.ui.presets:
                self.ui.preset_idx = self.ui.presets.index(prev)
        if self.show:
            self._wire_keys()                  # mode-local commands changed
        self.hud.toasts.flash(new.title, new.accent)
        return True

    def _register_shell_commands(self):
        """Menu + direct mode-switch keys (DESIGN.md §6.2: `m` opens the menu
        from any overlay state; each mode's letter switches directly)."""
        self.reg.add("menu.open", "Menu", "m",
                     lambda: self.menu.toggle(self.mode.id if self.mode else None))
        # unbound (TAB already reaches PANEL and is listed in the help table) —
        # this is the named command the HUD chevron posts, so the mouse path
        # and the key path are one implementation (DESIGN.md principle 7)
        self.reg.add(PANEL_OPEN, "Open the panel", None, self._open_panel)
        for cls in REGISTRY:
            # a mode whose id starts with a taken letter declares its own
            # `key` class attr (Physarum can't have 'p' — Particles owns it)
            self.reg.add(f"mode.{cls.id}", f"Switch to {cls.title}",
                         getattr(cls, "key", cls.id[:1]),
                         lambda mid=cls.id: self.request_mode(mid))

    def _open_panel(self):
        """Show the sidebar — the whole content of the `panel.open` command.

        Setting the overlay state alone was not enough, and the way it failed
        was invisible: `ui.open` is the sidebar's OWN collapsed/expanded flag,
        and a collapsed sidebar draws its expand button as a small box in
        exactly the top-right spot the HUD chevron occupies. So the first
        click swapped one near-identical box for another in the same place and
        no panel appeared — the exact 'my click did nothing' symptom the
        chevron was added to kill. Opening means both: the shell's overlay
        state AND the panel's own."""
        self.overlay = OverlayState.PANEL
        if self.ui is not None:
            self.ui.open = True

    def _wire_keys(self):
        """(Re)build the command registry for the active mode — called at boot
        and after every switch (mode-local commands differ per mode)."""
        self.reg = CommandRegistry()
        _wire_perform_keys(self.reg, self.ui, self.hud, self.ps,
                           lambda name: setattr(self.ui, "pending_preset", name),
                           mode_commands=self.mode.commands(),
                           safe_look=lambda: self.mode.safe_look(),
                           get_overlay=lambda: self.overlay,
                           debug_line=self.debug_line)
        self._register_shell_commands()
        self.help_rows = self.reg.table() + [("TAB", "Cycle overlay"),
                                             ("Esc", "Step toward hidden")]

    def _on_mouse(self, event, x, y, flags, param=None):
        """Window mouse routing: an open help modal swallows every mouse event
        (a click closes it — help already closes on any key); the open menu
        eats clicks (a card commits a switch); otherwise the panel gets the
        event."""
        if self.ps.help_open:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ps.help_open = False
            return
        if self.menu.open:
            if event == cv2.EVENT_LBUTTONDOWN:
                from_boot = self.menu.boot     # click() closes and clears it
                action, value = self.menu.click((x, y))
                if action == "switch":
                    self._menu_commit(value, from_boot)
                elif action == "auto":
                    self._toggle_auto()
            return
        if self.ui is not None:
            if event == cv2.EVENT_LBUTTONDOWN and self.auto.interrupt():
                # The panel is the other half of the instrument's input
                # surface, and none of it goes through _route_key — a slider
                # drag or a preset-row click never touched AUTO_RELEASE_KEYS.
                # dtouch.auto promises ANY scene-changing human input releases
                # it; a click on the panel is exactly that.
                self.hud.toasts.hint("auto off - you took over")
            self.ui.on_mouse(event, x, y, flags, param)

    def _route_key(self, key):
        """One waitKey code through the routing contract: the open menu
        consumes every key (DESIGN.md §3 — unknown keys hint, q closes the
        menu AND arms the quit confirm); then rename-typing consumes every
        key (Esc only cancels the rename — §6.2); then the perform layer."""
        if key == 255:
            return
        if self.menu.open:
            from_boot = self.menu.boot         # a commit closes and clears it
            action, mode_id = self.menu.key(key)
            if action == "switch":
                self._menu_commit(mode_id, from_boot)
            elif action == "quit":
                self.reg.dispatch(ord("q"))     # first press toasts (§6.2)
            elif action == "soon":
                # the reserved card is on screen and dashed: name it, rather
                # than deflect to a key map that cannot explain it either
                self.hud.toasts.hint(f"{mode_id.lower()} - coming soon")
            elif action == "auto":
                self._toggle_auto()
            elif action == "unknown":
                self.hud.toasts.hint("? for keys")
            return
        if self.ui is not None and self.ui.on_key(key):
            return                             # rename typing eats the key
        if key in AUTO_RELEASE_KEYS and self.auto.interrupt():
            self.hud.toasts.hint("auto off - you took over")
        self.overlay = _perform_key(key, self.overlay, self.ps,
                                    self.reg, self.hud)

    def _toggle_auto(self):
        on = self.auto.toggle()
        self.hud.toasts.hint("auto on - it plays itself, any scene key stops it"
                             if on else "auto off")

    def _auto_tick(self, dt):
        """One autopilot step, folded into the ordinary mailboxes.

        Everything it does is something a person could have done from the
        keyboard: post a look, post a mode, dispatch a named command. It gets
        no private reach into the mode, which is what keeps "it is playing"
        and "I am playing" the same code path.
        """
        if not self.auto.on or self.mode is None or self.ui is None:
            return
        looks = list(self.all_presets.keys())
        modes = [m.id for m in REGISTRY]
        current = self.ui.preset_name if self.ui.preset_idx < len(
            self.ui.presets) else None
        for kind, value in self.auto.tick(dt, self.mode.id, looks, modes,
                                          current=current):
            if kind == "preset":
                self.ui.pending_preset = value
            elif kind == "mode":
                self.pending_mode = value
            elif kind == "command":
                cmd = self.reg.get(value)
                if cmd is not None:      # a cast the current mode does not
                    cmd.run()            # have is a hint, not a demand
        if self.auto.last_reason:
            self.hud.toasts.hint(f"auto - {self.auto.last_reason}")
            self.auto.last_reason = ""

    def _draw_waiting_note(self, img):
        """No frame has ever arrived (DESIGN.md §6.4): a plain-language
        on-canvas explanation — never a traceback, never a frozen gray box."""
        h, w = img.shape[:2]
        uu = _u(h)
        ix = int(w * 0.035)                    # title-safe inset (§5)
        put_outlined(img, "waiting for camera...",
                     (ix, h // 2), max(int(1.0 * uu), 10), AMBER)
        put_outlined(img, "check camera permissions in System Settings",
                     (ix, h // 2 + int(1.4 * uu)), max(int(0.75 * uu), 8),
                     AMBER)

    # ----- preset plumbing (per-mode: looks, bank, setlist — DESIGN.md §7) -----
    def _load_presets(self):
        loaded = _presets.load(self.presets_path, mode=self.mode.id,
                               builtin=getattr(self.mode, "BUILTIN", {}))
        for note in _presets.take_notes():
            # store warnings (e.g. corrupt-file backup) surface as toasts —
            # silence on a data-loss event is a bug (DESIGN.md §9)
            self.hud.toasts.hint(note, AMBER)
        return loaded

    def _reload_presets(self):
        self.all_presets = self._load_presets()
        names = list(self.all_presets.keys())
        if self.ui is not None:
            self.ui.presets = names
            self.ui.user_presets = _presets.user_names(self.presets_path,
                                                       mode=self.mode.id)
            # delete/rename keep the stored bank+setlist consistent; mirror that
            stored = _presets.bank(self.presets_path, mode=self.mode.id)
            if stored is not None:
                self.ui.bank = stored
            stored = _presets.setlist(self.presets_path, mode=self.mode.id)
            if stored is not None:
                self.ui.setlist = stored
        return names

    def _seed_bank_setlist(self):
        """Seed the UI's bank + setlist for the active mode. Stored assignments
        win; a mode that never stored a bank gets its built-ins on slots 1..9
        in order (the instrument is playable blind out of the box — DESIGN.md
        principle 6; an explicitly emptied bank stays empty). An empty setlist
        means 'all looks, load order' and tracks saves live."""
        ui = self.ui
        stored = _presets.bank(self.presets_path, mode=self.mode.id)
        if stored is not None:
            ui.bank = stored
        else:
            builtin = list(getattr(self.mode, "BUILTIN", {}))[:BANK_SEED_MAX]
            ui.bank = {str(i + 1): n for i, n in enumerate(builtin)}
        ui.setlist = _presets.setlist(self.presets_path, mode=self.mode.id) or []

    def _autosave_state(self):
        """Autosave: active mode + preset + bank → state.json (DESIGN.md §6.4;
        crash restart resumes the same look). The bank's persistent authority
        is presets.json — the copy here is a crash-recovery snapshot."""
        ui = self.ui
        current = (ui.preset_name
                   if ui is not None and ui.preset_idx < len(ui.presets) else None)
        _presets.save_state({"mode": self.mode.id, "preset": current,
                             "bank": {self.mode.id: dict(ui.bank)} if ui else {}},
                            path=self.state_path)

    # ----- spec-derived HUD status (DESIGN.md §2.3) -----
    def _status_line(self):
        """'MODE TITLE  <status-marked widget values in spec order>  <tail>'
        — e.g. 'DITHER  blue noise  3-bit  bias auto  src still'. The
        body renders from the composed spec's `status` flags (the single
        schema authority); the mode contributes only the cam/src tail."""
        parts = [self.mode.title.upper()]
        ui = self.ui
        for w in ui.iter_widgets():
            st = getattr(w, "status", None)
            if not st:
                continue
            if not visible(ui, w):
                # a gated-off control (panelspec.visible) does not act, so
                # its value does not belong on the status line either —
                # "bias auto" under Floyd-Steinberg was the panel's dead-row
                # lie in a smaller font
                continue
            if isinstance(w, Cycle):
                opts = list(w.options)
                val = opts[int(getattr(ui, w.attr)) % len(opts)]
            else:
                val = getattr(ui, w.attr)
            parts.append(st(val) if callable(st) else st.format(val))
        tail = self.mode.status_tail(self.cam_name)
        if tail:
            parts.append(tail)
        return "  ".join(parts)

    def debug_line(self):
        """The 'i' status variant: fps / frame-time / res (DESIGN.md §6.2)."""
        rw, rh = self.res
        ms = 1000.0 / self.fps if self.fps > 0 else 0.0
        return f"{self.fps:4.1f}fps  {ms:5.1f}ms  {rw}x{rh}"

    def _capture_cfg(self):
        """Spec-derived capture (DESIGN.md §2.1/§7): walk the composed panel
        spec's save=True widgets — the single schema authority. The SIGNAL
        rack's block nests under "signal" (its Section declares store)."""
        return capture_look(self.ui, self.ui.spec)

    def _store_write(self, fn, *a, **kw):
        """Run a preset-store write that touches the disk. A full disk raises
        OSError out of json.dump — from the mailbox pump that would exit run()
        mid-performance. The show never dies for a failed save (DESIGN.md
        §6.4): toast it and carry on.

        Returns (ok, result): `ok` is "no exception", `result` is the store
        function's own verdict — a REFUSED write (poisoned file) raises
        nothing and returns False, so callers must check both before telling
        the operator their look was saved. Any note the store queued (the
        refusal reason) is drained onto the toasts here, because silence on a
        data-loss event is a bug (DESIGN.md §9)."""
        ok, result = True, None
        try:
            result = fn(*a, **kw)
        except OSError as e:                         # noqa: BLE001 — §6.4
            self.hud.toasts.flash("save failed - disk?", AMBER)
            self.hud.toasts.hint(str(e)[:80])
            print("preset store write failed:", e)
            ok = False
        for note in _presets.take_notes():
            self.hud.toasts.hint(note, AMBER)
        return ok, result

    def _apply_look(self, name, cfg):
        """Apply one stored look onto the shared UI state, contained.

        The file's SHAPE is validated by the store; its VALUES are not, and
        they reach `float()` here. A hand-edited (or half-merged) presets.json
        with `"contrast": "high"` is well-shaped and raises ValueError out of
        apply_look — at boot that killed the run before any window existed,
        and live it raised again every frame. apply_look now skips unusable
        values instead of raising, and this says which ones went (silence on a
        look that half-loaded is a bug — DESIGN.md §9); the guard stays for
        anything it cannot foresee. Returns True when the look was applied."""
        try:
            skipped = apply_look(self.ui, self.ui.spec, cfg,
                                 self.mode.DEFAULTS)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:                       # noqa: BLE001 — §6.4
            self.hud.toasts.flash("look not loaded - " + str(name), AMBER)
            self.hud.toasts.hint(f"{type(e).__name__}: {str(e)[:70]}", AMBER)
            print("preset apply failed:", name, e)
            return False
        if skipped:
            self.hud.toasts.hint("unusable values ignored - "
                                 + ", ".join(skipped[:3]), AMBER)
            print("preset", name, "has unusable values:", skipped)
        return True

    def _apply_pending_preset(self):
        """Spec-derived apply onto the shared UI state; engines pick the values
        up in the mode's next step() sync. apply="keep" widgets are untouched
        by look-switching; apply="reset" merges over the mode's defaults."""
        ui = self.ui
        name, ui.pending_preset = ui.pending_preset, None
        # the mailbox is cleared BEFORE the apply on purpose: clearing it after
        # meant a look that raised left the mailbox armed, so the next frame
        # applied the same bad look and raised again — forever, at frame rate,
        # with no way to select a different one
        if name in self.all_presets:
            self._apply_look(name, self.all_presets[name])
            self._autosave_state()

    def _assign_slot(self, name):
        """Slot-badge click (DESIGN.md §6.3): an unbanked look takes the next
        free slot (1-9); clicking an assigned look's badge clears its slot
        (the cheap, reversible correction path — §6.3 names only assignment,
        clearing is the resolved counterpart). Persists to presets.json (the
        bank's authority) and snapshots state.json."""
        ui, toasts = self.ui, self.hud.toasts
        slot = next((s for s, n in ui.bank.items() if n == name), None)
        if slot is not None:
            del ui.bank[slot]
            toasts.hint(f"slot {slot} cleared - {name}")
        else:
            free = next((str(i) for i in range(1, 10) if str(i) not in ui.bank),
                        None)
            if free is None:
                toasts.hint("bank full (1-9)")
                return
            ui.bank[free] = name
            toasts.flash(f"{free} - {name}")
        ok, stored = self._store_write(_presets.set_bank, ui.bank,
                                       path=self.presets_path,
                                       mode=self.mode.id)
        if ok and not stored:
            # the slot works for this session but presets.json is the bank's
            # authority — say it will not survive a restart, don't imply it will
            toasts.hint("bank not saved - it resets on restart", AMBER)
        self._autosave_state()

    def _drain_store_notes(self):
        """Surface any store warning no targeted drain took.

        The store queues its warnings on a module-global list, and the shell
        drains that list in exactly two places: after loading looks, and after
        a store WRITE — where the drained note is reported to the operator as
        the reason THAT write refused. Everything else that touches the store
        queues onto the same list and drains nothing: the boot state.json
        read, the autosave, the bank/setlist reads after a reload. So a note
        could sit in the queue until the next write and then be shown as that
        write's reason — a data-loss message about the wrong file, which §9
        treats as the same bug as silence about the right one. Draining once
        per frame keeps every note attached to the frame it happened on."""
        for note in _presets.take_notes():
            self.hud.toasts.hint(note, AMBER)

    def _pump_preset_mailboxes(self):
        """The pending_* mailboxes a panel click posts to (shipped semantics)."""
        ui = self.ui
        if ui.pending_preset:
            self._apply_pending_preset()
        if ui.pending_slot:
            self._assign_slot(ui.pending_slot)
            ui.pending_slot = None
        if ui.pending_save:
            # amended DESIGN.md §3: auto-name (suffix same-second collisions),
            # save, then immediately open the rename box — naming is one flow
            # — and confirm with a toast, never stdout alone.
            base = "mine_%s" % time.strftime("%H%M%S")
            existing = set(ui.presets) | set(ui.user_presets)
            name, n = base, 2
            while name in existing:
                name, n = "%s_%d" % (base, n), n + 1
            ok, wrote = self._store_write(_presets.save, name,
                                          self._capture_cfg(),
                                          path=self.presets_path,
                                          mode=self.mode.id)
            if ok and wrote:
                names = self._reload_presets()
                if name in names:
                    ui.preset_idx = names.index(name)
                ui.begin_rename(name)
                self.hud.toasts.hint("saved - " + name)
                print("saved preset", name)
            elif ok:
                # the store refused (poisoned file). Saying "saved" here and
                # opening the rename box put the operator in a text field on a
                # preset that does not exist — which then eats every keypress,
                # q and panic included, with nothing on screen to explain it.
                self.hud.toasts.flash("save refused", AMBER)
                print("preset save refused:", name)
            ui.pending_save = False
        if ui.pending_delete:
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            ok, gone = self._store_write(_presets.delete, ui.pending_delete,
                                         path=self.presets_path,
                                         mode=self.mode.id)
            if ok and gone:
                names = self._reload_presets()
                ui.preset_idx = names.index(sel) if sel in names else 0
                self.hud.toasts.hint("deleted - " + ui.pending_delete)
                print("deleted preset", ui.pending_delete)
            elif ok:
                self.hud.toasts.flash("delete refused", AMBER)
                print("preset delete refused:", ui.pending_delete)
            ui.pending_delete = None
        if getattr(ui, "pending_still_path", None):
            # still-image mailbox (v1: filled by the CLI --still flag or a
            # future drop event — there is no file dialog)
            path = ui.pending_still_path
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                self.hud.toasts.hint(f"could not read still: {path[-40:]}")
            else:
                self.still = frame
                self.hud.toasts.hint(f"still loaded - {os.path.basename(path)}")
            ui.pending_still_path = None
        if ui.pending_commands:
            # Action rows with no dedicated mailbox (e.g. the global 'Menu (M)'
            # row, §4.1) route through the command registry — the same named
            # command the 'm' key runs, so both paths stay one implementation.
            for name in ui.pending_commands:
                cmd = self.reg.get(name)
                if cmd is not None:
                    cmd.run()
                else:
                    self.hud.toasts.hint(f"unknown action {name}")
            ui.pending_commands = []
        if ui.pending_rename:
            old, new = ui.pending_rename
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            ok, done = self._store_write(
                _presets.rename, old, new, path=self.presets_path,
                mode=self.mode.id,
                builtin=getattr(self.mode, "BUILTIN", {}))
            if ok and done:
                names = self._reload_presets()
                target = new if sel == old else sel
                ui.preset_idx = names.index(target) if target in names else 0
                self.hud.toasts.hint("renamed - " + new)
                print("renamed preset", old, "->", new)
            elif ok:
                self.hud.toasts.hint("rename refused - name taken or invalid")
                print("rename refused (name taken or invalid):", old, "->", new)
            ui.pending_rename = None
        # anything the targeted drains above did not take belongs to this
        # frame, not to whichever write happens to run next
        self._drain_store_notes()

    # ----- host-owned per-frame sync (mirror / res / mic / recorder) -----
    def _sync_host_state(self):
        ui = self.ui
        self.mirror = ui.mirror
        nw, nh = ui.res_wh
        if (nw, nh) != self.res:
            if self.writer is not None:
                # imageio's ffmpeg writer needs a constant frame size — stop
                # the recording cleanly first (same path as 'r' off), then
                # apply the res change.
                self.writer.close()
                self.writer = None
                ui.record = False
                self.hud.toasts.hint("recording stopped - resolution changed")
                print("saved", self.rec_path)
                self.hud.toasts.hint("saved " + str(self.rec_path))
            self.res = (nw, nh)
            self.mode.on_resize(nw, nh)
            ui.w, ui.h = nw, nh   # AUTOSIZE window refits on next imshow
        if ui.audio and self.mic is None:
            self.mic = LiveMic(); self.mic.start()
        elif not ui.audio and self.mic is not None:
            self.mic.stop(); self.mic = None
        if ui.record and self.writer is None:
            self._start_recording()
        elif not ui.record and self.writer is not None:
            self.writer.close(); print("saved", self.rec_path)
            self.hud.toasts.hint("saved " + self.rec_path)   # filename toast on stop
            self.writer = None

    # ----- recorder (DESIGN.md §6.4: a take that can't start never ends the show) -----
    def _rec_path(self):
        """A recording filename that is not already taken. Names are
        second-resolution, so stopping and restarting a take inside one second
        used to reuse the name and silently truncate the first file — the
        take you just made, gone. Same same-second suffix pattern the preset
        auto-namer uses."""
        base = os.path.join(REC_DIR, "rec_%s" % time.strftime("%Y%m%d_%H%M%S"))
        path, n = base + ".mp4", 2
        while os.path.exists(path):
            path, n = "%s_%d.mp4" % (base, n), n + 1
        return path

    def _start_recording(self):
        """Open the writer for a new take, creating out/ on the way.

        Directory creation is deferred to HERE on purpose: `os.makedirs("out")`
        at boot raises PermissionError on a read-only launch dir (a run from a
        mounted DMG) or FileExistsError if something called `out` is already
        there — a traceback before the window existed, for a directory most
        sessions never use. A launch dir that cannot take recordings now costs
        the recordings, not the show."""
        path = None
        try:
            os.makedirs(REC_DIR, exist_ok=True)
            path = self._rec_path()
            self.writer = imageio.get_writer(path, fps=24, macro_block_size=8)
        except Exception as e:                       # noqa: BLE001 — §6.4
            self.writer = None
            self.ui.record = False                   # disarm; don't retry every frame
            self.hud.toasts.flash("cannot write recordings here", AMBER)
            self.hud.toasts.hint(str(e)[:80], AMBER)
            print("recording could not start:", e)
            return
        self.rec_path = path
        print("recording", path)

    def _recorder_failed(self, e):
        """A write that fails mid-take (disk fills, the volume goes away).
        Stop cleanly, keep whatever landed, and keep the show running — the
        recorder is not the performance (DESIGN.md §6.4)."""
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:                            # noqa: BLE001 — §6.4
            pass
        self.writer = None
        if self.ui is not None:
            self.ui.record = False
        self.hud.toasts.flash("recording stopped - write failed", AMBER)
        self.hud.toasts.hint(str(e)[:80], AMBER)
        print("recorder failed:", e)

    def _frame_error(self, e, internal=False):
        """One caught per-frame exception (DESIGN.md §6.4, §8 step 10: any
        traceback is a release blocker). Toast a human summary in amber and
        keep going.

        Rate-limited over a TIME WINDOW, not against the single last key.
        Remembering one (type, message) was defeated by any two alternating
        errors — A→B→A→B passes an equality check every single time, so the
        most reachable repeating failure (a mode that raises one error and a
        recorder or overlay that raises another) sprayed toasts and stdout
        twice per frame indefinitely, which is the exact burial the limit
        exists to prevent. Now every key seen inside the window is remembered:
        an identical error still re-toasts at most once per ERR_TOAST_S, a
        DIFFERENT error still toasts immediately, and past ERR_TOAST_MAX
        distinct errors in one window the stack stops growing and says where
        the rest went. Errors that vary their text every frame (a coordinate,
        a timestamp) are floods too — the cap catches those as well.

        The window only slides if the entries are allowed to age. Re-stamping
        the key on every arrival — including the suppressed ones — refreshed
        it 30 times a second, so the 5 s prune could never drop it and the
        limit latched forever: a permanently broken show (mode.step raising
        ModuleNotFoundError, one click on `portrait` with no mediapipe) said
        so ONCE, three seconds in, and was then silent for the rest of the
        night behind a black projector and a perfectly normal status line.
        The stamp now happens only on the arrival that is actually reported.

        `internal=True` marks an error raised by our OWN present half. When
        the produce half already failed this frame, that error is a symptom of
        the failure being reported and must never REPLACE it: the operator was
        reading `UnboundLocalError: frame` — our bug — instead of the camera
        error that caused it, forever, with no way to reach the real one."""
        key = (type(e).__name__, str(e)[:80])
        now = self._now()
        # drop keys that aged out, so the window slides instead of latching
        self._err_seen = {k: t for k, t in self._err_seen.items()
                          if now - t < ERR_TOAST_S}
        if key in self._err_seen:
            return
        if len(self._err_seen) < ERR_SEEN_MAX:
            # bounded: a flood of never-repeating text (a frame number, a
            # coordinate) grew this map one entry per frame — 300 entries,
            # rebuilt by the comprehension above on every single call. Past
            # the bound the map stops growing; it is already far past
            # ERR_TOAST_MAX, so the flood branch stays engaged either way.
            self._err_seen[key] = now
        if internal and not self._frame_ok:
            self._err_print("frame error (while reporting one):", e, now)
            return
        if len(self._err_seen) > ERR_TOAST_MAX:
            if not self._err_flood:
                self._err_flood = True
                self.hud.toasts.hint("more errors - see the terminal", AMBER)
            self._err_print("frame error:", e, now)
            return
        self._err_flood = False
        self.hud.toasts.flash("something went wrong - show continues", AMBER)
        self.hud.toasts.hint(f"{type(e).__name__}: {str(e)[:70]}", AMBER)
        self._err_print("frame error:", e, now)

    def _err_print(self, prefix, e, now):
        """One bounded stdout line for a frame error.

        The toast cap points the operator AT the terminal, so the terminal is
        the fuller record — but it was not bounded at all: 300 varying-text
        errors printed 300 lines while the toasts correctly capped at 3, which
        just moves the flood to the place the hint sends you. Same window,
        higher ceiling, and one line saying the rest were dropped."""
        if self._err_print_t0 is None or now - self._err_print_t0 >= ERR_TOAST_S:
            self._err_print_t0, self._err_prints = now, 0
        self._err_prints += 1
        if self._err_prints <= ERR_PRINT_MAX:
            print(prefix, type(e).__name__, e)
        elif self._err_prints == ERR_PRINT_MAX + 1:
            print(f"frame errors: more than {ERR_PRINT_MAX} in "
                  f"{ERR_TOAST_S:g}s - further ones suppressed")

    def _frame_failed(self, streak, err=None):
        """The produce half of one frame failed (raised, or never produced a
        frame at all). Throttle the loop, and headless give up eventually.

        Containment (073df99) turned a fatal produce error into a caught one —
        which is right — but a caught error that repeats every frame with
        nothing to wait on is a busy loop. Shown, that pegged a core at 983
        iterations/sec behind a frozen picture. Headless with no frame budget
        it was worse: no window, no key pump, no output, and only Ctrl-C to
        end it — a state the pre-containment code could not reach, so the fix
        created it.

        Returns the exception to end the run with (raised after the normal
        teardown), or None to keep going."""
        if not self.show and self.max_frames is None \
                and streak >= FRAME_FAIL_LIMIT:
            why = (f"{type(err).__name__}: {err}" if err is not None
                   else "no frame ever arrived")
            return RuntimeError(
                f"frame loop failed {streak} times in a row with no window to "
                f"show it and no frame limit to end it ({why})")
        if not self.show and self.max_frames is not None:
            # A bounded headless render is not a show: there is no window to
            # hold at a sane rate, no key pump to keep responsive, and nobody
            # watching it — and the frame budget already ends it. Idling a
            # frame-time per failure turned a 300-frame render against a
            # broken source into 11.2 s of sleeping for 0.4 s of work. The
            # UNBOUNDED headless case keeps both the throttle and the limit;
            # so does anything with a window.
            return None
        time.sleep(FRAME_FAIL_SLEEP_S)
        return None

    def _draw_panel_chevron(self, img):
        """The one always-clickable affordance in HUD state (DESIGN.md §6.3:
        the mouse cannot be required, but it must not be a dead end either).

        Boot state is HUD (§6.1), which draws no sidebar — and `ui._hot` is
        emptied whenever the panel is hidden, so after boot there was nothing
        on the whole frame a click could reach: an audit fired 576 clicks
        across it and got no response at all. The shipped build always drew
        the sidebar, so the double-click cohort HAD a mouse path to the panel
        and lost it, and the only remaining hint fades after 4 s.

        This is the panel-open handle, the same pixels the collapsed sidebar
        draws (imgui.panel_handle — sized, placed and outlined there, with
        the measurements), and it posts the same named command TAB reaches.

        It deliberately does NOT draw in HIDDEN. That state's whole contract
        is provably clean output — the OBS capture contract — and output is
        sacred; HIDDEN is the state you switch to precisely so nothing of
        ours is in the picture."""
        ui, g = self.ui, self.ui._gui
        h, w = img.shape[:2]
        ui._hot = g.begin(max(1.0, h / BASE_H), ui.mouse, ui.accent)
        r = panel_handle(img, hover=in_rect(panel_handle_rect(w, h), ui.mouse))
        ui._hot.append((r, PANEL_OPEN, None))

    # ----- the present half of a frame -----
    def _compose_frame(self, out, frame, camera_lost=False,
                       camera_black=False):
        """The window image for one frame: the mode's RGB output converted to
        BGR, then menu / panel / HUD / help on top.

        Called AFTER the recorder write in run() — recordings never contain
        HUD, panel, or menu (the shipped invariant, kept; the boot card is the
        one deliberate recorded UI frame, DESIGN.md §3). The BGR conversion
        happens only here, so a hidden window pays nothing.

        `frame` is the CAMERA frame behind the menu, and it is None until one
        has ever arrived — including when the very first read RAISED. The menu
        renders without a backdrop in that case; nothing here may assume a
        frame exists.

        The overlay draw is contained: chrome that fails is still only chrome,
        and the picture underneath is the show (DESIGN.md §6.4)."""
        ui, mode = self.ui, self.mode
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        try:
            if self.menu.open:
                # home menu (DESIGN.md §3): live camera through 1-bit blue
                # noise + 65% scrim + mode cards; the running mode keeps
                # stepping untouched behind it
                ui._hot = []
                self.menu.rects = draw_menu(bgr, frame, self.menu.cards,
                                            self.menu.sel, boot=self.menu.boot)
                # HIDDEN-state HUD = toasts + blackout tick only
                self.hud.draw(bgr, OverlayState.HIDDEN,
                              blackout=self.ps.blackout)
            else:
                if self.panel and self.overlay is OverlayState.PANEL:
                    ui.draw(bgr, {"status": ""})
                elif self.panel and self.overlay is OverlayState.HUD:
                    self._draw_panel_chevron(bgr)
                else:
                    ui._hot = []   # panel hidden: stale hit-rects must not eat clicks
                # the HUD paints after the handle (open or close glyph), so
                # its status line is capped short of the handle's rect —
                # otherwise a portrait frame's line runs across the handle
                status_max_x = None
                if self.panel and self.overlay in (OverlayState.HUD,
                                                   OverlayState.PANEL):
                    bh, bw = bgr.shape[:2]
                    status_max_x = (panel_handle_rect(bw, bh)[0]
                                    - int(0.5 * _u(bh)))
                self.hud.draw(bgr, self.overlay, status=self._status_line(),
                              debug_status=self.debug_line(),
                              recording=(self.writer is not None),
                              blackout=self.ps.blackout,
                              camera_lost=camera_lost,
                              camera_black=camera_black,
                              status_max_x=status_max_x)
            if self.ps.help_open:
                # help carries the active mode's accent (§5 one-accent)
                draw_help(bgr, self.help_rows, accent=mode.accent)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:                       # noqa: BLE001 — §6.4
            self._frame_error(e, internal=True)
        # Only what is on screen may hold the keyboard. Every branch above is
        # a way to take the rename box off it — the menu, a non-PANEL overlay,
        # a collapsed sidebar inside ui.draw, a mode switch that retired the
        # name — and one that draws it is the only one that keeps it. Runs
        # after the draw and before the waitKey that routes the next key, so
        # no key can land in a box the frame just stopped showing.
        ui.expire_offscreen_rename()
        return bgr

    # ----- the loop -----
    def run(self):
        rw, rh = self.res
        if self._boot_mode is None:
            # no explicit --mode: resume the last-used mode (DESIGN.md §3)
            self._boot_mode = self._resolve_boot_mode()
        if self._source is None:
            # never raises: a camera that will not open yields no frames and
            # keeps retrying, so the window still opens on the 'waiting for
            # camera' card and the operator can read it and quit (§6.4)
            self._source = CameraSource(self._device)
        self.cam_name = getattr(self._source, "name", "source")
        cam_error = getattr(self._source, "error", None)
        if cam_error is not None:
            self.hud.toasts.hint(str(cam_error)[:80], AMBER, ttl=4.0)
            print("camera unavailable:", cam_error)
        if self._still_path is not None and self.still is None:
            self.still = cv2.imread(self._still_path, cv2.IMREAD_COLOR)
            if self.still is None:
                print("could not read still:", self._still_path)

        if not self.set_mode(self._boot_mode):
            raise RuntimeError("boot mode failed to start")
        mode = self.mode
        self._mode_instances[mode.id] = mode
        self.all_presets = self._load_presets()
        preset = self._boot_preset
        if preset is None:
            # crash-restart resume (DESIGN.md §6.4): state.json is the single
            # last-mode/last-preset authority (see dtouch.presets docstring)
            st = _presets.load_state(self.state_path)
            if st.get("mode") in (None, mode.id):
                preset = st.get("preset")
        if preset is None:
            preset = mode.safe_look()
        if preset not in self.all_presets:
            # A boot preset this mode does not own. The Host's own default is a
            # Particles look, and --preset resolution can hand over a name that
            # belongs to the other mode, so this is reachable. It used to fall
            # straight through the apply below and leave the look UNAPPLIED:
            # the panel and the HUD showed a name, and the parameters were
            # whatever _UI_DEFAULTS happened to say. A UI that names a look it
            # did not load is lying, and silence on a half-loaded look is a bug
            # (DESIGN.md §9).
            #
            # It stayed invisible because Dither's first built-in used to be
            # `classic`, whose values are identical to its _UI_DEFAULTS — so
            # the no-op and the correct result looked the same. Changing the
            # first look is what made it show.
            preset = mode.safe_look()

        # The shared UI-state object ALWAYS exists — it is the mode's parameter
        # surface (spec capture/apply target + step()'s per-frame sync source).
        # `show`/`panel` only govern whether it is drawn and clickable.
        palettes = list(getattr(mode, "palettes", [])) or [""]
        mattes = list(getattr(mode, "mattes", [])) or [""]
        boot_palette = (getattr(getattr(mode, "pf", None), "palette", None)
                        or palettes[0])
        ui = self.ui = OverlayUI(rw, rh, list(self.all_presets.keys()),
                                 palettes, mattes,
                                 preset=preset,
                                 matte=getattr(mode, "matte_kind", mattes[0]),
                                 palette=boot_palette)
        ui.accent = mode.accent
        ui.panel_title = "lighteater - " + mode.title.upper()
        ui.set_spec(self.compose_spec(mode))
        ui.mirror = self.mirror
        ui.audio = self._boot_audio
        ui.video_bg = getattr(mode, "boot_video_bg", False)
        ui.video_mix = getattr(mode, "boot_video_mix", ui.video_mix)
        if preset in self.all_presets:
            # the look loaded at startup, same as a live switch (spec-derived)
            # — and contained the same way: a stored look with an unusable
            # value used to end the boot with a traceback and no window, which
            # is the one failure a performer cannot work around (§6.4)
            self._apply_look(preset, self.all_presets[preset])
        mode.configure_ui(ui)
        ui.user_presets = _presets.user_names(self.presets_path, mode=mode.id)
        self._seed_bank_setlist()
        self._autosave_state()

        if self.show:
            # AUTOSIZE: the window is fixed at the render resolution so the OS
            # can't maximize/scale it — that scaling was tanking fps (display
            # upscaling) and breaking click mapping. To go bigger, switch the
            # 'output' resolution; the window resizes to match natively.
            cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
            if self.panel:
                cv2.setMouseCallback(self.WIN, self._on_mouse)
            # Key routing goes through the command registry (DESIGN.md
            # principle 7) — the full perform layer, panel shown or not.
            self._wire_keys()
        else:
            _register_quit(self.reg, self.ps, self.hud.toasts)

        # Boot HUD hint (DESIGN.md §3): a 4 s fading pointer. ASCII only
        # (Hershey). Which pointer depends on where the launch landed — the
        # menu already names its own three keys along the bottom, so posting
        # the doors hint there would say the wrong thing in the wrong place
        # (and, before issue #18, on the same baseline). In the menu the
        # useful sentence is what Enter does; the doors hint is posted when
        # the menu is left (_menu_commit).
        if self._boot_menu:
            self.menu.show(self.mode.id, boot=True)
            self.hud.toasts.hint("%s %s" % (RESUME_HINT,
                                            self.mode.title.upper()), ttl=4.0)
        else:
            self.hud.toasts.hint("m menu - TAB panel - ? keys", ttl=4.0)

        # out/ is created by the first record, not here: makedirs on a
        # read-only launch dir used to end the boot with a traceback (§6.4).

        t0 = time.time(); fps = 0.0; count = 0
        presented = 0        # frames actually RENDERED since the last fps tick
        black_streak = 0
        last_frame = None    # last good camera frame, held across read failures
        camera_lost = False
        last_t = time.monotonic()
        out = None
        last_out = None      # last good PICTURE, held when a frame fails
        # bound BEFORE the loop: the present half passes `frame` to the menu
        # backdrop, and a first read that RAISES never binds it inside the
        # contained produce half. That made the present half raise
        # UnboundLocalError, which was caught — and then stood in the operator's
        # error line in place of the real cause, forever.
        frame = None
        fail_streak = 0      # consecutive frames the produce half did not make
        fatal = None         # ends the run AFTER teardown (headless only)
        try:
            while True:
                # DESIGN.md §6.4 / §8 step 10 (any traceback = release
                # blocker). Every line in the PRODUCE half below is reachable
                # mid-show — source.read, cv2.flip, mode.step, the recorder
                # write — and a single exception used to end the performance
                # with a stack trace. The most reachable path: a start.command
                # install with no mediapipe, one click on the shipped
                # `portrait` or `sigil` template, ModuleNotFoundError straight
                # out of mode.step().
                #
                # The produce half is contained here; the PRESENT half (draw,
                # imshow, waitKey, key routing) runs whether or not it
                # succeeded, so even a failure that repeats every single frame
                # still leaves a window that explains itself and answers `q`.
                try:
                    if self.pending_mode is not None:
                        mode_id, self.pending_mode = self.pending_mode, None
                        self._switch_mode(mode_id)
                        mode = self.mode

                    ok, frame = self._source.read()
                    # Still input (DESIGN.md §2.2: modes never special-case
                    # stills — the shell substitutes the loaded still for the
                    # camera when an accepts_still mode's input cycle selects it)
                    use_still = (getattr(mode, "accepts_still", False)
                                 and getattr(ui, "input_idx", 0) == 1)
                    if use_still and self.still is None:
                        ui.input_idx = 0     # snap back; v1 has no file dialog
                        self.hud.toasts.hint(
                            "no still loaded - launch with --still PATH")
                        use_still = False
                    if use_still:
                        frame, camera_lost = self.still, False
                    elif not ok:
                        # Camera loss: hold the last good frame and say so on the
                        # HUD (DESIGN.md §6.4); recovery is automatic when reads
                        # resume. Modes never see None (DESIGN.md §2.2).
                        if last_frame is not None:
                            frame, camera_lost = last_frame, True
                        else:
                            # No frame has EVER arrived (§6.4) — including the
                            # camera that never opened at all: never a frozen,
                            # unquittable window. Show an intentional black frame
                            # with the human fix and keep pumping keys through the
                            # normal path so q/quit works.
                            if self.show:
                                waiting = np.zeros((self.res[1], self.res[0], 3),
                                                   np.uint8)
                                self._draw_waiting_note(waiting)
                                self.hud.draw(waiting, self.overlay,
                                              blackout=self.ps.blackout)
                                cv2.imshow(self.WIN, waiting)
                                self._route_key(cv2.waitKey(1) & 0xFF)
                                if self.ps.quit or ui.quit:
                                    break
                                if cv2.getWindowProperty(
                                        self.WIN, cv2.WND_PROP_VISIBLE) < 0:
                                    break
                            if self.max_frames is not None:
                                break
                            # waiting is not spinning: throttle, and headless
                            # (no card, no key pump, no output at all) give up
                            # eventually instead of turning a missing camera
                            # into a silent busy loop
                            fail_streak += 1
                            last_t = time.monotonic()   # not a stall to integrate
                            presented, t0, fps = 0, time.time(), 0.0
                            self.fps = 0.0              # nothing is rendering
                            fatal = self._frame_failed(fail_streak)
                            if fatal is not None:
                                break
                            continue
                    else:
                        last_frame, camera_lost = frame, False
                        # a camera that appeared late (CameraSource retries the
                        # open) names itself only once it starts yielding
                        self.cam_name = getattr(self._source, "name", self.cam_name)
                    # perf: subsampled mean — a full-frame mean cost ~9 ms at 4K
                    # for a black-streak heuristic that only needs a coarse level
                    black_streak = (0 if use_still else
                                    black_streak + 1
                                    if float(frame[::16, ::16].mean()) < 3.0 else 0)

                    self._pump_preset_mailboxes()
                    self._sync_host_state()

                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    now_t = time.monotonic()
                    # Clamped (DESIGN.md §2.2). `last_t` only advanced here, so
                    # anything that failed EARLIER in the produce half kept
                    # accumulating: a measured 7-frame stall handed mode.step a
                    # single 0.409 s dt, and a long one would hand a mode a
                    # multi-second integration step — particles teleport, fades
                    # blow out, and the recovery frame looks worse than the
                    # stall. The failure paths reset the clock too, so this cap
                    # is the backstop, not the mechanism.
                    dt, last_t = min(now_t - last_t, DT_MAX), now_t
                    self._auto_tick(dt)
                    levels = (self.mic.levels()
                              if self.mic is not None and self.mic.available
                              else None)
                    out = mode.step(frame, levels, dt)

                    # SIGNAL rack post-FX (DESIGN.md §2.4). Applied here on
                    # purpose: after the mode's render + video composite (so it
                    # bends the whole picture), before the recorder (so captures
                    # match what you see) and before ui.draw (so the panel never
                    # gets glitched into unreadability). CircuitBent is documented
                    # for BGR; `out` is RGB, which only swaps which channel drifts
                    # left vs right — the offsets are independent symmetric draws,
                    # so the look is identical. Constructed lazily so a session
                    # that never enables it pays nothing.
                    if ui.glitch:
                        if self.cb is None:
                            self.cb = CircuitBent(seed=self.seed)
                        cb = self.cb
                        # one source of truth for panel -> rack config,
                        # including the dither suppression rule and the
                        # per-mode dither working rows (overlay_ui.sync_signal)
                        sync_signal(cb, ui, mode, self.res[1])
                        # A GL mode may have already applied the rack as
                        # fragment passes inside step() (physarum's ported
                        # path, dtouch.rack_gl) — it says so per frame via
                        # signal_done, and `out` is then the racked frame.
                        if not getattr(mode, "signal_done", False):
                            out = cb.process(out)
                    if self.ps.blackout:
                        # Hard black AFTER mode render/composite/glitch, BEFORE
                        # the recorder — blackout is part of the show and IS
                        # recorded; the UI/HUD still draw on top per overlay state
                        # (DESIGN.md §6.2).
                        out[:] = 0
                    if self.writer is not None:
                        try:
                            # the writer takes RGB `out` directly
                            self.writer.append_data(out)
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as e:       # noqa: BLE001 — §6.4
                            # a failing recorder stops recording; it does not
                            # stop the show, and it never takes the picture
                            # down with it
                            self._recorder_failed(e)
                except (KeyboardInterrupt, SystemExit):
                    raise                            # Ctrl-C still means stop
                except Exception as e:               # noqa: BLE001 — §6.4
                    # Hold the last good PICTURE. A broken frame freezes the
                    # image and says so; it never blanks the projector and it
                    # never drops the operator back to a shell prompt.
                    self._frame_ok = False
                    self._frame_error(e)
                    out = last_out
                    if out is None:
                        out = np.zeros((self.res[1], self.res[0], 3), np.uint8)
                    elif self.ps.blackout:
                        # Blackout is applied inside the try, so the held
                        # picture used to go out FULLY LIVE while the amber
                        # tick and the BLACKOUT flash both said the output was
                        # dead (measured: blackout=True, nonblack_px=51607).
                        # An armed indicator that lies about the projector is
                        # worse than no indicator, and blackout is the one key
                        # a performer hits when something is wrong on screen.
                        # A FRESH black frame, never `out[:] = 0`: `out` IS
                        # `last_out` here, and blacking it in place would
                        # destroy the held picture for good — the output would
                        # stay black after blackout was disarmed.
                        out = np.zeros_like(out)
                    # containment without a throttle is a busy loop: a produce
                    # half that fails every frame has nothing to wait on, and
                    # this used to spin at 983 iterations/sec behind a frozen
                    # picture (headless, forever and silently)
                    last_t = time.monotonic()        # the stall is not the mode's
                    fail_streak += 1
                    fatal = self._frame_failed(fail_streak, e)
                    if fatal is not None:
                        break
                else:
                    last_out = out
                    self._frame_ok = True
                    fail_streak = 0

                count += 1
                # fps counts PRESENTED frames (DESIGN.md §6.2). Counting loop
                # iterations meant a failing loop reported its own spin rate:
                # the HUD and the `i` readout showed 953.5 fps while the
                # picture was frozen and nothing was being rendered at all.
                # The window is time-based as well as frame-based so a stall
                # decays toward 0 instead of holding the last good number.
                if self._frame_ok:
                    presented += 1
                now = time.time()
                if presented >= 10 or now - t0 >= 1.0:
                    fps = presented / max(now - t0, 1e-6)
                    presented, t0 = 0, now
                self.fps = fps

                if self.show:
                    try:
                        if frame is None:
                            # nothing has EVER arrived — including a first
                            # read that RAISED. The §6.4 waiting card is the
                            # answer, not a black rectangle explaining nothing.
                            bgr = np.zeros((self.res[1], self.res[0], 3),
                                           np.uint8)
                            self._draw_waiting_note(bgr)
                            self.hud.draw(bgr, self.overlay,
                                          blackout=self.ps.blackout)
                        else:
                            bgr = self._compose_frame(
                                out, frame, camera_lost=camera_lost,
                                camera_black=(black_streak > 15))
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:           # noqa: BLE001 — §6.4
                        # internal: our own present half. It must never stand
                        # in the error line in place of the produce failure it
                        # was trying to draw.
                        self._frame_error(e, internal=True)
                        bgr = np.zeros((self.res[1], self.res[0], 3), np.uint8)
                        self.hud.toasts.draw(bgr)    # keep the explanation visible
                    cv2.imshow(self.WIN, bgr)
                    key = cv2.waitKey(1) & 0xFF   # pump GUI + mouse
                    # menu → rename box → perform layer (see _route_key)
                    self._route_key(key)
                    if self.ps.quit or ui.quit:
                        break
                    # quit only when the window is actually destroyed (red X) ->
                    # property is -1. A minimized window reports 0, so this does
                    # NOT quit on minimize.
                    if cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) < 0:
                        break
                if self.max_frames is not None and count >= self.max_frames:
                    break
        finally:
            if self.writer is not None:
                self.writer.close(); print("saved", self.rec_path)
                self.writer = None
            if self.mic is not None:
                self.mic.stop(); self.mic = None
            if hasattr(self._source, "release"):
                self._source.release()
            if self.mode is not None:
                self.mode.stop()
            if self.show:
                cv2.destroyAllWindows(); cv2.waitKey(1)
        if fatal is not None:
            # raised AFTER teardown, so the camera, mic and any take are
            # released first — and only ever headless, where there is no
            # window to carry the message and no `q` to end the run
            raise fatal
        return count, out
