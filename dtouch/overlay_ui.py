"""In-frame control panel — a generic walker over a panel spec, one window.

The widget primitives (rows, sliders, section headers, cycles, rename box,
manage buttons, tooltip, scrollbar) live in dtouch.imgui; the controls
themselves are declared as data in a panel spec (dtouch.panelspec, DESIGN.md
§2.3). OverlayUI walks the spec and draws it with the toolkit — today's spec
is `build_particles_spec`, reproducing the shipped Particles panel exactly
(pinned by tests/test_goldens.py); a future `Mode.panel_spec()` returns pieces
of the same thing.

Immediate-mode GUI drawn with cv2 primitives (Tk won't paint when launched
headless on macOS; cv2 windows do). Everything is a drawn, boxed,
hover-highlighting control with click feedback, and aligned hit-testing via
setMouseCallback. Because it's just pixels, it can be rendered headless and
inspected.

Every pixel dimension is authored at a 1080p baseline and multiplied by a scale
factor derived from the output resolution (self._s, set each draw from the
frame height). The factor floors at 1.0 — so 720p/1080p render exactly as
authored — and grows to 2x at 4K, keeping the panel the same fraction of the
frame instead of shrinking to a tiny fixed pixel box (issue #3).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import imgui
from .imgui import (PANEL, BTN, HOVER, INK, DIM, ACC, TRACK, HANDLE, RED, DARK,
                    in_rect as _in)
from .panelspec import (Slider, Toggle, Cycle, Action, Param, Readout,
                        PresetList, Section, display_fmt, nudge_to, quantise,
                        visible)

BASE_H = 1080   # resolution the layout literals are authored against

# Post-FX dither modes, in cycle order. "off" is a deliberate third choice rather than a
# disabled state — the glitch chain is still worth having without the dither.
DITHERS = ["bayer", "blue", "fs", "riemersma", "off"]

# SIGNAL rack dither-quality bias options (DESIGN.md §4.1): the ordered
# dithers' rounding direction — auto flips on dark frames, light/dark force it.
SIGNAL_BIASES = ["auto", "light", "dark"]
SIGNAL_BIAS_INVERT = {"auto": "auto", "light": False, "dark": True}


# ----- SIGNAL visibility gates (panelspec.visible; the magic-over-control
# ----- decision, 2026-08-24). The rack only runs while Glitch is ON
# ----- (dtouch/shell.py gates CircuitBent on ui.glitch; the physarum GPU
# ----- rack gates the same way), so every dependent row measured 0.000
# ----- rendered difference with Glitch off — a panel of working-looking
# ----- controls bending nothing. They now appear only where they act.

def _sig_dither_name(s):
    return DITHERS[int(getattr(s, "dither_idx", 0)) % len(DITHERS)]


def _sig_on(s):
    return bool(getattr(s, "glitch", False))


def _sig_dither_on(s):
    """Dither-quality rows: the rack must be on AND its dither not 'off'."""
    return _sig_on(s) and _sig_dither_name(s) != "off"


def _sig_ordered_on(s):
    """Bias only steers the ordered dithers — error diffusion self-corrects
    and ignores it (dtouch.dither), so the row hides under fs/riemersma."""
    return _sig_on(s) and _sig_dither_name(s) in ("bayer", "blue")


def sync_signal(cb, ui, mode, out_h):
    """Copy the SIGNAL section's live values onto a CircuitBent — the one
    source of truth for the rack's per-frame configuration. The shell's CPU
    rack and PhysarumMode's GPU rack both call it, so the two backends can
    never drift apart on what the panel means.

    Includes the suppression rule (DESIGN.md §2.4: a mode that claims
    "dither" owns dithering — the rack runs minus its dither) and the
    per-mode dither working resolution (`signal_dither_rows(out_h)`;
    default 72 rows, the lo-fi block look)."""
    cb.chroma_shift = ui.chroma
    cb.scan_drift = ui.drift
    cb.bit_crush = int(ui.crush)
    cb.scanlines = ui.scanlines
    # dither-quality controls (DESIGN.md §4.1: Bits int-snapped 1-4,
    # Gamma default ON, Bias auto/light/dark)
    cb.dither_bits = int(np.clip(round(ui.sig_bits), 1, 4))
    cb.dither_gamma = bool(ui.sig_gamma)
    cb.dither_invert = SIGNAL_BIAS_INVERT[
        SIGNAL_BIASES[int(ui.sig_bias_idx) % len(SIGNAL_BIASES)]]
    claimed = frozenset(getattr(mode, "claims", ()))
    cb.dither_mode = (None if "dither" in claimed or ui.dither_name == "off"
                      else ui.dither_name)
    # Pixel (the dither's block size, in output pixels). 0 = auto: the mode's
    # own working resolution (`signal_dither_rows(out_h)`; default 72 rows,
    # the lo-fi block look). A set value converts to working rows so both
    # backends (CircuitBent._apply_dither and SignalRackGL's downsample
    # stage) read the SAME dither_size — px 1 lands on out_h rows, which
    # both treat as full-res dithering.
    px = int(round(float(getattr(ui, "sig_px", 0.0))))
    if px > 0:
        cb.dither_size = max(1, round(out_h / px))
    else:
        rows_fn = getattr(mode, "signal_dither_rows", None)
        cb.dither_size = rows_fn(out_h) if callable(rows_fn) else 72


RES_OPTIONS = [("720p", 1280, 720), ("1080p", 1920, 1080),
               ("1440p", 2560, 1440), ("4K", 3840, 2160)]

_RANGES = {
    "fade": (0.50, 0.985), "exposure": (0.3, 4.0), "spark": (0.0, 1.0),
    "curl": (0.0, 1.0), "dot": (0.004, 0.020), "sens": (0.0, 3.0),
    # reseed's ceiling is 0.16, not 0.15, because the shipped `portrait` look is
    # authored at 0.16. A built-in is the authority on its own value: when
    # apply_look started clamping, a 0.15 ceiling silently retuned a look that
    # had rendered at 0.16 since before the panel had ranges at all.
    "count": (0.1, 1.0), "damp": (0.82, 0.985), "pull": (8.0, 40.0), "reseed": (0.005, 0.16),
    "video_mix": (0.05, 1.0),
    # MOTION (boids gains) and SIGNAL (circuit-bent) — see dtouch.flock / dtouch.circuit_bent
    "cohere": (0.0, 1.0), "align": (0.0, 1.0), "separate": (0.0, 1.0),
    "chroma": (0.0, 60.0), "drift": (0.0, 40.0), "crush": (0.0, 8.0),
}

# Sliders whose engine cannot use a fraction, and how a raw value lands on the
# whole number it will actually be used as (panelspec.Slider.step/snap). The
# rest of the rack is genuinely continuous and stays so.
#
# Crush TRUNCATES rather than rounds because the shell consumes it as
# `int(ui.crush)` — see dtouch/shell.py's SIGNAL sync. Matching that here is
# what makes the change invisible to already-saved looks: a stored Crush of
# 2.92 has always rendered at bit depth 2, it just used to *read* "3".
_QUANTISED = {
    "sig_bits": (1.0, "round"),   # int(clip(round(v), 1, 4)) — 4 real settings
    "sig_px":   (1.0, "round"),   # int(round(v)) whole pixels — 0 = auto
    "crush":    (1.0, "floor"),   # int(v) — whole output bits, 0 = off
}

_SLIDERS = [
    ("Trails", "fade", "How long particle motion trails linger before fading."),
    ("Glow", "exposure", "Overall brightness and bloom of the particles."),
    ("Spark", "spark", "How much fast motion scatters particles into bursts of sparks."),
    ("Flow", "curl", "Swirling, turbulent drift of the particles."),
    ("Size", "dot", "Size of each individual particle dot."),
    ("Count", "count", "How many particles (density of the cloud)."),
    ("Glide", "damp", "How long particles keep gliding. High = they overshoot into sharp "
                      "contour lines (the sigil look); low = they stop quickly."),
    ("Pull", "pull", "How sharply particles are pulled to the shape. Low = tighter, brighter "
                     "edge bands."),
    ("Reseed", "reseed", "How fast particles respawn. Low = they persist and trace flowing "
                         "lines; high = a constantly refreshing spray."),
]


# UI attr -> serialized preset key, where they differ (the v1 names — so a
# migrated look and a fresh capture are byte-compatible; DESIGN.md §7).
_SAVE_KEYS = {"curl": "curl_amp", "dot": "base_size", "reseed": "reseed_frac",
              "pull": "pull_falloff"}


def _slider_spec(label, attr, tip, **kw):
    lo, hi = _RANGES[attr]
    kw.setdefault("save_key", _SAVE_KEYS.get(attr))
    if attr in _QUANTISED:
        step, snap = _QUANTISED[attr]
        kw.setdefault("step", step)
        kw.setdefault("snap", snap)
        kw.setdefault("fmt", ".0f")
    return Slider(label, attr, lo, hi, tip=tip, **kw)


def build_particles_sections(palettes, mattes):
    """The Particles mode's own panel sections (DESIGN.md §4.1) — what
    ParticlesMode.panel_spec() returns. The shell appends the SIGNAL rack and
    the global rows below these (DESIGN.md §2.4)."""
    look_sliders = [_slider_spec(label, attr, tip) for label, attr, tip in _SLIDERS]
    return [
        Section("TEMPLATES", [PresetList()]),
        Section("SOURCE", [
            Cycle("matte", "matte_idx", list(mattes), save_key="matte",
                  status="matte {}"),
            Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                  key="res", save=False, nudge=False),
            Toggle("Video bg", "video_bg"),
            _slider_spec("Vid mix", "video_mix",
                         "How visible the raw camera footage is behind the particles.",
                         apply="keep",
                         show_when=lambda s: bool(getattr(s, "video_bg", False))),
        ]),
        Section("LOOK", [
            Cycle("color", "palette_idx", list(palettes), gap=4,
                  save_key="palette", status="{}"),
        ] + look_sliders + [
            # captured-but-undrawn: saved since v1, no panel control yet
            Param("attract_speed"),
        ]),
        # MOTION — boids steering (dtouch.flock). The section header + Flock
        # toggle stay visible while off (it still reads as a thing you can
        # turn on); the gain sliders appear only while Flock is ON, because
        # off means every gain is forced to 0 and the sliders steer nothing
        # (magic-over-control, 2026-08-24: a control that does nothing
        # perceptible in the current state is a bug).
        Section("MOTION", key_hint="F", widgets=[
            Toggle("Flock", "flock"),
            _slider_spec("Cohere", "cohere",
                         "Steer toward the local centre. Pulls the cloud into shoals.",
                         show_when=lambda s: bool(getattr(s, "flock", False))),
            _slider_spec("Align", "align",
                         "Match neighbours' direction. This is what makes it move as one.",
                         show_when=lambda s: bool(getattr(s, "flock", False))),
            _slider_spec("Separate", "separate",
                         "Push apart when crowded. Stops the shoal collapsing to a dot.",
                         show_when=lambda s: bool(getattr(s, "flock", False))),
        ], open=False),
    ]


def build_signal_section():
    """The shell's SIGNAL rack (DESIGN.md §2.4) — circuit-bent post-processing
    (dtouch.circuit_bent) applied to every mode's output after render + video
    composite, before recorder and panel (so the panel stays readable). Its
    state serializes under "signal" inside each mode's looks."""
    return Section("SIGNAL", key_hint="G", widgets=[
            Toggle("Glitch", "glitch"),
            # Everything below only bends the picture while Glitch is ON
            # (the shell runs CircuitBent behind ui.glitch), so the rows
            # gate their visibility on it (panelspec.visible): the section
            # opens as one honest toggle and unfolds when it starts acting.
            Cycle("dither", "dither_idx", list(DITHERS), save_key="dither",
                  show_when=_sig_on,
                  tip="Repaints the picture from small two-tone dots, like "
                      "newsprint or an old handheld console. Off = clean."),
            # dither-quality controls (DESIGN.md §2.4/§4.1: the audit's
            # quality controls grow the rack; a mode that claims them —
            # Dither owns ALL dither quality — hides them). They also gate
            # on the rack's dither being on at all: with dither 'off' they
            # move nothing, so they show nothing.
            Slider("Pixel", "sig_px", 0.0, 32.0, fmt=".0f", save_key="pixel",
                   step=_QUANTISED["sig_px"][0],
                   snap=_QUANTISED["sig_px"][1],
                   show_when=_sig_dither_on,
                   tip="Size of the dither's blocks, in pixels. 0 = auto: "
                       "each mode picks its own cell. 1 = full detail; "
                       "higher = chunkier."),
            Slider("Bits", "sig_bits", 1.0, 4.0, fmt=".0f", save_key="bits",
                   step=_QUANTISED["sig_bits"][0],
                   snap=_QUANTISED["sig_bits"][1],
                   show_when=_sig_dither_on,
                   tip="Dither bit depth. 1 = pure two-tone; higher keeps "
                       "more shades. Whole numbers only - there are four "
                       "settings."),
            Toggle("Gamma", "sig_gamma", save_key="gamma",
                   show_when=_sig_dither_on,
                   tip="Dither in linear light so mid-tones keep their "
                       "perceived brightness. Off = the crushed retro look."),
            Cycle("bias", "sig_bias_idx", list(SIGNAL_BIASES), save_key="bias",
                  show_when=_sig_ordered_on,
                  tip="Which way the dots lean on a mostly-dark or mostly-"
                      "bright picture. Auto decides per frame."),
            # Chroma and Drift look like pixel counts and are not: each is the
            # AMPLITUDE of a per-frame random draw that then lands on a whole
            # pixel. The fraction is used — 12.4 and 12.6 are different bleeds
            # — so the slider stays continuous and the tooltip says where the
            # whole numbers come in, rather than the panel pretending the
            # control is coarser than it is (DESIGN.md §4.1: the human
            # explanation lives in the `i` tooltip).
            _slider_spec("Chroma", "chroma",
                         "Colour bleed: red and blue drift apart, slowly. "
                         "This is how far they can go; each frame lands on a "
                         "whole pixel.",
                         show_when=_sig_on),
            _slider_spec("Drift", "drift",
                         "Scan-line sync loss. Rows slip sideways; sometimes a whole band tears. "
                         "This is the most any row can slip, in whole pixels.",
                         show_when=_sig_on),
            _slider_spec("Crush", "crush",
                         "Hard bit-depth reduction, in whole bits. 0 = off.",
                         show_when=_sig_on),
            Toggle("Scanlines", "scanlines", on_text="on", show_when=_sig_on),
        ], open=False, gap=6, store="signal")


def build_global_rows():
    """The shell's global rows (below every mode's sections) — the §4.1
    sketch: Sound react (A) - Sens - Record (R) - Mirror - Menu (M) - Quit.
    Key hints live in the labels; the Menu row is an Action posting the
    `menu.open` command through the walker's pending_commands mailbox."""
    return [
        Toggle("Sound react (A)", "audio"),
        # Sens scales the audio modulation, which is zero while Sound react
        # is off — the row only appears while the mic is actually driving
        # the picture (panelspec.visible).
        _slider_spec("Sens", "sens", "How strongly sound drives the visuals.",
                     apply="keep", gap=4,
                     show_when=lambda s: bool(getattr(s, "audio", False))),
        Toggle("Record (R)", "record", save=False,
               label_fn=lambda v: "Stop recording (R)" if v else "Record (R)"),
        Toggle("Mirror", "mirror", on_text="on", save=False, gap=4),
        Action("Menu (M)", "menu.open"),
        Action("Quit", "quit"),
    ]


def build_particles_spec(presets, palettes, mattes):
    """Today's Particles panel as data — the exact shipped section order,
    names, labels, and vertical rhythm (goldens hold). Composed the same way
    the shell composes any mode's panel: mode sections + SIGNAL rack + global
    rows (DESIGN.md §2.4). `presets` is unused (PresetList reads the walker's
    live list) but kept for signature stability."""
    return (build_particles_sections(palettes, mattes)
            + [build_signal_section()] + build_global_rows())


class OverlayUI:
    def __init__(self, w, h, presets, palettes, mattes,
                 preset="abstract", matte="auto", palette="ice"):
        self.w, self.h = w, h
        self.presets, self.palettes, self.mattes = presets, palettes, mattes
        self.preset_idx = presets.index(preset) if preset in presets else 0
        self.matte_idx = mattes.index(matte) if matte in mattes else 0
        self.palette_idx = palettes.index(palette) if palette in palettes else 0
        self.fade, self.exposure, self.spark, self.curl, self.dot = 0.90, 1.4, 0.35, 0.5, 0.011
        self.damp, self.pull, self.reseed = 0.90, 22.0, 0.06   # motion physics (sigil knobs)
        self.count = 0.75   # fraction of the allocated particles to render
        self.mirror, self.audio, self.sens, self.record = True, False, 1.0, False
        self.video_bg, self.video_mix = False, 0.5   # raw footage behind the particles
        # MOTION — boids steering on the particle cloud (dtouch.flock). Off by default:
        # every gain at 0 short-circuits the solver, so the existing look is untouched.
        self.flock = False
        self.cohere, self.align, self.separate = 0.35, 0.45, 0.30
        # SIGNAL — circuit-bent post-processing on the rendered frame (dtouch.circuit_bent).
        # `dither_idx` indexes DITHERS; "off" is a real choice, not a disabled state.
        self.glitch = False
        self.chroma, self.drift, self.crush = 10.0, 8.0, 0.0
        self.dither_idx = 0
        # dither-quality controls (DESIGN.md §4.1): defaults match CircuitBent's
        # shipped behaviour (3-bit, gamma-correct, auto bias)
        self.sig_bits, self.sig_gamma, self.sig_bias_idx = 3.0, True, 0
        # dither block size in output pixels; 0 = auto (the mode's own
        # working resolution via signal_dither_rows — sync_signal)
        self.sig_px = 0.0
        self.scanlines = True
        self.attract_speed = 4.5   # captured Param — no panel control yet
        self.res_options = list(RES_OPTIONS)
        self.res_idx = next((i for i, (_, rw, rh) in enumerate(self.res_options)
                             if (rw, rh) == (w, h)), None)
        if self.res_idx is None:
            # non-standard boot res (tests, custom rigs): a real option, so the
            # shell's res sync doesn't silently "correct" the output to 1080p
            self.res_options.insert(0, (f"{w}x{h}", w, h))
            self.res_idx = 0
        self.open = True
        self.quit = False
        self.scroll = 0          # panel scroll offset (px) — content taller than the window
        self._content_h = 0      # measured column height from the last draw
        self._scroll_drag = None
        self._thumb_drag = None  # grabbed the scrollbar thumb (not the content)
        self._tooltip = None
        self.pending_preset = None
        self.pending_save = False
        # per-mode perform state (DESIGN.md §7): explicit slot assignments and
        # setlist order for the ACTIVE mode, seeded by the shell. An empty
        # setlist means "all looks, load order".
        self.bank = {}               # {"1": name, ...} — digits 1-9 recall these
        self.setlist = []            # [ / ] walk this order
        self.pending_slot = None     # preset name whose slot badge was clicked
        self.pending_commands = []   # Action commands with no dedicated mailbox
        self.user_presets = set()    # names that can be renamed/deleted (saved looks)
        self.pending_delete = None   # name confirmed for deletion (live loop applies)
        self.pending_rename = None   # (old, new) committed via Enter (live loop applies)
        self.renaming = None         # name currently being renamed (typing mode)
        self.rename_buf = ""
        self._rename_drawn = False   # the box reached the screen on the last draw
        self._rename_reveal = False  # box just opened: scroll it into view if hidden
        self._reveal_scroll = None   # queued scroll correction (applied next draw)
        self._rename_grace = False   # one-frame token: a reveal was queued THIS draw
        self._del_armed = None       # first x-click arms; second confirms
        self._blink = 0
        self.panel_w = 290           # base (1080p) panel width; scaled by self._s when drawn
        self._s = 1.0                # current UI scale (set each draw from the frame height)
        self._panel_px = 290         # actual drawn panel width in frame px (panel_w * self._s)
        self.mouse = (-1, -1)
        self._hot = []
        self._drag = None
        self._flash = 0          # frames of click feedback left
        self._flash_key = None   # (kind, payload-identity) — re-located each draw
        self._scrim = None       # cached PANEL-colored blend buffer (perf)
        # per-mode accent (DESIGN.md §5): exactly one accent on screen, owned by
        # the active mode. Default ACC green = the shipped Particles chrome, so
        # a bare OverlayUI (goldens, tests) renders identical pixels.
        self.accent = ACC
        # panel title (DESIGN.md §4.1/§4.2 sketch: 'lighteater - MODE TITLE',
        # ASCII hyphen). The shell sets it per mode; the bare default matches
        # the goldens' Particles panel.
        self.panel_title = "lighteater - PARTICLES"
        # still-image mailbox (DESIGN.md §2.1: the shell owns still sources) —
        # a path posted here is loaded by the shell's mailbox pump.
        self.pending_still_path = None
        self._gui = imgui.Gui()
        self.set_spec(build_particles_spec(presets, palettes, mattes))

    def set_spec(self, spec):
        """Bind a panel spec: build the sections state (collapsible headers keep
        their defaults from the spec — the two new families start closed so the
        panel opens looking like it always did, and the default column still
        fits a 1080p window; tests/test_overlay_scroll.py) and the activation
        maps the generic _activate routes through."""
        self.spec = spec
        self.sections = {s.title: s.open for s in spec if isinstance(s, Section)}
        # keyboard param-nudge selection (DESIGN.md §6.2): an index into the
        # spec-order list of nudgeable widgets (Sliders + Cycles). Rebinding
        # the spec (mode switch) resets it to the first control.
        self.nudge_idx = 0
        # ...and the ANCHOR that index is re-derived from. The nudgeable list
        # is filtered by panelspec.visible, so it changes shape the moment a
        # master toggle (Glitch/Flock/Video bg) flips. An index alone would
        # then point at whatever row slid into that slot — see the shell's
        # nudge wiring.
        self.nudge_attr = None
        self._toggles = {}    # hit key -> Toggle
        self._cycles = {}     # hit key -> Cycle
        self._sliders = {}    # attr -> Slider
        self._quant = {}      # attr -> Slider, for the sliders with a quantum
        for wdg in self.iter_widgets():
            if isinstance(wdg, Toggle):
                self._toggles[wdg.attr] = wdg
            elif isinstance(wdg, Cycle):
                if wdg.attr == "res_idx":
                    # the instance may carry a custom boot resolution option
                    wdg.options = [n for n, _, _ in self.res_options]
                self._cycles[wdg.hit_key] = wdg
            elif isinstance(wdg, Slider):
                self._sliders[wdg.attr] = wdg
                if wdg.step:
                    self._quant[wdg.attr] = wdg
        # A spec swap (mode switch) can bind a quantum to an attr that is
        # already carrying an off-grid value from before — land it now, so the
        # panel never draws a number the engine is not using. Only where the
        # engine snaps, though: an `engine_snaps=False` slider (Scale, Hue)
        # can legitimately hold a stored look's fraction — the engine renders
        # it — and re-gridding it on every mode switch would silently retune
        # the look the operator is coming back to.
        for attr, wdg in self._quant.items():
            if wdg.engine_snaps and hasattr(self, attr):
                setattr(self, attr, quantise(wdg, getattr(self, attr)))

    def __setattr__(self, name, value):
        """Quantised sliders (DESIGN.md §4.1/§4.2) land on their own step here
        rather than at each of the four places that write one.

        Sliders, the nudge keys, preset apply and a mode's seeded defaults all
        write plain attributes on this object, and only one of them — the
        panel's own drag — lives in this file. Putting the rule at the write
        means there is exactly one answer to "what value does this control
        hold", which is the whole point: the readout, the OSD, the spec-derived
        HUD line and the engine all read that one number.

        The rule applies only where the ENGINE lands on the grid
        (`Slider.engine_snaps`, the default): for those, any fractional value
        would be a readout lie about a picture already rendered at the snapped
        number. A slider whose engine consumes the value continuously (Scale,
        Hue) is quantised at its input surfaces instead — the drag in
        `_set_from_track`, the nudge keys in the shell — because its one other
        writer, `apply_look`, is a stored look that is the authority on its
        own value and must pass through exactly (see panelspec's `step`
        docstring).

        `nudge_to` (not `quantise`) is the rule, so a 1/40-of-range nudge on a
        four-step slider still moves it; see its docstring for why that cannot
        misfire on the writers that mean an exact value.
        """
        quant = self.__dict__.get("_quant")
        wdg = quant.get(name) if quant else None
        if wdg is not None and wdg.engine_snaps \
                and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            value = nudge_to(wdg, value, self.__dict__.get(name))
        object.__setattr__(self, name, value)

    def iter_widgets(self):
        """Every widget in the spec, sections flattened (spec order)."""
        for item in self.spec:
            if isinstance(item, Section):
                yield from item.widgets
            else:
                yield item

    @property
    def typing(self): return self.renaming is not None
    @property
    def res_name(self): return self.res_options[self.res_idx][0]
    @property
    def res_wh(self): return self.res_options[self.res_idx][1:3]
    @property
    def matte_name(self): return self.mattes[self.matte_idx]
    @property
    def palette_name(self): return self.palettes[self.palette_idx]

    @property
    def dither_name(self): return DITHERS[self.dither_idx]
    @property
    def preset_name(self): return self.presets[self.preset_idx]

    def sync_from(self, pf, glow, matte):
        """Reflect engine state into the sliders (after a preset is applied)."""
        self.fade, self.exposure = glow.fade, glow.exposure
        self.spark, self.curl, self.dot = pf.spark, pf.curl_amp, pf.base_size
        self.damp, self.pull, self.reseed = pf.damp, pf.pull_falloff, pf.reseed_frac
        self.attract_speed = pf.attract_speed
        if matte in self.mattes: self.matte_idx = self.mattes.index(matte)
        if pf.palette in self.palettes: self.palette_idx = self.palettes.index(pf.palette)

    # ----- scaling -----
    def _S(self, n):
        """Scale a baseline pixel value by the current resolution factor (int for cv2)."""
        return self._gui.S(n)

    # ----- keyboard (rename typing) -----
    def begin_rename(self, name, buf=None):
        """Open the rename box on `name` (the pencil click, and the shell's
        save flow). The reveal flag covers the box that opens BELOW the fold:
        save appends the new look's row, and with enough saved looks that row
        is off-screen — the next draw scrolls it into view instead of letting
        `expire_offscreen_rename` silently cancel a rename the operator just
        asked for."""
        self.renaming = name
        self.rename_buf = name if buf is None else buf
        self._rename_reveal = True

    def cancel_rename(self):
        """Drop the rename without committing it — what Esc does."""
        self.renaming = None
        self.rename_buf = ""
        self._rename_reveal = False
        self._reveal_scroll = None

    def expire_offscreen_rename(self):
        """A rename box that did not reach the screen loses the keyboard.

        `on_key` below gives the box EVERY key by design — `q` must not quit
        mid-typing (DESIGN.md §6.2) — and that contract is only safe while the
        box is visible. It was not. Four routes left `renaming` set with
        nothing drawn: collapse the sidebar with the chevron, open the menu
        from the panel's own `Menu (M)` row, hide the overlay, or scroll the
        box off the top of the column. The screen then looked like a
        completely normal instrument, bottom hint and all (`m menu - TAB
        panel - ? keys` — three dead keys), while TAB, `m`, `?`, space, `0`
        and both presses of `q` were typed into a field nobody could see.
        Measured buffer for exactly that sequence: `"classicm? 0qq"`, with
        quit still False. Only Esc got out, and nothing on screen said so.

        The rule is visibility, not a list of routes: **the box may hold the
        keyboard only for as long as it is being painted.** Painted means
        PIXELS: cv2 clips an off-frame draw silently, so `_rename_drawn` is
        earned only when the box's rect actually intersects the frame (see
        `_draw_preset_list`), never merely because the draw walk reached the
        row. The shell calls this once per composed frame, right after the
        draw, so the flag is the just-rendered truth — a collapsed sidebar, an
        open menu, a hidden overlay, a mode switch that retired the name, or a
        wheel that scrolled the row away all cancel it for the same reason and
        without being enumerated here. Any future way to take the panel off
        screen is covered the day it is written.

        One deliberate grace: a box that just OPENED off-screen (save appends
        the new look's row, which can sit below the fold) has a scroll
        correction queued (`_reveal_scroll`) — the next draw paints it, so it
        is not cancelled in the gap. The grace is a one-frame token granted
        only by the draw walk that queued the reveal: a frame that never runs
        the walk grants nothing, so a menu or a hidden overlay still expires
        the box even mid-reveal.

        Cancelling is silent on purpose: the click that hid the box is its own
        feedback (the panel visibly collapsed / the menu opened), so a toast
        would be noise, and the rename was never committed — the look keeps
        the name it already had.
        """
        if self.renaming is not None and not self._rename_drawn \
                and not self._rename_grace:
            self.cancel_rename()
        # consumed, not merely read: on the frames where the panel is not drawn
        # at all — menu open, overlay HIDDEN or HUD — `draw()` never runs, so
        # this is the only place the flags can go stale, and a stale True is
        # exactly the deaf keyboard this exists to prevent.
        self._rename_drawn = False
        self._rename_grace = False

    def on_key(self, key):
        """Feed a cv2.waitKey code. Returns True if consumed (a rename box is open),
        so the caller knows not to treat 'q' as quit while the user is typing.

        Contract (DESIGN.md §6.2): rename-typing consumes EVERY key — while
        renaming, no global keys fire. Esc is consumed too, and cancels the
        rename only (it must not also step the overlay state or anything else);
        `Esc` then `0` is the two-press panic escape hatch."""
        if self.renaming is None:
            return False
        if key in (13, 10):              # enter — commit
            new = self.rename_buf.strip()
            if new and new != self.renaming:
                self.pending_rename = (self.renaming, new)
            self.renaming = None
            self._rename_reveal, self._reveal_scroll = False, None
        elif key == 27:                  # esc — cancel
            self.cancel_rename()
        elif key in (8, 127):            # backspace / delete
            self.rename_buf = self.rename_buf[:-1]
        elif 32 <= key <= 126 and len(self.rename_buf) < 22:
            self.rename_buf += chr(key)
        return True

    # ----- layout -----
    def draw(self, frame, info):
        g = self._gui
        self._tooltip = None
        h, w = frame.shape[:2]
        # scale the whole panel by the output resolution, floored at the 1080p baseline so
        # 720p/1080p are unchanged and 4K renders at 2x (same fraction of the frame).
        self._s = max(1.0, h / BASE_H)
        # cleared here and set again only if the rename box actually gets
        # painted below — see expire_offscreen_rename()
        self._rename_drawn = False
        self._hot = g.begin(self._s, self.mouse, self.accent)
        pw = g.S(self.panel_w)
        self._panel_px = pw
        if not self.open:
            # the same handle the HUD draws (imgui.panel_handle): one
            # affordance for "open the panel", wherever you meet it
            r = imgui.panel_handle(
                frame, hover=_in(imgui.panel_handle_rect(w, h), self.mouse))
            self._hot.append((r, "collapse", None))
            self._draw_status(frame, info)
            return frame

        px = w - pw
        # panel scrim restricted to the panel ROI (perf: the full-frame
        # copy+addWeighted was an identity op outside the panel — pixels
        # elsewhere are untouched, so the result is unchanged). The blend
        # runs on a contiguous copy of the ROI (cv2 on a column-sliced view
        # is slower than copying), against a cached PANEL-colored buffer.
        sx = max(px, 0)                # a frame narrower than the panel clamps
        roi = np.ascontiguousarray(frame[:, sx:])
        if self._scrim is None or self._scrim.shape != roi.shape:
            self._scrim = np.empty_like(roi)
            self._scrim[:] = PANEL
        cv2.addWeighted(self._scrim, 0.86, roi, 0.14, 0, dst=roi)
        frame[:, sx:] = roi
        # the column can be taller than the window (e.g. 720p) — scroll, clamped so it's
        # a no-op when everything fits. content height comes from the previous draw.
        if self._reveal_scroll is not None:
            # a rename box opened off-screen last draw: bring its row into
            # view (queued there, applied here, so the walk below paints it)
            self.scroll = self._reveal_scroll
            self._reveal_scroll = None
        self.scroll = imgui.clamp_scroll(self.scroll, self._content_h, h)
        x, cw, y = px + g.S(16), pw - g.S(32), g.S(30) - self.scroll
        # the title keeps clear of the close handle (drawn last, on top):
        # where its glyphs would reach into the handle's rows (at 720p the
        # 'LES' of 'PARTICLES' sat under the '>'; 1080p and up clear it
        # vertically), it shrinks to end S(8) left of the handle. Shrunk,
        # not ellipsized, so the mode's name still reads in full (the same
        # choice hud.fit_px makes for the containment flash).
        hx0, hy0, _, hy1 = imgui.panel_handle_rect(w, h)
        t_scale, t_thick = 0.62, 2
        (tw, th), tb = cv2.getTextSize(
            self.panel_title, cv2.FONT_HERSHEY_SIMPLEX, t_scale * g.s,
            max(1, int(round(t_thick * g.s))))
        max_w = hx0 - g.S(8) - x
        # the title's ink spans rows y-th-1 .. y+tb-1 (measured on
        # 'lighteater - PARTICLES' at 720p, 1080p and 4K: getTextSize's
        # baseline is one row past the lowest ink, its height one row short
        # of the highest)
        if y - th - 1 <= hy1 and y + tb - 1 >= hy0 and tw > max_w > 0:
            t_scale *= max_w / tw
            # Hershey advance is not linear in the scale: step down until
            # the rendered width really fits
            while t_scale > 0.2 and cv2.getTextSize(
                    self.panel_title, cv2.FONT_HERSHEY_SIMPLEX, t_scale * g.s,
                    max(1, int(round(t_thick * g.s))))[0][0] > max_w:
                t_scale -= 0.01
        g.text(frame, self.panel_title, x, y, self.accent, t_scale, t_thick)
        y += g.S(16)
        # the column starts below the close button (the 2.75u panel handle,
        # drawn last at a fixed spot), so at scroll 0 no row, section header
        # or slot badge sits under it: collapse wins every click on its rect,
        # which would leave whatever lay beneath unreachable. Scrolled rows
        # still pass under it, as they always did under the old close box.
        y = max(y + self.scroll,
                imgui.panel_handle_rect(w, h)[3] + g.S(6)) - self.scroll
        self._blink += 1

        # ----- the generic walk: sections, then the shell's global rows -----
        for item in self.spec:
            if isinstance(item, Section):
                y, sec_open = g.section(frame, item.title,
                                        self.sections.get(item.title, True), x, y, cw,
                                        key_hint=item.key_hint)
                if sec_open:
                    for wdg in item.widgets:
                        y = self._draw_widget(frame, wdg, x, y, cw, px)
                y += g.S(item.gap)
            else:
                y = self._draw_widget(frame, item, x, y, cw, px)
        self._content_h = y + self.scroll + g.S(8)   # column bottom incl. margin, unscrolled

        # collapse button drawn last so it stays fixed and clickable above
        # scrolled content. It is the panel handle itself, on the handle's
        # rect (imgui.panel_handle): a second click where the handle was
        # closes the panel instead of landing on the row under it.
        cr = imgui.panel_handle(
            frame, hover=_in(imgui.panel_handle_rect(w, h), self.mouse),
            glyph=">")
        self._hot.append((cr, "collapse", None))
        g.scrollbar(frame, px, h, self._content_h, self.scroll,
                    step=g.S(imgui.SCROLL_STEP))

        # click feedback: flash the last-clicked control — border-only (a fill
        # would hide the row's own content, e.g. the armed delete's 'sure?'),
        # re-located against THIS draw's hit rects so a res change, section
        # reflow, or scroll never leaves the flash on a stale rectangle. A
        # control that left the screen simply loses its flash.
        if self._flash > 0:
            rect = next((r for r, k, p in self._hot
                         if self._flash_id(k, p) == self._flash_key), None)
            if rect is not None:
                x0, y0, x1, y1 = rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255),
                              max(1, g.S(1)))   # crisp non-AA flash border
            self._flash -= 1
        self._tooltip = g.tooltip
        g.draw_tooltip(frame, self.h)
        self._draw_status(frame, info)
        return frame

    def _draw_widget(self, frame, wdg, x, y, cw, px):
        """Draw one spec widget at y; return the next y (the shipped rhythm:
        rows advance S(28+gap) in one rounding, sliders/cycles advance their own
        height then add S(gap) — exactly the shipped two-step literals)."""
        g = self._gui
        if not visible(self, wdg):
            # gated off (panelspec.visible): no row, no pixels, no hit rect —
            # the control's value is untouched and still captures into looks
            return y
        if isinstance(wdg, Slider):
            val = getattr(self, wdg.attr)
            # display_fmt, not wdg.fmt: a stored legacy fraction on an
            # engine-continuous slider is really rendering, so ".0f" would
            # print "46" over a 45.55-row grid (panelspec.display_fmt)
            y = g.slider(frame, wdg.label, wdg.attr, val, wdg.lo, wdg.hi, x, y, cw,
                         info=wdg.tip or None, fmt=display_fmt(wdg, val))
            return y + g.S(wdg.gap) if wdg.gap else y
        if isinstance(wdg, Toggle):
            val = bool(getattr(self, wdg.attr))
            label = (wdg.label_fn(val) if wdg.label_fn else
                     f"{wdg.label}: {wdg.on_text if val else wdg.off_text}")
            g.row(frame, label, wdg.attr, x, y, cw, active=val)
            return y + g.S(28 + wdg.gap)
        if isinstance(wdg, Cycle):
            idx = getattr(self, wdg.attr) % len(wdg.options)
            y = g.cycle(frame, wdg.label, wdg.options[idx], wdg.hit_key, x, y, cw,
                        info=wdg.tip or None)
            return y + g.S(wdg.gap) if wdg.gap else y
        if isinstance(wdg, Action):
            g.row(frame, wdg.label, wdg.command, x, y, cw)
            return y + g.S(28 + wdg.gap)
        if isinstance(wdg, PresetList):
            return self._draw_preset_list(frame, x, y, cw, px)
        if isinstance(wdg, Readout):
            return wdg.render(frame, g, x, y, cw)
        if isinstance(wdg, Param):
            return y                # captured, never drawn (no row, no pixels)
        raise TypeError(f"unknown panel-spec widget {wdg!r}")

    def _draw_preset_list(self, frame, x, y, cw, px):
        """TEMPLATES rows (slot badges, rename box, hover manage buttons) +
        Save current look. Badge hits go to the FRONT of `hot` (like the manage
        buttons) so they win over the full-row rect they sit on."""
        g = self._gui
        slots = {n: s for s, n in (self.bank or {}).items()}
        for i, name in enumerate(self.presets):
            if name == self.renaming:
                g.rename_box(frame, self.rename_buf, self._blink, x, y, cw, px)
                # "painted" means PIXELS, not "the walk got here": cv2 clips an
                # off-frame draw silently, so a box scrolled past the top was
                # "drawn" every frame while an invisible field kept the whole
                # keyboard (TAB - s - a few wheel notches was enough). The
                # flag — and with it the keyboard (expire_offscreen_rename) —
                # is earned only when the box's rect intersects the frame.
                if 0 < y + g.S(24) and y < frame.shape[0]:
                    self._rename_drawn = True
                    self._rename_reveal = False
                elif self._rename_reveal:
                    # just opened, below the fold (save appends the row):
                    # queue a scroll that shows it rather than letting the
                    # rename the operator just asked for silently expire.
                    # `ty` is the row's unscrolled column offset; the grace
                    # token buys exactly the one frame the queue needs.
                    ty = y + self.scroll
                    self._reveal_scroll = max(
                        ty - g.S(4) if y < 0
                        else ty + g.S(28) - frame.shape[0], 0)
                    self._rename_grace = True
                y += g.S(28)
                continue
            r = g.row(frame, name, "preset", x, y, cw,
                      active=(i == self.preset_idx), payload=i)
            br = g.slot_badge(frame, slots.get(name), x, y, cw)
            self._hot.insert(0, (br, "slot", name))
            if name in self.user_presets and (_in(r, self.mouse) or name == self._del_armed):
                g.manage_buttons(frame, name, x, y, cw, self._del_armed == name)
            y += g.S(28)
        g.row(frame, "+ Save current look", "save", x, y, cw)
        return y + g.S(30)

    def _draw_status(self, frame, info):
        g = self._gui
        g.text(frame, info.get("status", ""), g.S(12), g.S(24), self.accent, 0.5)
        if info.get("black"):
            h, w = frame.shape[:2]
            g.text(frame, "CAMERA IS BLACK - disable iPhone Continuity Camera",
                   g.S(12), h // 2, RED, 0.8, 2)
            g.text(frame, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                   g.S(12), h // 2 + g.S(28), (140, 140, 240), 0.5)
        if self.record:
            h, w = frame.shape[:2]
            if self.open:
                cv2.circle(frame, (w - self._panel_px - g.S(22), g.S(24)),
                           g.S(7), RED, -1)
            else:
                # beside the handle, not on it: the handle is title-safe inset
                # and 2.75u tall, and at 720p the old (w-70, 24) dot landed
                # inside it
                hx0, hy0, _, hy1 = imgui.panel_handle_rect(w, h)
                cv2.circle(frame, (hx0 - g.S(16), (hy0 + hy1) // 2),
                           g.S(7), RED, -1)

    # ----- mouse -----
    @staticmethod
    def _flash_id(kind, payload):
        """Stable identity for the click-flash relocation: a slider's payload
        carries track geometry that changes with res/reflow, so key on its
        attr; everything else's payload is already stable."""
        if kind == "slider":
            return (kind, payload[0])
        return (kind, payload)

    def on_mouse(self, event, x, y, flags, param=None):
        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # NOT a cursor position: on the Cocoa backend (x, y) IS the wheel
            # delta, so assigning it to self.mouse teleports every hover
            # highlight to the top-left corner mid-scroll. Decode and return.
            if event == cv2.EVENT_MOUSEWHEEL and self.open:
                self.scroll = imgui.wheel_scroll(
                    self.scroll, self._S(imgui.SCROLL_STEP),
                    imgui.wheel_delta(x, y, flags))
            return
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Fixed chrome first: the collapse button floats ABOVE scrolled
            # content, so a row that scrolled underneath it (including a
            # hovered preset's delete button, which front-inserts its hit)
            # must never steal its click — that click-theft could arm and
            # confirm a preset delete in two presses (data loss).
            for rect, kind, payload in self._hot:
                if kind == "collapse" and _in(rect, (x, y)):
                    self._flash_key, self._flash = self._flash_id(kind, payload), 4
                    self._activate(kind, payload, x, y)
                    return
            for rect, kind, payload in self._hot:
                if kind == "collapse":
                    continue
                if _in(rect, (x, y)):
                    self._flash_key, self._flash = self._flash_id(kind, payload), 4
                    self._activate(kind, payload, x, y)
                    return
            if self.open and x >= self.w - self._panel_px:
                self._scroll_drag = (y, self.scroll)   # empty panel area: drag to scroll
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag:
                self._set_from_track(self._drag, x)
            elif self._thumb_drag:
                # the THUMB follows the finger (imgui's convention block)
                y0, s0, travel, span = self._thumb_drag
                self.scroll = imgui.thumb_scroll(s0, y0, y, travel, span)
            elif self._scroll_drag:
                # the CONTENT follows the finger (imgui's convention block)
                y0, s0 = self._scroll_drag
                self.scroll = imgui.drag_scroll(s0, y0, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
            self._scroll_drag = None
            self._thumb_drag = None

    def _set_from_track(self, payload, x):
        """Turn a mouse x on a slider's track into that slider's value.

        Quantised the EXACT way (`quantise`, not `nudge_to`): a drag is a
        request for the value under the finger, so it must land on the nearest
        step and stay there while the finger wanders inside it — a drag that
        ratcheted a step per mouse-move event would be unusable. It is also
        what keeps the nudge rule honest: this write arrives already on the
        grid, so "the write did not move it" can only mean the finger is still
        inside the same step.
        """
        attr, x0, x1, lo, hi = payload
        t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
        val = lo + t * (hi - lo)
        wdg = self._quant.get(attr)
        setattr(self, attr, quantise(wdg, val) if wdg is not None else val)

    def _activate(self, kind, payload, x, y=0):
        """Generic activation: toggles flip their attr, cycles rotate through
        their options, sliders set their attr from the track position, actions
        post to their mailbox. Routing comes from the spec (set_spec's maps)."""
        if kind != "del" and self._del_armed:
            self._del_armed = None       # any other click disarms a pending delete
        if kind == "scroll":
            # a scrollbar arrow: payload is the signed step (down = positive =
            # show me what is below — imgui's convention block). draw() clamps.
            self.scroll += payload
            return
        if kind == "scrolltrack":
            ty0, track_h, thumb_h, span, page = payload
            travel = track_h - thumb_h
            top = ty0 + int(travel * (min(max(self.scroll, 0), span) / span))
            if top <= y <= top + thumb_h:
                self._thumb_drag = (y, self.scroll, travel, span)
            else:
                self.scroll += page if y > top else -page
            return
        if kind == "del":
            self._del_armed, confirmed = imgui.arm_delete(self._del_armed, payload)
            if confirmed:
                self.pending_delete = confirmed
            return
        if kind == "ren":
            self.begin_rename(payload)   # prefill with the current name
            return
        if kind == "slot":
            self.pending_slot = payload  # the shell assigns/clears + persists
            return
        if kind == "collapse":
            self.open = not self.open
        elif kind == "preset":
            self.preset_idx = payload
            self.pending_preset = self.presets[payload]
        elif kind == "cycle":
            key, d = payload
            wdg = self._cycles[key]
            setattr(self, wdg.attr, (getattr(self, wdg.attr) + d) % len(wdg.options))
        elif kind == "slider":
            self._drag = payload
            self._set_from_track(payload, x)
        elif kind == "section":
            self.sections[payload] = not self.sections.get(payload, True)
        elif kind in self._toggles:
            attr = self._toggles[kind].attr
            setattr(self, attr, not getattr(self, attr))
        elif kind == "save":
            self.pending_save = True
        elif kind == "quit":
            self.quit = True
        else:
            self.pending_commands.append(kind)   # Action with no dedicated mailbox
