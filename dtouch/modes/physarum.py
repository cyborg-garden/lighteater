"""Physarum — the slime-mold instrument (Mode protocol, DESIGN.md §2.2).

The trail map IS the picture: a Jones-model mold (dtouch.physarum) grows vein
networks, and the live video steers it two ways — the matte spatially blends
every simulation parameter between a "field" behavior point and a "body"
behavior point (whoever is in frame runs different physics than the room), and
the footage's luminance is food the sensors are drawn toward.

Three feel knobs sit on top of the model (MOLD section), each riding a
perceptual value**0.6 curve so the bottom third of the slider is already
clearly audible: **Weave** drives the engine's anti-thoroughfare levers
(sensor saturation, heading jitter, sub-population sense split, reseed churn,
diffuse trim) so the picture leans reticulated-network instead of a few fat
canals; **Evolve** hops the mold between REGIMES — distinct growth characters
(trunk highways, lace, shatter, flood, drift) with decisive crossfades and
randomized 15-45 s dwells — breathes lightly between hops, marches a phantom
food blob around the frame, and MELTS whatever stops changing (staleness =
low temporal variance -> local decay boost), so equilibrium is structurally
impossible; **React** turns motion into carving — motion history feeds the
sensors, holds trail decay where you swept, and fires local gather impulses
at the gesture. Z casts a random regime (points + look params from the
validated bands) with a forced hop and melt surge — the slot machine that
always pays.

**Depth** (LOOK section; H toggles it flat and back) lights the picture as a
relief of the trail under a key light that orbits slowly and drops on the
bass, so veins read as lit tubes stacked over one another. Both engines
render it (tonemap.frag on the GPU, dtouch.physarum.relief on numpy).

Two engines run the same model behind the same contract: the GPU field
(dtouch.physarum_gl — millions of agents on the full grid, moderngl
ping-pong) is tried first and the numpy field (dtouch.physarum) is the
fallback when no GL context can be had, toasted in amber (DESIGN.md §6.4: the
show still has to start). Headless-testable either way. The mode owns its
engine + matte, its panel sections (TEMPLATES / SOURCE / MOLD / LOOK), one
mode-local command (X swaps the two behavior points, the single most
theatrical move the instrument has), and its defaults + built-in looks.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..circuit_bent import CircuitBent
from ..commands import Command
from ..hud import AMBER
from ..matte import MatteUnavailable, make_matte, select_matte
from ..overlay_ui import RES_OPTIONS, sync_signal
from ..panelspec import Cycle, PresetList, Section, Slider, Toggle
from ..physarum import POINT_NAMES, PhysarumField
from ..physarum_gl import PhysarumFieldGL
from ..rack_gl import PhysarumOutGL
from .particles import MATTE_H, MATTE_W, MATTES, composite_video_bg

ACCENT = (40, 190, 250)          # BGR — amber, the color of the reference mold

# ----- evolve's regime table -----------------------------------------------
# The instrument must never settle (org design principle: magic and play over
# control). Instead of only breathing every parameter on gentle sines, evolve
# now hops between REGIMES — bundles of multipliers on the blended point
# parameters, each a distinct growth character — with decisive crossfades and
# randomized dwell. Our own regimes, derived from this mode's own points and
# offline sweeps; no external parameter tables.
#   sense/turn/spread/step/deposit — multipliers on the blended point params
#   gain/food — multipliers on the live slider values
#   jitter — ADDITIVE heading wobble (rad) on top of weave's
REGIMES = {
    "cruise":  dict(),                                       # the sliders as set
    "surge":   dict(sense=2.5, turn=0.70, spread=0.85, step=1.45,
                    deposit=1.6, gain=1.20, food=0.60),      # fat trunk highways
    "shatter": dict(sense=0.55, turn=1.80, spread=1.70, step=1.20,
                    deposit=0.75, gain=1.15, food=0.45, jitter=0.40),  # re-melt
    "knit":    dict(sense=0.45, turn=1.30, spread=0.60, step=0.65,
                    deposit=0.70, gain=0.90, food=1.40),     # fine local lace
    "flood":   dict(sense=1.3, turn=0.90, spread=1.25,
                    deposit=1.2, food=2.5),                  # chase the light
    "drift":   dict(sense=1.9, turn=1.50, spread=1.40, step=0.85,
                    food=0.20),                              # unmoored wander
}
REGIME_NAMES = list(REGIMES)
REGIME_KEYS = ("sense", "turn", "spread", "step", "deposit", "gain", "food")
# Perceptual curve on the three MOLD feel sliders. The contract
# (tests/test_physarum_alive.py) is that each is perceptible within 5 seconds
# AT 0.3, not merely at 1.0 — the bottom third of a slider has to be where
# most of the playing happens, or the control is a switch with decoration.
#
# It was 0.6. After the species and mosaic work the response went back-loaded:
# weave measured 0.044 at 0.3 against a 0.06 floor while reaching 0.448 at
# 1.0, and evolve 0.065 against 0.08 while reaching 0.138. The mechanisms got
# far stronger at the top and the curve no longer compensated at the bottom.
FEEL_CURVE = 0.30

REGIME_FADE = 3.0                # crossfade, seconds — decisive, not a jump cut
REGIME_DWELL = (15.0, 45.0)      # dwell range, seconds, order randomized

# Staleness melt: regions whose (blurred) picture stops changing start to
# dissolve — decay is locked-structure's predator, so equilibrium is
# structurally impossible while evolve is up. VAR floor is on blurred
# half-res luminance (grain dust averaged out); MELT_GAIN feeds the signed
# keep map's negative side (see dtouch.physarum MELT_DROP).
MELT_VAR = 3e-4
MELT_GAIN = 0.55

# ----- depth: the picture lit as a relief (art-director call 2026-09-24) -----
# The trail is read as a height field and lit by one key light
# (shaders/physarum/tonemap.frag), so veins read as lit tubes: a bright rim
# on one side, a dark crease on the other, thin veins visibly UNDER trunks.
# Chosen over stratified layers (3.4x the cost, nothing for the blob looks)
# and a volumetric glow (veiled faint strands instead of pushing them back).
#
# Per-look depth. The veins looks take it hardest (tubes and under-crossings
# read strongest; on lightning it also exposes bead-like density pulses along
# trunks, kept as a feature). The blob looks stay modest: ghost goes chrome
# near 1.0, and amoeba's and breath's biggest blobs saturate, band into
# contours and grow a hard shadow line there, while 0.5-0.6 pulls hidden
# anatomy out of their white blow-out. Custom looks and panic get the default.
LOOK_DEPTH = {"veinwork": 0.9, "lightning": 0.9, "ghost": 0.6,
              "amoeba": 0.5, "breath": 0.5}
DEPTH_DEFAULT = 0.6
# The light orbits. A light that never moves reads as an emboss filter; a
# moving one is the strongest shape cue a flat picture can give (creases
# slide around each tube, and which vein sits on top re-reads as it turns).
# These are starting values from stills: the orbit has NOT been seen in
# motion, and the motion playtest is where ORBIT_S (try 32-90 s) and the
# rake get tuned.
DEPTH_ORBIT_S = 48.0             # seconds per full turn of the key light
DEPTH_AZ0 = 0.75 * np.pi         # start azimuth, SCREEN space (+y up): upper left
# Elevation is the light's z (sine of its angle above the picture). The bass
# lowers it (the kick rakes the light across the relief, so creases deepen and
# shadows lengthen on the beat): elev = ELEV_HI - RAKE * sens * bass, clamped
# to [ELEV_LO, ELEV_HI]. ELEV_LO sits far above the shader's z floor
# (dtouch.physarum.RELIEF_MIN_Z), so the host clamp never engages here.
DEPTH_ELEV_HI = 0.63
DEPTH_ELEV_LO = 0.30
DEPTH_RAKE = 0.30
# Whether the Depth slider rides FEEL_CURVE like the MOLD sliders. It does
# not: linear depth already clears the perceptibility floor at 0.3 (the
# honest-slider test in tests/test_physarum_gl.py), and the curve would spend
# the bottom of the slider on a jump from flat to 70% relief.
DEPTH_FEEL_CURVE = False


def depth_light(t, bass=0.0, sens=1.0):
    """The key light at clock `t` seconds, in SCREEN space (+x right, +y up,
    +z toward the viewer), unit length: the orbit plus the bass rake. The
    browser port does the same arithmetic from looks.json's `depth` block."""
    az = DEPTH_AZ0 + 2.0 * np.pi * t / DEPTH_ORBIT_S
    elev = min(max(DEPTH_ELEV_HI - DEPTH_RAKE * sens * bass, DEPTH_ELEV_LO),
               DEPTH_ELEV_HI)
    ce = float(np.sqrt(1.0 - elev * elev))
    return (float(np.cos(az)) * ce, float(np.sin(az)) * ce, float(elev))


def screen_to_grid(light):
    """SCREEN-space light -> the grid-texel space tonemap.frag reads. Row 0 is
    the top of the desktop window, so screen-up is grid -y."""
    x, y, z = light
    return (x, -y, z)


def depth_feel(v):
    """Depth slider value -> engine depth (see DEPTH_FEEL_CURVE)."""
    v = min(max(float(v), 0.0), 1.0)
    return v ** FEEL_CURVE if DEPTH_FEEL_CURVE else v


def _regime_mults(name):
    """(sense, turn, spread, step, deposit, gain, food, jitter) for a regime."""
    b = REGIMES[name]
    return np.array([b.get(k, 1.0) for k in REGIME_KEYS] + [b.get("jitter", 0.0)],
                    np.float32)

PALETTES_PH = ["arctic", "fire", "aurora", "violet", "toxic", "rose", "mono", "video"]

# gradient stops (RGB), interpolated to a 256-entry LUT per palette
_PALETTE_STOPS = {
    "arctic": [(0, 0, 0), (18, 28, 88), (40, 118, 200), (120, 220, 255), (255, 255, 255)],
    "fire":   [(0, 0, 0), (80, 10, 0), (200, 60, 10), (255, 160, 20), (255, 250, 180)],
    "aurora": [(0, 0, 0), (8, 60, 48), (20, 180, 120), (140, 120, 220), (240, 240, 255)],
    "violet": [(0, 0, 0), (58, 10, 88), (160, 40, 180), (255, 120, 220), (255, 255, 255)],
    "toxic":  [(0, 0, 0), (6, 40, 8), (30, 140, 30), (150, 240, 80), (240, 255, 220)],
    "rose":   [(0, 0, 0), (70, 12, 40), (190, 40, 90), (255, 120, 150), (255, 235, 240)],
    "mono":   [(0, 0, 0), (255, 255, 255)],
}
_LUT_CACHE = {}


def _fmt_agents(n):
    """400000 -> '400k', 2000000 -> '2.0M' — the status line's agent count."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n // 1000}k"


def _palette_lut(name):
    """256x3 uint8 RGB ramp for a named palette, cached forever (tiny)."""
    lut = _LUT_CACHE.get(name)
    if lut is None:
        stops = np.array(_PALETTE_STOPS[name], np.float32)
        xs = np.linspace(0.0, 1.0, len(stops))
        t = np.linspace(0.0, 1.0, 256)
        lut = np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)],
                       axis=1).astype(np.uint8)
        _LUT_CACHE[name] = lut
    return lut


def _colorize(lum, name):
    """lum float32 [0,1] (h, w) -> uint8 RGB (h, w, 3) through the palette.

    cv2.applyColorMap with the ramp as a user colormap, not `lut[idx]`:
    numpy's fancy index costs 4.4 ms on the GPU engine's 1280x736 grid,
    more than the whole simulation; applyColorMap is 0.09 ms and
    pixel-identical (it indexes the 256 triplets as given, no channel swap).
    """
    idx = (lum * 255.0).astype(np.uint8)
    return cv2.applyColorMap(idx, _palette_lut(name).reshape(256, 1, 3))


class PhysarumMode:
    """Live video as the pen inside a slime-mold simulation."""

    id = "physarum"
    title = "Physarum"
    key = "o"                    # 'p' belongs to Particles; o as in Organism/Ooze
    accent = ACCENT
    accepts_still = False
    blurb = "slime-mold veins\neat the light"

    # option lists the shell's panel walker needs
    palettes = list(PALETTES_PH)
    mattes = list(MATTES)
    points = list(POINT_NAMES)

    # Ordered by playability — the first 9 seed bank slots 1-9.
    # weave/evolve/react per look keep each identity while tilting the whole
    # instrument toward network-not-thoroughfares (owner feedback 2026-08):
    # veinwork and lightning web hardest; breath stays the calmest.
    BUILTIN = {
        "veinwork":  dict(point_bg="veins", point_fg="fingers", palette="arctic",
                          matte="auto", food=0.35, gain=1.0, decay=0.94, exposure=3.5,
                          weave=0.7, evolve=0.5, react=0.7, depth=LOOK_DEPTH["veinwork"]),
        "amoeba":    dict(point_bg="cells", point_fg="storm", palette="fire",
                          matte="motion", food=0.50, gain=1.1, decay=0.92, exposure=3.0,
                          weave=0.55, evolve=0.6, react=0.8, depth=LOOK_DEPTH["amoeba"]),
        "ghost":     dict(point_bg="haze", point_fg="web", palette="mono",
                          matte="person", food=0.60, gain=0.9, decay=0.96, exposure=4.5,
                          weave=0.5, evolve=0.75, react=0.6, depth=LOOK_DEPTH["ghost"]),
        "lightning": dict(point_bg="web", point_fg="fingers", palette="violet",
                          matte="edges", food=0.45, gain=1.4, decay=0.90, exposure=3.0,
                          weave=0.65, evolve=0.6, react=0.8, depth=LOOK_DEPTH["lightning"]),
        "breath":    dict(point_bg="haze", point_fg="cells", palette="aurora",
                          matte="luma", food=0.30, gain=0.8, decay=0.95, exposure=4.0,
                          weave=0.45, evolve=0.7, react=0.5, depth=LOOK_DEPTH["breath"]),
    }

    # apply="reset" merges a look over these; matte / video_bg / video_mix are
    # deliberately absent (keep semantics — rig switches survive look hops).
    DEFAULTS = dict(point_bg="veins", point_fg="fingers", palette="arctic",
                    food=0.35, gain=1.0, decay=0.94, exposure=3.5, grain=0.2,
                    weave=0.6, evolve=0.5, react=0.7, depth=DEPTH_DEFAULT)

    _UI_DEFAULTS = dict(ph_matte_idx=0, ph_food=0.35, ph_video_bg=False,
                        ph_video_mix=0.5, ph_point_bg_idx=0, ph_point_fg_idx=2,
                        ph_gain=1.0, ph_decay=0.94, ph_palette_idx=0,
                        ph_exposure=3.5, ph_grain=0.2,
                        ph_weave=0.6, ph_evolve=0.5, ph_react=0.7,
                        ph_depth=DEPTH_DEFAULT, ph_quality_idx=0)

    # Per-engine sizing. The CPU field is budgeted at ~21 ms/frame on the
    # working grid; the GPU field runs 2M agents on a 1280x736 grid in ~8 ms
    # (experiments/08-physarum-gl). An explicit grid / n overrides both.
    ENGINES = ("auto", "gl", "cpu")
    # Agent DENSITY (agents per grid cell) is the load-bearing number, not the
    # agent count. Jones reticulation lives near 0.05-0.3 agents/cell; the
    # shipped 2.12/cell filled every cell with ~33 units of trail, so there was
    # no dark for a vein to be a vein against and measured vein/floor contrast
    # sat near 2-4x instead of 20-50x. These sizings hold ~0.5/cell.
    CPU_GRID, CPU_N = (576, 324), 100_000
    GL_GRID, GL_N = (1280, 736), 500_000

    # Render-quality tiers (GL engine only; the CPU fallback has no headroom).
    # perform = the shipped sizing; higher tiers raise the sim grid + agent
    # pool so the veins stay crisp on a 1440p/4K projector. Switching tiers
    # rebuilds the field live — the trail regrows in a couple of seconds.
    QUALITY = {
        "perform": ((1280, 736), 500_000),
        "balance": ((1920, 1104), 1_100_000),
        "quality": ((2560, 1472), 1_900_000),
    }
    QUALITY_NAMES = list(QUALITY)

    def __init__(self, matte="auto", grid=None, n=None, seed=1, engine="auto"):
        if engine not in self.ENGINES:
            raise ValueError(f"engine must be one of {self.ENGINES}, got {engine!r}")
        self.matte_kind = matte
        self._grid = tuple(grid) if grid is not None else None
        self._n = n
        self._quality = "perform"
        self.grid = self._grid or self.CPU_GRID
        self.n = self._n or self.CPU_N
        self.seed = seed
        self.engine_pref = engine
        self.engine = None           # "gl" / "cpu" once started
        self._px_scale = 1.0         # grid-px per CPU-ground-truth-px (set per engine)
        self.host = None
        self.pf = None
        self.mat = None
        self._burst_pending = False
        self._wave_pending = False
        self._t = 0.0                # evolve clock (sums clamped dt)
        # the depth key light's own clock, reset in start(): the shell caches
        # mode instances (shell.py _switch_mode), so on re-entry _t resumes
        # where it stopped, and the light would too if it rode _t
        self._orbit_t = 0.0
        self._prev_gray = None       # last grid-res luma (react's motion diff)
        self._motion = None          # lingering motion-energy map (react)
        # SIGNAL-on-GPU state (dtouch.rack_gl): the output composer + rack
        # on the field's context, and the per-frame flag telling the shell
        # its CPU rack already ran here (DESIGN.md §2.4)
        self._glout = None
        self._glout_key = None
        self._rack_gl_ok = True      # one GL failure disables the path (§6.4)
        self.signal_done = False
        # regime scheduler (evolve): decisive hops between growth characters
        self._reg_rng = np.random.default_rng(seed * 7919 + 17)
        self._reg_from = "cruise"
        self._reg_to = "cruise"
        self._reg_t = 0.0
        # first dwell samples the low end of the range so evolve's first
        # decisive hop arrives within the first half-minute of a session
        self._reg_dwell = float(self._reg_rng.uniform(REGIME_DWELL[0], 25.0))
        # staleness tracker (melt): EMA mean/var of blurred half-res luminance
        self._ema = None
        self._var = None
        self._last_lum = None        # last frame's luminance (staleness input)
        self._melt_pulse = 0.0       # extra melt right after a cast (random)
        self._cast_rng = np.random.default_rng(seed * 104729 + 31)
        self._depth_last = DEPTH_DEFAULT  # what h restores after FLAT

    # ----- lifecycle -----
    def start(self, host):
        self.host = host
        try:
            self.mat = make_matte(self.matte_kind)
        except MatteUnavailable as e:
            # missing optional dep: say so and boot on the default matte —
            # the show still has to start (DESIGN.md §6.4)
            host.hud.toasts.flash(str(e))
            self.matte_kind = "auto"
            self.mat = make_matte(self.matte_kind)
        self.pf = self._build_field(host)
        self._orbit_t = 0.0          # light back at upper left on every entry

    def _build_field(self, host):
        """The GPU field when it can be had, else the CPU field — sized per
        engine unless the caller fixed grid / n. GL failing to come up is
        not a reason to lose the show (DESIGN.md §6.4): it is toasted in
        amber, printed for the headless log, and the mold runs on numpy."""
        # any prior GL output composer belonged to the old field/context
        self._glout = None
        self._glout_key = None
        if self.engine_pref != "cpu":
            q_grid, q_n = self.QUALITY.get(self._quality, self.QUALITY["perform"])
            self.grid = self._grid or q_grid
            self.n = self._n or q_n
            gw, gh = self.grid
            try:
                pf = PhysarumFieldGL(n=self.n, gw=gw, gh=gh, seed=self.seed)
                self.engine = "gl"
                # The look/point parameters (sense, step — and the blur that
                # sets vein thickness) are calibrated in CPU-grid pixels. On
                # the GL grid a pixel covers ~1/2.2 as much of the frame, so
                # driving the raw values halved the mold's relative scale:
                # every look collapsed into the same fine wire-mesh and the
                # veins fell below what the projector/dither stage resolves.
                # Scale lengths by the grid ratio so the GL field renders the
                # CPU field's composition at higher fidelity.
                self._px_scale = gw / self.CPU_GRID[0]
                # keep the scaled base as FLOAT — the weave diffuse trim
                # rounds once, at the end, so it bites the same fraction at
                # every quality tier
                self._base_diffuse = pf.diffuse * self._px_scale
                pf.diffuse = max(1, round(self._base_diffuse))
                return pf
            except Exception as e:                   # noqa: BLE001 — §6.4
                msg = f"GPU physarum unavailable, running on CPU: {e}"
                host.hud.toasts.flash(msg[:80], AMBER)
                print(msg)
        self.grid = self._grid or self.CPU_GRID
        self.n = self._n or self.CPU_N
        gw, gh = self.grid
        self.engine = "cpu"
        self._px_scale = 1.0
        pf = PhysarumField(n=self.n, gw=gw, gh=gh, seed=self.seed)
        self._base_diffuse = float(pf.diffuse)
        return pf

    def stop(self):
        """Release the GPU field if that is what booted; idempotent."""
        if self.pf is not None:
            self.pf.release()      # the composer's objects die with the ctx
        self.pf = None
        self.mat = None
        self._glout = None
        self._glout_key = None

    def on_resize(self, w, h):
        pass                     # everything derives from host.res per frame

    def configure_ui(self, ui):
        """Seed this mode's ph_* attrs on the shared UI state without
        clobbering anything a look already applied (DitherGirl's pattern)."""
        for k, v in self._UI_DEFAULTS.items():
            if not hasattr(ui, k):
                setattr(ui, k, v)

    # ----- panel / commands -----
    def panel_spec(self):
        pts = list(POINT_NAMES)
        return [
            Section("TEMPLATES", [PresetList()]),
            Section("SOURCE", [
                Cycle("matte", "ph_matte_idx", list(MATTES), save_key="matte",
                      status="matte {}",
                      tip="How the camera finds you: your whole body, only "
                          "what moves, edges, or the brightest parts."),
                Slider("Food", "ph_food", 0.0, 1.5, save_key="food",
                       tip="How strongly the footage's light pulls the mold. "
                           "High: the network chases whatever is bright."),
                Toggle("Video bg", "ph_video_bg", save_key="video_bg"),
                # only acts while Video bg is on (both engines gate the
                # composite on ph_video_bg) — hidden otherwise
                # (panelspec.visible; magic-over-control, 2026-08-24)
                Slider("Vid mix", "ph_video_mix", 0.0, 1.0, save_key="video_mix",
                       apply="keep",
                       show_when=lambda s: bool(getattr(s, "ph_video_bg", False)),
                       tip="How visible the raw camera footage is under the veins."),
                Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                      key="res", save=False, nudge=False,
                      tip="The window / projector resolution."),
                Cycle("quality", "ph_quality_idx", list(self.QUALITY_NAMES),
                      save=False, nudge=False,
                      tip="How finely the mold itself is simulated. Higher "
                          "keeps veins crisp on a big projector, and costs "
                          "speed. Switching regrows the field in seconds."),
            ]),
            Section("MOLD", [
                Cycle("body", "ph_point_fg_idx", pts, save_key="point_fg",
                      status="body {}",
                      tip="How the mold behaves ON you — where the camera "
                          "sees you, it grows in this style."),
                Cycle("field", "ph_point_bg_idx", pts, save_key="point_bg",
                      status="field {}",
                      tip="How the mold behaves in the rest of the room, "
                          "away from you."),
                Slider("Tempo", "ph_gain", 0.4, 2.5, save_key="gain",
                       tip="Global speed — scales every agent's stride and reach."),
                Slider("Decay", "ph_decay", 0.80, 0.99, save_key="decay",
                       tip="How long trails persist. High: durable veins. "
                           "Low: nervous, fast-forgetting lace."),
                Slider("Weave", "ph_weave", 0.0, 1.0, save_key="weave",
                       tip="Fine webbing. Low: a few bold canals. High: a "
                           "dense net of thin threads and crossings."),
                Slider("Evolve", "ph_evolve", 0.0, 1.0, save_key="evolve",
                       tip="The mold rearranges itself over time, even when "
                           "nothing moves. Zero holds one structure."),
                Slider("React", "ph_react", 0.0, 1.0, save_key="react",
                       tip="How hard your movement carves it. High: motion "
                           "pours mold into the path you sweep."),
            ]),
            Section("LOOK", [
                Cycle("color", "ph_palette_idx", list(PALETTES_PH), gap=4,
                      save_key="palette", status="{}",
                      tip="The color the veins glow in. 'video' lights them "
                          "with the camera's own colors."),
                Slider("Exposure", "ph_exposure", 0.5, 8.0, save_key="exposure",
                       tip="Brightness curve on the trail. High burns the "
                           "veins white; low keeps only the trunk lines."),
                Slider("Grain", "ph_grain", 0.0, 1.5, save_key="grain",
                       tip="This frame's raw agent dust over the smooth "
                           "trail. Zero is airbrushed; high is sandstorm."),
                # the orbit and the bass rake ride under this one control:
                # they are the instrument breathing, not a knob of their own,
                # and depth 0 switches them off with everything else
                Slider("Depth", "ph_depth", 0.0, 1.0, save_key="depth",
                       tip="Lights the veins as raised tubes. Zero is flat "
                           "glow; high carves every trunk out of the dark."),
            ]),
        ]

    def cast_random(self):
        """Z: decisive jump to a coherent randomized regime — random point
        pairing plus look-space params drawn from the validated regime
        machinery, landed with a forced regime hop and a melt surge so the
        old composition visibly dissolves into the new one. A slot machine
        that always pays: every draw comes from bands the builtin looks and
        REGIMES already live in, never uniform noise over raw parameter
        space (magic and play over control)."""
        ui = self.host.ui
        rng = self._cast_rng
        pts = self.points
        cur = (int(getattr(ui, "ph_point_bg_idx", 0)) % len(pts),
               int(getattr(ui, "ph_point_fg_idx", 2)) % len(pts))
        while True:
            pair = (int(rng.integers(len(pts))), int(rng.integers(len(pts))))
            if pair != cur and pair[0] != pair[1]:
                break
        ui.ph_point_bg_idx, ui.ph_point_fg_idx = pair
        # validated bands: the envelope the builtin looks span, slightly
        # widened — never a degenerate corner (weave/evolve floors keep the
        # picture woven and moving; decay/gain bands keep it legible)
        ui.ph_weave = round(float(rng.uniform(0.35, 0.90)), 2)
        ui.ph_evolve = round(float(rng.uniform(0.45, 0.90)), 2)
        ui.ph_react = round(float(rng.uniform(0.40, 0.90)), 2)
        ui.ph_decay = round(float(rng.uniform(0.88, 0.96)), 3)
        ui.ph_gain = round(float(rng.uniform(0.80, 1.60)), 2)
        ui.ph_palette_idx = int(rng.integers(len(self.palettes)))
        # decisive landing: force a regime hop now + a melt surge
        self._reg_from = self._reg_to
        others = [n for n in REGIME_NAMES if n != self._reg_to]
        self._reg_to = others[int(rng.integers(len(others)))]
        # never flat: a cast is meant to land somewhere visibly new, and the
        # relief is part of every look now (0.4 is already where the blob
        # looks gain anatomy). Drawn last, which keeps the FIRST cast's
        # draws as they were before depth existed; every later cast shares
        # this generator, so its draws come from a shifted part of the stream.
        ui.ph_depth = round(float(rng.uniform(0.4, 1.0)), 2)
        self._reg_t = 0.0
        self._reg_dwell = float(self._reg_rng.uniform(*REGIME_DWELL))
        self._melt_pulse = 1.0
        return pair

    def commands(self):
        """X swaps body/field points; B pours agents onto the subject;
        W ripples the whole organism outward; Z casts a random regime; H
        toggles the relief flat and back.
        Burst/wave land at the matte's bright centroid, resolved on the next
        step (commands run between frames, and the shell owns the mouse)."""
        ui, toasts = self.host.ui, self.host.hud.toasts
        pts = self.points

        def _swap():
            ui.ph_point_bg_idx, ui.ph_point_fg_idx = (ui.ph_point_fg_idx,
                                                      ui.ph_point_bg_idx)
            toasts.flash("SWAP  body %s / field %s"
                         % (pts[ui.ph_point_fg_idx % len(pts)],
                            pts[ui.ph_point_bg_idx % len(pts)]))

        def _burst():
            self._burst_pending = True
            toasts.flash("BURST")

        def _wave():
            self._wave_pending = True
            toasts.flash("WAVE")

        def _random():
            bg_i, fg_i = self.cast_random()
            toasts.flash("RANDOM  body %s / field %s"
                         % (pts[fg_i], pts[bg_i]))

        def _depth():
            # h ("height"): flat against the last non-zero depth. A toggle,
            # not a scene change, so it stays out of AUTO_RELEASE_KEYS.
            cur = float(getattr(ui, "ph_depth", DEPTH_DEFAULT))
            if cur > 0.0:
                self._depth_last = cur
                ui.ph_depth = 0.0
                toasts.flash("FLAT")
            else:
                ui.ph_depth = self._depth_last
                toasts.flash("DEPTH %.1f" % ui.ph_depth)
        return {"physarum.swap": Command("physarum.swap",
                                         "Swap body/field points", "x", _swap),
                "physarum.burst": Command("physarum.burst",
                                          "Spawn burst on the subject", "b", _burst),
                "physarum.wave": Command("physarum.wave",
                                         "Radial wave", "w", _wave),
                "physarum.random": Command("physarum.random",
                                           "Random regime (always pays)", "z",
                                           _random),
                "physarum.depth": Command("physarum.depth",
                                          "Depth on/flat", "h", _depth)}

    def safe_look(self):
        return "veinwork"

    @staticmethod
    def signal_dither_rows(out_h):
        """SIGNAL-rack dither working rows for this mode: 1/6 of the output
        height (a ~6-px cell at any resolution) instead of the rack's fixed
        72 rows, which turn the mold's smooth veins into boulder-sized grain
        at 1080p+. The floor keeps the cell look at small windows. Expressed
        in rows so a GPU dither pass can mirror it as a quantized-UV cell."""
        return max(96, out_h // 6)

    def status_tail(self, cam_name):
        return f"{self.engine or 'cpu'} {_fmt_agents(self.n)}  cam {cam_name[:16]}"

    def _ui(self, attr, default):
        """Read a live value defensively — step() can run before the shell
        builds the shared UI state (the soak/test path)."""
        ui = self.host.ui if self.host is not None else None
        return getattr(ui, attr, default) if ui is not None else default

    # ----- per-frame -----
    def step(self, frame_bgr, audio_levels, dt):
        """Matte + luma from the frame, one sim frame, colorize the trail.

        The frame is already mirrored by the shell and never None; all float
        work happens at the sim grid, with ONE upscale to host.res at the end.
        """
        ui = self.host.ui if self.host is not None else None

        # quality tier switch: rebuild the GL field at the new sizing (the
        # trail regrows in a couple of seconds; a fixed grid/n override and
        # the CPU fallback both ignore the cycle — no headroom there)
        want_q = self.QUALITY_NAMES[int(self._ui("ph_quality_idx", 0))
                                    % len(self.QUALITY_NAMES)]
        if (want_q != self._quality and self.engine == "gl"
                and self._grid is None and self._n is None):
            self._quality = want_q
            self.pf.release()
            self.pf = self._build_field(self.host)
            if self.engine == "gl":
                g_w, g_h = self.grid
                self.host.hud.toasts.flash(
                    f"quality {want_q}  {g_w}x{g_h} {_fmt_agents(self.n)}")
        elif want_q != self._quality:
            self._quality = want_q      # remember; applies if GL boots later

        pf = self.pf
        gw, gh = self.grid
        rw, rh = self.host.res

        if ui is not None and getattr(ui, "ph_matte_idx", None) is not None:
            want = self.mattes[ui.ph_matte_idx % len(self.mattes)]
            if want != self.matte_kind:
                mat, self.matte_kind = select_matte(
                    ui, "ph_matte_idx", self.mattes, self.matte_kind,
                    self.host.hud.toasts)
                if mat is not None:
                    self.mat = mat

        pts = self.points
        pf.point_fg = pts[int(self._ui("ph_point_fg_idx", 2)) % len(pts)]
        pf.point_bg = pts[int(self._ui("ph_point_bg_idx", 0)) % len(pts)]
        decay = float(self._ui("ph_decay", 0.94))
        food = float(self._ui("ph_food", 0.35))
        pf.grain = float(self._ui("ph_grain", 0.5))
        gain = float(self._ui("ph_gain", 1.0))
        exposure = float(self._ui("ph_exposure", 3.5))
        weave = min(max(float(self._ui("ph_weave", 0.6)), 0.0), 1.0)
        evolve = min(max(float(self._ui("ph_evolve", 0.5)), 0.0), 1.0)
        react = min(max(float(self._ui("ph_react", 0.7)), 0.0), 1.0)

        # audio rides on top of the sliders for this frame only: bass pulses
        # the exposure, treble quickens the mold
        if audio_levels is not None:
            sens = float(self._ui("sens", 1.0))
            exposure *= 1.0 + 1.6 * sens * audio_levels["bass"]
            gain *= 1.0 + 0.8 * sens * audio_levels["treble"]

        # Perceptual slider mapping (owner playtest 2026-08-24: 0.3 on every
        # MOLD slider read as nothing): the sliders' bottom third has to
        # already be clearly audible, so the levers ride value**0.6 —
        # 0.3 -> 0.49, 0.6 -> 0.74, 1.0 -> 1.0 (wild) — instead of linear.
        e = evolve ** FEEL_CURVE

        # evolve: the mold hops between REGIMES — decisive crossfaded
        # transitions (REGIME_FADE seconds) between distinct growth
        # characters, dwelling REGIME_DWELL seconds in each, order
        # randomized — plus light incommensurate-sine breathing between
        # hops and the staleness melt below. The old gentle sine walks
        # alone read as a lava lamp: plenty of pixel drift, but the
        # network's CHARACTER (vein width, mesh scale) never changed.
        # Our own design: this mode's own parameters, no external tables.
        self._t += dt
        self._orbit_t += dt
        weave_base = weave
        extra_jitter = 0.0
        reg_fade_pulse = 0.0
        if evolve > 0:
            tau = self._t * (2.0 * np.pi)
            self._reg_t += dt
            if self._reg_t >= self._reg_dwell:
                self._reg_from = self._reg_to
                others = [n for n in REGIME_NAMES if n != self._reg_to]
                self._reg_to = others[int(self._reg_rng.integers(len(others)))]
                self._reg_t = 0.0
                self._reg_dwell = float(self._reg_rng.uniform(*REGIME_DWELL))
            f = min(self._reg_t / REGIME_FADE, 1.0)
            fade = f * f * (3.0 - 2.0 * f)               # smoothstep
            reg_fade_pulse = 4.0 * fade * (1.0 - fade)   # peaks mid-crossfade
            ra = _regime_mults(self._reg_from)
            rb = _regime_mults(self._reg_to)
            rm = ra + (rb - ra) * np.float32(fade)
            # perceptual amp: multipliers pulled toward 1 at low evolve
            rm[:7] = 1.0 + (rm[:7] - 1.0) * e
            extra_jitter = float(rm[7]) * e
            # breathing between hops (small — the regimes are the movers)
            gain *= float(rm[5]) * (1.0 + 0.18 * e * np.sin(tau / 19.0))
            food *= float(rm[6]) * (1.0 + 0.25 * e * np.sin(tau / 23.0 + 4.2))
            decay = min(max(decay + 0.015 * e * np.sin(tau / 29.0 + 2.1),
                            0.80), 0.995)
            weave = min(max(weave + 0.20 * e * np.sin(tau / 31.0 + 1.0),
                            0.0), 1.0)
            pf.mod_sense = float(rm[0]) * (1.0 + 0.50 * e * np.sin(tau / 17.0 + 0.7))
            pf.mod_turn = float(rm[1]) * (1.0 + 0.35 * e * np.sin(tau / 27.0 + 3.4))
            pf.mod_spread = float(rm[2]) * (1.0 + 0.30 * e * np.sin(tau / 13.0 + 5.5))
            pf.mod_step = float(rm[3])
            pf.mod_deposit = float(rm[4])
        else:
            pf.mod_sense = pf.mod_turn = pf.mod_spread = 1.0
            pf.mod_step = pf.mod_deposit = 1.0

        # evolve's spatial half. The regime table above moves the WHOLE frame
        # together, which is why the picture could churn constantly and still
        # read as one uniform texture. The mosaic partitions the grid into
        # drifting zones with their own multipliers and hard boundaries, so
        # several morphologies coexist and abut. Time axis and space axis,
        # both under the one knob: evolve is "how unlike itself it gets".
        pf.mosaic = 0.95 * e

        # weave: one knob onto the engine's anti-thoroughfare levers, tuned
        # offline (junction density several-x between 0 and 1 on a static
        # scene while veins stay coherent). 0 is the legacy bold-canal
        # behavior; tops raised 2026-08 so weave 1.0 is properly wild.
        # The diffuse trim rides the SLIDER value, not the evolve-modulated
        # one — an integer blur radius popping mid-oscillation would beat
        # visibly — and is computed in float from the unrounded base so the
        # trim bites identically at every quality tier (rounding the base
        # first made 'quality' veins relatively thinner than 'perform').
        w = weave ** FEEL_CURVE
        # sat is a MULTIPLE of the trail's own bright end. It must sit ABOVE
        # that end: capping at 0.22x p95 (the old 0.30*w) compressed the whole
        # field into a 3-unit band, so agents inside the network were steering
        # on noise and every point converged on the same mesh. Above 1.0 only
        # the fat canals compress, which is the anti-thoroughfare effect that
        # was actually wanted, and weave now tightens the cap toward the
        # network instead of blinding it.
        # The bottom of the knob stays the legacy engine: at weave 0 the cap is
        # off entirely. Above 0 it lands ABOVE the trail's bright end and
        # tightens toward it, so only the fat canals compress. (The old
        # mapping put it at ~0.2x the bright end, which flattened the whole
        # sensed field into a few units and left agents inside the network
        # steering on noise. It also had the cap jump from "off" to "crushing"
        # across weave 0; now it goes from off to nearly-inert.)
        pf.sat = 0.0 if weave <= 0.0 else 1.70 - 0.85 * w
        # jitter and reseed used to be weave's main levers and were the two
        # things preventing any structure at all: heading decorrelated in
        # 0.37 s and the entire population recycled once a second, so the
        # picture's statistics were pinned to the reseed distribution rather
        # than to anything self-organised. Both are now seasoning, not engine.
        pf.jitter = 0.14 * w + extra_jitter
        pf.hetero = w
        pf.reseed_frac = 0.004 + 0.006 * w * w
        # weave's real job. Three populations sense each other through a signed
        # matrix; `cross` is how hard they push. At 0 they are one organism and
        # you get the classic single-species transport network. As it rises the
        # populations carve exclusion membranes into each other and the picture
        # stops being one texture everywhere — territories, fronts, dark walls.
        # This is the lever that makes weave visible in under a second, which
        # the old satcap/jitter/reseed bundle never managed.
        # 0.68, not 1.0. The coefficient is a LOOK decision, not a spare
        # scale factor: an audit measured the default look's vein contrast
        # (p99/median) at 166 with 0.68 and 22.2 with 1.0 — the bottom edge of
        # the 20-50x band docs/ALIVENESS.md uses to define the original bug,
        # with the frame 80% brighter. Side by side, 0.68 is bold trunks
        # against real black with capillaries between them; 1.0 is a uniform
        # bright bundle, which is the complaint this project started from.
        # tests/test_physarum_alive.py pins the contrast so it cannot drift
        # again on the way to satisfying some other floor.
        pf.cross = 0.68 * w
        pf.sharpen = 0.18 * w
        pf.diffuse = max(1, round(self._base_diffuse
                                  * (1.0 - 0.45 * weave_base ** FEEL_CURVE)))

        pf.decay = decay
        pf.food = food
        # gain multiplies sense + step (both in grid px) in either engine, so
        # it doubles as the length-unit conversion onto the GL grid
        pf.gain = gain * self._px_scale
        pf.exposure = exposure
        # depth: the relief's amount, and the key light orbiting on its own
        # clock (reset in start(), so it is upper-left on every entry to the
        # mode, not only the first) with the bass
        # raking it lower. The light is authored in screen space and flipped
        # into the grid space the shader reads. Both engines take the same
        # two values; depth 0 is the flat picture on either.
        pf.depth = depth_feel(self._ui("ph_depth", DEPTH_DEFAULT))
        bass = float(audio_levels["bass"]) if audio_levels is not None else 0.0
        pf.light = screen_to_grid(
            depth_light(self._orbit_t, bass, float(self._ui("sens", 1.0))))

        small = cv2.resize(frame_bgr, (MATTE_W, MATTE_H))
        m = cv2.resize(self.mat.compute(small), (gw, gh))
        gray = cv2.resize(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (gw, gh))

        # react: motion carves. Frame-differenced luma builds a lingering
        # motion-energy map; it is poured into the food channel (sensors
        # chase it), respawn traffic is steered onto it (trails linger where
        # you swept), and a strong gesture fires one theatrical burst at its
        # centroid — episodic, so the pour reads as a cast spell instead of
        # continuously draining the rest of the organism.
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._motion = np.zeros_like(gray)
            self._cy, self._cx = np.mgrid[0:gh, 0:gw]
            # half-res coordinate grids for the evolve phantom (see below)
            self._hcy, self._hcx = np.mgrid[0:(gh + 1) // 2, 0:(gw + 1) // 2]
            self._hcy = (self._hcy * 2).astype(np.float32)
            self._hcx = (self._hcx * 2).astype(np.float32)
        # deadband under the diff: real cameras hold ~0.01-0.03 of per-pixel
        # sensor noise, which would otherwise read as permanent full-frame
        # "motion" and fire gather pulses at the frame centroid forever
        motion = np.maximum(np.abs(gray - self._prev_gray) - np.float32(0.04),
                            np.float32(0.0))
        self._prev_gray = gray
        np.maximum(self._motion * np.float32(0.975),
                   np.minimum(motion * np.float32(3.0), np.float32(1.0)),
                   out=self._motion)
        # evolve's phantom: a slow-wandering invisible food blob the mold
        # chases. Parameter breathing alone cannot re-knit the network — the
        # laid trail is an attractor and the layout locks onto the scene's
        # light — but a migrating attractor drags veins across the frame and
        # the field re-organizes behind it (~30-40 s circuits). Computed at
        # half res (one exp on a quarter of the cells) and upsampled.
        if evolve > 0:
            tau_p = self._t * (2.0 * np.pi)
            fx = gw * (0.5 + 0.38 * np.sin(tau_p / 37.0 + 0.9))
            fy = gh * (0.5 + 0.38 * np.sin(tau_p / 41.0 + 2.6))
            sig = 0.16 * min(gw, gh)
            blob = np.exp(((self._hcx - fx) ** 2 + (self._hcy - fy) ** 2)
                          * np.float32(-1.0 / (2.0 * sig * sig)))
            # perceptual amp, same mapping as the structural movers above.
            # 0.7: strong enough to drag veins across the frame, soft enough
            # that the drag reads as reorganization, not a global slosh —
            # the reference profile is calm at 2 s and big at 30 s
            gray = gray + (0.7 * e) * cv2.resize(blob, (gw, gh))

        # staleness tracker: EMA mean/variance of the (blurred) picture at
        # half res. Tracked ALWAYS (cheap), applied as melt only when evolve
        # is up — so the moment evolve rises it already knows what is stale.
        # Grain dust is blurred out first or it would hide stasis.
        if self._last_lum is not None:
            # input is the engine's lum_sample(): the full pre-relief
            # luminance (CPU) or the GL stats subsample — either resizes to
            # the half grid
            lh = cv2.resize(self._last_lum, ((gw + 1) // 2, (gh + 1) // 2),
                            interpolation=cv2.INTER_AREA)
            lh = cv2.blur(lh, (5, 5))
            if self._ema is None or self._ema.shape != lh.shape:
                self._ema = lh.copy()
                # warm start above the stale floor: nothing reads stale
                # until the estimate has genuinely settled low (~3 s)
                self._var = np.full_like(lh, 10.0 * MELT_VAR)
            d = lh - self._ema
            self._ema += np.float32(0.06) * d
            self._var = self._var * np.float32(0.97) + np.float32(0.03) * d * d

        self._burst_cool = max(getattr(self, "_burst_cool", 0.0) - dt, 0.0)
        self._melt_pulse = max(0.0, self._melt_pulse - dt / 2.5)
        r = react ** FEEL_CURVE      # perceptual mapping, same as the others
        keep_pos = None
        if react > 0:
            gray = gray + (5.0 * r) * self._motion
            # swept paths linger: motion history becomes a per-pixel decay
            # boost, so the veins you carve stay painted for a few seconds
            keep_pos = r * self._motion
            energy = float(motion.mean())
            if energy > 8e-4 and self._burst_cool <= 0.0:
                tot = float(motion.sum())
                mx = float((self._cx * motion).sum() / tot)
                my = float((self._cy * motion).sum() / tot)
                pf.gather(mx, my,
                          frac=min(0.9, r * (0.4 + 60.0 * energy)),
                          radius=40.0 * self._px_scale)
                self._burst_cool = 0.4

        # melt: bright structure whose picture stopped changing dissolves —
        # the signed keep map's negative side. Strength rides evolve, and
        # surges mid-crossfade and right after a random cast, so a regime
        # change visibly re-fluidizes the old composition instead of
        # painting the new regime under it.
        melt = None
        if evolve > 0 and self._var is not None:
            stale = (np.clip(1.0 - self._var * np.float32(1.0 / MELT_VAR),
                             0.0, 1.0)
                     * np.clip((self._ema - np.float32(0.25)) * np.float32(5.0),
                               0.0, 1.0))
            m_gain = MELT_GAIN * e * (1.0 + 1.5 * reg_fade_pulse
                                      + 2.0 * self._melt_pulse)
            melt = cv2.resize(stale * np.float32(m_gain), (gw, gh),
                              interpolation=cv2.INTER_LINEAR)

        keep = None
        if keep_pos is not None or melt is not None:
            keep = keep_pos if keep_pos is not None \
                else np.zeros((gh, gw), np.float32)
            if melt is not None:
                keep = keep - melt
            keep = np.clip(keep, -1.0, 1.0)

        if self._burst_pending or self._wave_pending:
            # land the impulse on the lit subject: weighted centroid of
            # matte*luma, falling back to frame center on an empty matte
            w = m * np.clip(gray, 0.05, 1.0)
            tot = float(w.sum())
            if tot > 1e-3:
                px_c = float((self._cx * w).sum() / tot)
                py_c = float((self._cy * w).sum() / tot)
            else:
                px_c, py_c = gw / 2.0, gh / 2.0
            if self._burst_pending:
                # the burst's footprint is sized in CPU-grid pixels too
                pf.spawn_burst(px_c, py_c, radius=6.0 * self._px_scale)
            if self._wave_pending:
                pf.wave(px_c, py_c)
            self._burst_pending = self._wave_pending = False

        pf.update(m, gray, keep)

        pal = self.palettes[int(self._ui("ph_palette_idx", 0)) % len(self.palettes)]
        # SIGNAL on the GPU (DESIGN.md §2.4; PR #26's measured port list):
        # with the GL engine and the rack ON, tonemap + colorize + upscale +
        # video composite + the whole rack run as fragment passes on the
        # field's own context — the trail never round-trips through numpy,
        # and the ONE readback is the final composed uint8 frame.
        # signal_done tells the shell its CPU rack already happened here.
        self.signal_done = False
        if (self.engine == "gl" and self._rack_gl_ok and self.host is not None
                and bool(self._ui("glitch", False))):
            out = self._step_gpu_rack(small, frame_bgr, pal)
            if out is not None:
                self.signal_done = True
                # staleness input without leaving the GPU path: the stats
                # subsample is already read back every frame
                samp = pf.lum_sample()
                if samp is not None:
                    self._last_lum = samp
                return out

        lum = pf.luminance()
        # staleness tracker input, next frame: the engine's flat sample, the
        # same one the GPU-rack path above feeds. Not `lum` — with depth > 0
        # that is relief-lit, and the lighting (dark flanks, the orbiting
        # light's variance) would read as the organism changing and weaken
        # the melt, so the same organism would melt differently depending on
        # depth and on whether the rack is on.
        # lum_sample() is None only on an empty trail, where `lum` is flat
        # zeros on both engines (no relief is applied to an empty picture)
        samp = pf.lum_sample()
        self._last_lum = samp if samp is not None else lum
        if pal == "video":
            # veins lit by the footage's own color — the mold as a lampshade
            color = cv2.cvtColor(cv2.resize(small, (gw, gh)),
                                 cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out_small = (lum[..., None] * (0.25 + 0.75 * color) * 255.0).astype(np.uint8)
        else:
            out_small = _colorize(lum, pal)
        out = cv2.resize(out_small, (rw, rh), interpolation=cv2.INTER_LINEAR)

        if bool(self._ui("ph_video_bg", False)):
            out = composite_video_bg(out, frame_bgr,
                                     float(self._ui("ph_video_mix", 0.5)))
        return np.ascontiguousarray(out)

    # ----- SIGNAL on the GPU -----
    def _ensure_glout(self, rw, rh):
        """The GL output composer + rack, rebuilt when the field, grid, or
        output resolution changes (all of them size its textures). Must be
        called with the field's context bound."""
        key = (id(self.pf), self.grid, (rw, rh))
        if self._glout is not None and self._glout_key != key:
            if self._glout.ctx is self.pf.ctx:
                self._glout.release()   # same live ctx: free the old textures
            self._glout = None
        if self._glout is None:
            gw, gh = self.grid
            self._glout = PhysarumOutGL(self.pf.ctx, gw, gh, rw, rh)
            self._glout_key = key
        return self._glout

    def _step_gpu_rack(self, small, frame_bgr, pal):
        """One frame of the ported output path: tonemap into the field's
        luminance texture, colorize + upscale + video composite, then the
        SIGNAL rack, all as fragment passes — one uint8 RGB readback at
        output res. Uses the shell's own CircuitBent (host.cb) for the
        rack's stochastic plan, so toggling engines stays continuous.

        Returns None on any GL failure and disables the path — the CPU
        rack takes over, toasted in amber (DESIGN.md §6.4: the show still
        has to start)."""
        host = self.host
        try:
            rw, rh = host.res
            cb = getattr(host, "cb", None)
            if cb is None:
                cb = CircuitBent(seed=getattr(host, "seed", 0))
                host.cb = cb
            sync_signal(cb, host.ui, self, rh)
            video_small = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                           if pal == "video" else None)
            cam, mix = None, 0.0
            if bool(self._ui("ph_video_bg", False)):
                mix = float(self._ui("ph_video_mix", 0.5))
                if mix > 0.0:
                    cam = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pf = self.pf
            with pf.ctx:
                glout = self._ensure_glout(rw, rh)
                pf.luminance_into_tex()
                lut = None if pal == "video" else _palette_lut(pal)
                src = glout.compose(pf.tex_lum, lut, video_small, cam, mix)
                glout.rack.run(cb, cb.plan(rh, rw), src)
                return glout.rack.read()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:               # noqa: BLE001 — §6.4
            self._rack_gl_ok = False
            self._glout = None
            msg = f"GPU rack unavailable, rack running on CPU: {e}"
            if host is not None:
                host.hud.toasts.flash(msg[:80], AMBER)
            print(msg)
            return None
