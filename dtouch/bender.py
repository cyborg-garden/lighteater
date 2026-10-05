"""Circuit Bender's looks, ladders and clocks: everything about the mode that is
a decision rather than a pixel operation, so it tests without a camera.

Circuit Bender is live JPEG databending of the camera: each frame is given a
sensor bend (dtouch.bender_sensor), encoded as a real baseline JPEG, bent
(dtouch.bender_jpeg), decoded, and shown. The mode is dtouch/modes/bender.py.

This module is the desktop's source of truth for the tables the browser page
runs on: dtouch.bender_looks exports them to
dtouch/shaders/bender/bender_looks.json, which cyborg-garden-site vendors
(it ports looks.js from that file instead of retyping it).

Distinct from dtouch/circuit_bent.py: that is the SIGNAL rack's stochastic
signal glitch on the frame; this is a whole mode, and nothing here comes from
it or is meant to match it.

Inspired by CyberShot Cam by @lixofuturista / @cebolander
(github.com/cebola4444/cybershot-cam), a pocket camera that bends its own
JPEGs. That repository carries no licence, so neither its code nor its
numbers are used here.
"""
from __future__ import annotations

import math

from .bender_jpeg import EFFECT_TITLES as JPEG_TITLES
from .bender_jpeg import EFFECTS as JPEG_EFFECTS
from .bender_sensor import SENSOR_EFFECTS, SENSOR_TITLES

# Every bend E steps through: BENT CAM first (the landing look: what a bent
# camera looks like at a glance), then the five JPEG bends and their stack,
# then the three single sensor faults.
EFFECTS = (SENSOR_EFFECTS[0],) + JPEG_EFFECTS + SENSOR_EFFECTS[1:]
EFFECT_TITLES = {**JPEG_TITLES, **SENSOR_TITLES}

# One look per effect, in E's order. Look 1, BENT CAM, is where the mode
# lands (and where panic resets to): a green or magenta cast, slipped bands,
# some in the other cast, and posterised bursts that keep the outlines. Each
# look is an effect with its own amount and post, tuned by eye on a camera.
LOOKS = {
    "bent": dict(effect="bent", amount=0.65, split=0, sort="off", copy=0, long=0),
    "zigzag": dict(effect="zigzag", amount=0.5, split=0, sort="off", copy=0, long=0),
    "erosion": dict(effect="erosion", amount=0.75, split=0, sort="off", copy=0, long=0),
    "remap": dict(effect="remap", amount=0.5, split=0, sort="off", copy=0, long=0),
    "swap": dict(effect="swap", amount=0.5, split=0, sort="off", copy=0, long=0),
    "chroma": dict(effect="chroma", amount=0.35, split=4, sort="off", copy=0, long=0),
    "stack": dict(effect="stack", amount=0.6, split=0, sort="off", copy=0.5, long=0),
    "hclock": dict(effect="hclock", amount=0.5, split=0, sort="off", copy=0, long=0),
    "vclock": dict(effect="vclock", amount=0.5, split=0, sort="off", copy=0, long=0),
    "adc": dict(effect="adc", amount=0.5, split=0, sort="off", copy=0, long=0),
}
LOOK_NAMES = tuple(LOOKS)
SAFE_LOOK = "bent"

# Ladders the keys and the panel step through.
AMOUNT_LADDER = (0.25, 0.5, 0.75, 1)
AMOUNT_NUDGE = 0.05
LONG_LADDER = (0, 3, 5, 10)            # seconds: off / 3 s / 5 s / 10 s
SORT_LADDER = ("off", "rows", "columns")
SPLIT_LADDER = (0, 4, 10)              # working pixels
COPY_LADDER = (0, 0.5, 1)
COPY_NAMES = {0: "off", 0.5: "some", 1: "lots"}
# The sort's brightness band: the mids sort, shadows and highlights hold shape.
SORT_BAND = (60, 205)

# Encode quality (0..1, the browser's convertToBlob scale; cv2 takes x100):
# lower is blockier before any bend.
JPEG_Q = 0.8
# Copy runs step one 16x16 MCU of a 4:2:0 JPEG.
BLOCK = 16


def next_in(ladder, v):
    """The ladder's next rung above `v`, wrapping to the first."""
    return next((x for x in ladder if x > v), ladder[0])


def clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return min(1.0, max(0.0, v))


# Settle: a bend that makes the decoder lose its place decodes to a different
# picture on almost every camera frame; shown raw that is a strobe. DHT REMAP
# can slip and STACK carries it, so the display folds their frames into a
# short running average (the long exposure's pass, time constant in seconds).
# SCAN SWAP gets a very short one that keeps its bands saturated. The table
# bends are steady; the sensor bends drift on their own slow clock.
SETTLE_S = dict(bent=0, zigzag=0, erosion=0, remap=0.35, swap=0.08, chroma=0,
                stack=0.12, hclock=0, vclock=0, adc=0)


def stack_tau(effect, long):
    """The running average's time constant: half the long exposure, or the
    effect's settle, whichever is longer; 0 = show frames as they come."""
    return max(long / 2 if long > 0 else 0, SETTLE_S.get(effect, 0))


# ----- the working size -----

# The camera that inspired this mode saves 1024 x 768 JPEGs, so an 8x8 block
# is a small fraction of the frame. The working size follows: never more
# than the camera delivers, within that camera's pixel budget. A slow machine
# lowers the budget (Governor); the size is never below 64.
MAX_PIXELS = 1024 * 768
MIN_PIXELS = 320 * 240


def working_size(out_w, out_h, src_w=math.inf, src_h=math.inf, budget=MAX_PIXELS):
    """[w, h, up]: the working size for an `out_w` x `out_h` output, up being
    output pixels per working pixel. The browser passes CSS pixels; the
    desktop passes its output resolution."""
    out_w, out_h = max(1, out_w), max(1, out_h)
    aspect = out_w / out_h
    crop_w = src_h * aspect if src_w / src_h > aspect else src_w
    fit = math.sqrt(max(MIN_PIXELS, budget) * aspect)
    w = max(64, int(math.floor(min(out_w, crop_w, fit))))
    h = max(64, int(math.floor(w / aspect)))
    return w, h, out_w / w


# The budget governor: below MIN_RATE bends a second, or with the mode's own
# per-frame cost on the main thread above MAX_STEP_MS, for SLOW_S seconds
# straight, the pixel budget steps down by STEP_DOWN, never under MIN_PIXELS.
# It only steps down, so the picture never pumps between two sizes.
# On the desktop both signals are costs, not frame rates: `rate` is what the
# worker could sustain (1000 / its own bend time) and `step_ms` is measured
# inside the mode's step. The shell's loop also waits on the camera, and a
# dim-light camera at 8 frames a second would otherwise ratchet the picture
# down to the floor through either. (The browser's governor watches the
# achieved bend rate only.)
MIN_RATE = 15
MAX_STEP_MS = 12
SLOW_S = 3
STEP_DOWN = 0.7


class Governor:
    def __init__(self, budget=MAX_PIXELS):
        self.budget = budget
        self.slow = 0.0
        self.steps = 0

    def step(self, dt, rate, step_ms=0.0):
        """dt seconds of a running loop at `rate` bends a second, the mode's
        step costing `step_ms` on the main thread; True when the budget
        changed."""
        if not dt > 0:
            return False
        slow = rate < MIN_RATE or step_ms > MAX_STEP_MS
        self.slow = self.slow + min(dt, 1) if slow else 0.0
        if self.slow < SLOW_S or self.budget <= MIN_PIXELS:
            return False
        self.slow = 0.0
        self.budget = max(MIN_PIXELS, int(math.floor(self.budget * STEP_DOWN + 0.5)))
        self.steps += 1
        return True

    def step_down(self):
        """One notch down now, on hard evidence (a bend that never came
        back); True when the budget changed."""
        if self.budget <= MIN_PIXELS:
            return False
        self.slow = 0.0
        self.budget = max(MIN_PIXELS, int(math.floor(self.budget * STEP_DOWN + 0.5)))
        self.steps += 1
        return True


# ----- the cut clock -----

# SCAN SWAP and COPY are driven by a seed: a held seed reads as a glitch that
# stays put and moves with the picture. The seed changes every CUT_EVERY_S, at
# once on a sound onset (at most every ONSET_GAP_S), and the phase drifts the
# cuts slowly in between. SCAN SWAP runs three times the cuts (CUT_TEMPO).
CUT_EVERY_S = 1.8
ONSET_GAP_S = 0.3
DRIFT_PER_S = 0.015
CUT_TEMPO = {"swap": {"cut": 3, "drift": 1}}
_CALM = {"cut": 1, "drift": 1}


def cut_tempo(effect):
    return CUT_TEMPO.get(effect, _CALM)


class CutClock:
    def __init__(self, seed=1):
        self.seed = int(seed) & 0xFFFFFFFF
        self.phase = 0.0
        self.t = 0.0
        self.since_onset = math.inf
        self.cuts = 0

    def step(self, dt, onset=False, tempo=_CALM):
        """True when the seed changed."""
        dt = min(max(float(dt or 0), 0.0), 0.25)
        self.t += dt
        self.since_onset += dt
        self.phase = math.fmod(self.phase + dt * DRIFT_PER_S * tempo["drift"], 1.0)
        beat = onset and self.since_onset >= ONSET_GAP_S
        if beat:
            self.since_onset = 0.0
        if self.t >= CUT_EVERY_S / tempo["cut"] or beat:
            self.t = 0.0
            self.seed = (self.seed * 1664525 + 1013904223) & 0xFFFFFFFF
            self.cuts += 1
            return True
        return False


# The desktop's mic (dtouch.audio) reports levels, not onsets, so an onset is
# the bass rising ONSET_RISE above its own slow average (time constant
# ONSET_TAU_S); it re-arms once the bass falls back under the average. The
# browser's analyser reports onsets itself.
ONSET_RISE = 0.25
ONSET_TAU_S = 0.4


class Onset:
    def __init__(self):
        self.avg = 0.0
        self.armed = True

    def step(self, bass, dt):
        """True on the frame the bass jumps."""
        bass = clamp01(bass)
        hit = self.armed and bass - self.avg > ONSET_RISE
        if hit:
            self.armed = False
        elif bass <= self.avg:
            self.armed = True
        k = 1 - math.exp(-max(dt, 0.0) / ONSET_TAU_S)
        self.avg += (bass - self.avg) * k
        return hit


# Sensor bends take a new touch every TOUCH_CUTS cuts: between touches the
# fault drifts on its own instead of jumping on every cut.
TOUCH_CUTS = 4


def touch_seed(cuts):
    k = cuts // TOUCH_CUTS + 1
    return ((k * 2654435761) & 0xFFFFFFFF) ^ 0x5EED


# ----- the back-off -----

# A bent JPEG can fail to decode, or decode to black. The mode never shows
# that: it holds the last good frame, backs the amount off, and tries again.
# After CLEAN_AFTER failures in a row the next frame is not bent at all.
CLEAN_AFTER = 3
BACKOFF = 0.6
RECOVER = 0.05
FLOOR = 0.1


class Backoff:
    def __init__(self):
        self.scale = 1.0
        self.streak = 0
        self.fails = 0
        self.max_streak = 0
        self.cleans = 0

    def amount(self, a):
        if self.streak >= CLEAN_AFTER:
            return 0.0
        return clamp01(a) * self.scale

    def fail(self):
        self.fails += 1
        self.streak += 1
        self.max_streak = max(self.max_streak, self.streak)
        self.scale = max(FLOOR, self.scale * BACKOFF)

    def ok(self, clean=False):
        if clean:
            self.cleans += 1
        self.streak = 0
        self.scale = min(1.0, self.scale + RECOVER)


def dead_frame(bent, src):
    """A decoded frame is dead when it came back black or flat while the
    source was not. Stats are dicts of mean and variance of luma 0..255."""
    if not bent:
        return True
    black = bent["mean"] < 6 and src["mean"] > 24
    flat = bent["variance"] < 2 and src["variance"] > 30
    return black or flat


def luma_stats(rgb):
    """Mean and variance of Rec. 601 luma over RGB pixels (a thumbnail)."""
    import numpy as np
    p = rgb.reshape(-1, rgb.shape[-1]).astype(np.float64)
    y = 0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2]
    if y.size == 0:
        return {"mean": 0.0, "variance": 0.0}
    m = float(y.mean())
    return {"mean": m, "variance": float((y * y).mean() - m * m)}


# ----- sound and auto -----

SOUND_LIFT = 0.3
AUTO_SWAY = 0.2
AUTO_SWAY_S = 14


def live_amount(base, bass=0.0, sens=1.0, auto=False, t=0.0):
    """Sound breathes the amount (bass adds up to SOUND_LIFT x sens on top);
    auto sways it by AUTO_SWAY over AUTO_SWAY_S. Never more than 1."""
    a = clamp01(base)
    if auto:
        a *= 1 - AUTO_SWAY / 2 + (AUTO_SWAY / 2) * math.sin(2 * math.pi * t / AUTO_SWAY_S)
    a += SOUND_LIFT * sens * clamp01(bass) * (0.4 + 0.6 * a)
    return clamp01(a)


def recast_pick(name, rng):
    """One autopilot re-cast: the picked look with a fresh amount and post,
    mostly one clear effect, some with a sort, copies, a split or a long
    exposure. `rng()` returns [0, 1)."""
    look = LOOKS.get(name, LOOKS[SAFE_LOOK])

    def any_of(seq):
        return seq[int(rng() * len(seq))]
    return dict(
        look=name if name in LOOKS else SAFE_LOOK,
        effect=look["effect"],
        amount=round((0.3 + 0.6 * rng()) * 100) / 100,
        split=any_of(SPLIT_LADDER[1:]) if rng() < 0.4 else 0,
        sort=any_of(SORT_LADDER[1:]) if rng() < 0.25 else "off",
        copy=any_of(COPY_LADDER[1:]) if rng() < 0.35 else 0,
        long=any_of((3, 5)) if rng() < 0.2 else 0,
    )
