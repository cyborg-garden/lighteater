"""Circuit Bender's sensor bends: what a circuit-bent CCD or CMOS camera does
to a picture BEFORE it is ever a JPEG. The JPEG bends (dtouch.bender_jpeg)
edit the file; these model the sensor and its readout, and the bent frame
then goes through the same JPEG encode and decode as every other frame.

The model, per frame, on the working-size RGB pixels:
  1. mosaic: keep one colour per pixel on an RGGB Bayer grid, the raw values
     a colour sensor actually reads
  2. the bend, on the raw samples in readout order, where a bender's wire
     lands:
       H CLOCK  line timing: full-width bands slide sideways and read on
                into the next line, some lean into diagonals, and the whole
                frame creeps with a slow clock drift
       V CLOCK  frame timing: repeated lines stretch down the frame, the
                frame start slips (a roll, with a dark blanking bar at the
                seam), highlights smear down their columns
       ADC BITS bridged and stuck data lines fold brightness into a few
                levels that still follow the picture, in bands where an
                intermittent contact makes and breaks
  3. demosaic: bilinear. A bend that moves samples by an odd number of sites
     puts red where the interpolation expects green: the magenta and green
     casts and rainbow edges fall out of the Bayer grid, as in a real bent
     camera.

BENT CAM, the landing look, is the three things every bent-camera account
shows at once: a pink or green cast, slipped bands, posterised bursts that
keep the outlines. Never 8x8 squares: those are a file's, not a sensor's.

THERMAL follows it, one fault taken all the way, tuned by eye against
bent-camera footage: wrapped data bits, so the brightness runs through a
repeating palette and gradients come back as many thin rainbow rings round
every light, shadows near-black, grain boiling, edges rimmed. It works on
RGB, after the demosaic.

Everything moves with `t` (seconds) through slow value noise: the fault
drifts like a finger on a circuit board. The seed picks which fault.

Port of cyborg-garden-site src/scripts/bender/sensor.js, held byte-identical
to it by dtouch/shaders/bender/bender_goldens.json; vectorised with numpy,
with the browser's integer maths (>> on non-negative ints, Uint8 truncation,
Math.round as half up) kept exactly.

Sources for the artefact classes:
  Phillip Stearns, the DCP series and the Fujifilm FinePix S9000 teardown,
    phillipstearns.wordpress.com
  Noystoise, "Glitch cameras and how", noystoise.com, 2023
  Gijs Gieskes, VU cam ("pins that give the whole image a nice color, pink
    or green"), gieskes.nl
"""
from __future__ import annotations

import math

import numpy as np

SENSOR_EFFECTS = ("bent", "thermal", "hclock", "vclock", "adc")
SENSOR_TITLES = {"bent": "bent cam", "thermal": "thermal", "hclock": "h clock",
                 "vclock": "v clock", "adc": "adc bits"}
# The first two are whole looks, iconic on their own; the rest are single
# faults.
ICONIC = 2

# BENT CAM's three faults, weighed
BENT_MIX = {"cast": 1.0, "slip": 0.7, "adc": 0.45}

# THERMAL: the rings across the brightness range (RINGS[0] + RINGS[1] x
# amount), where the colour starts (luma SHADOW[0], rising SHADOW[1] a
# level), the grain's density at amount 1 (out of 256) and how many times a
# second it is drawn again, and the edge rim (a luma step over EDGE[0],
# EDGE[1] a level). The palettes: two complementary hues per mode plus
# their neighbours, flat bands in turn, cyclic; the touch seed picks one, so
# a new touch is a hard cut to another mode.
THERMAL_RINGS = (4, 8)
THERMAL_SHADOW = (40, 4)
THERMAL_GRAIN = 40
THERMAL_NOISE_HZ = 24
THERMAL_EDGE = (110, 3)
# the ring field's box blur radius
THERMAL_BLUR = 3
THERMAL_PALETTES = (
    # neon: pink and green, with violet, cyan, yellow, orange
    ((255, 40, 200), (120, 30, 160), (40, 220, 230), (40, 230, 70), (240, 230, 50), (255, 120, 40)),
    # ccd: violet and yellow-green
    ((150, 60, 255), (60, 20, 120), (200, 240, 60), (120, 140, 30), (255, 230, 90), (255, 90, 180)),
    # night: red and green
    ((240, 40, 40), (90, 10, 20), (40, 220, 60), (10, 80, 20), (255, 140, 30), (190, 240, 40)),
)

_M32 = 0xFFFFFFFF


def _jround(x):
    return int(math.floor(x + 0.5))


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ----- noise -----

def hash01(a, b=0, c=0):
    """A 32-bit integer hash to [0, 1) (the browser's, Math.imul included)."""
    h = (((int(a) & _M32) ^ 0x9E3779B9) * 0x85EBCA6B) & _M32
    h = ((h ^ (int(b) & _M32) ^ (h >> 13)) * 0xC2B2AE35) & _M32
    h = ((h ^ (int(c) & _M32) ^ (h >> 16)) * 0x27D4EB2F) & _M32
    h ^= h >> 15
    return h / 4294967296.0


def hash32_grid(idx, b, c):
    """hash01's 32-bit integer for every value of the int64 array `idx`
    (vectorised; the browser's per-sample loop gives the same numbers)."""
    m = np.uint64(_M32)
    h = (((idx.astype(np.uint64) & m) ^ np.uint64(0x9E3779B9)) * np.uint64(0x85EBCA6B)) & m
    h = ((h ^ np.uint64(int(b) & _M32) ^ (h >> np.uint64(13))) * np.uint64(0xC2B2AE35)) & m
    h = ((h ^ np.uint64(int(c) & _M32) ^ (h >> np.uint64(16))) * np.uint64(0x27D4EB2F)) & m
    h ^= h >> np.uint64(15)
    return h


def vnoise(x, seed=0):
    """Smooth 1-D value noise in [0, 1): continuous in x, one knot per unit."""
    i = math.floor(x)
    f = x - i
    s = f * f * (3 - 2 * f)
    a = hash01(i, seed, 7)
    b = hash01(i + 1, seed, 7)
    return a + (b - a) * s


# ----- mosaic and demosaic -----

def site_colour(x, y):
    """RGGB: even rows R G R G, odd rows G B G B. 0 = R, 1 = G, 2 = B."""
    return (0 if (x & 1) == 0 else 1) if (y & 1) == 0 else (1 if (x & 1) == 0 else 2)


def mosaic(px):
    """RGB(A) pixels (h, w, >=3) to one raw uint8 sample per site (h, w)."""
    raw = np.empty(px.shape[:2], dtype=np.uint8)
    raw[0::2, 0::2] = px[0::2, 0::2, 0]
    raw[0::2, 1::2] = px[0::2, 1::2, 1]
    raw[1::2, 0::2] = px[1::2, 0::2, 1]
    raw[1::2, 1::2] = px[1::2, 1::2, 2]
    return raw


def demosaic(raw):
    """Bilinear demosaic of an RGGB raw (h, w) back into RGB uint8 (h, w, 3).
    Edge sites mirror their neighbours (x = -1 reads 1, x = w reads w - 2)."""
    h, w = raw.shape
    p = np.pad(raw.astype(np.int32), 1, mode="reflect")
    v = p[1:-1, 1:-1]
    n, s = p[:-2, 1:-1], p[2:, 1:-1]
    e, wv = p[1:-1, 2:], p[1:-1, :-2]
    cross = (n + s + e + wv) >> 2
    diag = (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]) >> 2
    horiz = (e + wv) >> 1
    vert = (n + s) >> 1
    out = np.empty((h, w, 3), dtype=np.int32)
    # R sites: (even row, even col)
    out[0::2, 0::2, 0] = v[0::2, 0::2]
    out[0::2, 0::2, 1] = cross[0::2, 0::2]
    out[0::2, 0::2, 2] = diag[0::2, 0::2]
    # B sites: (odd row, odd col)
    out[1::2, 1::2, 2] = v[1::2, 1::2]
    out[1::2, 1::2, 1] = cross[1::2, 1::2]
    out[1::2, 1::2, 0] = diag[1::2, 1::2]
    # G on a red row: red left/right, blue above/below
    out[0::2, 1::2, 1] = v[0::2, 1::2]
    out[0::2, 1::2, 0] = horiz[0::2, 1::2]
    out[0::2, 1::2, 2] = vert[0::2, 1::2]
    # G on a blue row: blue left/right, red above/below
    out[1::2, 0::2, 1] = v[1::2, 0::2]
    out[1::2, 0::2, 2] = horiz[1::2, 0::2]
    out[1::2, 0::2, 0] = vert[1::2, 0::2]
    return out.astype(np.uint8)


# ----- the bends, raw to raw -----

def h_clock(raw, amount, seed, t):
    """Each band of lines starts its readout `s` samples early or late, read
    from the raw as one continuous stream, so a shifted line carries on into
    the next one. A band holds a shift or leans (a line-length error, a
    diagonal); the band grid rolls slowly; the whole frame creeps."""
    h, w = raw.shape
    a = _clamp(amount, 0, 1)
    n = w * h
    flat = raw.reshape(-1)
    band_h = max(2, _jround(h * (0.012 + 0.05 * hash01(seed, 1))))
    roll = t * h * (0.004 + 0.012 * hash01(seed, 2))
    p_shift = 0.12 + 0.5 * a
    max_shift = w * (0.04 + 0.22 * a)
    creep = a * a * 0.35 * (vnoise(t * 0.13, seed + 5) - 0.5)
    out = np.empty(n, dtype=np.uint8)
    xs = np.arange(w, dtype=np.int64)
    prev_band, s0, lean, band_top, odd = None, 0.0, 0.0, 0, 0
    for y in range(h):
        band = math.floor((y + roll) / band_h)
        if band != prev_band:
            prev_band = band
            band_top = y
            hit = hash01(band, seed, 3) < p_shift
            breathe = 0.55 + 0.45 * vnoise(t * 0.15 + band * 0.37, seed + 11)
            s0 = (hash01(band, seed, 4) * 2 - 1) * max_shift * breathe if hit else 0.0
            lean = ((hash01(band, seed, 8) * 2 - 1) * (1 + 3 * a)
                    if hit and hash01(band, seed, 6) < 0.3 * a else 0.0)
            odd = 1 if hit and hash01(band, seed, 9) < 0.5 else 0
        # whole Bayer pairs plus the band's own odd site: a band keeps its
        # colour while its shift drifts
        s = 2 * _jround((s0 + lean * (y - band_top) + creep * y) / 2) + odd
        row = y * w
        start = row + s
        if 0 <= start and start + w <= n:
            out[row:row + w] = flat[start:start + w]
        else:
            out[row:row + w] = flat[(start + xs) % n]
    return out.reshape(h, w)


def v_clock(raw, amount, seed, t):
    """The frame start slips (the picture rolls, a dark blanking bar at the
    seam); in stuck bands the same line is read again and again; bright
    samples smear down their column."""
    h, w = raw.shape
    a = _clamp(amount, 0, 1)
    slip = vnoise(t * 0.07, seed + 1)
    odd = 1 if hash01(seed, 3) < 0.5 else 0
    roll = (2 * math.floor(math.fmod(slip * slip * 2.2 * a + t * 0.02 * a, 1.0)
                           * (h >> 1)) + odd) % h
    bar = _jround(h * 0.035 * a)
    band_h = max(3, _jround(h * (0.02 + 0.07 * hash01(seed, 2))))
    p_stuck = 0.08 + 0.35 * a
    drift = 2 * math.floor(t * h * 0.006)
    seam = (h - roll) % h
    src_rows = np.empty(h, dtype=np.int64)
    dark = np.zeros(h, dtype=bool)
    held_from, prev_band, stuck = -1, None, False
    for y in range(h):
        band = math.floor((y + drift) / band_h)
        if band != prev_band:
            prev_band = band
            stuck = (hash01(band, seed, 5) < p_stuck
                     and vnoise(t * 0.2 + band, seed + 9) > 0.35)
            held_from = -1
        sy = (y + roll) % h
        if stuck:
            if held_from < 0:
                held_from = sy
            sy = held_from
        src_rows[y] = sy
        dark[y] = bar > 0 and (y - seam + h) % h < bar
    out = raw[src_rows].copy()
    out[dark] >>= 3
    # smear: a running spill down each column from samples above the knee
    knee = 236 - _jround(24 * a)
    keep = 0.92 + 0.05 * a
    gain = 0.12 + 0.2 * a
    q = np.zeros(w, dtype=np.float64)
    for y in range(h):
        v = out[y].astype(np.float64)
        q = q * keep + np.where(v > knee, (v - knee) * gain, 0.0)
        hot = q > 0.5
        if hot.any():
            out[y, hot] = np.floor(np.minimum(255.0, v[hot] + q[hot])).astype(np.uint8)
    return out


def adc_faults(seed, amount):
    """Faults on the 8 data lines, picked by the seed: stuck high, stuck low,
    or a bridge (OR) from a lower line. The top bits carry the shapes."""
    a = _clamp(amount, 0, 1)
    n = 1 + _jround(a * 2.4 * hash01(seed, 20) + a)
    faults = []
    for k in range(n):
        r = hash01(seed, 21 + k)
        bit = 7 - math.floor(hash01(seed, 31 + k) * (2 + 3 * a))
        if r < 0.35:
            faults.append({"kind": "high", "bit": bit})
        elif r < 0.6:
            faults.append({"kind": "low", "bit": bit})
        else:
            faults.append({"kind": "bridge", "bit": bit,
                           "from": max(0, bit - 1 - math.floor(hash01(seed, 41 + k) * 3))})
    return faults


def apply_faults(v, faults):
    for f in faults:
        if f["kind"] == "high":
            v |= 1 << f["bit"]
        elif f["kind"] == "low":
            v &= ~(1 << f["bit"])
        elif (v >> f["from"]) & 1:
            v |= 1 << f["bit"]
    return v & 255


def adc_bits(raw, amount, seed, t):
    """The fault is intermittent: on in some bands of lines, off in others,
    and the bands crawl."""
    h, w = raw.shape
    a = _clamp(amount, 0, 1)
    faults = adc_faults(seed, a)
    lut = np.array([apply_faults(v, faults) for v in range(256)], dtype=np.uint8)
    on = 0.75 - 0.6 * a
    scale = 1 / max(4, h * 0.06)
    live = np.array([vnoise(y * scale + t * 0.7, seed + 13) > on for y in range(h)])
    out = raw.copy()
    out[live] = lut[raw[live]]
    return out


def green_bias(raw, bias):
    """Lift or sink the green sites (the cast): up reads green, down magenta.
    In place."""
    b = _jround(bias)
    if not b:
        return raw
    for rows, cols in ((slice(0, None, 2), slice(1, None, 2)),
                       (slice(1, None, 2), slice(0, None, 2))):
        raw[rows, cols] = np.clip(raw[rows, cols].astype(np.int32) + b, 0, 255)
    return raw


def luma(p):
    """Integer Rec. 601 luma 0..255 of RGB int arrays (..., 3)."""
    return (306 * p[..., 0] + 601 * p[..., 1] + 117 * p[..., 2]) >> 10


def box_blur(v, r):
    """Integer box mean over a (2r+1)^2 window, edges clamped: floor of the
    window sum over its area (the browser's running sums give the same)."""
    k = 2 * r + 1
    p = np.pad(v.astype(np.int64), r, mode="edge")
    c = np.cumsum(p, axis=1)
    c = np.pad(c, ((0, 0), (1, 0)))
    hs = c[:, k:] - c[:, :-k]
    c = np.cumsum(hs, axis=0)
    c = np.pad(c, ((1, 0), (0, 0)))
    return (c[k:] - c[:-k]) // (k * k)


def palette_lut(keys):
    """A cyclic 256-entry palette of flat bands, one per key colour (no
    blend: a blend between complementary hues is grey)."""
    n = len(keys)
    return np.array([keys[(i * n) >> 8] for i in range(256)], dtype=np.int64)


def thermal(px, amount, seed, t):
    """THERMAL, on RGB: shorted high data bits make values wrap, so the
    brightness runs through a repeating palette (palette(fract(luma x N)))
    and smooth gradients come back as many thin false-colour rings, denser
    toward the brights, hugging every light. The shadows stay near-black
    (the colour lives in the mids and highs), chunky saturated grain boils
    in the bands and is drawn again every frame, and outlines keep a thin
    bright rim. The rings follow the picture; the palette mode is a hard
    cut on a new touch."""
    h, w = px.shape[:2]
    a = _clamp(amount, 0, 1)
    p = px[..., :3].astype(np.int64)
    y = luma(p)
    mode = math.floor(hash01(seed, 80) * len(THERMAL_PALETTES))
    lut = palette_lut(THERMAL_PALETTES[mode])
    n = THERMAL_RINGS[0] + _jround(THERMAL_RINGS[1] * a)
    off = math.floor(256 * hash01(seed, 81))       # per touch: rings follow the picture only
    # the rings follow the light, not the texture: a box-blurred luma
    ys = box_blur(y, THERMAL_BLUR)
    col = lut[(((ys * ys * n) >> 8) + off) & 255]
    wgt = np.clip((y - THERMAL_SHADOW[0]) * THERMAL_SHADOW[1], 0, 256)
    shadow = np.stack([y >> 1, y >> 3, y >> 1], -1)
    out = (col * wgt[..., None] + shadow * (256 - wgt)[..., None]) >> 8
    # grain: 2x2 cells of a random palette colour, dim in the shadows
    frame = math.floor(t * THERMAL_NOISE_HZ)
    yy, xx = np.mgrid[0:h, 0:w]
    cell = (xx >> 1) + (yy >> 1) * ((w + 1) >> 1)
    r = hash32_grid(cell, frame, (seed ^ 0x6A1) & _M32).astype(np.int64)
    hit = (r >> 24) < _jround(a * THERMAL_GRAIN)
    grain = (lut[(r >> 8) & 255] * (64 + ((3 * wgt) >> 2))[..., None]) >> 8
    out = np.where(hit[..., None], grain, out)
    # edges: a thin bright rim where the luma steps
    yp = np.pad(y, 1, mode="edge")
    e = np.abs(yp[1:-1, 2:] - yp[1:-1, :-2]) + np.abs(yp[2:, 1:-1] - yp[:-2, 1:-1])
    k = np.clip((e - THERMAL_EDGE[0]) * THERMAL_EDGE[1], 0, 256)
    out = out + (((255 - out) * k[..., None]) >> 8)
    return out.astype(np.uint8)


# THERMAL works on RGB, after the camera's demosaic: what the processor hands
# on, not what the sensor read.
RGB_BENDS = {"thermal": thermal}


def bent_cam(raw, amount, seed, t, mix=None):
    """BENT CAM: the cast, the slipped bands, the posterised bursts."""
    mix = BENT_MIX if mix is None else mix
    a = _clamp(amount, 0, 1)
    sign = -1 if hash01(seed, 50) < 0.5 else 1
    breathe = 0.6 + 0.4 * vnoise(t * 0.11, seed + 51)
    green_bias(raw, sign * mix["cast"] * (14 + 46 * a) * breathe)
    cur = raw
    if mix["slip"] > 0:
        cur = h_clock(cur, mix["slip"] * a, seed, t)
    if mix["adc"] > 0:
        # JS `seed ^ 0x9e37` is an int32; every use goes through & 0xffffffff
        cur = adc_bits(cur, mix["adc"] * a, (seed ^ 0x9E37) & _M32, t)
    return cur


def sensor_bend(px, effect, amount, seed=1, t=0.0, mix=None):
    """One sensor bend on RGB uint8 pixels (h, w, 3); returns new RGB pixels.
    Amount 0, an unknown effect or a frame under 4x4 returns `px` itself."""
    try:
        a = float(amount)
    except (TypeError, ValueError):
        a = 0.0
    a = _clamp(a if math.isfinite(a) else 0.0, 0.0, 1.0)
    h, w = px.shape[:2]
    if a == 0 or effect not in SENSOR_EFFECTS or w < 4 or h < 4:
        return px
    seed = int(seed) & _M32
    if effect in RGB_BENDS:
        return RGB_BENDS[effect](px, a, seed, t)
    raw = mosaic(px)
    if effect == "bent":
        bent = bent_cam(raw, a, seed, t, mix)
    else:
        bent = {"hclock": h_clock, "vclock": v_clock, "adc": adc_bits}[effect](raw, a, seed, t)
    return demosaic(bent)
