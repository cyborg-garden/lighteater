"""The "alive" fractal step: its tuning table and the pure logic the shaders
mirror. Shared with the browser.

H's third step (depth + fractal veins) carries more than the stock fractal
veins on both hosts: motion as fast food the fine levels hunt, exploring
tips, a light that runs along a thread when it closes onto another vein,
pruning of threads nobody walks, a slow shuttle pulse travelling out along
the veins, veins steering along lines of equal depth, and, as the display,
the molten calligraphy (dtouch/shaders/alive/ink.frag). The passes are
dtouch/physarum_alive.py here and alive.js on cyborg-garden-site, which
vendors dtouch/shaders/alive/ (shaders + alive.json, the table below, by
`python -m dtouch.alive_looks`).

Clean room: Jones model, Tero-style flux pruning, Alim et al. shuttle
streaming as described in public writeups; no code or parameter tables from
any CC BY-NC-SA project.
"""
from __future__ import annotations

import math

import numpy as np

# The tuning table. Lengths are at the 576-wide reference grid (scaled by
# gw / PX_REF like every other length), times are seconds. Key for key the
# browser's ALIVE (src/scripts/physarum/alive-core.js); alive.json carries it.
ALIVE = {
    # A. hunting
    "freshHalfLife": 0.5,          # fresh motion food, s
    "skirtHalfLife": 0.8,          # its wide skirt, s
    "freshIn": 7,                  # deadbanded frame difference -> fresh (0..1)
    "freshFood": [0.4, 2.4, 3.4],  # fresh as food per level, x the trail's bright end
    "freshTurn": [0.04, 0.22, 0.28],  # largest lean toward the skirt per frame, rad
    "freshGain": 14,               # skirt slope per scene texel for a full lean
    "freshLand": 1.0,              # landing weight of fresh motion for reseeding fine agents
    "freshPull": 0.003,            # per-frame chance a fine agent relocates to the motion's edge
    "fluxHalfLife": 0.25,          # the deposit memory pruning reads, s
    "prune": [0.3, 1.6, 2.2],      # extra loss where a level is unused
    "pruneAt": [0.004, 0.02],      # used-ness ramp, deposits / trail
    "onOff": [1.0, 0.7],           # line presence, x compose's outline threshold per level
    "cell": 16,                    # connection-event block, grid px
    "cool": 4,                     # s between events in one block
    "young": 1.5,                  # s: a front (a thread still growing)
    "settled": 4,                  # s: a vein that has stood this long
    "reach": 7,                    # texels back along the thread that must be young line too
    "minAge": 0.3,                 # s the thread must have stood there
    "glowStep": [0.985, 0.9],      # light carried per texel: young, settled
    "glowFade": 0.8,               # per frame, once spent
    "glowRefr": 1.2,               # s
    "glowK": 1.0,
    "glowCalm": 0.12,              # share of connections away from motion that light
    "frontAge": 1.4,               # s a trail draws as fresh
    "front": [0.12, 0.55],         # fresh brightness boost: trunks, finer levels
    "pulsePeriod": 32,             # s per shuttle cycle (Physarum's is ~100 s)
    "pulseAmp": 0.2,               # brightness swing
    "pulseSwell": 0.35,            # hairline width swing
    "pulseWave": 0.09,             # rad per quarter-grid cell along the veins
    "pulseFar": 400,               # cells past which the pulse fades
    "pulseSrcQ": 0.97,             # the pulse starts on veins over the nearest 3% of the depth
    "pulseSrcFallback": 0.75,      # (the depth it starts above when that cannot be read back)
    "pulseCreep": 0.2,             # cells per frame the distance may grow
    "pulseOffCost": 4,             # a cell of empty ground costs this many cells of vein
    "pulseSound": 0.8,             # the phase runs up to 1 + this x sens x level faster with sound
    "headsTarget": 12000,          # agents sampled for tips
    "headLook": 3,                 # px at the reference grid
    "headTail": 7,                 # px at the reference grid
    "headPx": 2.5,                 # output px (x sqrt dpr)
    "headK": 0.4,
    # B. bending in
    "contour": [0.1, 0.06, 0.03],  # share of the angle to the depth contour turned per frame
    "contourGain": 30,             # depth slope per scene texel for full contour steering
    "slope": 0.8,
    "domeR": 22,                   # scene texels
    "domeW": [0.35, 0.9],          # share of depth from the matte dome: any matte, person
    "depthEase": 0.06,             # per frame
    "carve": [0.35, 0.5, 1.8, 3.0],  # groove floor, rim, refraction (lum texels), depth refraction (scene texels)
    "body": [0.45, 0.3, 0.65, 0.45],  # vein body over the camera: brightness floor, feed hue tint, contact line, drop shadow
    "contact": [1.25, 2.5],        # contact line radius, drop shadow length (output px, x sqrt DPR)
    "fine": [2.2, 1.2],            # over the camera: threshold raise for threads, veins (carve only)
    "bump": [7, 30],               # vein normal scale, depth steepness gain
    # C. molten calligraphy (shaders/alive/ink.frag): the display pass for
    # the fractal step, camera or not. ink -1 shows the carve/glow look.
    "ink": 1,                      # ground variant: 0 lacquer on light, 1 molten pool, 2 liquid chrome; -1 off
    "inkThr": [0.5, 1.0, 9.0],     # outline threshold raise: primaries, thorns, hairlines
    "press": [0.35, 50, 3],        # pressure: amount, wavelength (grid px), drift (grid px / s)
    "dry": [0.2, 2.6, 18, 1.3],    # dry brush: amount, streak scale across, along (grid px); glint line strength
    "sheen": [2.6, 2.0, 1.4, 180],  # bump, env strength, highlight strength, highlight power
    "ground": [0.9, 2.0, 0.3, 0.5],  # heat gain, heat blur (mip level), ground level, edge halo
    "edge": [0.4, 2600],           # outline anti-alias width (x fwidth), glint line sharpness
    "nib": [0.45, 2.1, 0.6, 0],    # broad nib: hairline gain, full gain, fixed pen angle (rad), drift (rad / s)
    "thorn": [1.8, 0.75, 3.5, 0.3],  # thorns: gain near the primaries, far out, mip level of 'near', nib hairline gain
    "tierBump": [0.8, 0.5],        # bevel of the thorns, of the hairlines (x the primaries')
    "inkBody": [1.5, 0.7],         # ink primaries' body: mip level, its share
    "chrome": [0.08, 0.3, 0.6],    # chrome posterise cuts (ink luma): dark grey, light grey, white
    "thornRim": 0.9,               # thorns' hard white rim on the side facing the light
    # D. the landing (the owner's call 2026-10-06): the mode lands, and 0 /
    # panic returns, on the first look's fractal step in the molten ink with
    # this palette under it. The camera's picture stays hidden (video bg
    # off), so the ground is black. Every other look still lands on depth
    # with the fractal off, as the physarum unit's looks.json says.
    "landing": {"palette": "violet"},
    # E. the ink's style options (the owner's call 2026-10-07, after the
    # TikTok study): paper (black ink on white, one accent) and a mirror
    # fold (2, 4 or 6), both off by default and never set by the landing
    # (ink.frag). The fold's source half follows the performer, the matte's
    # lit centroid (fold_side below): an axis moves only once the centroid
    # is `hyst` (a share of the frame) past the centre line; for 6 the
    # source wedge turns in 60 degree steps once the centroid is `reach`
    # from the centre and `angHyst` rad past the wedge's edge; and nothing
    # moves again for `hold` seconds, so a side change is one cut, never a
    # flicker.
    "fold": {"hyst": 0.08, "hold": 2.0, "reach": 0.12, "angHyst": 0.15},
    # F. flash safety for the style options (the owner's hard floor: never
    # more than 3 flashes a second, WCAG 2.3.1). Paper turns the whole frame
    # white, so: K, Y and (while paper shows) blackout take at most one
    # change per `cooldown` s, held or spammed; what the paper shows flips
    # at most once per `cooldown` s whatever asks (a key, a look, 0), and
    # then fades over `fade` s instead of cutting.
    "style": {"cooldown": 0.5, "fade": 0.3},
}

# The fold amounts Y steps through (0 off), and where a fold starts: the
# left half, the top-left quadrant, the wedge pointing up (image space,
# +y down), not yet moved (None).
MIRRORS = (0, 2, 4, 6)
FOLD_START = (-1.0, -1.0, -math.pi / 2, None)


def _wrap(a):
    """An angle into [-pi, pi), the same on both hosts (no % sign rules)."""
    return a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))


def _turn(ang, cx, cy, aspect, margin, t):
    """The 6-fold wedge's centre turned toward the centroid in whole 60
    degree steps, once it is `reach` out and `margin` rad past the wedge's
    edge (0: snap to the nearest wedge)."""
    dx, dy = (cx - 0.5) * aspect, cy - 0.5
    if math.hypot(dx, dy) > t["reach"]:
        off = _wrap(math.atan2(dy, dx) - ang)
        if abs(off) > math.pi / 6 + margin:
            step = math.pi / 3
            return _wrap(ang + math.floor(off / step + 0.5) * step)
    return ang


def fold_side(state, cx, cy, aspect, now, fold, table=None):
    """The fold's source as the performer moves: `state` (side x, side y,
    wedge angle, time of the last move or None) after the matte's lit
    centroid (cx, cy) in image space (0..1, +y down) at time `now` (s), for
    fold `fold`. Only the axes that fold reads can move (2: x; 4: x and y;
    6: the wedge), so only a change the picture shows stamps the hold.
    Mirrored by alive-core.js foldSide (tests on both hosts)."""
    t = ALIVE["fold"] if table is None else table
    sx, sy, ang, at = state
    if at is not None and now - at < t["hold"]:
        return state
    nsx, nsy, nang = sx, sy, ang
    if fold in (2, 4):
        nsx = -1.0 if cx < 0.5 - t["hyst"] else 1.0 if cx > 0.5 + t["hyst"] else sx
    if fold == 4:
        nsy = -1.0 if cy < 0.5 - t["hyst"] else 1.0 if cy > 0.5 + t["hyst"] else sy
    if fold == 6:
        nang = _turn(ang, cx, cy, aspect, t["angHyst"], t)
    if nsx == sx and nsy == sy and nang == ang:
        return state
    return (nsx, nsy, nang, now)


def fold_snap(state, cx, cy, aspect, now, fold, table=None):
    """A fold just turned on (or changed): its source goes straight to the
    side the performer is on (no hysteresis), and the hold starts, so it
    does not move again at once. alive-core.js foldSnap."""
    t = ALIVE["fold"] if table is None else table
    sx, sy, ang, _ = state
    if fold in (2, 4):
        sx = -1.0 if cx < 0.5 else 1.0
    if fold == 4:
        sy = -1.0 if cy < 0.5 else 1.0
    if fold == 6:
        ang = _turn(ang, cx, cy, aspect, 0.0, t)
    return (sx, sy, ang, now)


def style_ready(last, now, table=None):
    """May a style change (K, Y, blackout over paper, a paper flip) happen
    at `now` (s), the last one at `last` (None: never)?"""
    t = ALIVE["style"] if table is None else table
    return last is None or now - last >= t["cooldown"]


def paper_step(amount, on, dt, table=None):
    """What the paper shows (0..1), one frame on: toward `on` at 1 / fade
    per second, so a flip is a fade, never a cut. alive-core.js paperStep."""
    t = ALIVE["style"] if table is None else table
    d = max(float(dt), 0.0) / max(t["fade"], 1e-6)
    return min(amount + d, 1.0) if on else max(amount - d, 0.0)


def keep_for(half_life, dt):
    """Per-frame keep for a half-life in seconds (alive-core.js keepFor)."""
    return 0.5 ** (max(dt, 0.0) / half_life) if half_life > 0 else 0.0


def alive_active(enabled, failed, fractal):
    """The gate: the alive path runs only on H's fractal step, only when the
    host wants it and the GPU built it. Flat and depth-only are never alive."""
    return bool(enabled) and not failed and fractal > 0


def advance_pulse(phase, dt, period, sound=0.0, lock=0.0):
    """The shuttle pulse's phase, advanced by dt; sound (0..1) quickens it."""
    p = phase + (2.0 * math.pi * max(dt, 0.0) / period) * (1.0 + lock * max(sound, 0.0))
    return math.fmod(p, 2.0 * math.pi)


def depth_quantile(buf, q):
    """The q-quantile of the depth channel (.b) of an (..., 4) float buffer,
    on a stride so it stays cheap (alive-core.js depthQuantile)."""
    flat = np.asarray(buf, np.float32).reshape(-1, 4)
    texels = flat.shape[0]
    step = max(1, texels // 4096)
    v = np.sort(flat[::step, 2])
    if v.size == 0:
        return 0.0
    return float(v[min(v.size - 1, int(math.floor(q * v.size)))])


def head_stride(sim_n, target):
    """Every n-th agent is sampled for tips, so about `target` are looked at."""
    return max(1, int(math.floor(sim_n / max(1, target) + 0.5)))   # Math.round


def count_events(buf, since):
    """Connection events since the last readback: blocks whose
    seconds-since-last-event (.r of each RGBA texel) is under `since`."""
    flat = np.asarray(buf, np.float32).reshape(-1, 4)
    return int((flat[:, 0] < since).sum())
