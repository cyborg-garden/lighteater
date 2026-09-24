"""Physarum simulation — a slime-mold trail network that eats the live video.

A fixed pool of N agents follows the classic Jones transport-network model
(Jeff Jones, "Characteristics of pattern formation and evolution in
approximations of physarum transport networks", Artificial Life 16(2), 2010):
each agent senses the trail map at three points ahead of it (left / center /
right), turns toward the strongest reading, steps forward, and deposits trail;
the trail map is then diffused and decayed. Out of nothing but that loop,
vein networks grow, merge, and re-route.

Two couplings make it a *video* instrument rather than a screensaver:

- **The matte is the pen.** Every simulation parameter exists as a PAIR — a
  background value and a subject value — and each agent runs on the blend of
  the two by the matte value under it. Body and background literally run
  different physics, and the silhouette becomes a negotiated boundary between
  two organisms. (This spatial-interpolation idea is inspired by Etienne
  Jacob's interactive-physarum "pen"; the implementation here is independent
  and shares no code or parameter tables with it — that project is
  CC BY-NC-SA and this one is AGPL.)
- **Luminance is food.** The camera's luma is added to what the sensors read,
  so the network is drawn toward bright regions, and recycled agents respawn
  preferentially onto the lit subject. Wave a light around and the mold
  chases it.

Pure NumPy + a little OpenCV (box blur), all on a small working grid, same as
ParticleFlow. The trail map itself is the picture.
"""
from __future__ import annotations

import cv2
import numpy as np

# Behavior points — named parameter sets an agent can run on. Both ends of the
# matte blend (background / subject) pick from this table. Hand-tuned on the
# live instrument; the names describe the texture each one settles into.
# Re-spread 2026-08 (owner feedback: presets read too alike): each point now
# owns a distinct morphology niche — veins = the anchor mesh, cells = slow
# rounded compartments, fingers = fast bright darting runners, haze = pure
# fog, web = long straight strands, storm = wide-angle chaos. Verified by
# pairwise morphology distance on a static scene (min pair +38%).
#   sense:   sensor distance, in grid pixels
#   spread:  sensor half-angle, radians (how wide the left/right feelers sit)
#   turn:    rotation applied when a side sensor wins, radians
#   step:    move distance per frame, grid pixels
#   deposit: trail laid down per agent per frame
POINTS = {
    "veins":   dict(sense=9.0,  spread=0.39, turn=0.79, step=1.4, deposit=1.0),
    "cells":   dict(sense=14.0, spread=1.10, turn=0.30, step=0.8, deposit=1.0),
    "fingers": dict(sense=4.5,  spread=0.62, turn=0.70, step=3.4, deposit=2.0),
    "haze":    dict(sense=3.6,  spread=1.50, turn=0.08, step=0.35, deposit=0.25),
    "web":     dict(sense=34.0, spread=0.50, turn=1.10, step=1.8, deposit=0.6),
    "storm":   dict(sense=10.0, spread=1.55, turn=1.70, step=3.2, deposit=1.4),
}
POINT_NAMES = list(POINTS)

# A point can only steer on what the trail still resolves. The trail is box
# blurred every frame over a `1/(1-decay)`-frame memory, which accumulates to
# roughly SENSE_SIGMA_PX of smoothing at the shipped diffuse radius, so, with
# `scale` the engine's grid-px-per-table-unit:
#
#   sense * scale              >= SENSE_SIGMA_PX          reach past the blur
#   2*sense*scale*sin(spread)  >= 1.5 * SENSE_SIGMA_PX    resolve a gradient
#   step                       <  sense                   don't leap past it
#
# BOTH engines, and the CPU one binds. The blur radius is the same integer on
# each, so the smoothing is the same number of PIXELS — but the CPU grid is
# 2.22x narrower, so a point has 2.22x less reach there in the units that
# matter. Checking only PX_SCALE (the GL figure) let `fingers` ship at 2.36 px
# of separation on the fallback engine — WORSE than the 2.21 px this whole
# diagnosis calls blind, on the path every machine without a GPU actually
# runs. tests/test_physarum_points.py checks both.
#
# `fingers` used to violate all three (2.21 px of sensor separation on a field
# smoothed to 3.33 px, and a stride longer than its own reach), which made it a
# constant-turn random walk rather than a Jones walker — and it is the default
# body point. `haze` sensed inside its own blur kernel. tests/test_physarum.py
# pins these; widen a point rather than quietly reintroducing a blind one.
# Per-frame decay of the wave's ballistic phase (~0.55 s at 60 fps).
# Shared by both engines; physarum_gl re-exports the same constant.
BALLISTIC_DECAY = 0.90

# Exposure-reference smoothing, shared by both engines (physarum_gl re-exports
# this). The p95 the picture is normalized by is an estimate off a strided
# subsample; tracking it frame-to-frame pumps.
NORM_EMA = 0.85

PX_SCALE = 1280.0 / 576.0        # working grid -> GL grid, see modes.physarum
CPU_SCALE = 1.0                  # the CPU field runs at gain 1.0 on its own grid
SENSE_SIGMA_PX = 3.33            # accumulated trail smoothing, GL px

# ----- the spatial mosaic --------------------------------------------------
# One parameter set over the whole canvas grows one texture, everywhere, and
# no slider can change that — which is why every look read as the same thing
# recoloured. The grid is instead cut into slowly drifting Voronoi zones, and
# each zone runs ONE of these regimes outright. Hard seams, not a blend: the
# blend is what averages six characters back into one.
#
# Multipliers on the blended point parameters, plus a multiplier on the
# species cross-term (low = one organism, thick canals; high = woven, walled).
# Ours, derived from this model's own points; nothing external.
SPATIAL_REGIMES = {
    "mesh":  dict(sense=0.45, turn=1.35, spread=0.75, step=0.60, cross=1.45),
    "trunk": dict(sense=2.20, turn=0.65, spread=0.85, step=1.50, cross=0.25),
    "bloom": dict(sense=0.55, turn=1.90, spread=1.60, step=1.15, cross=1.10),
    "drift": dict(sense=1.80, turn=0.45, spread=1.30, step=0.90, cross=0.55),
    "knot":  dict(sense=0.75, turn=1.55, spread=0.45, step=0.50, cross=1.30),
    "calm":  dict(sense=1.00, turn=1.00, spread=1.00, step=1.00, cross=1.00),
}
SPATIAL_REGIME_NAMES = list(SPATIAL_REGIMES)
ZONE_REGIME_COUNT = 6            # the shader's u_zreg[] / u_zcross[] arity
assert len(SPATIAL_REGIMES) == ZONE_REGIME_COUNT

# The regime table as arrays, in the same order the GL host uploads u_zreg[] /
# u_zcross[] — columns are sense, turn, spread, step.
ZONE_REGIME_MULTS = np.array(
    [[SPATIAL_REGIMES[k][f] for f in ("sense", "turn", "spread", "step")]
     for k in SPATIAL_REGIMES], np.float32)
ZONE_REGIME_CROSS = np.array(
    [SPATIAL_REGIMES[k]["cross"] for k in SPATIAL_REGIMES], np.float32)

# The zones drift on slow sines (update.frag zone_at). A Voronoi lookup per
# agent per frame is a GPU luxury; on the CPU the lattice is rasterized into a
# zone-index map once every ZONE_REFRESH frames and read by agent position.
# At the shipped drift rates (0.07-0.17 rad/s) a site moves ~0.02 lattice
# cells in 20 frames, so the staleness is invisible and the cost amortizes to
# a fraction of a millisecond.
ZONE_REFRESH = 20
ZONE_MAP_MAX = 224               # zone map is rasterized at <= this and lifted
                                 # to the grid: interiors are flat, so only the
                                 # seam's cross-fade needs the resolution
ZONE_EDGE = 0.16                 # seam cross-fade width, lattice units
ZONE_WARP = 0.22                 # domain-warp amplitude, lattice units
TAU = float(2.0 * np.pi)

# Sensing further than a tenth of the frame is sensing globally: the three
# sensors stop reporting on a neighbourhood and the mold dissolves into a
# structureless cloud. The mosaic's long-range regimes would otherwise
# multiply an already-long point (`web`, sense 34) well past that.
SENSE_MAX_FRAC = 0.10

# Species cross-term: each species is repelled by the NEXT species' trail and
# mildly drawn to the PREVIOUS one, with this weight on the "drawn to" side.
# Mirrors update.frag's `float back = 0.35 * cross;`.
CROSS_BACK = 0.35

# Ceiling on the zone-scaled species cross-term, both engines.
REPEL_MAX = 1.5


def species_matrix(species, cross):
    """Row-major 3x3 species interaction matrix — the ONE definition.

    Both engines and the browser port implement this arrangement, and it used
    to be written out three times, which is how the GL field and the CPU
    field came to disagree about what cross 0 means.

    Diagonal is self-attraction (1). The off-diagonals BLEND from +1 at cross
    0 toward rock-paper-scissors at cross 1: each species repelled outright by
    the next, mildly drawn to the previous.

    All-ones is the load-bearing end. Three populations depositing into three
    channels and sensing their SUM is arithmetically the same field as one
    population depositing into one, so cross 0 really is the legacy
    single-channel model — which is what the bottom of `weave` promises. An
    identity row there instead gives three mutually invisible organisms at a
    third of the density each: a different picture wearing the same label,
    and measurably so (channel correlation 0.04 where the legacy field has
    them locked together).
    """
    if int(species) <= 1:
        return (1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0)
    c = max(float(cross), 0.0)
    wn = 1.0 - 2.0 * c                      # the next species: +1 -> -1
    wp = 1.0 - (1.0 - CROSS_BACK) * c       # the previous one: +1 -> +0.35
    return (1.0, wn, wp,
            wp, 1.0, wn,
            wn, wp, 1.0)

_PARAM_KEYS = ("sense", "spread", "turn", "step", "deposit")

# Signed keep map semantics (both engines): keep > 0 raises the effective
# decay toward KEEP_HOLD (trails linger — react's swept paths); keep < 0
# lowers it by up to MELT_DROP (trail re-fluidizes — evolve's stale-region
# melt), floored at MELT_FLOOR so a melt never hard-erases a region in one
# frame, it dissolves over a second or two.
KEEP_HOLD = 0.995
MELT_DROP = 0.12
MELT_FLOOR = 0.70

# Sense-range multipliers for the `hetero` sub-populations: a third of the
# pool feels close, a third mid, a third far. Multi-scale sensing grows
# multi-scale structure — filigree between the trunk lines.
HETERO_MULTS = np.array([0.45, 1.0, 1.9], np.float32)

# ----- depth: the picture lit as a relief -----------------------------------
# The numpy twin of the relief branch in shaders/physarum/tonemap.frag, so the
# Depth control does the same thing on the CPU fallback. A visible control
# that silently does nothing on one engine fails the honest-controls bar.
# The constants are the shader's, copied by value (GLSL cannot import them);
# tests/test_physarum_gl.py renders both on a fixed grown field and on tall
# synthetic ridges and holds them within 1e-3; a one-sided change to any of
# these constants should fail there (checked by mutation when written).
RELIEF_K = 0.25          # height compression (trail/norm -> h)
RELIEF_BUMP = 5.0        # normal steepness
RELIEF_BROAD = 0.5       # share of the 3 px gradient in the normal
RELIEF_AMB = 0.30        # key-light floor on the far side of a vein
RELIEF_SHADOW = 2.5      # cast shadow strength
RELIEF_CAVITY = 1.0      # cavity darkening strength
RELIEF_GROUND = 0.8      # faint ground brightness (1 = untouched)
# The shading divides by the light's z; the host keeps it at or above this
# (the shader deliberately has no guard of its own — see its header).
RELIEF_MIN_Z = 0.05
# Key light in GRID-TEXEL space (+y = row index) that both engines start
# with: upper-left ON SCREEN, 51 degrees up. Row 0 is the top of the desktop
# window (tests/test_physarum_gl.py
# test_trail_rows_and_columns_are_grid_coordinates), so "up" on screen is -y
# here. The prototype's constant was (-0.5443, +0.5443, 0.6234) because its
# harness drew row 0 at the bottom; copied as is, it lit the desktop from the
# lower left.
RELIEF_LIGHT = (-0.5443, -0.5443, 0.6234)


def relief_light(light):
    """The `u_light` a host may hand the shader: `light` normalised, with z
    raised to RELIEF_MIN_Z when it sits lower (then re-normalised). Both
    engines route their light through here, once per frame."""
    v = np.asarray(light, np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        v = np.asarray(RELIEF_LIGHT, np.float64)
        n = float(np.linalg.norm(v))
    v = v / n
    if v[2] < RELIEF_MIN_Z:
        xy = v[:2]
        r = float(np.linalg.norm(xy))
        s = np.sqrt(1.0 - RELIEF_MIN_Z * RELIEF_MIN_Z) / r if r > 0 else 0.0
        v = np.array([xy[0] * s, xy[1] * s, RELIEF_MIN_Z])
    return (float(v[0]), float(v[1]), float(v[2]))


def _smoothstep(e0, e1, x):
    t = np.clip((x - np.float32(e0)) * np.float32(1.0 / (e1 - e0)), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def relief(lum, total, inv_norm, depth, light):
    """Light the tonemapped `lum` (gh, gw) as a relief of the trail height,
    exactly as tonemap.frag's u_depth > 0 branch does: `total` is the
    channel-summed trail, `inv_norm` 1/p95, `light` the grid-space key light
    (already through relief_light). Returns lum itself at depth <= 0.

    Neighbours clamp at the grid edge, like the shader's texelFetch clamp
    (np.roll would wrap, which the shader does not). The shader's `round` on
    the shadow-march offsets may round exact halves either way (GLSL leaves
    it to the implementation); np.round goes to even. Only an azimuth that
    lands a march offset on exactly .5 can tell them apart."""
    if depth <= 0.0:
        return lum
    gh, gw = lum.shape
    h = (1.0 - np.exp(np.float32(-RELIEF_K * inv_norm) * total)).astype(np.float32)
    P = 5                                          # widest reach: the 5 px march
    hp = np.pad(h, P, mode="edge")

    def at(dx, dy):                                # h at (col + dx, row + dy)
        return hp[P + dy:P + dy + gh, P + dx:P + dx + gw]

    e1, w1, n1, s1 = at(1, 0), at(-1, 0), at(0, 1), at(0, -1)
    e3, w3, n3, s3 = at(3, 0), at(-3, 0), at(0, 3), at(0, -3)
    b = np.float32(RELIEF_BROAD)
    gx = (e1 - w1) * np.float32(0.5) * (1 - b) + (e3 - w3) * np.float32(1 / 6.0) * b
    gy = (n1 - s1) * np.float32(0.5) * (1 - b) + (n3 - s3) * np.float32(1 / 6.0) * b
    nx, ny = -gx * np.float32(RELIEF_BUMP), -gy * np.float32(RELIEF_BUMP)
    inv = np.float32(1.0) / np.sqrt(nx * nx + ny * ny + np.float32(1.0))
    nx, ny, nz = nx * inv, ny * inv, inv
    lx, ly, lz = (np.float32(c) for c in light)
    shade = (np.float32(RELIEF_AMB) + np.float32(1.0 - RELIEF_AMB)
             * np.maximum(nx * lx + ny * ly + nz * lz, 0.0) / lz)
    r = float(np.hypot(lx, ly))
    dx, dy = (float(lx) / r, float(ly) / r) if r > 0 else (0.0, 0.0)
    o2 = at(int(np.round(dx * 2.0)), int(np.round(dy * 2.0))) - h
    o5 = at(int(np.round(dx * 5.0)), int(np.round(dy * 5.0))) - h
    shadow = 1.0 - np.minimum(
        np.float32(0.75),
        np.float32(RELIEF_SHADOW) * np.maximum(0.0, np.maximum(o2, o5 * np.float32(0.7))))
    ring = np.float32(0.25) * (e3 + w3 + n3 + s3)
    cavity = 1.0 - np.minimum(np.float32(0.8),
                              np.float32(RELIEF_CAVITY) * np.maximum(0.0, ring - h))
    ground = (np.float32(RELIEF_GROUND)
              + np.float32(1.0 - RELIEF_GROUND) * _smoothstep(0.05, 0.45, h))
    lit = np.clip(lum * shade * shadow * cavity * ground, 0.0, 1.0)
    d = np.float32(depth)
    return (lum + (lit - lum) * d).astype(np.float32)


class PhysarumField:
    def __init__(self, n=250_000, gw=480, gh=270, seed=0,
                 point_bg="veins", point_fg="fingers",
                 decay=0.94, diffuse=1, food=0.35, exposure=3.5,
                 grain=0.2, reseed_frac=0.004, gain=1.0,
                 sat=0.0, jitter=0.0, hetero=0.0, species=3, cross=0.0,
                 sharpen=0.0, mosaic=0.0, zones=7.0):
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if gw <= 0 or gh <= 0:
            raise ValueError(f"grid must be positive, got {gw}x{gh}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if not (0.0 <= reseed_frac <= 1.0):
            raise ValueError(f"reseed_frac must be in [0, 1], got {reseed_frac}")
        self.n, self.gw, self.gh = n, gw, gh
        rng = np.random.default_rng(seed)
        self.px = rng.uniform(0, gw, n).astype(np.float32)
        self.py = rng.uniform(0, gh, n).astype(np.float32)
        self.heading = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        # Species is a LINEAGE, not a per-frame lookup: drawn once here (same
        # rng, same draw order as PhysarumFieldGL's agent-texture .w slot) and
        # never rewritten. Reseeds, spawn_burst, wave and gather all move
        # agents around by index, so an agent keeps its species for life.
        self.species = max(1, min(3, int(species)))
        self._species = rng.integers(0, self.species, n).astype(np.int32)
        # Three species channels. Each species lays into its own channel and
        # senses all three through a signed matrix, so one population can be
        # REPELLED by another's trail — the mechanism behind exclusion
        # membranes and travelling fronts, which a single-channel attract-only
        # field cannot express at any parameter setting. `trail_total` is the
        # 2-D picture (the whole organism) and what the statistics run on.
        self.trail = np.zeros((gh, gw, 3), np.float32)
        # params
        self.point_bg = point_bg
        self.point_fg = point_fg
        self.decay = decay
        self.diffuse = diffuse          # box-blur radius in pixels; 0 = off
        self.food = food                # how strongly video luma attracts sensors
        self.exposure = exposure        # tonemap gain on the trail
        self.grain = grain              # raw per-frame deposits in the picture
        # depth: 0 = the flat picture; up to 1 lights it as a relief under
        # `light` (grid-texel space). The mode drives both every frame.
        self.depth = 0.0
        self.light = RELIEF_LIGHT
        self.reseed_frac = reseed_frac
        self.gain = gain                # global multiplier on sense+step (tempo)
        # Anti-thoroughfare levers (the "weave" family — see PhysarumMode):
        self.sat = sat                  # sensor saturation: sensed trail is
                                        # softly capped at sat x the trail's
                                        # bright end, so a fat vein stops
                                        # out-competing thin ones. 0 = off.
        self.jitter = jitter            # random per-step heading wobble (rad)
        self.hetero = hetero            # 0..1 blend toward a 3-sub-population
                                        # sense-range split (short/mid/long)
        self.cross = float(cross)       # off-diagonal strength of the species
                                        # matrix (0 = three independent molds)
        self.sharpen = float(sharpen)   # lateral inhibition in the diffusion
                                        # pass (centre-surround, not a smear)
        self.mosaic = float(mosaic)     # 0..1 spatial parameter mosaic strength
        self.zones = float(zones)       # zone lattice density across the grid
        # slow structural modulation (the mode's `evolve` drives these):
        # multipliers on the blended point parameters — moving the
        # sense/turn/spread/step geometry re-organizes the network topology,
        # where gain alone only re-scales its tempo
        self.mod_sense = 1.0
        self.mod_turn = 1.0
        self.ballistic = 0.0   # steering suppression, decays after a wave
        self.mod_spread = 1.0
        self.mod_step = 1.0
        self.mod_deposit = 1.0
        self._rng = rng
        self._laid = np.zeros((gh, gw, 3), np.float32)
        self._norm = 0.0                # last luminance() percentile (sat cap ref)
        self._flat_lum = None           # last luminance() before relief (lum_sample)
        # per-agent sense-range multiplier groups for `hetero` (1/3 each)
        self._sense_group = HETERO_MULTS[np.arange(n) % 3].astype(np.float32)
        self.frame = 0                  # drives the zone lattice's slow drift
        self._zmap = None               # (gh, gw) int8 regime index per pixel
        self._zmap_frame = -10 ** 9
        self._zmap_zones = None
        self._zhash = None              # (Z, Z, 3) per-lattice-cell hash
        # the zone lattice must not perturb the agent stream's determinism
        self._zrng_seed = int(seed) ^ 0x5A0E

    # ----- the picture, and its species decomposition -----
    @property
    def trail_total(self):
        """The whole organism: the species channels summed, (gh, gw) float32.

        The picture, the p95 normalization and the `sat` reference all run on
        this — matching stats.frag / tonemap.frag, which sum .rgb."""
        tr = self.trail
        return tr[..., 0] + tr[..., 1] + tr[..., 2]

    @property
    def trail_species(self):
        """Per-species trail, (gh, gw, 3) float32 — mirrors PhysarumFieldGL."""
        return self.trail

    def species_of(self):
        """Per-agent species index, int32 (n,) — mirrors PhysarumFieldGL."""
        return self._species.copy()

    def interaction_matrix(self):
        """This field's species matrix — see `species_matrix`."""
        return species_matrix(self.species, self.cross)

    # ----- the spatial mosaic -----
    def _zone_mults(self):
        """(gh*gw, 5) float32 of per-pixel regime multipliers — sense, turn,
        spread, step, cross — rebuilt every ZONE_REFRESH frames.

        This is update.frag's zone_at() rasterized. Same lattice: `zones`
        sites across the grid, each drifting on its own slow sine, wrapping on
        the torus, nearest site wins. Three details carry the character:

        - **Domain warp.** A Voronoi diagram is made of straight lines, and a
          straight line across a slime mold reads instantly as machinery. Two
          cheap sine octaves bend the boundaries into something the organism
          could have grown. In lattice units, so it scales with zone size.
        - **A zone picks ONE regime outright**, not a blend of six —
          continuous multipliers average back into a single texture.
        - **...except on the seam itself**, where the nearest and second
          nearest zones cross-fade over a narrow band, so the transition is a
          front rather than a visible polygon edge.

        The GPU does this per agent per frame; here it is a map, because the
        lattice drifts at 0.07-0.17 rad/s and a 20-frame-old seam has moved
        about 0.02 of a cell. Storing the BLENDED multipliers rather than a
        zone index folds the seam cross-fade in for free and leaves one
        gather per agent instead of two plus a mix."""
        z = max(float(self.zones), 1.0)
        if (self._zmap is not None and self._zmap_zones == z
                and self.frame - self._zmap_frame < ZONE_REFRESH):
            return self._zmap
        gw, gh = self.gw, self.gh
        nz = max(1, int(round(z)))
        if self._zhash is None or self._zhash.shape[0] != nz:
            self._zhash = np.random.default_rng(
                self._zrng_seed ^ (nz * 0x9E3779B9)).random(
                    (nz, nz, 3)).astype(np.float32)
        step = max(1, int(np.ceil(max(gw, gh) / float(ZONE_MAP_MAX))))
        cw, ch = max(1, gw // step), max(1, gh // step)
        t = np.float32(self.frame * (1.0 / 60.0))
        nx0 = (((np.arange(cw, dtype=np.float32) + 0.5) * (gw / cw)) / gw * z)
        ny0 = (((np.arange(ch, dtype=np.float32) + 0.5) * (gh / ch)) / gh * z)
        # the warp reads the UNWARPED other axis (GLSL evaluates the whole
        # vec2 before the +=), so each offset stays one-dimensional
        wx = ZONE_WARP * (np.sin(ny0 * 2.7 + t * 0.05)
                          + 0.5 * np.sin(ny0 * 6.1 - t * 0.031))
        wy = ZONE_WARP * (np.cos(nx0 * 2.3 - t * 0.043)
                          + 0.5 * np.cos(nx0 * 5.3 + t * 0.037))
        NX = nx0[None, :] + wx[:, None]
        NY = ny0[:, None] + wy[None, :]
        i0x = np.floor(NX).astype(np.int32)
        i0y = np.floor(NY).astype(np.int32)
        # the drift is a property of the LATTICE CELL, not of the pixel, and
        # there are only nz*nz cells: evaluate the sines once per cell here
        # instead of once per sample per neighbour (18 transcendentals over
        # the whole raster, which dominated the rebuild)
        r0t, r1t, r2t = (self._zhash[..., 0], self._zhash[..., 1],
                         self._zhash[..., 2])
        dxt = (0.42 * np.sin(t * (0.09 + 0.08 * r0t) + TAU * r1t)).ravel()
        dyt = (0.42 * np.cos(t * (0.07 + 0.07 * r1t) + TAU * r2t)).ravel()
        r0t = r0t.ravel()
        big = np.float32(1e18)
        best = np.full((ch, cw), big, np.float32)
        nxt = np.full((ch, cw), big, np.float32)
        hv = np.full((ch, cw), 0.5, np.float32)
        hv2 = np.full((ch, cw), 0.5, np.float32)
        for dy in (-1, 0, 1):
            gy = i0y + dy
            wyi = np.mod(gy, nz) * nz
            gyf = gy.astype(np.float32) + 0.5
            for dx in (-1, 0, 1):
                gx = i0x + dx
                ci = wyi + np.mod(gx, nz)                      # flat cell id
                r0 = r0t[ci]
                sx = gx.astype(np.float32) + 0.5 + dxt[ci]
                sy = gyf + dyt[ci]
                ddx, ddy = NX - sx, NY - sy
                dd = ddx * ddx + ddy * ddy
                win = dd < best
                run = (~win) & (dd < nxt)
                nxt = np.where(win, best, np.where(run, dd, nxt))
                hv2 = np.where(win, hv, np.where(run, r0, hv2))
                best = np.where(win, dd, best)
                hv = np.where(win, r0, hv)
        # 0 deep inside a zone, 0.5 on the seam: narrow, so the seam stays a
        # seam and not a gradient
        gap = np.sqrt(np.maximum(nxt, 0.0)) - np.sqrt(np.maximum(best, 0.0))
        u = np.clip(gap * np.float32(1.0 / ZONE_EDGE), 0.0, 1.0)
        edge = (0.5 * (1.0 - u * u * (3.0 - 2.0 * u))).astype(np.float32)
        za = np.clip(np.floor(hv * ZONE_REGIME_COUNT),
                     0, ZONE_REGIME_COUNT - 1).astype(np.int32)
        zb = np.clip(np.floor(hv2 * ZONE_REGIME_COUNT),
                     0, ZONE_REGIME_COUNT - 1).astype(np.int32)
        e = edge[..., None]
        m4 = ZONE_REGIME_MULTS[za] * (1.0 - e) + ZONE_REGIME_MULTS[zb] * e
        mc = (ZONE_REGIME_CROSS[za] * (1.0 - edge)
              + ZONE_REGIME_CROSS[zb] * edge)
        small = np.concatenate([m4, mc[..., None]], axis=2).astype(np.float32)
        full = cv2.resize(small, (gw, gh), interpolation=cv2.INTER_LINEAR)
        self._zmap = np.ascontiguousarray(
            full.reshape(gh * gw, 5).T)                    # (5, gh*gw)
        self._zmap_frame = self.frame
        self._zmap_zones = z
        return self._zmap

    # ----- parameter blending -----
    def _blend_params(self, t):
        """Per-agent parameter arrays: background point blended toward the
        subject point by t = matte value under each agent (the pen)."""
        a, b = POINTS[self.point_bg], POINTS[self.point_fg]
        out = {}
        for k in _PARAM_KEYS:
            lo, hi = np.float32(a[k]), np.float32(b[k])
            out[k] = lo + (hi - lo) * t
        if self.mod_sense != 1.0:
            out["sense"] = out["sense"] * np.float32(self.mod_sense)
        if self.mod_turn != 1.0:
            out["turn"] = out["turn"] * np.float32(self.mod_turn)
        if self.mod_spread != 1.0:
            out["spread"] = out["spread"] * np.float32(self.mod_spread)
        if self.mod_step != 1.0:
            out["step"] = out["step"] * np.float32(self.mod_step)
        if self.mod_deposit != 1.0:
            out["deposit"] = out["deposit"] * np.float32(self.mod_deposit)
        return out

    def swap_points(self):
        self.point_bg, self.point_fg = self.point_fg, self.point_bg

    # ----- one simulation frame -----
    def _diffuse(self):
        # diffuse + decay. A plain box blur is the most structure-destroying
        # kernel there is at a given radius: it only ever smears. `sharpen`
        # subtracts a slice of the WIDER surround, turning diffusion into a
        # centre-surround operator, so a strong vein suppresses its own
        # neighbourhood — sharpening the vein and digging the dark halo around
        # it. Those halos are most of what reads as carved rather than smoked.
        #
        # The inhibition is SPLIT across the two axes, HALF STRENGTH ON EACH,
        # matching physarum_gl._blur_decay, which sets u_sharpen to
        # sharpen*0.5 on both the H and the V pass. On one axis only it is a
        # directional operator and the picture laminates along it; at full
        # strength on both it doubles the operator and collapses into a
        # pixel-scale Turing dot pattern. Half and half is the isotropic one.
        r = int(self.diffuse) if self.diffuse > 0 else 0
        sh = max(float(self.sharpen), 0.0)
        if r > 0 or sh > 0:
            k = 2 * r + 1
            if sh > 0:
                rw = max(3 * r, r + 2)
                w = 2 * rw + 1
                half = np.float32(sh * 0.5)

                def centre_surround(img, ksz, wsz):
                    blurred = cv2.boxFilter(img, -1, ksz)
                    surround = cv2.boxFilter(img, -1, wsz)
                    return np.maximum(blurred - half * (surround - blurred),
                                      np.float32(0.0))

                # cv2 ksize is (width, height): (k, 1) is the H pass.
                tmp = centre_surround(self.trail, (k, 1), (w, 1))
                self.trail = centre_surround(tmp, (1, k), (1, w))
            else:
                self.trail = cv2.boxFilter(self.trail, -1, (k, k))

    def update(self, matte, gray, keep=None):
        """matte, gray: float32 (gh, gw) in [0,1]. Advances agents one frame
        and rebuilds the trail map.

        `keep` (optional, same shape, [0,1]): per-pixel decay boost — where
        keep is 1 the trail decays at 0.995 instead of `decay`, so swept
        paths linger (the mode's react machinery paints it from motion)."""
        gw, gh = self.gw, self.gh
        n = self.n
        px, py, h = self.px, self.py, self.heading

        # float32 modulo can land exactly ON the upper bound (a hair below gw
        # rounds up in float32), so every grid index wraps in INT space
        cy = py.astype(np.int32) % gh
        cx = px.astype(np.int32) % gw
        t = matte[cy, cx]
        p = self._blend_params(t)
        sense_d = p["sense"] * self.gain
        step_d = p["step"] * self.gain
        turn_a = p["turn"]
        spread_a = p["spread"]
        if self.hetero > 0:
            # blend each agent's sense range toward its sub-population's
            sense_d = sense_d * (1.0 + (self._sense_group - 1.0) * self.hetero)

        # ----- the spatial mosaic -----
        # One parameter set over the whole canvas grows one texture, and no
        # slider changes that. The zone an agent stands in picks ONE regime
        # outright and multiplies its geometry (and its species cross-term) by
        # that regime's entries, mixed in by `mosaic` — the same
        # mix(1.0, r, u_mosaic) update.frag does. Hard seams, not a blend: the
        # blend is what averages six characters back into one.
        cross_base = max(float(self.cross), 0.0) if self.species > 1 else 0.0
        cross = cross_base
        mos = min(max(float(self.mosaic), 0.0), 1.0)
        if mos > 0.0:
            reg = self._zone_mults()
            here = cy * gw + cx
            m = np.float32(mos)
            one = np.float32(1.0)

            def zmul(row, base):
                return base * (one + (reg[row][here] - one) * m)
            sense_d = zmul(0, sense_d)
            turn_a = zmul(1, turn_a)
            spread_a = zmul(2, spread_a)
            step_d = zmul(3, step_d)
            if cross > 0.0:
                cross = zmul(4, cross)
                # The zones scale it; do not let it run away. Mirrors
                # update.frag's `repel = min(repel, 1.5)`. Unreachable today
                # (max is cross<=1 x the `mesh` regime's 1.45) but the shader
                # grew the clamp and this did not, which is exactly the kind
                # of silent asymmetry the shared species_matrix() exists to
                # prevent — a latent one is still one.
                cross = np.minimum(cross, np.float32(REPEL_MAX))
        # ...but never past a tenth of the frame. `web` already sits at the
        # ceiling by design, and the mosaic's long-range regimes would
        # otherwise multiply it past the point where any local structure can
        # survive — the sensors stop reporting on a neighbourhood at all.
        sense_max = np.float32(SENSE_MAX_FRAC * min(gw, gh))
        sense_d = np.minimum(sense_d, sense_max)

        # ----- what the sensors read -----
        # Species s reads dot(row_s, trail[p]): its own channel attracts, the
        # others pull with the signed weights of interaction_matrix(). Written
        # as total - cross * (0.65*previous + 2*next), which is that row
        # rearranged: at cross 0 it collapses to the plain channel SUM, i.e.
        # the legacy single-channel field. Folding the two channel
        # permutations into one plane means a sample is
        # two gathers instead of a 3x3 mix of the whole grid.
        tr_f = self.trail.reshape(-1)                # index cell*3 + species
        sp = self._species
        # `multi` decides WHICH field is read, `use_cross` only whether the
        # repulsion term is worth computing. They are not the same switch: at
        # cross 0 a three-species field still reads the channel SUM (the
        # legacy single-channel model), never its own channel alone.
        multi = self.species > 1
        use_cross = multi and cross_base > 0.0
        mix_f = tot_f = None
        if multi:
            tot_f = self.trail.sum(axis=2, dtype=np.float32).reshape(-1)
        if use_cross:
            mix_f = (np.float32(1.0 - CROSS_BACK) * self.trail[..., [2, 0, 1]]
                     + np.float32(2.0) * self.trail[..., [1, 2, 0]]).reshape(-1)
        # `sat` softly caps the sensed trail at sat x its own bright end, so a
        # saturated fat vein reads the same as a merely strong thin one — the
        # single strongest anti-thoroughfare lever. The dot product is SIGNED
        # once the off-diagonals bite, and exp(-t/C) on a negative t diverges,
        # so cap the MAGNITUDE and keep the sign (update.frag's `food`).
        cap = float(self.sat) * self._norm
        # food is in TRAIL units (see physarum_gl._food_norm): gray is
        # [0,1] and the trail is ~33 at equilibrium, so an unscaled
        # term cannot steer anything once a network exists.
        fnorm = self._norm if self._norm > 0.0 else 1.0
        gfood = (None if self.food <= 0
                 else ((self.food * fnorm) * gray).reshape(-1))

        def _sample(angle_off):
            sx = px + np.cos(h + angle_off) * sense_d
            sy = py + np.sin(h + angle_off) * sense_d
            c = (sy.astype(np.int32) % gh) * gw + (sx.astype(np.int32) % gw)
            i3 = c * 3 + sp
            if use_cross:
                v = tot_f[c] - cross * mix_f[i3]
            elif multi:
                v = tot_f[c]
            else:
                v = tr_f[i3]
            if cap > 0:
                a = np.abs(v)
                np.exp(a * np.float32(-1.0 / cap), out=a)
                v = np.copysign(np.float32(cap) * (1.0 - a), v)
            if gfood is not None:
                v = v + gfood[c]
            return v

        f_c = _sample(0.0)
        f_l = _sample(-spread_a)
        f_r = _sample(spread_a)

        # Jones steering: hold when ahead wins; flip a coin when ahead loses to
        # both sides; otherwise turn toward the stronger side.
        rand_sign = np.where(self._rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        turn = np.where(
            (f_c > f_l) & (f_c > f_r), 0.0,
            np.where((f_c < f_l) & (f_c < f_r), rand_sign,
                     np.where(f_l > f_r, -1.0, 1.0))).astype(np.float32)
        # The wave's ballistic phase: for a beat after a wave, steering and
        # wobble are suppressed so the outward front actually travels instead
        # of being steered back within a frame or two. Decays every update.
        b = np.float32(1.0 - min(max(self.ballistic, 0.0), 1.0))
        h += turn * (turn_a * b)
        if self.jitter > 0:
            # per-step heading wobble: highways stop being perfectly straight
            # attractors and the mold keeps probing sideways
            h += ((self._rng.random(n, dtype=np.float32) * 2.0 - 1.0)
                  * np.float32(self.jitter) * b)

        px += np.cos(h) * step_d
        py += np.sin(h) * step_d
        px %= gw
        py %= gh

        # deposit — one bincount over flattened grid indices, the cheap way to
        # scatter-add N agents without np.add.at's per-element dispatch. Each
        # agent lands in its OWN species channel (deposit.vert's one-hot
        # v_mask), so the index carries the species in its low digit.
        idx = ((py.astype(np.int32) % gh) * gw
               + (px.astype(np.int32) % gw)) * 3 + sp
        laid = np.bincount(idx, weights=p["deposit"], minlength=gw * gh * 3)
        self._laid = laid.reshape(gh, gw, 3).astype(np.float32)
        self.trail += self._laid

        self._diffuse()
        if keep is None:
            self.trail *= self.decay
        else:
            k = np.clip(keep, -1.0, 1.0).astype(np.float32)
            eff = (np.float32(self.decay)
                   + np.float32(KEEP_HOLD - self.decay) * np.maximum(k, 0.0)
                   - np.float32(MELT_DROP) * np.maximum(-k, 0.0))
            self.trail *= np.clip(eff, np.float32(MELT_FLOOR),
                                  np.float32(KEEP_HOLD))[..., None]

        # recycle a trickle of agents onto the lit subject, so the network
        # keeps finding whoever is in frame instead of ossifying
        budget = int(self.reseed_frac * n)
        if budget > 0:
            pick = self._rng.integers(0, n, budget)
            rx, ry = self._sample_weighted(matte * np.clip(gray, 0.05, 1.0), budget)
            self.px[pick] = rx
            self.py[pick] = ry
            self.heading[pick] = self._rng.uniform(0, 2 * np.pi, budget).astype(np.float32)
            # species is NOT redrawn: a reseed relocates a lineage, it does
            # not replace it (update.frag carries a.w through untouched)
        self.ballistic *= BALLISTIC_DECAY
        self.frame += 1

    def _sample_weighted(self, weight, k):
        w = weight.ravel().astype(np.float64)
        s = w.sum()
        if s <= 0:
            return (self._rng.uniform(0, self.gw, k).astype(np.float32),
                    self._rng.uniform(0, self.gh, k).astype(np.float32))
        cdf = np.cumsum(w) / s
        idx = np.searchsorted(cdf, self._rng.random(k))
        gy_i, gx_i = np.divmod(idx, self.gw)
        return ((gx_i + self._rng.random(k)).astype(np.float32),
                (gy_i + self._rng.random(k)).astype(np.float32))

    # ----- interactions -----
    def spawn_burst(self, x, y, frac=0.08, radius=6.0):
        """Teleport a fraction of the pool into a tight gaussian at (x, y)
        with fresh random headings — Bleuje-style particle spawning, the
        theatrical 'pour more mold HERE' move."""
        k = int(frac * self.n)
        if k <= 0:
            return
        pick = self._rng.integers(0, self.n, k)
        self.px[pick] = (x + self._rng.normal(0, radius, k)).astype(np.float32) % self.gw
        self.py[pick] = (y + self._rng.normal(0, radius, k)).astype(np.float32) % self.gh
        self.heading[pick] = self._rng.uniform(0, 2 * np.pi, k).astype(np.float32)

    def wave(self, x, y):
        """Point every agent's heading away from (x, y), and hold it there.

        The impulse alone is one frame of new headings, and steering takes
        most of them back before the front has gone anywhere. Arming the
        ballistic phase suppresses steering while it decays, so the ring
        actually travels; then the mold reknits.
        """
        self.heading = np.arctan2(self.py - y, self.px - x).astype(np.float32)
        self.ballistic = 1.0

    def gather(self, x, y, frac=0.5, radius=60.0):
        """Rush agents already within `radius` of (x, y) into a tight knot
        there — a LOCAL impulse that leaves the rest of the organism alone
        (the react 'spell' move; spawn_burst teleports from the whole pool)."""
        if frac <= 0 or radius <= 0:
            return
        dx = self.px - np.float32(x)
        dy = self.py - np.float32(y)
        dx -= self.gw * np.floor(dx / self.gw + 0.5)     # shortest torus offset
        dy -= self.gh * np.floor(dy / self.gh + 0.5)
        near = dx * dx + dy * dy < np.float32(radius * radius)
        pick = np.flatnonzero(near & (self._rng.random(self.n) < frac))
        k = pick.size
        if k == 0:
            return
        r = 0.15 * radius
        self.px[pick] = (x + self._rng.normal(0, r, k)).astype(np.float32) % self.gw
        self.py[pick] = (y + self._rng.normal(0, r, k)).astype(np.float32) % self.gh
        self.heading[pick] = self._rng.uniform(0, 2 * np.pi, k).astype(np.float32)

    # ----- lifecycle -----
    def release(self):
        """Nothing to free — here so both engines share one stop() contract."""

    # ----- picture -----
    def luminance(self):
        """Tonemapped trail in [0,1] float32 (gh, gw) — the mode colorizes it.

        The raw trail's scale depends on agent density and decay (equilibrium
        is roughly agents-per-cell x deposit x decay/(1-decay)), so it is
        normalized by its own bright end before the exposure curve — the
        exposure slider then means the same thing at any agent count.

        `grain` mixes the CURRENT frame's raw deposits (pre-blur agent
        positions) over the smooth diffused trail — that per-frame dust is
        what makes the organism look granular and alive instead of airbrushed."""
        tr = self.trail
        total = tr[..., 0] + tr[..., 1] + tr[..., 2]
        norm = float(np.percentile(total, 95.0))
        # Smooth it, exactly as PhysarumFieldGL does with NORM_EMA. A raw
        # per-frame p95 renormalizes the picture against its own noise every
        # frame, which reads as a slow pump — and it also feeds the `sat` and
        # `food` scales, so an unsmoothed reference makes the SENSING wobble
        # too. The two engines used different references here until this
        # comment was written, which is precisely the kind of quiet divergence
        # "both engines implement the same model" is supposed to exclude.
        if self._norm > 0.0 and norm > 0.0:
            norm = self._norm * NORM_EMA + norm * (1.0 - NORM_EMA)
        self._norm = norm               # feedback for the `sat` sensing cap
        if norm <= 0:
            self._flat_lum = None
            return np.zeros((self.gh, self.gw), np.float32)
        x = total * (1.0 / norm)
        if self.grain > 0:
            ld = self._laid
            laid = ld[..., 0] + ld[..., 1] + ld[..., 2]
            gnorm = float(laid.mean()) * 4.0
            if gnorm > 0:
                x = x + self.grain * (laid * (1.0 / gnorm))
        lum = (1.0 - np.exp(-self.exposure * x)).astype(np.float32)
        # kept BEFORE the relief: the relief is lighting on the picture, not
        # a property of the organism, so the mode's staleness tracker reads
        # this flat copy (lum_sample) and depth cannot move the evolve melt
        self._flat_lum = lum
        if self.depth > 0.0:
            lum = relief(lum, total, 1.0 / norm, float(self.depth),
                         relief_light(self.light))
        return lum

    def lum_sample(self):
        """The last luminance() picture WITHOUT the depth relief, (gh, gw)
        float32 in [0,1], or None before the first luminance() / on an empty
        trail. Same contract as PhysarumFieldGL.lum_sample: the staleness
        tracker's input, which must see the organism rather than the light
        orbiting over it (a relief-lit input pulls the tracker's mean under
        the melt gate and adds the orbit's variance, so depth would weaken
        the melt)."""
        return self._flat_lum
