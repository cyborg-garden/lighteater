"""Dithering algorithms — ordered (Bayer, blue-noise) and error-diffusion
(Floyd-Steinberg, Riemersma).

All algorithms are public-domain concepts; these are independent
reimplementations. Inputs/outputs are float32 arrays in [0, 1]. Works on
grayscale (H, W) or colour (H, W, C) arrays.

Gamma correctness: by default (``gamma=True``) inputs are treated as
sRGB-encoded, linearized before quantisation, and re-encoded on output —
dithering happens in linear light, so dithered mid-tones keep their perceived
brightness. Pass ``gamma=False`` for the historical treat-sRGB-as-linear
behaviour (visibly darker mid-tones at low bit depths).

GPU note: ordered dithering (Bayer, blue-noise) is a pure per-pixel threshold
test against a tiled texture — trivially parallel and shader-friendly. Error
diffusion (Floyd-Steinberg, Riemersma) carries state from pixel to pixel —
inherently sequential, CPU-only.
"""
from __future__ import annotations

import os

import numpy as np

try:                     # cv2.LUT is SIMD + threaded — ~15× faster than numpy
    import cv2 as _cv2   # fancy indexing at 720p. The project depends on
except ImportError:      # OpenCV anyway; the fallback keeps this module
    _cv2 = None          # usable standalone.


# ---------------------------------------------------------------------------
# sRGB transfer functions
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB-encoded values [0, 1] → linear light [0, 1].

    Piecewise sRGB EOTF (IEC 61966-2-1): linear segment below 0.04045,
    γ=2.4 power segment above. Vectorised; returns float32.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045,
                    x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light [0, 1] → sRGB-encoded values [0, 1].

    Inverse of :func:`srgb_to_linear` (piecewise, γ=2.4 segment).
    Vectorised; returns float32.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308,
                    x * 12.92,
                    1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


# 256-entry forward LUT: the pipeline's frames come from uint8, so quantising
# the float input to 8-bit sRGB code values before linearizing loses nothing
# and replaces a per-pixel np.power with a table lookup (~5× faster at 720p).
# Cached per scale factor so the ordered core can fold its ``* levels``
# multiply into the same lookup (scale 1.0 = plain linearization).
_SRGB2LIN_LUTS: dict[float, np.ndarray] = {}


def _srgb2lin_lut(scale: float = 1.0) -> np.ndarray:
    lut = _SRGB2LIN_LUTS.get(scale)
    if lut is None:
        lut = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
        lut = (lut * np.float32(scale)).astype(np.float32)
        _SRGB2LIN_LUTS[scale] = lut
    return lut


def _lut_apply(u8: np.ndarray, lut256: np.ndarray) -> np.ndarray:
    """Apply a 256-entry float32 LUT to a uint8 array."""
    if _cv2 is not None:
        # reshape: cv2 drops a trailing singleton channel dim on (H, W, 1).
        return _cv2.LUT(u8, lut256).reshape(u8.shape)
    return lut256[u8]


def _to_u8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] → uint8 code values (round-half-up via truncation)."""
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _linearize(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """float [0, 1] sRGB → linear light (× *scale*) via the 256-entry LUT."""
    return _lut_apply(_to_u8(img), _srgb2lin_lut(scale))


# Level-encoding LUTs, cached per (levels, gamma): 256 entries so cv2.LUT
# applies; only the first levels+1 slots are ever indexed.
_ENCODE_LUTS: dict[tuple[int, bool], np.ndarray] = {}


def _encode_levels(idx: np.ndarray, levels: int, gamma: bool) -> np.ndarray:
    """Map quantised level indices [0, levels] (uint8) to output float32.

    With gamma, level values are re-encoded from linear light back to sRGB via
    the LUT — the output holds only ``levels + 1`` distinct values, so a tiny
    table is exact and avoids a full-frame np.power.
    """
    lut = _ENCODE_LUTS.get((levels, gamma))
    if lut is None:
        lut = np.zeros(256, dtype=np.float32)
        lut[:levels + 1] = np.arange(levels + 1, dtype=np.float32) / float(levels)
        if gamma:
            lut[:levels + 1] = linear_to_srgb(lut[:levels + 1])
            lut[levels] = 1.0   # 1.055 - 0.055 rounds to 0.99999994 in float32
        _ENCODE_LUTS[(levels, gamma)] = lut
    if idx.dtype != np.uint8:
        idx = idx.astype(np.uint8)
    return _lut_apply(idx, lut)


# A frame is "dark" (for invert="auto") when its mean is below mid-grey in the
# working domain. Mid-grey is sRGB 0.5 — which is ~0.214 in linear light, not
# 0.5: linear mean 0.5 would be sRGB ~0.735, classifying nearly every real
# frame as dark.
_MID_GREY_LINEAR = float(srgb_to_linear(np.float32(0.5)))   # ≈ 0.214


# ---------------------------------------------------------------------------
# Ordered dithering (Bayer, blue-noise)
# ---------------------------------------------------------------------------

def _bayer_matrix(n: int) -> np.ndarray:
    """Recursively generate a normalised n×n Bayer threshold matrix.

    n must be a power of 2. If sub = M(n)/n² is the normalised n×n matrix,
    the normalised 2n×2n matrix is:

        [[sub,        sub + 2/N²],
         [sub + 3/N², sub + 1/N²]]    where N² = (2n)²

    This is derived from the standard unnormalised recursion
    M(2n) = [[4·M(n), 4·M(n)+2], [4·M(n)+3, 4·M(n)+1]] by dividing both
    sides by N² = 4n² and using sub = M(n)/n².
    """
    if n == 1:
        return np.array([[0.0]], dtype=np.float32)
    sub = _bayer_matrix(n // 2)    # already normalised by (n//2)²
    N2 = float(n * n)              # normalisation factor for the output matrix
    return np.block([
        [sub,             sub + 2.0 / N2],
        [sub + 3.0 / N2,  sub + 1.0 / N2],
    ]).astype(np.float32)


def _ordered_dither(img: np.ndarray, mat: np.ndarray, bits: int,
                    invert: bool | str, gamma: bool) -> np.ndarray:
    """Shared tile-threshold-quantise core for ordered dithering: the level
    indices of :func:`_ordered_indices`, encoded to output values."""
    idx, _ = _ordered_indices(img, mat, bits, invert, gamma)
    return _encode_levels(idx, (1 << bits) - 1, gamma)


def _ordered_indices(img: np.ndarray, mat: np.ndarray, bits: int,
                     invert: bool | str, gamma: bool) -> tuple[np.ndarray, bool]:
    """Tile-threshold-quantise to uint8 level indices [0, levels], plus the
    resolved invert flag (``"auto"`` decided on this frame's mean).

    Split out of :func:`_ordered_dither` so the browser port's goldens
    (``python -m dtouch.dither_looks``) can pin the indices themselves.

    ``floor(p · levels + t)`` rounds *down* on average (a value between two
    levels lands on the lower one more often than its fraction warrants), so
    dark frames lose their dim detail to black. ``invert`` flips the
    comparison by dithering the photographic negative and inverting back —
    exactly mirrored spatial pattern, upward-rounding density — which
    preserves dim detail. ``"auto"`` flips per frame when the working-domain
    mean is below mid-grey (see ``_MID_GREY_LINEAR``).
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1–8, got {bits}")

    levels = (1 << bits) - 1
    h, w = img.shape[:2]
    # ``work`` carries p * levels (the × levels multiply is folded into the
    # linearization LUT on the gamma path — one fewer full-frame pass).
    if gamma:
        work = _linearize(img, scale=float(levels))
    else:
        work = np.clip(img.astype(np.float32), 0.0, 1.0) * levels

    if invert == "auto":
        mean = float(work.mean()) / levels
        invert = mean < (_MID_GREY_LINEAR if gamma else 0.5)

    mh, mw = mat.shape
    ty = int(np.ceil(h / mh))
    tx = int(np.ceil(w / mw))
    threshold = np.tile(mat, (ty, tx))[:h, :w]   # (H, W) in [0, 1)
    if img.ndim == 3:
        threshold = threshold[:, :, np.newaxis]

    if invert:
        work = levels - work
    # floor(p * levels + threshold) maps uniformly-distributed threshold in
    # [0,1) to a dithering pattern that preserves p=0 → 0 and p=1 → 1 exactly.
    # Both operands are ≥ 0 and their sum is < levels + 1 ≤ 256, so the uint8
    # cast's truncation *is* the floor and no output clip is needed.
    idx = (work + threshold).astype(np.uint8)
    if invert:
        np.subtract(np.uint8(levels), idx, out=idx)
    return idx, bool(invert)


def bayer_dither(img: np.ndarray, bits: int = 2, matrix_size: int = 4,
                 invert: bool | str = "auto", gamma: bool = True) -> np.ndarray:
    """Ordered dithering via Bayer threshold matrix.

    Fast and fully vectorised — suitable for real-time use. Preserves exact
    black (0.0) and white (1.0): the threshold bias is additive in the
    quantisation domain so floor(1.0 * levels + t) / levels = 1 for all t<1.

    GPU-suitability: excellent — a pure per-pixel threshold test against a
    tiled texture; ports to a one-line fragment shader (one texture fetch +
    step). No inter-pixel dependency.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)
    matrix_size : Bayer matrix side length; must be a power of 2 (2, 4, 8, …)
    invert : bool or "auto"
        False = standard threshold comparison (rounds down on average —
        crushes dim detail on dark frames). True = flipped comparison (rounds
        up). "auto" (default) flips when the frame mean is below mid-grey in
        the working domain (linear ~0.214 with gamma on, 0.5 off).
    gamma : dither in linear light (default). False = historical behaviour.

    Returns
    -------
    float32, same shape as *img*, values quantised to ``2**bits`` levels.
    """
    if matrix_size < 1 or (matrix_size & (matrix_size - 1)) != 0:
        raise ValueError(f"matrix_size must be a power of 2, got {matrix_size}")
    return _ordered_dither(img, _bayer_matrix(matrix_size), bits, invert, gamma)


# Blue-noise threshold texture: 64×64 float32 rank matrix in [0, 1 - 1/4096],
# generated offline by scripts/gen_bluenoise.py (void-and-cluster) and shipped
# as a package asset. Lazy-loaded once, then tiled exactly like Bayer.
_BLUE_NOISE: np.ndarray | None = None
_BLUE_NOISE_PATH = os.path.join(os.path.dirname(__file__), "assets",
                                "bluenoise64.npy")


def _blue_noise_matrix() -> np.ndarray:
    global _BLUE_NOISE
    if _BLUE_NOISE is None:
        _BLUE_NOISE = np.load(_BLUE_NOISE_PATH).astype(np.float32)
    return _BLUE_NOISE


def blue_noise_dither(img: np.ndarray, bits: int = 2,
                      invert: bool | str = "auto",
                      gamma: bool = True) -> np.ndarray:
    """Ordered dithering via a precomputed 64×64 blue-noise threshold texture.

    Same tile-threshold core (and cost) as :func:`bayer_dither`, but the
    threshold map has no low-frequency energy, so the pattern reads as
    organic grain instead of Bayer's crosshatch. Texture generated offline
    with the void-and-cluster method (see scripts/gen_bluenoise.py).

    GPU-suitability: excellent — a pure per-pixel threshold test against a
    tiled texture; ports to a one-line fragment shader (one texture fetch +
    step). No inter-pixel dependency.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)
    invert : bool or "auto" — see :func:`bayer_dither`.
    gamma : dither in linear light (default). False = historical behaviour.

    Returns
    -------
    float32, same shape as *img*, values quantised to ``2**bits`` levels.
    """
    return _ordered_dither(img, _blue_noise_matrix(), bits, invert, gamma)


# ---------------------------------------------------------------------------
# Error-diffusion dithering (Floyd-Steinberg, Riemersma)
# ---------------------------------------------------------------------------

def floyd_steinberg(img: np.ndarray, bits: int = 2,
                    gamma: bool = True) -> np.ndarray:
    """Floyd-Steinberg error-diffusion dithering.

    Distributes quantisation error to four right/below neighbours using the
    classic 7/16 · 3/16 · 5/16 · 1/16 kernel. Sequential by nature — best
    suited to small images or offline use; use :func:`bayer_dither` for
    real-time paths.

    GPU-suitability: poor — each pixel depends on errors diffused from all
    previously visited pixels; the raster-order dependency chain defeats
    parallelisation.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)
    gamma : diffuse in linear light (default). False = historical behaviour.

    Returns
    -------
    float32, same shape as *img*, dithered.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1–8, got {bits}")

    levels = float((1 << bits) - 1)
    squeezed = img.ndim == 2
    src = _linearize(img) if gamma else img.astype(np.float32)
    if squeezed:
        src = src[:, :, np.newaxis]
    h, w, channels = src.shape

    out = np.empty((h, w, channels), dtype=np.float32)
    for c in range(channels):
        # Per-channel raster loop on plain Python floats (rows via tolist):
        # ~20× faster than per-pixel numpy scalar ops at live working sizes,
        # same diffusion arithmetic. round() is banker's rounding, matching
        # the previous np.round behaviour.
        rows = src[:, :, c].astype(np.float64).tolist()
        for y in range(h):
            row = rows[y]
            below = rows[y + 1] if y + 1 < h else None
            for x in range(w):
                old = row[x]
                new = round(old * levels) / levels
                row[x] = new
                err = old - new
                if x + 1 < w:
                    row[x + 1] += err * (7.0 / 16.0)
                if below is not None:
                    if x > 0:
                        below[x - 1] += err * (3.0 / 16.0)
                    below[x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        below[x + 1] += err * (1.0 / 16.0)
        plane = np.clip(np.asarray(rows, dtype=np.float32), 0.0, 1.0)
        if gamma:
            # Every pixel holds a quantised level after the loop (clipping only
            # snaps out-of-range values onto the end levels), so re-encoding via
            # the small level LUT is exact.
            idx = np.rint(plane * levels).astype(np.int32)
            plane = _encode_levels(idx, int(levels), gamma=True)
        out[:, :, c] = plane
    return (out[:, :, 0] if squeezed else out).astype(np.float32)


# Hilbert traversal order cache: (h, w) → (row_indices, col_indices).
# Building the order is vectorised but not free; live use hits one size.
_HILBERT_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _hilbert_order(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/col indices of every pixel of an h×w image in Hilbert-curve order.

    Computed on the smallest 2^k × 2^k grid covering the image via the
    standard iterative index→(x, y) bit transform (vectorised over all grid
    points), then masked to in-bounds pixels — every pixel is visited exactly
    once, in an order that stays spatially local. Cached per (h, w).
    """
    cached = _HILBERT_CACHE.get((h, w))
    if cached is not None:
        return cached

    side = 1
    while side < max(h, w):
        side *= 2
    t = np.arange(side * side, dtype=np.int64)
    x = np.zeros(side * side, dtype=np.int64)
    y = np.zeros(side * side, dtype=np.int64)
    s = 1
    while s < side:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        # Rotate the sub-quadrant when ry == 0 (mirror first when rx == 1).
        flip = (ry == 0) & (rx == 1)
        xf = np.where(flip, s - 1 - x, x)
        yf = np.where(flip, s - 1 - y, y)
        swap = ry == 0
        x, y = np.where(swap, yf, xf), np.where(swap, xf, yf)
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1

    mask = (y < h) & (x < w)
    result = (y[mask].astype(np.intp), x[mask].astype(np.intp))
    _HILBERT_CACHE[(h, w)] = result
    return result


def riemersma_dither(img: np.ndarray, bits: int = 2, ratio: float = 1.0 / 8.0,
                     history: int = 32, gamma: bool = True) -> np.ndarray:
    """Riemersma dithering — error diffusion along a Hilbert curve.

    Pixels are visited in Hilbert-curve order; each is corrected by an
    exponentially decaying history of the last *history* quantisation errors
    (newest weight 1, oldest weight *ratio*, weights normalised to sum to 1
    so mean luminance is preserved). The space-filling curve breaks up the
    directional worm artefacts of raster-order diffusion. Sequential by
    nature — best suited to small images (e.g. the reduced dither_size path)
    or offline use.

    GPU-suitability: poor — strictly sequential along the Hilbert curve with
    a sliding error history; inherently serial.

    Parameters
    ----------
    img : float32 (H, W) or (H, W, C) in [0, 1]
    bits : output bit depth per channel (1–8)
    ratio : weight of the oldest history entry relative to the newest,
        strictly inside (0, 1) — 1.0 would zero the decay denominator
    history : number of past errors carried along the curve (≥ 2)
    gamma : diffuse in linear light (default). False = historical behaviour.

    Returns
    -------
    float32, same shape as *img*, dithered.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1–8, got {bits}")
    if not 0.0 < ratio < 1.0:
        # ratio=1.0 would make the decay 1.0 and the weight sum divide by zero
        raise ValueError(
            f"ratio must be strictly between 0 and 1 (exclusive), got {ratio}")
    if history < 2:
        raise ValueError(f"history must be >= 2, got {history}")

    levels = (1 << bits) - 1
    squeezed = img.ndim == 2
    src = img[:, :, np.newaxis] if squeezed else img
    h, w, channels = src.shape
    ys, xs = _hilbert_order(h, w)

    decay = ratio ** (1.0 / (history - 1))   # per-step weight decay
    drop = decay ** history                  # weight at which an error retires
    wsum = (1.0 - drop) / (1.0 - decay)      # Σ decay^j, j = 0 … history-1

    out = np.empty((h, w, channels), dtype=np.float32)
    for c in range(channels):
        plane = src[:, :, c]
        work = _linearize(plane) if gamma else plane.astype(np.float32)
        vals = work[ys, xs].astype(np.float64).tolist()
        res = np.empty(len(vals), dtype=np.float64)
        errs = [0.0] * history   # circular buffer of raw errors
        head = 0                 # index of the oldest entry
        acc = 0.0                # running Σ decay^age · err (unnormalised)
        for i, v in enumerate(vals):
            corrected = v + acc / wsum
            q = round(corrected * levels) / levels
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            res[i] = q
            e = corrected - q
            # Age the history one step, retire the oldest error, admit the new.
            acc = acc * decay + e - errs[head] * drop
            errs[head] = e
            head = (head + 1) % history
        plane_out = np.empty((h, w), dtype=np.float64)
        plane_out[ys, xs] = res
        if gamma:
            idx = np.rint(plane_out * levels).astype(np.int32)
            out[:, :, c] = _encode_levels(idx, levels, gamma=True)
        else:
            out[:, :, c] = plane_out.astype(np.float32)
    return (out[:, :, 0] if squeezed else out).astype(np.float32)
