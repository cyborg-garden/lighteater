"""Circuit Bender's post effects on the CPU: the pixel sort, COPY and SPLIT,
and the long exposure's running average.

- `sort_runs` is the classic threshold pixel sort (every stretch of a row or
  column whose brightness sits inside a band is sorted dark to light, stably).
  A port of the browser's sort.js, held byte-identical to it by the goldens.
- `post` is `dtouch/shaders/bender/post.frag` (COPY: block runs and echoed
  bands; SPLIT: chromatic aberration) in numpy. The shader is what the
  browser runs on the GPU; this is the same rule for the desktop's headless
  pipeline. The hash is evaluated in float32 like the shader's; on the GPU
  tested (tests/test_bender_shared.py) the two agree pixel for pixel, but a
  GPU may round its last place differently, so the goldens do not cover it.
  The COPY map depends only on the size, amount and seed (which changes a
  few times a second), so it is built once and reused (`_COPY_CACHE`).
- `fold` is `dtouch/shaders/bender/stack.frag`'s running mean, without the
  RGBA8 dither (the desktop stacks in float32).

"pixel sort", "copy fx" and "chromatic aberration" are names from the
CyberShot Cam guide's web gallery (@lixofuturista / @cebolander); these are
independent implementations of the general techniques, not that project's
code.
"""
from __future__ import annotations

import cv2
import numpy as np


def _luma(rgb):
    """Rec. 601 luma in 0..255, integer maths (x 1024), as sort.js."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (306 * r + 601 * g + 117 * b) >> 10


def sort_runs(px, lo, hi, direction="rows"):
    """Sort RGB uint8 `px` (h, w, 3) in place. `lo`..`hi` (inclusive) is the
    brightness band a run must stay inside; `direction` is 'rows' or
    'columns'. Runs shorter than 2 are left alone; equal brightness keeps its
    order. Returns the number of runs sorted."""
    view = px if direction != "columns" else px.transpose(1, 0, 2)
    lines, length = view.shape[:2]
    flat = view.reshape(lines * length, -1) if view.flags.c_contiguous else None
    work = view.reshape(lines * length, -1) if flat is not None else view.copy().reshape(lines * length, -1)
    y = _luma(work)
    inband = ((y >= lo) & (y <= hi)).reshape(lines, length)
    start = inband.copy()
    start[:, 1:] &= ~inband[:, :-1]
    run_id = np.cumsum(start.reshape(-1))            # unique per run, in order
    idx = np.flatnonzero(inband.reshape(-1))
    if idx.size == 0:
        return 0
    ids = run_id[idx]
    lens = np.bincount(ids)
    runs = int(np.count_nonzero(lens >= 2))
    order = np.argsort(ids.astype(np.int64) * 256 + y[idx], kind="stable")
    work[idx] = work[idx[order]]
    if flat is None:
        view[...] = work.reshape(view.shape)
    return runs


def _hash(x, y, z):
    """post.frag's hash, in float32."""
    p = np.stack(np.broadcast_arrays(np.float32(x), np.float32(y), np.float32(z)), -1).astype(np.float32)
    p = p * np.array([0.1031, 0.1030, 0.0973], dtype=np.float32)
    p = p - np.floor(p)
    d = (p * (p[..., [1, 0, 2]] + np.float32(33.33))).sum(-1, dtype=np.float32)
    p = p + d[..., None]
    v = (p[..., 0] + p[..., 1]) * p[..., 2]
    return (v - np.floor(v)).astype(np.float32)


def _copy_src(h, w, copy, seed, block):
    """post.frag copySrc for every pixel: (sy, sx) index planes."""
    ys, xs = np.mgrid[0:h, 0:w]
    sy, sx = ys.copy(), xs.copy()
    if copy <= 0:
        return sy, sx
    bw, bh = -(-w // block), -(-h // block)
    bx = np.arange(bw)
    by = np.arange(bh)
    start = 0.16 * copy
    starts = _hash(bx[None, :], by[:, None], seed) < start          # (bh, bw)
    run_len = 2 + (_hash(bx[None, :], by[:, None], seed + 7.13) * 9.0).astype(np.int32)
    # per block: the run start it repeats, if any (searching up to 10 left)
    src_bx = bx[None, :].repeat(bh, 0).copy()
    done = starts.copy()                                             # starts never copy
    for k in range(1, 11):
        s = bx - k
        valid = s >= 0
        sc = np.clip(s, 0, None)
        hit = ~done & valid[None, :] & starts[:, sc]
        take = hit & (k <= run_len[:, sc])
        src_bx[take] = sc[None, :].repeat(bh, 0)[take]
        done |= hit | ~valid[None, :]
    echo = _hash(-3.0, by, seed + 1.7) < 0.1 * copy                  # per block row
    up = block * (1 + (_hash(-5.0, by, seed) * 3.0).astype(np.int32))
    bxi = xs // block
    byi = ys // block
    moved = src_bx[byi, bxi] != bxi
    sx = np.where(moved, src_bx[byi, bxi] * block + xs % block, xs)
    echo_px = echo[byi] & ~moved
    sy = np.where(echo_px, np.maximum(0, ys - up[byi]), ys)
    return sy, sx


_COPY_CACHE = {}
_COPY_CACHE_MAX = 4


def _copy_maps(h, w, copy, seed, block):
    """The COPY map as cv2.remap fixed-point maps, cached on everything it
    depends on. The seed changes on each cut (a few times a second), so a
    frame almost always reuses the last map: building it is about 20 ms at
    the working size, a NEAREST remap through it well under 1. The maps are
    whole pixels, so the remap is an exact lookup."""
    key = (h, w, float(copy), float(seed), int(block))
    maps = _COPY_CACHE.get(key)
    if maps is None:
        sy, sx = _copy_src(h, w, copy, seed, block)
        # one CV_16SC2 map of whole-pixel (x, y), no fraction table
        maps = (np.ascontiguousarray(np.dstack([np.clip(sx, 0, w - 1),
                                                np.clip(sy, 0, h - 1)]).astype(np.int16)),
                None)
        if len(_COPY_CACHE) >= _COPY_CACHE_MAX:
            _COPY_CACHE.pop(next(iter(_COPY_CACHE)))
        _COPY_CACHE[key] = maps
    return maps


def post(rgb, split=0.0, copy=0.0, seed=0.5, block=16):
    """COPY then SPLIT on RGB uint8 (h, w, 3); returns new pixels (or `rgb`
    itself when both are off). SPLIT takes red from `split` pixels right and
    blue from as far left of the copied picture, clamped at the frame edge.
    That is post.frag's rule exactly: it clamps the split's sample point
    into the frame, then looks it up through COPY."""
    s = int(np.floor(float(split) + 0.5))
    if s == 0 and copy <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if copy > 0:
        m1, m2 = _copy_maps(h, w, copy, seed, block)
        rgb = cv2.remap(rgb, m1, m2, cv2.INTER_NEAREST)
    if not s:
        return rgb
    xs = np.arange(w)
    out = rgb.copy()
    out[..., 0] = rgb[:, np.clip(xs + s, 0, w - 1), 0]
    out[..., 2] = rgb[:, np.clip(xs - s, 0, w - 1), 2]
    return out


def fold(stack, cur, alpha):
    """stack.frag's running mean: mix(stack, cur, alpha), float32, updated in
    place (cv2.accumulateWeighted) and returned. `stack` None starts a fresh
    exposure. `show` turns it back into display pixels."""
    if stack is None or alpha >= 1.0 or stack.shape != cur.shape:
        return cur.astype(np.float32)
    cv2.accumulateWeighted(cur, stack, float(alpha))
    return stack


def show(stack):
    """A float32 stack as uint8 pixels, rounded and saturated."""
    return cv2.convertScaleAbs(stack)
