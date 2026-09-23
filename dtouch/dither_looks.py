"""Export the dither tables and goldens the browser port needs as JSON.

`dtouch/shaders/dither/` is the second browser-shared unit (the first is
`../physarum/`): cyborg-garden-site's /games/lighteater page vendors
`dither.frag` verbatim next to the two files this module writes, and this
module is their single source of truth. Every table is read from the live
desktop code, never retyped.

    python -m dtouch.dither_looks        # rewrite both JSON files

- `dither_looks.json` (bundled by the page): the Dither mode's card metadata,
  algorithm lists, palettes and their resolved 1-bit pairs, built-in looks,
  defaults, constants, the two threshold matrices as exact integers, and the
  linearize / level-encode LUTs `dither.frag` samples. Also, under `modes`,
  the home-menu card of every mode the page shows, so the web menu vendors
  its cards instead of retyping them.
- `dither_goldens.json` (imported only by the site's tests and verifier):
  level indices from `dtouch.dither._ordered_indices` on three rule-defined
  64x64 fixtures, Floyd-Steinberg indices for the follow-up port, and
  `_palette_map` RGB rows.

tests/test_dither_shared.py fails when either file on disk no longer
matches `dumps_looks()` / `dumps_goldens()`, so a change to any table here
must come with a regenerate. Output is deterministic: a rerun is a no-op.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np

from .dither import (_MID_GREY_LINEAR, _bayer_matrix, _blue_noise_matrix,
                     _encode_levels, _ordered_indices, _srgb2lin_lut,
                     floyd_steinberg)
from .menu import MENU_SCRIM, registry_cards
from .modes.dithergirl import (ALGOS, AUTHORED_INVERSE, BIASES, HUE_HI, HUE_LO,
                               LEGACY_PALETTES, LEGIBILITY_FLOOR, ORDERED,
                               PALETTES, SCALE_DEFAULT, SCALE_HI, SCALE_LO,
                               SLOW_SCALE, TINT_LUMA_FLOOR, DitherGirlMode,
                               _BIAS_INVERT, _dither, low_depth_stops,
                               palette_stops)

UNIT_DIR = os.path.join(os.path.dirname(__file__), "shaders", "dither")
LOOKS_PATH = os.path.join(UNIT_DIR, "dither_looks.json")
GOLDENS_PATH = os.path.join(UNIT_DIR, "dither_goldens.json")

# The home-menu cards the page shows, in the page's card order (the design's
# two-card menu: no Particles, no AUTO, no reserved Flocking card). The card
# fields themselves come from dtouch/menu.py registry_cards().
WEB_MODES = ("physarum", "dithergirl")

# What the browser may claim: the ordered algorithms `_dither` renders.
# ASCII is ordered but never reaches `_dither` (dtouch/modes/dithergirl.py:361
# raises; dtouch.ascii_art owns it, its ramp measured against cv2's Hershey
# raster), so it is not portable as a table.
WEB_ALGOS = tuple(a for a in ALGOS if a in ORDERED and a != "ASCII")

# The matrix each ordered algorithm tiles: bayer_dither's default 4x4
# (dtouch/dither.py:213, 243) and the 64x64 blue-noise asset
# (dtouch/dither.py:286).
_MATRIX_OF = {"Bayer": "bayer4", "Blue noise": "bluenoise64"}

# The levels a Bits 1..4 control produces: (1 << bits) - 1
# (dtouch/dither.py:181).
LEVELS = tuple((1 << b) - 1 for b in (1, 2, 3, 4))

GOLDEN_SIZE = 64


# ---------------------------------------------------------------------------
# dither_looks.json
# ---------------------------------------------------------------------------

def _rgb(bgr):
    return [int(v) for v in bgr[::-1]]


def _stops(stops):
    return [[int(v) for v in c] for c in stops]


def _cards():
    by_id = {c.id: c for c in registry_cards()}
    out = {}
    for mode_id in WEB_MODES:
        c = by_id[mode_id]
        out[mode_id] = {"id": c.id, "title": c.title, "key": c.key,
                        "blurb": c.blurb, "accent": _rgb(c.accent)}
    return out


def _matrices():
    bayer = _bayer_matrix(4).astype(np.float64) * 16.0
    assert np.array_equal(bayer, np.round(bayer)), "bayer4 * 16 is not integer"
    bayer = bayer.astype(np.int64)
    assert sorted(bayer.ravel().tolist()) == list(range(16))
    blue = _blue_noise_matrix().astype(np.float64) * 4096.0
    assert blue.shape == (64, 64)
    assert np.array_equal(blue, np.round(blue)), "bluenoise64 * 4096 is not integer"
    blue = blue.astype(np.int64)
    assert sorted(blue.ravel().tolist()) == list(range(4096)), \
        "bluenoise64 ranks are not a permutation of 0..4095"
    return {"bayer4": bayer.tolist(), "bluenoise64": blue.tolist()}


def _enc_lut(levels, gamma):
    """The whole 256-entry level-encode table: indexing it with every code
    returns the table itself."""
    return _encode_levels(np.arange(256, dtype=np.uint8), levels, gamma)


def _luts():
    lin = {str(lv): [float(v) for v in _srgb2lin_lut(float(lv))] for lv in LEVELS}
    enc = {}
    for lv in LEVELS:
        for gamma in (True, False):
            enc[f"{lv},{str(gamma).lower()}"] = [float(v) for v in _enc_lut(lv, gamma)]
    return {"lin": lin, "enc": enc}


def looks_payload():
    """The JSON-able dict: plain lists, ints, floats and strings, stable key
    order."""
    m = DitherGirlMode
    card = _cards()["dithergirl"]
    names = list(PALETTES)
    web_builtin = [n for n, look in m.BUILTIN.items()
                   if look["algorithm"] in WEB_ALGOS]
    return {
        "title": card["title"],
        "key": card["key"],
        "blurb": card["blurb"],
        "accent": card["accent"],
        "algos": list(ALGOS),
        "ordered": list(ORDERED),
        "web_algos": list(WEB_ALGOS),
        "biases": list(BIASES),
        "bias_invert": dict(_BIAS_INVERT),
        "palette_names": names,
        "palettes": {n: {"stops": _stops(PALETTES[n])} for n in names},
        "authored_inverse": {n: _stops(s) for n, s in AUTHORED_INVERSE.items()},
        "legacy_palettes": {n: dict(v) for n, v in LEGACY_PALETTES.items()},
        # what a 1-bit dither renders: the stops step() hands _palette_map at
        # two levels (dtouch/modes/dithergirl.py:930, low_depth_stops)
        "one_bit_pairs": {n: {"normal": _stops(low_depth_stops(palette_stops(n, False), 2)),
                              "inverted": _stops(low_depth_stops(palette_stops(n, True), 2))}
                          for n in names},
        "builtin": {n: json.loads(json.dumps(look)) for n, look in m.BUILTIN.items()},
        "web_builtin": web_builtin,
        "defaults": dict(m.DEFAULTS),
        "constants": {
            "scale_lo": SCALE_LO, "scale_hi": SCALE_HI,
            "scale_default": SCALE_DEFAULT, "slow_scale": SLOW_SCALE,
            "legibility_floor": LEGIBILITY_FLOOR,
            "mid_grey_linear": _MID_GREY_LINEAR,
            "tint_luma_floor": TINT_LUMA_FLOOR,
            "hue_lo": HUE_LO, "hue_hi": HUE_HI,
            "menu_scrim": MENU_SCRIM,
        },
        "matrices": _matrices(),
        "luts": _luts(),
        "modes": _cards(),
    }


# ---------------------------------------------------------------------------
# dither_goldens.json
# ---------------------------------------------------------------------------

FIXTURE_RULES = {
    "gradient": "(4 * x + (y & 3)) & 255",
    "dark": "gradient >> 2",
    "light": "192 + (gradient >> 2)",
}


def fixture(name, size=GOLDEN_SIZE):
    """uint8 codes (size, size), row 0 = y 0, for one of FIXTURE_RULES."""
    y, x = np.mgrid[0:size, 0:size]
    grad = ((4 * x + (y & 3)) & 255).astype(np.uint8)
    if name == "gradient":
        return grad
    if name == "dark":
        return (grad >> 2).astype(np.uint8)
    if name == "light":
        return (192 + (grad >> 2)).astype(np.uint8)
    raise KeyError(name)


def _plane(codes):
    """The float plane the desktop dithers: dithergirl.step()'s
    `u8.astype(float32) / 255.0` (dtouch/modes/dithergirl.py:924-925)."""
    return codes.astype(np.float32) / 255.0


def _b64(idx):
    return base64.b64encode(np.ascontiguousarray(idx, dtype=np.uint8).tobytes()).decode("ascii")


def _slug(algo):
    return algo.lower().replace(" ", "-")


def _matrix(algo):
    return _bayer_matrix(4) if _MATRIX_OF[algo] == "bayer4" else _blue_noise_matrix()


def _core_case(fx, algo, bits, gamma, bias):
    codes = fixture(fx)
    img = _plane(codes)
    idx, inv = _ordered_indices(img, _matrix(algo), bits, _BIAS_INVERT[bias], gamma)
    levels = (1 << bits) - 1
    # the indices, encoded, ARE what the mode renders through _dither
    assert np.array_equal(_encode_levels(idx, levels, gamma),
                          _dither(img, algo, bits, gamma, bias))
    assert int(idx.max()) <= levels
    return {"name": f"{_slug(algo)}-{bits}bit-gamma-{'on' if gamma else 'off'}-{bias}-{fx}",
            "fixture": fx, "algo": algo, "bits": bits, "gamma": gamma,
            "bias": bias, "invert": inv, "idx": _b64(idx)}


def _core():
    cases = []
    # every bit depth, both gamma paths, both forced biases, on the gradient
    for algo in WEB_ALGOS:
        for bits in (1, 2, 3, 4):
            for gamma in (True, False):
                for bias in ("light", "dark"):
                    cases.append(_core_case("gradient", algo, bits, gamma, bias))
    # bias auto on the clearly dark and clearly light fixtures, gamma on (the
    # web page's only gamma). Not on the gradient: there, auto with gamma on
    # always resolves to light (linear mean 0.311 vs mid_grey_linear 0.214),
    # so it would only duplicate the forced-light cases; with gamma off its
    # mean is exactly 0.5, right on that threshold.
    for algo in WEB_ALGOS:
        for bits in (1, 2, 3, 4):
            for fx in ("dark", "light"):
                cases.append(_core_case(fx, algo, bits, True, "auto"))
    return cases


def _fs():
    cases = []
    img = _plane(fixture("gradient"))
    for bits in (1, 2, 4):
        levels = (1 << bits) - 1
        for gamma in (True, False):
            out = floyd_steinberg(img, bits=bits, gamma=gamma)
            enc = _enc_lut(levels, gamma)[:levels + 1]
            hit = out[..., None] == enc[None, None, :]
            assert np.all(hit.sum(axis=-1) == 1), "fs output not on the enc table"
            idx = hit.argmax(axis=-1).astype(np.uint8)
            cases.append({"name": f"fs-{bits}bit-gamma-{'on' if gamma else 'off'}-gradient",
                          "fixture": "gradient", "bits": bits, "gamma": gamma,
                          "idx": _b64(idx)})
    return cases


RGB_PALETTES = ("mono", "aurora", "cga")


def _rgb_cases():
    cases = []
    for name in RGB_PALETTES:
        for bits in (1, 2):
            levels = (1 << bits) - 1
            enc = _enc_lut(levels, True)[:levels + 1]
            for invert in (False, True):
                stops = low_depth_stops(palette_stops(name, invert), 1 << bits)
                rgb = DitherGirlMode._palette_map(enc[None, :], stops)[0]
                cases.append({"name": f"{name}-{bits}bit-{'inverted' if invert else 'normal'}",
                              "palette": name, "bits": bits, "invert": invert,
                              "enc": [float(v) for v in enc],
                              "stops": _stops(stops),
                              "rgb": rgb.astype(int).tolist()})
    return cases


def goldens_payload():
    grad = fixture("gradient")
    assert len(np.unique(grad)) == 256, "gradient fixture must hold every code"
    return {
        "size": [GOLDEN_SIZE, GOLDEN_SIZE],
        "layout": "uint8 per pixel, row-major, byte y * 64 + x, row 0 = y 0 (image space)",
        "fixtures": dict(FIXTURE_RULES),
        "plane": "code / 255 in float32 (dithergirl.step's u8 / 255.0); grayscale",
        "matrix_of": dict(_MATRIX_OF),
        "core": _core(),
        "fs": _fs(),
        "rgb": _rgb_cases(),
    }


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def _scalar(v):
    return not isinstance(v, (dict, list))


def _emit(v, depth):
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [f"{inner}{json.dumps(k)}: {_emit(x, depth + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list):
        if all(_scalar(x) for x in v):          # a row, a LUT, a stop: one line
            return json.dumps(v, separators=(",", ":"))
        items = [f"{inner}{_emit(x, depth + 1)}" for x in v]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(v)


def _dumps(d):
    """Indented one space per level for review, but every list of scalars
    (a matrix row, a LUT, a stop) sits on one line."""
    return _emit(d, 0) + "\n"


def dumps_looks():
    return _dumps(looks_payload())


def dumps_goldens():
    return _dumps(goldens_payload())


def write():
    paths = []
    for path, text in ((LOOKS_PATH, dumps_looks()), (GOLDENS_PATH, dumps_goldens())):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write():
        print("wrote", p)
