"""Export Circuit Bender's tables and goldens the browser page needs as JSON.

`dtouch/shaders/bender/` is the third browser-shared unit (after
`../physarum/` and `../dither/`): cyborg-garden-site's /games/lighteater page
vendors the directory verbatim, and this module is the single source of
truth for its two JSON files. Every table is read from the live desktop
code, never retyped.

    python -m dtouch.bender_looks        # rewrite both JSON files

- `bender_looks.json` (bundled by the page): the mode's card metadata, the
  effects in E's order and their titles, the looks, the ladders, the settle
  and cut-clock constants, the working-size budget and its governor, the
  back-off, sound and autopilot sway, BENT CAM's mix, THERMAL's constants,
  STACK's weights, and
  the bend loop's rate limits.
- `bender_goldens.json` (imported only by the site's tests): the bends'
  outputs, as sha256 of the bytes or pixels, on the unit's three fixture
  JPEGs (one each from cv2, ffmpeg and Chrome) and on a rule-defined pixel
  fixture, plus the seeded generator's first values. The browser's jpeg.js,
  sensor.js and sort.js must reproduce every one of them, which is what holds
  the two ports byte-identical.

tests/test_bender_shared.py fails when either file on disk no longer matches
`dumps_looks()` / `dumps_goldens()`. Output is deterministic: a rerun is a
no-op.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import bender as B
from .bender_jpeg import (EFFECTS as JPEG_EFFECTS, STACK_WEIGHTS, bend,
                          mcu_offsets, rng32, scan_smear)
from .bender_post import sort_runs
from . import bender_sensor as S
from .bender_sensor import BENT_MIX, SENSOR_EFFECTS, sensor_bend
from .dither_looks import _dumps, _rgb
from .modes import bender as mode

UNIT_DIR = os.path.join(os.path.dirname(__file__), "shaders", "bender")
LOOKS_PATH = os.path.join(UNIT_DIR, "bender_looks.json")
GOLDENS_PATH = os.path.join(UNIT_DIR, "bender_goldens.json")

# The fixture JPEGs, shipped in the unit (no EXIF: JFIF, plus Chrome's ICC
# profile). Read, never regenerated, so an encoder upgrade cannot move them.
FIXTURES = ("fixture-cv2.jpg", "fixture-ffmpeg.jpg", "fixture-chrome.jpg")

GOLDEN_AMOUNTS = (0.25, 0.5, 1)
GOLDEN_SEEDS = (1, 99)
GOLDEN_PHASE = 0.37
SENSOR_SIZES = ((96, 64), (101, 37))     # an odd width and height too
SENSOR_AMOUNTS = (0.3, 0.65, 1)
SENSOR_SEEDS = (1, 7, 3735928559)     # 7 lands THERMAL on its third palette
SENSOR_TIMES = (0, 7.25)


def _num(v):
    """Whole floats as ints, so the JSON reads 1 rather than 1.0 (and the
    page's JS numbers print the same)."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _look(look):
    return {k: _num(v) for k, v in look.items()}


def looks_payload():
    cls = mode.BenderMode
    return {
        "title": cls.title,
        "key": cls.key,
        "blurb": cls.blurb,
        "accent": _rgb(cls.accent),
        "effects": list(B.EFFECTS),
        "effect_titles": {e: B.EFFECT_TITLES[e] for e in B.EFFECTS},
        "jpeg_effects": list(JPEG_EFFECTS),
        "sensor_effects": list(SENSOR_EFFECTS),
        "looks": {k: _look(v) for k, v in B.LOOKS.items()},
        "look_names": list(B.LOOK_NAMES),
        "safe_look": B.SAFE_LOOK,
        "ladders": {
            "amount": [_num(v) for v in B.AMOUNT_LADDER],
            "amount_nudge": B.AMOUNT_NUDGE,
            "long": list(B.LONG_LADDER),
            "sort": list(B.SORT_LADDER),
            "split": list(B.SPLIT_LADDER),
            "copy": [_num(v) for v in B.COPY_LADDER],
            "copy_names": {str(_num(k)): v for k, v in B.COPY_NAMES.items()},
        },
        "sort_band": list(B.SORT_BAND),
        "settle_s": {k: _num(v) for k, v in B.SETTLE_S.items()},
        "settle_stretch": B.SETTLE_STRETCH,
        "cut": {
            "every_s": B.CUT_EVERY_S,
            "onset_gap_s": B.ONSET_GAP_S,
            "drift_per_s": B.DRIFT_PER_S,
            "tempo": B.CUT_TEMPO,
            "touch_cuts": B.TOUCH_CUTS,
        },
        "working": {"max_pixels": B.MAX_PIXELS, "min_pixels": B.MIN_PIXELS},
        "governor": {"min_rate": B.MIN_RATE, "max_step_ms": B.MAX_STEP_MS,
                     "slow_s": B.SLOW_S, "step_down": B.STEP_DOWN},
        "backoff": {"clean_after": B.CLEAN_AFTER, "backoff": B.BACKOFF,
                    "recover": B.RECOVER, "floor": B.FLOOR},
        "sound": {"lift": B.SOUND_LIFT, "auto_sway": B.AUTO_SWAY,
                  "auto_sway_s": B.AUTO_SWAY_S},
        "jpeg": {"quality": B.JPEG_Q, "block": B.BLOCK},
        "bent_mix": {k: _num(v) for k, v in BENT_MIX.items()},
        "iconic": S.ICONIC,
        "thermal": {"rings": list(S.THERMAL_RINGS), "shadow": list(S.THERMAL_SHADOW),
                    "grain": S.THERMAL_GRAIN, "noise_hz": S.THERMAL_NOISE_HZ,
                    "edge": list(S.THERMAL_EDGE), "blur": S.THERMAL_BLUR,
                    "palettes": [[list(c) for c in p] for p in S.THERMAL_PALETTES]},
        "stack_weights": [[name, w] for name, w in STACK_WEIGHTS],
        "loop": {"max_bend_hz": mode.MAX_BEND_HZ, "stall_s": mode.STALL_S,
                 "respawn_gap_s": mode.RESPAWN_GAP_S,
                 "respawn_max_s": mode.RESPAWN_MAX_S,
                 "max_losses": mode.MAX_LOSSES},
    }


# ---------------------------------------------------------------------------
# goldens

def _sha(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def sensor_fixture(w, h):
    """The rule-defined pixel fixture (RGB uint8, h x w): channel values the
    page computes the same way with integer maths and & 255."""
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 5 + y * 3) & 255, (x * y + 40) & 255,
                     (255 - 2 * x + y * 7) & 255], -1).astype(np.uint8)


def _read_fixture(name):
    with open(os.path.join(UNIT_DIR, name), "rb") as fh:
        return fh.read()


def _jpeg_cases():
    cases = []
    for name in FIXTURES:
        data = _read_fixture(name)
        for effect in JPEG_EFFECTS:
            for a in GOLDEN_AMOUNTS:
                for seed in GOLDEN_SEEDS:
                    out = bend(data, effect, a, rng32(seed), GOLDEN_PHASE)
                    cases.append({"fixture": name, "effect": effect, "amount": _num(a),
                                  "seed": seed, "phase": GOLDEN_PHASE,
                                  "length": len(out), "sha256": _sha(out)})
        out = scan_smear(data, 0.7, rng32(5), 0.1)
        cases.append({"fixture": name, "effect": "smear", "amount": 0.7, "seed": 5,
                      "phase": 0.1, "length": len(out), "sha256": _sha(out)})
    return cases


def _walks():
    out = {}
    for name in FIXTURES:
        w = mcu_offsets(_read_fixture(name))
        out[name] = {"complete": w["complete"], "mcus": w["mcus"],
                     "offsets": len(w["offsets"]), "last": int(w["offsets"][-1])}
    return out


def _sensor_cases():
    cases = []
    for w, h in SENSOR_SIZES:
        for effect in SENSOR_EFFECTS:
            for a in SENSOR_AMOUNTS:
                for seed in SENSOR_SEEDS:
                    for t in SENSOR_TIMES:
                        px = sensor_bend(sensor_fixture(w, h), effect, a, seed, t)
                        cases.append({"w": w, "h": h, "effect": effect, "amount": _num(a),
                                      "seed": seed, "t": _num(t), "sha256": _sha(px.tobytes())})
    return cases


def _sort_cases():
    cases = []
    lo, hi = B.SORT_BAND
    for w, h in SENSOR_SIZES:
        for d in B.SORT_LADDER[1:]:
            px = sensor_fixture(w, h)
            runs = sort_runs(px, lo, hi, d)
            cases.append({"w": w, "h": h, "direction": d, "lo": lo, "hi": hi,
                          "runs": runs, "sha256": _sha(px.tobytes())})
    return cases


# settle_alpha's inputs: per bend, the stretch floor, a held frame, the
# camera fallback, a long exposure, an effect with no settle
SETTLE_CASES = (
    ("remap", 0, 1 / 60, 1), ("remap", 0, 1 / 16, 1), ("remap", 0, 0.5, 1),
    ("remap", 0, 0.05, 0), ("swap", 0, 0.01, 2), ("stack", 0, 0.03, 3),
    ("remap", 3, 0.1, 5), ("bent", 0, 0.1, 1),
)


def _settle_cases():
    cases = []
    for effect, long, dt, n in SETTLE_CASES:
        for camera in (False, True):
            cases.append({"effect": effect, "long": long, "dt": dt, "new_bends": n,
                          "max_hz": mode.MAX_BEND_HZ, "camera": camera,
                          "alpha": B.settle_alpha(effect, long, dt, n, mode.MAX_BEND_HZ, camera)})
    return cases


def goldens_payload():
    r = rng32(12345)
    return {
        "hash": "sha256 of the output bytes (JPEG) or of the RGB pixels, "
                "row-major, 3 bytes a pixel (sensor bends, sort)",
        "rng32": {"seed": 12345, "first": [r() for _ in range(4)]},
        "fixtures": list(FIXTURES),
        "sensor_fixture": "r = (5x + 3y) & 255, g = (x*y + 40) & 255, "
                          "b = (255 - 2x + 7y) & 255",
        "walks": _walks(),
        "jpeg": _jpeg_cases(),
        "sensor": _sensor_cases(),
        "sort": _sort_cases(),
        "settle": _settle_cases(),
    }


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
