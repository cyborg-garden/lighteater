"""Export the physarum tables the browser port needs as JSON.

`dtouch/shaders/physarum/looks.json` is vendored verbatim by
cyborg-garden-site's /physarum page next to the shaders; this module is its
single source of truth. Payload: the behavior POINTS the shaders' uniforms
are fed from, the mode's DEFAULTS and BUILTIN looks, every palette as
both its gradient stops and the resolved 256-entry RGB LUT, and the relief
light's orbit + bass rake (`depth`).

    python -m dtouch.physarum_looks        # rewrite looks.json

tests/test_physarum_gl.py fails when the file on disk no longer matches
`payload()`, so a change to POINTS / BUILTIN / DEFAULTS / palette stops / the
DEPTH_* constants must come with a regenerate.
"""
from __future__ import annotations

import json
import os

from .modes.physarum import (DEPTH_AZ0, DEPTH_ELEV_HI, DEPTH_ELEV_LO,
                             DEPTH_FEEL_CURVE, DEPTH_ORBIT_S, DEPTH_RAKE,
                             FEEL_CURVE,
                             _PALETTE_STOPS, PALETTES_PH, PhysarumMode,
                             _palette_lut)
from .physarum import POINT_NAMES, POINTS

LOOKS_PATH = os.path.join(os.path.dirname(__file__), "shaders", "physarum", "looks.json")


def payload():
    """The JSON-able dict — plain lists and floats only, stable key order."""
    palettes = {}
    for name in PALETTES_PH:
        if name == "video":
            continue                      # lit by the footage; no LUT
        palettes[name] = {
            "stops": [list(map(int, s)) for s in _PALETTE_STOPS[name]],
            "lut": _palette_lut(name).tolist(),
        }
    return {
        "points": {name: {k: float(v) for k, v in POINTS[name].items()}
                   for name in POINT_NAMES},
        "point_names": list(POINT_NAMES),
        "defaults": dict(PhysarumMode.DEFAULTS),
        "builtin": {name: dict(look) for name, look in PhysarumMode.BUILTIN.items()},
        "palette_names": list(PALETTES_PH),
        "palettes": palettes,
        # the relief light's motion, so the browser vendors the numbers
        # instead of retyping them (each look carries its own `depth` in
        # `builtin` / `defaults`). az0 is radians in SCREEN space (+y up):
        # a host whose row 0 is the screen top passes -y into u_light.
        # light = (cos az * ce, sin az * ce, elev), ce = sqrt(1 - elev^2),
        # az = az0 + 2 pi t / orbit_s, elev = clamp(elev_hi - rake * sens *
        # bass, elev_lo, elev_hi); feel_curve_applies says whether the
        # slider value is raised to the MOLD FEEL_CURVE before it is u_depth.
        "depth": {
            "orbit_s": float(DEPTH_ORBIT_S),
            "elev_hi": float(DEPTH_ELEV_HI),
            "elev_lo": float(DEPTH_ELEV_LO),
            "rake": float(DEPTH_RAKE),
            "az0": float(DEPTH_AZ0),
            "feel_curve_applies": bool(DEPTH_FEEL_CURVE),
            "feel_curve": float(FEEL_CURVE),
        },
    }


def dumps():
    """Indented for review, except each 256x3 LUT sits on one line — with
    indent=1 alone the seven LUTs are 75 KB of one-number lines."""
    d = payload()
    luts = {}
    for name, pal in d["palettes"].items():
        luts[name] = pal["lut"]
        pal["lut"] = f"@@LUT:{name}@@"
    text = json.dumps(d, indent=1)
    for name, lut in luts.items():
        text = text.replace(f'"@@LUT:{name}@@"', json.dumps(lut, separators=(",", ":")))
    return text + "\n"


def write(path=LOOKS_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps())
    return path


if __name__ == "__main__":
    print("wrote", write())
