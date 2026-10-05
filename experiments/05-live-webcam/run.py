#!/usr/bin/env python3
"""Experiment 05 — real-time live preview (interactive).

Default mode 'flow': camera -> subject-agnostic matte (whatever moves/stands out) -> a smoothly
flowing cloud of glowing particles. Works for a dancer, a crowd, a boat — not person-only.
Mode 'grid' is the older luminance-displaced grid.

    python run.py                       # the HOME MENU, last-used mode selected
    python run.py --mode flow           # skip the menu: straight into flow
    python run.py --matte motion        # key on motion only (great for a dancer)
    python run.py --matte person        # multi-person segmentation
    python run.py --device 1            # a specific camera index
    python run.py --mode grid           # the old displacement-grid effect

    python run.py --flock               # start with the particle cloud flocking
    python run.py --glitch              # start with the circuit-bent signal chain on

    python run.py --mode dithergirl     # boot into Dither (live dithering)
    python run.py --mode physarum       # boot into Physarum (slime-mold veins eat the light)
    python run.py --mode bender         # boot into Circuit Bender (live JPEG databending)
    python run.py --still photo.jpg     # load a still and imply dithergirl

Launching with no mode-implying flag opens the home menu with the live camera
behind it and the last-used mode already selected — Enter (or Esc) starts it.
Any mode-implying flag skips the menu entirely.

Controls: press `?` for the live key map — it is generated from the command
registry, so it is the only listing that cannot go stale.

    TAB overlay (hidden / HUD / panel)   Esc step back toward hidden
    q q quit (twice)                     m menu        ? keys
    space BLACKOUT (hard black, and it IS recorded)    0 panic reset
    r record        a sound react        g glitch      i debug readout
    1-9 recall a bank slot               [ ] walk the setlist
    , . select a control                 - = nudge it  (_ + nudge x5)
    s save the current look (panel only)

The control panel groups everything into collapsible sections: TEMPLATES, SOURCE, LOOK,
MOTION (flocking) and SIGNAL (glitch + dithering). Click a header to open it.
"""
from __future__ import annotations

import argparse
import multiprocessing

from dtouch import presets as _presets
from dtouch.camera import list_cameras
from dtouch.live import live
from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.bender import BenderMode
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.modes.particles import ParticlesMode
from dtouch.modes.physarum import PhysarumMode
from dtouch.shell import Host


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


# Engine flags that only Particles/flow can honour. A value different from the
# default is an explicit request, so it means "boot into flow" (DESIGN.md §3:
# CLI flags mean "boot into") — otherwise the flag would be silently dropped by
# the state.json resume path, which constructs the mode with DEFAULT args.
ENGINE_DEFAULTS = dict(matte="auto", grid="416x234", particles=200000,
                       flock=False, glitch=False)


def engine_flags_given(args):
    """The particles-implying flags the user actually set, as CLI spellings."""
    return ["--" + name for name, default in ENGINE_DEFAULTS.items()
            if getattr(args, name) != default]


def boot_mode_name(mode_arg, still_arg, particles_flags=False):
    """Boot precedence: --mode > --still > engine flags > state.json.

    --still PATH implies dithergirl unless --mode is given (DESIGN.md §3: CLI
    flags mean 'boot into'). Any particles-implying engine flag set to a
    non-default value (--matte/--grid/--particles/--flock/--glitch) boots flow
    WITH those args. With no mode-implying flag at all, returns None — the
    shell opens the HOME MENU with the last-used mode from state.json running
    behind it and selected (§3, amended: launch opens on the menu; first run:
    Particles is the selection)."""
    if mode_arg:
        return mode_arg
    if still_arg:
        return "dithergirl"
    return "flow" if particles_flags else None


# run.py's --mode spellings vs Mode.id — the launcher predates the registry,
# so 'flow' is the CLI name for the particles mode ('grid' is the legacy
# non-shell path and owns no looks).
MODE_ARG = {"particles": "flow", "dithergirl": "dithergirl", "physarum": "physarum",
            "bender": "bender"}
MODE_ID = {v: k for k, v in MODE_ARG.items()}


def preset_owners(name, path="presets.json"):
    """Mode ids whose look set contains `name`, in REGISTRY order — built-ins
    plus that mode's saved looks, exactly what the shell would resolve."""
    return [cls.id for cls in REGISTRY
            if name in _presets.load(path, mode=cls.id,
                                     builtin=getattr(cls, "BUILTIN", {}))]


def resolve_preset(name, boot, path="presets.json"):
    """Resolve `--preset NAME` against a real mode's looks. Returns
    (boot_mode_arg, preset) and prints the note when it had to choose.

    `--preset` is NOT an entry in ENGINE_DEFAULTS on purpose. That rule is
    "any non-default value boots flow", and a preset name can legitimately
    belong to either mode — it would send every Dither look to Particles.
    The name is resolved against the modes' actual look sets instead:

    - No higher-precedence mode flag: the mode that OWNS the look boots. The
      name IS the request. (Previously the flag was dropped entirely once
      state.json remembered another mode — `run.py --mode dithergirl`, quit,
      `run.py --preset embers` booted Dither on `classic` and said
      nothing. On main `--preset` always worked.)
    - One already chosen (--mode / --still / an engine flag): that wins, per
      the documented precedence. A look it does not own is called out and we
      land on that mode's safe_look, rather than silently starting on
      whatever index 0 happens to be.
    - A name no mode owns is called out, and the normal resume happens.
    """
    if not name:
        return boot, name
    owners = preset_owners(name, path)
    if boot is None:
        if not owners:
            print('note: no preset named "%s" - booting the last-used look'
                  % name)
            return None, None
        if len(owners) > 1:
            print('note: "%s" is a look in %s - booting %s'
                  % (name, " and ".join(mode_by_id(m).title for m in owners),
                     mode_by_id(owners[0]).title))
        return MODE_ARG[owners[0]], name
    cls = mode_by_id(MODE_ID.get(boot))
    if cls is None:                       # 'grid': the legacy path has no looks
        print('note: --preset %s ignored - %s has no looks' % (name, boot))
        return boot, None
    if cls.id in owners:
        return boot, name
    safe = cls().safe_look()
    print('note: preset "%s" is not a %s look - booting %s'
          % (name, cls.title, safe))
    return boot, safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None,
                    choices=["flow", "grid", "dithergirl", "physarum", "bender"])
    ap.add_argument("--still", default=None, metavar="PATH",
                    help="load a still image (implies --mode dithergirl)")
    ap.add_argument("--ui", default="panel", choices=["panel", "keys"],
                    help="panel = render + slider window (default); keys = render only")
    ap.add_argument("--audio", action="store_true", help="start with mic reactivity on")
    ap.add_argument("--device", default="builtin",
                    help="'builtin' (laptop cam), an index, or a name substring")
    ap.add_argument("--matte", default="auto",
                    choices=["auto", "motion", "saliency", "person", "edges", "luma"])
    ap.add_argument("--preset", default=None,
                    help="named look: abstract | portrait | textured | embers | aurora | <your saved>")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--grid", default="416x234")
    ap.add_argument("--particles", type=int, default=200000,
                    help="max particles allocated; the Count slider scales how many render")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--flock", action="store_true",
                    help="start with boids steering on the particle cloud (MOTION panel)")
    ap.add_argument("--glitch", action="store_true",
                    help="start with the circuit-bent post-FX chain on (SIGNAL panel)")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("  (AVFoundation enumeration unavailable)")
        for i, name, dtype in cams:
            tag = dtype.replace("AVCaptureDeviceType", "")
            print(f"  [{i}] {name}  ({tag})")
        return

    device = int(args.device) if args.device.isdigit() else args.device
    given = engine_flags_given(args)
    boot = boot_mode_name(args.mode, args.still, particles_flags=bool(given))
    if given and boot != "flow":
        # a higher-precedence flag won; say so instead of dropping the flag —
        # but only for flags the boot mode really drops (physarum eats --matte)
        used = {"physarum": {"--matte"}}.get(boot, set())
        ignored = [f for f in given if f not in used]
        if ignored:
            print("note: %s ignored - booting %s" % (", ".join(ignored), boot))
    # --preset resolves against a mode's real looks; with no mode-implying
    # flag, the mode that owns the named look is the one that boots
    boot, preset = resolve_preset(args.preset, boot)
    if boot == "grid":
        live(device=device, res=parse_wh(args.res), mirror=not args.no_mirror)
        return
    # thin launcher: shell + mode (DESIGN.md §8 step 6)
    if boot is None:
        mode = None      # shell opens the home menu over the last-used mode (§3)
    elif boot == "dithergirl":
        mode = DitherGirlMode(still=bool(args.still))
    elif boot == "physarum":
        mode = PhysarumMode(matte=args.matte)
    elif boot == "bender":
        mode = BenderMode()
    else:
        mode = ParticlesMode(matte=args.matte, grid=parse_wh(args.grid),
                             n=args.particles, flock=args.flock, glitch=args.glitch)
    host = Host(mode, device=device, res=parse_wh(args.res),
                preset=preset, audio=args.audio, still=args.still,
                panel=(args.ui == "panel"), mirror=not args.no_mirror)
    host.run()


if __name__ == "__main__":
    # Circuit Bender bends in a spawned worker process; a frozen build must
    # not re-run the launcher in it
    multiprocessing.freeze_support()
    main()
