"""Autopilot — the instrument playing itself.

The app is a performance instrument, and the most common thing anyone wants
from one is to put it on and watch. Autopilot is that: it re-casts the scene
on a slow, randomized cadence — a new look, sometimes a new palette or
behavior pair, sometimes a theatrical burst or wave — and, when it is allowed
to, walks between the three modes so the piece has movements.

It is a shell concern, not a Mode. It does not know what a physarum is; it
drives the same named commands and the same mailboxes a person's fingers
would, so anything a person can do it can do, and anything it does a person
can interrupt. That last part is the contract: ANY scene-changing input from
a human turns it off, immediately and without a fight (`interrupt`). It is a
guest, not a driver.

The shell owns the wiring; this owns the decisions, with the clock injected
so the whole thing is testable without a window.
"""
from __future__ import annotations

import numpy as np

# Cadence, seconds. Long enough to watch a scene become itself, short enough
# that leaving the room and coming back shows you something new. Randomized
# so it never feels like a slideshow on a timer.
DWELL = (25.0, 55.0)
# A mode change is a bigger event than a look change, so it is rarer.
MODE_EVERY = (4, 8)          # re-casts between mode hops, inclusive range
# How often a re-cast also throws a cast (burst or wave). Not every time —
# a gesture that happens every time stops reading as a gesture.
CAST_CHANCE = 0.45
# The very first re-cast comes fast: a control that appears to do nothing for
# the first 40 seconds is a control the user believes is broken.
FIRST_DWELL = 3.0
# The most a single tick may advance the clock, seconds. A stalled frame (a
# mode rebuild, a window drag, a laptop lid) must not fire a re-cast the
# instant it resumes (see `tick`).
DT_MAX = 0.25
# The theatrical moves a re-cast may throw (see `_cast`). Named here, not
# inline, so dtouch/dither_looks.py can export them for the browser port.
CASTS = ("physarum.burst", "physarum.wave", "physarum.random",
         "physarum.swap")


class Autopilot:
    """Decides what the instrument should do next, and when.

    `tick` returns a list of (kind, value) intents for the shell to apply:
      ("preset", name)    recall a look in the current mode
      ("mode", mode_id)   switch modes
      ("command", name)   dispatch a named command (casts, swaps)
    The shell applies them through the ordinary mailboxes, so autopilot has
    exactly the reach a person has and no more.
    """

    def __init__(self, seed=None, cross_mode=True):
        self.on = False
        self.cross_mode = cross_mode
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._next = 0.0
        self._until_mode_hop = 0
        self.last_reason = ""

    # ----- control -----
    def toggle(self):
        self.set(not self.on)
        return self.on

    def set(self, on):
        on = bool(on)
        if on == self.on:
            return
        self.on = on
        if on:
            self._t = 0.0
            self._next = FIRST_DWELL
            self._until_mode_hop = int(self._rng.integers(*MODE_EVERY))
            self.last_reason = "engaged"
        else:
            self.last_reason = "released"

    def interrupt(self):
        """A human touched a scene control. Autopilot yields at once.

        Deliberately silent about which control: the rule is simple enough to
        learn by accident, which is the only way anyone learns it.
        """
        if self.on:
            self.set(False)
            self.last_reason = "you took over"
            return True
        return False

    # ----- the loop -----
    def _dwell(self):
        return float(self._rng.uniform(*DWELL))

    def tick(self, dt, mode_id, looks, mode_ids=(), current=None):
        """Advance the clock and return this frame's intents (usually none).

        `looks` are the current mode's recallable look names, `current` the
        one on screen (never re-picked), and `mode_ids` the modes it may hop
        between. dt is clamped: a stalled frame (a mode
        rebuild, a window drag, a laptop lid) must not fire a re-cast the
        instant it resumes, which is the wall-clock bug the browser build had.
        """
        if not self.on:
            return []
        self._t += min(max(float(dt), 0.0), DT_MAX)
        if self._t < self._next:
            return []
        self._t = 0.0
        self._next = self._dwell()

        out = []
        hop = (self.cross_mode and len(mode_ids) > 1
               and self._until_mode_hop <= 0)
        if hop:
            others = [m for m in mode_ids if m != mode_id]
            target = others[int(self._rng.integers(len(others)))]
            self._until_mode_hop = int(self._rng.integers(*MODE_EVERY))
            self.last_reason = f"-> {target}"
            out.append(("mode", target))
            return out

        self._until_mode_hop -= 1
        # Never re-cast to the look already on screen. Uniform picking put one
        # re-cast in five back where it started (measured 20.4% over 40 seeds),
        # so every fifth dwell the picture simply did not change for 50-110 s —
        # the exact "this control does nothing" reading FIRST_DWELL exists to
        # prevent. Falls through when there is only one look to pick.
        pool = [l for l in looks if l and l != current] or [l for l in looks if l]
        if pool:
            pick = pool[int(self._rng.integers(len(pool)))]
            self.last_reason = pick
            out.append(("preset", pick))
        if pool and self._rng.random() < CAST_CHANCE:
            out.append(("command", self._cast()))
        return out

    def _cast(self):
        """A theatrical move. Only ever mode-local commands that are safe to
        dispatch when the mode does not have them — the shell drops unknown
        names rather than raising, so this stays a hint, not a demand."""
        return CASTS[int(self._rng.integers(len(CASTS)))]
