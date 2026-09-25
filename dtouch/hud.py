"""HUD layer — u-unit geometry, outlined text, toasts, OSD, status line, corner tick.

The perform surface (DESIGN.md §5–6): everything here is authored in the u-unit
(`u = frame_height / 45` — ~16 px @720p, 24 @1080p, 48 @4K), double-drawn for
worst-case contrast (black under near-white — legible against a white wall), and
transient except the status line and the blackout corner tick.

Overlay states (DESIGN.md §6.1): HIDDEN (provably clean output — the OBS capture
contract; nothing draws once toasts expire) → HUD (status line + momentary
toasts/OSD) → PANEL (full sidebar; HUD still active). TAB cycles forward, Esc
always steps toward HIDDEN and never quits.

Everything draws onto small ROIs (text boxes, a corner triangle) — no full-frame
ops — to hold the ≤1 ms/frame overlay budget. ASCII text only (Hershey fonts
render non-ASCII as '?').
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# BGR — matched to the panel's chrome (overlay_ui) plus the one amber addition.
INK = (215, 222, 218)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
RED = (70, 70, 235)
AMBER = (0, 191, 255)          # armed / warning / transient (DESIGN.md §5)
OUTLINE = (0, 0, 0)
PLATE = (34, 32, 30)           # == imgui.PANEL — the panel's own chrome ground
HELP_PLATE_ALPHA = 0.86        # ...and the panel's own opacity over it

FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAP_PX = 22.0                 # Hershey simplex cap height at fontScale 1.0

TITLE_SAFE = 0.035             # 3.5% title-safe inset (DESIGN.md §5)

# Camera notes (DESIGN.md §6.4). Two different faults, two different fixes:
# frames ARRIVING but black is the Continuity Camera symptom the original
# message names, and its wording is kept verbatim. Frames not arriving at all
# is an unplugged camera or one another app has taken — telling that operator
# to turn Continuity Camera off sends them somewhere there is nothing to find.
CAMERA_NOTE = "CAMERA IS BLACK - disable iPhone Continuity Camera"
CAMERA_FIX = "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off"
CAMERA_LOST_NOTE = "CAMERA STOPPED SENDING FRAMES - holding the last one"
CAMERA_LOST_FIX = "check it is still connected, and not in use by another app"

CENTER_TOAST_S = 1.2           # mode/center flash fade (DESIGN.md §5 type table)
HINT_TOAST_S = 1.5
OSD_FADE_S = 1.5


def u(frame_h):
    """The u-unit: frame-relative geometry base (DESIGN.md §5)."""
    return frame_h / 45.0


class OverlayState(Enum):
    HIDDEN = 0
    HUD = 1
    PANEL = 2


def cycle_overlay(state):
    """TAB: HIDDEN -> HUD -> PANEL -> HIDDEN."""
    return {OverlayState.HIDDEN: OverlayState.HUD,
            OverlayState.HUD: OverlayState.PANEL,
            OverlayState.PANEL: OverlayState.HIDDEN}[state]


def esc_overlay(state):
    """Esc: one step toward HIDDEN; in HIDDEN it does nothing. Esc never quits."""
    return {OverlayState.PANEL: OverlayState.HUD,
            OverlayState.HUD: OverlayState.HIDDEN,
            OverlayState.HIDDEN: OverlayState.HIDDEN}[state]


def _font(px):
    scale = px / _CAP_PX
    thick = max(1, int(round(scale)))
    return scale, thick


def put_outlined(img, text, org, px, color=INK, thick_mul=1.0):
    """Double-drawn Hershey text: black under the ink, so it survives any background."""
    scale, thick = _font(px)
    thick = max(1, int(round(thick * thick_mul)))
    cv2.putText(img, text, org, FONT, scale, OUTLINE, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_size(text, px):
    scale, thick = _font(px)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick + 2)
    return tw, th, base


def fit_px(text, px, max_w, min_px=6):
    """The largest type size at or below `px` whose rendered width fits
    `max_w` (DESIGN.md §5: nothing crosses the title-safe inset).

    The u-unit scales type with the FRAME, not with the string, so a long
    line at a fixed size overflows every resolution equally: the containment
    flash is 1435 px wide in a 1280 px frame at 720p, 2152 in 1920, 4303 in
    3840 — clipped at both ends, rendering as 'mething went wrong - show
    continu'. That is the one message the whole containment mechanism exists
    to show, so it shrinks to fit rather than being cropped to nonsense.

    Hershey advance is not linear in the size (integer scale and thickness),
    so the proportional guess is corrected downward until it really fits."""
    if px <= min_px or max_w <= 0:
        return max(px, min_px)
    tw, _, _ = text_size(text, px)
    if tw <= max_w or tw <= 0:
        return px
    px = max(min_px, int(px * max_w / tw))
    while px > min_px and text_size(text, px)[0] > max_w:
        px -= 1
    return px


def blend_outlined(img, text, org, px, color, alpha):
    """Outlined text at partial opacity, blended over a small ROI only (budget)."""
    if alpha <= 0.0:
        return
    if alpha >= 0.999:
        put_outlined(img, text, org, px, color)
        return
    h, w = img.shape[:2]
    tw, th, base = text_size(text, px)
    m = 4  # outline margin
    x0 = max(org[0] - m, 0); y0 = max(org[1] - th - m, 0)
    x1 = min(org[0] + tw + m, w); y1 = min(org[1] + base + m, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    put_outlined(over, text, (org[0] - x0, org[1] - y0), px, color)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_corner_tick(img, size_px, color=AMBER):
    """Filled corner triangle, top-right, legs `size_px` — the one persistent
    element allowed on a blacked-out frame (DESIGN.md §5: blackout armed)."""
    h, w = img.shape[:2]
    s = int(round(size_px))
    pts = np.array([[w - s, 0], [w, 0], [w, s]], np.int32)
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)


HELP_COLS_MAX = 3      # past three columns the map stops being scannable
HELP_MIN_PX = 8        # ...and below this the type stops being readable


def help_key_text(key):
    """What a key is CALLED on the map (space has no glyph)."""
    return {" ": "space"}.get(key, key)


def help_layout(w, h, rows):
    """Where every row of the key map goes, for a `w`x`h` frame.

    Split out and made fit-aware because the map ran off the bottom of the
    frame: one fixed column of 1.5-px leading, so a 33-row registry put the
    last two rows — `TAB Cycle overlay` and `Esc Step toward hidden` — below
    the frame at 720p (baseline 754) and 1080p (1114), and three rows off at
    4K. The two keys that walk you back OUT of an overlay state were the two
    the map could not show, and it had neither a fit nor a scroll.

    Columns first (the map stays full size and scannable), then, only if even
    HELP_COLS_MAX columns will not fit, the type shrinks. Everything returned
    sits inside the title-safe box (DESIGN.md §5).

    Returns (title_org, title_px, px, placed) where `placed` is a list of
    (key_text, label, key_x, label_x, baseline_y)."""
    uu = u(h)
    inset_x, inset_y = int(w * TITLE_SAFE), int(h * TITLE_SAFE)
    safe_w, safe_h = max(1, w - 2 * inset_x), max(1, h - 2 * inset_y)
    title_px = max(HELP_MIN_PX, int(1.4 * uu))
    head = int(2.6 * title_px)
    keys = [help_key_text(k) for k, _ in rows]
    labels = [l for _, l in rows]
    n = max(1, len(rows))

    px = max(HELP_MIN_PX, int(0.9 * uu))
    while True:
        row_h = max(1, int(1.5 * px))
        gap = int(1.4 * px)                                   # key -> label
        gutter = int(2.0 * px)                                # column -> column
        key_w = max([text_size(k, px)[0] for k in keys], default=0)
        lab_w = max([text_size(l, px)[0] for l in labels], default=0)
        col_w = key_w + gap + lab_w + gutter
        for cols in range(1, HELP_COLS_MAX + 1):
            per = -(-n // cols)                               # ceil
            if cols * col_w <= safe_w and head + per * row_h <= safe_h:
                break
        else:
            if px > HELP_MIN_PX:
                px -= 1
                continue                                      # try smaller type
            cols, per = HELP_COLS_MAX, -(-n // HELP_COLS_MAX)  # best effort
        break

    block_w = cols * col_w - gutter
    block_h = head + per * row_h
    x0 = max(inset_x, (w - block_w) // 2)
    top = inset_y + max(0, (safe_h - block_h) // 2)      # centered when it fits
    placed = []
    for i, (key, label) in enumerate(zip(keys, labels)):
        x = x0 + (i // per) * col_w
        placed.append((key, label, x, x + key_w + gap,
                       top + head + (i % per) * row_h))
    return (x0, top + title_px), title_px, px, placed


def draw_help(img, rows, accent=ACC):
    """The live key map over a 65% scrim (DESIGN.md §6.2 `?`). `rows` is
    (key, label) pairs — the registry's table plus any caller extras (TAB/Esc).
    Any key closes it; it works in every overlay state. Full-frame scrim is fine
    here: help is a modal, not part of the steady-state overlay budget.
    `accent` is the active mode's (DESIGN.md §5: exactly one accent on
    screen, owned by the active mode)."""
    h, w = img.shape[:2]
    np.copyto(img, cv2.convertScaleAbs(img, alpha=0.35))   # 65% scrim
    title_org, title_px, px, placed = help_layout(w, h, rows)
    _help_plate(img, title_org, title_px, px, placed)
    put_outlined(img, "KEYS", title_org, title_px, accent)
    for key, label, kx, lx, y in placed:
        put_outlined(img, key, (kx, y), px, accent)
        put_outlined(img, label, (lx, y), px, INK)


def _help_plate(img, title_org, title_px, px, placed):
    """A flat ground under the key table, the panel's own (PANEL at 0.86).

    The frame scrim above is a multiply, so it darkens the picture without
    flattening it: a 1-bit output is 0 vs 255, and 65% of that is 0 vs 89 —
    still hard-edged, still full-contrast, and at 720p the dither cells land
    at the same spatial scale as the glyph strokes. The table stayed legible
    but fought the background the whole time, and this is the screen that
    teaches the keys, so it has to be readable over ANY output, not most.

    The menu does not have this problem because its content sits on drawn
    chrome. The panel does not either — it lays PANEL at 0.86 under its whole
    column. This is that same plate, sized to the block, so help reads the
    same over a 1-bit Floyd-Steinberg picture as over a still.

    Cost is a blend of the block's rect only, on the frames where the modal is
    open. The steady-state overlay budget is untouched — nothing here runs
    unless `?` put the map on screen.
    """
    if not placed:
        return
    x0, ty = title_org
    left = min([x0] + [kx for _k, _l, kx, _lx, _y in placed])
    right = max(lx + text_size(label, px)[0] for _k, label, _kx, lx, _y in placed)
    top = ty - title_px
    bottom = max(y for *_r, y in placed) + int(0.35 * px)
    pad = int(0.9 * px)
    h, w = img.shape[:2]
    x_a, y_a = max(0, left - pad), max(0, top - pad)
    x_b, y_b = min(w, right + pad), min(h, bottom + pad)
    if x_b <= x_a or y_b <= y_a:
        return
    roi = img[y_a:y_b, x_a:x_b]
    plate = np.empty_like(roi)
    plate[:] = PLATE
    cv2.addWeighted(plate, HELP_PLATE_ALPHA, roi, 1.0 - HELP_PLATE_ALPHA, 0,
                    dst=roi)


@dataclass
class _Toast:
    text: str
    color: tuple
    t0: float
    ttl: float

    def alpha(self, now):
        left = self.ttl - (now - self.t0)
        if left <= 0:
            return 0.0
        return min(1.0, left / (0.45 * self.ttl))   # hold, then fade out


class Toasts:
    """Momentary feedback: one big center flash + small stacked hints.
    Silence-on-input is a bug (DESIGN.md principle 4)."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self._center = None
        self._hints = []

    def flash(self, text, color=INK):
        """Center 3.0u flash (mode change, RESET, BLACKOUT...). Replaces the last."""
        self._center = _Toast(text, color, self._now(), CENTER_TOAST_S)

    def hint(self, text, color=DIM, ttl=HINT_TOAST_S):
        """Small 0.75u hint ('? for keys', 'q again to quit'...). `ttl` covers
        the long-lived boot hint (DESIGN.md §3: 4 s), default stays 1.5 s."""
        self._hints.append(_Toast(text, color, self._now(), ttl))
        self._hints = self._hints[-3:]

    def retire(self, prefix):
        """Drop any live hint starting with *prefix* — it has been answered.

        A hint is a pointer, and a pointer outlives its usefulness the moment
        the operator follows it. The boot menu's 'enter resumes PARTICLES' is
        posted with a 4 s ttl and Enter is usually pressed inside one, so it
        stacked above the doors hint that replaces it and the operator got two
        sentences, one of them about a menu they had just left."""
        self._hints = [t for t in self._hints if not t.text.startswith(prefix)]

    def active(self):
        now = self._now()
        if self._center is not None and self._center.alpha(now) > 0:
            return True
        return any(t.alpha(now) > 0 for t in self._hints)

    def draw(self, img):
        now = self._now()
        h, w = img.shape[:2]
        uu = u(h)
        # every toast fits inside the title-safe box, whatever it says: these
        # carry exception text and file paths, and a clipped explanation of a
        # failure is not an explanation (DESIGN.md §5, §6.4)
        safe_w = max(1, w - 2 * int(w * TITLE_SAFE))
        if self._center is not None:
            a = self._center.alpha(now)
            if a <= 0:
                self._center = None
            else:
                px = fit_px(self._center.text, int(3.0 * uu), safe_w)
                tw, _, _ = text_size(self._center.text, px)
                blend_outlined(img, self._center.text, ((w - tw) // 2, (h + px) // 2),
                               px, self._center.color, a)
        self._hints = [t for t in self._hints if t.alpha(now) > 0]
        base_px = int(0.75 * uu)
        y = h - int(h * TITLE_SAFE)
        for t in reversed(self._hints):
            px = fit_px(t.text, base_px, safe_w)
            tw, _, _ = text_size(t.text, px)
            blend_outlined(img, t.text, ((w - tw) // 2, y), px, t.color, t.alpha(now))
            y -= int(1.5 * base_px)


class Osd:
    """Param readout: name + value + bar, 1.5u, bottom-left, fades in 1.5 s."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self._show = None      # (name, value_text, fill 0..1 or None, t0)

    def show(self, name, value, lo=None, hi=None, fmt="{:.2f}"):
        fill = None
        if lo is not None and hi is not None and hi > lo:
            fill = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        text = fmt.format(value) if not isinstance(value, str) else value
        self._show = (name, text, fill, self._now())

    def draw(self, img):
        if self._show is None:
            return
        name, text, fill, t0 = self._show
        left = OSD_FADE_S - (self._now() - t0)
        if left <= 0:
            self._show = None
            return
        a = min(1.0, left / (0.45 * OSD_FADE_S))
        h, w = img.shape[:2]
        uu = u(h)
        px = int(1.5 * uu)
        x = int(w * TITLE_SAFE)
        y = h - int(h * TITLE_SAFE) - int(1.2 * uu)
        blend_outlined(img, f"{name}  {text}", (x, y), px, INK, a)
        if fill is not None:
            bw, bh = int(10 * uu), max(2, int(0.5 * uu))
            y0 = y + int(0.4 * uu)
            roi = img[y0:y0 + bh, x:x + bw]
            over = roi.copy()
            cv2.rectangle(over, (0, 0), (bw - 1, bh - 1), OUTLINE, -1)
            cv2.rectangle(over, (0, 0), (int(bw * fill), bh - 1), INK, -1)
            cv2.rectangle(over, (0, 0), (bw - 1, bh - 1), DIM, 1)
            cv2.addWeighted(over, a, roi, 1.0 - a, 0, dst=roi)


class Hud:
    """The perform-surface renderer. Draw AFTER the recorder write — recordings
    never contain HUD/panel (the existing ordering invariant, kept)."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self.toasts = Toasts(now)
        self.osd = Osd(now)
        self.debug = False     # 'i' — status line variant: fps / frame-time / res

    def draw(self, img, state, status="", debug_status="", recording=False,
             blackout=False, camera_lost=False, camera_black=False,
             status_max_x=None):
        """`status_max_x`: the x the status line (and its record dot) must
        end before, or None for no limit. The shell passes the panel handle's
        left edge minus a gap: on portrait frames the line is wider than the
        room left of the handle (1080x1920: it ran from x=37 to 1250, across
        the handle at 925..1043), and the HUD draws after the handle."""
        h, w = img.shape[:2]
        uu = u(h)
        if blackout:
            draw_corner_tick(img, 2.0 * uu)   # all states, including HIDDEN
        if state is OverlayState.HIDDEN:
            # Provably clean: only already-ticking toasts and the transient
            # param OSD (both fade out), then pixels untouched. The OSD draws
            # here too because param nudging works in every overlay state
            # (DESIGN.md §6.2).
            self.toasts.draw(img)
            self.osd.draw(img)
            return img
        ix, iy = int(w * TITLE_SAFE), int(h * TITLE_SAFE)
        px = int(0.75 * uu)
        line = debug_status if self.debug else status
        # shrink (fit_px), not crop, when a limit is given: the record dot
        # needs 0.9u + its radius after the text, so that comes off first
        lpx = px
        if line and status_max_x is not None:
            room = status_max_x - ix - (int(0.9 * uu) + max(2, int(0.3 * uu))
                                        if recording else 0)
            lpx = fit_px(line, px, room)
        if line:
            put_outlined(img, line, (ix, iy + px), lpx, DIM)
        if recording:
            tw = text_size(line, lpx)[0] if line else 0
            cv2.circle(img, (ix + tw + int(0.9 * uu), iy + px // 2 + 2),
                       max(2, int(0.3 * uu)), RED, -1, cv2.LINE_AA)
        if camera_lost or camera_black:
            # persistent note sits top-left UNDER the status line, title-safe
            # (DESIGN.md §5: frame center is reserved for transient toasts;
            # only the pre-show 'waiting for camera' message may sit centered)
            #
            # A failed READ takes precedence over a black streak: when nothing
            # is arriving, the held frame's own darkness is a consequence, not
            # the fault. Every failed read used to print the Continuity
            # Camera note, so an unplugged camera — or one Zoom had taken —
            # sent the operator into iPhone settings after a phone that was
            # never involved.
            note, fix = ((CAMERA_LOST_NOTE, CAMERA_LOST_FIX) if camera_lost
                         else (CAMERA_NOTE, CAMERA_FIX))
            npx = int(1.0 * uu)
            ny = iy + px + int(1.6 * npx)
            put_outlined(img, note, (ix, ny), npx, AMBER)
            put_outlined(img, fix, (ix, ny + int(1.4 * npx)),
                         int(0.75 * uu), AMBER)
        self.toasts.draw(img)
        self.osd.draw(img)
        return img
