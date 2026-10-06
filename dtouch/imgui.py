"""Immediate-mode widget toolkit — the panel's drawing + hit-rect primitives.

Lifted out of OverlayUI (DESIGN.md §2.3 / §8 step 4) so any surface — the
Particles panel today, the home menu and per-mode panels later — draws with the
same widgets and gets the same pixels. Everything is a drawn, boxed,
hover-highlighting control with click feedback; hit-testing is a list of
(rect, kind, payload) tuples appended as widgets draw.

The toolkit is standalone: a `Gui` context carries the per-draw state (scale,
mouse, hit list, pending tooltip); widgets take an image + values + geometry
and append to `gui.hot`. Nothing here knows about OverlayUI, presets, or the
engine. ASCII glyphs only — cv2's Hershey font can't render •/■/≡/— (they show
as ???).

Every pixel dimension is authored at a 1080p baseline and multiplied by the
scale factor `Gui.s` (see `Gui.S`), floored at 1.0 by the caller so 720p/1080p
render exactly as authored and 4K renders at 2x (issue #3).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .hud import TITLE_SAFE, u

# BGR chrome — shared by the panel and every future toolkit surface.
PANEL = (34, 32, 30)
BTN = (54, 52, 50)
HOVER = (82, 86, 82)
INK = (215, 222, 218)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
TRACK = (70, 74, 72)
HANDLE = (160, 230, 175)
RED = (70, 70, 235)
DARK = (20, 22, 20)


def in_rect(rect, p):
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


# ----- scroll mechanics (pure) -----
#
# ==========================================================================
# THE SCROLL CONVENTION — read this before touching any scroll path.
#
# `scroll` is how many pixels the content column has been pushed UP past the
# top of the view. It is always >= 0. Widgets draw at `y = y0 - scroll`, so:
#
#       scroll INCREASES  ->  content moves UP  ->  you see what is BELOW
#       scroll DECREASES  ->  content moves DOWN ->  you see what is ABOVE
#
# Every input path must agree with that sentence. There are four, and they are
# only consistent if you say each one out loud:
#
#   wheel   `wheel_scroll`  delta < 0 (toward the user) INCREASES scroll
#   drag    `drag_scroll`   finger UP (y < y0)          INCREASES scroll
#                           -- the CONTENT follows the finger
#   thumb   `thumb_scroll`  finger DOWN (y > y0)        INCREASES scroll
#                           -- the THUMB follows the finger, so the content
#                              goes the OTHER way. This is the mirror of
#                              drag_scroll and it is correct: you are dragging
#                              a different object.
#   arrows  `Gui.scrollbar` the DOWN arrow               INCREASES scroll
#
# If you add a fifth path, add a line here and make it obey. Two paths that
# disagree read to the user as "the menu scrolls up and down inconsistently",
# which is exactly the bug this block exists to prevent recurring.
# ==========================================================================

SCROLL_STEP = 40    # baseline px moved by one wheel notch / one arrow click
WHEEL_NOTCH = 120   # Win32/GTK encode one wheel detent as +-120


def wheel_delta(x, y, flags):
    """Normalize one cv2 EVENT_MOUSEWHEEL callback into a single signed delta.

    The live cv2 backends disagree about WHERE the delta is, and the difference
    is not cosmetic — reading the wrong place breaks scrolling outright:

    * **Win32 / GTK / Qt** pack the delta into the HIGH 16 bits of `flags` as a
      signed short, in multiples of +-120 (one detent). `(x, y)` is the cursor.
    * **macOS / Cocoa** passes the delta in `(x, y)` — literally
      `int(event.scrollingDeltaY)`, already corrected for the system "natural
      scrolling" preference by AppKit — and `flags` carries ONLY the
      Shift/Ctrl/Alt modifier bits. See `cvSendMouseEvent:type:flags:` and
      `cvMouseEvent:` in opencv/modules/highgui/src/window_cocoa.mm.

    So on macOS `flags` is 0 on every unmodified wheel event, and any code that
    reads the direction out of it (`flags > 0`) scrolls the same way forever —
    the shipped bug: both directions scrolled DOWN, and holding Shift was the
    only way to scroll up.

    The discriminator is safe both ways: the modifier bits are all below 0x40,
    so the high word is nonzero ONLY on the flags-encoding backends, and those
    never deliver a wheel event with a zero delta.

    Note there is no supported decoder to lean on — OpenCV's
    `cv::getMouseWheelDelta` is C++-only and is not exported to the Python
    bindings (verified on cv2 4.13.0), which is most likely why the high word
    got missed in the first place.

    Sign, after this: **positive = pushed away from the user = show me what is
    above = scroll decreases**. Uniform on every backend.
    """
    hi = (flags >> 16) & 0xFFFF
    if hi:
        return hi - 0x10000 if hi >= 0x8000 else hi
    return y            # cocoa; horizontal wheels arrive as EVENT_MOUSEHWHEEL


def wheel_scroll(scroll, step, delta):
    """One wheel event, given a delta already normalized by `wheel_delta`.

    Magnitude is deliberately coarse: a detent-encoded delta (|d| >= 120) moves
    one `step` per detent, and anything smaller — a macOS trackpad's per-frame
    pixel delta — moves exactly one step per event. A truncated-to-zero delta
    moves nothing (it used to move a whole step DOWN, which is the jitter half
    of "scrolls inconsistently"). Clamping happens at draw time.
    """
    if not delta:
        return scroll
    notches = round(abs(delta) / WHEEL_NOTCH) if abs(delta) >= WHEEL_NOTCH else 1
    return scroll - step * notches if delta > 0 else scroll + step * notches


def clamp_scroll(scroll, content_h, view_h):
    """Clamp so scrolling is a no-op when everything fits."""
    return min(max(scroll, 0), max(0, content_h - view_h))


def drag_scroll(scroll0, y0, y):
    """Drag-on-empty-panel scrolling: the CONTENT follows the finger, so
    dragging UP increases scroll (see the convention block)."""
    return scroll0 + (y0 - y)


def thumb_scroll(scroll0, y0, y, travel, span):
    """Scrollbar-thumb drag: the THUMB follows the finger, so dragging DOWN
    increases scroll — the mirror of `drag_scroll`, because you have hold of a
    different object (see the convention block).

    `travel` is how far the thumb can slide (track_h - thumb_h); `span` is how
    far the content can scroll (content_h - view_h). Ungrabbable thumbs
    (travel <= 0) hold still rather than divide by zero.
    """
    if travel <= 0:
        return scroll0
    return int(round(scroll0 + (y - y0) * span / travel))


def arm_delete(armed, name):
    """Two-click destructive arm: first x-click arms, second confirms.
    Returns (new_armed, confirmed_name_or_None)."""
    if armed == name:
        return None, name
    return name, None


# ----- the panel handle (the one way back to the sidebar) -----
#
# Owner report: "on the local version sometimes the button to expand out the
# side menu was not noticeable." Measured on the old 36x30 px PANEL box with a
# 1 px TRACK border: 1.28:1 against a black frame and 1.03:1 against physarum
# veins at 720p — the box itself was invisible on the dark pictures this
# instrument mostly makes, leaving a small grey glyph. It was also 1.5u x
# 1.25u against §5's 2.75u hit target, sat outside title-safe, and the
# blackout corner tick (hud.draw_corner_tick, drawn after it) covered its
# top-right corner. The HUD chevron and the collapsed sidebar also drew two
# different glyphs for the same "open the panel" action.
#
# The fix is one handle, drawn by this one function for both callers:
#   - 2.75u square (§5 hit target), top-right corner on the 3.5% title-safe
#     inset (§5 layout discipline). The blackout tick's legs are 2u, and its
#     hypotenuse at the handle's right edge reaches y = 2u - TITLE_SAFE*w,
#     below the handle's top (TITLE_SAFE*h) for any frame wider than ~0.27x
#     its height, so the two never share a pixel — pinned by a pixel test
#     rather than asserted here.
#   - a two-tone outline, the same idea as §5's double-drawn text: a black
#     outer ring and an INK inner ring. Whatever the ground, one of the two is
#     far from it — black carries white/grey grounds, INK carries black ones —
#     so the edge clears WCAG 1.4.11's 3:1 non-text contrast without knowing
#     the picture. The worst ground is the one equally far from both rings:
#     INK-on-black is 15.41:1 (§5), so that ground sees sqrt(15.41) = 3.93:1
#     from each. Measured on rendered pixels at 1080p (2026-09-24): black
#     15.41, white 21.0, mid-grey 5.32, 1-bit noise 15.41, and 3.95 on
#     synthetic blurred physarum veins, whose soft vein edges sit near that
#     worst ground; the old box scored 1.44 on black by the same instrument.
#     tests/test_panel_handle.py pins >= 3:1 on the flat and noise grounds.
#   - the plate inside is the panel's own 86% PANEL scrim (§5), and the glyph
#     is INK in both states (the old HUD chevron used DIM). Hovered, the
#     plate is HANDLE_HOVER, not the widgets' HOVER: HOVER at 86% over a
#     white ground left the INK glyph at 3.80:1 (review 2026-09-24, below
#     the 4.5:1 text bar). HANDLE_HOVER is HOVER blended halfway to PANEL,
#     still visibly lighter than the resting plate; the hover cases of
#     tests/test_panel_handle.py measure it on every ground.
#   - no fade. A handle that fades is the "sometimes not noticeable" bug again.
#   - the open panel's close button is this same handle with ">" (the panel
#     slides back out to the right), on the same rect. Review 2026-09-24: when
#     the close button kept its old (w-40, 12) box, a second click on the
#     handle landed on whatever panel row sat under it (the TEMPLATES
#     section at 720p, preset 0 at 1080p and 4K) and loaded or folded it.
#     On one rect the second click closes the panel again, as before.
# Whether it draws at all is the caller's decision: the shell does not draw it
# in HIDDEN (the clean-output contract).

HANDLE_U = 2.75          # side, in u — DESIGN.md §5 hit-target floor
HANDLE_SCRIM = 0.86      # == the panel scrim (DESIGN.md §5)
HANDLE_HOVER = tuple((a + b) // 2 for a, b in zip(HOVER, PANEL))


def panel_handle_rect(w, h):
    """(x0, y0, x1, y1) of the panel handle for a `w`x`h` frame."""
    uu = u(h)
    side = int(math.ceil(HANDLE_U * uu))
    x1 = w - int(w * TITLE_SAFE)
    y0 = int(h * TITLE_SAFE)
    return (x1 - side, y0, x1, y0 + side)


def _rounded(img, x0, y0, x1, y1, r, color):
    """Filled rounded rectangle (inclusive corners), anti-aliased corners."""
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    if r:
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def panel_handle(img, hover=False, glyph="<"):
    """Draw the panel handle on `img` in place; return its hit rect. `glyph`
    is "<" to open the panel, ">" to close it."""
    h, w = img.shape[:2]
    rect = panel_handle_rect(w, h)
    x0, y0, x1, y1 = rect
    uu = u(h)
    dark = max(2, int(round(0.12 * uu)))     # outer black ring
    light = max(2, int(round(0.08 * uu)))    # inner INK ring
    rad = max(3, int(round(0.5 * uu)))
    roi = img[y0:y1 + 1, x0:x1 + 1]
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return rect
    # scrim against the picture itself, computed before the rings paint over it
    plate = np.empty_like(roi)
    plate[:] = HANDLE_HOVER if hover else PANEL
    scrim = cv2.addWeighted(plate, HANDLE_SCRIM, roi, 1.0 - HANDLE_SCRIM, 0)
    _rounded(roi, 0, 0, rw - 1, rh - 1, rad, (0, 0, 0))
    _rounded(roi, dark, dark, rw - 1 - dark, rh - 1 - dark,
             rad - dark, INK)
    inset = dark + light
    mask = np.zeros((rh, rw), np.uint8)
    _rounded(mask, inset, inset, rw - 1 - inset, rh - 1 - inset,
             rad - inset, 255)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    roi[:] = (scrim * a + roi * (1.0 - a) + 0.5).astype(np.uint8)
    # "<": the panel slides in from the right (">" mirrors it: back out).
    # Drawn as strokes, not Hershey text, so it centres exactly and scales
    # with u.
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    arm = int(round(0.45 * uu))
    t = max(2, int(round(0.16 * uu)))
    sgn = 1 if glyph == "<" else -1
    tip = (cx - sgn * (arm // 2), cy)
    cv2.polylines(img, [np.array([(cx + sgn * (arm // 2), cy - arm), tip,
                                  (cx + sgn * (arm // 2), cy + arm)], np.int32)],
                  False, INK, t, cv2.LINE_AA)
    return rect


class Gui:
    """Per-draw immediate-mode context: resolution scale, mouse position, the
    hit list widgets append to, and the tooltip deferred to end-of-draw."""

    def __init__(self):
        self.s = 1.0            # current UI scale (1080p baseline; caller floors at 1.0)
        self.mouse = (-1, -1)
        self.hot = []           # (rect, kind, payload) — appended as widgets draw
        self.tooltip = None     # (text, x, y) — set on info-badge hover, drawn last
        self.accent = ACC       # per-mode accent (DESIGN.md §5): selected/active chrome

    def begin(self, s, mouse, accent=ACC):
        """Start a draw pass: set scale + mouse + accent, reset hit list and tooltip."""
        self.s = s
        self.mouse = mouse
        self.accent = accent
        self.hot = []
        self.tooltip = None
        return self.hot

    # ----- scaling -----
    def S(self, n):
        """Scale a baseline pixel value by the current resolution factor (int for cv2)."""
        return int(round(n * self.s))

    # ----- primitives -----
    def text(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * self.s, color,
                    max(1, int(round(thick * self.s))), cv2.LINE_AA)

    def text_outlined(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        """Double-drawn text — black under the ink (DESIGN.md §5), for text
        that lands outside the panel scrim (rename hint, armed-delete
        confirm) where the live picture is the background."""
        t = max(1, int(round(thick * self.s)))
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * self.s,
                    (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * self.s,
                    color, t, cv2.LINE_AA)

    def box(self, img, rect, fill, border=TRACK):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, max(1, self.S(1)), cv2.LINE_AA)

    # ----- widgets -----
    def row(self, img, label, key, x, y, w, h=None, active=False, payload=None):
        """A full-width clickable row (button / toggle face)."""
        if h is None:
            h = self.S(24)
        rect = (x, y, x + w, y + h)
        hovered = in_rect(rect, self.mouse)
        fill = self.accent if active else (HOVER if hovered else BTN)
        self.box(img, rect, fill)
        self.text(img, label, x + self.S(10), y + h - self.S(8), DARK if active else INK, 0.46)
        self.hot.append(((x, y, x + w, y + h), key, payload))
        return rect

    def slider(self, img, label, attr, val, lo, hi, x, y, w, info=None, fmt=".2f"):
        """Labelled slider with an `i` tooltip badge. Hit payload carries the
        track geometry so the caller can turn a drag into a value. Returns next y."""
        rect = (x, y, x + w, y + self.S(26))
        hovered = in_rect(rect, self.mouse)
        self.box(img, rect, HOVER if hovered else BTN)
        # info badge
        ic = (x + self.S(14), y + self.S(13))
        irect = (x + self.S(4), y + self.S(3), x + self.S(24), y + self.S(23))
        cv2.circle(img, ic, self.S(8), (90, 110, 150), -1, cv2.LINE_AA)
        self.text(img, "i", ic[0] - self.S(2), ic[1] + self.S(5), (235, 240, 245), 0.42, 1)
        if info and in_rect(irect, self.mouse):
            self.tooltip = (info, x, y)
        self.text(img, label, x + self.S(30), y + self.S(17), DIM, 0.42)
        tx0, tx1 = x + self.S(96), x + w - self.S(44)
        cv2.line(img, (tx0, y + self.S(13)), (tx1, y + self.S(13)), TRACK,
                 max(1, self.S(3)), cv2.LINE_AA)
        # clamped to the track: a value outside [lo, hi] (a hand-edited preset,
        # a built-in authored against an older range) must still draw a handle
        # ON its own track — a handle floating past the end reads as a broken
        # widget, and the click that "fixes" it silently destroys the value.
        frac = min(max((val - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0
        hx = int(tx0 + frac * (tx1 - tx0))
        cv2.circle(img, (hx, y + self.S(13)), self.S(6), HANDLE, -1, cv2.LINE_AA)
        self.text(img, f"{val:{fmt}}", x + w - self.S(38), y + self.S(17), INK, 0.42)
        self.hot.append(((tx0 - self.S(8), y, tx1 + self.S(8), y + self.S(26)),
                         "slider", (attr, tx0, tx1, lo, hi)))
        return y + self.S(30)

    def section(self, img, title, is_open, x, y, w, key_hint=None):
        """Clickable section header. Returns (next_y, is_open).

        `key_hint` (DESIGN.md §4.1: MOTION (F), SIGNAL (G)) renders DIM,
        right-aligned in the header row. ASCII markers only: cv2's Hershey
        font renders no glyph for the usual disclosure triangles, so they
        come out as '???' (see module docstring)."""
        rect = (x - self.S(4), y - self.S(4), x + w, y + self.S(14))
        if in_rect(rect, self.mouse):
            self.box(img, rect, HOVER)
        self.text(img, ("- " if is_open else "+ ") + title, x, y + self.S(8),
                  INK if is_open else DIM, 0.4)
        if key_hint:
            hint = f"({key_hint})"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.4 * self.s, 1)
            self.text(img, hint, x + w - tw - self.S(6), y + self.S(8), DIM, 0.4)
        self.hot.append((rect, "section", title))
        return y + self.S(16), is_open

    def cycle(self, img, label, value, key, x, y, w, info=None):
        """< value > cycle control with an optional `i` tooltip badge (same
        contract as slider's). Hits are ("cycle", (key, ±1)). Returns next y."""
        lx = x
        if info:
            ic = (x + self.S(8), y + self.S(6))
            irect = (x - self.S(2), y - self.S(4), x + self.S(18), y + self.S(16))
            cv2.circle(img, ic, self.S(8), (90, 110, 150), -1, cv2.LINE_AA)
            self.text(img, "i", ic[0] - self.S(2), ic[1] + self.S(5),
                      (235, 240, 245), 0.42, 1)
            if in_rect(irect, self.mouse):
                self.tooltip = (info, x, y)
            lx = x + self.S(22)
        self.text(img, label, lx, y + self.S(10), DIM, 0.42)
        ry = y + self.S(16)
        bw, bh = self.S(28), self.S(24)
        lb = (x, ry, x + bw, ry + bh)
        rb = (x + w - bw, ry, x + w, ry + bh)
        self.box(img, lb, HOVER if in_rect(lb, self.mouse) else BTN)
        self.box(img, rb, HOVER if in_rect(rb, self.mouse) else BTN)
        self.text(img, "<", x + self.S(9), ry + self.S(17), INK, 0.5, 2)
        self.text(img, ">", x + w - self.S(19), ry + self.S(17), INK, 0.5, 2)
        self.text(img, str(value), x + self.S(40), ry + self.S(17), self.accent, 0.5)
        self.hot.append((lb, "cycle", (key, -1)))
        self.hot.append((rb, "cycle", (key, +1)))
        return ry + self.S(32)

    def slot_badge(self, img, slot, x, y, cw):
        """Bank-slot badge on a preset row (DESIGN.md §4.1 / §6.3): an assigned
        row shows its digit in the accent; an unassigned row shows an empty
        well. Clicking assigns the next free slot (the shell owns the policy).
        Sits left of the hover manage buttons — no overlap. Returns its rect;
        the caller appends the hit with the row's name."""
        r = (x + cw - self.S(78), y + self.S(2), x + cw - self.S(56), y + self.S(22))
        if slot:
            self.box(img, r, DARK, border=self.accent)
            self.text(img, str(slot), r[0] + self.S(7), r[3] - self.S(6),
                      self.accent, 0.42)
        else:
            self.box(img, r, HOVER if in_rect(r, self.mouse) else BTN)
        return r

    def manage_buttons(self, img, name, x, y, cw, armed):
        """Hover-revealed rename (~) and delete (x) buttons on a saved look's row.
        Their hit rects go to the FRONT of `hot` so they win over the full-row rect.
        `armed` = this name's delete is one click from confirming (see arm_delete)."""
        db = (x + cw - self.S(26), y + self.S(2), x + cw - self.S(4), y + self.S(22))
        rb = (x + cw - self.S(52), y + self.S(2), x + cw - self.S(30), y + self.S(22))
        self.box(img, rb, HOVER if in_rect(rb, self.mouse) else BTN)
        self.text(img, "~", rb[0] + self.S(6), rb[3] - self.S(7), INK, 0.45)
        self.box(img, db, RED if armed else (HOVER if in_rect(db, self.mouse) else BTN))
        self.text(img, "x", db[0] + self.S(7), db[3] - self.S(7), INK, 0.45, 2 if armed else 1)
        if armed:
            self.text_outlined(img, "sure? x again", x - self.S(118),
                               y + self.S(16), RED, 0.42, 1)
        self.hot.insert(0, (db, "del", name))
        self.hot.insert(0, (rb, "ren", name))

    def rename_box(self, img, buf, blink, x, y, cw, px):
        """The row being renamed becomes a text input (Enter saves, Esc cancels).
        `blink` is a frame counter driving the cursor; `px` is the panel's left edge."""
        rect = (x, y, x + cw, y + self.S(24))
        self.box(img, rect, DARK, border=self.accent)
        cur = "_" if (blink // 12) % 2 == 0 else ""
        self.text(img, buf + cur, x + self.S(10), y + self.S(16), self.accent, 0.46)
        self.text_outlined(img, "type name: enter=save esc=cancel",
                           px - self.S(240), y + self.S(16), self.accent, 0.42)

    # ----- deferred / chrome -----
    def draw_tooltip(self, img, clamp_h):
        """Draw the tooltip deferred by the last slider badge hover (if any)."""
        if not self.tooltip:
            return
        text, sx, sy = self.tooltip
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 30:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            lines.append(cur)
        bw, bh = self.S(250), self.S(16) * len(lines) + self.S(16)
        bx = max(sx - bw - self.S(14), self.S(10))
        by = max(min(sy, clamp_h - bh - self.S(10)), self.S(10))
        self.box(img, (bx, by, bx + bw, by + bh), (44, 48, 56), border=(120, 140, 170))
        for i, ln in enumerate(lines):
            self.text(img, ln, bx + self.S(10), by + self.S(20) + i * self.S(16), INK, 0.42)

    def scrollbar_geom(self, px, view_h, content_h, scroll):
        """Geometry of the panel's left-gutter scrollbar, or None when the
        content fits — in which case nothing draws and nothing is clickable.

        Everything lives in the S(16) gutter between the panel's left edge and
        the controls' left edge, so no control is DRAWN under it — with one
        exception that matters for hit-testing rather than for pixels: a
        section header's rect starts S(4) into the gutter, so the two overlap
        by a quarter of its width. `scrollbar` front-inserts its hits for
        exactly this reason (fixed chrome above scrolled content), which is
        what makes the overlap harmless instead of a header that eats arrow
        clicks near the top and bottom of the panel.

        The arrow BUTTONS are drawn at the shipped chrome's button scale (the
        cycle `<` `>` buttons are S(28)xS(24), the manage buttons S(22)xS(20)).
        Their HIT rects are the full gutter width and 2.75u tall — DESIGN.md
        §5's minimum target, u = view_h/45 — which costs nothing, because the
        gutter beside them is empty. Drawn small, hit large.

        Returns a dict of (x0, y0, x1, y1) rects: up_box/down_box (drawn),
        up_hit/down_hit (clickable), track, thumb — plus thumb_h/travel/span,
        the numbers `thumb_scroll` needs. `thumb` is None if the track is too
        short to hold one.
        """
        if content_h <= view_h:
            return None
        S = self.S
        gx0, gx1 = px, px + S(16)
        bx0, bx1, bh = px + S(2), px + S(14), S(22)
        # Targets first, flush against the top and bottom edges of the frame —
        # an edge target is infinitely tall to a mouse (Fitts), and flush is
        # also where the click-flash outline reads best. The button is then
        # centred inside its target.
        th = max(bh + S(4), int(round(2.75 * view_h / 45.0)))      # DESIGN.md §5
        th = min(th, max(view_h // 3, bh))
        up_hit = (gx0, 0, gx1, th)
        down_hit = (gx0, view_h - th, gx1, view_h)
        def _box(hit):
            top = (hit[1] + hit[3]) // 2 - bh // 2
            return (bx0, top, bx1, top + bh)
        up_box, down_box = _box(up_hit), _box(down_hit)
        ty0, ty1 = up_hit[3] + S(2), down_hit[1] - S(2)
        track_h, span = max(ty1 - ty0, 0), max(content_h - view_h, 1)
        thumb_h = min(max(int(track_h * view_h / content_h), S(30)), track_h)
        thumb = None
        if thumb_h > 0:
            travel = track_h - thumb_h
            top = ty0 + int(travel * (min(max(scroll, 0), span) / span))
            thumb = (bx0, top, bx1, top + thumb_h)
        return {"up_box": up_box, "down_box": down_box,
                "up_hit": up_hit, "down_hit": down_hit,
                "track": (gx0, ty0, gx1, ty0 + track_h), "thumb": thumb,
                "track_y0": ty0, "track_h": track_h,
                "thumb_h": thumb_h, "travel": track_h - thumb_h, "span": span}

    def scrollbar(self, img, px, view_h, content_h, scroll, step):
        """Scrollbar in the panel's left gutter — only when content overflows.

        Up arrow, thumb, down arrow. The arrows are real hit targets so the
        panel is scrollable with clicks alone (DESIGN.md §6.3: the mouse layer
        is a fallback that must reach everything, and a cv2 window's wheel is
        the least portable input we have — see `wheel_delta`). One click moves
        one `step`, the same step one wheel notch moves.

        Hits are front-inserted, like the collapse button and the manage
        buttons: this is fixed chrome floating above scrolled content, so a row
        that scrolled under it must never steal the click.
        """
        geom = self.scrollbar_geom(px, view_h, content_h, scroll)
        if geom is None:
            return None
        if geom["thumb"]:
            x0, y0, x1, y1 = geom["thumb"]
            hot = in_rect(geom["track"], self.mouse)
            # recessed groove first, so the empty part of the track still reads
            # as somewhere you can click (DARK is the same well the unassigned
            # slot badge uses)
            cv2.rectangle(img, (x0, geom["track_y0"]),
                          (x1, geom["track_y0"] + geom["track_h"]), DARK, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), HOVER if hot else TRACK, -1)
            self.hot.insert(0, (geom["track"], "scrolltrack",
                                (geom["track_y0"], geom["track_h"],
                                 geom["thumb_h"], geom["span"],
                                 max(view_h - step, step))))
        for box, hit, sign in ((geom["up_box"], geom["up_hit"], -1),
                               (geom["down_box"], geom["down_hit"], +1)):
            self.box(img, box, HOVER if in_rect(hit, self.mouse) else BTN)
            self._arrow(img, box, sign)
            self.hot.insert(0, (hit, "scroll", sign * step))
        return geom

    def _arrow(self, img, box, sign):
        """Filled triangle inside an arrow button — a polygon, not a glyph:
        Hershey has no triangle and `^`/`v` do not read as a matched pair
        (see the module docstring on ASCII-only text)."""
        x0, y0, x1, y1 = box
        mx, my = self.S(2), self.S(6)
        cx = (x0 + x1) // 2
        if sign < 0:
            pts = [(cx, y0 + my), (x1 - mx, y1 - my), (x0 + mx, y1 - my)]
        else:
            pts = [(cx, y1 - my), (x1 - mx, y0 + my), (x0 + mx, y0 + my)]
        cv2.fillConvexPoly(img, np.array(pts, np.int32), INK, cv2.LINE_AA)
