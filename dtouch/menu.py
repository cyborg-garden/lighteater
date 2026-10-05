"""Home menu + boot card — an in-app switcher screen, not a launcher (DESIGN.md §3).

The menu is a shell overlay state, not a Mode: one window, the live camera
behind it rendered through 1-bit blue-noise dither and dimmed under a 65%
scrim (the menu proves the camera works and demos the newest feature before a
single click). Cards come from the mode REGISTRY — one import + one line adds
a card — plus one dimmed reserved card for Flocking (DESIGN.md §2.5: the
landing site for issue #4's body-driven flock).

Navigation (DESIGN.md §3): the mode's letter or digit selects-and-enters,
','/'.' move selection, Enter commits, Esc returns to the running mode
untouched, mouse clicks a card. `Menu` is pure state (unit-testable without a
window); `draw_menu` renders and returns the click rects.

BOOT menu (DESIGN.md §3, amended 2026-08-15 — the user's call): the app now
opens on this menu, so a menu can be showing before any mode has been *entered*.
`show(active_id, boot=True)` marks that state, and it changes exactly one thing:
the dismiss keys (Esc, `m`) commit the selection instead of "returning to the
running mode", because at boot there is no running mode to return to — a
dismiss that left the menu open, or dropped you behind it, would be the dead
end principle 5 forbids. Everything else about the menu is identical.

Mode switches draw a static boot card first (mode glyph + name in the mode's
accent, one frame, no animation — the cv2 loop is single-threaded, so nothing
can animate during GL teardown). The recorder captures the card, not a gray
flash.

ASCII text only (Hershey fonts render non-ASCII as '?').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .dither import blue_noise_dither
from .hud import DIM, INK, TITLE_SAFE, put_outlined, text_size, u
from .imgui import in_rect
from .modes import REGISTRY

MENU_SCRIM = 0.65               # DESIGN.md §5: menu scrim 65%

# The autopilot card. It is not a Mode — there is nothing to switch INTO — so
# it commits to its own verb and the shell toggles dtouch.auto.Autopilot.
# It sits on the menu because the menu is where you decide what the next
# stretch of time looks like, and "you decide" and "it decides" belong on the
# same screen.
AUTO_ID = "auto"
AUTO_ACCENT = (120, 230, 255)   # BGR

# Arrow keys as they arrive here: cv2's waitKey code masked with `& 0xFF` by
# the shell loop. macOS reports 63232..63235, which mask to these four.
#
# The menu is the first screen anyone meets and it is a row of cards, so the
# arrows are the obvious thing to press — a child reaches for them before they
# reach for `,` and `.`. They used to land on the unknown branch and hint
# "? for keys", which is feedback pointing at a key map that does not mention
# menu navigation, so it answered a question nobody asked.
#
# This does NOT make the arrows load-bearing (DESIGN.md §6.2 — their codes are
# platform-dependent, and these values are macOS's). `,`/`.` remain the
# documented navigation and the menu's own hint line still names them; on a
# platform that reports something else the arrows fall through to the same
# hint they gave before. Nothing is reachable ONLY by arrow.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


@dataclass
class Card:
    id: Optional[str]           # mode id, or None for a reserved card
    title: str
    blurb: str
    accent: tuple               # BGR
    key: Optional[str]          # select-and-enter letter, or None
    enabled: bool = True


def registry_cards():
    """One card per registered mode, plus the dimmed reserved Flocking card
    (DESIGN.md §2.5 — 'coming soon', not selectable)."""
    cards = [Card(m.id, m.title, getattr(m, "blurb", ""), m.accent,
                  getattr(m, "key", m.id[:1]))
             for m in REGISTRY]
    cards.append(Card(AUTO_ID, "AUTO", "it plays itself", AUTO_ACCENT, "a"))
    cards.append(Card(None, "FLOCKING", "coming soon", DIM, None, enabled=False))
    return cards


class Menu:
    """Menu state machine — open flag, selection, key/click routing.

    `key()` returns ("switch", mode_id) when a mode card is committed,
    ("auto", None) when the AUTO card is committed (autopilot is not a mode —
    see AUTO_ID),
    ("close", None) when the menu dismisses, ("quit", None) when q closes the
    menu and asks the shell to arm the quit confirm (DESIGN.md §3),
    ("soon", title) when the key names the reserved card, ("unknown", None) for
    keys the menu doesn't know (the shell hints — silence-on-input is a bug
    there too), and (None, None) only when the key DID something visible on
    screen. Commit and dismiss both close the menu; the shell routes the
    switch.

    In the BOOT menu (`show(..., boot=True)`) the dismiss keys commit instead,
    so Esc/`m` return ("switch", mode_id) too — see the module docstring."""

    def __init__(self, cards=None):
        self.cards = list(cards) if cards is not None else registry_cards()
        self.open = False
        self.boot = False        # opened at launch, before any mode was entered
        self.sel = 0
        self.rects = []          # (rect, card) from the last draw — click targets

    # ----- lifecycle -----
    def show(self, active_id=None, boot=False):
        self.open = True
        self.boot = bool(boot)
        self.sel = next((i for i, c in enumerate(self.cards)
                         if c.enabled and c.id == active_id), self.sel)
        if not self._enabled(self.sel):
            self.sel = next(i for i, c in enumerate(self.cards) if c.enabled)

    def close(self):
        self.open = False
        self.boot = False

    def toggle(self, active_id=None):
        if self.open:
            self.close()
        else:
            self.show(active_id)

    # ----- navigation -----
    def _enabled(self, i):
        return 0 <= i < len(self.cards) and self.cards[i].enabled

    def move(self, d):
        """Step selection by d, skipping disabled cards, wrapping."""
        n = len(self.cards)
        i = self.sel
        for _ in range(n):
            i = (i + d) % n
            if self.cards[i].enabled:
                self.sel = i
                return

    def _commit(self, i):
        self.sel = i
        self.close()
        card = self.cards[i]
        if card.id == AUTO_ID:
            # not a mode switch: the shell hands this to the autopilot, which
            # then drives the ordinary mode and preset mailboxes from inside
            return "auto", None
        return "switch", card.id

    def _soon(self, i):
        """Answer for a key that named a card that is on screen but not
        selectable. `3` used to return (None, None) — the reserved 'coming
        soon' card is RIGHT THERE and dashed, and pressing its number did
        nothing at all, which reads as a broken menu rather than a card that
        is not built yet. Anything that names no card is just unknown."""
        if 0 <= i < len(self.cards):
            return "soon", self.cards[i].title
        return "unknown", None

    def key(self, code):
        """Route one cv2.waitKey code. See class docstring for returns."""
        if code == 27 or code in (ord("m"), ord("M")):   # Esc / m: back untouched
            if self.boot and self._enabled(self.sel):
                # boot menu: there is nothing behind to go back TO, so the
                # dismiss keys enter the selected mode (DESIGN.md §3, amended)
                return self._commit(self.sel)
            self.close()
            return "close", None
        if code in (ord("q"), ord("Q")):
            # DESIGN.md §3: q closes the menu AND arms the quit confirm
            self.close()
            return "quit", None
        if code in (13, 10):                             # Enter commits
            if self._enabled(self.sel):
                return self._commit(self.sel)
            return self._soon(self.sel)
        if code in (ord(","), LEFT, UP):
            self.move(-1)
            return None, None                            # the selection moved
        if code in (ord("."), RIGHT, DOWN):
            self.move(+1)
            return None, None
        if 32 <= code <= 126:
            ch = chr(code).lower()
            if ch.isdigit():                             # digit selects-and-enters
                i = int(ch) - 1
                if self._enabled(i):
                    return self._commit(i)
                # `0` (i = -1) and every digit past the last card are simply
                # unknown; a digit that names the reserved card gets the real
                # answer instead of a deflection to the key map.
                return self._soon(i)
            for i, c in enumerate(self.cards):           # letter selects-and-enters
                if c.enabled and c.key and ch == c.key.lower():
                    return self._commit(i)
        return "unknown", None      # §3: unknown keys hint, never vanish

    def click(self, pt):
        """A mouse click at pt: the (action, value) of the card under it, or
        (None, None) when the click hit no enabled card.

        This used to return `_commit(i)[1]` — the VALUE only. That is a mode
        id for a mode card and None for the AUTO card, so a click on AUTO was
        indistinguishable from a click on empty space: the menu closed and
        nothing happened, while the same card's letter key worked. Returning
        the verb keeps mouse and keyboard on one contract, which is what the
        rest of the menu already promises.
        """
        for rect, card in self.rects:
            if card.enabled and in_rect(rect, pt):
                return self._commit(self.cards.index(card))
        return None, None


def _dashed_rect(img, x0, y0, x1, y1, color, thickness=1, dash=8, gap=6):
    """Dashed rectangle border (short segments) — the reserved 'coming soon'
    card's frame (DESIGN.md §3 sketch: it is drawn as not-yet-solid)."""
    step = dash + gap

    def _seg(pa, pb):
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)

    for x in range(x0, x1, step):
        _seg((x, y0), (min(x + dash, x1), y0))
        _seg((x, y1), (min(x + dash, x1), y1))
    for y in range(y0, y1, step):
        _seg((x0, y), (x0, min(y + dash, y1)))
        _seg((x1, y), (x1, min(y + dash, y1)))


HINT_TOAST_ROWS = 3     # hud.Toasts keeps at most 3 hints, stacked upward


def menu_hint(boot=False):
    """The menu's bottom line. At boot Esc does not go 'back' (there is nothing
    behind it) — it enters the selection, so listing it would be a lie; the
    other door, `q`, is named instead."""
    if boot:
        return ", . move - enter select - q quit"
    return ", . move - enter select - esc back"


def hint_baseline(h):
    """Baseline y for the menu's bottom hint line — issue #18.

    The HUD's hint toasts stack UPWARD from `h - title-safe inset`
    (hud.Toasts.draw), which is exactly where this line used to sit: at boot,
    the 4 s doors hint and the menu's own hint were drawn centered on one
    baseline and rendered as a single unreadable smear — two messages, neither
    readable, on the first frame the user ever sees.

    The menu's line is layout and the toasts are transient, so the layout moves
    up and leaves the stack its full room. The offset is FIXED (the whole
    stack, not the current occupancy) so the line never jumps around as toasts
    come and go — a hint line that moves while you are reading it is its own
    small bug."""
    uu = u(h)
    hp = int(0.75 * uu)                       # same size hud.Toasts uses
    row = int(1.5 * hp)                       # ...and the same row pitch
    return h - int(h * TITLE_SAFE) - HINT_TOAST_ROWS * row - hp // 2


def draw_menu(img, cam_bgr, cards, sel, boot=False):
    """Render the menu over `img` (BGR, in place): 1-bit blue-noise-dithered
    live camera under a 65% scrim, 'lighteater' top-left, a centered row of mode
    cards. Returns the click rects [(rect, card), ...]."""
    h, w = img.shape[:2]
    uu = u(h)

    # background: the live camera through the newest feature (1-bit blue
    # noise). Perf: dithered at HALF res then NEAREST-upscaled — the doubled
    # dither cells read as deliberate texture, and the full-res version cost
    # ~38 ms/frame at 4K for a dimmed backdrop.
    if cam_bgr is not None:
        hw, hh = max(1, w // 2), max(1, h // 2)
        cam = cv2.resize(cam_bgr, (hw, hh))
        gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lit = blue_noise_dither(gray, bits=1)
        shade = np.uint8(round(255 * (1.0 - MENU_SCRIM)))   # dimmed under the scrim
        small = (lit * float(shade)).astype(np.uint8)
        img[:] = cv2.resize(small, (w, h),
                            interpolation=cv2.INTER_NEAREST)[:, :, None]
    else:
        img[:] = 0

    ix, iy = int(w * TITLE_SAFE), int(h * TITLE_SAFE)
    tpx = int(1.4 * uu)
    put_outlined(img, "lighteater", (ix, iy + tpx), tpx, INK)

    # centered row of cards
    n = len(cards)
    cw, ch, gap = int(10.0 * uu), int(7.0 * uu), int(1.5 * uu)
    total = n * cw + (n - 1) * gap
    x0 = max((w - total) // 2, ix)
    y0 = (h - ch) // 2
    rects = []
    for i, c in enumerate(cards):
        x = x0 + i * (cw + gap)
        rect = (x, y0, x + cw, y0 + ch)
        color = c.accent if c.enabled else DIM
        selected = i == sel
        if c.enabled:
            cv2.rectangle(img, (x, y0), (x + cw, y0 + ch), color,
                          max(2, int(0.2 * uu)) if selected else 1, cv2.LINE_AA)
        else:
            # reserved card (DESIGN.md §3): dashed border, stays dimmed
            _dashed_rect(img, x, y0, x + cw, y0 + ch, color,
                         dash=max(4, int(0.4 * uu)), gap=max(3, int(0.3 * uu)))
        # title (CAPS per the §3 sketch, centered), blurb, key hint — every
        # state redundantly coded (color + the selection border + [key] hint)
        title = c.title.upper()
        tp = int(1.0 * uu)
        tw = text_size(title, tp)[0]
        pad = int(0.6 * uu)
        if tw > cw - 2 * pad:                # CIRCUIT BENDER: shrink to fit
            tp = max(int(tp * (cw - 2 * pad) / tw), 8)
            tw = text_size(title, tp)[0]
        put_outlined(img, title, (x + (cw - tw) // 2, y0 + int(2.2 * uu)), tp,
                     color if (selected or not c.enabled) else INK)
        bp = int(0.62 * uu)
        for j, line in enumerate(c.blurb.split("\n")):
            bw = text_size(line, bp)[0]
            put_outlined(img, line, (x + (cw - bw) // 2,
                                     y0 + int(3.6 * uu) + j * int(1.0 * uu)),
                         bp, DIM)
        if c.key:
            hint = f"[{c.key.upper()}]"
            hw = text_size(hint, bp)[0]
            put_outlined(img, hint, (x + (cw - hw) // 2, y0 + ch - int(0.8 * uu)),
                         bp, color if selected else DIM)
        rects.append((rect, c))

    # bottom hint line (DESIGN.md §3): part of the layout — 0.75u, DIM,
    # bottom-center, title-safe, and clear of the HUD's toast stack (issue #18)
    hint = menu_hint(boot)
    hp = int(0.75 * uu)
    hw = text_size(hint, hp)[0]
    put_outlined(img, hint, ((w - hw) // 2, hint_baseline(h)), hp, DIM)
    return rects


def render_boot_card(res, title, accent, glyph=None):
    """The static boot card shown for one frame during a mode switch
    (DESIGN.md §3): mode glyph + name in the mode's accent on black. Returns a
    BGR image at `res` — the caller shows it and (if recording) writes its RGB
    conversion, so the recorder captures the card, not a gray flash."""
    w, h = res
    img = np.zeros((h, w, 3), np.uint8)
    uu = u(h)
    glyph = (glyph or title[:1]).upper()
    gp = int(8.0 * uu)
    gw = text_size(glyph, gp)[0]
    put_outlined(img, glyph, ((w - gw) // 2, h // 2), gp, accent)
    tp = int(3.0 * uu)
    tw = text_size(title, tp)[0]
    put_outlined(img, title, ((w - tw) // 2, h // 2 + int(4.2 * uu)), tp, accent)
    return img
