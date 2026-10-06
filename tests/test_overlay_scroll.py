"""Panel scrolling — at small outputs (720p) the control column is taller than the
window; wheel / arrows / thumb / drag-on-empty-panel must all make every control
reachable, and must all agree about which way is down (dtouch.imgui's convention
block).

**The bug these tests were rewritten for.** The wheel path read its direction out
of `flags > 0`. That is the Win32/GTK encoding — those backends pack a signed
+-120 delta into the HIGH 16 bits of `flags`. OpenCV's Cocoa backend does not:
it passes the delta in `(x, y)` and puts only Shift/Ctrl/Alt in `flags`
(window_cocoa.mm, `cvSendMouseEvent:type:flags:`). So on macOS `flags` was 0 on
every unmodified wheel event, `flags > 0` was always false, and the panel
scrolled DOWN in both directions — holding Shift was the only way to go up. The
old tests could not see it, because they passed the raw delta (`-120`/`+120`)
as `flags`, which is a shape no backend actually sends. Every wheel test below
now goes through `_wheel_*` emitters that reproduce a real backend byte-for-byte.
"""
import cv2
import numpy as np

from dtouch import imgui
from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui(w=1280, h=720):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _quit_bottom(ui, w=1280, h=720):
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return next(r for r, k, _ in ui._hot if k == "quit")[3]


# ----- the two real backend shapes (see the module docstring) -----

def _wheel_win32(ui, notches, cursor=(1100, 300)):
    """Win32/GTK/Qt: delta in the high 16 bits of flags, cursor in (x, y).
    Positive = pushed away from the user = show me what is above."""
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, cursor[0], cursor[1], (notches * 120) << 16)


def _wheel_cocoa(ui, delta, modifiers=0):
    """macOS/Cocoa: `int(event.scrollingDeltaY)` in y, modifiers only in flags.
    Positive = show me what is above (AppKit has already applied the system
    'natural scrolling' preference for us)."""
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, delta, modifiers)


BACKENDS = {"win32": lambda ui, n: _wheel_win32(ui, n),
            "cocoa": lambda ui, n: _wheel_cocoa(ui, n * 3)}


def test_wheel_scrolls_panel_down():
    for name, wheel in BACKENDS.items():
        ui = _ui()
        before = _quit_bottom(ui)
        wheel(ui, -1)                                     # toward the user
        assert _quit_bottom(ui) < before, f"{name}: wheel-down did not scroll down"


def test_wheel_scrolls_panel_back_up():
    """THE regression: on Cocoa both directions used to scroll DOWN."""
    for name, wheel in BACKENDS.items():
        ui = _ui()
        for _ in range(4):
            wheel(ui, -1)
        ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
        scrolled = ui.scroll
        assert scrolled > 0, f"{name}: precondition — nothing scrolled"
        wheel(ui, +1)                                     # away from the user
        ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
        assert ui.scroll < scrolled, f"{name}: wheel-up did not scroll up"


def test_wheel_direction_does_not_depend_on_held_modifiers():
    """Shift/Ctrl/Alt live in the same `flags` int the old code read the
    direction from, so a held modifier used to be the ONLY way to scroll up."""
    for mods in (0, cv2.EVENT_FLAG_SHIFTKEY, cv2.EVENT_FLAG_CTRLKEY,
                 cv2.EVENT_FLAG_ALTKEY):
        ui = _ui()
        _wheel_cocoa(ui, -3, mods)
        down = _quit_bottom(ui)
        _wheel_cocoa(ui, +3, mods)
        assert _quit_bottom(ui) > down, f"modifiers={mods} broke the direction"


def test_zero_delta_wheel_event_does_nothing():
    """A macOS trackpad drag slower than one point per frame truncates to
    `int(0.4) == 0`. Those used to scroll a whole step down each — the jitter
    half of 'scrolls up and down inconsistently'."""
    ui = _ui()
    for _ in range(10):
        _wheel_cocoa(ui, 0)
    assert _quit_bottom(ui) == _quit_bottom(_ui())


def test_wheel_does_not_move_the_hover_cursor():
    """On Cocoa (x, y) IS the delta, not a cursor. Feeding it to self.mouse
    teleported every hover highlight to the top-left corner mid-scroll."""
    ui = _ui()
    ui.mouse = (1100, 300)
    _wheel_cocoa(ui, -3)
    assert ui.mouse == (1100, 300)


def test_wheel_and_drag_agree_about_which_way_is_down():
    """The convention block in dtouch.imgui, asserted: a wheel pushed toward
    the user and a finger dragged UP both mean 'show me what is below'."""
    wheeled = _ui()
    _wheel_cocoa(wheeled, -3)
    wheeled.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    dragged = _ui()
    px = 1280 - 6
    dragged.on_mouse(cv2.EVENT_LBUTTONDOWN, px, 400, 0)
    dragged.on_mouse(cv2.EVENT_MOUSEMOVE, px, 360, cv2.EVENT_FLAG_LBUTTON)
    dragged.on_mouse(cv2.EVENT_LBUTTONUP, px, 360, 0)
    dragged.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert wheeled.scroll > 0 and dragged.scroll > 0


def test_wheel_delta_decodes_both_backends():
    """Unit-level pin on the decoder, including the discriminator: the modifier
    bits are all below 0x40, so a nonzero high word means (and only means) a
    flags-encoding backend."""
    assert imgui.wheel_delta(1100, 300, (-120) << 16) == -120
    assert imgui.wheel_delta(1100, 300, ((120) << 16) | cv2.EVENT_FLAG_SHIFTKEY) == 120
    assert imgui.wheel_delta(0, -3, 0) == -3
    assert imgui.wheel_delta(0, +3, cv2.EVENT_FLAG_CTRLKEY) == 3
    assert imgui.wheel_delta(0, 0, 0) == 0


def test_scroll_clamps_to_zero_at_top():
    for name, wheel in BACKENDS.items():
        ui = _ui()
        wheel(ui, +1)                                     # wheel up at the top
        assert _quit_bottom(ui) == _quit_bottom(_ui()), name


def test_quit_reachable_at_720p_after_scrolling():
    for name, wheel in BACKENDS.items():
        ui = _ui()
        for _ in range(60):                               # scroll all the way down
            wheel(ui, -1)
        assert _quit_bottom(ui) <= 720, name


def test_no_scroll_when_content_fits_1080p():
    ui = _ui(1920, 1080)
    before = _quit_bottom(ui, 1920, 1080)
    for _ in range(20):
        _wheel_cocoa(ui, -3)
    assert _quit_bottom(ui, 1920, 1080) == before         # clamped: content fits


def test_drag_on_empty_panel_scrolls():
    ui = _ui()
    before = _quit_bottom(ui)
    # a spot inside the panel that hits no control: just under the collapse button
    px = 1280 - 6
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, px, 400, 0)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, px, 250, cv2.EVENT_FLAG_LBUTTON)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, px, 250, 0)
    assert _quit_bottom(ui) < before


def test_collapse_button_wins_over_scrolled_row_under_it():
    """DATA-LOSS guard: the collapse button is fixed chrome floating above
    scrolled content. When a user preset's row (and its hover manage
    buttons, which front-insert their hits) scrolls underneath it, a click
    must toggle the collapse and fire nothing else: two clicks on an armed
    delete would silently destroy a saved look. Since the close button
    became the 2.75u panel handle (imgui.panel_handle), the row's delete
    sits just right of it at 720p and its rename button sits under it, so
    whichever manage button overlaps is the one this stages."""
    ui = OverlayUI(1280, 720, PRESETS + ["mine"], list(PALETTES), MATTES)
    ui.user_presets = {"mine"}
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    collapse = next(r for r, k, _ in ui._hot if k == "collapse")
    row = next(r for r, k, p in ui._hot
               if k == "preset" and p == ui.presets.index("mine"))
    # scroll the panel so the 'mine' row slides under the collapse button's
    # centre (the button is the 2.75u panel handle, taller than a row)
    cy = (collapse[1] + collapse[3]) // 2
    ui.scroll = row[1] - (cy - (row[3] - row[1]) // 2)
    ui.mouse = ((collapse[0] + collapse[2]) // 2, cy)
    ui.draw(frame, {"status": ""})       # hover draw reveals manage buttons
    hits = []
    for r, k, p in ui._hot:
        if k not in ("del", "ren") or p != "mine":
            continue
        ix0, iy0 = max(r[0], collapse[0]), max(r[1], collapse[1])
        ix1, iy1 = min(r[2], collapse[2]), min(r[3], collapse[3])
        if ix1 > ix0 and iy1 > iy0:
            hits.append(((ix0 + ix1) // 2, (iy0 + iy1) // 2))
    assert hits, "precondition: a manage button overlaps collapse"
    cx, cy = hits[0]
    ui.mouse = (cx, cy)
    ui.draw(frame, {"status": ""})
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
    assert ui.open is False              # the collapse click landed
    assert ui._del_armed is None         # the delete never armed
    assert ui.pending_delete is None
    assert ui.renaming is None
    assert ui.pending_preset is None


def test_slider_drag_still_works_with_scrolling():
    ui = _ui()
    ui.video_bg = True     # Vid mix only draws while Video bg is ON
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    payload = next(p for _, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    attr, x0, x1, lo, hi = payload
    rect = next(r for r, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    ymid = (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, x1, ymid, 0)
    assert abs(ui.video_mix - hi) < 1e-6


# ----- clickable scrollbar arrows (the mouse-only path, DESIGN.md §6.3) -----

def _arrows(ui):
    """(up_rect, down_rect) from the live hit list — payload is the signed step."""
    up = [r for r, k, p in ui._hot if k == "scroll" and p < 0]
    down = [r for r, k, p in ui._hot if k == "scroll" and p > 0]
    return up, down


def _click(ui, rect):
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, cx, cy, 0)


def test_arrows_draw_only_when_the_content_overflows():
    """720p overflows (column 1012 px); 1080p closed does not, and gets no
    scrollbar at all — the same rule the goldens hold us to."""
    up, down = _arrows(_ui(1280, 720))
    assert len(up) == 1 and len(down) == 1
    up, down = _arrows(_ui(1920, 1080))
    assert up == [] and down == []


def test_arrows_appear_when_opening_a_section_makes_the_column_overflow():
    ui = _ui(1920, 1080)
    assert _arrows(ui) == ([], [])
    for title in ("MOTION", "SIGNAL"):
        ui.sections[title] = True
    # the gated rows only draw once their master toggles are on
    # (panelspec.visible) — turn them on so the sections actually expand
    ui.flock = ui.glitch = True
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    up, down = _arrows(ui)
    assert len(up) == 1 and len(down) == 1


def test_arrows_vanish_with_the_collapsed_panel():
    ui = _ui()
    ui.open = False
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert _arrows(ui) == ([], [])


def test_arrows_meet_the_minimum_hit_target():
    """DESIGN.md §5: hit targets >= 2.75u, u = frame_height/45. The drawn
    button is deliberately smaller than its hit rect — the gutter beside it is
    empty, so the target costs nothing."""
    for w, h in ((1280, 720), (3840, 2160)):
        ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
        for title in ("MOTION", "SIGNAL"):
            ui.sections[title] = True
        ui.flock = ui.glitch = True   # expand the gated rows (panelspec.visible)
        frame = np.zeros((h, w, 3), np.uint8)
        ui.draw(frame, {"status": ""})
        ui.draw(frame, {"status": ""})
        (up,), (down,) = _arrows(ui)
        need = int(2.75 * h / 45)
        for r in (up, down):
            assert r[3] - r[1] >= need, f"{h}p: {r[3] - r[1]}px tall, need {need}"


def test_arrows_sit_at_the_top_and_bottom_of_the_panel():
    ui = _ui()
    (up,), (down,) = _arrows(ui)
    assert up[1] <= 0 + 4 and down[3] >= 720 - 4
    assert up[3] < down[1]


def test_arrows_never_cover_a_control():
    """They live in the gutter left of the controls column, so no click that
    was meant for a row can land on one."""
    ui = _ui()
    (up,), (down,) = _arrows(ui)
    controls = [r for r, k, _ in ui._hot
                if k not in ("scroll", "scrolltrack", "section", "collapse")]
    assert controls
    for arrow in (up, down):
        for c in controls:
            assert arrow[2] <= c[0], f"arrow {arrow} overlaps control {c}"


def test_down_arrow_scrolls_down_and_up_arrow_comes_back():
    ui = _ui()
    top = _quit_bottom(ui)
    (up,), (down,) = _arrows(ui)
    _click(ui, down)
    scrolled = _quit_bottom(ui)
    assert scrolled < top, "down arrow did not scroll down"
    (up,), (down,) = _arrows(ui)
    _click(ui, up)
    assert _quit_bottom(ui) == top, "up arrow did not undo it"


def test_arrow_step_matches_one_wheel_notch():
    """One click, one notch — the panel has one scroll unit, not two."""
    clicked = _ui()
    (_,), (down,) = _arrows(clicked)
    _click(clicked, down)
    clicked.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    wheeled = _ui()
    _wheel_cocoa(wheeled, -3)
    wheeled.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert clicked.scroll == wheeled.scroll == imgui.SCROLL_STEP


def test_arrow_click_activates_nothing_else():
    """Fixed chrome above scrolled content: an arrow click must not also reach
    a row that happens to have scrolled under it."""
    ui = OverlayUI(1280, 720, PRESETS + ["mine"], list(PALETTES), MATTES)
    ui.user_presets = {"mine"}
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    for _ in range(3):
        (up,), (down,) = _arrows(ui)
        _click(ui, down)
        ui.draw(frame, {"status": ""})
        assert ui.pending_preset is None
        assert ui.pending_delete is None and ui._del_armed is None
        assert ui.pending_save is False and ui.quit is False
        assert ui.pending_commands == []
        assert ui.sections == {"TEMPLATES": True, "SOURCE": True, "LOOK": True,
                               "MOTION": False, "SIGNAL": False}


def test_arrows_clamp_at_both_ends():
    ui = _ui()
    for _ in range(40):
        (up,), (down,) = _arrows(ui)
        _click(ui, down)
        ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert ui.scroll == ui._content_h - 720
    for _ in range(40):
        (up,), (down,) = _arrows(ui)
        _click(ui, up)
        ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert ui.scroll == 0


# ----- the thumb, which used to run away from the finger -----

def _geom(ui, h=720):
    return ui._gui.scrollbar_geom(1280 - ui._panel_px, h, ui._content_h, ui.scroll)


def test_thumb_drag_follows_the_finger():
    """REGRESSION: the thumb was drawn but not hit-tested, so grabbing it fell
    through to the drag-the-content path and the thumb ran UP as you pulled it
    DOWN — the single most legible 'scrolls inconsistently' symptom."""
    ui = _ui()
    g0 = _geom(ui)
    tx = (g0["thumb"][0] + g0["thumb"][2]) // 2
    ty = (g0["thumb"][1] + g0["thumb"][3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, tx, ty, 0)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, tx, ty + 120, cv2.EVENT_FLAG_LBUTTON)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, tx, ty + 120, 0)
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    g1 = _geom(ui)
    assert ui.scroll > 0, "dragging the thumb down did not scroll down"
    assert g1["thumb"][1] > g0["thumb"][1], "the thumb moved away from the finger"


def test_thumb_tracks_the_scroll_position_end_to_end():
    ui = _ui()
    top = _geom(ui)
    ui.scroll = 10 ** 6
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    bottom = _geom(ui)
    assert top["thumb"][1] == top["track_y0"]
    assert bottom["thumb"][3] == top["track_y0"] + top["track_h"]


def test_clicking_the_empty_track_pages():
    ui = _ui()
    g = _geom(ui)
    below = g["thumb"][3] + (g["track"][3] - g["thumb"][3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, (g["track"][0] + g["track"][2]) // 2, below, 0)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, (g["track"][0] + g["track"][2]) // 2, below, 0)
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    assert ui.scroll > imgui.SCROLL_STEP    # a page, not a step


def test_drag_on_empty_panel_and_thumb_drag_go_opposite_ways():
    """Both are correct: one drags the content, the other drags the thumb.
    Pinned so a future 'consistency' cleanup cannot quietly align them."""
    content = _ui()
    px = 1280 - 6
    content.on_mouse(cv2.EVENT_LBUTTONDOWN, px, 400, 0)
    content.on_mouse(cv2.EVENT_MOUSEMOVE, px, 500, cv2.EVENT_FLAG_LBUTTON)
    content.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})

    thumb = _ui()
    thumb.scroll = 150
    thumb.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    g = _geom(thumb)
    tx = (g["thumb"][0] + g["thumb"][2]) // 2
    ty = (g["thumb"][1] + g["thumb"][3]) // 2
    thumb.on_mouse(cv2.EVENT_LBUTTONDOWN, tx, ty, 0)
    thumb.on_mouse(cv2.EVENT_MOUSEMOVE, tx, ty + 100, cv2.EVENT_FLAG_LBUTTON)
    thumb.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})

    assert content.scroll == 0            # dragged content down: already at top
    assert thumb.scroll > 150             # dragged thumb down: scrolled down
