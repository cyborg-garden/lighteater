"""The panel handle — the one button that opens the side panel.

Owner report: "on the local version sometimes the button to expand out the
side menu was not noticeable." The old box measured 1.28:1 against a black
frame (1.03:1 on physarum veins at 720p), was 1.5u x 1.25u against DESIGN.md
§5's 2.75u hit target, and the blackout tick covered its corner. These pin the
fix on RENDERED PIXELS, through both callers (the HUD chevron in shell.py and
the collapsed sidebar in overlay_ui.py), so a later restyle cannot quietly
reintroduce any of the four.
"""
import types

import numpy as np
import pytest

from dtouch import imgui
from dtouch.hud import TITLE_SAFE, Hud, OverlayState, u
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.overlay_ui import OverlayUI
from dtouch.shell import PANEL_OPEN, Host

from test_shell_fixes import SyntheticSource, _paths

SIZES = [(1280, 720), (1920, 1080), (3840, 2160)]
WCAG_NON_TEXT = 3.0      # WCAG 2.x 1.4.11 non-text contrast
WCAG_TEXT = 4.5          # the glyph is the label, so it is held to the text bar


def _lum(bgr):
    c = np.asarray(bgr, np.float64)[..., ::-1] / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    # elementwise, not `@`: macOS Accelerate matmul raises spurious FP warnings
    return (c * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1)


def _cr(a, b):
    a, b = max(a, b), min(a, b)
    return (a + 0.05) / (b + 0.05)


def _grounds(w, h):
    rng = np.random.default_rng(7)
    return {
        "black": np.zeros((h, w, 3), np.uint8),
        "white": np.full((h, w, 3), 255, np.uint8),
        "midgrey": np.full((h, w, 3), 128, np.uint8),
        "panelgrey": np.full((h, w, 3), imgui.PANEL, np.uint8),
        # per-pixel 1-bit noise: the busiest frame the dither modes make
        "busy_1bit": ((rng.random((h, w, 1)) > 0.5).astype(np.uint8)
                      .repeat(3, 2) * 255),
    }


def _hud_chevron(img, mouse=(-1, -1)):
    h, w = img.shape[:2]
    ui = OverlayUI(w, h, ["abstract"], ["ice"], ["auto"])
    ui.mouse = mouse
    Host._draw_panel_chevron(types.SimpleNamespace(ui=ui), img)
    return next(r for r, k, _ in ui._hot if k == PANEL_OPEN)


def _collapsed_sidebar(img, mouse=(-1, -1)):
    h, w = img.shape[:2]
    ui = OverlayUI(w, h, ["abstract"], ["ice"], ["auto"])
    ui.mouse = mouse
    ui.open = False
    ui.draw(img, {"status": ""})
    return next(r for r, k, _ in ui._hot if k == "collapse")


CALLERS = {"hud": _hud_chevron, "sidebar": _collapsed_sidebar}


def _ring_samples(img, rect, depth):
    """Pixels `depth` px inside the handle edge, from the middle half of each
    of the four sides (the rounded corners are left out)."""
    x0, y0, x1, y1 = rect
    qx, qy = (x1 - x0) // 4, (y1 - y0) // 4
    return np.concatenate([
        img[y0 + depth, x0 + qx:x1 - qx],
        img[y1 - depth, x0 + qx:x1 - qx],
        img[y0 + qy:y1 - qy, x0 + depth],
        img[y0 + qy:y1 - qy, x1 - depth],
    ])


@pytest.mark.parametrize("caller", sorted(CALLERS))
@pytest.mark.parametrize("w,h", SIZES)
def test_handle_edge_clears_3_to_1_on_every_ground(caller, w, h):
    uu = u(h)
    dark = max(2, int(round(0.12 * uu)))
    light = max(2, int(round(0.08 * uu)))
    for name, ground in _grounds(w, h).items():
        img = ground.copy()
        rect = CALLERS[caller](img)
        outer = np.median(_lum(_ring_samples(img, rect, dark // 2)))
        inner = np.median(_lum(_ring_samples(img, rect, dark + light // 2)))
        # every luminance the ground actually has (1-bit noise has two): the
        # edge must hold against the WORST of them
        for g in np.unique(np.round(_lum(ground[:8, :8].reshape(-1, 3)), 4)):
            best = max(_cr(outer, g), _cr(inner, g))
            assert best >= WCAG_NON_TEXT, (
                f"{caller} {w}x{h} on {name}: edge {best:.2f}:1 < 3:1")


@pytest.mark.parametrize("hover", [False, True])
@pytest.mark.parametrize("caller", sorted(CALLERS))
@pytest.mark.parametrize("w,h", SIZES)
def test_glyph_reads_against_its_plate(caller, w, h, hover):
    """Both states: the hovered plate (review 2026-09-24) once left the
    glyph at 3.80:1 on white, and only the resting state was drawn here."""
    uu = u(h)
    inset = max(2, int(round(0.12 * uu))) + max(2, int(round(0.08 * uu)))
    hx0, hy0, hx1, hy1 = imgui.panel_handle_rect(w, h)
    mouse = ((hx0 + hx1) // 2, (hy0 + hy1) // 2) if hover else (-1, -1)
    for name, ground in _grounds(w, h).items():
        img = ground.copy()
        x0, y0, x1, y1 = CALLERS[caller](img, mouse)
        L = _lum(img[y0 + inset + 2:y1 - inset - 1,
                     x0 + inset + 2:x1 - inset - 1].reshape(-1, 3))
        plate, glyph = np.median(L), L.max()
        assert _cr(glyph, plate) >= WCAG_TEXT, (
            f"{caller} {w}x{h} on {name}: glyph {_cr(glyph, plate):.2f}:1")


@pytest.mark.parametrize("caller", sorted(CALLERS))
@pytest.mark.parametrize("w,h", SIZES)
def test_handle_is_a_full_hit_target_inside_title_safe(caller, w, h):
    ground = np.full((h, w, 3), 255, np.uint8)
    img = ground.copy()
    x0, y0, x1, y1 = CALLERS[caller](img)
    need = 2.75 * u(h)
    assert x1 - x0 >= need and y1 - y0 >= need
    assert x1 <= w - int(w * TITLE_SAFE) and y0 >= int(h * TITLE_SAFE)
    # the DRAWN handle fills the hit target, not a small box inside a big rect
    changed = np.any(img != ground, axis=2)
    ys, xs = np.nonzero(changed[y0 - 1:y1 + 2, x0 - 1:x1 + 2])
    assert xs.max() - xs.min() >= need and ys.max() - ys.min() >= need


@pytest.mark.parametrize("w,h", SIZES)
def test_hud_and_sidebar_draw_the_same_handle(w, h):
    rng = np.random.default_rng(1)
    ground = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a, b = ground.copy(), ground.copy()
    ra, rb = _hud_chevron(a), _collapsed_sidebar(b)
    assert ra == rb
    x0, y0, x1, y1 = ra
    assert np.array_equal(a[y0:y1 + 1, x0:x1 + 1], b[y0:y1 + 1, x0:x1 + 1])


@pytest.mark.parametrize("caller", sorted(CALLERS))
@pytest.mark.parametrize("w,h", SIZES + [(1080, 1080), (1080, 1920)])
def test_blackout_tick_never_covers_the_handle(caller, w, h):
    """Same pixels with blackout armed or not, drawn in compose order (the
    handle, then the HUD with its tick on top) — square and portrait frames
    included, where the tick reaches furthest toward the handle."""
    state = OverlayState.HUD if caller == "hud" else OverlayState.PANEL
    shots = []
    for blackout in (False, True):
        img = np.zeros((h, w, 3), np.uint8)
        rect = CALLERS[caller](img)
        Hud().draw(img, state, blackout=blackout)
        shots.append(img)
    x0, y0, x1, y1 = rect
    assert not np.array_equal(shots[0], shots[1]), "precondition: tick drawn"
    assert np.array_equal(shots[0][y0:y1 + 1, x0:x1 + 1],
                          shots[1][y0:y1 + 1, x0:x1 + 1])


@pytest.mark.parametrize("w,h", SIZES)
def test_second_click_on_the_handle_closes_the_panel_and_changes_nothing(w, h):
    """The double-click cohort (shell.py) clicks the handle twice. When the
    open panel's close button sat at the old (w-40, 12) box, the second
    click landed on the row under the handle instead: TEMPLATES folded at
    720p, preset 0 loaded at 1080p and 4K. The close button is the same
    handle on the same rect, so the second click closes the panel and no
    look, section, slot or rename moves."""
    cv2 = pytest.importorskip("cv2")
    ui = OverlayUI(w, h, ["abstract", "portrait"], ["ice"], ["auto"])
    ui.bank = {1: "abstract", 2: "portrait"}
    ui.open = False
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    x0, y0, x1, y1 = imgui.panel_handle_rect(w, h)
    c = ((x0 + x1) // 2, (y0 + y1) // 2)
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, *c, 0)
    assert ui.open is True
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    closes = [r for r, k, _ in ui._hot if k == "collapse"]
    assert closes == [(x0, y0, x1, y1)]
    sections, preset_idx = dict(ui.sections), ui.preset_idx
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, *c, 0)
    assert ui.open is False
    assert ui.sections == sections
    assert ui.preset_idx == preset_idx and ui.pending_preset is None
    assert ui.pending_slot is None and ui.renaming is None


@pytest.mark.parametrize("w,h", SIZES)
def test_record_dot_sits_beside_the_collapsed_handle_not_on_it(w, h):
    plain = np.zeros((h, w, 3), np.uint8)
    rec = plain.copy()
    _collapsed_sidebar(plain)
    ui = OverlayUI(w, h, ["abstract"], ["ice"], ["auto"])
    ui.mouse, ui.open, ui.record = (-1, -1), False, True
    ui.draw(rec, {"status": ""})
    x0, y0, x1, y1 = imgui.panel_handle_rect(w, h)
    assert not np.array_equal(plain, rec), "precondition: dot drawn"
    assert np.array_equal(plain[y0:y1 + 1, x0:x1 + 1],
                          rec[y0:y1 + 1, x0:x1 + 1])


def test_handle_is_drawn_in_hud_and_absent_in_hidden(tmp_path):
    """Through the real per-frame compose: HUD shows the handle; HIDDEN (the
    clean-output contract) leaves its rect exactly as the picture was."""
    w, h = 1280, 720
    host = Host(DitherGirlMode(), source=SyntheticSource(), res=(w, h),
                show=False, preset=None, max_frames=1, **_paths(tmp_path))
    host.run()
    host.menu.open = False
    host.ps.blackout = False
    x0, y0, x1, y1 = imgui.panel_handle_rect(w, h)
    out = np.full((h, w, 3), 255, np.uint8)      # white: the edge is black

    host.overlay = OverlayState.HUD
    bgr = host._compose_frame(out.copy(), None)
    assert any(k == PANEL_OPEN for _, k, _ in host.ui._hot)
    assert not np.array_equal(bgr[y0:y1 + 1, x0:x1 + 1],
                              out[y0:y1 + 1, x0:x1 + 1])

    host.overlay = OverlayState.HIDDEN
    bgr = host._compose_frame(out.copy(), None)
    assert host.ui._hot == []
    assert np.array_equal(bgr[y0:y1 + 1, x0:x1 + 1],
                          out[y0:y1 + 1, x0:x1 + 1])


@pytest.mark.parametrize("w,h", SIZES + [(1080, 1920)])
def test_open_panel_title_stays_clear_of_the_close_handle(w, h):
    """The open panel's title (drawn first) never shares a pixel with the
    close handle (drawn last, on top). Review 2026-09-24: at 720p the '>'
    covered the 'LES' of 'lighteater - PARTICLES' and the regenerated
    goldens had it baked in. Drawn with the title alone vs with it blanked,
    the difference must stay left of the handle."""
    shots = []
    for title in ("lighteater - PARTICLES", ""):
        ui = OverlayUI(w, h, ["abstract"], ["ice"], ["auto"])
        ui.mouse, ui.open = (-1, -1), True
        ui.panel_title = title
        img = np.zeros((h, w, 3), np.uint8)
        ui.draw(img, {"status": ""})
        shots.append(img)
    diff = np.any(shots[0] != shots[1], axis=2)
    assert diff.any(), "precondition: title drawn"
    x0, y0, x1, y1 = imgui.panel_handle_rect(w, h)
    assert not diff[y0:y1 + 1, x0:x1 + 1].any()
    assert not diff[y0:y1 + 1, x0 - 4:x0].any()   # a visible gap too


@pytest.mark.parametrize("overlay", [OverlayState.HUD, OverlayState.PANEL])
@pytest.mark.parametrize("w,h", [(1080, 1920), (1280, 720)])
def test_status_line_stays_clear_of_the_handle(tmp_path, w, h, overlay):
    """Through the real compose, with the real status line (Host._status_line
    — the blackout test above draws Hud with an empty one): the handle's
    rect is identical whether or not the status line is drawn. Review
    2026-09-24: at 1080x1920 the line ran from x=37 to 1250 and the HUD,
    drawn after the handle, painted it across the '<'."""
    host = Host(DitherGirlMode(), source=SyntheticSource(), res=(w, h),
                show=False, preset=None, max_frames=1, **_paths(tmp_path))
    host.run()
    host.menu.open = False
    host.ps.blackout = False
    host.overlay = overlay
    assert host._status_line(), "precondition: a real status string"
    out = np.full((h, w, 3), 255, np.uint8)
    with_line = host._compose_frame(out.copy(), None)
    real = host._status_line
    host._status_line = lambda: ""
    try:
        without = host._compose_frame(out.copy(), None)
    finally:
        host._status_line = real
    assert not np.array_equal(with_line, without), "precondition: line drawn"
    x0, y0, x1, y1 = imgui.panel_handle_rect(w, h)
    assert np.array_equal(with_line[y0:y1 + 1, x0:x1 + 1],
                          without[y0:y1 + 1, x0:x1 + 1])
