"""The ink's style options (dtouch/shaders/alive/ink.frag): paper and the
mirror fold, and the inkblot look that carries both over the camera.

Holds: off is the shipped ink pixel for pixel (against a frozen copy of the
pre-style shader); paper is light, the fold is a real mirror and follows
the performer's side through a hysteresis and a hold; K and Y are free keys
that leave AUTO running; inkblot lands on the fractal step with paper, fold
4 and the video background on, every other look and panic turn the options
off, and the landing never sets them. GPU tests skip where no standalone GL
context can be made (CI).
"""
import math
import os

import numpy as np
import pytest

from dtouch.alive import ALIVE, FOLD_START, MIRRORS, fold_side
from dtouch.modes.physarum import INK_LOOKS, LOOK_FRACTAL, PALETTES_PH, PhysarumMode
from dtouch.physarum_alive import GLSL_VERSION_LINE, alive_source
from dtouch.shell import AUTO_RELEASE_KEYS

from test_physarum_ink import VIOLET, _booted, _field, _ink, _scene  # noqa: F401
from dtouch.modes.physarum import _lut_rows

PRE = os.path.join(os.path.dirname(__file__), "fixtures", "alive", "ink_pre_style.frag")
T = ALIVE["fold"]


# ---------- fold_side: the source follows the performer ----------

def test_fold_starts_left_top_up_and_unmoved():
    assert FOLD_START == (-1.0, -1.0, -math.pi / 2, None)
    assert MIRRORS == (0, 2, 4, 6)


def test_fold_side_moves_only_past_the_hysteresis():
    s = FOLD_START
    # inside the dead band either side of the centre lines: the sides stay
    # (the 6-fold wedge has its own rule, below)
    assert fold_side(s, 0.5, 0.5, 16 / 9, 10.0) is s
    for cx, cy in ((0.5 + T["hyst"] * 0.9, 0.5 - T["hyst"] * 0.9),
                   (0.5 - T["hyst"] * 0.9, 0.5 + T["hyst"] * 0.9)):
        assert fold_side(s, cx, cy, 16 / 9, 10.0)[:2] == s[:2]
    # well right and below: both axes flip, stamped with the time
    s2 = fold_side(s, 0.9, 0.9, 16 / 9, 10.0)
    assert s2[:2] == (1.0, 1.0) and s2[3] == 10.0


def test_fold_side_holds_after_a_move():
    s = fold_side(FOLD_START, 0.9, 0.5, 16 / 9, 10.0)
    assert s[0] == 1.0
    # back to the left inside the hold: kept, so a side change is one cut
    assert fold_side(s, 0.1, 0.5, 16 / 9, 10.0 + T["hold"] * 0.9) is s
    s3 = fold_side(s, 0.1, 0.5, 16 / 9, 10.0 + T["hold"] * 1.1)
    assert s3[0] == -1.0


def test_fold_side_never_flips_faster_than_the_hold():
    """A performer standing on the centre line and jittering across it
    cannot strobe the fold: over 60 s of 60 fps jitter it moves at most once
    per hold (the flash limit is 3 a second; this is one per 2 s at most)."""
    rng = np.random.default_rng(1)
    s, moves = FOLD_START, []
    for i in range(3600):
        t = i / 60.0
        cx = 0.5 + rng.uniform(-0.3, 0.3)
        n = fold_side(s, cx, 0.5, 16 / 9, t)
        if n is not s:
            moves.append(t)
        s = n
    assert moves, "jitter that far past the band must move it sometimes"
    gaps = np.diff(moves)
    assert len(gaps) == 0 or gaps.min() >= T["hold"]


def test_fold_six_turns_toward_the_performer_in_sixty_degree_steps():
    s = FOLD_START                                   # wedge up (-pi/2)
    # straight below the centre: the wedge turns to point down (+pi/2)
    s2 = fold_side(s, 0.5, 0.95, 1.0, 5.0)
    assert math.isclose(s2[2], math.pi / 2, abs_tol=1e-9) or \
        math.isclose(s2[2], -math.pi / 2 + math.pi, abs_tol=1e-9)
    # a step is always a whole number of 60 degree turns from the start
    k = (s2[2] - s[2]) / (math.pi / 3)
    assert math.isclose(k, round(k), abs_tol=1e-9)
    # near the centre (inside `reach`), no turn
    s3 = fold_side(s, 0.5, 0.5 + T["reach"] * 0.5, 1.0, 5.0)
    assert s3[2] == s[2]
    # inside the current wedge plus its margin: no turn either
    a = s[2] + (math.pi / 6) * 0.9
    s4 = fold_side(s, 0.5 + 0.4 * math.cos(a), 0.5 + 0.4 * math.sin(a), 1.0, 5.0)
    assert s4[2] == s[2]


def test_fold_angles_stay_wrapped():
    s = FOLD_START
    t = 0.0
    for a in np.linspace(0, 6 * math.pi, 40):
        t += T["hold"] + 0.1
        s = fold_side(s, 0.5 + 0.45 * math.cos(a), 0.5 + 0.45 * math.sin(a), 1.0, t)
        assert -math.pi <= s[2] < math.pi


# ---------- the shader: off is the shipped ink ----------

def _swap_ink(f, frag_src):
    """Point the field's ink program at another fragment source."""
    A = f._alive
    ctx = f.ctx
    with ctx:
        p = ctx.program(vertex_shader=alive_source("quad.vert"), fragment_shader=frag_src)
        A.prog["ink"].release()
        A.vao["ink"].release()
        A.prog["ink"], A.vao["ink"] = p, ctx.vertex_array(p, [])


def _frame(f, rows):
    return f.ink_frame(rows, (VIOLET + 0.5) / len(PALETTES_PH))


def _draw(f, rows):
    """ink.frag alone over the field's current state (no sim, stats or
    compose pass in between), read back raw."""
    A = f._alive
    with f.ctx:
        A.ink_draw(rows, (VIOLET + 0.5) / len(PALETTES_PH))
        return np.frombuffer(A.ink_out[0].read(), np.uint8).copy()


def test_off_is_the_shipped_ink_pixel_for_pixel():
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        rows = _lut_rows(tuple(PALETTES_PH))
        _ink(f, frames=40)
        _frame(f, rows)                       # fields, compose, mips: in place
        a, a2 = _draw(f, rows), _draw(f, rows)
        assert np.array_equal(a, a2), "the pass redrawn is the same picture"
        with open(PRE, encoding="utf-8") as fh:
            pre = fh.read()
        with open(os.path.join(os.path.dirname(alive_source.__code__.co_filename),
                               "shaders", "alive", "video.glsl"), encoding="utf-8") as fh:
            video = fh.read()
        _swap_ink(f, GLSL_VERSION_LINE + "#line 1\n" + video + "#line 1\n" + pre)
        assert np.array_equal(_draw(f, rows), a)
        # and the options really change the picture (the test is not blind)
        _swap_ink(f, alive_source("video.glsl", "ink.frag"))
        assert np.array_equal(_draw(f, rows), a)
        f.ink_style = {"paper": True, "fold": 0}
        assert not np.array_equal(_draw(f, rows), a)
    finally:
        f.release()


def _paperish(img):
    """Share of light, near-neutral pixels: paper, not the molten glow."""
    return float(((img.min(2) > 150) & ((img.max(2) - img.min(2)) < 30)).mean())


def test_paper_is_light_and_the_folds_are_mirrors():
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        rows = _lut_rows(tuple(PALETTES_PH))
        _ink(f, frames=40)
        black = _frame(f, rows)
        f.ink_style = {"paper": True, "fold": 0}
        paper = _frame(f, rows)
        # white paper between black ink (this test field is dense, so ink
        # covers most of it): many times the light neutral share of the
        # black ground (measured 0.06 -> 0.34), and real ink on it
        assert _paperish(paper) > 0.2 and _paperish(paper) > 3 * _paperish(black)
        assert (paper.max(2) < 30).mean() > 0.2
        for fold, side in ((2, (-1.0, -1.0)), (2, (1.0, -1.0)), (4, (-1.0, -1.0)), (4, (1.0, 1.0))):
            f.ink_style = {"paper": False, "fold": fold, "side": side, "angle": -math.pi / 2}
            img = _frame(f, rows).astype(np.int16)
            # a mirror across the vertical centre line (and the horizontal
            # one for 4), within a level of the bilinear upscale
            assert np.abs(img - img[:, ::-1]).mean() < 4.0, (fold, side)
            if fold == 4:
                assert np.abs(img - img[::-1, :]).mean() < 4.0, (fold, side)
        # the side picks which half is the source: left and right differ
        f.ink_style = {"fold": 2, "side": (-1.0, -1.0)}
        left = _frame(f, rows).astype(np.int16)
        f.ink_style = {"fold": 2, "side": (1.0, -1.0)}
        right = _frame(f, rows).astype(np.int16)
        assert np.abs(left - right).mean() > 2.0
        # 6: a kaleidoscope about the centre. Its mirror lines are the
        # wedges' edges, every 60 degrees from the source wedge's centre +-
        # 30; for the wedge pointing up that includes the horizontal line
        # through the centre, so top and bottom mirror (and it is not the
        # 2-fold's left/right mirror)
        f.ink_style = {"fold": 6, "side": (-1.0, -1.0), "angle": -math.pi / 2}
        six = _frame(f, rows).astype(np.int16)
        assert np.abs(six - six[::-1, :]).mean() < 4.0
        assert np.abs(six - six[:, ::-1]).mean() > 4.0
        assert f.alive_error is None
    finally:
        f.release()


# ---------- the mode: keys, the look, panic, the landing ----------

def test_k_and_y_are_free_and_leave_auto_running(tmp_path):
    host, _ = _booted(tmp_path)
    m = host.mode
    try:
        # wiring registers every key (the registry refuses a second binding,
        # so a clash with a shell or mode key would raise right here)
        host._wire_keys()
        assert host.reg._by_key[ord("k")].name == "physarum.ink_paper"
        assert host.reg._by_key[ord("y")].name == "physarum.ink_fold"
        for k in ("k", "y"):
            assert ord(k) not in AUTO_RELEASE_KEYS
        assert not m.ink_paper and m.ink_fold == 0
        host.reg.dispatch(ord("k"))
        assert m.ink_paper
        seen = []
        for _ in range(4):
            host.reg.dispatch(ord("y"))
            seen.append(m.ink_fold)
        assert seen == [2, 4, 6, 0]
    finally:
        m.really_stop()


def test_inkblot_is_a_look_on_the_fractal_step_with_paper_fold_and_veil(tmp_path):
    host, _ = _booted(tmp_path)
    ui, m = host.ui, host.mode
    try:
        assert INK_LOOKS == ("inkblot",)
        names = list(PhysarumMode.BUILTIN)
        assert names.index("inkblot") == 5 and names[0] == "veinwork"   # key 6, never the landing
        cfg = PhysarumMode.BUILTIN["inkblot"]
        ui.ph_video_bg = False
        assert host._apply_look("inkblot", cfg)
        assert m.ink_paper and m.ink_fold == 4 and ui.ph_video_bg
        assert ui.ph_fractal == LOOK_FRACTAL["inkblot"] > 0.0
        # every other look turns the options off and leaves the video bg alone
        assert host._apply_look("amoeba", PhysarumMode.BUILTIN["amoeba"])
        assert not m.ink_paper and m.ink_fold == 0 and ui.ph_video_bg
        # options set by key, then panic: off again, and the landing as shipped
        m.ink_paper, m.ink_fold = True, 6
        host._wire_keys()
        host.reg.dispatch(ord("0"))
        host._apply_pending_preset()
        assert not m.ink_paper and m.ink_fold == 0
        if m.engine == "gl":
            assert ui.ph_palette_idx == VIOLET and ui.ph_fractal == 1.0
    finally:
        m.really_stop()


def test_boot_never_sets_the_options(tmp_path):
    host, _ = _booted(tmp_path, frames=6)
    m = host.mode
    try:
        assert not m.ink_paper and m.ink_fold == 0
        if m.engine == "gl":
            assert m.pf.ink_style is None
    finally:
        m.really_stop()


def test_the_fold_follows_the_performer_in_the_running_mode(tmp_path):
    host, _ = _booted(tmp_path, frames=4)
    m = host.mode
    if m.engine != "gl":
        m.really_stop()
        pytest.skip("no GL context available (CI)")
    try:
        m.ink_fold = 2
        m._fold = FOLD_START
        # a synthetic frame lit only on the right: the side moves right
        frame = np.zeros((108, 192, 3), np.uint8)
        frame[:, 150:] = 255
        for _ in range(6):
            m.step(frame, None, 1 / 30)
        assert m._fold[0] == 1.0
        assert m.pf.ink_style["fold"] == 2 and m.pf.ink_style["side"][0] == 1.0
    finally:
        m.really_stop()


def test_the_autopilot_never_recasts_onto_an_ink_look(tmp_path):
    """Paper turns the whole frame white: a gesture the performer makes, not
    one the autopilot springs on a room (the browser's AUTO_LOOKS)."""
    host, _ = _booted(tmp_path)
    m = host.mode
    try:
        assert PhysarumMode.AUTO_SKIP == INK_LOOKS
        host.auto.on = True
        posted = []
        for _ in range(4000):
            host.ui.pending_preset = None
            host._auto_tick(0.5)
            if host.ui.pending_preset:
                posted.append(host.ui.pending_preset)
        assert len(set(posted)) >= 3, posted[:10]      # it really re-cast
        assert "inkblot" not in posted
    finally:
        m.really_stop()
