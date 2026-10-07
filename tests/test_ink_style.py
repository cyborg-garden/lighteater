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

from dtouch.alive import (ALIVE, FOLD_START, MIRRORS, fold_side, fold_snap,
                          paper_step, style_ready)
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
    assert fold_side(s, 0.5, 0.5, 16 / 9, 10.0, 4) is s
    for cx, cy in ((0.5 + T["hyst"] * 0.9, 0.5 - T["hyst"] * 0.9),
                   (0.5 - T["hyst"] * 0.9, 0.5 + T["hyst"] * 0.9)):
        assert fold_side(s, cx, cy, 16 / 9, 10.0, 4) is s
    # well right and below: both axes flip, stamped with the time
    s2 = fold_side(s, 0.9, 0.9, 16 / 9, 10.0, 4)
    assert s2[:2] == (1.0, 1.0) and s2[3] == 10.0


def test_only_the_axes_a_fold_reads_move_and_stamp():
    s = FOLD_START
    # fold 2 reads x only: a performer low in the frame or round the side
    # moves neither y nor the wedge, and stamps no hold
    assert fold_side(s, 0.5, 0.95, 1.0, 10.0, 2) is s
    s2 = fold_side(s, 0.9, 0.95, 1.0, 10.0, 2)
    assert s2[0] == 1.0 and s2[1] == s[1] and s2[2] == s[2]
    # fold 6 reads the wedge only
    s3 = fold_side(s, 0.9, 0.5, 1.0, 10.0, 6)
    assert s3[:2] == s[:2] and s3[2] != s[2]


def test_fold_side_holds_after_a_move():
    s = fold_side(FOLD_START, 0.9, 0.5, 16 / 9, 10.0, 2)
    assert s[0] == 1.0
    assert fold_side(s, 0.1, 0.5, 16 / 9, 10.0 + T["hold"] * 0.9, 2) is s
    assert fold_side(s, 0.1, 0.5, 16 / 9, 10.0 + T["hold"] * 1.1, 2)[0] == -1.0


def test_fold_side_never_flips_faster_than_the_hold():
    """60 s of 60 fps centroid jitter across the centre lines: the fold
    moves, but never twice inside the hold (well under the 3 a second
    floor)."""
    rng = np.random.default_rng(1)
    for fold in (2, 4, 6):
        s, moves = FOLD_START, []
        for i in range(3600):
            t = i / 60.0
            n = fold_side(s, 0.5 + rng.uniform(-0.3, 0.3), 0.5 + rng.uniform(-0.3, 0.3),
                          16 / 9, t, fold)
            if n is not s:
                moves.append(t)
            s = n
        assert moves, fold
        assert len(moves) < 2 or np.diff(moves).min() >= T["hold"], fold


def test_fold_six_turns_toward_the_performer_in_sixty_degree_steps():
    s = FOLD_START                                   # wedge up (-pi/2)
    s2 = fold_side(s, 0.5, 0.95, 1.0, 5.0, 6)        # straight below: down
    assert math.isclose(s2[2], math.pi / 2, abs_tol=1e-9)
    k = (s2[2] - s[2]) / (math.pi / 3)
    assert math.isclose(k, round(k), abs_tol=1e-9)
    assert fold_side(s, 0.5, 0.5 + T["reach"] * 0.5, 1.0, 5.0, 6)[2] == s[2]
    a = s[2] + (math.pi / 6) * 0.9
    assert fold_side(s, 0.5 + 0.4 * math.cos(a), 0.5 + 0.4 * math.sin(a), 1.0, 5.0, 6)[2] == s[2]


def test_fold_angles_stay_wrapped():
    s, t = FOLD_START, 0.0
    for a in np.linspace(0, 6 * math.pi, 40):
        t += T["hold"] + 0.1
        s = fold_side(s, 0.5 + 0.45 * math.cos(a), 0.5 + 0.45 * math.sin(a), 1.0, t, 6)
        assert -math.pi <= s[2] < math.pi


def test_a_new_fold_snaps_to_the_performer_and_starts_the_hold():
    # just past the centre (inside the hysteresis band): fold_side would
    # keep the old side, a fresh fold goes straight to the performer's
    s = fold_snap(FOLD_START, 0.53, 0.47, 16 / 9, 3.0, 2)
    assert s[0] == 1.0 and s[3] == 3.0
    assert fold_side(s, 0.1, 0.5, 16 / 9, 3.5, 2) is s        # held
    s4 = fold_snap(FOLD_START, 0.2, 0.8, 16 / 9, 3.0, 4)
    assert s4[:2] == (-1.0, 1.0)
    s6 = fold_snap(FOLD_START, 0.5, 0.9, 1.0, 3.0, 6)
    assert math.isclose(s6[2], math.pi / 2, abs_tol=1e-9)


# ---------- flash safety: the 3 a second floor ----------

def test_the_paper_fades_and_flips_at_most_once_per_cooldown():
    st = ALIVE["style"]
    assert st == {"cooldown": 0.5, "fade": 0.3}
    a, frames = 0.0, 0
    while a < 1.0:
        a = paper_step(a, True, 1 / 60)
        frames += 1
    assert frames == math.ceil(st["fade"] * 60)                # ~300 ms, not a cut
    assert paper_step(1.0, False, 1 / 60) == pytest.approx(1 - 1 / 60 / st["fade"])
    assert style_ready(None, 0.0) and not style_ready(1.0, 1.4) and style_ready(1.0, 1.5)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _transitions(xs):
    return sum(1 for a, b in zip(xs, xs[1:]) if a != b)


def test_held_or_spammed_k_y_and_blackout_stay_under_three_a_second(tmp_path):
    """A held key repeats at 30/s on the desktop (waitKey cannot tell repeats
    from presses): K, Y and, over paper, blackout change at most twice a
    second. Off paper, blackout stays free."""
    host, _ = _booted(tmp_path)
    m = host.mode
    try:
        clock = _Clock()
        m._clock = clock
        host.scene_clock = clock                 # the shell's blackout hold too
        host._wire_keys()
        space = ord(" ")
        for key, read in ((ord("k"), lambda: m.ink_paper),
                          (ord("y"), lambda: m.ink_fold),
                          (space, lambda: host.ps.blackout)):
            if key == space:
                m.ink_paper = True               # blackout over paper
            seen = [read()]
            for _ in range(90):                  # 3 s at 30 presses / s
                clock.t += 1 / 30
                host.reg.dispatch(key)
                seen.append(read())
            n = _transitions(seen)
            # K and Y: one change per 0.5 s. Blackout: ON at once, OFF held
            # 0.5 s, so at most 2 flashes (4 changes) a second
            limit = 4.0 if key == space else 2.0
            assert 3 <= n and n / 3.0 <= limit + 1e-9, (chr(key), n)
        # off paper too: ON is instant, OFF waits out the hold (flash guards)
        m.ink_paper, m._paper_on, m._paper_amt = False, False, 0.0
        host.ps.blackout = False
        clock.t += 1.0
        host.reg.dispatch(space)
        assert host.ps.blackout is True           # ON at once
        for _ in range(10):                       # a third of a second of presses
            clock.t += 1 / 30
            host.reg.dispatch(space)
        assert host.ps.blackout is True           # OFF held
    finally:
        m.really_stop()


def _lum_run(host, frames, act, every):
    """Step the running mode `frames` frames at 60 fps, calling act(i) every
    `every` frames, with the scene clock on the frame clock; return each
    rendered frame's mean luminance (0..1)."""
    m = host.mode
    t = [1000.0]
    host.scene_clock = lambda: t[0]
    m._clock = lambda: t[0]
    frame = np.zeros((108, 192, 3), np.uint8)
    lums = []
    for i in range(frames):
        if i % every == 0:
            act(i // every)
        out = m.step(frame, None, 1 / 60)
        t[0] += 1 / 60
        o = out.astype(np.float32)
        lums.append(float((o[..., 0] * 0.2126 + o[..., 1] * 0.7152 + o[..., 2] * 0.0722).mean()) / 255.0)
    return lums


def _jumps(lums, step=0.1):
    """Frame-to-frame luminance changes over `step` (10% of full range, the
    WCAG flash threshold); two opposing ones make one flash."""
    return sum(1 for a, b in zip(lums, lums[1:]) if abs(b - a) > step)


def test_rapid_look_switching_cannot_strobe_the_rendered_frame(tmp_path):
    """inkblot <-> amoeba applied directly (no key gate: the panel, a
    resume) every 6 and every 12 frames for 3 s: the paper only ever fades,
    the ink holds through a fade-out, so the rendered frame jumps by more
    than 10% at most twice a second (the floor is 3 flashes, 6 such jumps, a
    second). Before the fix the 6-frame run jumped 17 times in 2 s. Faster
    than that no person reaches: every key path is gated to 2 a second
    (the test below)."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    if m.engine != "gl":
        m.really_stop()
        pytest.skip("no GL context available (CI)")
    try:
        names = ("amoeba", "inkblot")
        apply = lambda k: host._apply_look(names[k % 2], PhysarumMode.BUILTIN[names[k % 2]])  # noqa: E731
        host._apply_look("inkblot", PhysarumMode.BUILTIN["inkblot"])
        _lum_run(host, 60, lambda k: None, 1000)           # settle in the ink
        for every in (6, 12):
            lums = _lum_run(host, 180, apply, every)
            assert _jumps(lums) <= 2 * 3, (every, _jumps(lums))
    finally:
        m.really_stop()


def test_held_h_and_look_keys_cannot_strobe_the_rendered_frame(tmp_path):
    """H held (a repeat every 2 frames, 30/s) on veinwork, and 6 / 2 spammed
    through the bank keys: one scene change per 0.5 s, so at most two >10%
    jumps a second. Before, a held H jumped 32 times in 2 s."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    if m.engine != "gl":
        m.really_stop()
        pytest.skip("no GL context available (CI)")
    try:
        host._wire_keys()
        assert host.ui.bank.get("6") == "inkblot" and host.ui.bank.get("2") == "amoeba"
        for start, act in (
                ("veinwork", lambda k: host.reg.dispatch(ord("h"))),
                ("inkblot", lambda k: host.reg.dispatch(ord("h"))),
                ("inkblot", lambda k: host.reg.dispatch(ord("62"[k % 2])))):
            host._apply_look(start, PhysarumMode.BUILTIN[start])
            host._apply_pending_preset()
            _lum_run(host, 40, lambda k: None, 1000)
            lums = []
            for _ in range(3):                             # 3 s, 1 s at a time
                host.ui.pending_preset = None
                chunk = []
                for i in range(60):
                    if i % 2 == 0:
                        act(i // 2)
                        host._apply_pending_preset()
                    chunk += _lum_run(host, 1, lambda k: None, 1000)
                lums += chunk
            assert _jumps(lums) <= 2 * 3, (start, _jumps(lums))
    finally:
        m.really_stop()


def test_scene_keys_take_one_change_per_half_second(tmp_path):
    from dtouch.shell import FREE_COMMANDS, SCENE_COOLDOWN_S
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    try:
        assert SCENE_COOLDOWN_S == 0.5
        assert "physarum.depth" not in FREE_COMMANDS and "preset.recall.6" not in FREE_COMMANDS
        host._wire_keys()
        t = [50.0]
        host.scene_clock = lambda: t[0]
        seen = []
        for _ in range(90):                                # 30 presses / s, 3 s
            t[0] += 1 / 30
            host.reg.dispatch(ord("h"))
            seen.append((host.ui.ph_depth, host.ui.ph_fractal))
        n = _transitions(seen)
        assert 3 <= n <= 6, n
        # panic stays free: the way back is never held
        t[0] += 0.01
        host.reg.dispatch(ord("0"))
        assert host.ui.pending_preset == m.safe_look()
    finally:
        m.really_stop()


def test_resuming_into_inkblot_fades_the_paper_in(tmp_path):
    """A crash-resume (or any boot) straight into inkblot starts the paper
    at 0 and fades it in: no white frame on the first frame."""
    host, _ = _booted(tmp_path, frames=2)
    m = host.mode
    if m.engine != "gl":
        m.really_stop()
        pytest.skip("no GL context available (CI)")
    try:
        assert m._paper_amt == 0.0
        host._apply_look("inkblot", PhysarumMode.BUILTIN["inkblot"])
        frame = np.zeros((108, 192, 3), np.uint8)
        m.step(frame, None, 1 / 60)
        assert 0.0 < m._paper_amt <= (1 / 60) / ALIVE["style"]["fade"] + 1e-9
        for _ in range(30):
            m.step(frame, None, 1 / 60)
        assert m._paper_amt == 1.0
    finally:
        m.really_stop()


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
        w, h = A.ink_out[0].size
        return np.frombuffer(A.ink_out[0].read(), np.uint8).reshape(h, w, 4)[..., :3].copy()


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


def test_the_paper_fade_changes_the_frame_smoothly():
    """Frame by frame at 60 fps, a paper flip moves the picture's mean
    luminance by a small step (no frame jumps by 10% of full range, the
    WCAG flash threshold), and the whole fade does change it a lot."""
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        rows = _lut_rows(tuple(PALETTES_PH))
        _ink(f, frames=40)
        _frame(f, rows)
        lum, a = [], 0.0
        for on in [True] * 30 + [False] * 30:
            a = paper_step(a, on, 1 / 60)
            f.ink_style = {"paper": a, "fold": 0}
            img = _draw(f, rows).astype(np.float32)
            lum.append(float((img[..., 0] * 0.2126 + img[..., 1] * 0.7152
                              + img[..., 2] * 0.0722).mean()) / 255.0)
        steps = np.abs(np.diff(lum))
        assert max(lum) - min(lum) > 0.1, "the fade really moves the picture"
        assert steps.max() < 0.1, steps.max()
    finally:
        f.release()


def test_fold_six_keeps_the_picture_s_own_scale():
    """A pixel inside the 6-fold's source wedge maps to itself (no zoom), so
    the wedge's content is the unfolded picture there."""
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        rows = _lut_rows(tuple(PALETTES_PH))
        _ink(f, frames=40)
        _frame(f, rows)
        f.ink_style = None
        plain = _draw(f, rows)
        f.ink_style = {"fold": 6, "side": (-1.0, -1.0), "angle": -math.pi / 2}
        six = _draw(f, rows)
        h, w = plain.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        # canvas rows run bottom-up; image space is +y down from the top
        u = (xx + 0.5) / w
        v = 1.0 - (yy + 0.5) / h
        px, py = (u - 0.5) * (w / h), v - 0.5
        ang = np.arctan2(py, px)
        inside = (np.abs(ang - (-math.pi / 2)) < (math.pi / 6) * 0.7) & (np.hypot(px, py) > 0.06) \
            & (np.hypot(px, py) < 0.45)
        assert inside.sum() > 200
        d = np.abs(plain.astype(np.int16) - six.astype(np.int16))[inside]
        assert d.mean() < 1.0 and np.percentile(d, 99) <= 6, (d.mean(), np.percentile(d, 99))
        # and outside the wedge it is folded, not the plain picture
        out = ~inside & (np.hypot(px, py) > 0.1)
        assert np.abs(plain.astype(np.int16) - six.astype(np.int16))[out].mean() > 5.0
    finally:
        f.release()


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
        clock = _Clock()
        m._clock = clock
        host.scene_clock = clock          # K and Y take the shell's shared budget
        host.reg.dispatch(ord("k"))
        assert m.ink_paper
        seen = []
        for _ in range(4):
            clock.t += ALIVE["style"]["cooldown"] + 0.01   # presses a person makes
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
