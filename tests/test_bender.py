"""Circuit Bender: the bends, the tables and the mode.

The bends' cross-port contract (byte-identical to the browser's) is held by
tests/test_bender_shared.py through the goldens; this suite holds what each
piece promises on its own: the JPEG bends keep the stream decodable, amount 0
is the identity, the sensor bends model a Bayer sensor, the clocks and
back-off behave, and the mode plugs into the shell headless.
"""
import os
import time

import numpy as np
import pytest

from dtouch import bender as B
from dtouch import bender_jpeg as J
from dtouch import bender_sensor as S
from dtouch.bender_post import fold, post, sort_runs
from dtouch.bender_looks import FIXTURES, UNIT_DIR
from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes import bender as M
from dtouch.modes.bender import BenderMode, bend_frame, cover_crop, decode_jpeg
from dtouch.shell import AUTO_RELEASE_KEYS, Host

from test_dithergirl import SyntheticSource, _paths


def _fixture(name):
    with open(os.path.join(UNIT_DIR, name), "rb") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def jpegs():
    return {n: _fixture(n) for n in FIXTURES}


def _scene(h=96, w=128, seed=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    img = np.stack([120 + 90 * np.sin(xx / 9.0), 110 + 80 * np.cos(yy / 7.0),
                    100 + 60 * np.sin((xx + yy) / 13.0)], -1)
    img += rng.normal(0, 10, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- the JPEG bends ----------

def test_jround_is_javascripts_half_up():
    assert [J._jround(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)] == [1, 2, 3, 0, -1, -2]


def test_rng32_is_seeded_and_in_range():
    a, b = J.rng32(7), J.rng32(7)
    xs = [a() for _ in range(100)]
    assert xs == [b() for _ in range(100)]
    assert all(0 <= x < 1 for x in xs) and len(set(xs)) == 100


@pytest.mark.parametrize("effect", J.EFFECTS)
def test_amount_zero_is_an_identical_copy(jpegs, effect):
    for data in jpegs.values():
        assert J.bend(data, effect, 0, J.rng32(1)) == data


@pytest.mark.parametrize("effect", J.EFFECTS)
def test_every_bend_stays_a_valid_decodable_jpeg(jpegs, effect):
    for name, data in jpegs.items():
        for a in (0.1, 0.5, 1.0):
            for seed in (1, 2, 3):
                out = J.bend(data, effect, a, J.rng32(seed), 0.3)
                assert J.validate_jpeg(out)["ok"], (name, effect, a, seed)
                assert decode_jpeg(out) is not None, (name, effect, a, seed)


def test_bends_never_touch_their_input(jpegs):
    data = bytearray(jpegs["fixture-cv2.jpg"])
    keep = bytes(data)
    for effect in J.EFFECTS:
        J.bend(data, effect, 1.0, J.rng32(4), 0.5)
    assert bytes(data) == keep


def test_parse_rejects_what_it_cannot_walk():
    with pytest.raises(J.JpegError):
        J.parse_jpeg(b"not a jpeg")
    with pytest.raises(J.JpegError):
        J.bend(b"\xff\xd8\xff", "zigzag", 0.5)


def test_dht_remap_only_permutes_symbols(jpegs):
    data = jpegs["fixture-cv2.jpg"]
    out = J.dht_remap(data, 1.0)
    j = J.parse_jpeg(data)
    assert out != data
    for h in j["dht"]:
        a = data[h["symbols"]:h["symbols"] + h["n"]]
        b = out[h["symbols"]:h["symbols"] + h["n"]]
        assert sorted(a) == sorted(b)
        if h["cls"] == 0:
            assert a == b                       # DC tables are left alone
        assert [x for x in b if x & 15 == 0] == [x for x in a if x & 15 == 0]


def test_zigzag_fold_grows_with_the_amount(jpegs):
    data = jpegs["fixture-cv2.jpg"]
    j = J.parse_jpeg(data)

    def moved(a):
        out = J.zigzag_perm(data, a)
        return sum(out[t["at"] + k] != data[t["at"] + k] for t in j["dqt"] for k in range(64))
    counts = [moved(a) for a in (0.1, 0.4, 0.7, 1.0)]
    assert counts == sorted(counts) and counts[0] < counts[-1]
    assert J.fold_span(0) == 0 and J.fold_span(1) == 63


def test_dqt_erosion_keeps_dc_and_erodes_both_ends(jpegs):
    data = jpegs["fixture-cv2.jpg"]
    out = J.dqt_erosion(data, 1.0)
    for t in J.parse_jpeg(data)["dqt"]:
        assert out[t["at"]] == data[t["at"]]
        ac = list(out[t["at"] + 1:t["at"] + 64])
        assert ac.count(1) >= 16
        assert max(ac) > max(data[t["at"] + 1:t["at"] + 64])


def test_chroma_amp_leaves_the_luma_table_alone(jpegs):
    data = jpegs["fixture-cv2.jpg"]                      # separate chroma table
    j = J.parse_jpeg(data)
    luma = j["sof"]["comps"][0]["tq"]
    out = J.chroma_amp(data, 1.0)
    for t in j["dqt"]:
        a, b = data[t["at"]:t["at"] + 64], out[t["at"]:t["at"] + 64]
        assert (a == b) == (t["id"] == luma)


def test_scan_swap_keeps_the_header_and_the_mcu_count(jpegs):
    data = jpegs["fixture-cv2.jpg"]
    j = J.parse_jpeg(data)
    out = J.scan_swap(data, 1.0, J.rng32(3), 0.2)
    assert out[:j["scan"]["start"]] == data[:j["scan"]["start"]]
    walk = J.mcu_offsets(out)
    assert walk["complete"] and walk["mcus"] == J.mcu_offsets(data)["mcus"]


def test_scan_smear_never_grows_and_repairs_stuffing(jpegs):
    data = jpegs["fixture-cv2.jpg"]
    out = J.scan_smear(data, 1.0, J.rng32(9), 0.4)
    assert len(out) == len(data) and out != data
    assert J.validate_jpeg(out)["ok"]


def test_repair_stuffing_turns_unstuffed_ff_into_fe():
    b = bytearray(b"\x01\xff\x00\xff\x12\xff")
    J.repair_stuffing(b, 0, len(b))
    assert bytes(b) == b"\x01\xff\x00\xfe\x12\xfe"


def test_stack_cuts_the_scan_first(jpegs):
    """STACK's swap must see the encoder's own tables, so it always walks
    clean: its first hand equals SCAN SWAP alone at the stacked weight."""
    data = jpegs["fixture-ffmpeg.jpg"]
    assert J.STACK_WEIGHTS[0][0] == "swap"
    first = J.scan_swap(data, J.STACK_WEIGHTS[0][1], J.rng32(1), 0.2)
    assert J.mcu_offsets(first)["complete"]


# ---------- the sensor bends ----------

def test_sensor_amount_zero_returns_the_pixels_themselves():
    px = _scene()
    for e in S.SENSOR_EFFECTS:
        assert S.sensor_bend(px, e, 0, 3, 1.0) is px
    assert S.sensor_bend(px, "nope", 1, 3, 1.0) is px


def test_mosaic_demosaic_round_trips_a_flat_colour():
    px = np.zeros((16, 20, 3), np.uint8)
    px[...] = (200, 120, 40)
    assert np.array_equal(S.demosaic(S.mosaic(px)), px)


@pytest.mark.parametrize("effect", S.SENSOR_EFFECTS)
def test_sensor_bends_change_the_picture_and_replay(effect):
    px = _scene()
    a = S.sensor_bend(px, effect, 0.8, 11, 2.0)
    b = S.sensor_bend(px, effect, 0.8, 11, 2.0)
    c = S.sensor_bend(px, effect, 0.8, 12, 2.0)
    assert a.shape == px.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.abs(a.astype(int) - px).mean() > 2


def test_bent_cam_casts_the_whole_frame_pink_or_green():
    """The cast is the first thing a bent camera shows: green minus the mean
    of red and blue moves well off the source's, one way or the other."""
    px = np.full((64, 96, 3), 128, np.uint8)

    def tilt(img):
        f = img.astype(float)
        return f[..., 1].mean() - (f[..., 0].mean() + f[..., 2].mean()) / 2
    for seed in (1, 2, 3, 4):
        out = S.sensor_bend(px, "bent", 0.65, seed, 0.0)
        assert abs(tilt(out)) > 8


def test_h_clock_moves_whole_lines():
    raw = np.tile(np.arange(64, dtype=np.uint8), (40, 1))
    out = S.h_clock(raw, 1.0, 5, 0.0)
    moved = [not np.array_equal(out[y], raw[y]) for y in range(40)]
    assert any(moved)


def _hue_jumps(row):
    """How many times the dominant channel changes along a row of RGB."""
    dom = row.astype(int).argmax(-1)
    return int((dom[1:] != dom[:-1]).sum())


def _ring_count(row):
    """Colour changes along a row: a hue jump or a big step in value."""
    r = row.astype(int)
    return int((np.abs(np.diff(r, axis=0)).sum(-1) > 90).sum())


def test_thermal_turns_a_smooth_ramp_into_many_thin_rings():
    """A smooth grey ramp, the sky round every light in the footage, comes
    back as many thin false-colour rings, denser toward the brights."""
    ramp = np.tile(np.linspace(0, 255, 512).astype(np.uint8)[None, :, None], (8, 1, 3))
    out = S.sensor_bend(ramp, "thermal", 0.6, 7, 3.0)
    row = out[4]
    assert _hue_jumps(ramp[4]) <= 2
    assert _ring_count(row) >= 10
    assert _ring_count(row[384:]) > _ring_count(row[128:256])


def test_thermal_keeps_the_shadows_near_black_and_colours_the_brights():
    px = np.zeros((64, 128, 3), np.uint8)
    px[:, :64] = 18                       # shadow
    px[:, 64:] = 200                      # light
    out = S.sensor_bend(px, "thermal", 0.6, 5, 1.0).astype(int)
    shadow, light = out[4:-4, 4:60], out[4:-4, 68:-4]
    assert np.median(shadow.max(-1)) < 24
    sat = light.max(-1) - light.min(-1)
    assert np.median(sat) > 100


def test_thermal_grain_boils_every_frame_but_the_rings_hold():
    ramp = np.tile(np.linspace(60, 255, 128).astype(np.uint8)[None, :, None], (64, 1, 3))
    a = S.sensor_bend(ramp, "thermal", 0.6, 7, 3.00)
    b = S.sensor_bend(ramp, "thermal", 0.6, 7, 3.05)
    moved = (np.abs(a.astype(int) - b).sum(-1) > 0).mean()
    assert 0.01 < moved < 0.3             # the grain moved, the rings did not


def test_thermal_rims_a_hard_edge():
    px = np.full((32, 64, 3), 10, np.uint8)
    px[:, 32:] = 220
    out = S.sensor_bend(px, "thermal", 0.6, 2, 0.0).astype(int)
    rim = out[8:24, 31:33].sum(-1).mean()
    inside = np.median(out[8:24, 44:56].sum(-1))
    assert rim > inside + 60


def test_the_iconic_bends_come_first():
    assert B.EFFECTS[:2] == ("bent", "thermal")
    assert B.LOOK_NAMES[:2] == ("bent", "thermal")
    assert "streak" not in B.EFFECTS


# ---------- sort and post ----------

def _luma(px):
    p = px.astype(np.int32)
    return (306 * p[..., 0] + 601 * p[..., 1] + 117 * p[..., 2]) >> 10


@pytest.mark.parametrize("direction", ["rows", "columns"])
def test_sort_orders_runs_inside_the_band_only(direction):
    px = _scene(40, 50, seed=2)
    before = px.copy()
    n = sort_runs(px, 60, 205, direction)
    assert n > 0
    y0, y1 = _luma(before), _luma(px)
    out_band = (y0 < 60) | (y0 > 205)
    assert np.array_equal(px[out_band], before[out_band])
    lines = y1 if direction == "rows" else y1.T
    band = (y0 >= 60) & (y0 <= 205)
    band = band if direction == "rows" else band.T
    for li in range(lines.shape[0]):
        run = []
        for v, inb in zip(lines[li], band[li]):
            if inb:
                run.append(v)
            else:
                assert run == sorted(run)
                run = []
        assert run == sorted(run)


def test_post_off_returns_the_picture_itself():
    px = _scene()
    assert post(px, 0, 0) is px


def test_split_pulls_red_and_blue_apart():
    px = np.zeros((8, 32, 3), np.uint8)
    px[:, 16] = 255
    out = post(px, 4, 0)
    assert out[0, 12, 0] == 255 and out[0, 20, 2] == 255 and out[0, 16, 1] == 255


def test_fold_is_a_running_mean():
    a = np.zeros((2, 2, 3), np.uint8)
    b = np.full((2, 2, 3), 100, np.uint8)
    s = fold(None, a, 0.3)
    s = fold(s, b, 0.25)
    assert np.allclose(s, 25)


# ---------- the tables and clocks ----------

def test_looks_cover_every_effect_in_order():
    assert [v["effect"] for v in B.LOOKS.values()] == list(B.EFFECTS)
    assert B.SAFE_LOOK == B.EFFECTS[0] == "bent"
    assert set(B.SETTLE_S) == set(B.EFFECTS)


def test_next_in_wraps():
    assert B.next_in(B.AMOUNT_LADDER, 0.5) == 0.75
    assert B.next_in(B.AMOUNT_LADDER, 1) == 0.25


def test_working_size_follows_output_source_and_budget():
    w, h, up = B.working_size(640, 360)
    assert (w, h, up) == (640, 360, 1.0)
    w, h, _ = B.working_size(1920, 1080)
    assert w * h <= B.MAX_PIXELS and abs(w / h - 16 / 9) < 0.02
    w, h, _ = B.working_size(1920, 1080, 640, 480)       # the camera caps it
    assert w <= 640
    w, h, _ = B.working_size(1920, 1080, budget=B.MIN_PIXELS)
    assert w * h <= B.MIN_PIXELS * 1.01


def test_governor_only_steps_down():
    g = B.Governor()
    assert not g.step(1.0, 30)
    for _ in range(3):
        g.step(1.0, 5)
    assert g.steps == 1 and g.budget < B.MAX_PIXELS
    b = g.budget
    for _ in range(10):
        g.step(1.0, 60)
    assert g.budget == b


def test_cut_clock_ticks_and_swap_runs_three_times_the_cuts():
    calm, fast = B.CutClock(1), B.CutClock(1)
    for _ in range(int(B.CUT_EVERY_S * 30) + 1):
        calm.step(1 / 30)
        fast.step(1 / 30, tempo=B.cut_tempo("swap"))
    assert calm.cuts == 1 and fast.cuts == 3
    assert 0 < calm.phase < 1


def test_onset_cuts_at_once_but_not_twice_in_a_gap():
    c = B.CutClock(1)
    assert c.step(0.01, onset=True)
    assert not c.step(0.01, onset=True)


def test_backoff_sends_a_clean_frame_after_three_fails():
    bo = B.Backoff()
    for _ in range(B.CLEAN_AFTER):
        assert bo.amount(1.0) > 0
        bo.fail()
    assert bo.amount(1.0) == 0
    bo.ok(clean=True)
    assert bo.amount(1.0) > 0 and bo.cleans == 1


def test_dead_frame_catches_black_and_flat():
    src = {"mean": 120, "variance": 400}
    assert B.dead_frame({"mean": 2, "variance": 0}, src)
    assert B.dead_frame({"mean": 90, "variance": 1}, src)
    assert not B.dead_frame({"mean": 90, "variance": 300}, src)


def test_live_amount_stays_in_range():
    for base in (0, 0.5, 1):
        for bass in (0, 1):
            for t in (0, 3.5, 7):
                v = B.live_amount(base, bass=bass, sens=2, auto=True, t=t)
                assert 0 <= v <= 1


def test_recast_pick_is_a_valid_state():
    rng = np.random.default_rng(1)
    for name in list(B.LOOKS) + ["nope"]:
        s = B.recast_pick(name, rng.random)
        assert s["look"] in B.LOOKS and s["effect"] in B.EFFECTS
        assert 0.3 <= s["amount"] <= 0.9
        assert s["split"] in B.SPLIT_LADDER and s["sort"] in B.SORT_LADDER


# ---------- one bend, the worker's job ----------

@pytest.mark.parametrize("effect", B.EFFECTS)
def test_bend_frame_bends_every_effect(effect):
    px = _scene()
    r = bend_frame(px, effect, 0.7, 5, 0.2, 1.0)
    assert r["status"] == "ok"
    assert r["rgb"].shape == px.shape
    assert np.abs(r["rgb"].astype(int) - px).mean() > 1


def test_bend_frame_reports_a_dead_frame(monkeypatch):
    monkeypatch.setattr(M, "decode_jpeg", lambda data: np.zeros((96, 128, 3), np.uint8))
    assert bend_frame(_scene(), "zigzag", 0.5, 1)["status"] == "dead"


def test_bend_frame_reports_an_undecodable_frame(monkeypatch):
    monkeypatch.setattr(M, "decode_jpeg", lambda data: None)
    assert bend_frame(_scene(), "zigzag", 0.5, 1)["status"] == "decode"


def test_bend_frame_sorts_when_asked():
    px = _scene()
    a = bend_frame(px, "zigzag", 0.5, 1)["rgb"]
    b = bend_frame(px, "zigzag", 0.5, 1, sort="rows")["rgb"]
    assert not np.array_equal(a, b)


def test_cover_crop_keeps_the_centre():
    f = np.zeros((100, 200, 3), np.uint8)
    assert cover_crop(f, 1.0).shape[:2] == (100, 100)
    assert cover_crop(f, 4.0).shape[:2] == (50, 200)


# ---------- the mode ----------

RES = (320, 180)


def _booted(tmp_path):
    host = Host(BenderMode(), source=SyntheticSource(), res=RES, show=False,
                preset=None, max_frames=1, **_paths(tmp_path))
    host.run()
    return host


def test_mode_protocol_attrs():
    m = BenderMode()
    assert m.id == "bender" and m.title == "Circuit Bender" and m.key == "j"
    assert m.accepts_still is False
    assert m.safe_look() == "bent cam" == next(iter(BenderMode.BUILTIN))
    assert set(BenderMode.BUILTIN) == {B.EFFECT_TITLES[e] for e in B.EFFECTS}


def test_registered_with_a_free_key():
    assert BenderMode in REGISTRY and mode_by_id("bender") is BenderMode
    keys = [getattr(m, "key", m.id[:1]) for m in REGISTRY]
    assert len(keys) == len(set(keys))
    assert ord("j") in AUTO_RELEASE_KEYS


def test_distinct_from_the_signal_racks_circuit_bent():
    import dtouch.circuit_bent as cb
    assert not hasattr(cb, "BenderMode")
    assert "circuit_bent" not in M.__dict__


def test_lands_on_bent_cam_and_steps_headless(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    assert host.ui.preset_name == "bent cam" and m.effect() == "bent"
    m.stop()
    m.stop()                                   # idempotent
    m.start(host)
    frame = _scene(180, 320)
    deadline = time.time() + 10
    out = m.step(frame, None, 1 / 30)
    while m.bends < 3 and time.time() < deadline:
        time.sleep(0.02)
        out = m.step(frame, None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    assert m.bends >= 3 and m.size == RES
    assert m.bent is not None
    m.stop()


def test_commands_bind_free_keys_and_step_the_state(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    cmds = m.commands()
    assert {c.key for c in cmds.values()} == {"e", "b", "x", "c", "k", "l"}
    host._wire_keys()                          # no clash with the shell's keys
    before = m.effect()
    cmds["bender.effect"].run()
    assert m.effect() != before
    cmds["bender.amount"].run()
    assert m.amount() in B.AMOUNT_LADDER
    cmds["bender.split"].run()
    assert m.split() == B.SPLIT_LADDER[1]
    m.stop()


def test_every_builtin_look_applies(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    for name, cfg in BenderMode.BUILTIN.items():
        assert host._apply_look(name, cfg)
        assert B.EFFECT_TITLES[m.effect()] == cfg["effect"]
        assert m.amount() == pytest.approx(cfg["amount"])


def test_the_long_title_stays_inside_its_menu_card():
    """CIRCUIT BENDER at the menu's title size ran past both card edges."""
    from dtouch.menu import draw_menu, registry_cards
    for w, h in ((1280, 720), (1920, 1080), (960, 540)):
        img = np.zeros((h, w, 3), np.uint8)
        # the selected card's thick border overhangs its rect: select the
        # first card and skip the gap beside it
        rects = draw_menu(img, None, registry_cards(), 0)
        boxes = sorted(r for r, _ in rects)
        for (x0, y0, x1, y1), (nx0, _, _, _) in list(zip(boxes, boxes[1:]))[1:]:
            gap = img[y0:y1, x1 + 3:nx0 - 2]
            assert not gap.any(), (w, h, x1, nx0)


def test_a_stalled_bend_is_abandoned_not_waited_on(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    gate = __import__("threading").Event()

    def stuck(*a, **kw):
        gate.wait(5)
        return dict(status="decode")
    monkeypatch.setattr(M, "bend_frame", stuck)
    monkeypatch.setattr(M, "STALL_S", 0.05)
    monkeypatch.setattr(M, "POOL", "thread")         # a process cannot see the patch
    m.stop()
    m.start(host)
    frame = _scene(180, 320)
    m.step(frame, None, 1 / 30)
    assert m.inflight is not None
    time.sleep(0.1)
    m.step(frame, None, 1 / 30)
    assert m.stalls == 1
    gate.set()
    m.stop()
