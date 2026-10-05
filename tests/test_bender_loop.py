"""Circuit Bender's loop on the desktop: what keeps the main thread cheap
(cached COPY maps and Huffman tables, the bend in its own process, a governor
that watches the display) and how the mode plays with sound and AUTO."""
import numpy as np
import pytest

from dtouch import bender as B
from dtouch import bender_jpeg as J
from dtouch import bender_post as P
from dtouch.bender_post import post
from dtouch.modes import bender as M
from dtouch.modes.bender import BenderMode
from dtouch.shell import AUTO_RELEASE_KEYS

from test_bender import _booted, _fixture, _scene


# ---------- the main thread stays cheap ----------

def test_copy_lookup_is_cached_across_frames(monkeypatch):
    calls = []
    real = P._copy_src
    monkeypatch.setattr(P, "_copy_src", lambda *a: calls.append(a) or real(*a))
    P._COPY_CACHE.clear()
    px = _scene(64, 96)
    a = post(px, 4, 0.5, seed=7.5)
    b = post(px, 4, 0.5, seed=7.5)
    assert np.array_equal(a, b) and len(calls) == 1
    post(px, 4, 0.5, seed=8.5)                       # a new seed is a new map
    assert len(calls) == 2


def test_cached_post_matches_the_uncached_rule():
    px = _scene(70, 101, seed=4)
    xs = np.arange(101)
    for split, copy, seed in ((0, 1.0, 3.5), (4, 0.5, 9.5), (10, 0.0, 1.5)):
        P._COPY_CACHE.clear()
        got = post(px, split, copy, seed=seed)
        if copy > 0:
            sy, sx = P._copy_src(70, 101, copy, seed, 16)
            ref = px[np.clip(sy, 0, 69), np.clip(sx, 0, 100)]
        else:
            ref = px
        want = ref.copy()
        if split:
            want[..., 0] = ref[:, np.clip(xs + split, 0, 100), 0]
            want[..., 2] = ref[:, np.clip(xs - split, 0, 100), 2]
        assert np.array_equal(got, want)


def test_huffman_luts_are_built_once_per_table(monkeypatch):
    calls = []
    real = J._build_huff_lut
    monkeypatch.setattr(J, "_build_huff_lut", lambda key: calls.append(key) or real(key))
    J._HUFF_CACHE.clear()
    data = _fixture("fixture-cv2.jpg")
    J.mcu_offsets(data)
    n = len(calls)
    J.mcu_offsets(data)
    J.scan_swap(data, 0.8, J.rng32(2), 0.1)
    assert n > 0 and len(calls) == n


def test_governor_steps_down_on_a_slow_display_too():
    g = B.Governor()
    for _ in range(4):
        g.step(1.0, 30, display=10)                  # bends fine, display not
    assert g.steps == 1
    ok = B.Governor()
    for _ in range(4):
        ok.step(1.0, 30, display=60)
    assert ok.steps == 0


def test_bends_run_in_a_process_by_default():
    from concurrent.futures import ProcessPoolExecutor
    assert M.POOL == "process"
    pool = M._make_pool()
    try:
        assert isinstance(pool, ProcessPoolExecutor)
        r = pool.submit(M.bend_frame, _scene(), "swap", 0.6, 3, 0.2).result(timeout=60)
        assert r["status"] == "ok"
    finally:
        pool.shutdown(wait=True)


# ---------- sound, rate and AUTO ----------

def test_onset_comes_from_rising_bass():
    o = B.Onset()
    assert not any(o.step(0.2, 1 / 30) for _ in range(30))   # steady: none
    assert o.step(0.9, 1 / 30)                                # a kick
    assert not o.step(0.9, 1 / 30)                            # held: once


def test_auto_release_rule_is_per_mode():
    """j switches modes (global, like p/d/o); the bend keys interrupt AUTO
    only inside Circuit Bender, where they change the scene."""
    assert ord("j") in AUTO_RELEASE_KEYS
    assert ord("e") not in AUTO_RELEASE_KEYS
    assert set(BenderMode.release_keys) == set(M.COMMAND_KEYS.values())


def test_shell_honours_the_modes_release_keys(tmp_path):
    host = _booted(tmp_path)
    host.mode.start(host)
    host._wire_keys()
    host.auto.on = True
    host._route_key(ord("e"))
    assert host.auto.on is False
    host.mode.stop()


def test_bend_rate_limit_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "POOL", "thread")
    monkeypatch.setattr(M, "bend_frame", lambda *a, **k: dict(status="decode"))
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    t, submits = 100.0, 0
    for _ in range(300):                             # 3 s at 100 Hz
        before = m.last_submit
        m._kick(frame, t)
        if m.last_submit != before:
            submits += 1
            m.inflight = None                        # pretend it finished
        t += 0.01
    assert submits <= M.MAX_BEND_HZ * 3 + 1
    m.stop()


def test_auto_recasts_amount_and_post_on_each_look(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    host.auto.on = True
    seen = set()
    for _ in range(12):
        host._apply_look("dht remap", BenderMode.BUILTIN["dht remap"])
        assert m.effect() == "remap"
        seen.add(round(m.amount(), 2))
    assert len(seen) > 3 and all(0.3 <= a <= 0.9 for a in seen)
    host.auto.on = False
    host._apply_look("dht remap", BenderMode.BUILTIN["dht remap"])
    assert m.amount() == pytest.approx(0.5)


def test_auto_sways_the_live_amount(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    assert m.live_amount() == pytest.approx(m.amount())
    host.auto.on = True
    vals = set()
    for t in np.linspace(0, B.AUTO_SWAY_S, 9):
        m.t = float(t)
        vals.add(round(m.live_amount(), 3))
    assert len(vals) > 3
