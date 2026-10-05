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


def test_governor_steps_down_when_the_step_costs_too_much():
    g = B.Governor()
    for _ in range(4):
        g.step(1.0, 30, step_ms=40)                  # bends fine, main thread not
    assert g.steps == 1
    ok = B.Governor()
    for _ in range(4):
        ok.step(1.0, 30, step_ms=3)
    assert ok.steps == 0


def test_a_slow_camera_does_not_shrink_the_picture(tmp_path, monkeypatch):
    """A dim-light camera delivers 8 frames a second: the loop waits on it,
    the step itself stays cheap, and the working size must hold."""
    monkeypatch.setattr(M, "POOL", "thread")
    monkeypatch.setattr(M, "bend_frame",
                        lambda rgb, *a, **k: dict(status="ok", rgb=rgb, bend_ms=20.0))
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    import time
    t_end = time.time() + 0.1
    for _ in range(80):                              # 10 s of an 8 fps camera
        m.step(frame, None, 1 / 8)
        while m.inflight is not None and time.time() < t_end + 5:
            m._settle(time.perf_counter())
            if m.inflight is not None:
                time.sleep(0.001)
    assert m.governor.steps == 0
    m.stop()


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


def test_a_slow_bend_does_shrink_the_picture(tmp_path, monkeypatch):
    """The other side: a worker that needs 120 ms a bend cannot reach
    MIN_RATE at this size, whatever the camera does."""
    monkeypatch.setattr(M, "POOL", "thread")
    monkeypatch.setattr(M, "bend_frame",
                        lambda rgb, *a, **k: dict(status="ok", rgb=rgb, bend_ms=120.0))
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    import time
    for _ in range(300):                             # 10 s at 30 fps
        m.step(frame, None, 1 / 30)
        while m.inflight is not None:
            m._settle(time.perf_counter())
    assert m.governor.steps >= 1
    m.stop()


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
    for key in "kb":                                 # not in the global set
        assert ord(key) not in AUTO_RELEASE_KEYS
        host.auto.on = True
        host._route_key(ord(key))
        assert host.auto.on is False, key
    host.mode.stop()


def test_a_bend_key_outside_bender_leaves_auto_on(tmp_path):
    from dtouch.modes.dithergirl import DitherGirlMode
    from dtouch.shell import Host
    from test_dithergirl import SyntheticSource, _paths
    host = Host(DitherGirlMode(), source=SyntheticSource(), res=(192, 108), show=False,
                preset=None, max_frames=1, **_paths(tmp_path))
    host.run()
    host.auto.on = True
    host._route_key(ord("k"))
    assert host.auto.on is True


def _rate(tmp_path, monkeypatch, loop_hz, seconds=3.0):
    monkeypatch.setattr(M, "POOL", "thread")
    monkeypatch.setattr(M, "bend_frame", lambda *a, **k: dict(status="decode"))
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    t, submits = 100.0, 0
    for _ in range(int(seconds * loop_hz)):
        before = m.inflight
        m._kick(frame, t)
        if m.inflight is not before:
            submits += 1
            m.inflight = None                        # pretend it finished
        t += 1.0 / loop_hz
    m.stop()
    return submits / seconds


def test_a_30hz_loop_reaches_the_ceiling(tmp_path, monkeypatch):
    """The ceiling is a deadline, not a minimum gap: on a 30 Hz loop a gap
    check undershoots to about 20 a second."""
    assert _rate(tmp_path, monkeypatch, 30) >= M.MAX_BEND_HZ - 1


def test_a_fast_loop_never_passes_the_ceiling(tmp_path, monkeypatch):
    assert _rate(tmp_path, monkeypatch, 100) <= M.MAX_BEND_HZ + 1


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


# ---------- the worker process ----------

def _children():
    import multiprocessing
    return [p for p in multiprocessing.active_children() if p.is_alive()]


def _wait_bends(m, frame, n, timeout=60):
    import time
    end = time.time() + timeout
    start = m.bends
    while m.bends - start < n and time.time() < end:
        m.step(frame, None, 1 / 30)
        time.sleep(0.01)
    return m.bends - start


def test_a_killed_worker_is_replaced_and_bending_goes_on(tmp_path):
    import os
    import signal
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    assert _wait_bends(m, frame, 2) >= 2
    for pid in list(m._pool._processes):
        os.kill(pid, signal.SIGKILL)
    assert _wait_bends(m, frame, 3) >= 3              # no exception, bends again
    assert m.worker_deaths >= 1
    m.stop()


def test_stop_leaves_no_worker_behind_and_reentry_bends(tmp_path):
    import time
    host = _booted(tmp_path)
    m = host.mode
    frame = _scene(180, 320)
    for _ in range(2):                                # enter, bend, leave, twice
        m.start(host)
        assert _wait_bends(m, frame, 2) >= 2
        m.stop()
        end = time.time() + 5
        while _children() and time.time() < end:
            time.sleep(0.05)
        assert _children() == []


def test_the_worker_exits_when_its_parent_is_killed(tmp_path):
    """A force-quit app must not orphan the bend worker (ppid 1 forever)."""
    import os
    import signal
    import subprocess
    import sys
    import time
    code = (
        "import sys, time\n"
        "from dtouch.modes import bender as M\n"
        "pool = M._make_pool()\n"
        "pool.submit(M._warm).result(timeout=60)\n"
        "print(list(pool._processes)[0], flush=True)\n"
        "time.sleep(60)\n")
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              text=True, env=env)
    try:
        child = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)
        end = time.time() + 10
        alive = True
        while alive and time.time() < end:
            try:
                os.kill(child, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                alive = False
        assert not alive, "the worker outlived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()
