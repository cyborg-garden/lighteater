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


def _until(worker, timeout=60):
    import time
    end = time.time() + timeout
    while time.time() < end:
        got = worker.poll()
        if got is not None:
            return got
        time.sleep(0.01)
    raise AssertionError("no reply")


def test_bends_run_in_a_process_by_default():
    from dtouch.bender_worker import ProcessWorker
    import time
    assert M.POOL == "process"
    w = M.make_worker()
    try:
        assert isinstance(w, ProcessWorker)
        assert not w.submit(M.bend_frame, _scene(), "swap", 0.6, 3, 0.2) or w.ready
        end = time.time() + 60
        while not w.ready and time.time() < end:
            w.poll()
            time.sleep(0.01)
        assert w.submit(M.bend_frame, _scene(), "swap", 0.6, 3, 0.2)
        status, r = _until(w)
        assert status == "ok" and r["status"] == "ok"
    finally:
        w.kill()


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
            while m._worker.poll() is None:          # let it finish at once
                pass
            m.inflight = None
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
    os.kill(m._worker.pid, signal.SIGKILL)
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
        "w = M.make_worker()\n"
        "while not w.ready:\n"
        "    w.poll(); time.sleep(0.01)\n"
        "print(w.pid, flush=True)\n"
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



# ---------- quitting never hangs (audit round 3) ----------

def _run_exits(code, timeout=30):
    """Run `code` in a fresh interpreter; True when it exits by itself in time."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    p = subprocess.Popen([sys.executable, "-c", code], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode == 0, out
    except subprocess.TimeoutExpired:
        p.kill()
        return False, "hung"


MIDWRITE = (
    "import time, numpy as np\n"
    "from dtouch.bender_worker import ProcessWorker\n"
    "w = ProcessWorker()\n"
    "while not w.ready:\n"
    "    w.poll(); time.sleep(0.01)\n"
    "w.submit(np.ones, 30_000_000)\n"      # a 240 MB reply: a long write back
    "time.sleep({delay})\n"
    "w.kill()\n"
    "print('killed', flush=True)\n")


@pytest.mark.parametrize("delay", [0.08, 0.3, 0.6])
def test_a_kill_mid_reply_never_hangs_the_exit(delay):
    """ProcessPoolExecutor hung forever here: its manager thread blocked in
    recv on the half-written reply, and interpreter exit joined it."""
    ok, out = _run_exits(MIDWRITE.format(delay=delay))
    assert ok, out


def test_stopping_during_a_large_bend_exits_promptly():
    code = (
        "import time, pathlib, tempfile\n"
        "import numpy as np\n"
        "import sys; sys.path.insert(0, 'tests')\n"
        "from test_bender import _booted, _scene\n"
        "host = _booted(pathlib.Path(tempfile.mkdtemp()))\n"
        "host.res = (1920, 1080)\n"
        "m = host.mode\n"
        "frame = _scene(1080, 1920)\n"
        "for i in range(6):\n"
        "    m.start(host)\n"
        "    end = time.time() + 0.3 + 0.2 * i\n"
        "    while time.time() < end:\n"
        "        m.step(frame, None, 1 / 30); time.sleep(0.005)\n"
        "    t = time.perf_counter(); m.stop()\n"
        "    assert time.perf_counter() - t < 2, 'slow stop'\n"
        "print('done', flush=True)\n")
    ok, out = _run_exits(code, timeout=90)
    assert ok, out


# ---------- lost workers: rate-limited, backed off, given up ----------

class _DeadWorker:
    """A worker that is ready and then always dead."""
    made = 0

    def __init__(self):
        _DeadWorker.made += 1
        self.ready, self.dead, self.busy, self.pid = True, False, False, None

    def submit(self, *a):
        return True

    def poll(self):
        return "dead", None

    def kill(self):
        self.dead = True


class _SilentWorker(_DeadWorker):
    """Ready, takes the job, never replies (a stall)."""

    def poll(self):
        return None


class _NeverReady(_DeadWorker):
    """Started, never ready, never dead (a hung import)."""

    def __init__(self):
        super().__init__()
        self.ready, self.born = False, 0.0

    def poll(self):
        return None


def test_a_worker_that_never_gets_ready_is_lost(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    monkeypatch.setattr(M, "make_worker", _NeverReady)
    m.start(host)
    frame = _scene(180, 320)
    m._kick(frame, M.STARTUP_S - 1)
    assert m._worker is not None and m.stalls == 0
    m._kick(frame, M.STARTUP_S + 1)
    assert m._worker is None and m.stalls == 1 and m.respawn_at is not None
    m.stop()


def test_lost_workers_back_off_and_then_give_up(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    monkeypatch.setattr(M, "make_worker", _DeadWorker)
    m.start(host)
    hints = []
    monkeypatch.setattr(host.hud.toasts, "hint", lambda t, *a, **k: hints.append(t))
    frame = _scene(180, 320)
    t, gaps, last = 0.0, [], None
    for _ in range(4000):                            # 400 s at 10 Hz
        m._kick(frame, t)
        if m.inflight is not None:
            m._settle(t)
        if m.respawn_at is not None and m.respawn_at != last:
            gaps.append(round(m.respawn_at - t, 3))
            last = m.respawn_at
        t += 0.1
        if m.gave_up:
            break
    assert m.gave_up and m.losses == M.MAX_LOSSES
    assert gaps == [min(M.RESPAWN_MAX_S, M.RESPAWN_GAP_S * 2 ** i)
                    for i in range(M.MAX_LOSSES - 1)]
    assert any("stopped bending" in h for h in hints)
    # the camera shows at the full working size, not a shrunk one
    host.res = (1920, 1080)
    m.governor.budget = B.MIN_PIXELS
    big = _scene(1080, 1920)
    out = m.step(big, None, 1 / 30)
    assert out.shape == (1080, 1920, 3)
    assert m.size == B.working_size(1920, 1080, 1920, 1080, B.MAX_PIXELS)[:2]
    host.res = (320, 180)
    made = _DeadWorker.made
    for _ in range(100):
        m._kick(frame, t)
        t += 0.1
    assert _DeadWorker.made == made                 # no respawn after giving up
    assert m.step(frame, None, 1 / 30).shape == (180, 320, 3)   # camera shows
    m.stop()


def test_a_stall_is_rate_limited_and_counts_as_slow(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    monkeypatch.setattr(M, "make_worker", _SilentWorker)
    m.start(host)
    frame = _scene(180, 320)
    m._kick(frame, 0.0)
    assert m.inflight is not None
    m._settle(M.STALL_S + 0.1)
    assert m.stalls == 1 and m._worker is None
    assert m.respawn_at >= M.STALL_S + 0.1 + M.RESPAWN_GAP_S
    assert m.governor.steps == 1                      # hard evidence: one notch
    assert m.bend_ms == 0                             # and nothing lingers
    m.stop()


def test_a_stall_steps_down_once_and_waits_say_nothing(tmp_path, monkeypatch):
    """Audit round 4: a stale stall ratcheted the size to the floor during
    respawn waits and after giving up. Each stall is one slow sample; the
    waits between workers are silent."""
    import types
    clock = [0.0]
    monkeypatch.setattr(M, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(M, "make_worker", _SilentWorker)
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    calls = []
    real = m.governor.step
    monkeypatch.setattr(m.governor, "step", lambda *a, **k: calls.append(a) or real(*a, **k))
    frame = _scene(180, 320)
    for _ in range(int(120 * 30)):                    # two minutes at 30 fps
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    assert m.gave_up and m.bends == 0
    assert m.stalls == M.MAX_LOSSES
    assert calls == []                                # waits say nothing
    assert m.governor.steps == M.MAX_LOSSES           # one notch per stall, no more
    m.stop()


def test_a_slow_startup_says_nothing_to_the_governor(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    monkeypatch.setattr(M, "make_worker", _NeverReady)
    m.start(host)
    m._kick(_scene(180, 320), M.STARTUP_S + 1)
    assert m.stalls == 1 and m.governor.steps == 0 and m.bend_ms == 0
    m.stop()



# ---------- the worker never blocks the app (audit round 4) ----------

def test_a_frozen_worker_cannot_block_submit():
    """A worker that stops reading (SIGSTOP here) makes submit give up in
    bounded time instead of blocking the display on the pipe."""
    import os
    import signal
    import time
    from dtouch import bender_worker as W
    w = W.ProcessWorker()
    try:
        end = time.time() + 60
        while not w.ready and time.time() < end:
            w.poll()
            time.sleep(0.01)
        os.kill(w.pid, signal.SIGSTOP)
        big = np.zeros((1080, 1920, 3), np.uint8)    # far past a pipe buffer
        t0 = time.perf_counter()
        ok = w.submit(np.copy, big)
        took = time.perf_counter() - t0
        assert not ok and w.dead
        assert took < W.SEND_S + 0.5
    finally:
        w.kill()


def test_pipe_copy_is_not_counted_as_the_steps_cost(tmp_path, monkeypatch):
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)

    class Slow:
        io_s = 0.0
        dead, ready = False, True

    w = Slow()
    m._worker = w

    def fake_step(*a):
        import time
        time.sleep(0.05)
        w.io_s += 0.05                                # all of it was pipe copy
        return np.zeros((180, 320, 3), np.uint8)
    monkeypatch.setattr(m, "_step", fake_step)
    for _ in range(30):
        m.step(None, None, 1 / 30)
    assert m.step_ms < 5
    m._worker = None
    m.stop()


def test_a_killed_thread_worker_never_holds_up_exit():
    import threading
    from dtouch.bender_worker import ThreadWorker
    gate = threading.Event()
    w = ThreadWorker()
    w.submit(gate.wait, 30)
    w.kill()
    ts = [t for t in threading.enumerate() if t.name == "bender-thread"]
    assert ts and all(t.daemon for t in ts)
    gate.set()



# ---------- a slow machine shrinks, it does not give up (audit round 5) ----------

def _slow_machine(tmp_path, monkeypatch, t_full, secs=120):
    """Bend time proportional to pixels, `t_full` seconds at MAX_PIXELS
    (scratchpad/audit5/slow_machine.py)."""
    import types
    clock = [0.0]
    monkeypatch.setattr(M, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))

    class Slow:
        def __init__(self):
            self.ready, self.dead, self.busy, self.pid, self.born = True, False, False, None, clock[0]
            self.done, self.io_s = None, 0.0

        def submit(self, fn, rgb, *a):
            self.busy, self.rgb = True, rgb
            self.ms = rgb.shape[0] * rgb.shape[1] / B.MAX_PIXELS * t_full * 1000
            self.done = clock[0] + self.ms / 1000
            return True

        def poll(self):
            if self.dead:
                return "dead", None
            if self.done is not None and clock[0] >= self.done:
                self.done, self.busy = None, False
                return "ok", dict(status="ok", rgb=self.rgb, bend_ms=self.ms)
            return None

        def kill(self):
            self.dead = True

    monkeypatch.setattr(M, "make_worker", Slow)
    host = _booted(tmp_path)
    host.res = (1920, 1080)
    m = host.mode
    m.start(host)
    frame = _scene(1080, 1920)
    for _ in range(int(secs * 30)):
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    got = dict(gave_up=m.gave_up, budget=m.governor.budget, steps=m.governor.steps,
               bends=m.bends, stalls=m.stalls)
    m.stop()                                          # (stop resets the governor)
    return got


@pytest.mark.parametrize("t_full", [0.05, 0.3, 1.0, 1.6, 2.0, 4.0])
def test_a_slow_machine_shrinks_to_what_it_can_bend(tmp_path, monkeypatch, t_full):
    """e243122 gave up at 1.6 s and beyond: a stall added only a third of the
    governor's window, and a fresh worker's empty measurement reset it."""
    got = _slow_machine(tmp_path, monkeypatch, t_full)
    assert not got["gave_up"]
    fit = min(1.0, (1000 / B.MIN_RATE) / (t_full * 1000))   # the share that keeps up
    want = max(B.MIN_PIXELS, B.MAX_PIXELS * fit)
    assert got["budget"] <= want / B.STEP_DOWN + 1   # shrunk to (about) what it can do
    if fit >= 1:
        assert got["steps"] == 0                     # a fast machine keeps full size
    assert got["bends"] > 100


def test_giving_up_shows_the_live_camera_not_the_last_bend(tmp_path, monkeypatch):
    import types
    clock = [0.0]
    monkeypatch.setattr(M, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
    phase = ["ok"]

    class W:
        def __init__(self):
            self.ready, self.dead, self.busy, self.pid, self.born = True, False, False, None, clock[0]
            self.p, self.io_s = None, 0.0

        def submit(self, fn, rgb, *a):
            self.p = np.full_like(rgb, 7)             # a distinctive bent frame
            return True

        def poll(self):
            if phase[0] == "die":
                return "dead", None
            if self.p is None:
                return None
            p, self.p = self.p, None
            return "ok", dict(status="ok", rgb=p, bend_ms=10.0)

        def kill(self):
            self.dead = True

    monkeypatch.setattr(M, "make_worker", W)
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    for _ in range(60):
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    phase[0] = "die"
    for _ in range(int(90 * 30)):
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    assert m.gave_up
    white = np.full_like(frame, 255)
    for _ in range(5):
        clock[0] += 1 / 30
        out = m.step(white, None, 1 / 30)
    assert out.mean() > 250
    m.stop()



# ---------- audit round 6 ----------

def _fake_clock(monkeypatch):
    import types
    clock = [0.0]
    monkeypatch.setattr(M, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
    return clock


def test_one_off_stalls_on_a_fast_machine_keep_the_size(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    stall = [False]

    class Quick:
        def __init__(self):
            self.ready, self.dead, self.busy, self.pid, self.born = True, False, False, None, clock[0]
            self.p, self.io_s = None, 0.0

        def submit(self, fn, rgb, *a):
            self.p = rgb
            return True

        def poll(self):
            if self.dead:
                return "dead", None
            if self.p is None or stall[0]:
                return None
            p, self.p = self.p, None
            return "ok", dict(status="ok", rgb=p, bend_ms=10.0)

        def kill(self):
            self.dead = True

    monkeypatch.setattr(M, "make_worker", Quick)
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)

    def run(secs):
        for _ in range(int(secs * 30)):
            clock[0] += 1 / 30
            m.step(frame, None, 1 / 30)
    run(3)
    for _ in range(2):                                # two one-off hiccups
        stall[0] = True
        run(M.STALL_S + 0.2)
        stall[0] = False
        run(10)
    assert m.stalls == 2 and not m.gave_up
    assert m.governor.budget == B.MAX_PIXELS and m.governor.steps == 0
    m.stop()


def test_a_failed_submit_is_not_charged_to_the_step(tmp_path, monkeypatch):
    """scratchpad/audit6/failed_submit.py: the 0.5 s a frozen worker costs
    submit is waiting, and the worker is lost in the same step."""
    clock = _fake_clock(monkeypatch)
    stuck = [False]

    class W:
        def __init__(self):
            self.ready, self.dead, self.busy, self.pid, self.born = True, False, False, None, clock[0]
            self.io_s, self.p = 0.0, None

        def submit(self, fn, rgb, *a):
            if stuck[0]:
                clock[0] += 0.5
                self.io_s += 0.5                      # as ProcessWorker does
                self.dead = True
                return False
            self.p = rgb
            return True

        def poll(self):
            if self.dead:
                return "dead", None
            if self.p is None:
                return None
            p, self.p = self.p, None
            return "ok", dict(status="ok", rgb=p, bend_ms=10.0)

        def kill(self):
            self.dead = True

    monkeypatch.setattr(M, "make_worker", W)
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    for _ in range(90):
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    before = m.step_ms
    stuck[0] = True
    clock[0] += 1 / 30
    m.step(frame, None, 1 / 30)
    assert m.worker_deaths == 1
    assert m.step_ms - before < 1.0
    m.stop()


def test_a_lost_worker_shows_the_live_camera_while_waiting(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    dying = [False]

    class W:
        def __init__(self):
            self.ready, self.dead, self.busy, self.pid, self.born = True, False, False, None, clock[0]
            self.io_s, self.p = 0.0, None

        def submit(self, fn, rgb, *a):
            self.p = np.full_like(rgb, 7)
            return True

        def poll(self):
            if dying[0]:
                return "dead", None
            if self.p is None:
                return None
            p, self.p = self.p, None
            return "ok", dict(status="ok", rgb=p, bend_ms=10.0)

        def kill(self):
            self.dead = True

    monkeypatch.setattr(M, "make_worker", W)
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)
    frame = _scene(180, 320)
    for _ in range(30):
        clock[0] += 1 / 30
        m.step(frame, None, 1 / 30)
    dying[0] = True
    clock[0] += 1 / 30
    m.step(frame, None, 1 / 30)                       # lost; waiting to respawn
    assert m._worker is None and not m.gave_up
    clock[0] += 1 / 30
    out = m.step(np.full_like(frame, 255), None, 1 / 30)
    assert out.mean() > 250
    m.stop()
