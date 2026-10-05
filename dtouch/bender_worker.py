"""Circuit Bender's bend worker: one spawned process and two pipes, owned here.

Why not concurrent.futures.ProcessPoolExecutor: killing its worker while the
worker is writing a result (a bent frame is a few megabytes) leaves the
executor's manager thread blocked forever in recv, and interpreter exit joins
that thread, so quitting the app hung. Here the app's main thread is the
only reader, and it never blocks on the reply pipe: it looks with `poll(0)`
and reads only a message that has started to arrive, which the worker
finishes writing without waiting on anyone. A kill is then always safe:
kill, join with a timeout, close our ends, and whatever half-message was in
a pipe is discarded with it.

Nor does the app block on the request pipe. A job is written to a
non-blocking pipe against a deadline (SEND_S): a worker frozen in a way that
stops it reading (stopped, swapped out, wedged in a C call) makes `submit`
return False in bounded time, and the mode treats that worker as lost.

The protocol, one job at a time:
  worker -> app   ("ready",)                  once its imports are done
  app -> worker   (job, fn, args)             only when ready and idle
  worker -> app   (job, "ok", result) or (job, "error", repr)

The worker watches its parent and exits the moment the app is gone (a force
quit never runs shutdown), so it is never left re-parented to launchd.

`io_s` counts the seconds the app spent copying frames through the pipes,
so the mode can leave that copy out of its own cost (the governor's signal).

`ThreadWorker` has the same interface over a daemon thread, for tests that
monkeypatch the bend (a spawned process cannot see the patch).
"""
from __future__ import annotations

import multiprocessing
import multiprocessing.process
import os
import select
import struct
import threading
import time
from multiprocessing.reduction import ForkingPickler

KILL_JOIN_S = 1.0          # how long a kill waits for the process to go...
KILL_REJOIN_S = 0.5        # ...and once more, before letting go of it
SEND_S = 0.5               # a job not fully written by now: the worker is frozen


def _watch_parent(parent):
    while True:
        if os.getppid() != parent:
            os._exit(0)
        time.sleep(0.5)


def _worker_main(requests, replies, parent):
    threading.Thread(target=_watch_parent, args=(parent,), daemon=True,
                     name="bender-parent-watch").start()
    replies.send(("ready",))
    while True:
        try:
            job, fn, args = requests.recv()
        except (EOFError, OSError):
            os._exit(0)                       # the app closed its end
        try:
            msg = (job, "ok", fn(*args))
        except Exception as e:                # noqa: BLE001: report, keep serving
            msg = (job, "error", repr(e))
        try:
            replies.send(msg)
        except (BrokenPipeError, OSError):
            os._exit(0)


def _frame(obj):
    """`obj` framed as multiprocessing.Connection.recv expects it (a 4-byte
    big-endian length, or -1 and an 8-byte length past 2 GB)."""
    data = bytes(ForkingPickler.dumps(obj))
    n = len(data)
    head = struct.pack("!i", n) if n <= 0x7FFFFFFF else struct.pack("!iQ", -1, n)
    return head + data


class ProcessWorker:
    """One bend at a time in a spawned process. Every method returns in
    bounded time."""

    def __init__(self):
        ctx = multiprocessing.get_context("spawn")
        req_r, self._req_w = ctx.Pipe(duplex=False)
        self._rep_r, rep_w = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=_worker_main,
                                 args=(req_r, rep_w, os.getpid()),
                                 daemon=True, name="bender-worker")
        self._proc.start()
        req_r.close()                         # EOF reaches each side when the
        rep_w.close()                         # other one goes
        os.set_blocking(self._req_w.fileno(), False)
        self.born = time.perf_counter()       # the startup timeout counts from here
        self.ready = False
        self.busy = False
        self.dead = False
        self.io_s = 0.0
        self._job = 0

    @property
    def pid(self):
        return self._proc.pid

    def _write(self, buf, deadline):
        fd = self._req_w.fileno()
        view = memoryview(buf)
        while view:
            left = deadline - time.perf_counter()
            if left <= 0:
                return False
            _, ready, _ = select.select([], [fd], [], left)
            if not ready:
                return False
            try:
                n = os.write(fd, view)
            except BlockingIOError:
                continue
            view = view[n:]
        return True

    def submit(self, fn, *args):
        """Start a job; False when the worker is not ready, busy, dead, or
        not reading (the job could not be written within SEND_S)."""
        if self.dead or not self.ready or self.busy:
            return False
        self._job += 1
        t0 = time.perf_counter()
        try:
            ok = self._write(_frame((self._job, fn, args)), t0 + SEND_S)
        except (BrokenPipeError, OSError):
            ok = False
        self.io_s += time.perf_counter() - t0
        if not ok:
            self.dead = True                  # a half-written job: unusable
            return False
        self.busy = True
        return True

    def poll(self):
        """('ok', result), ('error', why), ('dead', None), or None (nothing
        yet). Reads only what has started to arrive; never waits."""
        while not self.dead:
            t0 = time.perf_counter()
            try:
                if not self._rep_r.poll(0):
                    break
                msg = self._rep_r.recv()
            except (EOFError, OSError):
                self.dead = True
                break
            finally:
                self.io_s += time.perf_counter() - t0
            if msg == ("ready",):
                self.ready = True
                continue
            job, status, value = msg
            if job != self._job:
                continue                      # a stale reply: never ours now
            self.busy = False
            return status, value
        if self.dead or not self._proc.is_alive():
            self.dead = True
            self.busy = False
            return "dead", None
        return None

    def kill(self):
        """Gone, now. Safe mid-write: we own the only reader and drop it. A
        process that will not go is let go of, so exit never waits on it."""
        self.dead = True
        try:
            self._proc.kill()
        except Exception:                     # noqa: BLE001: already gone
            pass
        self._proc.join(KILL_JOIN_S)
        if self._proc.exitcode is None:
            self._proc.join(KILL_REJOIN_S)
        if self._proc.exitcode is None:
            # multiprocessing joins every child it knows of at exit, without
            # a timeout; this one must not be among them
            multiprocessing.process._children.discard(self._proc)
        for conn in (self._req_w, self._rep_r):
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._proc.close()
        except (ValueError, AttributeError):
            pass                              # still running after the joins


class ThreadWorker:
    """The same interface over a daemon thread per job (tests). A thread
    cannot be killed: a killed ThreadWorker's bend finishes on its own and is
    dropped, and being a daemon it never holds up exit."""

    def __init__(self):
        self.born = time.perf_counter()
        self.ready = True
        self.dead = False
        self.pid = None
        self.io_s = 0.0
        self._slot = None                     # [done, status, value] of the job

    @property
    def busy(self):
        return self._slot is not None

    def submit(self, fn, *args):
        if self.dead or self._slot is not None:
            return False
        slot = self._slot = [False, None, None]

        def run():
            try:
                slot[1:] = ["ok", fn(*args)]
            except Exception as e:            # noqa: BLE001
                slot[1:] = ["error", repr(e)]
            slot[0] = True
        threading.Thread(target=run, daemon=True, name="bender-thread").start()
        return True

    def poll(self):
        if self.dead:
            return "dead", None
        slot = self._slot
        if slot is None or not slot[0]:
            return None
        self._slot = None
        return slot[1], slot[2]

    def kill(self):
        self.dead = True
        self._slot = None
