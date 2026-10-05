"""Circuit Bender's bend worker: one spawned process and one Pipe, owned here.

Why not concurrent.futures.ProcessPoolExecutor: killing its worker while the
worker is writing a result (a bent frame is a few megabytes) leaves the
executor's manager thread blocked forever in recv, and interpreter exit joins
that thread, so quitting the app hung. Here the app's main thread is the
only reader, and it never blocks on the pipe: it looks with `poll(0)` and
reads only a message that has started to arrive, which the worker finishes
writing without waiting on anyone. A kill is then always safe: kill, join
with a timeout, close our end, and whatever half-message was in the pipe is
discarded with it.

The protocol, one job at a time:
  worker -> app   ("ready",)                  once its imports are done
  app -> worker   (job, fn, args)             only when ready and idle
  worker -> app   (job, "ok", result) or (job, "error", repr)
The app sends a job only to an idle worker that said ready, so the worker is
always sitting in recv and the send (a working-size frame) never waits on it.

The worker watches its parent and exits the moment the app is gone (a force
quit never runs shutdown), so it is never left re-parented to launchd.

`ThreadWorker` has the same interface over one thread, for tests that
monkeypatch the bend (a spawned process cannot see the patch).
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

KILL_JOIN_S = 1.0          # how long a kill waits for the process to go


def _watch_parent(parent):
    while True:
        if os.getppid() != parent:
            os._exit(0)
        time.sleep(0.5)


def _worker_main(conn, parent):
    threading.Thread(target=_watch_parent, args=(parent,), daemon=True,
                     name="bender-parent-watch").start()
    conn.send(("ready",))
    while True:
        try:
            job, fn, args = conn.recv()
        except (EOFError, OSError):
            os._exit(0)                       # the app closed its end
        try:
            msg = (job, "ok", fn(*args))
        except Exception as e:                # noqa: BLE001: report, keep serving
            msg = (job, "error", repr(e))
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError):
            os._exit(0)


class ProcessWorker:
    """One bend at a time in a spawned process. Every method returns at once."""

    def __init__(self):
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=_worker_main, args=(child, os.getpid()),
                                 daemon=True, name="bender-worker")
        self._proc.start()
        child.close()                         # EOF reaches us when it dies
        self.born = time.perf_counter()       # the startup timeout counts from here
        self.ready = False
        self.busy = False
        self._job = 0
        self.dead = False

    @property
    def pid(self):
        return self._proc.pid

    def submit(self, fn, *args):
        """Start a job; False when the worker is not ready, busy or dead."""
        if self.dead or not self.ready or self.busy:
            return False
        self._job += 1
        try:
            self._conn.send((self._job, fn, args))
        except (BrokenPipeError, EOFError, OSError):
            self.dead = True
            return False
        self.busy = True
        return True

    def poll(self):
        """('ok', result), ('error', why), ('dead', None), or None (nothing
        yet). Reads only what has started to arrive; never waits."""
        while not self.dead:
            try:
                if not self._conn.poll(0):
                    break
                msg = self._conn.recv()
            except (EOFError, OSError):
                self.dead = True
                break
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
        """Gone, now. Safe mid-write: we own the only reader and drop it."""
        self.dead = True
        try:
            self._proc.kill()
        except Exception:                     # noqa: BLE001: already gone
            pass
        self._proc.join(KILL_JOIN_S)
        try:
            self._conn.close()
        except OSError:
            pass
        try:
            self._proc.close()
        except (ValueError, AttributeError):
            pass                              # still running after the join


class ThreadWorker:
    """The same interface over one thread (tests). A thread cannot be
    killed: a killed ThreadWorker's bend finishes on its own and is dropped."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bender")
        self._fut = None
        self.born = time.perf_counter()
        self.ready = True
        self.dead = False
        self.pid = None

    @property
    def busy(self):
        return self._fut is not None

    def submit(self, fn, *args):
        if self.dead or self._fut is not None:
            return False
        self._fut = self._pool.submit(fn, *args)
        return True

    def poll(self):
        if self.dead:
            return "dead", None
        f = self._fut
        if f is None or not f.done():
            return None
        self._fut = None
        try:
            return "ok", f.result()
        except Exception as e:                # noqa: BLE001
            return "error", repr(e)

    def kill(self):
        self.dead = True
        self._fut = None
        self._pool.shutdown(wait=False, cancel_futures=True)
