"""Circuit Bender: live JPEG databending of the camera, LightEater's fourth mode
here (the third on the web page, which has no Particles).

Per bend, on a worker thread (`bend_frame`):
  1. the camera frame, cover-cropped to the output's aspect, at the working
     size (dtouch.bender.working_size: about the inspiring camera's 1024 x
     768, never more than the camera delivers)
  2. a sensor bend (dtouch.bender_sensor: BENT CAM, H CLOCK, V CLOCK, ADC
     BITS) on the pixels, before any JPEG exists
  3. a real baseline JPEG, encoded by cv2 (quality JPEG_Q)
  4. a JPEG bend on the bytes (dtouch.bender_jpeg: ZIGZAG PERM, DQT EROSION,
     DHT REMAP, SCAN SWAP, CHROMA AMP, STACK)
  5. Pillow decodes it; a failure or a dead frame is reported, never shown: the
     mode holds the last good frame and backs off (dtouch.bender.Backoff)
  6. an optional pixel sort (dtouch.bender_post.sort_runs)
and on every display frame the main thread draws COPY and SPLIT, the long
exposure (or the bitstream bends' settle), and scales to the output with
NEAREST, so JPEG blocks stay square.

One bend is in flight at a time and the display shows the newest finished
one, so the picture runs at the shell's rate and the bends at whatever this
machine sustains; under MIN_RATE for a few seconds the working size steps
down (Governor). SCAN SWAP walks the scan in Python and is the slow bend, so
it is the one that steps the size down on a slow machine.

Pure numpy/cv2, no GL: the mode runs headless (tests, CI). The browser page
runs the same bends (held byte-identical by the shared goldens) and the
shared post shaders in dtouch/shaders/bender/.

Inspired by CyberShot Cam by @lixofuturista / @cebolander
(github.com/cebola4444/cybershot-cam), a pocket camera that bends its own
JPEGs. That repository carries no licence, so neither its code nor its
numbers are used. Distinct from dtouch/circuit_bent.py, the SIGNAL rack's
signal glitch.

Privacy: every frame is encoded and decoded in memory. Nothing is written.
ASCII text only in panel strings (Hershey fonts render non-ASCII as '?').
"""
from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from .. import bender as B
from ..bender_jpeg import EFFECTS as JPEG_EFFECTS
from ..bender_jpeg import JpegError, bend, rng32
from ..bender_post import fold, post, sort_runs
from ..bender_sensor import SENSOR_EFFECTS, sensor_bend
from ..imgui import DIM
from ..overlay_ui import RES_OPTIONS
from ..panelspec import Cycle, PresetList, Readout, Section, Slider, Toggle

# Electric cyan, clear of Physarum's amber, Dither's magenta and AUTO's gold:
# BGR (255, 225, 90) = RGB (90, 225, 255), the web card's accent.
ACCENT = (255, 225, 90)

MAX_BEND_HZ = 30          # past about 30 bends a second the eye gains nothing
STALL_S = 1.5             # a bend still unsettled after this is abandoned
MAX_ABANDONED = 2         # worker threads left with a stalled bend, at most
GOVERN_EVERY_S = 0.5      # how often the governor looks at the bend rate
RATE_TAU_S = 1.0          # smoothing for the bend-rate readout
THUMB = (16, 9)           # the dead-frame check's thumbnail

# Panel option labels. A Cycle stores its displayed VALUE in a look, so the
# looks below speak in these labels; the maps turn them back into numbers.
EFFECT_LABELS = [B.EFFECT_TITLES[e] for e in B.EFFECTS]
SPLIT_LABELS = ["off" if v == 0 else f"{v} px" for v in B.SPLIT_LADDER]
COPY_LABELS = [B.COPY_NAMES[v] for v in B.COPY_LADDER]
SORT_LABELS = list(B.SORT_LADDER)
LONG_LABELS = ["off" if v == 0 else f"{v} s" for v in B.LONG_LADDER]


def _look_cfg(look):
    """dtouch.bender.LOOKS entry -> a store-key look dict (panel labels)."""
    return dict(effect=B.EFFECT_TITLES[look["effect"]],
                amount=float(look["amount"]),
                split=SPLIT_LABELS[B.SPLIT_LADDER.index(look["split"])],
                copy=COPY_LABELS[B.COPY_LADDER.index(look["copy"])],
                sort=look["sort"],
                long=LONG_LABELS[B.LONG_LADDER.index(look["long"])])


def decode_jpeg(data):
    """RGB uint8 pixels of a (bent) JPEG, or None when it will not decode.
    Pillow, not cv2: both are libjpeg underneath and give the same pixels,
    but cv2 prints libjpeg's "Corrupt JPEG data" warning to stderr for every
    frame a bitstream bend desynchronises (DHT REMAP does it on most), which
    at 30 bends a second buries the terminal. Pillow keeps them quiet."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:                 # noqa: BLE001: any undecodable frame
        return None


def _thumb_stats(rgb):
    return B.luma_stats(cv2.resize(rgb, THUMB, interpolation=cv2.INTER_AREA))


def bend_frame(rgb, effect, amount, seed, phase=0.0, t=0.0, sort="off",
               quality=B.JPEG_Q):
    """One bend, pure: RGB uint8 working-size pixels in, a dict out with
    `status` ('ok' | 'decode' | 'dead'), `rgb` when ok, `parse_fail`, and
    per-stage milliseconds. Runs on the worker thread."""
    t0 = time.perf_counter()
    a = B.clamp01(amount)
    src_stats = _thumb_stats(rgb)
    sensor_ms = 0.0
    img = rgb
    if a > 0 and effect in SENSOR_EFFECTS:
        s0 = time.perf_counter()
        img = sensor_bend(rgb, effect, a, seed, t)
        sensor_ms = (time.perf_counter() - s0) * 1000
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(round(quality * 100))])
    if not ok:
        return dict(status="decode", parse_fail=False, sensor_ms=sensor_ms,
                    enc_ms=0.0, dec_ms=0.0, sort_ms=0.0)
    data = enc.tobytes()
    t1 = time.perf_counter()
    parse_fail = False
    if a > 0 and effect in JPEG_EFFECTS:
        try:
            data = bend(data, effect, a, rng32(seed), phase)
        except JpegError:
            parse_fail = True
    out = decode_jpeg(data)
    t2 = time.perf_counter()
    timing = dict(parse_fail=parse_fail, sensor_ms=sensor_ms,
                  enc_ms=(t1 - t0) * 1000 - sensor_ms, dec_ms=(t2 - t1) * 1000,
                  sort_ms=0.0)
    if out is None or out.shape[:2] != rgb.shape[:2]:
        return dict(status="decode", **timing)
    out = np.array(out)               # writable: the sort works in place
    if a > 0 and B.dead_frame(_thumb_stats(out), src_stats):
        return dict(status="dead", **timing)
    if sort and sort != "off":
        s0 = time.perf_counter()
        sort_runs(out, B.SORT_BAND[0], B.SORT_BAND[1], sort)
        timing["sort_ms"] = (time.perf_counter() - s0) * 1000
    return dict(status="ok", rgb=out, **timing)


def cover_crop(frame, aspect):
    """The centre of `frame` cropped to `aspect` (w / h), cover-fit."""
    h, w = frame.shape[:2]
    if w / h > aspect:
        cw = max(1, int(round(h * aspect)))
        x0 = (w - cw) // 2
        return frame[:, x0:x0 + cw]
    ch = max(1, int(round(w / aspect)))
    y0 = (h - ch) // 2
    return frame[y0:y0 + ch]


class BenderMode:
    """Live JPEG databending as a shell plugin (Mode protocol, DESIGN.md §2.2)."""

    id = "bender"
    title = "Circuit Bender"
    key = "j"                    # j for jpeg: c, b, e and the like are its own keys
    accent = ACCENT
    accepts_still = False
    blurb = "live jpeg\ndatabending"

    # Built-in looks (DESIGN.md §7): one per effect, in E's order, from the
    # shared table (dtouch.bender.LOOKS). The first, BENT CAM, is where the
    # mode lands and the panic target.
    BUILTIN = {B.EFFECT_TITLES[k]: _look_cfg(v) for k, v in B.LOOKS.items()}
    DEFAULTS = _look_cfg(B.LOOKS[B.SAFE_LOOK])

    _UI_DEFAULTS = dict(bd_effect_idx=0, bd_amount=B.LOOKS[B.SAFE_LOOK]["amount"],
                        bd_split_idx=0, bd_copy_idx=0, bd_sort_idx=0,
                        bd_long_idx=0)

    def __init__(self):
        self.host = None
        self._pool = None
        self._reset()

    def _reset(self):
        self.cut = B.CutClock(1)
        self.backoff = B.Backoff()
        self.governor = B.Governor()
        self.size = (0, 0)
        self.bent = None              # the newest good bent frame (RGB, working size)
        self.stack = None             # the running mean (float32), or None
        self.inflight = None          # (future, submitted_at, clean)
        self.abandoned = []           # futures given up on, still running
        self.last_submit = -math.inf
        self.t = 0.0
        self.bends = 0
        self.rate = 0.0
        self.since_govern = 0.0
        self.decode_fails = self.dead_rejects = self.parse_fails = self.stalls = 0
        self.last_timing = {}
        self.bass = 0.0

    # ----- lifecycle -----
    def start(self, host):
        self.host = host
        self._reset()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bender")

    def stop(self):
        """Idempotent (DESIGN.md §2.2). The worker thread is released without
        waiting: a bend in flight finishes on its own and is dropped."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._reset()

    def on_resize(self, w, h):
        pass                          # the working size follows host.res per step

    # ----- ui plumbing -----
    def configure_ui(self, ui):
        for k, v in self._UI_DEFAULTS.items():
            if not hasattr(ui, k):
                setattr(ui, k, v)

    def _ui(self, attr, default):
        ui = self.host.ui if self.host is not None else None
        return getattr(ui, attr, default) if ui is not None else default

    def _pick(self, attr, options, values):
        return values[int(self._ui(attr, 0)) % len(options)]

    def effect(self):
        return self._pick("bd_effect_idx", EFFECT_LABELS, B.EFFECTS)

    def amount(self):
        return B.clamp01(self._ui("bd_amount", self._UI_DEFAULTS["bd_amount"]))

    def split(self):
        return self._pick("bd_split_idx", SPLIT_LABELS, B.SPLIT_LADDER)

    def copy(self):
        return self._pick("bd_copy_idx", COPY_LABELS, B.COPY_LADDER)

    def sort(self):
        return self._pick("bd_sort_idx", SORT_LABELS, B.SORT_LADDER)

    def long(self):
        return self._pick("bd_long_idx", LONG_LABELS, B.LONG_LADDER)

    # ----- panel -----
    def panel_spec(self):
        return [
            Section("TEMPLATES", [PresetList()]),
            Section("SOURCE", [
                Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                      key="res", save=False, nudge=False),
                Toggle("Mirror", "mirror", on_text="on", save=False),
            ]),
            Section("BEND", [
                Cycle("effect", "bd_effect_idx", list(EFFECT_LABELS),
                      save_key="effect", status=str,
                      tip="E steps through them. Bent cam is what a "
                          "circuit-bent camera looks like at a glance; the "
                          "next six bend the JPEG file, the last three the "
                          "sensor."),
                Slider("Amount", "bd_amount", 0.0, 1.0, save_key="amount",
                       status="{:.0%}",
                       tip="How hard the bend bites. B steps 25/50/75/100%. "
                           "A bend that breaks the frame backs itself off."),
                Readout(self._draw_rate),
            ]),
            Section("POST", [
                Cycle("split", "bd_split_idx", list(SPLIT_LABELS),
                      save_key="split",
                      tip="Chromatic aberration: red right, blue left (X)."),
                Cycle("copy", "bd_copy_idx", list(COPY_LABELS),
                      save_key="copy",
                      tip="Block runs and echoed bands, like a decoder that "
                          "lost its place (C)."),
                Cycle("sort", "bd_sort_idx", list(SORT_LABELS),
                      save_key="sort",
                      tip="Pixel sort of the mid-tones, by rows or columns "
                          "(K)."),
                Cycle("long exposure", "bd_long_idx", list(LONG_LABELS),
                      save_key="long",
                      tip="Stack the last few seconds of bends: what holds "
                          "still stays sharp, what moves smears (L)."),
            ]),
        ]

    def commands(self):
        """E effect, B amount ladder, X split, C copy, K sort, L long."""
        from ..commands import Command
        ui, toasts = self.host.ui, self.host.hud.toasts

        def _step(attr, labels, name):
            def run():
                setattr(ui, attr, (int(getattr(ui, attr, 0)) + 1) % len(labels))
                label = labels[getattr(ui, attr)]
                toasts.flash((f"{name} {label}" if name else label).upper())
            return run

        def _amount():
            ui.bd_amount = float(B.next_in(B.AMOUNT_LADDER, self.amount() + 1e-9))
            toasts.flash("AMOUNT %d%%" % round(ui.bd_amount * 100))

        return {
            "bender.effect": Command("bender.effect", "Next bend", "e",
                                     _step("bd_effect_idx", EFFECT_LABELS, "")),
            "bender.amount": Command("bender.amount", "Bend amount 25-100%",
                                     "b", _amount),
            "bender.split": Command("bender.split", "Split (aberration)", "x",
                                    _step("bd_split_idx", SPLIT_LABELS, "split")),
            "bender.copy": Command("bender.copy", "Copy runs", "c",
                                   _step("bd_copy_idx", COPY_LABELS, "copy")),
            "bender.sort": Command("bender.sort", "Pixel sort", "k",
                                   _step("bd_sort_idx", SORT_LABELS, "sort")),
            "bender.long": Command("bender.long", "Long exposure", "l",
                                   _step("bd_long_idx", LONG_LABELS, "long")),
        }

    def safe_look(self):
        return B.EFFECT_TITLES[B.SAFE_LOOK]

    def status_tail(self, cam_name):
        w, h = self.size
        return f"{w}x{h} jpeg  {self.rate:.0f}/s  src {cam_name[:16]}"

    def _draw_rate(self, frame, g, x, y, cw):
        """Bend rate and working size: the honest perf note (§4.2)."""
        w, h = self.size
        g.text(frame, f"{self.rate:.0f} bends/s at {w}x{h}", x, y + g.S(12),
               DIM, 0.42)
        return y + g.S(20)

    # ----- the bend loop -----
    def _working_size(self, frame):
        rw, rh = self.host.res
        fh, fw = frame.shape[:2]
        w, h, _ = B.working_size(rw, rh, fw, fh, self.governor.budget)
        return w, h

    def _settle(self, now):
        """Collect a finished bend; abandon one that stalled."""
        fut, at, clean = self.inflight
        if fut.done():
            self.inflight = None
            try:
                r = fut.result()
            except Exception:             # noqa: BLE001: a bend never kills the show
                self.backoff.fail()
                self.decode_fails += 1
                return
            self.bends += 1
            self.last_timing = r
            if r.get("parse_fail"):
                self.parse_fails += 1
            if r["status"] == "ok":
                self.backoff.ok(clean=clean)
                self.bent = r["rgb"]
            else:
                if r["status"] == "dead":
                    self.dead_rejects += 1
                else:
                    self.decode_fails += 1
                self.backoff.fail()
        elif now - at > STALL_S:
            # a thread cannot be killed: leave it to finish on a pool of its
            # own and bend on a fresh one, at most MAX_ABANDONED at a time
            self.abandoned = [f for f in self.abandoned if not f.done()]
            if len(self.abandoned) >= MAX_ABANDONED:
                return
            self.stalls += 1
            self.abandoned.append(fut)
            self.inflight = None
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="bender")

    def _kick(self, frame, now):
        if self._pool is None or self.inflight is not None:
            return
        if now - self.last_submit < 1.0 / MAX_BEND_HZ - 0.008:
            return
        w, h = self._working_size(frame)
        if (w, h) != self.size:
            self.size = (w, h)
            self.stack = None
        src = cv2.resize(cover_crop(frame, w / h), (w, h),
                         interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        effect = self.effect()
        want = B.live_amount(self.amount(), bass=self.bass,
                             sens=float(self._ui("sens", 1.0)))
        a = self.backoff.amount(want)
        seed = B.touch_seed(self.cut.cuts) if effect in SENSOR_EFFECTS else self.cut.seed
        fut = self._pool.submit(bend_frame, rgb, effect, a, seed,
                                self.cut.phase, self.t, self.sort())
        # a clean frame: the back-off ran out of tries and sent it unbent
        self.inflight = (fut, now, a == 0 and want > 0)
        self.last_submit = now

    def step(self, frame_bgr, audio_levels, dt):
        """Camera frame (pre-mirrored by the shell) -> the newest bent frame,
        post, settle -> RGB at host.res."""
        rw, rh = self.host.res
        now = time.perf_counter()
        dt = max(0.0, float(dt or 0.0))
        self.t += dt
        onset = False
        self.bass = 0.0
        if audio_levels is not None:
            self.bass = float(audio_levels.get("bass", 0.0))
            onset = bool(audio_levels.get("onset", False))
        self.cut.step(dt, onset, B.cut_tempo(self.effect()))
        bends_before = self.bends
        if self.inflight is not None:
            self._settle(now)
        self._kick(frame_bgr, now)
        # the bend rate, smoothed; the governor steps the size down on it
        inst = (self.bends - bends_before) / dt if dt > 0 else 0.0
        k = 1 - math.exp(-dt / RATE_TAU_S) if dt > 0 else 0.0
        self.rate += (inst - self.rate) * k
        self.since_govern += dt
        if self.since_govern >= GOVERN_EVERY_S and self.bends > 0:
            self.governor.step(self.since_govern, self.rate)
            self.since_govern = 0.0

        pic = self.bent
        if pic is None or pic.shape[:2] != (self.size[1], self.size[0]):
            # nothing bent at this size yet: the camera, unbent, not black
            w, h = self.size if self.size[0] else self._working_size(frame_bgr)
            pic = cv2.cvtColor(cv2.resize(cover_crop(frame_bgr, w / h), (w, h),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
        pic = post(pic, self.split(), self.copy(),
                   seed=(self.cut.seed % 9973) + 0.5, block=B.BLOCK)
        tau = B.stack_tau(self.effect(), self.long())
        if tau > 0:
            alpha = 1.0 if self.stack is None else 1 - math.exp(-dt / tau)
            if self.stack is not None and self.stack.shape != pic.shape:
                self.stack = None
                alpha = 1.0
            self.stack = fold(self.stack, pic, alpha)
            pic = np.clip(self.stack + 0.5, 0, 255).astype(np.uint8)
        else:
            self.stack = None
        return cv2.resize(pic, (rw, rh), interpolation=cv2.INTER_NEAREST)
