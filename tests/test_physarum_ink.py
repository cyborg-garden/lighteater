"""The alive fractal step and the molten-ink landing on the GPU engine.

Holds: the ink draws on H's third step and only there (flat and depth run
the stock programs bit for bit); the mode lands in the ink over violet at
boot and on panic, every other look still lands on depth; any alive
failure falls back to the stock fractal and puts the shared landing back.
GPU tests skip where no standalone GL context can be made (CI).
"""
import numpy as np
import pytest

from dtouch.alive import ALIVE
from dtouch.modes.physarum import PALETTES_PH, PhysarumMode, _lut_rows
from dtouch.physarum_alive import AliveGL, scene_size
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable
from dtouch.shell import Host

from test_shell import SyntheticSource

VIOLET = PALETTES_PH.index("violet")


def _field(**kw):
    kw.setdefault("n", 20000)
    kw.setdefault("gw", 128)
    kw.setdefault("gh", 72)
    kw.setdefault("seed", 5)
    try:
        return PhysarumFieldGL(**kw)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")


def _scene(f):
    yy, xx = np.mgrid[0:f.gh, 0:f.gw]
    matte = (((xx - f.gw / 2) ** 2 + (yy - f.gh / 2) ** 2) < (f.gh * 0.35) ** 2).astype(np.float32)
    gray = (0.3 + 0.4 * matte).astype(np.float32)
    sw, sh = scene_size(f.gw, f.gh)
    import cv2
    f.alive_scene_in = (np.zeros((sh, sw, 4), np.float32), cv2.resize(matte, (sw, sh)))
    return matte, gray


def _ink(f, frames=60):
    matte, gray = _scene(f)
    rows = _lut_rows(tuple(PALETTES_PH))
    img = None
    for _ in range(frames):
        f.update(matte, gray)
        img = f.ink_frame(rows, (VIOLET + 0.5) / len(PALETTES_PH))
    return img


def test_the_ink_draws_on_the_fractal_step_in_violet():
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        img = _ink(f)
        assert f.alive_error is None and f.fractal_error is None
        assert img is not None and img.shape == (144, 256, 3) and img.dtype == np.uint8
        # black ground, purple glow, hard chrome whites: dark on average,
        # with real highlights, and violet (red and blue over green)
        r, g, b = (float(img[..., i].mean()) for i in range(3))
        assert 2.0 < (r + g + b) / 3 < 120.0 and img.max() > 200
        assert b > g and r > g
        st = f.alive_stats()
        assert st is not None and st["heads"] > 0 and not st["readFailed"]
        # luminance() reads the alive picture too (rack and colourise paths)
        assert f.luminance().shape == (144, 256)
    finally:
        f.release()


def test_flat_and_depth_never_touch_the_alive_step():
    """Fractal 0: no alive pass is built or run, and the picture is the
    stock engine's bit for bit, with the step wanted or not."""
    a, b = _field(seed=9), _field(seed=9)
    try:
        b.alive = True
        for depth in (0.0, 0.6):
            for f in (a, b):
                f.depth = depth
            matte, gray = _scene(a)
            _scene(b)
            for _ in range(20):
                a.update(matte, gray)
                b.update(matte, gray)
            assert np.array_equal(a.luminance_u8(), b.luminance_u8())
        assert b._alive is None and not b.alive_display()
    finally:
        a.release()
        b.release()


def test_without_the_step_the_fractal_is_the_stock_one():
    f = _field()
    try:
        f.fractal, f.out_size = 1.0, (256, 144)
        matte, gray = _scene(f)
        for _ in range(5):
            f.update(matte, gray)
        assert f._alive is None and not f.alive_display()
    finally:
        f.release()


def test_an_alive_failure_falls_back_to_the_stock_fractal(monkeypatch):
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        _ink(f, frames=10)

        def boom(self, dt):
            raise RuntimeError("out of memory (test)")
        monkeypatch.setattr(AliveGL, "step_network", boom)
        img = _ink(f, frames=3)
        assert img is None and "out of memory" in f.alive_error
        assert f._alive is None and not f.alive_display()
        lum = f.luminance()                      # the stock fractal, still drawing
        assert lum.shape == (144, 256) and lum.max() > 0
    finally:
        f.release()


def test_a_build_failure_keeps_the_stock_fractal(monkeypatch):
    def boom(self, field):
        raise RuntimeError("no programs (test)")
    monkeypatch.setattr(AliveGL, "__init__", boom)
    f = _field()
    try:
        f.fractal, f.out_size, f.alive = 1.0, (256, 144), True
        assert _ink(f, frames=3) is None
        assert "no programs" in f.alive_error and f.luminance().max() > 0
    finally:
        f.release()


# ----- the mode: the landing ---------------------------------------------------

def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


class _Kept(PhysarumMode):
    """The shell stops the mode when its run ends; keep the field alive for
    the asserts (each test stops it for real)."""
    def stop(self):
        pass

    def really_stop(self):
        PhysarumMode.stop(self)


def _booted(tmp_path, frames=3, **mode_kw):
    host = Host(_Kept(**mode_kw), source=SyntheticSource(), res=(192, 108),
                show=False, preset=None, max_frames=frames, **_paths(tmp_path))
    _, out = host.run()
    return host, out


def test_boot_lands_in_the_ink_over_violet_with_the_camera_hidden(tmp_path):
    host, out = _booted(tmp_path, frames=8)
    ui, m = host.ui, host.mode
    if m.engine != "gl":
        pytest.skip("no GL context available (CI)")
    try:
        assert ALIVE["landing"]["palette"] == "violet"
        assert ui.ph_palette_idx == VIOLET and ui.ph_fractal == 1.0 and ui.ph_depth == 0.9
        assert not ui.ph_video_bg
        assert m.pf.alive and m.pf.alive_error is None and m.pf.alive_display()
        assert out is not None and out.shape == (108, 192, 3)
        assert float(out.mean()) < 120.0          # the black ground
    finally:
        m.really_stop()


def test_panic_lands_in_the_ink_and_a_look_lands_on_depth(tmp_path):
    host, _ = _booted(tmp_path)
    ui, m = host.ui, host.mode
    if m.engine != "gl":
        pytest.skip("no GL context available (CI)")
    try:
        assert host._apply_look("amoeba", PhysarumMode.BUILTIN["amoeba"])
        assert ui.ph_fractal == 0.0 and PALETTES_PH[ui.ph_palette_idx] == "fire"
        host._wire_keys()
        host.reg.dispatch(ord("0"))
        assert ui.land_pending and ui.pending_preset == "veinwork"
        host._apply_pending_preset()
        assert ui.ph_palette_idx == VIOLET and ui.ph_fractal == 1.0
        assert not ui.land_pending
        # a recall of the same look by name lands where looks land: depth
        assert host._apply_look("veinwork", PhysarumMode.BUILTIN["veinwork"])
        assert ui.ph_fractal == 0.0 and PALETTES_PH[ui.ph_palette_idx] == "arctic"
    finally:
        m.really_stop()


def test_alive_off_and_the_cpu_engine_keep_the_shared_landing(tmp_path):
    host, _ = _booted(tmp_path / "a", alive=False)
    try:
        assert host.ui.ph_fractal == 0.0 and PALETTES_PH[host.ui.ph_palette_idx] == "arctic"
        if host.mode.engine == "gl":
            assert host.mode.pf._alive is None
    finally:
        host.mode.really_stop()
    host, _ = _booted(tmp_path / "b", engine="cpu")
    try:
        assert host.ui.ph_fractal == 0.0 and PALETTES_PH[host.ui.ph_palette_idx] == "arctic"
    finally:
        host.mode.really_stop()


def test_an_alive_failure_puts_the_shared_landing_back(tmp_path, monkeypatch):
    def boom(self, field):
        raise RuntimeError("no programs (test)")
    monkeypatch.setattr(AliveGL, "__init__", boom)
    host, out = _booted(tmp_path, frames=4)
    ui, m = host.ui, host.mode
    if m.engine != "gl":
        pytest.skip("no GL context available (CI)")
    try:
        assert "no programs" in m.pf.alive_error
        assert ui.ph_fractal == 0.0 and PALETTES_PH[ui.ph_palette_idx] == "arctic"
        assert out is not None
    finally:
        m.really_stop()
