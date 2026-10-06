"""dtouch/shaders/alive/: the alive fractal step's browser-shared unit.

The files here are vendored verbatim by cyborg-garden-site (alive.js), the
same way the physarum, dither and bender units are. Holds:
  1. every file is marked as browser-shared and stays inside GLSL 3.30 ∩
     ES 3.00, and every program the desktop builds compiles under 330 core;
  2. the alive copies of update/blur/compose are the physarum unit's files
     plus marked alive+ / alive- blocks and nothing else, every addition
     behind u_alive (at 0 the copy runs the stock step);
  3. alive.json is dtouch.alive.ALIVE, and every uniform a pass declares is
     set by dtouch/physarum_alive.py, at the declared vector length;
  4. the pure helpers the shaders mirror match the browser's.
"""
import json
import os
import re

import numpy as np
import pytest

from dtouch.alive import (ALIVE, advance_pulse, alive_active, count_events,
                          depth_quantile, head_stride, keep_for)
from dtouch.alive_looks import ALIVE_JSON_PATH, dumps
from dtouch.physarum_alive import ALIVE_DIR, PROGRAMS, alive_source

from test_physarum_gl import _lint_es300

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSARUM_DIR = os.path.join(ROOT, "dtouch", "shaders", "physarum")
ALIVE_PY = os.path.join(ROOT, "dtouch", "physarum_alive.py")
MARK = "browser-shared: dtouch/shaders/alive"
COPIES = ("update.frag", "blur.frag", "compose.frag")
SHADERS = sorted(n for n in os.listdir(ALIVE_DIR)
                 if n.endswith((".frag", ".vert", ".glsl")))
BLOCK = re.compile(r"^[ \t]*// alive\+\n([\s\S]*?)^[ \t]*// alive-\n", re.M)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _own(name):
    """What the alive step adds to a file: its blocks for the three copies,
    the whole file otherwise."""
    src = _read(os.path.join(ALIVE_DIR, name))
    return "\n".join(BLOCK.findall(src)) if name in COPIES else src


def test_the_unit_has_its_files_and_each_is_marked():
    want = {"blur.frag", "carve.frag", "compose.frag", "events.frag", "glow.frag",
            "heads.frag", "heads.vert", "ink.frag", "life.frag", "pulse.frag",
            "quad.vert", "scene.frag", "update.frag", "video.glsl"}
    assert want <= set(SHADERS)
    for name in SHADERS:
        src = _read(os.path.join(ALIVE_DIR, name))
        assert MARK in _own(name), f"{name} is marked as browser-shared"
        assert "#version" not in src and "precision " not in src, name
    assert os.path.exists(os.path.join(ALIVE_DIR, "README.md"))
    assert os.path.exists(ALIVE_JSON_PATH)


@pytest.mark.parametrize("name", SHADERS)
def test_shader_stays_inside_glsl_es_300(name):
    _lint_es300(name, _read(os.path.join(ALIVE_DIR, name)))


def test_every_program_compiles_under_330_core():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    from dtouch.physarum_gl import load_shader
    try:
        fs, quad = load_shader("fullscreen.vert"), alive_source("quad.vert")
        progs = dict(PROGRAMS)
        progs["carve"] = ("quad", ("video.glsl", "carve.frag"))   # browser only, still shared
        for key, (vert, frags) in progs.items():
            vs = fs if vert == "fs" else quad if vert == "quad" else alive_source(vert)
            ctx.program(vertex_shader=vs, fragment_shader=alive_source(*frags)).release()
    finally:
        ctx.release()


@pytest.mark.parametrize("name", COPIES)
def test_the_copies_are_the_physarum_files_plus_marked_blocks(name):
    copy = _read(os.path.join(ALIVE_DIR, name))
    assert BLOCK.sub("", copy) == _read(os.path.join(PHYSARUM_DIR, name)), (
        f"alive/{name} minus its alive blocks is physarum/{name}, byte for byte")
    assert BLOCK.findall(copy), f"{name} carries alive blocks"
    assert re.search(r"^uniform float u_alive;", copy, re.M)


def test_nothing_alive_leaks_into_the_physarum_unit():
    for name in os.listdir(PHYSARUM_DIR):
        assert not re.search(r"u_alive|alive\+", _read(os.path.join(PHYSARUM_DIR, name))), name


def test_alive_json_matches_the_python_source():
    assert _read(ALIVE_JSON_PATH) == dumps(), (
        "dtouch/shaders/alive/alive.json is stale: python -m dtouch.alive_looks")
    assert json.loads(_read(ALIVE_JSON_PATH))["alive"]["landing"] == {"palette": "violet"}


def test_alive_keys_are_unique_in_the_source():
    """A repeated key in the ALIVE literal silently keeps the last one (the
    browser's body/inkBody collision broke the carve once)."""
    src = _read(os.path.join(ROOT, "dtouch", "alive.py"))
    table = src[src.index("ALIVE = {"):src.index("\n}\n")]
    keys = re.findall(r'^\s+"(\w+)":', table, re.M)
    assert len(keys) == len(set(keys)), [k for k in keys if keys.count(k) > 1]


def test_every_uniform_a_desktop_pass_declares_is_set_at_its_length():
    py = _read(ALIVE_PY)
    used = {f for _, frags in PROGRAMS.values() for f in frags} | {
        v for v, _ in PROGRAMS.values() if v not in ("fs", "quad")}
    for name in sorted(used):
        for kind, uname in re.findall(r"^uniform (\w+) (u_\w+)", _own(name), re.M):
            assert f'"{uname}"' in py, f"{name}: {uname} is set by physarum_alive.py"
            m = re.search(rf'"{uname}", tuple\(ALIVE\["(\w+)"\]\)', py)
            if m and kind.startswith("vec"):
                assert len(ALIVE[m.group(1)]) == int(kind[3]), (uname, m.group(1))


def test_no_em_dashes_in_anything_the_alive_step_adds():
    for name in SHADERS:
        assert "—" not in _own(name), name
    for f in ("alive.py", "alive_looks.py", "physarum_alive.py"):
        assert "—" not in _read(os.path.join(ROOT, "dtouch", f)), f
    assert "—" not in _read(os.path.join(ALIVE_DIR, "README.md"))


# ----- the pure helpers (alive-core.js) --------------------------------------

def test_half_lives_and_the_gate():
    assert keep_for(0.5, 0.5) == pytest.approx(0.5)
    assert keep_for(0.5, 0.0) == 1.0 and keep_for(0.0, 0.1) == 0.0
    assert alive_active(True, False, 1.0) and not alive_active(True, False, 0.0)
    assert not alive_active(False, False, 1.0) and not alive_active(True, True, 1.0)


def test_the_pulse_is_slow_and_sound_only_quickens_it():
    p = advance_pulse(0.0, 1.0, 32.0)
    assert p == pytest.approx(2 * np.pi / 32)
    assert advance_pulse(0.0, 1.0, 32.0, sound=1.0, lock=0.8) == pytest.approx(p * 1.8)
    assert 0.0 <= advance_pulse(6.0, 10.0, 32.0) < 2 * np.pi


def test_sampling_and_counting_helpers():
    assert head_stride(120000, 12000) == 10 and head_stride(10, 12000) == 1
    assert head_stride(18000, 12000) == 2          # Math.round, not banker's
    buf = np.zeros((100, 4), np.float32)
    buf[:, 2] = np.arange(100)
    assert depth_quantile(buf, 0.97) == 97.0
    ev = np.zeros((10, 4), np.float32)
    ev[:, 0] = np.arange(10)
    assert count_events(ev, 3.5) == 4
