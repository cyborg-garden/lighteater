"""The browser-shared Circuit Bender unit (dtouch/shaders/bender/README.md).

cyborg-garden-site vendors `dtouch/shaders/bender/` verbatim, so this suite
holds the contract the page relies on: both JSON files equal what the Python
source generates, the shaders stay inside GLSL 3.30 ∩ ES 3.00 and compile
under 330 core, post.frag draws what `dtouch.bender_post.post` computes, and
the goldens are real (distinct, non-trivial outputs). The GL tests skip (not
fail) where no context can be created.
"""
import json
import os

import numpy as np
import pytest

from dtouch import bender as B
from dtouch.bender_jpeg import mcu_offsets, validate_jpeg
from dtouch.bender_looks import (FIXTURES, GOLDENS_PATH, LOOKS_PATH, UNIT_DIR,
                                 dumps_goldens, dumps_looks)
from dtouch.bender_post import post
from dtouch.bender_sensor import SENSOR_EFFECTS
from dtouch.modes.bender import BenderMode
from dtouch.physarum_gl import load_shared_shader, load_shader

from test_physarum_gl import _lint_es300

SHADERS = ("post.frag", "stack.frag", "view.frag")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _body(name):
    return load_shared_shader("bender", name, version_line="").split("#line 1\n", 1)[1]


@pytest.fixture(scope="module")
def looks():
    return json.loads(_load(LOOKS_PATH))


@pytest.fixture(scope="module")
def goldens():
    return json.loads(_load(GOLDENS_PATH))


def _ctx():
    try:
        import moderngl
        return moderngl.create_standalone_context()
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")


# ---------- the files are what the source generates ----------

def test_looks_json_is_the_generated_text():
    assert _load(LOOKS_PATH) == dumps_looks(), \
        "bender_looks.json is stale: run `python -m dtouch.bender_looks`"


def test_goldens_json_is_the_generated_text():
    assert _load(GOLDENS_PATH) == dumps_goldens(), \
        "bender_goldens.json is stale: run `python -m dtouch.bender_looks`"


def test_exporter_rerun_is_a_no_op():
    assert dumps_looks() == dumps_looks()
    assert dumps_goldens() == dumps_goldens()


def test_unit_holds_only_what_the_readme_lists():
    names = sorted(n for n in os.listdir(UNIT_DIR) if not n.startswith("."))
    assert names == sorted(["README.md", "bender_goldens.json", "bender_looks.json",
                            *SHADERS, *FIXTURES])


# ---------- the shaders ----------

@pytest.mark.parametrize("name", SHADERS)
def test_shader_carries_no_host_lines(name):
    body = _body(name)
    assert body.strip()
    assert "#version" not in body
    assert "precision " not in body


@pytest.mark.parametrize("name", SHADERS)
def test_shader_stays_inside_glsl_es_300(name):
    _lint_es300(name, _body(name))


@pytest.mark.parametrize("name", SHADERS)
def test_shader_compiles_under_330_core(name):
    ctx = _ctx()
    try:
        src = load_shared_shader("bender", name)
        assert src.startswith("#version 330 core\n#line 1\n")
        ctx.program(vertex_shader=load_shader("fullscreen.vert"), fragment_shader=src)
    finally:
        ctx.release()


def _gpu_post(rgb, split, copy, seed, block=B.BLOCK):
    """post.frag on the GPU over `rgb` (RGB uint8), read back as RGB."""
    ctx = _ctx()
    try:
        h, w = rgb.shape[:2]
        prog = ctx.program(vertex_shader=load_shader("fullscreen.vert"),
                           fragment_shader=load_shared_shader("bender", "post.frag"))
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
        tex = ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes())
        out = ctx.texture((w, h), 4)
        fbo = ctx.framebuffer(color_attachments=[out])
        tex.use(0)
        for name, v in (("u_src", 0), ("u_size", (w, h)), ("u_split", float(split)),
                        ("u_copy", float(copy)), ("u_seed", float(seed)),
                        ("u_block", block)):
            if name in prog:
                prog[name].value = v
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        vao = ctx.vertex_array(prog, [(tri, "2f", "in_vert")])
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        vao.render(vertices=3)
        px = np.frombuffer(fbo.read(components=4), np.uint8).reshape(h, w, 4)
        return px[..., :3].copy()
    finally:
        ctx.release()


@pytest.mark.parametrize("split,copy", [(0, 1.0), (4, 0.0), (10, 0.5)])
def test_post_frag_draws_what_post_computes(split, copy):
    """The desktop's numpy COPY + SPLIT is the shader's rule. The hash runs
    in float32 on both, and a GPU may round its last place differently, so a
    sliver of blocks may pick differently: at least 99% of pixels agree."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (96, 160, 3), np.uint8)
    seed = 1234.5
    gpu = _gpu_post(rgb, split, copy, seed)
    cpu = post(rgb, split, copy, seed=seed, block=B.BLOCK)
    agree = np.all(gpu == cpu, axis=-1).mean()
    assert agree >= 0.99, agree
    if copy > 0:
        assert not np.array_equal(cpu, rgb)          # the copy did something


# ---------- the tables ----------

def test_looks_card_is_the_mode(looks):
    assert looks["title"] == BenderMode.title
    assert looks["key"] == BenderMode.key
    assert looks["blurb"] == BenderMode.blurb
    assert looks["accent"] == list(BenderMode.accent[::-1])


def test_looks_tables_are_the_python_tables(looks):
    assert looks["effects"] == list(B.EFFECTS)
    assert looks["safe_look"] == B.SAFE_LOOK == looks["effects"][0]
    assert list(looks["looks"]) == list(B.LOOKS)
    for name, look in looks["looks"].items():
        assert look == {k: (int(v) if isinstance(v, float) and v.is_integer() else v)
                        for k, v in B.LOOKS[name].items()}
    assert looks["settle_s"].keys() == set(B.EFFECTS)
    assert looks["jpeg"]["quality"] == B.JPEG_Q


# ---------- the goldens are real ----------

def test_fixtures_are_baseline_jpegs_that_walk(goldens):
    for name in FIXTURES:
        with open(os.path.join(UNIT_DIR, name), "rb") as fh:
            data = fh.read()
        assert validate_jpeg(data)["ok"], name
        assert b"Exif" not in data, f"{name} carries EXIF"
        assert mcu_offsets(data)["complete"], name
        assert goldens["walks"][name]["complete"]


def test_goldens_are_distinct_and_cover_every_bend(goldens):
    """A golden that repeats or never changes its input proves nothing. The
    table bends ignore the seed, so their two seeds share a hash; every
    (fixture, effect, amount) differs, and the seeded bends differ by seed."""
    import hashlib
    inputs = set()
    for name in FIXTURES:
        with open(os.path.join(UNIT_DIR, name), "rb") as fh:
            inputs.add(hashlib.sha256(fh.read()).hexdigest())
    groups = {}
    for c in goldens["jpeg"]:
        groups.setdefault((c["fixture"], c["effect"], c["amount"]), set()).add(c["sha256"])
    firsts = [sorted(h)[0] for h in groups.values()]
    assert len(set(firsts)) == len(firsts)
    # a light DHT REMAP touches only the sizes 8 and up, which these small
    # fixtures barely use: it may leave one unchanged. Nothing else may.
    same = {(c["fixture"], c["effect"], c["amount"]) for c in goldens["jpeg"]
            if c["sha256"] in inputs}
    assert all(e == "remap" and a == 0.25 for _, e, a in same), same
    for (_, effect, _), hs in groups.items():
        if effect in ("swap", "stack"):
            assert len(hs) == 2, effect
    effects = {c["effect"] for c in goldens["jpeg"]}
    assert effects == set(B.EFFECTS) - set(SENSOR_EFFECTS) | {"smear"}
    sens = [c["sha256"] for c in goldens["sensor"]]
    assert len(set(sens)) == len(sens)
    # LINE STREAK remembers frames, so it has sequences instead (streak_seq)
    assert {c["effect"] for c in goldens["sensor"]} == set(SENSOR_EFFECTS) - {"streak"}
    assert set(SENSOR_EFFECTS) == {"bent", "thermal", "streak", "hclock", "vclock", "adc"}
    seq = [f["sha256"] for c in goldens["streak_seq"] for f in c["frames"]]
    assert len(set(seq)) == len(seq)
    assert all(c["runs"] > 0 for c in goldens["sort"])
    first = goldens["rng32"]["first"]
    assert len(set(first)) == len(first) and all(0 <= v < 1 for v in first)
