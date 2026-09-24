"""The browser-shared dither unit (dtouch/shaders/dither/README.md).

cyborg-garden-site vendors `dtouch/shaders/dither/` verbatim, so this suite
holds the contract the page relies on: both JSON files equal what the
Python source generates, the shader stays inside GLSL 3.30 ∩ ES 3.00 and
compiles under 330 core, and the tables and goldens mean what they claim.
The GL compile test skips (not fails) where no context can be created.
"""
import base64
import json

import numpy as np
import pytest

from dtouch.dither import (_encode_levels, _ordered_indices, bayer_dither,
                           blue_noise_dither)
from dtouch.dither_looks import (GOLDENS_PATH, LOOKS_PATH, dumps_goldens,
                                 dumps_looks, fixture)
from dtouch.modes.dithergirl import (LEGIBILITY_FLOOR, PALETTES, DitherGirlMode,
                                     contrast_ratio, palette_pair)
from dtouch.rack_gl import load_shared_shader

from test_physarum_gl import _lint_es300


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _body():
    return load_shared_shader("dither", "dither.frag", version_line="").split("#line 1\n", 1)[1]


@pytest.fixture(scope="module")
def looks():
    return json.loads(_load(LOOKS_PATH))


@pytest.fixture(scope="module")
def goldens():
    return json.loads(_load(GOLDENS_PATH))


def test_looks_json_is_the_generated_text():
    assert _load(LOOKS_PATH) == dumps_looks(), \
        "dither_looks.json is stale: run `python -m dtouch.dither_looks`"


def test_goldens_json_is_the_generated_text():
    assert _load(GOLDENS_PATH) == dumps_goldens(), \
        "dither_goldens.json is stale: run `python -m dtouch.dither_looks`"


def test_shared_frag_carries_no_host_lines():
    body = _body()
    assert body.strip()
    assert "#version" not in body
    assert "precision " not in body


def test_shared_frag_stays_inside_glsl_es_300():
    _lint_es300("dither.frag", _body())


def test_shared_frag_compiles_under_330_core():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        from dtouch.rack_gl import load_rack_shader
        src = load_shared_shader("dither", "dither.frag")
        assert src.startswith("#version 330 core\n#line 1\n")
        ctx.program(vertex_shader=load_rack_shader("fullscreen.vert"),
                    fragment_shader=src)
    finally:
        ctx.release()


def test_every_pair_clears_the_legibility_floor(looks):
    for name in PALETTES:
        for inv in (False, True):
            off, on = palette_pair(name, inv)
            assert contrast_ratio(off, on) >= LEGIBILITY_FLOOR, (name, inv)
    for name, pair in looks["one_bit_pairs"].items():
        for side in ("normal", "inverted"):
            off, on = pair[side]
            assert contrast_ratio(off, on) >= LEGIBILITY_FLOOR, (name, side)


def test_web_builtin_names_exist_and_are_renderable(looks):
    assert looks["web_builtin"], "no built-in look the page can render"
    for name in looks["web_builtin"]:
        look = looks["builtin"][name]
        assert look["algorithm"] in looks["web_algos"]
        assert look["palette"] in looks["palettes"]
    assert set(looks["web_algos"]) <= set(looks["ordered"])
    assert set(looks["builtin"]) == set(DitherGirlMode.BUILTIN)


def test_matrices_are_the_desktop_thresholds(looks):
    """The exported integers, divided back, dither exactly like the public
    bayer_dither / blue_noise_dither."""
    bayer = np.array(looks["matrices"]["bayer4"], np.float32) / 16.0
    blue = np.array(looks["matrices"]["bluenoise64"], np.float32) / 4096.0
    img = np.random.default_rng(5).random((70, 70)).astype(np.float32)
    for mat, ref in ((bayer, bayer_dither), (blue, blue_noise_dither)):
        for gamma in (True, False):
            idx, _ = _ordered_indices(img, mat, 2, False, gamma)
            assert np.array_equal(_encode_levels(idx, 3, gamma),
                                  ref(img, bits=2, invert=False, gamma=gamma))


def test_gradient_fixture_holds_every_code():
    assert len(np.unique(fixture("gradient"))) == 256
    assert fixture("dark").max() < 64 and fixture("light").min() >= 192


def test_goldens_have_the_designed_shape(goldens):
    assert len(goldens["core"]) == 48
    assert len(goldens["fs"]) == 6
    assert len(goldens["rgb"]) == 12
    names = [c["name"] for c in goldens["core"] + goldens["fs"] + goldens["rgb"]]
    assert len(names) == len(set(names))


def test_every_core_golden_is_a_real_pattern(goldens):
    for case in goldens["core"]:
        idx = np.frombuffer(base64.b64decode(case["idx"]), np.uint8)
        assert idx.size == 64 * 64, case["name"]
        assert len(np.unique(idx)) >= 2, f"{case['name']} is flat"
        assert idx.max() <= (1 << case["bits"]) - 1, case["name"]


def test_auto_bias_resolves_both_ways(goldens):
    auto = {c["fixture"]: set() for c in goldens["core"] if c["bias"] == "auto"}
    for c in goldens["core"]:
        if c["bias"] == "auto":
            auto[c["fixture"]].add(c["invert"])
    assert auto == {"dark": {True}, "light": {False}}


def test_shared_frag_reproduces_the_core_goldens_with_identity_enc(looks, goldens):
    """The browser's recipe on the real shader: identity `u_enc_lut`, the
    exported integer matrices and `luts.lin`, the fixture as u8 codes in an
    RGBA8 texture. `.r * 255` must be every core golden's level index."""
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    from dtouch.rack_gl import _nearest, load_rack_shader
    try:
        prog = ctx.program(vertex_shader=load_rack_shader("fullscreen.vert"),
                           fragment_shader=load_shared_shader("dither", "dither.frag"))
        vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])
        out = _nearest(ctx.texture((64, 64), 4, dtype="f4"))
        fbo = ctx.framebuffer(color_attachments=[out])
        ident = np.arange(256, dtype=np.float32) / np.float32(255.0)
        ident = _nearest(ctx.texture((256, 1), 1, ident.tobytes(), dtype="f4"))
        div = {"bayer4": 16.0, "bluenoise64": 4096.0}
        for case in goldens["core"]:
            codes = fixture(case["fixture"])
            rgba = np.ascontiguousarray(np.repeat(codes[..., None], 4, axis=2))
            src = _nearest(ctx.texture((64, 64), 4, rgba.tobytes()))
            mname = goldens["matrix_of"][case["algo"]]
            m = np.array(looks["matrices"][mname], np.float32) / np.float32(div[mname])
            mat = _nearest(ctx.texture(m.shape[::-1], 1,
                                       np.ascontiguousarray(m).tobytes(), dtype="f4"))
            levels = (1 << case["bits"]) - 1
            lut = np.array(looks["luts"]["lin"][str(levels)], np.float32)
            lin = _nearest(ctx.texture((256, 1), 1, lut.tobytes(), dtype="f4"))
            for unit, tex in enumerate((src, mat, lin, ident)):
                tex.use(unit)
            for name, unit in (("u_src", 0), ("u_mat", 1), ("u_lin_lut", 2),
                               ("u_enc_lut", 3)):
                prog[name].value = unit
            prog["u_mat_size"].value = m.shape[::-1]
            prog["u_levels"].value = float(levels)
            prog["u_invert"].value = int(case["invert"])
            prog["u_gamma"].value = int(case["gamma"])
            fbo.use()
            vao.render(moderngl.TRIANGLES)
            got = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32)
            got = np.rint(got.reshape(64, 64, 4)[..., 0] * 255.0).astype(np.uint8)
            want = np.frombuffer(base64.b64decode(case["idx"]), np.uint8).reshape(64, 64)
            # texture row 0 is uploaded data row 0 and fbo.read returns
            # framebuffer row 0 first, both at gl_FragCoord.y 0: no flip
            assert np.array_equal(got, want), case["name"]
            for tex in (src, mat, lin):
                tex.release()
    finally:
        ctx.release()


def test_autopilot_block_is_dtouch_auto(looks):
    """The browser's autopilot port runs on these numbers, so they must be
    dtouch/auto.py's own, not a copy that drifted."""
    from dtouch import auto
    a = looks["autopilot"]
    assert a["dwell"] == list(auto.DWELL)
    assert a["mode_every"] == list(auto.MODE_EVERY)
    assert a["cast_chance"] == auto.CAST_CHANCE
    assert a["first_dwell"] == auto.FIRST_DWELL
    assert a["dt_max"] == auto.DT_MAX
    assert a["casts"] == list(auto.CASTS)
    # the claim the port relies on: integers(lo, hi) never returns hi
    assert a["mode_every_high"] == "exclusive"
    rng = np.random.default_rng(0)
    draws = {int(rng.integers(*auto.MODE_EVERY)) for _ in range(2000)}
    assert draws == set(range(auto.MODE_EVERY[0], auto.MODE_EVERY[1]))


def test_modes_carry_the_auto_card(looks):
    from dtouch.menu import AUTO_ACCENT, AUTO_ID, registry_cards
    card = next(c for c in registry_cards() if c.id == AUTO_ID)
    assert looks["modes"][AUTO_ID] == {
        "id": AUTO_ID, "title": card.title, "key": card.key,
        "blurb": card.blurb, "accent": list(AUTO_ACCENT[::-1])}
    # the page's card order: the two modes, then AUTO
    assert list(looks["modes"]) == ["physarum", "dithergirl", AUTO_ID]


def test_exporter_rerun_is_a_no_op():
    assert dumps_looks() == dumps_looks()
    assert dumps_goldens() == dumps_goldens()
