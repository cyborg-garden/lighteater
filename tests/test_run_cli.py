"""Boot-precedence pins for the launcher (experiments/05-live-webcam/run.py).

DESIGN.md §3: CLI flags mean "boot into". The launcher's own docstring
advertises `--matte person`, so an engine flag must reach the constructed mode
instead of being swallowed by the state.json resume path (which builds the mode
with DEFAULT args). Precedence: --mode > --still > engine flags > state.json.
"""
import importlib.util
import os
import sys

import pytest

RUN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments", "05-live-webcam", "run.py")


def _load_run():
    spec = importlib.util.spec_from_file_location("dtouch_run_cli", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load_run()


class FakeMode:
    """Records the args a boot would have constructed the mode with."""

    def __init__(self, **kw):
        self.kw = kw


class FakeHost:
    made = None

    def __init__(self, mode, **kw):
        self.mode, self.kw = mode, kw
        FakeHost.made = self

    def run(self):
        self.ran = True


@pytest.fixture
def launch(monkeypatch):
    """main() with the heavy bits stubbed; returns the FakeHost it built."""
    monkeypatch.setattr(run, "Host", FakeHost)
    monkeypatch.setattr(run, "ParticlesMode", FakeMode)
    monkeypatch.setattr(run, "DitherGirlMode", FakeMode)
    monkeypatch.setattr(run, "live", lambda **kw: None)

    def go(*argv):
        FakeHost.made = None
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        run.main()
        return FakeHost.made

    return go


# ---------- engine flags reach the mode ----------

def test_matte_flag_reaches_the_mode_without_a_mode_flag(launch):
    """`python run.py --matte person` — the launcher's own documented example."""
    host = launch("--matte", "person")
    assert host.mode.kw["matte"] == "person"


def test_grid_and_particles_flags_reach_the_mode(launch):
    host = launch("--grid", "200x100", "--particles", "1234")
    assert host.mode.kw["grid"] == (200, 100)
    assert host.mode.kw["n"] == 1234


def test_flock_and_glitch_still_boot_flow(launch):
    host = launch("--flock", "--glitch")
    assert host.mode.kw["flock"] is True
    assert host.mode.kw["glitch"] is True


# ---------- state.json is only consulted with no mode-implying flag ----------

def test_engine_flags_skip_the_state_json_resume(launch):
    """A constructed mode (not None) means the shell never resolves from
    state.json — the resume path is what dropped the flags."""
    assert launch("--matte", "motion").mode is not None
    assert run.boot_mode_name(None, None, particles_flags=True) == "flow"


def test_no_flags_at_all_still_resumes_from_state_json(launch):
    """Defaults must NOT count as 'given' — bare `run.py` still resumes."""
    assert launch().mode is None
    assert run.boot_mode_name(None, None, particles_flags=False) is None


def test_engine_flags_given_reports_only_non_default_values():
    ap_args = type("A", (), dict(run.ENGINE_DEFAULTS))()
    assert run.engine_flags_given(ap_args) == []
    ap_args.matte = "person"
    ap_args.flock = True
    assert run.engine_flags_given(ap_args) == ["--matte", "--flock"]


# ---------- precedence: --mode > --still > engine flags ----------

def test_still_wins_over_an_engine_flag_and_says_so(launch, capsys, tmp_path):
    still = str(tmp_path / "x.jpg")
    host = launch("--still", still, "--flock")
    out = capsys.readouterr().out
    assert "--flock" in out and "ignored" in out and "dithergirl" in out
    assert host.mode.kw == {"still": True}          # DitherGirlMode, not flow
    assert host.kw["still"] == still


def test_mode_flag_wins_over_still_and_engine_flags(launch, tmp_path):
    still = str(tmp_path / "x.jpg")
    host = launch("--mode", "flow", "--still", still, "--matte", "person")
    assert host.mode.kw["matte"] == "person"


def test_explicit_dithergirl_mode_warns_about_dropped_engine_flags(launch,
                                                                  capsys):
    launch("--mode", "dithergirl", "--matte", "person")
    assert "ignored" in capsys.readouterr().out


# ---------- --preset resolves against a mode's real looks ----------
# `python run.py --mode dithergirl` once, quit, then `python run.py --preset
# embers` used to boot Dither Girl on `classic` and say nothing: the flag was
# swallowed by the state.json resume path, forever, for that install. On main
# --preset always worked.


@pytest.fixture
def empty_store(tmp_path, monkeypatch):
    """Resolve against the built-ins only, not whatever the repo's own
    presets.json happens to hold."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_preset_boots_the_mode_that_owns_the_look(launch, empty_store):
    """No mode-implying flag: the NAME is the request. `embers` is a Particles
    built-in, so Particles boots — regardless of what state.json remembers."""
    host = launch("--preset", "embers")
    assert host.mode is not None                    # not the state.json resume
    assert "matte" in host.mode.kw                  # ParticlesMode's signature
    assert host.kw["preset"] == "embers"


def test_preset_boots_dither_girl_for_a_dither_girl_look(launch, empty_store):
    host = launch("--preset", "newsprint")
    assert host.mode.kw == {"still": False}         # DitherGirlMode
    assert host.kw["preset"] == "newsprint"


def test_a_mode_flag_still_wins_but_an_alien_preset_is_called_out(
        launch, empty_store, capsys):
    """Precedence is unchanged (--mode > --still > engine flags > --preset).
    What changes is that the preset is not silently dropped onto index 0."""
    host = launch("--mode", "dithergirl", "--preset", "embers")
    out = capsys.readouterr().out
    assert 'preset "embers" is not a Dither look' in out
    assert "booting menu" in out
    assert host.kw["preset"] == "menu"              # the mode's safe look


def test_a_preset_the_booted_mode_owns_passes_through_quietly(
        launch, empty_store, capsys):
    host = launch("--mode", "dithergirl", "--preset", "phosphor")
    assert host.kw["preset"] == "phosphor"
    assert "not a" not in capsys.readouterr().out


def test_an_unknown_preset_says_so_instead_of_booting_index_zero(
        launch, empty_store, capsys):
    host = launch("--preset", "nope-not-a-look")
    assert 'no preset named "nope-not-a-look"' in capsys.readouterr().out
    assert host.kw["preset"] is None                # normal resume, not index 0
    assert host.mode is None


def test_a_name_both_modes_own_names_the_one_it_picked(launch, empty_store,
                                                       capsys):
    """A look name can legitimately belong to either mode — the reason
    --preset cannot just be bolted onto ENGINE_DEFAULTS."""
    from dtouch import presets

    presets.save("twin", {"fade": 0.9}, path="presets.json", mode="particles")
    presets.save("twin", {"scale": 90.0}, path="presets.json",
                 mode="dithergirl")
    host = launch("--preset", "twin")
    out = capsys.readouterr().out
    assert "Particles and Dither" in out and "booting Particles" in out
    assert host.kw["preset"] == "twin"


def test_preset_owners_reads_builtins_and_saved_looks(empty_store):
    from dtouch import presets

    assert run.preset_owners("embers") == ["particles"]
    assert run.preset_owners("newsprint") == ["dithergirl"]
    assert run.preset_owners("nothing-here") == []
    presets.save("mine", {"fade": 0.9}, path="presets.json", mode="dithergirl")
    assert run.preset_owners("mine") == ["dithergirl"]


def test_mode_bender_boots_circuit_bender(launch, monkeypatch):
    monkeypatch.setattr(run, "BenderMode", FakeMode)
    host = launch("--mode", "bender")
    assert isinstance(host.mode, FakeMode)


def test_a_bender_look_boots_circuit_bender(launch, empty_store, monkeypatch):
    """A preset only Circuit Bender owns resolves to it (MODE_ARG must know
    the mode, or the lookup raises KeyError at launch)."""
    assert run.preset_owners("bent cam") == ["bender"]
    host = launch("--preset", "bent cam")
    assert type(host.mode).__name__ == "BenderMode"
    assert host.kw["preset"] == "bent cam"
    assert run.resolve_preset("bent cam", None) == ("bender", "bent cam")


def test_grid_mode_says_the_preset_is_ignored(launch, empty_store, capsys):
    launch("--mode", "grid", "--preset", "embers")
    assert "--preset embers ignored" in capsys.readouterr().out


# ---------- --no-mirror reaches the shell ----------

def test_no_mirror_reaches_the_host(launch):
    assert launch("--no-mirror").kw["mirror"] is False
    assert launch().kw["mirror"] is True


def test_no_mirror_reaches_the_legacy_grid_path(launch, monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "live", lambda **kw: seen.update(kw))
    launch("--mode", "grid", "--no-mirror")
    assert seen["mirror"] is False


# ---------- the module docstring advertises the keys that exist ----------

def test_docstring_does_not_advertise_retired_keys():
    """It used to promise "space freeze". `space` is output.blackout, and
    blackout is written INTO the recording — a user following the docstring
    mid-take records black."""
    doc = run.__doc__
    assert "space freeze" not in doc
    assert "n cycle matte" not in doc
    assert "m mirror" not in doc                    # `m` is the menu
    assert "?" in doc and "BLACKOUT" in doc
    assert "recorded" in doc                        # the dangerous part, named
