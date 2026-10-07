"""Boot to the HOME MENU — DESIGN.md §3 (amended 2026-08-15, the user's call).

The app used to launch straight into the last-used mode. It now opens on the
menu, with that mode running live behind it and already selected, so Enter is a
one-key resume. An explicit mode (--mode / --still / an engine flag) still goes
straight in — a flag means "boot into", and a flag that has to wait through a
menu is a worse launch than we had.

Everything here drives the real Host loop headless with DitherGirlMode (pure
numpy/cv2, no GL), reached through state.json exactly the way a real launch
reaches it.
"""
import numpy as np
import pytest

from dtouch import presets
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.shell import Host

RES = (192, 108)

DOORS_HINT = "m menu - TAB panel - ? keys"


def _gl_available():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


# Resuming Particles builds its GL engines; the rest of this file resumes
# Dither Girl (pure numpy/cv2), same split test_shell.py uses.
_gl = pytest.mark.skipif(not _gl_available(),
                         reason="no GL context available (CI)")


class SyntheticSource:
    def __init__(self, w=64, h=36):
        self.reads = 0
        self.w, self.h = w, h
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        rng = np.random.default_rng(self.reads)
        return True, rng.integers(0, 256, (self.h, self.w, 3), np.uint8)

    def release(self):
        pass


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _last_used(tmp_path, mode_id):
    p = _paths(tmp_path)
    presets.save_state({"mode": mode_id, "preset": None, "bank": {}},
                       path=p["state_path"])
    return p


def _booted(tmp_path, mode="NO-MODE", last_used="dithergirl", **kw):
    """Run one frame of the real loop. Default: no explicit mode (the launch
    the user actually types) with `last_used` remembered in state.json."""
    p = _last_used(tmp_path, last_used)
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    kw.setdefault("max_frames", 1)
    host = Host(None if mode == "NO-MODE" else mode,
                source=SyntheticSource(), **p, **kw)
    host.run()
    return host


def _hints(host):
    return [t.text for t in host.hud.toasts._hints]


def _sel_id(host):
    return host.menu.cards[host.menu.sel].id


# ---------- boots to the menu ----------

def test_launch_with_no_mode_opens_the_home_menu(tmp_path):
    host = _booted(tmp_path)
    assert host.menu.open is True
    assert host.menu.boot is True


def test_the_last_used_mode_runs_live_behind_the_menu(tmp_path):
    """The menu is not a dead screen (DESIGN.md §3): the mode is started and
    stepping behind it, so the camera-backed picture is already live."""
    host = _booted(tmp_path)
    assert host.mode.id == "dithergirl"


def test_menu_selection_defaults_to_the_last_used_mode(tmp_path):
    """Enter is a one-key resume — the pre-selected card is the last mode."""
    assert _sel_id(_booted(tmp_path, last_used="dithergirl")) == "dithergirl"


@_gl
def test_menu_selection_follows_whichever_mode_was_last_used(tmp_path):
    assert _sel_id(_booted(tmp_path, last_used="particles")) == "particles"


def test_first_run_with_no_state_resolves_particles(tmp_path):
    """No state.json at all (first ever launch): Particles is what boots
    behind the menu, so Particles is what the menu opens on."""
    from dtouch.modes.particles import ParticlesMode

    host = Host(None, source=SyntheticSource(), res=RES, show=False,
                **_paths(tmp_path))
    assert host._boot_menu is True
    assert isinstance(host._resolve_boot_mode(), ParticlesMode)


# ---------- an explicit mode skips it ----------

def test_explicit_mode_boots_straight_in(tmp_path):
    """--mode / --still / an engine flag: no menu, no waiting."""
    host = _booted(tmp_path, mode=DitherGirlMode(), last_used="particles")
    assert host.menu.open is False
    assert host.menu.boot is False
    assert host.mode.id == "dithergirl"


def test_explicit_mode_still_gets_the_three_doors_hint(tmp_path):
    host = _booted(tmp_path, mode=DitherGirlMode())
    assert DOORS_HINT in _hints(host)


# ---------- the boot hint says what Enter does, and stays off the menu's line ----------

def test_boot_menu_hint_names_the_resume_not_the_doors(tmp_path):
    """The menu already prints its own key line along the bottom; posting the
    doors hint there would say the wrong thing in the wrong place (and, before
    issue #18, on the very same baseline)."""
    host = _booted(tmp_path)
    hints = _hints(host)
    assert "enter resumes DITHER" in hints
    assert DOORS_HINT not in hints


def test_boot_hint_keeps_the_4s_ttl(tmp_path):
    host = _booted(tmp_path)
    hint = next(t for t in host.hud.toasts._hints
                if t.text.startswith("enter resumes"))
    assert hint.ttl == 4.0


def test_leaving_the_boot_menu_posts_the_doors_hint(tmp_path):
    """The three doors belong to the MODE, so the hint is posted as the menu
    is left — where it is both true and unobstructed."""
    host = _booted(tmp_path)
    host._route_key(13)                       # Enter
    assert DOORS_HINT in _hints(host)


def test_leaving_the_boot_menu_retires_the_resume_hint(tmp_path):
    """A hint is a pointer, and 'enter resumes X' points at Enter. It carries
    a 4 s ttl and Enter usually lands inside one, so it used to survive the
    commit and stack ABOVE the hint that replaces it — two sentences, one of
    them about a menu the operator has already left."""
    from dtouch.shell import RESUME_HINT

    host = _booted(tmp_path)
    assert any(t.startswith(RESUME_HINT) for t in _hints(host))
    host._route_key(13)                       # Enter
    assert not any(t.startswith(RESUME_HINT) for t in _hints(host))
    assert _hints(host) == [DOORS_HINT]


# ---------- Esc at boot enters the selection ----------

def test_esc_at_boot_enters_the_selected_mode(tmp_path):
    """DESIGN.md §3 amended: Esc 'returns to the running mode' — at boot there
    is nothing behind the menu to return to, so it enters the selection rather
    than dropping the operator on a mode they never chose."""
    host = _booted(tmp_path)
    host._route_key(27)
    assert host.menu.open is False
    assert host.pending_mode is None          # already running: not a switch
    assert host.mode.id == "dithergirl"


def test_esc_at_boot_after_moving_enters_what_is_selected(tmp_path):
    host = _booted(tmp_path, last_used="dithergirl")
    host._route_key(ord(","))                 # move to the other card
    picked = _sel_id(host)
    assert picked != "dithergirl"
    host._route_key(27)
    assert host.menu.open is False
    assert host.pending_mode == picked


def test_m_at_boot_enters_too(tmp_path):
    host = _booted(tmp_path)
    host._route_key(ord("m"))
    assert host.menu.open is False


def test_esc_after_boot_still_returns_untouched(tmp_path):
    """Only the BOOT menu commits on Esc. Reopened mid-set, Esc is 'back'."""
    host = _booted(tmp_path)
    t = [100.0]
    host.scene_clock = lambda: t[0]
    host._route_key(13)                       # leave the boot menu
    host.menu.show(host.mode.id)              # reopen it — not a boot menu
    assert host.menu.open is True and host.menu.boot is False
    host.pending_mode = None
    t[0] += 0.6          # a person's pause: the commit took the flash budget
    host._route_key(27)
    assert host.menu.open is False
    assert host.pending_mode is None          # nothing switched


# ---------- entering the pre-selected mode is not "already in X" ----------

def test_enter_on_the_running_mode_does_not_scold(tmp_path):
    """`request_mode` hints 'already in X' for a same-mode switch, which is
    right mid-set and wrong on the one card the boot menu pre-selected."""
    host = _booted(tmp_path)
    host._route_key(13)
    assert not any("already in" in h for h in _hints(host))


def test_entering_the_pre_selected_mode_flashes_its_title(tmp_path):
    """Silence-on-input is a bug (principle 4): entering gets the same
    mode-title flash a real switch gets."""
    host = _booted(tmp_path)
    host._route_key(13)
    assert host.hud.toasts._center is not None
    assert host.hud.toasts._center.text == "Dither"


def test_picking_a_different_card_at_boot_switches(tmp_path):
    host = _booted(tmp_path, last_used="dithergirl")
    host._route_key(ord("p"))                 # Particles: letter selects-and-enters
    assert host.menu.open is False
    assert host.pending_mode == "particles"


def test_boot_menu_click_commits_the_same_way(tmp_path):
    """The mouse path and the key path are one implementation (principle 7)."""
    import cv2

    host = _booted(tmp_path)
    host.menu.rects = [((0, 0, 10, 10), host.menu.cards[host.menu.sel])]
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert host.menu.open is False
    assert host.pending_mode is None          # entering, not switching
    assert DOORS_HINT in _hints(host)


def test_q_in_the_boot_menu_still_arms_the_quit_confirm(tmp_path):
    host = _booted(tmp_path)
    host._route_key(ord("q"))
    assert host.menu.open is False
    assert host.ps.quit is False              # first press only arms it
    assert any("quit" in h for h in _hints(host))
