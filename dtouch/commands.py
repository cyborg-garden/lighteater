"""Command registry — every action is a named, listable, bindable command.

DESIGN.md principle 7: performance actions are commands (`output.blackout`,
`preset.recall.3`), bound to keys through one table so keyboard now — and
MIDI/OSC/pedals later — are bindings, not rewrites. `?` renders `table()` live.

Key routing is case-folded (DESIGN.md §6.2): the table binds lowercase codes and
shifted variants alias unshifted — `cv2.waitKey` reports the character actually
produced, so an uppercase-only binding would silently demand Shift. Binding a
letter therefore routes both cases; non-letter keys bind the exact character
(`?` IS the shifted key — that's its identity, not an alias).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Command:
    name: str                      # "output.blackout" — namespaced, stable
    label: str                     # "Blackout" — for the help overlay / toasts
    key: Optional[str]             # bound character, or None (unbound command)
    run: Callable[[], None]


# waitKey codes that are a REAL keypress but are not printable characters, and
# so would fall outside the 32..126 hint window (DESIGN.md principle 4:
# silence-on-input is a bug).
#
# The four arrows are the first thing anyone tries on a menu or a slider and
# they are deliberately not load-bearing (DESIGN.md §6.2 — their waitKey codes
# are platform-dependent), so they must SAY that rather than do nothing: macOS
# reports them as 63232..63235, which the loop masks with `& 0xFF` to 0..3.
# Enter is 13 (LF 10), Backspace 8, Delete 127.
#
# `waitKey`'s idle code (-1 masked to 255) is deliberately NOT here: no key was
# pressed, so a hint would be a toast nobody asked for, three times a second.
HINTABLE_NONPRINTING = frozenset({0, 1, 2, 3, 8, 10, 13, 127})


def hintable(keycode: int) -> bool:
    """Is this code a keypress an operator would expect an answer to?"""
    return 32 <= keycode <= 126 or keycode in HINTABLE_NONPRINTING


@dataclass
class CommandRegistry:
    _by_name: dict = field(default_factory=dict)
    _by_key: dict = field(default_factory=dict)      # keycode (int) -> Command
    on_unknown: Optional[Callable[[int], None]] = None

    @staticmethod
    def _codes_for(key: str):
        """All waitKey codes a bound character routes from (case-folded)."""
        ch = key.lower()
        codes = [ord(ch)]
        if ch != ch.upper() and len(ch.upper()) == 1:   # letter: shifted aliases
            codes.append(ord(ch.upper()))
        return codes

    def register(self, cmd: Command) -> Command:
        if cmd.name in self._by_name:
            raise ValueError(f"command {cmd.name!r} already registered")
        self._by_name[cmd.name] = cmd
        if cmd.key is not None:
            if len(cmd.key) != 1:
                raise ValueError(f"key binding must be one character: {cmd.key!r}")
            for code in self._codes_for(cmd.key):
                if code in self._by_key:
                    raise ValueError(
                        f"key {chr(code)!r} already bound to {self._by_key[code].name}")
                self._by_key[code] = cmd
        return cmd

    def add(self, name, label, key, run) -> Command:
        return self.register(Command(name, label, key, run))

    def get(self, name: str) -> Optional[Command]:
        return self._by_name.get(name)

    def dispatch(self, keycode: int) -> bool:
        """Route one cv2.waitKey code. Returns True when a command ran.
        Unbound keys go to the unknown-key hook (gentle toast) — printable
        ones and the non-printing keys people actually press (`hintable`)."""
        cmd = self._by_key.get(keycode)
        if cmd is not None:
            cmd.run()
            return True
        if self.on_unknown is not None and hintable(keycode):
            self.on_unknown(keycode)
        return False

    def commands(self):
        """Every registered command, in registration order."""
        return list(self._by_name.values())

    def table(self):
        """(key, label) rows for the help overlay, in registration order."""
        return [(c.key, c.label) for c in self._by_name.values() if c.key is not None]
