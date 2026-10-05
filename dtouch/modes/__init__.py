"""Mode plugins — modes rack into the shell that owns the show (DESIGN.md §2).

A mode owns its engine objects (sim + renderer), its per-frame image, its panel
sections, its mode-local commands, and its defaults + built-in looks. The shell
(dtouch.shell.Host) owns everything else. The protocol (DESIGN.md §2.2):

    class Mode(Protocol):
        id: str              # "particles" — preset + command namespace
        title: str           # "Particles" — menu + toast display
        accent: tuple        # per-mode accent hue (BGR)
        accepts_still: bool

        def start(self, host) -> None: ...   # build engines; may raise — shell
                                             # catches, toasts, reinstates previous
        def stop(self) -> None: ...          # release GL/matte/etc; idempotent
        def panel_spec(self) -> list: ...    # declared controls; shell renders +
                                             # appends SIGNAL rack + global rows
        def commands(self) -> dict: ...      # mode-local named commands
        def step(self, frame_bgr, audio, dt): ...  # sim+render -> RGB at host.res
        def safe_look(self) -> str: ...      # the panic target
        def status_tail(self, cam_name) -> str: ...  # cam/src tail of the HUD
                                             # status (body is spec-derived,
                                             # DESIGN.md §2.3)
        def on_resize(self, w, h) -> None: ...

Modes register in REGISTRY (graft from Contract): one import + one line adds a
mode to the menu, key table, and preset store.
"""
from .bender import BenderMode
from .dithergirl import DitherGirlMode
from .particles import ParticlesMode
from .physarum import PhysarumMode

REGISTRY = [ParticlesMode, DitherGirlMode, PhysarumMode, BenderMode]


def mode_by_id(mode_id):
    """The registered mode class for `mode_id`, or None."""
    return next((m for m in REGISTRY if m.id == mode_id), None)
