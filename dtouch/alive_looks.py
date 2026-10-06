"""Export the alive step's table the browser port needs as JSON.

`dtouch/shaders/alive/alive.json` is vendored verbatim by cyborg-garden-site
next to the alive shaders; dtouch.alive.ALIVE is its single source of truth.

    python -m dtouch.alive_looks        # rewrite alive.json

tests/test_alive_shared.py fails when the file on disk no longer matches
`payload()`, so a change to ALIVE must come with a regenerate.
"""
from __future__ import annotations

import json
import os

from .alive import ALIVE

ALIVE_JSON_PATH = os.path.join(os.path.dirname(__file__), "shaders", "alive", "alive.json")


def payload():
    """The JSON-able dict: the table key for key (the browser's own names,
    so it can stand in for its ALIVE literal)."""
    return {"alive": json.loads(json.dumps(ALIVE))}


def dumps():
    return json.dumps(payload(), indent=1) + "\n"


def write(path=ALIVE_JSON_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps())
    return path


if __name__ == "__main__":
    print("wrote", write())
