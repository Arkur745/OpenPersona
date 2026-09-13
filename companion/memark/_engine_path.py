"""Puts `companion/memark/_vendor/` on `sys.path` so the vendored
MemEngine files' own absolute imports (e.g. `from engine.core
.memory_node import MemoryNode`) resolve — see `_vendor/PROVENANCE.md`
for exactly what's vendored and why. Call `ensure_engine_importable()`
before importing anything from `engine.*`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).parent / "_vendor"


def ensure_engine_importable() -> None:
    path_str = str(_VENDOR_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
