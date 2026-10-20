"""`Orchestrator`: composes the interfaces into conversation turns.
`handle_text_turn` is request/response; `handle_voice_turn` is a whole
session's listening/speaking state machine with barge-in. Both query,
de-duplicate and write memory, and both route personas with tools through
the optional `agent`. Read `orchestrator.py`'s docstrings and
`tests/test_orchestrator_voice_turn.py`'s module docstring before changing
the voice loop's concurrency (one reader per async generator).
"""

from companion.core.orchestrator.orchestrator import Orchestrator

__all__ = ["Orchestrator"]
