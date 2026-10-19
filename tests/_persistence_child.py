"""Child-process helper for tests/test_memory_persistence_across_processes.py.

Each invocation is one short-lived process that does a single thing
against a real on-disk MemoryStore and prints a JSON result to stdout,
then exits. The tests run several of these one after another against the
same directory, so "the second process sees what the first wrote" is a
real process restart, not a second store object in the same interpreter
(which is all tests/test_memark_retrieval.py's
`test_data_survives_a_fresh_store_instance_same_persist_dir` can prove —
module-level caches, open SQLite connections or anything else living in
process memory would make that test pass for the wrong reason).

The store is built through `companion.app.build_memory`, the same call
`python -m companion` makes, so the `memory.backend` config value is what
picks Chroma vs MemArk here too. Both use their real default embedding
function (Chroma's ONNX MiniLM), not a fake.

Not collected by pytest (leading underscore, no `test_` prefix).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from companion.app import build_memory
from companion.config.loader import BackendConfig
from companion.core.interfaces import LLMBackend, SpeechToText, TextToSpeech, Turn
from companion.core.interfaces.types import Persona
from companion.core.interfaces.vad import VoiceActivityDetector
from companion.core.orchestrator import Orchestrator
from companion.core.prompt_builder import PromptBuilder

CANNED_REPLY = "Noted."


def _turn_to_json(turn: Turn) -> dict:
    return {"role": turn.role, "text": turn.text, "timestamp": turn.timestamp.isoformat()}


def _turn_from_json(data: dict) -> Turn:
    return Turn(
        role=data["role"], text=data["text"], timestamp=datetime.fromisoformat(data["timestamp"])
    )


class _RecordingLLM(LLMBackend):
    """Stands in for Ollama so the chat command is deterministic; records
    exactly what PromptBuilder handed it so the test can check whether
    memory written by an earlier process reached the prompt."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Turn]]] = []

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.calls.append((prompt, context))
        yield CANNED_REPLY


def _build_store(args: argparse.Namespace):
    config = SimpleNamespace(
        memory=BackendConfig(backend=args.backend, options={"persist_dir": args.persist_dir})
    )
    return build_memory(config)


async def _run(args: argparse.Namespace) -> dict:
    store = _build_store(args)
    if args.command == "write":
        turns = [_turn_from_json(t) for t in json.loads(args.turns)]
        for turn in turns:
            await store.write(args.user, args.persona, turn)
        return {"written": len(turns)}
    if args.command == "query":
        results = await store.query(args.user, args.persona, args.query, k=args.k)
        return {"results": [_turn_to_json(t) for t in results]}
    if args.command == "delete":
        await store.delete(args.user, args.persona)
        return {"deleted": True}
    if args.command in ("add-note", "search-notes"):
        # The notes tool, built the way the agent builds it, against a
        # tools data dir inside persist_dir.
        from pathlib import Path

        from companion.tools.base import ToolContext
        from companion.tools.registry import build_persona_tools

        persona = Persona(id=args.persona, name="P", tone="t", boundaries="b",
                          voice_id="piper:unused", tools=["notes"])
        ctx = ToolContext(user_id=args.user, persona_id=args.persona, memory=store,
                          data_dir=Path(args.persist_dir) / "tools")
        tools = {t.name: t.func for t in build_persona_tools(persona, ctx, allow_network=False)}
        if args.command == "add-note":
            return {"result": await tools["add_note"](text=args.message)}
        return {"result": await tools["search_notes"](query=args.query)}
    if args.command == "chat":
        llm = _RecordingLLM()
        orchestrator = Orchestrator(
            stt=Mock(spec=SpeechToText),
            tts=Mock(spec=TextToSpeech),
            llm=llm,
            vad=Mock(spec=VoiceActivityDetector),
            memory=store,
            prompt_builder=PromptBuilder(),
        )
        persona = Persona(
            id=args.persona,
            name="Test Persona",
            tone="Plain.",
            boundaries="None.",
            voice_id="piper:unused",
        )
        # A fresh process has no live conversation: recent_turns starts
        # empty, exactly like the first turn after `python -m companion`
        # restarts. Anything about earlier sessions can only come from disk.
        reply = await orchestrator.handle_text_turn(
            user_id=args.user, persona=persona, recent_turns=[], message=args.message
        )
        prompt, context = llm.calls[0]
        return {
            "reply": reply,
            "prompt": prompt,
            "context": [_turn_to_json(t) for t in context],
        }
    raise ValueError(f"unknown command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=["write", "query", "delete", "chat", "add-note", "search-notes"])
    parser.add_argument("--backend", required=True, choices=["chroma", "memark"])
    parser.add_argument("--persist-dir", required=True)
    parser.add_argument("--user", default="u1")
    parser.add_argument("--persona", default="p1")
    parser.add_argument("--turns", default="[]", help="JSON list of turns, for write")
    parser.add_argument("--query", default="")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--message", default="")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(_run(args))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
