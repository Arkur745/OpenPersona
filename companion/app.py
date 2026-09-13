"""Application bootstrap: reads config, constructs real backend
implementations per the config-driven selection pattern (see
config/config.example.yaml's own header), and wires them into an
`Orchestrator` ready to run — plus the text-mode conversation loop
itself. `companion/__main__.py` is a thin CLI shim over this module;
everything testable lives here.

## Where backend-name branching is allowed

CLAUDE.md's architecture rule is: "if you're about to write
`if backend_name == "piper":` anywhere outside the code that reads
config.example.yaml, stop — that branching belongs here." This module
*is* that code — `build_stt`/`build_llm`/`build_vad` reject any backend
name other than the one real implementation each category has (there is
no second class to dispatch to yet, e.g. `whisper_cpp` doesn't exist as
a class at all); `build_tts_for_persona` branches for real between
`piper`/`coqui` since both are genuinely implemented and their
`from_voice_id()` signatures differ (`voices_dir` vs. `model_name`); and
`build_memory` now branches for real too, between `chroma`
(`VectorMemoryStore`) and `memark` (`MemArkMemoryStore`,
`companion/memark/retrieval.py`) — see that function's own comment for
why `memark`'s import is deferred into its branch rather than living
alongside the other backend imports at module top.

## What this does NOT wire up

- **Voice mode.** `mode: voice` in config is recognized and rejected with
  a clear message (see `companion/__main__.py`) rather than silently
  running text mode instead — this pass only drives `handle_text_turn`.
  `Orchestrator.handle_voice_turn` itself is already real (build-order
  step 6); what a voice-mode CLI loop would still need beyond what exists
  today: an actual mic-capture `AsyncIterator[bytes]` source and an audio
  *playback* sink for the bytes `handle_voice_turn` yields. Neither
  exists yet, and this module doesn't invent them.
- **`llm.voice_mode_override`** (config.example.yaml). This key predates
  this bootstrap pass and describes a mode-based model swap that
  `Orchestrator` has never actually implemented — there is no mode-aware
  branch inside `Orchestrator` itself, only one `llm: LLMBackend`
  reference used uniformly by both `handle_text_turn` and
  `handle_voice_turn`. Text mode (this pass) never needs it, so it's
  parsed through by the config loader (kept, not rejected as unknown)
  but not read or acted on here. Flagged explicitly rather than quietly
  implementing mode-switching logic inside `Orchestrator` that wasn't
  asked for.
- **Persona selection UI.** Nothing before this pass defined how a
  persona gets chosen for a session. `select_persona()` below picks a
  reasonable default — an explicit `--persona <id>` CLI arg if given,
  else the first persona `PersonaLoader.load_all()` returns (which is
  alphabetical by filename, since it scans `sorted(directory.glob(...))`)
  — rather than inventing a selection UI that wasn't requested.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from companion.backends.llm.ollama_local import OllamaLLM
from companion.backends.stt.whisper_local import FasterWhisperSTT
from companion.backends.tts.coqui_local import CoquiTTS
from companion.backends.tts.piper_local import PiperTTS
from companion.backends.vad.silero_vad import SileroVAD
from companion.config.loader import AppConfig, BackendConfig
from companion.core.interfaces.types import Persona, Turn
from companion.core.orchestrator import Orchestrator
from companion.core.persona_loader import PersonaLoader
from companion.core.prompt_builder import PromptBuilder
from companion.memory.store import VectorMemoryStore

if TYPE_CHECKING:
    from companion.core.interfaces import (
        LLMBackend,
        MemoryStore,
        SpeechToText,
        TextToSpeech,
        VoiceActivityDetector,
    )

_EXIT_COMMANDS = {"exit", "quit"}


class AppSetupError(RuntimeError):
    """Something about constructing the app from config failed in a way
    that isn't already one of the more specific errors a lower layer
    raises (`ConfigError`, `PersonaError`, or a backend's own
    connection/model error, e.g. `OllamaConnectionError`)."""


def _build_single_backend(section: BackendConfig, expected_backend: str, cls, category: str):
    if section.backend != expected_backend:
        raise AppSetupError(
            f"{category}.backend {section.backend!r} is not implemented — "
            f"only {expected_backend!r} is."
        )
    return cls(**section.options)


def build_stt(config: AppConfig) -> SpeechToText:
    return _build_single_backend(config.stt, "faster_whisper", FasterWhisperSTT, "stt")


def build_llm(config: AppConfig) -> LLMBackend:
    return _build_single_backend(config.llm, "ollama", OllamaLLM, "llm")


def build_vad(config: AppConfig) -> VoiceActivityDetector:
    return _build_single_backend(config.vad, "silero", SileroVAD, "vad")


def build_memory(config: AppConfig) -> MemoryStore:
    if config.memory.backend == "memark":
        # Importing companion.memark.retrieval itself requires
        # networkx/faiss (it imports the vendored engine.* modules at
        # module level, unlike FasterWhisperSTT/PiperTTS/SileroVAD, whose
        # heavy deps stay lazy *inside* methods so their own modules
        # import fine with nothing installed — see e.g.
        # companion/backends/vad/silero_vad.py's module docstring on why
        # torch is deferred that way). Importing MemArkMemoryStore here,
        # inside this branch, rather than unconditionally alongside the
        # other backend classes above, means selecting "chroma" never
        # needs the memory-memark extra installed at all — and selecting
        # "memark" without it fails right here, at startup, with the same
        # unwrapped ImportError every other optional-extra-gated backend
        # in this project surfaces on its own first real use. Install:
        # pip install "companion[memory-memark]".
        from companion.memark.retrieval import MemArkMemoryStore

        return MemArkMemoryStore(**config.memory.options)
    # VectorMemoryStore's own constructor already raises NotImplementedError
    # for anything but "chroma" (see companion/memory/store.py) — reused
    # rather than duplicating that check here.
    return VectorMemoryStore(backend=config.memory.backend, **config.memory.options)


def build_tts_for_persona(config: AppConfig, persona: Persona) -> TextToSpeech:
    """TTS is the one backend category that's persona-bound, not global —
    a `TextToSpeech` instance is bound to one voice for its lifetime,
    resolved via that backend's own `from_voice_id()` classmethod (see
    docs/ARCHITECTURE.md's "Persona voice binding" note).
    """
    options = dict(config.tts.options)
    if config.tts.backend == "piper":
        options.pop("voice_model_path", None)  # from_voice_id sets this itself
        return PiperTTS.from_voice_id(persona.voice_id, config.tts.voices_dir, **options)
    if config.tts.backend == "coqui":
        model_name = options.pop("model_name", None)
        if not model_name:
            raise AppSetupError("tts.options.model_name is required for backend: coqui")
        options.pop("speaker", None)  # from_voice_id sets this itself from voice_id
        return CoquiTTS.from_voice_id(persona.voice_id, model_name=model_name, **options)
    raise AppSetupError(
        f"tts.backend {config.tts.backend!r} is not implemented — only "
        f"'piper' and 'coqui' are."
    )


def load_personas(config: AppConfig) -> dict[str, Persona]:
    """Raises `PersonaError` (from companion.core.persona_loader) if any
    persona file in `config.personas_dir` is malformed — not caught here,
    left to propagate with its own already-actionable message."""
    return PersonaLoader(config.personas_dir).load_all()


def select_persona(personas: dict[str, Persona], requested_id: str | None) -> Persona:
    if not personas:
        raise AppSetupError(
            "No personas found — check personas.directory in your config."
        )
    if requested_id is not None:
        if requested_id not in personas:
            available = ", ".join(sorted(personas))
            raise AppSetupError(
                f"No persona with id {requested_id!r}. Available: {available}"
            )
        return personas[requested_id]
    return next(iter(personas.values()))


def build_orchestrator(config: AppConfig, persona: Persona) -> Orchestrator:
    return Orchestrator(
        stt=build_stt(config),
        tts=build_tts_for_persona(config, persona),
        llm=build_llm(config),
        vad=build_vad(config),
        memory=build_memory(config),
        prompt_builder=PromptBuilder(),
    )


async def _default_read_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run_text_loop(
    orchestrator: Orchestrator,
    persona: Persona,
    user_id: str,
    *,
    read_input: Callable[[str], Awaitable[str]] = _default_read_input,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Minimal read/generate/print loop against `handle_text_turn`.

    `read_input`/`print_fn` are injectable so this loop's mechanics
    (exit commands, blank-input handling, `recent_turns` accumulation)
    are testable without real stdin/stdout — see tests/test_app.py.
    Real usage (`companion/__main__.py`) uses the defaults: `input()`
    bridged onto a thread (so it doesn't block the event loop, matching
    how blocking calls are bridged elsewhere in this codebase — see
    `backends/tts/_sync_stream.py`) and plain `print()`.
    """
    print_fn(f"Talking to {persona.name} ({persona.id}). Type 'exit' to quit.\n")
    recent_turns: list[Turn] = []
    while True:
        try:
            message = await read_input("you> ")
        except EOFError:
            break
        message = message.strip()
        if not message:
            continue
        if message.lower() in _EXIT_COMMANDS:
            break

        reply = await orchestrator.handle_text_turn(
            user_id=user_id, persona=persona, recent_turns=recent_turns, message=message
        )
        print_fn(f"{persona.name}> {reply}\n")

        recent_turns.append(Turn(role="user", text=message))
        if reply.strip():
            recent_turns.append(Turn(role="assistant", text=reply))
