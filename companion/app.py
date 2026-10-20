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

## Modes and model selection

- **Text mode** runs `run_text_loop` below; **voice mode** runs
  `companion/voice_app.py`'s `run_voice_session` (real mic capture and
  speaker playback around `Orchestrator.handle_voice_turn`).
  `companion/__main__.py` dispatches on `config.mode`.
- **`llm.voice_mode_override`**: in voice mode, `build_llm` merges these
  options (typically a smaller, faster model) over `llm.options`. Mode is
  fixed per session, so `Orchestrator` still holds a single `llm` and has
  no mode-aware branch.

## Persona selection

Nothing defined how a persona gets chosen for a session, so
`select_persona()` below picks a reasonable default: an explicit
`--persona <id>` CLI arg if given, else the first persona
`PersonaLoader.load_all()` returns (alphabetical by filename, since it
scans `sorted(directory.glob(...))`). The Streamlit UI has its own picker.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from companion.backends.llm.ollama_local import OllamaLLM
from companion.backends.stt.whisper_local import FasterWhisperSTT
from companion.backends.tts.coqui_local import CoquiTTS
from companion.backends.tts.piper_local import PiperTTS
from companion.backends.vad.silero_vad import SileroVAD
from companion.config.loader import AppConfig, BackendConfig
from companion.core.agent import AgentEvent, ToolFinished, ToolStarted
from companion.core.interfaces.types import Persona, Turn
from companion.core.orchestrator import Orchestrator
from companion.core.persona_loader import PersonaLoader
from companion.core.prompt_builder import PromptBuilder
from companion.memory.store import VectorMemoryStore
from companion.memory.summarizer import RollingSummarizer
from companion.tools.registry import validate_persona_tools
from companion.tools.store import ToolStore

if TYPE_CHECKING:
    from companion.core.agent import AgentRunner
    from companion.core.interfaces import (
        LLMBackend,
        MemoryStore,
        SpeechToText,
        TextToSpeech,
        VoiceActivityDetector,
    )

_EXIT_COMMANDS = {"exit", "quit"}
_DEFAULT_TOOL_DATA_DIR = "./data/tools"
REMINDER_POLL_S = 20.0
SUMMARY_TEMPERATURE = 0.2  # see build_summarizer


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
    """In voice mode, `llm.voice_mode_override.options` (typically a
    smaller, faster model) are merged over `llm.options`. Mode is fixed for
    a whole session, so this is decided once here; `Orchestrator` still
    holds a single `llm`. The Streamlit UI's push-to-talk goes through the
    text path, so it uses the base options."""
    section = config.llm
    if config.mode == "voice" and config.llm_voice_override:
        section = BackendConfig(
            backend=section.backend, options={**section.options, **config.llm_voice_override}
        )
    return _build_single_backend(section, "ollama", OllamaLLM, "llm")


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


def build_summarizer(config: AppConfig, memory: MemoryStore) -> RollingSummarizer | None:
    """None when `memory.summarizer.trigger_turn_count` is missing or 0.
    Gets its own LLM instance built from the base `llm.options`, never with
    `voice_mode_override` applied: those options exist to keep spoken
    replies short and would cut a summary off (docs/ROLLING_SUMMARIZER.md
    section 4)."""
    if config.summarizer_trigger_turn_count is None:
        return None
    # A summary replaces the raw turns, so it should be faithful, not
    # lively: always a low temperature, whatever the chat uses.
    section = BackendConfig(
        backend=config.llm.backend,
        options={**config.llm.options, "temperature": SUMMARY_TEMPERATURE},
    )
    llm = _build_single_backend(section, "ollama", OllamaLLM, "llm")
    return RollingSummarizer(
        memory=memory, llm=llm, trigger_turn_count=config.summarizer_trigger_turn_count
    )


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
    persona file in `config.personas_dir` is malformed, or `UnknownToolError`
    if one lists a tool that doesn't exist — not caught here, left to
    propagate with their own already-actionable messages. Tool names are
    checked even with no `agent:` section, since an unknown name is a typo
    either way."""
    personas = PersonaLoader(config.personas_dir).load_all()
    validate_persona_tools(personas.values())
    return personas


def build_agent(config: AppConfig, memory: MemoryStore) -> AgentRunner | None:
    """The agent loop for personas with `tools:`. None when config has no
    `agent:` section, in which case every persona uses the plain LLM."""
    if config.agent is None:
        return None
    if config.agent.backend != "langgraph":
        raise AppSetupError(
            f"agent.backend {config.agent.backend!r} is not implemented — only 'langgraph' is."
        )
    import importlib.util

    missing = [m for m in ("langgraph", "langchain", "langchain_ollama")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise AppSetupError(
            f"the agent: section needs the agents extra ({', '.join(missing)} not installed). "
            'Run: pip install -e ".[agents]", or remove the agent: section.'
        )
    from companion.backends.agent.langgraph_ollama import LangGraphOllamaAgent

    return LangGraphOllamaAgent(memory=memory, **config.agent.options)


def tool_data_dir(config: AppConfig) -> str:
    if config.agent is None:
        return _DEFAULT_TOOL_DATA_DIR
    return config.agent.options.get("data_dir", _DEFAULT_TOOL_DATA_DIR)


def take_due_reminders(
    config: AppConfig, user_id: str, personas: dict[str, Persona]
) -> list[str]:
    """Due reminders for this user, as display lines, each returned exactly
    once. Empty when there's no agent (so no reminders tool) configured."""
    from datetime import datetime

    if config.agent is None:
        return []
    due = ToolStore(tool_data_dir(config)).take_due_reminders(
        user_id, datetime.now().astimezone()
    )
    lines = []
    for reminder in due:
        setter = personas.get(reminder.persona_id)
        source = f" (from {setter.name})" if setter else ""
        lines.append(f"⏰ Reminder{source}: {reminder.text}")
    return lines


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
    memory = build_memory(config)
    return Orchestrator(
        stt=build_stt(config),
        tts=build_tts_for_persona(config, persona),
        llm=build_llm(config),
        vad=build_vad(config),
        memory=memory,
        prompt_builder=PromptBuilder(),
        agent=build_agent(config, memory),
        summarizer=build_summarizer(config, memory),
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
    check_reminders: Callable[[], list[str]] | None = None,
    reminder_poll_s: float = REMINDER_POLL_S,
) -> None:
    """Minimal read/generate/print loop against `handle_text_turn`.

    `check_reminders` (e.g. `take_due_reminders` bound to this config and
    user) is polled by a background task every `reminder_poll_s` seconds,
    so a reminder shows up even while you're idle at the prompt. When a
    persona uses a tool, a short "(using ...)" line is printed.

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

    async def poll_reminders() -> None:
        while True:
            for line in await asyncio.to_thread(check_reminders):
                print_fn(f"\n{line}\n")
            await asyncio.sleep(reminder_poll_s)

    def show_tool_use(event: AgentEvent) -> None:
        if isinstance(event, ToolStarted):
            print_fn(f"  (using {event.name}...)")
        elif isinstance(event, ToolFinished) and not event.ok:
            # Shown even if the model then claims success.
            print_fn(f"  ({event.name} reported a problem: {event.detail or 'unknown error'})")

    poller = asyncio.create_task(poll_reminders()) if check_reminders else None
    try:
        await _text_loop_body(orchestrator, persona, user_id, recent_turns,
                              read_input, print_fn, show_tool_use)
    finally:
        if poller is not None:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller


async def _text_loop_body(
    orchestrator: Orchestrator,
    persona: Persona,
    user_id: str,
    recent_turns: list[Turn],
    read_input: Callable[[str], Awaitable[str]],
    print_fn: Callable[[str], None],
    on_agent_event: Callable[[AgentEvent], None],
) -> None:
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
            user_id=user_id, persona=persona, recent_turns=recent_turns, message=message,
            on_agent_event=on_agent_event,
        )
        print_fn(f"{persona.name}> {reply}\n")

        recent_turns.append(Turn(role="user", text=message))
        if reply.strip():
            recent_turns.append(Turn(role="assistant", text=reply))
