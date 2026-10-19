"""Tests for companion/app.py — the bootstrap that wires real backends
together, plus the text-mode conversation loop.

Split in two halves, matching what's actually being checked:

1. `build_*`/`select_persona`/`load_personas` construct the *real*
   backend classes with the right arguments from a given config/persona
   set — safe to assert on directly (no backend does network/model I/O
   in its own `__init__`, verified in each backend's own module), without
   re-testing what each backend's methods actually do (that's each
   backend's own test file's job).
2. `run_text_loop`'s mechanics (exit commands, blank-input handling,
   `recent_turns` accumulation, an interruption-free multi-turn run) are
   tested against fakes via injectable `read_input`/`print_fn`, plus one
   fully-wired integration test building a real `Orchestrator` from a
   real config file + real persona file + real `PromptBuilder`, with only
   the LLM/MemoryStore faked (not practical to run for real in CI — see
   the gated `test_real_end_to_end_text_conversation_smoke` at the bottom
   for the version of this that *does* use real Ollama + real Chroma,
   skipped cleanly if either isn't available).
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
import yaml

from companion.app import (
    AppSetupError,
    build_llm,
    build_memory,
    build_orchestrator,
    build_stt,
    build_tts_for_persona,
    build_vad,
    load_personas,
    run_text_loop,
    select_persona,
)
from companion.backends.llm.ollama_local import OllamaLLM
from companion.backends.stt.whisper_local import FasterWhisperSTT
from companion.backends.tts.coqui_local import CoquiTTS
from companion.backends.tts.piper_local import PiperTTS
from companion.backends.vad.silero_vad import SileroVAD
from companion.config.loader import AppConfig, BackendConfig, TTSConfig, load_config
from companion.core.interfaces import LLMBackend, MemoryStore, Turn
from companion.core.interfaces.types import Persona
from companion.core.orchestrator import Orchestrator
from companion.core.prompt_builder import PromptBuilder
from companion.memory.store import VectorMemoryStore

VALID_PERSONA_YAML = {
    "id": "mentor",
    "name": "The Old Mentor",
    "tone": "Calm.",
    "boundaries": "Not a therapist.",
    "voice_id": "piper:en_US-mentor-medium",
}


def _persona(**overrides) -> Persona:
    defaults = {
        "id": "mentor",
        "name": "The Old Mentor",
        "tone": "Calm.",
        "boundaries": "Not a therapist.",
        "voice_id": "piper:en_US-mentor-medium",
    }
    defaults.update(overrides)
    return Persona(**defaults)


def _config(**overrides) -> AppConfig:
    defaults = {
        "personas_dir": "./companion/personas",
        "stt": BackendConfig(backend="faster_whisper", options={"model_size": "tiny"}),
        "tts": TTSConfig(
            backend="piper",
            voices_dir="/tmp/voices",
            options={"voice_model_path": "/tmp/voices/default.onnx", "use_cuda": False},
        ),
        "llm": BackendConfig(backend="ollama", options={"model": "qwen2.5:7b-instruct"}),
        "vad": BackendConfig(backend="silero", options={"threshold": 0.6}),
        "memory": BackendConfig(backend="chroma", options={"persist_dir": "/tmp/mem"}),
        "audio": BackendConfig(backend="sounddevice", options={"sample_rate": 16000}),
        "mode": "text",
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


# --- build_* construct the right real class, safely (no I/O in __init__) ---


def test_build_stt_constructs_faster_whisper_with_options():
    stt = build_stt(_config())

    assert isinstance(stt, FasterWhisperSTT)
    assert stt.model_size == "tiny"


def test_build_stt_rejects_unimplemented_backend():
    with pytest.raises(AppSetupError, match="whisper_cpp"):
        build_stt(_config(stt=BackendConfig(backend="whisper_cpp", options={})))


def test_build_llm_constructs_ollama_with_options():
    llm = build_llm(_config())

    assert isinstance(llm, OllamaLLM)
    assert llm.model == "qwen2.5:7b-instruct"


def test_build_llm_applies_voice_override_only_in_voice_mode():
    override = {"model": "qwen2.5:3b-instruct", "temperature": 0.2}

    text_llm = build_llm(_config(mode="text", llm_voice_override=override))
    voice_llm = build_llm(_config(mode="voice", llm_voice_override=override))

    assert (text_llm.model, text_llm.temperature) == ("qwen2.5:7b-instruct", 0.7)
    assert (voice_llm.model, voice_llm.temperature) == ("qwen2.5:3b-instruct", 0.2)


def test_build_llm_in_voice_mode_without_override_uses_base_options():
    llm = build_llm(_config(mode="voice"))

    assert llm.model == "qwen2.5:7b-instruct"


def test_build_llm_rejects_unimplemented_backend():
    with pytest.raises(AppSetupError, match="gpt4all"):
        build_llm(_config(llm=BackendConfig(backend="gpt4all", options={})))


def test_build_vad_constructs_silero_with_options():
    vad = build_vad(_config())

    assert isinstance(vad, SileroVAD)
    assert vad.threshold == 0.6


def test_build_memory_constructs_chroma_backed_store():
    memory = build_memory(_config())

    assert isinstance(memory, VectorMemoryStore)
    assert memory.persist_dir == "/tmp/mem"


def test_build_memory_rejects_unimplemented_backend_via_stores_own_check():
    # VectorMemoryStore's own constructor raises this — build_memory
    # deliberately doesn't duplicate the check, just reuses it.
    with pytest.raises(NotImplementedError, match="lancedb"):
        build_memory(_config(memory=BackendConfig(backend="lancedb", options={})))


def test_build_memory_constructs_memark_backed_store_when_selected():
    # MemArkMemoryStore's own module needs networkx/faiss/chromadb at
    # import time (see companion/memark/retrieval.py's module docstring
    # and build_memory's own comment) — skip cleanly, not error, if the
    # memory-memark extra isn't installed, same convention as
    # tests/test_memark_retrieval.py.
    pytest.importorskip("networkx")
    pytest.importorskip("faiss")
    pytest.importorskip("chromadb")
    from companion.memark.retrieval import MemArkMemoryStore

    memory = build_memory(
        _config(memory=BackendConfig(backend="memark", options={"persist_dir": "/tmp/mem-memark"}))
    )

    assert isinstance(memory, MemArkMemoryStore)
    assert str(memory._graph_store.persist_dir).replace("\\", "/") == "/tmp/mem-memark"


def test_build_tts_for_persona_piper_resolves_voice_id():
    tts = build_tts_for_persona(_config(), _persona(voice_id="piper:en_US-mentor-medium"))

    assert isinstance(tts, PiperTTS)
    assert tts.voice_model_path.replace("\\", "/") == "/tmp/voices/en_US-mentor-medium.onnx"
    assert tts.use_cuda is False


def test_build_tts_for_persona_coqui_resolves_speaker_from_voice_id():
    config = _config(
        tts=TTSConfig(
            backend="coqui",
            voices_dir="/tmp/voices",
            options={"model_name": "tts_models/multilingual/multi-dataset/xtts_v2"},
        )
    )

    tts = build_tts_for_persona(config, _persona(voice_id="coqui:some-speaker"))

    assert isinstance(tts, CoquiTTS)
    assert tts.speaker == "some-speaker"
    assert tts.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"


def test_build_tts_for_persona_coqui_requires_model_name():
    config = _config(tts=TTSConfig(backend="coqui", voices_dir="/tmp/voices", options={}))

    with pytest.raises(AppSetupError, match="model_name"):
        build_tts_for_persona(config, _persona(voice_id="coqui:some-speaker"))


def test_build_tts_for_persona_rejects_unimplemented_backend():
    config = _config(tts=TTSConfig(backend="espeak", voices_dir="/tmp", options={}))

    with pytest.raises(AppSetupError, match="espeak"):
        build_tts_for_persona(config, _persona())


def test_build_orchestrator_wires_all_real_components():
    orchestrator = build_orchestrator(_config(), _persona())

    assert isinstance(orchestrator, Orchestrator)
    assert isinstance(orchestrator.stt, FasterWhisperSTT)
    assert isinstance(orchestrator.tts, PiperTTS)
    assert isinstance(orchestrator.llm, OllamaLLM)
    assert isinstance(orchestrator.vad, SileroVAD)
    assert isinstance(orchestrator.memory, VectorMemoryStore)
    assert isinstance(orchestrator.prompt_builder, PromptBuilder)


def test_build_orchestrator_wires_memark_memory_when_selected():
    pytest.importorskip("networkx")
    pytest.importorskip("faiss")
    pytest.importorskip("chromadb")
    from companion.memark.retrieval import MemArkMemoryStore

    config = _config(
        memory=BackendConfig(backend="memark", options={"persist_dir": "/tmp/mem-memark"})
    )

    orchestrator = build_orchestrator(config, _persona())

    assert isinstance(orchestrator.memory, MemArkMemoryStore)


# --- load_personas / select_persona ----------------------------------------


def test_load_personas_wraps_persona_loader(tmp_path):
    (tmp_path / "mentor.yaml").write_text(yaml.safe_dump(VALID_PERSONA_YAML), encoding="utf-8")

    personas = load_personas(_config(personas_dir=str(tmp_path)))

    assert set(personas) == {"mentor"}
    assert personas["mentor"].name == "The Old Mentor"


def test_select_persona_by_explicit_id():
    personas = {"mentor": _persona(id="mentor"), "friend": _persona(id="friend")}

    assert select_persona(personas, "friend").id == "friend"


def test_select_persona_unknown_id_lists_available():
    personas = {"mentor": _persona(id="mentor"), "friend": _persona(id="friend")}

    with pytest.raises(AppSetupError, match="friend, mentor"):
        select_persona(personas, "nonexistent")


def test_select_persona_defaults_to_first_when_none_requested():
    personas = {"friend": _persona(id="friend"), "mentor": _persona(id="mentor")}

    assert select_persona(personas, None).id == "friend"  # dict insertion order


def test_select_persona_no_personas_raises():
    with pytest.raises(AppSetupError, match="No personas found"):
        select_persona({}, None)


# --- run_text_loop: mechanics via fakes -------------------------------------


class _FakeLLM(LLMBackend):
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.received_calls: list[tuple[str, list[Turn]]] = []

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.received_calls.append((prompt, context))
        yield next(self._replies, "ok.")


class _FakeMemory(MemoryStore):
    def __init__(self) -> None:
        self.written: list[tuple[str, str, Turn]] = []

    async def query(self, user_id: str, persona_id: str, query: str, k: int = 5) -> list[Turn]:
        return []

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        self.written.append((user_id, persona_id, turn))

    async def delete(self, user_id: str, persona_id: str | None = None) -> None:
        raise NotImplementedError


def _scripted_input(messages: list[str]):
    it = iter(messages)

    async def _read(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _read


def _make_fake_orchestrator(llm: LLMBackend, memory: MemoryStore) -> Orchestrator:
    from unittest.mock import Mock

    return Orchestrator(
        stt=Mock(),
        tts=Mock(),
        llm=llm,
        vad=Mock(),
        memory=memory,
        prompt_builder=PromptBuilder(),
    )


@pytest.mark.asyncio
async def test_run_text_loop_exit_command_stops_the_loop():
    orchestrator = _make_fake_orchestrator(_FakeLLM(["should not be reached"]), _FakeMemory())
    printed: list[str] = []

    await run_text_loop(
        orchestrator,
        _persona(),
        "u1",
        read_input=_scripted_input(["exit"]),
        print_fn=printed.append,
    )

    assert not any("should not be reached" in line for line in printed)


@pytest.mark.asyncio
async def test_run_text_loop_skips_blank_input():
    llm = _FakeLLM(["real reply"])
    orchestrator = _make_fake_orchestrator(llm, _FakeMemory())

    await run_text_loop(
        orchestrator,
        _persona(),
        "u1",
        read_input=_scripted_input(["   ", "hello", "exit"]),
        print_fn=lambda _: None,
    )

    assert len(llm.received_calls) == 1


@pytest.mark.asyncio
async def test_run_text_loop_eof_ends_gracefully():
    orchestrator = _make_fake_orchestrator(_FakeLLM([]), _FakeMemory())

    # _scripted_input([]) raises EOFError on the very first read
    await run_text_loop(
        orchestrator, _persona(), "u1", read_input=_scripted_input([]), print_fn=lambda _: None
    )
    # no assertion needed beyond "this returned instead of hanging/raising"


@pytest.mark.asyncio
async def test_run_text_loop_accumulates_recent_turns_across_messages():
    llm = _FakeLLM(["first reply.", "second reply."])
    orchestrator = _make_fake_orchestrator(llm, _FakeMemory())

    await run_text_loop(
        orchestrator,
        _persona(),
        "u1",
        read_input=_scripted_input(["first message", "second message", "exit"]),
        print_fn=lambda _: None,
    )

    assert len(llm.received_calls) == 2
    _prompt, second_context = llm.received_calls[1]
    context_texts = [t.text for t in second_context]
    assert "first message" in context_texts
    assert "first reply." in context_texts
    assert "second message" in context_texts


@pytest.mark.asyncio
async def test_run_text_loop_prints_persona_name_and_reply():
    llm = _FakeLLM(["a real reply"])
    orchestrator = _make_fake_orchestrator(llm, _FakeMemory())
    printed: list[str] = []

    await run_text_loop(
        orchestrator,
        _persona(name="The Old Mentor"),
        "u1",
        read_input=_scripted_input(["hi", "exit"]),
        print_fn=printed.append,
    )

    assert any("The Old Mentor> a real reply" in line for line in printed)


@pytest.mark.asyncio
async def test_full_wiring_from_real_config_and_persona_files_with_fake_llm_and_memory(
    tmp_path,
):
    """The integration-style test: real config.yaml + real persona.yaml,
    loaded through the real loaders, wired into a real Orchestrator with a
    real PromptBuilder — only LLM/MemoryStore are faked, since real
    Ollama/Chroma aren't practical to require for this test to run in CI.
    Confirms the *wiring*, not each backend's internals."""
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "mentor.yaml").write_text(
        yaml.safe_dump(VALID_PERSONA_YAML), encoding="utf-8"
    )

    config_data = {
        "personas": {"directory": str(personas_dir)},
        "stt": {"backend": "faster_whisper", "options": {}},
        "tts": {"backend": "piper", "voices_dir": str(tmp_path), "options": {}},
        "llm": {"backend": "ollama", "options": {"model": "does-not-matter"}},
        "vad": {"backend": "silero", "options": {}},
        "memory": {"backend": "chroma", "options": {"persist_dir": str(tmp_path / "mem")}},
        "audio": {"backend": "sounddevice", "options": {"sample_rate": 16000}},
        "mode": "text",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = load_config(config_path)
    personas = load_personas(config)
    persona = select_persona(personas, None)

    fake_llm = _FakeLLM(["Hello there, real wiring works."])
    fake_memory = _FakeMemory()
    orchestrator = Orchestrator(
        stt=build_stt(config),
        tts=build_tts_for_persona(config, persona),
        llm=fake_llm,
        vad=build_vad(config),
        memory=fake_memory,
        prompt_builder=PromptBuilder(),
    )

    printed: list[str] = []
    await run_text_loop(
        orchestrator,
        persona,
        "local",
        read_input=_scripted_input(["hi there", "exit"]),
        print_fn=printed.append,
    )

    assert any("Hello there, real wiring works." in line for line in printed)
    assert len(fake_memory.written) == 2  # user turn + assistant turn


# --- Fully real smoke test: real Ollama + real Chroma -----------------------


@pytest.mark.asyncio
async def test_real_end_to_end_text_conversation_smoke(tmp_path):
    """Builds and runs the app entirely for real: real FasterWhisperSTT/
    PiperTTS/SileroVAD construction (unused by text mode, just proving
    build_orchestrator doesn't choke on them), a real OllamaLLM against a
    locally running server with at least one model pulled, and a real
    Chroma-backed MemoryStore. Skipped, not failed, if Ollama or chromadb
    aren't available — same pattern as every other real-backend smoke
    test in this repo."""
    pytest.importorskip("chromadb")
    pytest.importorskip("ollama")
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except httpx.HTTPError:
        pytest.skip("No local Ollama server reachable at localhost:11434")
    if not models:
        pytest.skip("Ollama server has no models pulled")

    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "mentor.yaml").write_text(
        yaml.safe_dump(VALID_PERSONA_YAML), encoding="utf-8"
    )

    config = _config(
        personas_dir=str(personas_dir),
        tts=TTSConfig(backend="piper", voices_dir=str(tmp_path), options={}),
        llm=BackendConfig(backend="ollama", options={"model": models[0]}),
        memory=BackendConfig(backend="chroma", options={"persist_dir": str(tmp_path / "mem")}),
    )
    personas = load_personas(config)
    persona = select_persona(personas, "mentor")
    orchestrator = build_orchestrator(config, persona)

    reply = await orchestrator.handle_text_turn(
        user_id="test-user",
        persona=persona,
        recent_turns=[],
        message="Reply with a single short sentence confirming you can hear me.",
    )

    assert reply.strip()
    retrieved = await orchestrator.memory.query("test-user", persona.id, "hear me", k=5)
    assert any("hear me" in t.text.lower() for t in retrieved)


# --- agent wiring -------------------------------------------------------------


def test_build_agent_is_none_without_an_agent_section():
    from companion.app import build_agent

    assert build_agent(_config(), memory=Mock()) is None


def test_build_agent_constructs_the_langgraph_runner():
    pytest.importorskip("langgraph")
    from companion.app import build_agent
    from companion.backends.agent.langgraph_ollama import LangGraphOllamaAgent

    agent = build_agent(
        _config(agent=BackendConfig(backend="langgraph", options={"max_steps": 3})), memory=Mock()
    )

    assert isinstance(agent, LangGraphOllamaAgent)
    assert agent.max_steps == 3


def test_build_agent_rejects_unknown_backend():
    from companion.app import build_agent

    with pytest.raises(AppSetupError, match="crewai"):
        build_agent(_config(agent=BackendConfig(backend="crewai", options={})), memory=Mock())


def test_load_personas_rejects_unknown_tool_names(tmp_path):
    from companion.tools.registry import UnknownToolError

    (tmp_path / "p.yaml").write_text(
        yaml.safe_dump({**VALID_PERSONA_YAML, "tools": ["teleport"]}), encoding="utf-8"
    )

    with pytest.raises(UnknownToolError, match="teleport"):
        load_personas(_config(personas_dir=str(tmp_path)))


def test_take_due_reminders_names_the_persona_that_set_it(tmp_path):
    from datetime import datetime, timedelta

    from companion.app import take_due_reminders
    from companion.tools.store import ToolStore

    now = datetime.now().astimezone()
    ToolStore(tmp_path).add_reminder("u1", "mentor", "Stretch", now - timedelta(minutes=1), now)
    config = _config(agent=BackendConfig(backend="langgraph", options={"data_dir": str(tmp_path)}))

    lines = take_due_reminders(config, "u1", {"mentor": _persona()})

    assert lines == ["⏰ Reminder (from The Old Mentor): Stretch"]
    assert take_due_reminders(config, "u1", {"mentor": _persona()}) == []


async def test_text_loop_prints_due_reminders_while_idle_and_tool_use():
    from companion.core.agent import TextDelta, ToolStarted

    class _Agent:
        def handles(self, persona):
            return True

        async def run_stream(self, persona, user_id, built):
            yield ToolStarted("set_reminder")
            yield TextDelta("Done.")

    orchestrator = Orchestrator(stt=Mock(), tts=Mock(), llm=Mock(), vad=Mock(),
                                memory=_FakeMemory(), prompt_builder=PromptBuilder(),
                                agent=_Agent())
    printed = []
    pending = [["⏰ Reminder: Stretch"]]
    inputs = iter(["remind me", "exit"])

    async def read_input(prompt):
        await asyncio.sleep(0.05)  # idle long enough for the poller to run
        return next(inputs)

    await run_text_loop(orchestrator, _persona(), "u1", read_input=read_input,
                        print_fn=printed.append,
                        check_reminders=lambda: pending.pop() if pending else [],
                        reminder_poll_s=0.01)

    assert "\n⏰ Reminder: Stretch\n" in printed
    assert "  (using set_reminder...)" in printed
    assert "The Old Mentor> Done.\n" in printed
