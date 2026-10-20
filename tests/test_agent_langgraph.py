"""Tests for companion/backends/agent/langgraph_ollama.py.

The real LangGraph loop runs against a scripted fake chat model (no
Ollama), with real tools writing to a temporary data dir. A gated smoke
test at the end runs the real thing against a local qwen2.5 model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_ollama")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from companion.backends.agent.langgraph_ollama import (
    STUCK_MESSAGE,
    LangGraphOllamaAgent,
)
from companion.core.agent import AgentRunner, TextDelta, ToolFinished, ToolStarted
from companion.core.interfaces import MemoryStore, Turn
from companion.core.interfaces.types import Persona
from companion.core.prompt_builder import PromptBuilder
from companion.tools.store import ToolStore


class _ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order (the last one repeats) and
    records what it was sent."""

    script: list
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        message = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        if message.tool_calls:  # real models give every call a fresh id
            message = message.model_copy(update={"tool_calls": [
                {**c, "id": f"call_{len(self.seen)}_{i}"} for i, c in enumerate(message.tool_calls)
            ]})
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _NoMemory(MemoryStore):
    async def query(self, user_id, persona_id, query, k=5):
        return []

    async def write(self, user_id, persona_id, turn):
        pass

    async def delete(self, user_id, persona_id=None):
        pass

    async def list_turns(self, user_id, persona_id):
        return []

    async def replace_turns(self, user_id, persona_id, old, new):
        return False


def _persona(*tools: str) -> Persona:
    return Persona(id="planner", name="Planner", tone="Brisk.", boundaries="None.",
                   voice_id="piper:x", tools=list(tools))


def _agent(tmp_path, script, **kwargs) -> tuple[LangGraphOllamaAgent, _ScriptedModel]:
    agent = LangGraphOllamaAgent(memory=_NoMemory(), data_dir=str(tmp_path), **kwargs)
    model = _ScriptedModel(script=script, seen=[])
    agent._chat_model = model
    return agent, model


async def _run(agent, persona, message="hi") -> list:
    built = PromptBuilder().build(persona=persona, recent_turns=[
        Turn(role="user", text="earlier question"), Turn(role="assistant", text="earlier answer"),
    ], retrieved_memory=[], user_message=message)
    return [e async for e in agent.run_stream(persona, "u1", built)]


def test_satisfies_the_agent_runner_protocol(tmp_path):
    agent, _ = _agent(tmp_path, [AIMessage("x")])
    assert isinstance(agent, AgentRunner)


async def test_tool_call_then_answer(tmp_path):
    agent, model = _agent(tmp_path, [
        _tool_call("set_reminder", {"what": "Stretch", "when": "in 10 minutes"}),
        AIMessage("Done, I'll remind you to stretch in 10 minutes."),
    ])

    events = await _run(agent, _persona("reminders"), "Remind me to stretch in 10 minutes")

    assert events == [
        ToolStarted("set_reminder"),
        ToolFinished("set_reminder", ok=True),
        TextDelta("Done, I'll remind you to stretch in 10 minutes."),
    ]
    reminders = ToolStore(tmp_path).pending_reminders("u1")
    assert [r.text for r in reminders] == ["Stretch"]
    assert reminders[0].due_at - datetime.now(UTC) < timedelta(minutes=10, seconds=5)

    first_call = model.seen[0]
    assert first_call[0].type == "system"
    assert [m.type for m in first_call[1:]] == ["human", "ai", "human"]
    assert first_call[-1].content == "Remind me to stretch in 10 minutes"
    tool_result = model.seen[1][-1]
    assert isinstance(tool_result, ToolMessage) and tool_result.content.startswith("Reminder #1")


async def test_bad_tool_arguments_go_back_to_the_model_as_an_error(tmp_path):
    from companion.backends.agent.langgraph_ollama import failed_tool_note

    agent, model = _agent(tmp_path, [
        _tool_call("set_reminder", {"what": "Stretch", "when": "whenever"}),
        AIMessage("When should I remind you?"),
    ])

    events = await _run(agent, _persona("reminders"))

    assert events[1].name == "set_reminder" and not events[1].ok
    assert events[1].detail.startswith("Can't tell when to remind.")
    assert model.seen[1][-1].content.startswith("Error: Can't tell when to remind.")
    # Never retried, so the honesty note is appended after the model's text.
    assert events[-2:] == [TextDelta("When should I remind you?"),
                           TextDelta(failed_tool_note(["set_reminder"]))]


async def test_unretried_failure_is_flagged_even_if_the_model_claims_success(tmp_path):
    """Seen in manual testing: set_reminder failed, then the model said
    "I've set a reminder for you". The note keeps the reply truthful."""
    from companion.backends.agent.langgraph_ollama import failed_tool_note

    agent, _ = _agent(tmp_path, [
        _tool_call("set_reminder", {"what": "Stretch", "when": "whenever"}),
        AIMessage("I've set a reminder for you to stretch!"),
    ])

    events = await _run(agent, _persona("reminders"))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))

    assert text.endswith(failed_tool_note(["set_reminder"]))
    assert "set reminder didn't go through" in text
    assert ToolStore(tmp_path).pending_reminders("u1") == []


async def test_successful_retry_clears_the_failure(tmp_path):
    agent, _ = _agent(tmp_path, [
        _tool_call("set_reminder", {"what": "Stretch", "when": "whenever"}),
        _tool_call("set_reminder", {"what": "Stretch", "when": "in 1 minute"}),
        AIMessage("Done, in a minute."),
    ])

    events = await _run(agent, _persona("reminders"))

    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("Done, in a minute.")]
    assert [r.text for r in ToolStore(tmp_path).pending_reminders("u1")] == ["Stretch"]


async def test_hitting_max_steps_gives_an_apology_not_an_exception(tmp_path):
    agent, _ = _agent(tmp_path, [_tool_call("get_datetime", {})], max_steps=2)

    events = await _run(agent, _persona("get_datetime"))

    assert events[-1] == TextDelta(STUCK_MESSAGE)


async def test_tool_timeout_is_reported_to_the_model(tmp_path, monkeypatch):
    import asyncio

    from companion.tools import offline

    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    agent, model = _agent(tmp_path, [_tool_call("add_note", {"text": "x"}), AIMessage("ok")],
                          tool_timeout_s=0.05)
    monkeypatch.setattr(offline.asyncio, "to_thread", slow)

    events = await _run(agent, _persona("notes"))

    assert ToolFinished("add_note", ok=False, detail="add_note timed out after 0.05s.") in events
    assert model.seen[1][-1].content == "Error: add_note timed out after 0.05s."


def test_handles_only_personas_with_tools_enabled_under_this_config(tmp_path):
    offline_agent, _ = _agent(tmp_path, [], allow_network=False)
    online_agent, _ = _agent(tmp_path, [], allow_network=True)

    assert not offline_agent.handles(_persona())
    assert offline_agent.handles(_persona("notes"))
    assert not offline_agent.handles(_persona("weather"))
    assert online_agent.handles(_persona("weather"))


# --- real Ollama ------------------------------------------------------------


def _model_available(name: str) -> bool:
    httpx = pytest.importorskip("httpx")
    try:
        tags = httpx.get("http://localhost:11434/api/tags", timeout=2.0).json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return False
    return any(m["name"] == name for m in tags.get("models", []))


async def test_real_qwen_uses_the_calculator(tmp_path):
    if not _model_available("qwen2.5:7b-instruct"):
        pytest.skip("needs a local Ollama server with qwen2.5:7b-instruct pulled")
    agent = LangGraphOllamaAgent(memory=_NoMemory(), data_dir=str(tmp_path), temperature=0)

    events = await _run(agent, _persona("calculator"), "What is 17% of 2,340? Use the calculator.")

    assert ToolStarted("calculate") in events
    answer = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "397.8" in answer


async def test_malformed_tool_call_is_not_announced(tmp_path, caplog):
    """A tool call that can't be parsed never runs, so it must not show up
    as "(using ...)", which is what made a reminder look set when it wasn't."""
    bad = AIMessage(content="", invalid_tool_calls=[
        {"name": "set_reminder", "args": "{not json", "id": "x1", "error": "bad json",
         "type": "invalid_tool_call"}])
    agent, _ = _agent(tmp_path, [bad])

    with caplog.at_level("WARNING"):
        events = await _run(agent, _persona("reminders"))

    assert not any(isinstance(e, ToolStarted) for e in events)
    assert "malformed tool call" in caplog.text
    assert ToolStore(tmp_path).pending_reminders("u1") == []


async def test_text_alongside_a_tool_call_is_not_streamed_as_the_answer(tmp_path):
    """Narration in the same model step as a tool call ("I'll use the
    calculator") isn't part of the answer, and in voice mode it would be
    spoken."""
    narrated_call = AIMessage(
        content="Sure! I'll use the calculator for this one.",
        tool_calls=[{"name": "calculate", "args": {"expression": "0.23 * 1875"}, "id": "c1"}],
    )
    agent, _ = _agent(tmp_path, [narrated_call, AIMessage("23% of 1,875 is 431.25.")])

    events = await _run(agent, _persona("calculator"))

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["23% of 1,875 is 431.25."]


def test_chat_model_is_rebuilt_for_a_new_event_loop(tmp_path):
    """Regression for Streamlit's "Event loop is closed": ChatOllama's async
    client is loop-bound, so each new loop gets a fresh one."""
    import asyncio

    agent = LangGraphOllamaAgent(memory=_NoMemory(), data_dir=str(tmp_path))

    async def get():
        return agent._ensure_chat_model(), agent._ensure_chat_model()

    a1, a2 = asyncio.run(get())
    b1, _ = asyncio.run(get())

    assert a1 is a2
    assert b1 is not a1


def test_real_qwen_works_across_separate_event_loops(tmp_path):
    import asyncio

    if not _model_available("qwen2.5:7b-instruct"):
        pytest.skip("needs a local Ollama server with qwen2.5:7b-instruct pulled")
    agent = LangGraphOllamaAgent(memory=_NoMemory(), data_dir=str(tmp_path), temperature=0)

    def turn():
        return asyncio.run(_run(agent, _persona("get_datetime"), "What day of the week is it?"))

    for _ in range(2):
        assert any(isinstance(e, TextDelta) for e in turn())
