"""`AgentRunner` on LangGraph + Ollama: the tool-calling loop for personas
that list `tools:` (see companion/core/agent/ and docs/AGENTIC_TOOLS.md).

Uses `langchain.agents.create_agent` (LangGraph's prebuilt ReAct loop; in
LangGraph 1.x `langgraph.prebuilt.create_react_agent` is deprecated in its
favour) with `ChatOllama`. Needs a model that supports tool calling: check
with `ollama show <model>` and look for "tools" under Capabilities.
`qwen2.5:7b-instruct` is the default.

Per turn:
- The persona's tools are built for this user (companion/tools/registry.py)
  and wrapped so a tool can never fail the turn: bad arguments, timeouts
  and crashes go back to the model as an "Error: ..." result it can react
  to.
- The input is exactly `PromptBuilder`'s output: `built.prompt` as the
  system message, `built.context` as the conversation. No prompt text is
  added here; PromptBuilder stays the only place prompts are assembled.
- Events are streamed: `ToolStarted` when the model calls a tool,
  `ToolFinished` when it returns, and `TextDelta` for answer text.
- `max_steps` bounds the loop (LangGraph's recursion limit). Hitting it
  produces an apology as the answer instead of an exception.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from companion.backends.llm.ollama_local import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from companion.core.agent import AgentEvent, TextDelta, ToolFinished, ToolStarted
from companion.tools.base import Tool, ToolContext, ToolInputError
from companion.tools.registry import build_persona_tools, enabled_tool_names

if TYPE_CHECKING:
    from companion.core.interfaces import MemoryStore
    from companion.core.interfaces.types import Persona
    from companion.core.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)

STUCK_MESSAGE = "Sorry, I got stuck working that out. Could you ask in a different way?"


def failed_tool_note(tool_names: list[str]) -> str:
    """Appended to the reply when a tool failed this turn and was never
    successfully retried. qwen2.5:7b sometimes says "Done!" after a failed
    call; this keeps the spoken reply, the text and the saved memory
    truthful whatever the model claims."""
    names = ", ".join(name.replace("_", " ") for name in tool_names)
    return f" (Heads up: {names} didn't go through, so nothing was saved. Please try again.)"


class LangGraphOllamaAgent:
    def __init__(
        self,
        memory: MemoryStore,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_steps: int = 6,
        tool_timeout_s: float = 10.0,
        allow_network: bool = False,
        data_dir: str = "./data/tools",
        num_ctx: int | None = None,
    ) -> None:
        self.memory = memory
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_steps = max_steps
        self.tool_timeout_s = tool_timeout_s
        self.allow_network = allow_network
        self.data_dir = Path(data_dir)
        self.num_ctx = num_ctx
        self._chat_model = None  # built lazily, like OllamaLLM's client
        self._chat_model_loop = None  # the event loop it was built on

    def handles(self, persona: Persona) -> bool:
        return bool(enabled_tool_names(persona, self.allow_network))

    def _ensure_chat_model(self):
        """One `ChatOllama` per event loop, for the same reason as
        `OllamaLLM._ensure_client`: its async HTTP client is bound to the
        loop it was created on, and Streamlit runs a new loop per message. A
        model set directly (tests inject fakes this way) has no recorded
        loop and is always kept."""
        loop = asyncio.get_running_loop()
        stale = self._chat_model_loop is not None and self._chat_model_loop is not loop
        if self._chat_model is None or stale:
            from langchain_ollama import ChatOllama

            kwargs = {"model": self.model, "base_url": self.host, "temperature": self.temperature}
            if self.num_ctx is not None:
                kwargs["num_ctx"] = self.num_ctx
            self._chat_model = ChatOllama(**kwargs)
            self._chat_model_loop = loop
        return self._chat_model

    def _wrap(self, tool: Tool):
        from langchain_core.tools import StructuredTool

        schema = StructuredTool.from_function(
            coroutine=tool.func, name=tool.name, description=tool.description
        ).args_schema

        async def call(**kwargs) -> str:
            try:
                return await asyncio.wait_for(tool.func(**kwargs), self.tool_timeout_s)
            except ToolInputError as exc:
                logger.info("tool %s rejected %s: %s", tool.name, kwargs, exc)
                return f"Error: {exc}"
            except TimeoutError:
                logger.warning("tool %s timed out (%s)", tool.name, kwargs)
                return f"Error: {tool.name} timed out after {self.tool_timeout_s:g}s."
            except Exception:
                logger.exception("tool %s failed", tool.name)
                return f"Error: {tool.name} failed unexpectedly."

        return StructuredTool(name=tool.name, description=tool.description,
                              args_schema=schema, coroutine=call)

    def _messages(self, built: BuiltPrompt) -> list:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [SystemMessage(built.prompt)]
        for turn in built.context:
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(turn.text))
        return messages

    async def run_stream(
        self, persona: Persona, user_id: str, built: BuiltPrompt
    ) -> AsyncIterator[AgentEvent]:
        from langchain.agents import create_agent
        from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
        from langgraph.errors import GraphRecursionError

        ctx = ToolContext(user_id=user_id, persona_id=persona.id, memory=self.memory,
                          data_dir=self.data_dir)
        tools = [self._wrap(t) for t in build_persona_tools(persona, ctx, self.allow_network)]
        agent = create_agent(self._ensure_chat_model(), tools)
        # Each step is one model call or one round of tool calls.
        config = {"recursion_limit": 2 * self.max_steps + 1}

        announced: set[str] = set()  # tool-call ids already reported as started
        failed: dict[str, None] = {}  # tools whose last call this turn failed, in order
        try:
            async for chunk, meta in agent.astream(
                {"messages": self._messages(built)}, config=config, stream_mode="messages"
            ):
                if isinstance(chunk, ToolMessage):
                    content = str(chunk.content)
                    ok = chunk.status != "error" and not content.startswith("Error:")
                    name = chunk.name or "tool"
                    if ok:
                        failed.pop(name, None)  # a successful retry clears it
                    else:
                        failed[name] = None
                    yield ToolFinished(name, ok=ok,
                                       detail="" if ok else content.removeprefix("Error: "))
                    continue
                if not isinstance(chunk, AIMessage) or meta.get("langgraph_node") != "model":
                    continue
                # Announce only parsed, valid calls (`tool_calls`), never raw
                # `tool_call_chunks`: a malformed call shows up in the stream
                # but never runs, and announcing it made the CLI print
                # "(using set_reminder...)" for a reminder that was never
                # set. Ollama delivers each call whole, in one chunk, so this
                # costs the voice filler no latency.
                calls = chunk.tool_calls or []
                for bad in chunk.invalid_tool_calls or []:
                    logger.warning("agent for %s produced a malformed tool call: %s",
                                   persona.id, bad)
                pending_call = bool(calls or chunk.invalid_tool_calls
                                    or (isinstance(chunk, AIMessageChunk)
                                        and chunk.tool_call_chunks))
                for call in calls:
                    key = call.get("id") or call.get("name")
                    if call.get("name") and key not in announced:
                        announced.add(key)
                        yield ToolStarted(call["name"])
                if chunk.content and not pending_call:
                    yield TextDelta(chunk.text)
            if failed:
                yield TextDelta(failed_tool_note(list(failed)))
        except GraphRecursionError:
            logger.warning("agent for %s hit max_steps=%d", persona.id, self.max_steps)
            yield TextDelta(STUCK_MESSAGE)
        except Exception as exc:
            raise _translate_ollama_error(exc, self.model, self.host) from exc


def _translate_ollama_error(exc: Exception, model: str, host: str) -> Exception:
    """Same actionable errors OllamaLLM raises, so the CLI and UI show
    "pull the model" / "start Ollama" instead of a library traceback."""
    from ollama import ResponseError

    if isinstance(exc, ResponseError) and exc.status_code == 404:
        return OllamaModelNotFoundError(
            f"Agent model {model!r} isn't pulled. Run: ollama pull {model}"
        )
    if isinstance(exc, ConnectionError) or type(exc).__name__ in ("ConnectError", "ConnectTimeout"):
        return OllamaConnectionError(f"Can't reach Ollama at {host}. Is `ollama serve` running?")
    return exc
