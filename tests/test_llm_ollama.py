"""Tests for companion/backends/llm/ollama_local.py.

Same two-tier pattern as the STT/TTS suites: fake-client-injected tests
(bypass `_ensure_client`'s lazy creation by setting `._client` directly)
cover OllamaLLM's own message-building/option-passing/error-translation
logic deterministically and offline — `ollama` and `httpx` are in the
`dev` extra specifically so real `ResponseError`/`httpx.TransportError`
instances can be used here without a live server, not just guessed at.
`test_real_ollama_*` additionally exercises a real local Ollama server,
skipping itself (not failing) if one isn't reachable or has no models
pulled — unlike the STT/TTS smoke tests, this one can't auto-provision a
model to test with (LLM weights are multi-GB, not a small download), so
it uses whatever's already pulled rather than pulling something itself.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from ollama import ResponseError

from companion.backends.llm.ollama_local import (
    OllamaConnectionError,
    OllamaLLM,
    OllamaModelNotFoundError,
)
from companion.core.interfaces.types import Turn


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.message = SimpleNamespace(content=content) if content is not None else None


class _FakeClient:
    """Stands in for ollama.AsyncClient. `chat()` returns an async
    generator that either streams scripted chunks or raises a scripted
    exception mid-iteration — matching real ollama-python's behavior of
    deferring the actual request to first iteration (verified against a
    live server; see the module docstring in ollama_local.py)."""

    def __init__(self, chunks: list[_FakeChunk] | None = None, exception: Exception | None = None) -> None:
        self._chunks = chunks or []
        self._exception = exception
        self.received_calls: list[tuple] = []

    async def chat(self, model, messages, stream=True, options=None):
        self.received_calls.append((model, messages, stream, options))
        return self._stream()

    async def _stream(self):
        for chunk in self._chunks:
            yield chunk
        if self._exception is not None:
            raise self._exception


@pytest.mark.asyncio
async def test_generate_stream_yields_content_in_order():
    fake = _FakeClient(chunks=[_FakeChunk("Hello"), _FakeChunk(", "), _FakeChunk("world")])
    llm = OllamaLLM(model="unused")
    llm._client = fake

    received = [t async for t in llm.generate_stream("system prompt", [])]

    assert received == ["Hello", ", ", "world"]


@pytest.mark.asyncio
async def test_generate_stream_skips_falsy_content_chunks():
    fake = _FakeClient(chunks=[_FakeChunk("a"), _FakeChunk(None), _FakeChunk(""), _FakeChunk("b")])
    llm = OllamaLLM(model="unused")
    llm._client = fake

    received = [t async for t in llm.generate_stream("prompt", [])]

    assert received == ["a", "b"]


@pytest.mark.asyncio
async def test_generate_stream_builds_system_plus_context_messages():
    fake = _FakeClient(chunks=[_FakeChunk("ok")])
    llm = OllamaLLM(model="my-model")
    llm._client = fake

    context = [
        Turn(role="user", text="hi there"),
        Turn(role="assistant", text="hello!"),
    ]
    _ = [t async for t in llm.generate_stream("you are helpful", context)]

    assert len(fake.received_calls) == 1
    model, messages, stream, options = fake.received_calls[0]
    assert model == "my-model"
    assert stream is True
    assert messages == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]


@pytest.mark.asyncio
async def test_generate_stream_passes_generation_options():
    fake = _FakeClient(chunks=[_FakeChunk("ok")])
    llm = OllamaLLM(model="unused", temperature=0.2, top_p=0.9, seed=42)
    llm._client = fake

    _ = [t async for t in llm.generate_stream("prompt", [])]

    _, _, _, options = fake.received_calls[0]
    assert options == {"temperature": 0.2, "top_p": 0.9, "seed": 42}


@pytest.mark.asyncio
async def test_generate_stream_omits_unset_optional_generation_params():
    fake = _FakeClient(chunks=[_FakeChunk("ok")])
    llm = OllamaLLM(model="unused")  # defaults: only temperature is set
    llm._client = fake

    _ = [t async for t in llm.generate_stream("prompt", [])]

    _, _, _, options = fake.received_calls[0]
    assert options == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_model_not_found_raises_clear_error_with_pull_command():
    error = ResponseError(error="model 'ghost:latest' not found", status_code=404)
    fake = _FakeClient(chunks=[], exception=error)
    llm = OllamaLLM(model="ghost:latest")
    llm._client = fake

    with pytest.raises(OllamaModelNotFoundError, match="ollama pull ghost:latest"):
        async for _ in llm.generate_stream("prompt", []):
            pass


@pytest.mark.asyncio
async def test_other_response_errors_propagate_unwrapped():
    error = ResponseError(error="internal server error", status_code=500)
    fake = _FakeClient(chunks=[], exception=error)
    llm = OllamaLLM(model="unused")
    llm._client = fake

    with pytest.raises(ResponseError):
        async for _ in llm.generate_stream("prompt", []):
            pass


@pytest.mark.asyncio
async def test_connect_error_raises_clear_ollama_connection_error():
    fake = _FakeClient(chunks=[], exception=httpx.ConnectError("all connection attempts failed"))
    llm = OllamaLLM(model="unused", host="http://localhost:1")
    llm._client = fake

    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        async for _ in llm.generate_stream("prompt", []):
            pass


@pytest.mark.asyncio
async def test_partial_stream_then_connection_error_still_raises_clear_error():
    """A connection drop mid-stream (after some real tokens already came
    through) must still translate to OllamaConnectionError, not surface
    as a raw httpx error after silently swallowing the partial output."""
    fake = _FakeClient(
        chunks=[_FakeChunk("partial")], exception=httpx.ReadTimeout("timed out")
    )
    llm = OllamaLLM(model="unused")
    llm._client = fake

    received = []
    with pytest.raises(OllamaConnectionError):
        async for token in llm.generate_stream("prompt", []):
            received.append(token)

    assert received == ["partial"]


def _reachable_ollama_model_or_skip(host: str = "http://localhost:11434") -> str:
    from ollama import AsyncClient

    async def _probe():
        client = AsyncClient(host=host)
        resp = await client.list()
        return [m.model for m in resp.models]

    try:
        models = asyncio.run(_probe())
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {host!r} in this environment: {exc}")

    if not models:
        pytest.skip(f"Ollama at {host!r} has no models pulled — nothing to test against")
    return models[0]


def test_real_ollama_backend_smoke():
    """Optional real end-to-end check against a local Ollama server, using
    whatever model is already pulled (an LLM is too large to auto-pull
    for a test the way the STT/TTS smoke tests auto-download a small
    voice/model) — skipped (not failed) if no server or no models."""
    pytest.importorskip("ollama")
    model = _reachable_ollama_model_or_skip()
    llm = OllamaLLM(model=model, temperature=0.0)

    async def run():
        return [
            t
            async for t in llm.generate_stream(
                "You are a terse assistant. Answer in under 8 words.",
                [Turn(role="user", text="What is the capital of France?")],
            )
        ]

    tokens = asyncio.run(run())

    assert len(tokens) >= 1
    assert all(isinstance(t, str) for t in tokens)
    assert "".join(tokens).strip() != ""


def test_real_ollama_model_not_found_raises():
    pytest.importorskip("ollama")
    _reachable_ollama_model_or_skip()  # just to confirm the server itself is up

    llm = OllamaLLM(model="definitely-not-a-real-model:latest")

    async def run():
        async for _ in llm.generate_stream("prompt", []):
            pass

    with pytest.raises(OllamaModelNotFoundError):
        asyncio.run(run())
