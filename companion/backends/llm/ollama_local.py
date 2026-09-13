"""Local LLM backend: Ollama.

Model name is config-driven, not hardcoded — e.g. a smaller ~3B-class model
for voice mode (latency-sensitive) and a larger 7-8B instruction-tuned
model for text mode is one config value swapped (`llm.voice_mode_override`
in config.example.yaml), not two code paths.

## Why `qwen2.5:7b-instruct` is the default

Comparable conversational quality to Llama 3.1 8B for this use case, but
Apache-2.0 licensed — Llama 3.1's model license is source-available with
usage restrictions (an acceptable-use policy, and conditions that kick in
above 700M MAU), not OSI-approved. For a project whose own code is MIT and
that explicitly tracks bundled-model licensing (THIRD_PARTY_LICENSES.md),
defaulting to the more permissively licensed model is the better fit — not
a claim that it's strictly a better model. Both are one `model:` edit away
in config.yaml either way; nothing about this backend assumes Qwen.

## Streaming via `ollama`'s `AsyncClient.chat()`

Verified against a real local Ollama server (0.24.0) with `stream=True`:
`await client.chat(...)` returns immediately with an async generator —
it does not itself perform the request. The actual HTTP connection (and
therefore any connection/model-not-found error) only happens once you
start iterating it, which is why the try/except below wraps the `async
for`, not the `await client.chat(...)` call — wrapping only the latter
would silently miss every real failure mode.

`context: list[Turn]` (per the LLMBackend interface — recent conversation
+ retrieved memory, already merged and ordered by PromptBuilder) maps onto
Ollama's `chat()` message list directly: each `Turn.role` is already one of
"user"/"assistant"/"system", matching Ollama's expected message roles, so
turns become `{"role": turn.role, "content": turn.text}` entries following
one `{"role": "system", "content": prompt}` message built from `prompt`
(the persona/grounding text). This couples PromptBuilder's job to this
shape — see docs/PROMPT_BUILDER.md's design note for the seam this
assumes memory retrieval produces `Turn`s that fit here.

## Error handling — a real gotcha, not a hypothetical

Two distinct failure modes, and they do NOT share an exception type:
- Ollama unreachable (server not running): raises `ollama`'s own
  `ConnectionError` when NOT streaming, but that wrapping does not apply
  to the `stream=True` code path used here — verified directly. What
  actually surfaces during iteration is `httpx.TransportError` (e.g.
  `httpx.ConnectError`), since `stream=True` defers the connection to
  first iteration, bypassing ollama-python's own connect-time wrapping.
  Both are caught below and re-raised as `OllamaConnectionError`.
- Model not pulled locally: raises `ollama.ResponseError` with
  `status_code == 404` — re-raised as `OllamaModelNotFoundError` with the
  `ollama pull <model>` command to fix it, rather than propagating a bare
  404 with no actionable next step.

Install: pip install "companion[llm-ollama]" and have the Ollama server
running locally (`ollama pull <model>` for whichever model config selects).
"""

from typing import AsyncIterator

from companion.core.interfaces.llm_backend import LLMBackend
from companion.core.interfaces.types import Turn


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server can't be reached at all."""


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the configured model hasn't been pulled locally."""


class OllamaLLM(LLMBackend):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_predict: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_predict = num_predict
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.stop = stop

        self._client = None  # created lazily — see _ensure_client

    def _ensure_client(self):
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self.host)
        return self._client

    def _options(self) -> dict:
        options: dict = {"temperature": self.temperature}
        for key in ("num_predict", "top_p", "top_k", "seed", "stop"):
            value = getattr(self, key)
            if value is not None:
                options[key] = value
        return options

    async def generate_stream(
        self, prompt: str, context: list[Turn]
    ) -> AsyncIterator[str]:
        import httpx
        from ollama import ResponseError

        client = self._ensure_client()
        messages = [{"role": "system", "content": prompt}]
        messages.extend({"role": turn.role, "content": turn.text} for turn in context)

        try:
            stream = await client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=self._options(),
            )
            async for chunk in stream:
                content = chunk.message.content if chunk.message else None
                if content:
                    yield content
        except ResponseError as exc:
            if exc.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model {self.model!r} isn't pulled on the Ollama server at "
                    f"{self.host!r}. Run: ollama pull {self.model}"
                ) from exc
            raise
        except (httpx.TransportError, ConnectionError) as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host!r} — is the server running "
                f"(`ollama serve`)? ({exc})"
            ) from exc
