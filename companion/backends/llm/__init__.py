"""LLM backends. `ollama_local.OllamaLLM` streams chat completions from a
local Ollama server and turns connection and missing-model failures into
`OllamaConnectionError` / `OllamaModelNotFoundError` with actionable
messages. Tool-calling personas don't use this; they go through
`companion/backends/agent/`.
"""
