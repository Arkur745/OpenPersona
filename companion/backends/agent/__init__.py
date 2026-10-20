"""Agent runners: implementations of `companion.core.agent.AgentRunner`.

`langgraph_ollama.LangGraphOllamaAgent` runs the tool-calling loop for
personas that list `tools:` (LangGraph's `create_agent` + `ChatOllama`).
It's the only module in the project that imports LangGraph or LangChain.
It builds each persona's tools per user from `companion/tools/registry.py`,
turns tool errors and timeouts into results the model can react to, and
bounds the loop with `max_steps`. Needs the `agents` extra and an Ollama
model with tool support. See docs/ARCHITECTURE.md "Agentic personas".
"""
