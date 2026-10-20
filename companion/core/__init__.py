"""The backend-agnostic core:
- `interfaces/`: the five abstract contracts plus `Turn` and `Persona`.
- `orchestrator/`: the text and voice turn loops.
- `prompt_builder/`: the only place prompts are assembled.
- `persona_loader/`: YAML to `Persona`, with strict validation.
- `agent/`: the `AgentRunner` protocol and events for personas with tools.

Nothing here imports a concrete backend; see docs/ARCHITECTURE.md.
"""
