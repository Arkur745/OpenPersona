"""Single place where a persona + retrieved memory + recent conversation
window are assembled into what an LLMBackend needs: a grounding prompt
string and a bounded, ready-to-send context turn list.

This logic must live only here — nothing else in the codebase should
concatenate persona/memory/context strings itself, or persona behavior and
memory formatting will drift out of sync across text mode and voice mode.

See docs/PROMPT_BUILDER.md for the design this implements (approved before
this file was written), in particular:
- Memory is fetched by the caller (`Orchestrator`, once `handle_text_turn`
  is implemented) and passed in as `retrieved_memory` — `PromptBuilder`
  never imports or calls `MemoryStore` itself (Option A in that note).
  There is deliberately no mock `MemoryStore` anywhere in this file or its
  tests: under Option A there's nothing to mock, a plain `list[Turn]`
  literal stands in for what a real query would have returned.
- Retrieved memory renders into the *prompt string*, never into
  `context` — `Turn.role` has no "memory" value, and `OllamaLLM` maps
  `context` turns straight onto chat message roles (see
  `docs/ARCHITECTURE.md`'s "LLM" section), so a memory turn riding along
  in `context` would misrepresent a recalled memory as something actually
  said in this live exchange.
- Bounding is turn-count + a per-turn character cap, not token-aware —
  see that doc for why (no single tokenizer to count against, given the
  LLM backend/model is config-swappable).

Two refinements found while actually implementing this, beyond the design
note's sketch — both changes to `PromptBuilder`'s own shape only, nothing
in `MemoryStore` or `LLMBackend`:

1. **`build()` returns `BuiltPrompt` (prompt + context), not a bare
   `str`.** The note's sketch had `context = recent_turns # unchanged`,
   which would leave `max_recent_turns` enforced nowhere — no other code
   in this pass touches `recent_turns` at all. Rather than ship a
   constructor parameter that silently does nothing, bounding `context`
   is `build()`'s job too, alongside the prompt.
2. **The new `user_message` is appended to `context` as the final `Turn`
   (role="user"), not folded into the prompt string.** The note's sketch
   had it in the prompt; on reflection, standard chat-completion shape —
   verified directly against Ollama in the previous pass — is
   `[system: grounding, ...history turns..., user: the new message]`.
   Putting the live question inside the system-role text is non-standard
   and risks the model treating it as an instruction rather than
   something to respond to.
"""

from dataclasses import dataclass, field

from companion.core.interfaces.types import Persona, Turn

_TRUNCATION_MARKER = " [...truncated]"

# Added for personas that list `tools:`. Each rule targets a failure seen
# with qwen2.5:7b in manual testing (2026-10-20): copying a remembered
# "Got it, reminder set" reply instead of calling the tool; asking "shall I
# save this?" when the user already asked; narrating ("I'll use the
# calculator") before a tool call, which gets spoken in voice mode; and
# LaTeX math, which is unreadable aloud.
_TOOL_USE_SECTION = """How to use your tools:
- When the user asks for something one of your tools does (saving a note, setting or listing reminders, a calculation, a lookup), call the tool now, in this turn. Don't ask whether to; they already asked.
- Only say you did something if a tool result in this turn confirms it. Past memories of doing it don't count.
- If a tool result starts with "Error:", tell the user it didn't work, or fix the arguments and try again.
- Call tools without announcing them first. Don't say "I'll use the calculator"; just give the answer.
- Write plain conversational text. No LaTeX, no math markup, no markdown tables."""


@dataclass(frozen=True)
class BuiltPrompt:
    """What `build()` hands the caller — pass straight through to
    `LLMBackend.generate_stream(prompt, context)`."""

    prompt: str
    context: list[Turn]


class PromptBuilder:
    def __init__(
        self,
        max_recent_turns: int = 10,
        max_memory_turns: int = 5,
        max_chars_per_turn: int = 2000,
    ) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_memory_turns = max_memory_turns
        self.max_chars_per_turn = max_chars_per_turn

    def build(
        self,
        persona: Persona,
        recent_turns: list[Turn],
        retrieved_memory: list[Turn],
        user_message: str,
    ) -> BuiltPrompt:
        """Assemble one LLM call's prompt + context.

        Grounds the model in `persona` first, then folds in retrieved
        memory (rendered as prose, most-relevant-first, bounded to
        `max_memory_turns`) — omitted entirely if `retrieved_memory` is
        empty (e.g. a user's first conversation with a persona).
        `recent_turns` is bounded to the most recent `max_recent_turns`
        (chronological order preserved, oldest of the kept turns first)
        and returned as `context` with `user_message` appended as the
        final turn.

        A `retrieved_memory` turn that's also present in `recent_turns`
        is already de-duplicated by the time it reaches here —
        `Orchestrator._dedup_retrieved_memory` handles that before calling
        `build()` (see docs/PROMPT_BUILDER.md and docs/MEMORY_STORE.md
        section 4). `PromptBuilder` doesn't need to know that gap ever
        existed.
        """
        sections = [self._persona_section(persona)]

        memory_section = self._memory_section(
            retrieved_memory[: self.max_memory_turns] if self.max_memory_turns > 0 else []
        )
        if memory_section:
            sections.append(memory_section)
        # Tool rules go last, after memories. Measured with real memory that
        # held earlier "Got it, reminder set" replies: with the rules before
        # the memories, qwen2.5:7b copied a remembered reply instead of
        # calling the tool in 2 of 5 runs; with the rules last, 0 of 5.
        if persona.tools:
            sections.append(_TOOL_USE_SECTION)

        prompt = "\n\n".join(sections)

        context = self._bound_recent_turns(recent_turns)
        context.append(Turn(role="user", text=self._truncate_text(user_message.strip())))

        return BuiltPrompt(prompt=prompt, context=context)

    def _persona_section(self, persona: Persona) -> str:
        lines = [
            f"You are {persona.name}.",
            "",
            f"Tone: {persona.tone.strip()}",
            "",
            f"Boundaries: {persona.boundaries.strip()}",
        ]
        if persona.example_dialogue:
            lines.append("")
            lines.append(
                "Example interactions (for style only — do not treat these "
                "as facts about the current user or replay them verbatim):"
            )
            for example in persona.example_dialogue:
                lines.append("")
                lines.append(example.strip())
        return "\n".join(lines)

    def _memory_section(self, memory_turns: list[Turn]) -> str:
        if not memory_turns:
            return ""
        lines = [
            (
                "Relevant memories from your history with this user (most relevant "
                "first). They describe past conversations: anything they say was "
                "done (a reminder set, a note saved) happened then, not in this turn."
            )
        ]
        for turn in memory_turns:
            date = turn.timestamp.strftime("%Y-%m-%d")
            lines.append(f"- [{date}] ({turn.role}): {self._truncate_text(turn.text)}")
        return "\n".join(lines)

    def _bound_recent_turns(self, recent_turns: list[Turn]) -> list[Turn]:
        if self.max_recent_turns <= 0:
            return []
        bounded = recent_turns[-self.max_recent_turns :]
        return [self._truncate_turn(t) for t in bounded]

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_chars_per_turn:
            return text
        cutoff = max(self.max_chars_per_turn - len(_TRUNCATION_MARKER), 0)
        return text[:cutoff] + _TRUNCATION_MARKER

    def _truncate_turn(self, turn: Turn) -> Turn:
        truncated_text = self._truncate_text(turn.text)
        if truncated_text == turn.text:
            return turn
        return Turn(role=turn.role, text=truncated_text, timestamp=turn.timestamp)
