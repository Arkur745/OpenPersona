"""`PromptBuilder.build()` turns persona + retrieved memory + recent turns +
the new message into `BuiltPrompt(prompt, context)`. It's a pure function
with no `MemoryStore` dependency, and the only place persona prompts are
assembled. See docs/PROMPT_BUILDER.md.
"""

from companion.core.prompt_builder.prompt_builder import BuiltPrompt, PromptBuilder

__all__ = ["BuiltPrompt", "PromptBuilder"]
