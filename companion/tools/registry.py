"""Every tool a persona can list in its `tools:` field, by name.

Adding a tool = a `build_*` function returning `Tool`s plus one entry in
`TOOL_SPECS`. Personas pick tools by name in YAML, so giving a persona a
tool never needs code.

Network tools (`requires_network=True`) are only built when the config
allows it (`agent.options.allow_network: true`). When it doesn't, a persona
that lists one simply goes without it, so the same persona file works
offline and online.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from companion.core.interfaces.types import Persona
from companion.tools.base import Tool, ToolContext, ToolSpec
from companion.tools.offline import (
    build_calculator_tools,
    build_datetime_tools,
    build_notes_tools,
    build_recall_tools,
    build_reminders_tools,
)
from companion.tools.online import build_weather_tools, build_web_search_tools

logger = logging.getLogger(__name__)

TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec("get_datetime", "current date and time", build_datetime_tools),
        ToolSpec("calculator", "exact arithmetic", build_calculator_tools),
        ToolSpec("notes", "save, search and delete notes", build_notes_tools),
        ToolSpec("reminders", "set, list and cancel reminders", build_reminders_tools),
        ToolSpec("recall_memory", "search past conversations", build_recall_tools),
        ToolSpec("web_search", "search the web", build_web_search_tools, requires_network=True),
        ToolSpec("weather", "weather forecast", build_weather_tools, requires_network=True),
    )
}


class UnknownToolError(ValueError):
    pass


def validate_persona_tools(personas: Iterable[Persona]) -> None:
    """Fails loudly at startup if any persona lists a tool that doesn't
    exist, naming the persona and the bad name."""
    for persona in personas:
        unknown = [name for name in persona.tools if name not in TOOL_SPECS]
        if unknown:
            raise UnknownToolError(
                f"persona {persona.id!r} lists unknown tool(s) {unknown}. "
                f"Available: {', '.join(sorted(TOOL_SPECS))}"
            )


def enabled_tool_names(persona: Persona, allow_network: bool) -> list[str]:
    """The persona's tools that will actually run under this config."""
    return [name for name in persona.tools
            if allow_network or not TOOL_SPECS[name].requires_network]


def build_persona_tools(persona: Persona, ctx: ToolContext, allow_network: bool) -> list[Tool]:
    validate_persona_tools([persona])
    names = enabled_tool_names(persona, allow_network)
    skipped = [n for n in persona.tools if n not in names]
    if skipped:
        logger.info(
            "persona %s: network tools %s are off (agent.options.allow_network is false)",
            persona.id, skipped,
        )
    return [tool for name in names for tool in TOOL_SPECS[name].build(ctx)]
