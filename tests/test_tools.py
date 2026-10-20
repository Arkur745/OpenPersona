"""Tests for companion/tools: every tool's own logic, without an LLM.

Fixed clock, a temporary SQLite store, a fake MemoryStore, and mocked HTTP
for the online tools (httpx.MockTransport; ddgs replaced in sys.modules),
so nothing here touches the network or a model.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from companion.core.interfaces import MemoryStore, Turn
from companion.core.interfaces.types import Persona
from companion.tools.base import ToolContext, ToolInputError
from companion.tools.offline import evaluate
from companion.tools.online import fetch_weather
from companion.tools.registry import (
    TOOL_SPECS,
    UnknownToolError,
    build_persona_tools,
    enabled_tool_names,
    validate_persona_tools,
)
from companion.tools.store import ToolStore

TZ = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 19, 18, 0, tzinfo=TZ)


class _FakeMemory(MemoryStore):
    def __init__(self, turns=()):
        self.turns = list(turns)
        self.queries = []

    async def query(self, user_id, persona_id, query, k=5):
        self.queries.append((user_id, persona_id, query, k))
        return self.turns[:k]

    async def write(self, user_id, persona_id, turn):
        pass

    async def delete(self, user_id, persona_id=None):
        pass

    async def list_turns(self, user_id, persona_id):
        return []

    async def replace_turns(self, user_id, persona_id, old, new):
        return False


def _ctx(tmp_path, user="u1", persona="p1", memory=None, now=NOW) -> ToolContext:
    return ToolContext(user_id=user, persona_id=persona, memory=memory or _FakeMemory(),
                       data_dir=tmp_path, now=lambda: now)


def _persona(*tools: str) -> Persona:
    return Persona(id="p1", name="P", tone="t", boundaries="b", voice_id="piper:x",
                   tools=list(tools))


def _tools(ctx, *names, allow_network=False) -> dict:
    return {t.name: t.func for t in build_persona_tools(_persona(*names), ctx, allow_network)}


# --- get_datetime ----------------------------------------------------------


async def test_get_datetime_reports_local_time_and_weekday(tmp_path):
    tools = _tools(_ctx(tmp_path), "get_datetime")

    assert await tools["get_datetime"]() == "Monday, 19 October 2026, 18:00 (UTC+05:30, UTC+0530)"


# --- calculator ------------------------------------------------------------


@pytest.mark.parametrize("expr, expected", [
    ("0.17 * 2340", 397.8),
    ("2,340 * 0.17", 397.8),
    ("2^10", 1024),
    ("sqrt(16) + abs(-2)", 6.0),
    ("(1 + 2) * 3 // 2", 4),
    ("round(pi, 2)", 3.14),
    ("min(10,200)", 10),
    ("1,234,567 + 1", 1234568),
    ("3 × 4 ÷ 2", 6),
])
def test_evaluate_supports_arithmetic(expr, expected):
    assert evaluate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "(1).__class__",
    "x + 1",
    "[1, 2]",
    "lambda: 1",
    "'a' * 3",
])
def test_evaluate_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ToolInputError):
        evaluate(expr)


def test_evaluate_caps_exponents_and_handles_division_by_zero():
    with pytest.raises(ToolInputError, match="Exponent too large"):
        evaluate("9 ** 9 ** 9")
    with pytest.raises(ToolInputError, match="Division by zero"):
        evaluate("1 / 0")


async def test_calculate_tool_formats_integers_cleanly(tmp_path):
    calc = _tools(_ctx(tmp_path), "calculator")["calculate"]

    assert await calc(expression="0.25 * 400") == "0.25 * 400 = 100"
    assert await calc(expression="1 / 3") == "1 / 3 = 0.3333333333"


# --- notes -----------------------------------------------------------------


async def test_notes_add_search_delete(tmp_path):
    tools = _tools(_ctx(tmp_path), "notes")

    assert await tools["add_note"](text="Learn Rust this winter") == \
        "Saved note #1: Learn Rust this winter"
    await tools["add_note"](text="Buy running shoes")

    listing = await tools["search_notes"]()
    assert listing.splitlines()[0].startswith("#2 ")  # newest first
    assert "Rust" in await tools["search_notes"](query="rust")
    assert await tools["search_notes"](query="python") == "No matching notes."

    assert await tools["delete_note"](note_id=1) == "Deleted note #1."
    assert await tools["delete_note"](note_id=1) == "There is no note #1."
    assert "Rust" not in await tools["search_notes"]()


async def test_notes_are_scoped_per_user_and_persona(tmp_path):
    mine = _tools(_ctx(tmp_path, persona="mentor"), "notes")
    other_persona = _tools(_ctx(tmp_path, persona="planner"), "notes")
    other_user = _tools(_ctx(tmp_path, user="u2", persona="mentor"), "notes")

    await mine["add_note"](text="secret plan")

    assert await other_persona["search_notes"]() == "No notes saved yet."
    assert await other_user["search_notes"]() == "No notes saved yet."
    assert await other_user["delete_note"](note_id=1) == "There is no note #1."


def test_note_search_treats_like_wildcards_literally(tmp_path):
    store = ToolStore(tmp_path)
    store.add_note("u", "p", "100% done", NOW)
    store.add_note("u", "p", "nothing here", NOW)

    assert [n.text for n in store.search_notes("u", "p", "%")] == ["100% done"]
    assert store.search_notes("u", "p", "_") == []


# --- reminders -------------------------------------------------------------


async def test_set_reminder_relative_and_absolute(tmp_path):
    tools = _tools(_ctx(tmp_path), "reminders")

    rel = await tools["set_reminder"](what="Stretch", when="in 10 minutes")
    abs_ = await tools["set_reminder"](what="Call mom", when="2026-10-20T09:00")

    assert rel == "Reminder #1 set for Mon 19 Oct 2026 18:10: Stretch"
    assert abs_ == "Reminder #2 set for Tue 20 Oct 2026 09:00: Call mom"
    listing = await tools["list_reminders"]()
    assert listing.index("Stretch") < listing.index("Call mom")


@pytest.mark.parametrize("when, message", [
    ("", "Can't tell when"),
    ("whenever", "Can't tell when"),
    ("in 0 minutes", "Can't tell when"),
    ("13pm", "Can't tell when"),
    ("at 25:00", "Can't tell when"),
    ("2026-10-19T17:00", "in the past"),
])
async def test_set_reminder_rejects_bad_times(tmp_path, when, message):
    tools = _tools(_ctx(tmp_path), "reminders")

    with pytest.raises(ToolInputError, match=message):
        await tools["set_reminder"](what="x", when=when)


# NOW is Monday 19 Oct 2026, 18:00 (+05:30).
@pytest.mark.parametrize("when, expected", [
    ("in 1 minute", "Mon 18:01"),
    ("1 min", "Mon 18:01"),
    ("in a minute", "Mon 18:01"),
    ("in 2 hours", "Mon 20:00"),
    ("an hour", "Mon 19:00"),
    ("half an hour", "Mon 18:30"),
    ("in 5 minutes from now", "Mon 18:05"),
    ("18:30", "Mon 18:30"),
    ("today at 7pm", "Mon 19:00"),
    ("tonight", "Mon 20:00"),
    ("6pm", "Tue 18:00"),  # exactly now, so tomorrow
    ("5pm", "Tue 17:00"),  # already passed today
    ("at 9:15 am", "Tue 09:15"),
    ("tomorrow at 9am", "Tue 09:00"),
    ("tomorrow", "Tue 09:00"),
    ("noon", "Tue 12:00"),
    ("2026-10-20T17:00", "Tue 17:00"),
])
def test_parse_when_understands_what_people_say(tmp_path, when, expected):
    from companion.tools.offline import parse_when

    assert f"{parse_when(_ctx(tmp_path), when):%a %H:%M}" == expected


async def test_cancel_reminder(tmp_path):
    tools = _tools(_ctx(tmp_path), "reminders")
    await tools["set_reminder"](what="Stretch", when="in 10 minutes")

    assert await tools["cancel_reminder"](reminder_id=1) == "Cancelled reminder #1."
    assert await tools["cancel_reminder"](reminder_id=1) == "There is no upcoming reminder #1."
    assert await tools["list_reminders"]() == "No upcoming reminders."


def test_due_reminders_are_delivered_exactly_once_to_any_persona(tmp_path):
    store = ToolStore(tmp_path)
    store.add_reminder("u1", "planner", "Stretch", NOW + timedelta(minutes=5), NOW)
    store.add_reminder("u1", "planner", "Later", NOW + timedelta(hours=5), NOW)
    store.add_reminder("u2", "planner", "Not yours", NOW, NOW)

    assert store.take_due_reminders("u1", NOW) == []
    due = store.take_due_reminders("u1", NOW + timedelta(minutes=6))

    assert [(r.text, r.persona_id) for r in due] == [("Stretch", "planner")]
    assert store.take_due_reminders("u1", NOW + timedelta(minutes=7)) == []
    assert [r.text for r in store.pending_reminders("u1")] == ["Later"]


# --- recall_memory ---------------------------------------------------------


async def test_recall_memory_queries_this_users_persona_scope(tmp_path):
    memory = _FakeMemory([Turn(role="user", text="My sister is Maya",
                               timestamp=datetime(2026, 9, 1, tzinfo=UTC))])
    tools = _tools(_ctx(tmp_path, memory=memory), "recall_memory")

    result = await tools["recall_memory"](query="sister")

    assert result == "- 01 Sep 2026 user: My sister is Maya"
    assert memory.queries == [("u1", "p1", "sister", 5)]


async def test_recall_memory_with_nothing_found(tmp_path):
    tools = _tools(_ctx(tmp_path), "recall_memory")

    assert await tools["recall_memory"](query="x") == "Nothing relevant found in past conversations."


# --- registry --------------------------------------------------------------


def test_unknown_tool_names_fail_loudly():
    with pytest.raises(UnknownToolError, match=r"persona 'p1'.*\['teleport'\]"):
        validate_persona_tools([_persona("notes", "teleport")])


def test_network_tools_are_dropped_unless_allowed(tmp_path):
    persona = _persona("get_datetime", "weather", "web_search")

    assert enabled_tool_names(persona, allow_network=False) == ["get_datetime"]
    assert enabled_tool_names(persona, allow_network=True) == ["get_datetime", "weather",
                                                               "web_search"]
    offline = [t.name for t in build_persona_tools(persona, _ctx(tmp_path), False)]
    assert offline == ["get_datetime"]


def test_every_tool_has_a_description_and_unique_name(tmp_path):
    names = []
    for spec in TOOL_SPECS.values():
        for tool in spec.build(_ctx(tmp_path)):
            assert tool.description.strip()
            names.append(tool.name)
    assert len(names) == len(set(names))


# --- online tools (mocked) -------------------------------------------------


def _weather_transport(request: httpx.Request) -> httpx.Response:
    if "geocoding" in request.url.host:
        if request.url.params["name"] == "Nowhere":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"results": [
            {"name": "Lisbon", "admin1": "Lisbon", "country": "Portugal",
             "latitude": 38.7, "longitude": -9.1}]})
    return httpx.Response(200, json={
        "current": {"temperature_2m": 21.5, "weather_code": 1},
        "daily": {"time": ["2026-10-19", "2026-10-20"],
                  "temperature_2m_min": [15.0, 14.2], "temperature_2m_max": [23.1, 20.0],
                  "precipitation_probability_max": [5, 70], "weather_code": [1, 63]},
    })


async def test_weather_summarizes_open_meteo_response():
    async with httpx.AsyncClient(transport=httpx.MockTransport(_weather_transport)) as client:
        text = await fetch_weather(client, "Lisbon", 2)

    assert text.splitlines() == [
        "Weather for Lisbon, Lisbon, Portugal:",
        "Now: 21.5°C, mainly clear",
        "2026-10-19: 15.0–23.1°C, mainly clear, 5% chance of rain",
        "2026-10-20: 14.2–20.0°C, rain, 70% chance of rain",
    ]


async def test_weather_unknown_place():
    async with httpx.AsyncClient(transport=httpx.MockTransport(_weather_transport)) as client:
        assert await fetch_weather(client, "Nowhere", 1) == "Couldn't find a place called 'Nowhere'."


async def test_web_search_formats_results(tmp_path, monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results):
            return [{"title": "Rust Book", "body": "Learn Rust.", "href": "https://doc.rust-lang.org"}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    search = _tools(_ctx(tmp_path), "web_search", allow_network=True)["web_search"]

    assert await search(query="rust") == "- Rust Book: Learn Rust. (https://doc.rust-lang.org)"


async def test_web_search_failure_is_reported_not_raised(tmp_path, monkeypatch):
    class BrokenDDGS:
        def text(self, query, max_results):
            raise RuntimeError("rate limited")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=BrokenDDGS))
    search = _tools(_ctx(tmp_path), "web_search", allow_network=True)["web_search"]

    assert await search(query="rust") == "Web search failed: RuntimeError: rate limited"


# Speech-to-text spells times differently from typing. qwen passes them on
# verbatim (measured: `when='9:22 p.m.'` from "Set a reminder at 9.22 p.m."),
# and before these were accepted every such voice reminder failed (0/3).
@pytest.mark.parametrize("when, expected", [
    ("9:22 p.m.", "Mon 21:22"),
    ("9:22 P.M.", "Mon 21:22"),
    ("9.22 p.m.", "Mon 21:22"),
    ("9.22 PM", "Mon 21:22"),
    ("9 22 pm", "Mon 21:22"),
    ("9 p.m.", "Mon 21:00"),
    ("tomorrow at 9 a.m.", "Tue 09:00"),
    ("for 10 minutes", "Mon 18:10"),
    ("after 2 hours", "Mon 20:00"),
])
def test_parse_when_understands_speech_to_text_spellings(tmp_path, when, expected):
    from companion.tools.offline import parse_when

    assert f"{parse_when(_ctx(tmp_path), when):%a %H:%M}" == expected
