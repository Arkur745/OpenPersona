"""Tools that run entirely on this machine (no network).

Each `build_*` function takes a `ToolContext` and returns the `Tool`s for
one registry entry (see registry.py). Return values are short plain text
written for the model to read, not for the user.
"""

from __future__ import annotations

import ast
import asyncio
import math
import operator
import re
from datetime import datetime, timedelta
from typing import Annotated

from companion.tools.base import Tool, ToolContext, ToolInputError
from companion.tools.store import ToolStore

# --- get_datetime ----------------------------------------------------------


def build_datetime_tools(ctx: ToolContext) -> list[Tool]:
    async def get_datetime() -> str:
        now = ctx.now()
        return now.strftime("%A, %d %B %Y, %H:%M (%Z, UTC%z)")

    return [Tool(
        "get_datetime",
        "Get the current local date, weekday and time. Use it before working "
        "out dates like 'tomorrow' or 'next Friday'.",
        get_datetime,
    )]


# --- calculator ------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000
_MAX_LEN = 300
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def evaluate(expression: str) -> float | int:
    """Safely evaluates an arithmetic expression: numbers, + - * / // % **,
    parentheses, and the functions/constants above. No names, attributes or
    calls beyond those, so it can't run code. Exponents are capped so a
    model can't hang the process with something like 9**9**9.

    Commas are ambiguous ("2,340" vs "min(10,200)"), so the expression is
    first evaluated as written, and only if that fails is it retried with
    thousands separators removed."""
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expression) > _MAX_LEN:
        raise ToolInputError(f"Expression too long (max {_MAX_LEN} characters).")
    try:
        return _evaluate_exact(expression)
    except ToolInputError:
        without_separators = _THOUSANDS_SEP.sub("", expression)
        if without_separators == expression:
            raise
        return _evaluate_exact(without_separators)


def _evaluate_exact(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolInputError(f"Not a valid arithmetic expression: {expression!r}") from None

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ToolInputError(f"Exponent too large (max {_MAX_EXPONENT}).")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ToolInputError(
            f"Unsupported element in expression: {ast.dump(node)[:60]}. Use numbers, "
            f"+ - * / // % ** and {', '.join(sorted(_FUNCS))}."
        )

    try:
        return ev(tree)
    except ZeroDivisionError:
        raise ToolInputError("Division by zero.") from None
    except (OverflowError, ValueError, TypeError) as exc:
        raise ToolInputError(f"Can't evaluate {expression!r}: {exc}") from None


def build_calculator_tools(ctx: ToolContext) -> list[Tool]:
    async def calculate(
        expression: Annotated[str, "Arithmetic expression, e.g. '0.17 * 2340' or 'sqrt(2) * 10'"],
    ) -> str:
        result = evaluate(expression)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 10)
        return f"{expression} = {result}"

    return [Tool(
        "calculate",
        "Evaluate an arithmetic expression exactly. Use it for any maths "
        "instead of working it out yourself. For percentages write 0.17 * 2340.",
        calculate,
    )]


# --- notes -----------------------------------------------------------------


def _fmt_dt(dt: datetime, ctx: ToolContext) -> str:
    return dt.astimezone(ctx.now().tzinfo).strftime("%a %d %b %Y %H:%M")


def build_notes_tools(ctx: ToolContext) -> list[Tool]:
    store = ToolStore(ctx.data_dir)

    async def add_note(text: Annotated[str, "The note to save, in the user's words"]) -> str:
        text = text.strip()
        if not text:
            raise ToolInputError("Note text is empty.")
        note = await asyncio.to_thread(store.add_note, ctx.user_id, ctx.persona_id, text, ctx.now())
        return f"Saved note #{note.id}: {text}"

    async def search_notes(
        query: Annotated[str, "Words to look for; empty string lists the latest notes"] = "",
    ) -> str:
        notes = await asyncio.to_thread(store.search_notes, ctx.user_id, ctx.persona_id, query)
        if not notes:
            return "No matching notes." if query.strip() else "No notes saved yet."
        return "\n".join(f"#{n.id} ({_fmt_dt(n.created_at, ctx)}): {n.text}" for n in notes)

    async def delete_note(note_id: Annotated[int, "The note number, e.g. 3 for #3"]) -> str:
        deleted = await asyncio.to_thread(store.delete_note, ctx.user_id, ctx.persona_id, note_id)
        return f"Deleted note #{note_id}." if deleted else f"There is no note #{note_id}."

    return [
        Tool("add_note", "Save a note for the user to look up later. Use it whenever "
             "they say 'note that', 'make a note', 'write down', 'remember that' or "
             "similar; save it right away, don't ask first.", add_note),
        Tool("search_notes", "Find the user's saved notes, newest first. Use it when "
             "they ask what notes they have or what they asked you to note.", search_notes),
        Tool("delete_note", "Delete one saved note by its number. Only when the user asks.",
             delete_note),
    ]


# --- reminders -------------------------------------------------------------


_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "few": 3}
_RELATIVE_RE = re.compile(
    r"^(?:(?:in|for|after)\s+)?(half\s+an?\s+hour|(\d+(?:\.\d+)?|a|an|one|two|three|four|five|ten|"
    r"fifteen|twenty|thirty|few)\s*([a-z]+))(?:\s+from\s+now)?$"
)
# Speech-to-text writes times as "9:22 p.m.", "9.22 PM" or "9 22 pm"; all
# of them are accepted (am/pm spellings are normalised in parse_when).
_CLOCK_RE = re.compile(r"^(?:at\s+)?(\d{1,2})(?:[:. ](\d{2}))?\s*(am|pm)?$")
_NAMED_TIMES = {"noon": "12:00", "midday": "12:00", "midnight": "00:00",
                "morning": "09:00", "afternoon": "15:00", "evening": "18:00",
                "tonight": "20:00"}


def _clock(text: str) -> tuple[int, int] | None:
    text = _NAMED_TIMES.get(text, text)
    m = _CLOCK_RE.match(text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ampm.startswith("p") else 0)
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_when(ctx: ToolContext, when: str) -> datetime:
    """Turns what people say into a due time, relative to `ctx.now()`.

    Takes a single free-text field on purpose: given two optional arguments
    (`minutes_from_now` or `at`), qwen2.5:7b often filled in neither and then
    claimed the reminder was set (measured 0/5 on one phrasing). Accepts
    "in 10 minutes", "2 hours", "half an hour", "6pm", "9:22 p.m.", "at 18:30",
    "tomorrow at 9am", "tonight", "noon" and ISO "2026-10-20T17:00". A clock
    time that has already passed today means tomorrow.
    """
    now = ctx.now()
    raw = (when or "").strip()
    text = re.sub(r"\s+", " ", raw.lower()).strip(" .")
    text = re.sub(r"(\d)\s*([ap])\.?\s?m\b\.?", r"\1\2m", text)  # "9:22 p.m." -> "9:22pm"
    if not text:
        raise ToolInputError(_WHEN_HELP)

    m = _RELATIVE_RE.match(text)
    # Only a known unit makes it relative: "6pm" also matches the pattern
    # (6, "pm") and must fall through to clock times below.
    if m and (m.group(1).startswith("half") or m.group(3) in _UNIT_SECONDS):
        if m.group(1).startswith("half"):
            return now + timedelta(minutes=30)
        amount = m.group(2)
        number = _WORD_NUMBERS.get(amount)
        number = float(amount) if number is None else number
        if number <= 0:
            raise ToolInputError(_WHEN_HELP)
        return now + timedelta(seconds=number * _UNIT_SECONDS[m.group(3)])

    day_offset, rest = 0, text
    for word, offset in (("tomorrow", 1), ("today", 0), ("tonight", 0)):
        if text == word or text.startswith(word + " "):
            day_offset = offset
            rest = text[len(word):].strip() or ("tonight" if word == "tonight" else "morning")
            if word == "today" and rest == "morning":
                rest = text  # "today" alone isn't a time
            break
    hm = _clock(rest)
    if hm is not None:
        due = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        due += timedelta(days=day_offset)
        if day_offset == 0 and due <= now:
            due += timedelta(days=1)  # "6pm" said at 7pm means tomorrow
        return due

    try:
        due = datetime.fromisoformat(raw)
    except ValueError:
        raise ToolInputError(_WHEN_HELP) from None
    if due.tzinfo is None:
        due = due.replace(tzinfo=now.tzinfo)
    if due <= now:
        raise ToolInputError(
            f"{raw} is in the past (it is now {now:%Y-%m-%d %H:%M}). Pick a future time."
        )
    return due


_WHEN_HELP = (
    "Can't tell when to remind. Pass `when` as plain words, e.g. 'in 10 minutes', "
    "'in 2 hours', '6pm', 'tomorrow at 9am', or an ISO time like '2026-10-20T17:00'."
)


def build_reminders_tools(ctx: ToolContext) -> list[Tool]:
    store = ToolStore(ctx.data_dir)

    async def set_reminder(
        what: Annotated[str, "What to remind the user about, e.g. 'drink water'; "
                             "just 'reminder' if the user didn't say"],
        when: Annotated[str, "When, in the user's words: 'in 10 minutes', '6pm', "
                             "'tomorrow at 9am'"],
    ) -> str:
        # `what`/`when` rather than `text`/`when`: with `text`, qwen2.5:7b
        # (Planner persona) filled only `when` in 5/5 runs.
        if not what.strip():
            raise ToolInputError("Say what to remind about in `what`, e.g. 'drink water'.")
        due = parse_when(ctx, when)
        reminder = await asyncio.to_thread(
            store.add_reminder, ctx.user_id, ctx.persona_id, what.strip(), due, ctx.now()
        )
        return f"Reminder #{reminder.id} set for {_fmt_dt(due, ctx)}: {reminder.text}"

    async def list_reminders() -> str:
        pending = await asyncio.to_thread(store.pending_reminders, ctx.user_id)
        if not pending:
            return "No upcoming reminders."
        return "\n".join(f"#{r.id} at {_fmt_dt(r.due_at, ctx)}: {r.text}" for r in pending)

    async def cancel_reminder(reminder_id: Annotated[int, "The reminder number"]) -> str:
        cancelled = await asyncio.to_thread(store.cancel_reminder, ctx.user_id, reminder_id)
        return (f"Cancelled reminder #{reminder_id}." if cancelled
                else f"There is no upcoming reminder #{reminder_id}.")

    return [
        Tool("set_reminder", "Set a reminder that will be shown to the user when it's due. "
             "Call it every time the user asks to be reminded, even if a similar "
             "reminder was set before. Always pass both `what` (e.g. 'drink water') "
             "and `when` in plain words as the user said it ('in 1 minute', '6pm', "
             "'tomorrow at 9am').", set_reminder),
        Tool("list_reminders", "List the user's upcoming reminders.", list_reminders),
        Tool("cancel_reminder", "Cancel an upcoming reminder by its number.", cancel_reminder),
    ]


# --- recall_memory ---------------------------------------------------------


def build_recall_tools(ctx: ToolContext) -> list[Tool]:
    async def recall_memory(
        query: Annotated[str, "What to look for in past conversations"],
    ) -> str:
        turns = await ctx.memory.query(ctx.user_id, ctx.persona_id, query, k=5)
        if not turns:
            return "Nothing relevant found in past conversations."
        return "\n".join(
            f"- {t.timestamp.astimezone(ctx.now().tzinfo):%d %b %Y} "
            f"{'user' if t.role == 'user' else 'you'}: {t.text}"
            for t in turns
        )

    return [Tool(
        "recall_memory",
        "Search your long-term memory of past conversations with this user, "
        "for when they ask what you remember or refer to something from before.",
        recall_memory,
    )]
