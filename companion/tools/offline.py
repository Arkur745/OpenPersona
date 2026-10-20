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


def parse_due_time(ctx: ToolContext, minutes_from_now: int | None, at: str | None) -> datetime:
    now = ctx.now()
    if (minutes_from_now is None) == (at is None or not at.strip()):
        raise ToolInputError(
            "Give exactly one of minutes_from_now or at. For 'in 5 minutes' pass "
            "minutes_from_now=5; for a clock time pass at='2026-10-20T17:00'."
        )
    if minutes_from_now is not None:
        if minutes_from_now <= 0:
            raise ToolInputError("minutes_from_now must be positive.")
        return now + timedelta(minutes=minutes_from_now)
    try:
        due = datetime.fromisoformat(at.strip())
    except ValueError:
        raise ToolInputError(
            f"Can't read {at!r} as a time. Use ISO format like 2026-10-20T09:00."
        ) from None
    if due.tzinfo is None:
        due = due.replace(tzinfo=now.tzinfo)
    if due <= now:
        raise ToolInputError(
            f"{at} is in the past (it is now {now:%Y-%m-%dT%H:%M}). Pick a future time."
        )
    return due


def build_reminders_tools(ctx: ToolContext) -> list[Tool]:
    store = ToolStore(ctx.data_dir)

    async def set_reminder(
        text: Annotated[str, "What to remind the user about"],
        minutes_from_now: Annotated[int | None, "Delay in minutes, for 'in 10 minutes'"] = None,
        at: Annotated[str | None, "Exact local time in ISO format, e.g. 2026-10-20T09:00"] = None,
    ) -> str:
        if not text.strip():
            raise ToolInputError("Reminder text is empty.")
        due = parse_due_time(ctx, minutes_from_now, at)
        reminder = await asyncio.to_thread(
            store.add_reminder, ctx.user_id, ctx.persona_id, text.strip(), due, ctx.now()
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
             "reminder was set before. Use minutes_from_now for relative times "
             "(e.g. 1 for 'in a minute'), or at for a specific time (call "
             "get_datetime first if you need today's date).", set_reminder),
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
