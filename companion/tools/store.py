"""SQLite storage for the notes and reminders tools.

One file, `<data_dir>/tools.sqlite3`, next to (not inside) the memory
store's directory. Scoping:

- **Notes** belong to one (user, persona) pair, the same isolation
  guarantee `MemoryStore` gives: the mentor never sees the planner's notes.
- **Reminders** belong to a user. Whichever persona the user is talking to
  delivers a due reminder, and the reminder records which persona set it.
  A reminder only reaching you if you happen to open the right persona
  would defeat the point.

Calls are synchronous and short. The tools run them through
`asyncio.to_thread` so they never block the event loop. Each call opens
its own connection, so the store is safe to share across threads.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_scope ON notes (user_id, persona_id);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    text TEXT NOT NULL,
    due_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    delivered_at TEXT,
    cancelled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS reminders_user ON reminders (user_id, delivered_at, cancelled);
"""


@dataclass(frozen=True)
class Note:
    id: int
    text: str
    created_at: datetime


@dataclass(frozen=True)
class Reminder:
    id: int
    persona_id: str
    text: str
    due_at: datetime


def _iso(dt: datetime) -> str:
    """Stored in UTC so string comparison orders correctly."""
    if dt.tzinfo is None:
        raise ValueError("datetimes stored by ToolStore must be timezone-aware")
    return dt.astimezone(UTC).isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


class ToolStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "tools.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _run(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            with conn:
                return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # --- notes -----------------------------------------------------------

    def add_note(self, user_id: str, persona_id: str, text: str, now: datetime) -> Note:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO notes (user_id, persona_id, text, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, persona_id, text, _iso(now)),
                )
                note_id = cur.lastrowid
        finally:
            conn.close()
        return Note(id=note_id, text=text, created_at=now)

    def search_notes(self, user_id: str, persona_id: str, query: str = "",
                     limit: int = 10) -> list[Note]:
        """Newest first. Matches notes containing every word of `query`
        (case-insensitive); an empty query returns the most recent notes."""
        sql = "SELECT id, text, created_at FROM notes WHERE user_id = ? AND persona_id = ?"
        params: list = [user_id, persona_id]
        for word in query.split():
            sql += " AND text LIKE ? ESCAPE '\\'"
            escaped = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [Note(r["id"], r["text"], _parse(r["created_at"]))
                for r in self._run(sql, tuple(params))]

    def delete_note(self, user_id: str, persona_id: str, note_id: int) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM notes WHERE id = ? AND user_id = ? AND persona_id = ?",
                    (note_id, user_id, persona_id),
                )
                return cur.rowcount == 1
        finally:
            conn.close()

    # --- reminders -------------------------------------------------------

    def add_reminder(self, user_id: str, persona_id: str, text: str, due_at: datetime,
                     now: datetime) -> Reminder:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO reminders (user_id, persona_id, text, due_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, persona_id, text, _iso(due_at), _iso(now)),
                )
                reminder_id = cur.lastrowid
        finally:
            conn.close()
        return Reminder(reminder_id, persona_id, text, due_at)

    def pending_reminders(self, user_id: str) -> list[Reminder]:
        """Not yet delivered and not cancelled, soonest first."""
        rows = self._run(
            "SELECT id, persona_id, text, due_at FROM reminders "
            "WHERE user_id = ? AND delivered_at IS NULL AND cancelled = 0 ORDER BY due_at",
            (user_id,),
        )
        return [Reminder(r["id"], r["persona_id"], r["text"], _parse(r["due_at"])) for r in rows]

    def cancel_reminder(self, user_id: str, reminder_id: int) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "UPDATE reminders SET cancelled = 1 "
                    "WHERE id = ? AND user_id = ? AND delivered_at IS NULL AND cancelled = 0",
                    (reminder_id, user_id),
                )
                return cur.rowcount == 1
        finally:
            conn.close()

    def take_due_reminders(self, user_id: str, now: datetime) -> list[Reminder]:
        """Returns reminders that are due and marks them delivered, in one
        transaction, so each is delivered exactly once even if two callers
        race (e.g. two Streamlit tabs)."""
        conn = self._connect()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT id, persona_id, text, due_at FROM reminders WHERE user_id = ? "
                    "AND delivered_at IS NULL AND cancelled = 0 AND due_at <= ? ORDER BY due_at",
                    (user_id, _iso(now)),
                ).fetchall()
                conn.executemany(
                    "UPDATE reminders SET delivered_at = ? WHERE id = ?",
                    [(_iso(now), r["id"]) for r in rows],
                )
        finally:
            conn.close()
        return [Reminder(r["id"], r["persona_id"], r["text"], _parse(r["due_at"])) for r in rows]
