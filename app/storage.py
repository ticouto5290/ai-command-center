"""Small local store for resumable chat sessions and generated images."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "data" / "command_center.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS generated_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        image_url TEXT NOT NULL,
        prompt TEXT NOT NULL,
        checkpoint TEXT NOT NULL,
        loras TEXT NOT NULL,
        created_at TEXT NOT NULL,
        seed INTEGER,
        width INTEGER,
        height INTEGER,
        steps INTEGER,
        cfg REAL,
        sampler TEXT
    )""")
    image_columns = {row[1] for row in connection.execute("PRAGMA table_info(generated_images)")}
    for name, declaration in (("seed", "INTEGER"), ("width", "INTEGER"), ("height", "INTEGER"),
                              ("steps", "INTEGER"), ("cfg", "REAL"), ("sampler", "TEXT")):
        if name not in image_columns:
            connection.execute(f"ALTER TABLE generated_images ADD COLUMN {name} {declaration}")
    connection.execute("""CREATE TABLE IF NOT EXISTS session_state (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        data TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS custom_tools (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        method TEXT NOT NULL,
        input_schema TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS mcp_servers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        command TEXT NOT NULL,
        args TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_session() -> dict:
    session_id = str(uuid.uuid4())
    now = _now()
    with _connect() as db:
        db.execute("INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", (session_id, "New conversation", now, now))
    return {"id": session_id, "title": "New conversation", "created_at": now, "updated_at": now}


def list_sessions() -> list[dict]:
    with _connect() as db:
        rows = db.execute("SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 50").fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict | None:
    with _connect() as db:
        row = db.execute("SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        messages = db.execute("SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        images = db.execute("SELECT filename, image_url AS url, prompt, checkpoint, loras, created_at, seed, width, height, steps, cfg, sampler FROM generated_images WHERE session_id = ? ORDER BY id DESC", (session_id,)).fetchall()
        state_row = db.execute("SELECT data FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
    session = dict(row)
    session["messages"] = [dict(message) for message in messages]
    session["images"] = [{**dict(image), "loras": json.loads(image["loras"])} for image in images]
    session["state"] = json.loads(state_row["data"]) if state_row else {}
    return session


def save_messages(session_id: str, messages: list[dict]) -> dict | None:
    normalized = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        normalized.append((role, content[:20000], _now()))
    if len(normalized) > 2000:
        normalized = normalized[-2000:]
    first_user = next((content for role, content, _ in normalized if role == "user"), "New conversation")
    title = " ".join(first_user.split())[:64] or "New conversation"
    now = _now()
    with _connect() as db:
        exists = db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not exists:
            return None
        previous_times = [row[0] for row in db.execute("SELECT created_at FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()]
        normalized = [(role, content, previous_times[index] if index < len(previous_times) else created) for index, (role, content, created) in enumerate(normalized)]
        db.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, session_id))
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.executemany("INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)", [(session_id, role, content, created) for role, content, created in normalized])
    return {"id": session_id, "title": title, "updated_at": now}


def save_image(session_id: str, image: dict) -> None:
    now = _now()
    with _connect() as db:
        exists = db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not exists:
            return
        db.execute(
            "INSERT INTO generated_images(session_id, filename, image_url, prompt, checkpoint, loras, created_at, seed, width, height, steps, cfg, sampler) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, image["filename"], image["url"], image["prompt"], image["checkpoint"], json.dumps(image["loras"]), now,
             image.get("seed"), image.get("width"), image.get("height"), image.get("steps"), image.get("cfg"), image.get("sampler")),
        )
        db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))


def save_state(session_id: str, data: dict) -> None:
    allowed = {"model", "image_checkpoint", "loras", "lora_strength", "negative_prompt", "image_prompt",
               "seed_mode", "seed", "seed_offset", "width", "height", "steps", "cfg", "sampler", "scheduler", "count", "pdf_tools"}
    state = {key: value for key, value in data.items() if key in allowed}
    with _connect() as db:
        if not db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
            return
        db.execute(
            "INSERT INTO session_state(session_id, data) VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET data = excluded.data",
            (session_id, json.dumps(state)),
        )


def list_custom_tools() -> list[dict]:
    with _connect() as db:
        rows = db.execute("SELECT * FROM custom_tools ORDER BY created_at DESC").fetchall()
    return [{**dict(row), "input_schema": json.loads(row["input_schema"])} for row in rows]


def save_custom_tool(tool: dict) -> dict:
    tool_id = tool.get("id") or str(uuid.uuid4())
    created_at = _now()
    with _connect() as db:
        db.execute(
            "INSERT INTO custom_tools(id, name, description, endpoint, method, input_schema, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description, endpoint=excluded.endpoint, method=excluded.method, input_schema=excluded.input_schema",
            (tool_id, tool["name"], tool["description"], tool["endpoint"], tool["method"], json.dumps(tool["input_schema"]), created_at),
        )
        row = db.execute("SELECT * FROM custom_tools WHERE id = ?", (tool_id,)).fetchone()
    return {**dict(row), "input_schema": json.loads(row["input_schema"])}


def delete_custom_tool(tool_id: str) -> bool:
    with _connect() as db:
        cursor = db.execute("DELETE FROM custom_tools WHERE id = ?", (tool_id,))
    return cursor.rowcount > 0


def list_mcp_servers() -> list[dict]:
    with _connect() as db:
        rows = db.execute("SELECT * FROM mcp_servers ORDER BY created_at DESC").fetchall()
    return [{**dict(row), "args": json.loads(row["args"])} for row in rows]


def save_mcp_server(server: dict) -> dict:
    server_id = server.get("id") or str(uuid.uuid4())
    created_at = _now()
    with _connect() as db:
        db.execute(
            "INSERT INTO mcp_servers(id, name, command, args, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, command=excluded.command, args=excluded.args",
            (server_id, server["name"], server["command"], json.dumps(server["args"]), created_at),
        )
        row = db.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
    return {**dict(row), "args": json.loads(row["args"])}


def delete_mcp_server(server_id: str) -> bool:
    with _connect() as db:
        cursor = db.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
    return cursor.rowcount > 0
