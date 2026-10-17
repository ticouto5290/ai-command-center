"""Coding workspace tools: project-scoped list/read/write/run with basic safety."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PREFS_PATH = DATA / "coding_prefs.json"
PLAYBOOK_PATH = DATA / "coding_playbook.json"
MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 500_000
MAX_LIST_ENTRIES = 400
MAX_COMMAND_SECONDS = 45
BLOCKED_COMMAND = re.compile(
    r"(rm\s+-rf\s+/|mkfs|dd\s+if=|:(){:|fork\s*\(|shutdown|reboot|diskutil\s+erase)",
    re.IGNORECASE,
)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_prefs() -> dict:
    data = _load_json(PREFS_PATH, {})
    return {
        "project_root": data.get("project_root") or "",
        "selected_files": data.get("selected_files") or [],
    }


def set_project_root(path: str) -> dict:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError("Choose an existing folder on this Mac.")
    # Refuse obviously dangerous roots
    if candidate in (Path("/"), Path.home().resolve()):
        raise ValueError("Pick a project folder, not the home or system root.")
    prefs = get_prefs()
    prefs["project_root"] = str(candidate)
    prefs["selected_files"] = []
    _save_json(PREFS_PATH, prefs)
    return prefs


def set_selected_files(files: list) -> dict:
    prefs = get_prefs()
    root = _project_root()
    clean = []
    for item in files or []:
        if not isinstance(item, str):
            continue
        rel = item.replace("\\", "/").lstrip("/")
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            continue
        if target.is_file():
            clean.append(rel)
        if len(clean) >= 40:
            break
    prefs["selected_files"] = clean
    _save_json(PREFS_PATH, prefs)
    return prefs


def _project_root() -> Path:
    prefs = get_prefs()
    root = prefs.get("project_root") or ""
    if not root:
        raise ValueError("Set a project folder first.")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("The saved project folder is missing. Choose it again.")
    return path


def _safe_rel(path: str) -> Path:
    root = _project_root()
    rel = str(path or ".").replace("\\", "/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("Paths may not leave the project folder.")
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError("That path is outside the project folder.")
    return target


def list_dir(path: str = ".", depth: int = 2) -> dict:
    root = _project_root()
    target = _safe_rel(path)
    if not target.is_dir():
        raise ValueError("Not a directory.")
    depth = max(0, min(int(depth), 4))
    entries = []

    def walk(current: Path, rel: str, level: int) -> None:
        if len(entries) >= MAX_LIST_ENTRIES:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") and child.name not in (".gitignore", ".env.example"):
                continue
            child_rel = f"{rel}/{child.name}".lstrip("/") if rel != "." else child.name
            item = {
                "path": child_rel,
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
            }
            if child.is_file():
                try:
                    item["size"] = child.stat().st_size
                except OSError:
                    item["size"] = 0
            entries.append(item)
            if child.is_dir() and level < depth:
                walk(child, child_rel, level + 1)

    walk(target, "." if path in ("", ".") else str(path).replace("\\", "/").lstrip("/"), 0)
    return {"root": str(root), "path": path or ".", "entries": entries[:MAX_LIST_ENTRIES]}


def read_file(path: str) -> dict:
    target = _safe_rel(path)
    if not target.is_file():
        raise ValueError("File not found in the project.")
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(f"File is larger than {MAX_READ_BYTES // 1000} KB; open a smaller file.")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("File is not valid UTF-8 text.") from None
    return {"path": path, "size": size, "content": text}


def write_file(path: str, content: str, create_dirs: bool = True) -> dict:
    target = _safe_rel(path)
    if not isinstance(content, str):
        raise ValueError("Content must be text.")
    data = content.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ValueError(f"Content exceeds {MAX_WRITE_BYTES // 1000} KB limit.")
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    before = ""
    if target.is_file():
        try:
            before = target.read_text(encoding="utf-8")
        except Exception:
            before = ""
    target.write_text(content, encoding="utf-8")
    return {
        "path": path,
        "bytes": len(data),
        "created": before == "",
        "changed": before != content,
        "preview": content[:2000],
    }


def run_command(command: str) -> dict:
    root = _project_root()
    cmd = str(command or "").strip()
    if not cmd or len(cmd) > 2000:
        raise ValueError("Enter a short shell command.")
    if BLOCKED_COMMAND.search(cmd):
        raise ValueError("That command is blocked for safety.")
    # Disallow absolute path escapes outside project via cd /
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=MAX_COMMAND_SECONDS,
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {MAX_COMMAND_SECONDS}s.") from None
    return {
        "command": cmd,
        "cwd": str(root),
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[:20_000],
        "stderr": (completed.stderr or "")[:8_000],
        "duration_sec": round(time.time() - started, 2),
    }


def project_context() -> dict:
    """Build context string pieces for the coding agent."""
    prefs = get_prefs()
    if not prefs.get("project_root"):
        return {"prefs": prefs, "context": "No project folder selected yet."}
    root = Path(prefs["project_root"])
    listing = list_dir(".", depth=1)
    readme = ""
    for name in ("README.md", "readme.md", "README"):
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size < 30_000:
            readme = candidate.read_text(encoding="utf-8", errors="replace")[:8_000]
            break
    selected = []
    for rel in prefs.get("selected_files") or []:
        try:
            selected.append(read_file(rel))
        except Exception:
            continue
    git_log = ""
    if (root / ".git").is_dir():
        try:
            git_log = subprocess.run(
                ["git", "log", "-5", "--oneline"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout[:2000]
        except Exception:
            git_log = ""
    playbook = _load_json(PLAYBOOK_PATH, {"lessons": []})
    lessons = playbook.get("lessons") or []
    parts = [
        f"Project root: {root}",
        "Top-level entries: " + ", ".join(e["path"] for e in listing.get("entries", [])[:40]),
    ]
    if readme:
        parts.append("README excerpt:\n" + readme)
    if git_log:
        parts.append("Recent git log:\n" + git_log)
    if selected:
        for item in selected[:8]:
            parts.append(f"Selected file {item['path']}:\n{item['content'][:4000]}")
    if lessons:
        parts.append("Playbook lessons:\n- " + "\n- ".join(str(x)[:300] for x in lessons[-12:]))
    return {"prefs": prefs, "context": "\n\n".join(parts)}


def add_lesson(text: str) -> dict:
    note = str(text or "").strip()[:500]
    if not note:
        raise ValueError("Lesson text is empty.")
    data = _load_json(PLAYBOOK_PATH, {"lessons": []})
    lessons = data.get("lessons") or []
    lessons.append({"at": time.time(), "text": note})
    data["lessons"] = lessons[-50:]
    _save_json(PLAYBOOK_PATH, data)
    return data


CODING_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders under the project root (relative paths only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory, default ."},
                    "depth": {"type": "integer", "description": "0-4, default 2"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file under the project root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a UTF-8 text file under the project root. Prefer small focused edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a short shell command with cwd set to the project root. It runs with the current macOS user's permissions and is not OS-sandboxed.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_lesson",
            "description": "Store a short lesson about this project for future coding sessions.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


def run_coding_tool(name: str, arguments: dict) -> dict:
    args = arguments if isinstance(arguments, dict) else {}
    if name == "list_dir":
        return list_dir(args.get("path", "."), args.get("depth", 2))
    if name == "read_file":
        return read_file(str(args.get("path", "")))
    if name == "write_file":
        return write_file(str(args.get("path", "")), str(args.get("content", "")))
    if name == "run_command":
        return run_command(str(args.get("command", "")))
    if name == "add_lesson":
        return add_lesson(str(args.get("text", "")))
    raise ValueError(f"Unknown coding tool: {name}")
