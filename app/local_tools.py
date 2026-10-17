"""Local Python tools: generate, review, register, and run user tools."""

from __future__ import annotations

import ast
import json
import re
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "data" / "tools"
REGISTRY_PATH = ROOT / "data" / "local_tools_registry.json"
SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,63}$")

# Minimal safe builtins for tool execution
_SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "zip": zip,
}


def _load_registry() -> list[dict]:
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_registry(items: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_local_tools() -> list[dict]:
    return _load_registry()


def _validate_source(source: str) -> None:
    if not source or len(source) > 80_000:
        raise ValueError("Tool source must be under 80 KB.")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error: {exc}") from exc
    # Ban obvious dangerous imports / calls at parse level
    banned = {"subprocess", "socket", "ctypes", "multiprocessing", "shutil"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned:
                    raise ValueError(f"Import of '{alias.name}' is not allowed in local tools.")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in banned:
                raise ValueError(f"Import from '{node.module}' is not allowed in local tools.")
    # Require a run function
    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body
    )
    if not has_run:
        raise ValueError("Tool source must define def run(**kwargs) or def run(...).")


def save_local_tool(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    if not SAFE_NAME.match(name):
        raise ValueError("Tool name: 2–64 letters, numbers, underscore; must start with a letter.")
    description = str(payload.get("description", "")).strip()[:500] or f"Local tool {name}"
    source = str(payload.get("source", ""))
    _validate_source(source)
    parameters = payload.get("parameters") or {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON object (JSON Schema).")

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOLS_DIR / f"{name}.py"
    # Version previous file if exists
    if path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path.rename(TOOLS_DIR / f"{name}.{stamp}.py.bak")
    path.write_text(source, encoding="utf-8")

    tool = {
        "id": payload.get("id") or str(uuid.uuid4()),
        "name": name,
        "description": description,
        "parameters": parameters,
        "path": str(path),
        "kind": "local_python",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    items = [t for t in _load_registry() if t.get("name") != name]
    items.insert(0, tool)
    _save_registry(items)
    return tool


def delete_local_tool(name: str) -> bool:
    items = _load_registry()
    kept = [t for t in items if t.get("name") != name]
    if len(kept) == len(items):
        return False
    _save_registry(kept)
    path = TOOLS_DIR / f"{name}.py"
    if path.is_file():
        path.unlink()
    return True


def get_tool(name: str) -> dict | None:
    return next((t for t in _load_registry() if t.get("name") == name), None)


def run_local_tool(name: str, arguments: dict | None = None) -> dict:
    tool = get_tool(name)
    if not tool:
        raise ValueError("Local tool not found.")
    path = Path(tool["path"])
    if not path.is_file():
        raise ValueError("Tool file is missing on disk.")
    source = path.read_text(encoding="utf-8")
    return run_source(source, arguments)


def run_source(source: str, arguments: dict | None = None) -> dict:
    """Run reviewed draft source without registering it as an approved tool."""
    _validate_source(source)
    args = arguments if isinstance(arguments, dict) else {}
    namespace = {"__builtins__": _SAFE_BUILTINS}
    # Allow a small set of pure modules
    import math
    import json as json_mod
    import re as re_mod
    import datetime as datetime_mod
    namespace.update({"math": math, "json": json_mod, "re": re_mod, "datetime": datetime_mod})
    try:
        exec(compile(source, "<local-tool-draft>", "exec"), namespace, namespace)
        fn = namespace.get("run")
        if not callable(fn):
            raise ValueError("run() is not callable.")
        result = fn(**args)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
    if isinstance(result, dict):
        return {"ok": True, **result} if "ok" not in result else result
    return {"ok": True, "result": result}


def tool_schemas_for_llm() -> list[dict]:
    schemas = []
    for tool in _load_registry():
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or tool["name"],
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return schemas


def generation_prompt(user_request: str) -> str:
    return f"""You are generating a local Python tool for AI Command Center.

Requirements:
- Define exactly one function: def run(**kwargs): ...
- Return a dict, preferably with keys ok, result, and optional files.
- Use only pure Python plus math, json, re, datetime. No network, no subprocess, no file system access outside what the host injects later.
- Keep the code short and readable.
- After the code block, output a JSON schema for parameters on its own line starting with SCHEMA:

User request:
{user_request}

Reply with:
1) A single ```python code fence containing the full tool file
2) One line: SCHEMA: {{ ... json schema for parameters ... }}
"""


def parse_generated_tool(assistant_text: str) -> dict:
    """Extract python source and optional schema from model output."""
    text = str(assistant_text or "")
    source = ""
    m = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        source = m.group(1).strip()
    else:
        m2 = re.search(r"```\s*([\s\S]*?)```", text)
        if m2:
            source = m2.group(1).strip()
    if not source:
        raise ValueError("No Python code block found in the model reply.")
    # Default name from def or generic
    name_match = re.search(r"^def\s+run\b", source, re.M)
    if not name_match:
        # allow module level and require run later via validate
        pass
    name = "custom_tool"
    nm = re.search(r"(?i)tool[_ ]name[:\s]+([A-Za-z][A-Za-z0-9_]{1,63})", text)
    if nm:
        name = nm.group(1)
    parameters = {"type": "object", "properties": {}, "additionalProperties": True}
    sm = re.search(r"SCHEMA:\s*(\{[\s\S]*\})", text)
    if sm:
        try:
            parameters = json.loads(sm.group(1))
        except json.JSONDecodeError:
            pass
    description = "User-generated local tool"
    dm = re.search(r"(?i)description[:\s]+(.+)", text)
    if dm:
        description = dm.group(1).strip()[:500]
    return {
        "name": name,
        "description": description,
        "source": source,
        "parameters": parameters,
    }
