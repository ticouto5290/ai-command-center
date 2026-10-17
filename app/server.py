"""Local-only web server for the AI Command Center prototype."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import select
import socket
import shutil
import urllib.parse
import uuid
from threading import Lock, Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import storage
import image_runtime
import pdf_tools


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = os.environ.get("AICC_HOST", "127.0.0.1")
PORT = int(os.environ.get("AICC_PORT", "8765"))
LM_STUDIO_MODELS = Path(os.environ.get("AICC_LM_STUDIO_MODELS", str(Path.home() / ".lmstudio" / "models"))).expanduser()
MODEL_REGISTRY = ROOT / "data" / "local_models.json"
MODEL_STORE = ROOT / "data" / "models"
LIBRARY_DIR = ROOT / "data" / "library"
MAX_LIBRARY_UPLOAD_BYTES = 30 * 1024 * 1024
LIBRARY_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".csv", ".json", ".png", ".jpg", ".jpeg", ".webp"}
LLAMA_RUNTIME = ROOT.parent / ".venv" / "llama-runtime"
LLAMA_CPU_SERVER = LLAMA_RUNTIME / "cpu" / "llama-server"
LLAMA_METAL_SERVER = LLAMA_RUNTIME / "metal" / "llama-server"
MLX_PYTHON = ROOT.parent / ".venv" / "mlx-runtime" / "bin" / "python"
MLX_LM_MODELS = LM_STUDIO_MODELS / "lmstudio-community"
with socket.socket() as _port_socket:
    _port_socket.bind((HOST, 0))
    LLAMA_RUNTIME_PORT = _port_socket.getsockname()[1]
LLAMA_RUNTIME_BASE = f"http://127.0.0.1:{LLAMA_RUNTIME_PORT}"
_model_lock = Lock()
_runtime_process: subprocess.Popen | None = None
_loaded_model_path: str | None = None
_mlx_process: subprocess.Popen | None = None
_loaded_mlx_path: str | None = None
_mlx_runtime_port: int | None = None
_download_lock = Lock()
_model_downloads: dict[str, dict] = {}


def registered_models() -> list[dict]:
    """Return models with MLX entries preferred (listed first) on native macOS."""
    try:
        saved = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved = []
    discovered = discover_local_models() + discover_mlx_models()
    available = {item["path"]: item for item in discovered}
    portable = {(item["name"], item["file"]): item for item in discovered}
    registered = []
    for item in saved:
        if not isinstance(item, dict):
            continue
        resolved = available.get(item.get("path"))
        if resolved is None:
            # A saved model can be mounted at a different path in Docker.
            # Match the same model folder and filename in the mounted catalog.
            resolved = portable.get((item.get("name"), item.get("file")))
        if resolved is not None and resolved not in registered:
            registered.append(resolved)
    registered_paths = {item["path"] for item in registered}
    extras = [item for item in discover_mlx_models() if item["path"] not in registered_paths]
    combined = registered + extras
    # Prefer Apple MLX models first so the UI default and dropdown favor them.
    mlx_models = [m for m in combined if m.get("backend") == "mlx"]
    other_models = [m for m in combined if m.get("backend") != "mlx"]
    return mlx_models + other_models


def register_gguf(model_path: str) -> dict:
    candidate = Path(model_path).expanduser().resolve()
    allowed_roots = [LM_STUDIO_MODELS.resolve(), MODEL_STORE.resolve()]
    if not any(root == candidate or root in candidate.parents for root in allowed_roots) or not candidate.is_file() or candidate.suffix.lower() != ".gguf":
        raise ValueError("Choose a GGUF model from LM Studio or this app's model folder.")
    models = registered_models()
    model = next((item for item in discover_local_models() if item["path"] == str(candidate)), None)
    if model is None:
        raise ValueError("That GGUF model could not be found.")
    if not any(item["path"] == model["path"] for item in models):
        models.append(model)
    MODEL_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY.write_text(json.dumps(models, indent=2), encoding="utf-8")
    return model


def runtime_is_ready() -> bool:
    try:
        with urlopen(f"{LLAMA_RUNTIME_BASE}/health", timeout=0.5) as response:
            return response.status == 200
    except (URLError, TimeoutError, OSError):
        return False


def mlx_runtime_is_ready() -> bool:
    if _mlx_runtime_port is None:
        return False
    try:
        with urlopen(f"http://127.0.0.1:{_mlx_runtime_port}/v1/models", timeout=0.5) as response:
            return response.status == 200
    except (URLError, TimeoutError, OSError):
        return False


def stop_model_runtime() -> None:
    global _runtime_process, _loaded_model_path
    if _runtime_process is not None and _runtime_process.poll() is None:
        _runtime_process.terminate()
        try:
            _runtime_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _runtime_process.kill()
    _runtime_process = None
    _loaded_model_path = None


def stop_mlx_runtime() -> None:
    global _mlx_process, _loaded_mlx_path, _mlx_runtime_port
    if _mlx_process is not None and _mlx_process.poll() is None:
        _mlx_process.terminate()
        try:
            _mlx_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _mlx_process.kill()
    _mlx_process = None
    _loaded_mlx_path = None
    _mlx_runtime_port = None


def start_model_runtime(model_path: str, server_binary: Path, gpu_layers: int) -> bool:
    global _runtime_process, _loaded_model_path
    if not server_binary.is_file():
        return False
    log_dir = ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "model-runtime.log"
    with log_path.open("ab") as log_file:
        _runtime_process = subprocess.Popen(
            [str(server_binary), "--model", model_path, "--host", "127.0.0.1", "--port", str(LLAMA_RUNTIME_PORT), "--ctx-size", "2048", "--n-gpu-layers", str(gpu_layers), "--no-webui"],
            stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, start_new_session=True,
        )
    for _ in range(180):
        if runtime_is_ready():
            _loaded_model_path = model_path
            return True
        if _runtime_process.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def get_local_model(model_path: str) -> str:
    global _loaded_model_path
    with _model_lock:
        if _loaded_model_path == model_path and runtime_is_ready():
            return "local"
        stop_mlx_runtime()
        stop_model_runtime()
        metal_binary = LLAMA_METAL_SERVER if LLAMA_METAL_SERVER.is_file() else None
        if metal_binary and start_model_runtime(model_path, metal_binary, -1):
            return "local"
        stop_model_runtime()
        if start_model_runtime(model_path, LLAMA_CPU_SERVER, 0):
            return "local"
        stop_model_runtime()
        raise RuntimeError("The in-app model engine could not load this model. Check the local model runtime log and try again.")


def get_mlx_model(model_path: str) -> str:
    global _mlx_process, _loaded_mlx_path, _mlx_runtime_port
    if not MLX_PYTHON.is_file():
        raise RuntimeError("The app's Apple MLX model support is still setting up. Restart the app to finish setup.")
    with _model_lock:
        if _loaded_mlx_path == model_path and mlx_runtime_is_ready():
            return "mlx"
        stop_model_runtime()
        stop_mlx_runtime()
        with socket.socket() as probe:
            probe.bind((HOST, 0))
            _mlx_runtime_port = probe.getsockname()[1]
        log_path = ROOT / "data" / "mlx-runtime.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            _mlx_process = subprocess.Popen(
                [str(MLX_PYTHON), "-m", "mlx_lm", "server", "--model", model_path,
                 "--host", "127.0.0.1", "--port", str(_mlx_runtime_port)],
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, start_new_session=True,
            )
        for _ in range(900):
            if mlx_runtime_is_ready():
                _loaded_mlx_path = model_path
                return "mlx"
            if _mlx_process.poll() is not None:
                stop_mlx_runtime()
                raise RuntimeError("The local MLX model could not start. Open the app's MLX engine log for details.")
            time.sleep(1)
        stop_mlx_runtime()
        raise RuntimeError("The local MLX model took too long to load. Check the MLX engine log and available memory.")


def model_by_name(name: str) -> dict | None:
    return next((item for item in registered_models() if item["name"] == name), None)


def model_chat(messages: list[dict], backend: str = "llama.cpp", tools: list[dict] | None = None) -> dict:
    # Leave enough room for code responses to finish as complete files instead
    # of silently cutting off at the previous 512-token chat limit.
    payload = {"model": "default_model" if backend == "mlx" else "local", "messages": messages, "max_tokens": 2048, "temperature": 0.7}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    base = f"http://127.0.0.1:{_mlx_runtime_port}" if backend == "mlx" else LLAMA_RUNTIME_BASE
    request = Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def library_items() -> list[dict]:
    items = []
    for folder, kind in ((LIBRARY_DIR, "file"), (image_runtime.IMAGE_DIR, "image")):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            try:
                if not path.is_file() or path.name.startswith("."):
                    continue
                items.append({"id": path.name, "filename": path.name, "kind": kind,
                              "content_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                              "size": path.stat().st_size, "modified": path.stat().st_mtime,
                              "url": f"/api/library/view/{urllib.parse.quote(path.name)}"})
            except OSError:
                continue
    return sorted(items, key=lambda item: item["modified"], reverse=True)[:500]


def library_file(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    candidate = (LIBRARY_DIR / filename).resolve()
    if candidate.parent == LIBRARY_DIR.resolve() and candidate.is_file():
        return candidate
    image_candidate = (image_runtime.IMAGE_DIR / filename).resolve()
    if image_candidate.parent == image_runtime.IMAGE_DIR.resolve() and image_candidate.is_file():
        return image_candidate
    return None


PDF_TOOL_SCHEMAS = {
    "create_pdf": {"type": "function", "function": {"name": "create_pdf", "description": "Create a polished PDF document from a title and supplied text, then save it in the local Library.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Document title"}, "text": {"type": "string", "description": "Document body text"}}, "required": ["title", "text"]}}},
    "read_pdf": {"type": "function", "function": {"name": "read_pdf", "description": "Read and extract text from a PDF already saved in the Library. Use the exact filename listed in the Library context.", "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Exact PDF filename from the Library"}}, "required": ["filename"]}}},
    "edit_pdf": {"type": "function", "function": {"name": "edit_pdf", "description": "Edit a PDF in the Library by creating a revised copy with an appended note page and/or an updated title. The source is preserved.", "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Exact source PDF filename from the Library"}, "note": {"type": "string", "description": "Optional text to append as a new note page"}, "title": {"type": "string", "description": "Optional replacement PDF title metadata"}}, "required": ["filename"]}}},
}


def pdf_tool_context(enabled: list[str]) -> tuple[list[dict], str]:
    tools = [PDF_TOOL_SCHEMAS[name] for name in enabled if name in PDF_TOOL_SCHEMAS]
    if not tools:
        return [], ""
    files = [item["filename"] for item in library_items() if item["filename"].lower().endswith(".pdf")]
    context = "You have local PDF tools enabled. Use them when the user's request calls for creating, reading, or editing a PDF. Only access PDFs in the Library; never invent filenames. Editing makes a revised copy and preserves the source.\nPDFs currently in Library: " + (", ".join(files) if files else "none") + "."
    return tools, context


def run_pdf_tool(name: str, arguments: dict) -> dict:
    if name == "create_pdf":
        return pdf_tools.create_pdf(arguments.get("title"), arguments.get("text"))
    if name == "read_pdf":
        return pdf_tools.read_pdf(arguments.get("filename"))
    if name == "edit_pdf":
        return pdf_tools.edit_pdf(arguments.get("filename"), arguments.get("note"), arguments.get("title"))
    raise ValueError("This PDF tool is not available.")


def chat_with_pdf_tools(messages: list[dict], backend: str, enabled: list[str]) -> tuple[dict, list[dict]]:
    tool_schemas, instructions = pdf_tool_context(enabled)
    if not tool_schemas:
        return model_chat(messages, backend), []
    working = list(messages)
    working.insert(0, {"role": "system", "content": instructions})
    actions = []
    for _ in range(4):
        result = model_chat(working, backend, tool_schemas)
        choice = (result.get("choices") or [{}])[0]
        assistant = choice.get("message") or {}
        calls = assistant.get("tool_calls") or []
        if not calls and isinstance(assistant.get("content"), str):
            # Qwen chat templates may serialize tool requests as tagged JSON
            # instead of OpenAI's tool_calls field.
            content = assistant["content"]
            candidates = []
            for pattern in (r"<tool>\s*(.*?)\s*</tool>", r"<response>\s*(.*?)\s*</response>", r"<tools>\s*(.*?)\s*</tools>"):
                candidates.extend(match.group(1) for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
            candidates += [match.group(1) for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content, re.IGNORECASE)]
            if not candidates and content.lstrip().startswith("{"):
                candidates.append(content.strip())
            for candidate in candidates:
                try:
                    request_data = json.loads(candidate)
                    name = request_data.get("name") or (request_data.get("function") or {}).get("name")
                    arguments = request_data.get("arguments", (request_data.get("function") or {}).get("arguments", {}))
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if name in enabled and isinstance(arguments, dict):
                        calls.append({"id": uuid.uuid4().hex, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}})
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue
        if not calls:
            return result, actions
        working.append(assistant)
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if name not in enabled or not isinstance(arguments, dict):
                    raise ValueError("The requested PDF action is not enabled for this chat.")
                outcome = run_pdf_tool(name, arguments)
                actions.append({"name": name, "result": outcome})
            except Exception as error:
                outcome = {"error": str(error)[:1000]}
            working.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(outcome, ensure_ascii=False)[:65000]})
    raise RuntimeError("The assistant reached the PDF tool call limit for one reply.")


def validate_custom_tool(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    description = str(payload.get("description", "")).strip()
    endpoint = str(payload.get("endpoint", "")).strip()
    method = str(payload.get("method", "POST")).upper()
    schema = payload.get("input_schema", {"type": "object", "properties": {}})
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{1,63}", name):
        raise ValueError("Use a tool name with 2–64 letters, numbers, underscores, or dashes.")
    if not description or len(description) > 1000:
        raise ValueError("Add a description of up to 1,000 characters.")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or len(endpoint) > 2000:
        raise ValueError("Use a valid http or https endpoint.")
    if method not in ("GET", "POST"):
        raise ValueError("Choose GET or POST for the first version of custom tools.")
    if not isinstance(schema, dict) or schema.get("type") != "object" or len(json.dumps(schema)) > 16000:
        raise ValueError("Input schema must be a JSON object schema under 16 KB.")
    return {"id": payload.get("id"), "name": name, "description": description, "endpoint": endpoint, "method": method, "input_schema": schema}


def run_custom_tool(tool: dict, arguments: dict) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    endpoint = tool["endpoint"]
    method = tool["method"]
    if method == "GET":
        separator = "&" if "?" in endpoint else "?"
        url = endpoint + (separator + urllib.parse.urlencode(arguments, doseq=True) if arguments else "")
        request = Request(url, method="GET")
    else:
        request = Request(endpoint, data=json.dumps(arguments).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=20) as response:
        content = response.read(1_000_000).decode("utf-8", errors="replace")
        try:
            return {"status": response.status, "result": json.loads(content)}
        except json.JSONDecodeError:
            return {"status": response.status, "result": content}


def mcp_stdio_request(server: dict, method: str, params: dict | None = None, timeout: float = 8.0) -> dict:
    command = server["command"]
    args = server.get("args", [])
    if not command or len(command) > 256 or not isinstance(args, list) or len(args) > 64 or any(not isinstance(arg, str) or len(arg) > 2048 for arg in args):
        raise ValueError("MCP server command or arguments are invalid.")
    process = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(message: dict) -> None:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def receive(request_id: int) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            line = process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message", "MCP request failed."))
                return message.get("result", {})
        raise RuntimeError("MCP server did not respond in time. Check its command and that it writes protocol messages to stdout.")

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "AI Command Center", "version": "0.1"}}})
        initialized = receive(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}})
        result = receive(2)
        return {"server": initialized.get("serverInfo", {}), **result}
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()


def discover_local_models() -> list[dict]:
    """List GGUF files in the app's model store and LM Studio's model folder."""
    models = []
    seen: set[str] = set()
    for root in (MODEL_STORE, LM_STUDIO_MODELS):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.gguf")):
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            in_app_store = root == MODEL_STORE
            models.append({
                "name": path.stem if in_app_store else path.parent.name,
                "file": path.name,
                "path": str(path),
                "size": path.stat().st_size,
                "source": "Command Center" if in_app_store else "LM Studio",
                "backend": "llama.cpp",
            })
            if len(models) >= 100:
                return models
    return models


def discover_mlx_models() -> list[dict]:
    """Find complete MLX model folders that can run through the app's MLX engine."""
    models = []
    if sys.platform != "darwin":
        return models
    if not MLX_LM_MODELS.is_dir():
        return models
    for folder in sorted(MLX_LM_MODELS.iterdir()):
        if not folder.is_dir() or "mlx" not in folder.name.casefold() or not folder.joinpath("config.json").is_file():
            continue
        shards = list(folder.glob("*.safetensors"))
        if not shards:
            continue
        size = sum(item.stat().st_size for item in shards if item.is_file())
        models.append({
            "name": folder.name,
            "file": "MLX · " + ", ".join(item.name for item in shards[:2]),
            "path": str(folder.resolve()),
            "size": size,
            "source": "LM Studio",
            "backend": "mlx",
        })
    return models[:100]


def discover_lm_studio_models() -> list[dict]:
    """Compatibility helper for the model import list."""
    registered_paths = {item["path"] for item in registered_models()}
    return [item for item in discover_local_models() if item["path"] not in registered_paths]


def valid_hf_repo(repo_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", repo_id))


def huggingface_gguf_files(repo_id: str) -> list[str]:
    if not valid_hf_repo(repo_id):
        raise ValueError("Enter a Hugging Face model repository such as Qwen/Qwen2.5-7B-Instruct-GGUF.")
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo_id, safe="/") + "?" + urllib.parse.urlencode({"expand[]": "siblings"})
    request = Request(url, headers={"User-Agent": "AI-Command-Center/0.1"})
    with urlopen(request, timeout=30) as response:
        info = json.loads(response.read().decode("utf-8"))
    files = [item.get("rfilename", "") for item in info.get("siblings", []) if isinstance(item, dict)]
    gguf_files = [
        name for name in files
        if name.lower().endswith(".gguf") and not re.search(r"-\d{5}-of-\d{5}\.gguf$", name, re.IGNORECASE)
    ]
    if not gguf_files:
        raise ValueError("This public repository has no single-file GGUF models. Choose a GGUF repository with one complete quantized file.")
    return gguf_files[:250]


def download_hf_model(job_id: str, repo_id: str, filename: str) -> None:
    safe_repo = repo_id.replace("/", "--")
    destination = (MODEL_STORE / safe_repo / filename).resolve()
    root = MODEL_STORE.resolve()
    if root not in destination.parents or destination.suffix.lower() != ".gguf":
        with _download_lock:
            _model_downloads[job_id].update({"status": "error", "error": "The selected model file path is invalid."})
        return
    partial = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded_file = urllib.parse.quote(filename, safe="/")
        url = f"https://huggingface.co/{urllib.parse.quote(repo_id, safe='/')}/resolve/main/{encoded_file}"
        request = Request(url, headers={"User-Agent": "AI-Command-Center/0.1"})
        with urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0") or 0)
            if total > 32 * 1024**3:
                raise ValueError("This file is over the 32 GB download limit for one model file.")
            if total and shutil.disk_usage(destination.parent).free < total:
                raise ValueError("There is not enough free space to download this model into the app.")
            with _download_lock:
                _model_downloads[job_id]["total"] = total
            downloaded = 0
            with partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    with _download_lock:
                        _model_downloads[job_id]["downloaded"] = downloaded
                output.flush()
        if total and downloaded != total:
            raise RuntimeError("The download ended before the complete model file arrived. Try again.")
        partial.replace(destination)
        model = register_gguf(str(destination))
        with _download_lock:
            _model_downloads[job_id].update({"status": "complete", "model": model["name"], "downloaded": downloaded})
    except Exception as error:
        partial.unlink(missing_ok=True)
        with _download_lock:
            _model_downloads[job_id].update({"status": "error", "error": str(error)[:800]})


def upload_gguf_file(file_name: str, content_length: int, source_stream) -> dict:
    safe_name = Path(urllib.parse.unquote(file_name)).name
    if not safe_name or safe_name in (".", "..") or not safe_name.lower().endswith(".gguf"):
        raise ValueError("Choose a .gguf model file.")
    if content_length <= 0 or content_length > 32 * 1024**3:
        raise ValueError("Model files must be between 1 byte and 32 GB.")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(MODEL_STORE).free < content_length:
        raise ValueError("There is not enough free space to copy this model into the app.")
    target_dir = MODEL_STORE / f"imported-{uuid.uuid4().hex[:10]}"
    target_dir.mkdir()
    target = target_dir / safe_name
    try:
        with target.open("wb") as output:
            remaining = content_length
            while remaining:
                chunk = source_stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("The upload ended before the complete model file arrived.")
                output.write(chunk)
                remaining -= len(chunk)
        return register_gguf(str(target))
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise


def generate_image(payload: dict) -> dict:
    """Route to PyTorch MPS or stable-diffusion.cpp based on engine preference."""
    engine = str(payload.get("engine", "auto")).strip().lower()
    if engine in ("pytorch", "pytorch-mps", "mps"):
        import image_pytorch
        return image_pytorch.generate_image(payload)
    if engine in ("sd", "sd_cpp", "sd-cpp", "cpp"):
        return image_runtime.generate_image(payload)
    # auto: prefer PyTorch when available on native macOS
    try:
        import image_pytorch
        status = image_pytorch.engine_status()
        if status.get("available") and not status.get("docker"):
            try:
                result = image_pytorch.generate_image(payload)
                result.setdefault("engine", "pytorch-mps" if status.get("device") == "mps" else "pytorch")
                return result
            except Exception:
                pass
    except Exception:
        pass
    return image_runtime.generate_image(payload)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "AICommandCenter/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request is too large.")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/library":
            self.send_json(200, {"files": library_items()})
            return
        if self.path.startswith("/api/library/view/"):
            filename = urllib.parse.unquote(self.path.removeprefix("/api/library/view/").split("?", 1)[0])
            path = library_file(filename)
            if not path:
                self.send_error(404, "Library file not found")
                return
            content = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix.lower() == ".html":
                content_type = "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path == "/api/tools":
            self.send_json(200, {"tools": storage.list_custom_tools(), "mcp_servers": storage.list_mcp_servers()})
            return
        if self.path.startswith("/api/mcp/servers/") and self.path.endswith("/tools"):
            server_id = self.path.removeprefix("/api/mcp/servers/").removesuffix("/tools").strip("/")
            server = next((item for item in storage.list_mcp_servers() if item["id"] == server_id), None)
            if not server:
                self.send_json(404, {"error": "MCP server not found."})
                return
            try:
                result = mcp_stdio_request(server, "tools/list")
                self.send_json(200, result)
            except (OSError, RuntimeError, ValueError) as error:
                self.send_json(502, {"error": str(error)})
            return
        if self.path == "/api/sessions":
            self.send_json(200, {"sessions": storage.list_sessions()})
            return
        if self.path.startswith("/api/sessions/"):
            session_id = self.path.removeprefix("/api/sessions/")
            session = storage.get_session(session_id)
            self.send_json(200 if session else 404, session or {"error": "Session not found."})
            return
        if self.path == "/api/image-catalog":
            self.send_json(200, image_runtime.image_catalog())
            return
        if self.path.startswith("/api/images/view/"):
            filename = urllib.parse.unquote(self.path.removeprefix("/api/images/view/").split("?", 1)[0])
            if not filename or Path(filename).name != filename or not filename.startswith("image-"):
                self.send_error(400, "Invalid image path")
                return
            stored_image = (image_runtime.IMAGE_DIR / filename).resolve()
            if image_runtime.IMAGE_DIR.resolve() not in stored_image.parents or not stored_image.is_file():
                self.send_error(404, "Image not found")
                return
            content = stored_image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path == "/api/status":
            self.send_json(200, self.local_status())
            return
        if self.path == "/api/models":
            runtime_installed = LLAMA_CPU_SERVER.is_file() or LLAMA_METAL_SERVER.is_file() or MLX_PYTHON.is_file()
            models = registered_models()
            self.send_json(200, {
                "available": runtime_is_ready() or mlx_runtime_is_ready(),
                "runtime_installed": runtime_installed,
                "models": models,
                "importable": discover_lm_studio_models(),
            })
            return
        if self.path.startswith("/api/huggingface/files?"):
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            repo_id = query.get("repo", [""])[0].strip()
            try:
                files = huggingface_gguf_files(repo_id)
                self.send_json(200, {"repo": repo_id, "files": files})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                self.send_json(502, {"error": f"Could not read this public Hugging Face model repository: {error}"})
            return
        if self.path.startswith("/api/model-downloads/"):
            job_id = self.path.removeprefix("/api/model-downloads/").split("?", 1)[0]
            with _download_lock:
                job = _model_downloads.get(job_id)
            self.send_json(200 if job else 404, job or {"error": "Model download not found."})
            return
        self.serve_static()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/library/upload":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_LIBRARY_UPLOAD_BYTES:
                    raise ValueError("Choose a file under 30 MB.")
                name = urllib.parse.unquote(self.headers.get("X-File-Name", "")).strip()
                basename = Path(name).name
                suffix = Path(basename).suffix.lower()
                if not basename or suffix not in LIBRARY_UPLOAD_EXTENSIONS:
                    raise ValueError("Choose a PDF, text, HTML, CSV, JSON, or image file.")
                content = self.rfile.read(length)
                if len(content) != length:
                    raise ValueError("The file upload did not finish.")
                LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(basename).stem).strip(" .")[:70] or "file"
                destination = LIBRARY_DIR / f"{safe_name}-{uuid.uuid4().hex[:8]}{suffix}"
                destination.write_bytes(content)
                self.send_json(201, {"file": next((item for item in library_items() if item["filename"] == destination.name), None)})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except OSError as error:
                self.send_json(500, {"error": f"Could not save this file: {error}"})
            return
        if self.path in ("/api/pdf/create", "/api/pdf/read", "/api/pdf/edit"):
            try:
                payload = self.read_json()
                operation = self.path.rsplit("/", 1)[-1]
                if operation == "create":
                    result = pdf_tools.create_pdf(payload.get("title"), payload.get("text"))
                elif operation == "read":
                    result = pdf_tools.read_pdf(payload.get("filename"))
                else:
                    result = pdf_tools.edit_pdf(payload.get("filename"), payload.get("note"), payload.get("title"))
                self.send_json(200, result)
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": f"PDF {self.path.rsplit('/', 1)[-1]} failed: {str(error)[:500]}"})
            return
        if self.path == "/api/tools":
            try:
                payload = validate_custom_tool(self.read_json())
                self.send_json(201, {"tool": storage.save_custom_tool(payload)})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(409, {"error": f"Could not save tool: {error}"})
            return
        if self.path.startswith("/api/tools/") and self.path.endswith("/run"):
            tool_id = self.path.removeprefix("/api/tools/").removesuffix("/run").strip("/")
            tool = next((item for item in storage.list_custom_tools() if item["id"] == tool_id), None)
            if not tool:
                self.send_json(404, {"error": "Tool not found."})
                return
            try:
                result = run_custom_tool(tool, self.read_json().get("arguments", {}))
                self.send_json(200, {"result": result})
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                self.send_json(502, {"error": str(error)})
            return
        if self.path == "/api/mcp/servers":
            try:
                payload = self.read_json()
                name = str(payload.get("name", "")).strip()
                command = str(payload.get("command", "")).strip()
                args = payload.get("args", [])
                if not name or len(name) > 100 or not command or len(command) > 256:
                    raise ValueError("Enter a server name and command.")
                if not isinstance(args, list) or len(args) > 64 or any(not isinstance(arg, str) or len(arg) > 2048 for arg in args):
                    raise ValueError("Arguments must be a JSON list of up to 64 text values.")
                server = storage.save_mcp_server({"id": payload.get("id"), "name": name, "command": command, "args": args})
                self.send_json(201, {"server": server})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            return
        if self.path == "/api/sessions":
            self.send_json(201, storage.create_session())
            return
        if self.path == "/api/images/generate":
            try:
                payload = self.read_json()
                with _model_lock:
                    stop_model_runtime()
                    stop_mlx_runtime()
                    result = generate_image(payload)
                result["prompt"] = str(payload.get("prompt", ""))
                session_id = payload.get("session_id")
                if isinstance(session_id, str):
                    for image in result["images"]:
                        storage.save_image(session_id, {**result, **image})
                self.send_json(200, result)
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except (HTTPError, RuntimeError, URLError, OSError, json.JSONDecodeError) as error:
                detail = error.read().decode("utf-8", errors="replace") if isinstance(error, HTTPError) else str(error)
                self.send_json(502, {"error": detail[:1200] or "Image generation failed."})
            return
        if self.path == "/api/models/import":
            try:
                payload = self.read_json()
                model = register_gguf(payload.get("path", ""))
                self.send_json(200, {"ok": True, "model": model["name"]})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                self.send_json(500, {"error": str(error) or "Model import failed."})
            return
        if self.path == "/api/models/upload":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                model = upload_gguf_file(self.headers.get("X-File-Name", ""), content_length, self.rfile)
                self.send_json(201, {"ok": True, "model": model["name"]})
            except ValueError as error:
                self.close_connection = True
                self.send_json(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                self.send_json(500, {"error": str(error) or "Could not add this model file."})
            return
        if self.path == "/api/models/huggingface/download":
            try:
                payload = self.read_json()
                repo_id = str(payload.get("repo", "")).strip()
                filename = str(payload.get("file", "")).strip()
                if filename not in huggingface_gguf_files(repo_id):
                    raise ValueError("Choose one of the GGUF files listed for this Hugging Face repository.")
                job_id = uuid.uuid4().hex
                with _download_lock:
                    _model_downloads[job_id] = {"status": "downloading", "repo": repo_id, "file": filename, "downloaded": 0, "total": 0}
                Thread(target=download_hf_model, args=(job_id, repo_id, filename), daemon=True).start()
                self.send_json(202, {"job_id": job_id})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                self.send_json(502, {"error": f"Could not start the Hugging Face download: {error}"})
            return
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Not found."})
            return
        try:
            payload = self.read_json()
            model = payload.get("model")
            messages = payload.get("messages")
            if not isinstance(model, str) or not model.strip():
                self.send_json(400, {"error": "Choose a local model first."})
                return
            if not isinstance(messages, list) or not messages:
                self.send_json(400, {"error": "A message is required."})
                return
            selected = model_by_name(model)
            if not selected:
                self.send_json(400, {"error": "Choose a model added to this app first."})
                return
            backend = selected.get("backend", "llama.cpp")
            if backend == "mlx":
                get_mlx_model(selected["path"])
            else:
                get_local_model(selected["path"])
            enabled_tools = payload.get("tools", [])
            if not isinstance(enabled_tools, list):
                enabled_tools = []
            enabled_tools = [name for name in enabled_tools if isinstance(name, str) and name in PDF_TOOL_SCHEMAS]
            result, tool_actions = chat_with_pdf_tools(messages, backend, enabled_tools)
            self.send_json(200, {"message": {"content": result["choices"][0]["message"].get("content", "")}, "tool_actions": tool_actions})
        except ValueError as error:
            detail = str(error)
            if "llama_context" in detail or "command queue" in detail:
                self.send_json(503, {
                    "error": "The model file is available, but macOS did not let this app launch create a Metal graphics context. Start Command Center with run.command from your desktop, then refresh this page."
                })
            else:
                self.send_json(400, {"error": detail})
        except RuntimeError as error:
            self.send_json(503, {"error": str(error)})
        except Exception as error:
            self.send_json(502, {"error": f"The in-app model could not start: {str(error)[:500]}"})

    def do_PUT(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/sessions/"):
            self.send_json(404, {"error": "Not found."})
            return
        session_id = self.path.removeprefix("/api/sessions/")
        try:
            payload = self.read_json()
            messages = payload.get("messages")
            if not isinstance(messages, list):
                self.send_json(400, {"error": "Messages must be a list."})
                return
            session = storage.save_messages(session_id, messages)
            state = payload.get("state")
            if session and isinstance(state, dict):
                storage.save_state(session_id, state)
            self.send_json(200 if session else 404, session or {"error": "Session not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path.startswith("/api/tools/"):
            tool_id = self.path.removeprefix("/api/tools/")
            removed = storage.delete_custom_tool(tool_id)
            self.send_json(200 if removed else 404, {"ok": removed})
            return
        if self.path.startswith("/api/mcp/servers/"):
            server_id = self.path.removeprefix("/api/mcp/servers/")
            removed = storage.delete_mcp_server(server_id)
            self.send_json(200 if removed else 404, {"ok": removed})
            return
        self.send_json(404, {"error": "Not found."})

    @staticmethod
    def local_status() -> dict:
        return {"runtime": "In-app model engine", "available": runtime_is_ready()}

    def serve_static(self) -> None:
        requested = "/index.html" if self.path in ("/", "") else self.path.split("?", 1)[0]
        file_path = (STATIC / requested.lstrip("/")).resolve()
        if STATIC.resolve() not in file_path.parents and file_path != STATIC.resolve():
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"AI Command Center is available at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AI Command Center.")
    finally:
        stop_model_runtime()
        stop_mlx_runtime()
        server.server_close()


if __name__ == "__main__":
    main()
