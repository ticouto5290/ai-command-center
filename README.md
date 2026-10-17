# AI Command Center

A local-first AI workspace for chat, image generation, coding, tools, and automations.

**Primary path: native macOS on Apple Silicon.** Docker is optional and limited to CPU-only inference.

## Run on this Mac (recommended)

Requires macOS on Apple Silicon, Python 3.11+, Git, and CMake. Double-click `run.command` to set up the app-managed runtimes the first time and open the app at <http://127.0.0.1:8765>.

The app binds to localhost. Local models and session data stay on this Mac. Model files already stored by LM Studio are read in place; the app does not launch or call LM Studio. Coding file tools are restricted to the selected project folder. Terminal commands run with your macOS account permissions and are not isolated by an OS sandbox.

The Models page discovers MLX folders and GGUF files in the LM Studio model directory. This Mac currently has Qwen3 4B MLX, Qwen2.5 Coder 14B MLX, and Qwen2.5 0.5B GGUF registered for selection. Model weights stay in their existing folders and load into the app-managed runtime when you send a message.

**Preferred engines on native macOS**
- Chat / coding: Apple **MLX first** (models listed first in the UI), then llama.cpp (GGUF + Metal).
- Images: **PyTorch MPS** when installed (reliability / modern models), otherwise stable-diffusion.cpp.

### Optional: enable PyTorch MPS image engine

On native macOS, after the first `run.command` setup:

```bash
.venv/bin/pip install torch torchvision diffusers transformers accelerate safetensors
```

Then open Image Studio and choose **PyTorch MPS** (or Auto). Docker remains on the sd.cpp / CPU path only.

### Recommended models (download manually)

The same list appears on the **Models** page inside the app. Prefer MLX builds when available.

| Role | Model | Link |
|------|--------|------|
| Primary chat + uncensored | Qwen3.8-27B Uncensored (MLX) | https://huggingface.co/sktthehacker/Qwen3.8-27B-Uncensored-MLX |
| Fast MoE | Qwen3.6-35B-A3B Abliterated | Search “Qwen3.6-35B-A3B abliterated” / MLX on Hugging Face |
| Coding | Qwen3-Coder / Qwen3.8 Coder family | https://huggingface.co/models?search=Qwen3-Coder |
| Smaller / faster | Qwen3.5-9B uncensored or Gemma-4 Heretic | Search on Hugging Face |

Existing on many machines: Qwen2.5-Coder-14B-Instruct-MLX-4bit.

## Docker (optional, CPU only)

Docker is no longer the recommended path. Use it only if you need a Linux/CPU environment.

Double-click `start-docker.command` if you still want it. The container cannot use MLX or Metal. GGUF models and image checkpoints work on CPU; large models need more Docker memory than the default ~8 GB.

Stop Docker before using native mode: `docker compose --env-file .env.docker down`.

## Working now

- Chat with local **MLX** models (preferred, listed first) and GGUF models through the app-managed engines.
- Recommended model links on the Models page for manual download.
- **Image Studio engines:** Auto / PyTorch MPS / stable-diffusion.cpp (select in UI). PyTorch path uses attention/VAE slicing and sequential CPU offload to lower peak memory (slower is OK). Optional: `pip install torch torchvision diffusers transformers accelerate safetensors` in `.venv`.
- **Code workspace (MVP):** set a project folder, browse tree, select context files, chat with coding agent tools (`list_dir`, `read_file`, `write_file`, `run_command`, `add_lesson`).
- **Sessions:** rename, archive and restore, export JSON, delete from the recent list menu (⋯). Session data cascades when deleted.
- **IRIS hybrid workspace (early UI):** live model/image/code/tool status, an orbital overview, and navigable Chat/Code/Images/Tools/Email/Voice nodes. Email and Voice are visual placeholders; their integrations remain to be built.
- **Local tool create/review (MVP):** Tools page → describe tool → generate → review/edit source → test a draft → approve. Draft tests do not register or save the tool. Approved tools are stored under `app/data/tools/`.
- PDF create/read/edit tools; Library; custom HTTP tools; MCP stdio config.

## Still to build

- Coding workspace polish: diff preview UI, stronger sandbox, richer playbook.
- Local tools: expose approved tools as per-chat LLM tools; better schema editor.
- Blank-frame detection for sd.cpp; stronger image runtime diagnostics.
- Model Center hardware controls; Model Workshop (LoRA training).
- MCP lifecycle controls; Automations (n8n); Jarvis voice; coding diff review and an OS-level terminal sandbox.
- Native macOS packaging (beyond `run.command`).

## App layout

```text
app/
  server.py          Local web server and API routes
  storage.py         SQLite sessions, preferences, and image history
  image_runtime.py   Local model discovery and direct image generation
  pdf_tools.py       Local PDF create/read/revise operations
  static/
    index.html       App shell and views
    styles.css       Shared visual system
    images.css       Image Studio layout
    app.js           Browser interactions
```

Downloaded models, generated images, runtime logs, and the SQLite database are kept under `app/data/` or `.venv/`; those local files are ignored by Git.
