# AI Command Center

A local-first AI workspace for chat, image generation, coding, tools, and automations.

**Primary path: native macOS on Apple Silicon.** Docker is optional and limited to CPU-only inference.

## Run on this Mac (recommended)

Requires macOS on Apple Silicon, Python 3.11+, Git, and CMake. Double-click `run.command` to set up the app-managed runtimes the first time and open the app at <http://127.0.0.1:8765>.

The app binds to localhost. Local models and session data stay on this Mac. Model files already stored by LM Studio are read in place; the app does not launch or call LM Studio.

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

- Chat with local **MLX** models (preferred) and GGUF models through the app-managed engines.
- Discover and run local Apple MLX models. Qwen2.5-Coder-14B-Instruct-MLX-4bit is preferred when present.
- Recommended model links are listed in the Models page for manual download.
- Find public single-file GGUF models on Hugging Face, download them into the app, or add a GGUF from this Mac.
- Generate images through the app-managed stable-diffusion.cpp engine without calling ComfyUI. The Image Studio reads local SD/SDXL checkpoints and LoRAs from the app's image folders and shared model folders.
- Set image seed mode (random, fixed, incrementing), seed, width, height, steps, guidance scale, sampler, images per run, and LoRA strength.
- Save and resume chats, image settings, generated images, and image metadata with each local session.
- Keep generated images, created PDFs, and files added from this Mac in the local Library.
- Create PDFs from text, extract text from saved PDFs, and revise them by adding a note page or updating title metadata. The source PDF stays intact.
- Enable PDF creation, reading, or editing for individual chats so the local assistant can invoke those tools.
- Create and run basic custom HTTP tools and configure local MCP stdio servers.

## Still to build

- Coding workspace: project folder access, project-aware skills, reviewable file edits, a scoped terminal, and a small-app workflow.
- Model Center: hardware-aware controls, provider/API connections, runtime status for each engine, and clearer model downloads across formats.
- Model Workshop: dataset preparation, LoRA/QLoRA training jobs, evaluation, and adding trained adapters to chat.
- Tools and integrations: a complete custom tool editor, MCP lifecycle controls, and third-party app connections. The first built-in PDF tools and per-chat tool selection are now available.
- Automations: integrate n8n inside the app and manage workflows and runs.
- Jarvis mode: voice input, speech output, and the alternate conversational interface.
- Session management: rename, search, archive, export, and delete sessions.
- Image reliability: blank-frame detection, safer native defaults, optional PyTorch-MPS engine.
- Native packaging: proper macOS app / improved launcher (move fully away from Docker as primary).

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
