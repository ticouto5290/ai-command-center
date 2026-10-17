# AI Command Center

A local-first AI workspace for chat, image generation, coding, tools, and automations.

## Run on this Mac

Requires macOS on Apple silicon, Python 3.11 and 3.14, Git, and CMake. Double-click `run.command` to set up the app-managed runtimes the first time and open the app at <http://127.0.0.1:8765>.

The app binds to localhost. Local models and session data stay on this Mac. Model files already stored by LM Studio are read in place; the app does not launch or call LM Studio.

## Run in Docker

Double-click `start-docker.command`; it opens Docker Desktop if needed. The first build compiles the in-app chat and image engines for Linux and may take several minutes. The app opens at <http://127.0.0.1:8765> and the container restarts whenever Docker starts. Stop it with `docker compose --env-file .env.docker down`.

The Docker setup keeps sessions, generated images, Library files, and downloaded models in this project's `app/data/` folder. It mounts existing LM Studio and ComfyUI model folders read-only, so model files are not copied into the container. The container uses CPU inference: macOS Metal and Apple's MLX cannot run inside Docker Desktop's Linux container. GGUF chat models and image checkpoints are supported; MLX-only models such as the current 14B coder will not appear in Docker mode. Docker Desktop currently has about 8 GB of memory available, so large image checkpoints may need lower resolution or a larger Docker memory allocation.

Use `run.command` for native macOS mode when you want Apple MLX or Metal acceleration. Stop Docker first with `docker compose --env-file .env.docker down`; use one mode at a time. Both launch modes use the same local session, Library, image, and model-download folders.

## Working now

- Chat with local GGUF models through the app-managed llama.cpp engine.
- Discover and run local Apple MLX models through the app-managed MLX-LM engine. The laptop has Qwen2.5-Coder-14B-Instruct-MLX-4bit (7.8 GB) and Qwen3-4B-Instruct-2507-MLX-4bit. The 14B coder model is selectable in chat and is preferred for new sessions.
- Find public single-file GGUF models on Hugging Face, download them into the app, or add a GGUF from this Mac.
- Generate images through the app-managed stable-diffusion.cpp Metal engine without calling ComfyUI. The Image Studio reads local SD/SDXL checkpoints and LoRAs from the app's image folders and the existing shared model folders.
- Set image seed mode (random, fixed, incrementing), seed, width, height, steps, guidance scale, sampler, images per run, and LoRA strength.
- Save and resume chats, image settings, generated images, and image metadata with each local session.
- Keep generated images, created PDFs, and files added from this Mac in the local Library.
- Create PDFs from text, extract text from saved PDFs, and revise them by adding a note page or updating title metadata. The source PDF stays intact.
- Enable PDF creation, reading, or editing for individual chats so the local assistant can invoke those tools.
- Create and run basic custom HTTP tools and configure local MCP stdio servers.

The Qwen 7B model is not currently present in the searched local model folders. Qwen2.5-Coder-14B is available instead in MLX 4-bit format.

## Still to build

- Coding workspace: project folder access, project-aware skills, reviewable file edits, a scoped terminal, and a small-app workflow.
- Model Center: hardware-aware controls, provider/API connections, runtime status for each engine, and clearer model downloads across formats.
- Model Workshop: dataset preparation, LoRA/QLoRA training jobs, evaluation, and adding trained adapters to chat.
- Tools and integrations: a complete custom tool editor, MCP lifecycle controls, and third-party app connections. The first built-in PDF tools and per-chat tool selection are now available.
- Automations: integrate n8n inside the app and manage workflows and runs.
- Jarvis mode: voice input, speech output, and the alternate conversational interface.
- Session management: rename, search, archive, export, and delete sessions.

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
