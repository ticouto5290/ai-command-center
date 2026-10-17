"""App-managed Stable Diffusion image generation for macOS."""

from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
IMAGE_RUNTIME = PROJECT / ".venv" / "stable-diffusion.cpp" / "build-command-center-metal" / "bin" / "sd-cli"
IMAGE_DIR = ROOT / "data" / "images"
HOME = Path.home()


def _configured_paths(variable: str) -> list[Path]:
    return [Path(item).expanduser() for item in os.environ.get(variable, "").split(os.pathsep) if item]


LOCAL_CHECKPOINT_ROOTS = [ROOT / "data" / "image-models" / "checkpoints"]
LOCAL_LORA_ROOTS = [ROOT / "data" / "image-models" / "loras"]

CHECKPOINT_ROOTS = [
    *LOCAL_CHECKPOINT_ROOTS,
    HOME / "ComfyUI-Shared" / "models" / "checkpoints",
    HOME / "ComfyUI-Installs" / "test" / "ComfyUI" / "models" / "checkpoints",
    HOME / "Downloads" / "workflow_comfyui" / "checkpoints",
    HOME / "Downloads" / "workflow_comfyui" / "models" / "checkpoints",
    HOME / "Downloads" / "local-ai-hub" / "~/ComfyUI" / "models" / "checkpoints",
    HOME / "Downloads" / "realisticVisionV60B1_v51HyperVAE.safetensors",
    *_configured_paths("AICC_CHECKPOINT_DIRS"),
]
LORA_ROOTS = [
    *LOCAL_LORA_ROOTS,
    HOME / "ComfyUI-Shared" / "models" / "loras",
    HOME / "ComfyUI-Installs" / "test" / "ComfyUI" / "models" / "loras",
    HOME / "Downloads" / "workflow_comfyui" / "lora",
    HOME / "Downloads" / "workflow_comfyui" / "loras",
    HOME / "Sites" / "img_pony" / "loras",
    *_configured_paths("AICC_LORA_DIRS"),
]
SUPPORTED_EXTENSIONS = {".safetensors", ".ckpt", ".gguf"}
MAX_MODEL_BYTES = 40 * 1024**3
MAX_IMAGE_PIXELS = 1536 * 1536
SAMPLERS = {"euler", "euler_a", "heun", "dpm++2m", "dpm++2s_a", "ddim_trailing", "lcm"}
SCHEDULERS = {"model", "discrete", "karras", "exponential", "ays", "sgm_uniform"}


def _catalog(roots: list[Path]) -> list[dict]:
    found: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir() and not root.is_file():
            continue
        for path in ([root] if root.is_file() else root.rglob("*")):
            try:
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if root in CHECKPOINT_ROOTS and re.search(r"(?:video|i2v|t2v|wan|ltx|audio)", path.stem, re.IGNORECASE):
                    continue
                size = path.stat().st_size
                if size <= 0 or size > MAX_MODEL_BYTES:
                    continue
                resolved = str(path.resolve())
                try:
                    relative = path.resolve().relative_to(HOME)
                    location = "/".join(relative.parts[:2])
                except ValueError:
                    location = path.parent.name
                found.setdefault(resolved, {"name": path.name, "path": resolved, "size": size, "location": location})
            except OSError:
                continue
    return sorted(found.values(), key=lambda item: item["name"].casefold())


def image_catalog() -> dict:
    return {
        "engine_ready": IMAGE_RUNTIME.is_file(),
        "docker_cpu": os.environ.get("AICC_DOCKER") == "1",
        "checkpoints": _catalog(CHECKPOINT_ROOTS),
        "loras": _catalog(LORA_ROOTS),
    }


def _allowed_model_path(value: object, roots: list[Path]) -> Path:
    if not isinstance(value, str):
        raise ValueError("Choose an image model from the local model list.")
    candidate = Path(value).expanduser().resolve()
    if not any((root.exists() and (candidate == root.resolve() or root.resolve() in candidate.parents)) for root in roots):
        raise ValueError("The selected model is outside the app's approved local image folders.")
    if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS or candidate.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("The selected image model file is not available or is too large.")
    return candidate


def _bounded_number(payload: dict, key: str, default: float, minimum: float, maximum: float, integer: bool = False):
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ValueError(f"{key.replace('_', ' ').capitalize()} must be between {minimum:g} and {maximum:g}.")
    if integer and int(value) != value:
        raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a whole number.")
    return int(value) if integer else float(value)


def _generation_settings(payload: dict) -> dict:
    width = _bounded_number(payload, "width", 1024, 256, 1536, integer=True)
    height = _bounded_number(payload, "height", 1024, 256, 1536, integer=True)
    if width % 8 or height % 8 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Width and height must be multiples of 8, with no more than 1536 × 1536 pixels.")
    seed_mode = payload.get("seed_mode", "random")
    if seed_mode not in ("random", "fixed", "increment"):
        raise ValueError("Choose random, fixed, or incrementing seed mode.")
    seed = _bounded_number(payload, "seed", 1, 0, 2**32 - 1, integer=True)
    if seed_mode == "random":
        seed = random.SystemRandom().randrange(0, 2**32)
    elif seed_mode == "increment":
        seed = (seed + _bounded_number(payload, "seed_offset", 0, 0, 2**32 - 1, integer=True)) % 2**32
    steps = _bounded_number(payload, "steps", 24, 1, 100, integer=True)
    cfg = _bounded_number(payload, "cfg", 7, 1, 30)
    count = _bounded_number(payload, "count", 1, 1, 4, integer=True)
    sampler = payload.get("sampler", "euler")
    if sampler not in SAMPLERS:
        raise ValueError("Choose a supported sampler.")
    scheduler = payload.get("scheduler", "model")
    if scheduler not in SCHEDULERS:
        raise ValueError("Choose a supported scheduler.")
    return {"width": width, "height": height, "seed_mode": seed_mode, "seed": seed, "steps": steps, "cfg": cfg, "count": count, "sampler": sampler, "scheduler": scheduler}


def generate_image(payload: dict) -> dict:
    if not IMAGE_RUNTIME.is_file():
        raise RuntimeError("The app's image engine has not been built yet. Restart the app to finish its one-time setup.")
    checkpoint = _allowed_model_path(payload.get("checkpoint"), CHECKPOINT_ROOTS)
    prompt = str(payload.get("prompt", "")).strip()
    negative = str(payload.get("negative", "")).strip()
    if not prompt or len(prompt) > 5000 or len(negative) > 3000:
        raise ValueError("Enter a prompt up to 5,000 characters and an optional negative prompt up to 3,000 characters.")
    settings = _generation_settings(payload)
    selected_loras = payload.get("loras", [])
    if not isinstance(selected_loras, list) or len(selected_loras) > 12:
        raise ValueError("Choose up to 12 LoRAs.")
    loras = [_allowed_model_path(path, LORA_ROOTS) for path in selected_loras]
    lora_strength = _bounded_number(payload, "lora_strength", 0.8, 0, 2)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex[:12]
    output_file = IMAGE_DIR / f"image-{image_id}.png"
    prompt_parts = [prompt]
    with tempfile.TemporaryDirectory(prefix="ai-command-center-loras-") as lora_folder:
        used_names: set[str] = set()
        for lora in loras:
            alias = re.sub(r"[^A-Za-z0-9._-]", "_", lora.stem)[:60] or "lora"
            if alias.casefold() in used_names:
                alias = f"{alias}-{abs(hash(str(lora))) & 0xffff:04x}"
            used_names.add(alias.casefold())
            os.symlink(lora, Path(lora_folder) / f"{alias}{lora.suffix}")
            prompt_parts.append(f"<lora:{alias}:{lora_strength:g}>")
        args = [
            str(IMAGE_RUNTIME), "-m", str(checkpoint), "-p", " ".join(prompt_parts),
            "-n", negative, "-o", str(output_file), "-W", str(settings["width"]),
            "-H", str(settings["height"]), "--steps", str(settings["steps"]),
            "--cfg-scale", str(settings["cfg"]), "--sampling-method", settings["sampler"],
            "--seed", str(settings["seed"]), "--batch-count", str(settings["count"]),
            "--lora-model-dir", lora_folder,
        ]
        if os.uname().machine == "arm64":
            # Metal diffusion inference is producing blank white images on
            # this Apple Silicon setup. Keep the text encoder and VAE on their
            # normal backend while running the diffusion model on the CPU.
            args += ["--backend", "diffusion=cpu"]
            # These are the engine's documented numerical-stability settings
            # for blank outputs on affected backend/model combinations.
            args += ["--linear-scale", "0.0078125", "--attn-scale", "0.0078125"]
        if loras:
            # The immediate merge path aborts on this Metal setup when the
            # LoRA graph tries to add into a pre-allocated CLIP tensor.
            # Runtime application is slower but is the engine's compatible path.
            args += ["--lora-apply-mode", "at_runtime"]
        if settings["scheduler"] != "model":
            args += ["--scheduler", settings["scheduler"]]
        log_path = ROOT / "data" / "image-runtime.log"
        try:
            environment = os.environ.copy()
            # Current stable-diffusion.cpp Metal tensor kernels can silently
            # produce white frames on Apple M5; the upstream issue documents
            # this switch as the working fallback for those devices.
            if os.uname().machine == "arm64":
                environment["GGML_METAL_TENSOR_DISABLE"] = "1"
            with log_path.open("ab") as log:
                result = subprocess.run(args, cwd=IMAGE_RUNTIME.parent, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800, check=False, env=environment)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Image generation took longer than 30 minutes and was stopped.") from None
        outputs = []
        for index in range(settings["count"]):
            seed = (settings["seed"] + index) % 2**32
            path = output_file if settings["count"] == 1 else output_file.with_name(f"{output_file.stem}_{index}.png")
            if result.returncode != 0 or not path.is_file():
                for partial in IMAGE_DIR.glob(f"{output_file.stem}*.png"):
                    partial.unlink(missing_ok=True)
                raise RuntimeError("Image generation failed. Open Image Studio's local engine log for details.")
            outputs.append({
                "url": f"/api/images/view/{path.name}", "filename": path.name,
                "checkpoint": checkpoint.name, "seed": seed,
            })
    return {
        **outputs[0], "images": outputs, "seed_mode": settings["seed_mode"],
        "width": settings["width"], "height": settings["height"],
        "steps": settings["steps"], "cfg": settings["cfg"], "sampler": settings["sampler"], "scheduler": settings["scheduler"],
        "loras": [path.name for path in loras], "generated_at": time.time(),
    }
