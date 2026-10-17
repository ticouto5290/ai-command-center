"""Optional PyTorch + MPS image generation engine for native macOS.

This is the reliability / feature path. The lightweight path remains
stable-diffusion.cpp in image_runtime.py. Both share the same checkpoint
and LoRA discovery roots where possible.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from pathlib import Path

import image_runtime

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "images"
LOG_PATH = ROOT / "data" / "image-pytorch.log"


def engine_status() -> dict:
    """Report whether PyTorch + MPS is usable on this machine."""
    status = {
        "available": False,
        "torch_installed": False,
        "mps_available": False,
        "device": "cpu",
        "message": "PyTorch is not installed yet.",
        "docker": os.environ.get("AICC_DOCKER") == "1",
    }
    if status["docker"]:
        status["message"] = "PyTorch MPS is native macOS only. Docker stays on the CPU engine."
        return status
    try:
        import torch  # noqa: F401
        status["torch_installed"] = True
    except ImportError:
        status["message"] = "Install PyTorch for Apple Silicon to enable this engine (see Models / README)."
        return status
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            status["mps_available"] = True
            status["available"] = True
            status["device"] = "mps"
            status["message"] = "PyTorch MPS ready."
        else:
            status["available"] = True
            status["device"] = "cpu"
            status["message"] = "PyTorch is installed but MPS is not available; will use CPU (slow)."
    except Exception as exc:  # pragma: no cover
        status["message"] = f"PyTorch check failed: {exc}"
    return status


def _load_pipeline(checkpoint: Path, device: str):
    """Load a diffusers pipeline from a single-file checkpoint when possible."""
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image

    dtype = torch.float16 if device == "mps" else torch.float32
    # Prefer AutoPipeline so SD1.5 / SDXL / newer single-file models are handled.
    try:
        pipe = AutoPipelineForText2Image.from_single_file(
            str(checkpoint),
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        # Fallback: try SDXL then SD1.5 explicit loaders.
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(checkpoint), torch_dtype=dtype, local_files_only=True
            )
        except Exception:
            pipe = StableDiffusionPipeline.from_single_file(
                str(checkpoint), torch_dtype=dtype, local_files_only=True
            )
    pipe = pipe.to(device)
    if device == "mps":
        # Recommended for memory stability on Apple Silicon.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe


def generate_image(payload: dict) -> dict:
    status = engine_status()
    if not status["torch_installed"]:
        raise RuntimeError(
            "PyTorch is not installed. On native macOS run: "
            "pip install torch torchvision diffusers transformers accelerate safetensors "
            "inside the app venv, or use the stable-diffusion.cpp engine instead."
        )
    if status["docker"]:
        raise RuntimeError("PyTorch MPS is only available on native macOS. Use the sd.cpp engine in Docker.")

    import torch

    checkpoint = image_runtime._allowed_model_path(payload.get("checkpoint"), image_runtime.CHECKPOINT_ROOTS)
    prompt = str(payload.get("prompt", "")).strip()
    negative = str(payload.get("negative", "")).strip()
    if not prompt or len(prompt) > 5000:
        raise ValueError("Enter a prompt up to 5,000 characters.")

    settings = image_runtime._generation_settings(payload)
    selected_loras = payload.get("loras", [])
    if not isinstance(selected_loras, list) or len(selected_loras) > 12:
        raise ValueError("Choose up to 12 LoRAs.")
    # LoRA loading via diffusers is left for a follow-up; report clearly for now.
    if selected_loras:
        raise RuntimeError(
            "LoRAs in the PyTorch engine are not wired yet. "
            "Switch to the stable-diffusion.cpp engine for LoRA runs, or generate without LoRAs."
        )

    device = status["device"]
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex[:12]
    output_file = IMAGE_DIR / f"image-{image_id}.png"

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.write(f"checkpoint={checkpoint}\ndevice={device}\nsize={settings['width']}x{settings['height']}\n")
        log.write(f"steps={settings['steps']} cfg={settings['cfg']} seed={settings['seed']}\n")
        log.write(f"prompt={prompt[:200]}\n")

    try:
        pipe = _load_pipeline(checkpoint, device)
    except Exception as exc:
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"load_error={exc}\n")
        raise RuntimeError(
            f"Could not load checkpoint with PyTorch: {exc}. "
            "Try a standard SD 1.5 or SDXL safetensors file, or switch engine to sd.cpp."
        ) from exc

    generator = torch.Generator(device=device if device != "mps" else "cpu")
    # MPS generators can be flaky; seed on CPU generator is widely used.
    generator.manual_seed(int(settings["seed"]))

    outputs = []
    try:
        for index in range(settings["count"]):
            seed = (settings["seed"] + index) % (2**32)
            generator.manual_seed(int(seed))
            result = pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                width=settings["width"],
                height=settings["height"],
                num_inference_steps=settings["steps"],
                guidance_scale=settings["cfg"],
                generator=generator,
            )
            image = result.images[0]
            path = output_file if settings["count"] == 1 else output_file.with_name(f"{output_file.stem}_{index}.png")
            image.save(path, format="PNG")
            # Basic blank detection
            extrema = image.convert("L").getextrema()
            if extrema[0] > 250 or extrema[1] < 5:
                path.unlink(missing_ok=True)
                raise RuntimeError(
                    "PyTorch generation produced a near-blank image. "
                    "Try fewer steps, a different checkpoint, or the sd.cpp engine."
                )
            outputs.append({
                "url": f"/api/images/view/{path.name}",
                "filename": path.name,
                "checkpoint": checkpoint.name,
                "seed": seed,
            })
    finally:
        try:
            del pipe
            if device == "mps":
                import torch
                torch.mps.empty_cache()
        except Exception:
            pass

    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"saved={outputs[0]['filename']} count={len(outputs)}\n")

    return {
        **outputs[0],
        "images": outputs,
        "seed_mode": settings["seed_mode"],
        "width": settings["width"],
        "height": settings["height"],
        "steps": settings["steps"],
        "cfg": settings["cfg"],
        "sampler": settings.get("sampler", "pytorch"),
        "scheduler": settings.get("scheduler", "default"),
        "loras": [],
        "engine": "pytorch-mps" if device == "mps" else "pytorch-cpu",
        "generated_at": time.time(),
    }
