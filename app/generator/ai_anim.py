# app/generator/ai_anim.py
from pathlib import Path
from typing import List, Optional
import random
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# ===== Opcional: limita hilos en CPU si te satura =====
try:
    # Ajusta según tu CPU; 4-6 suele ir bien en laptops
    torch.set_num_threads(6)
except Exception:
    pass

# ===== Caches globales para no recargar modelos cada petición =====
_TXT2IMG_CACHE = {}
_IMG2IMG_CACHE = {}

def _common_optimize(pipe):
    # Menor memoria en atención
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    # (Si tu versión lo soporta) reduce el chunk size de la atención
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe

def _from_pretrained(model_id: str, cls, device: str):
    kwargs = {"torch_dtype": torch.float32, "safety_checker": None}
    # Si accelerate está instalado, se usa low_cpu_mem_usage
    try:
        import accelerate  # noqa: F401
        kwargs["low_cpu_mem_usage"] = True
    except Exception:
        pass
    pipe = cls.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    return _common_optimize(pipe)

def _load_txt2img(model_id: str, device: str = "cpu"):
    key = (model_id, device, "txt2img")
    if key not in _TXT2IMG_CACHE:
        _TXT2IMG_CACHE[key] = _from_pretrained(model_id, StableDiffusionPipeline, device)
    return _TXT2IMG_CACHE[key]

def _load_img2img(model_id: str, device: str = "cpu"):
    key = (model_id, device, "img2img")
    if key not in _IMG2IMG_CACHE:
        _IMG2IMG_CACHE[key] = _from_pretrained(model_id, StableDiffusionImg2ImgPipeline, device)
    return _IMG2IMG_CACHE[key]

def generate_scene_keyframe(
    prompt: str,
    out_path: Path,
    width: int = 448,
    height: int = 448,
    steps: int = 24,
    guidance: float = 7.0,
    seed: Optional[int] = 1234,
    model_id: str = "runwayml/stable-diffusion-v1-5"
):
    device = "cpu"
    pipe = _load_txt2img(model_id, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, artifacts, text watermark",
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path

def animate_from_keyframe(
    keyframe_path: Path,
    out_dir: Path,
    prompt: str,
    energy_curve: List[float],
    fps: int,
    width: int = 448,
    height: int = 448,
    steps: int = 18,
    guidance: float = 6.5,
    base_strength: float = 0.10,
    max_strength: float = 0.22,
    seed: Optional[int] = 1234,
    model_id: str = "runwayml/stable-diffusion-v1-5"
) -> List[Path]:
    """
    Img2img encadenado con fuerza modulada por energía de voz (0..1).
    """
    device = "cpu"
    pipe = _load_img2img(model_id, device=device)
    rnd = random.Random(seed or 1234)

    base = Image.open(keyframe_path).convert("RGB").resize((width, height))
    current = base
    frames: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, e in enumerate(energy_curve):
        strength = base_strength + (max_strength - base_strength) * float(e)
        strength = max(0.05, min(0.9, strength))
        if seed is not None:
            torch.manual_seed(rnd.randint(0, 10_000_000))
        out = pipe(
            prompt=f"{prompt}, subtle motion, cinematic",
            negative_prompt="blurry, low quality, artifacts, text watermark",
            image=current,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]
        current = out
        fpath = out_dir / f"frame{idx:04d}.png"
        current.save(fpath)
        frames.append(fpath)
    return frames
