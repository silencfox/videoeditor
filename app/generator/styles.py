# app/generator/styles.py
from typing import Dict

STYLE_PRESETS: Dict[str, str] = {
    "anime": (
        "anime style, vibrant colors, clean lineart, detailed shading, "
        "dynamic composition, cinematic lighting, high quality"
    ),
    "comic": (
        "comic book style, bold ink lines, halftone shading, dramatic contrast, "
        "action pose, panel-like composition, high quality"
    ),
    "disney": (
        "Disney-like animation style, expressive characters, soft lighting, "
        "whimsical, cinematic, high quality, smooth gradients"
    ),
    "realistic": (
        "photorealistic, natural lighting, detailed textures, high quality"
    ),
}

def build_prompt(base: str, style: str) -> str:
    style_suffix = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["anime"])
    return f"{base}, {style_suffix}"
