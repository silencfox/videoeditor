from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

from .utils import run_cmd

def _load_font(size: int):
    """
    Intenta cargar DejaVuSans.ttf. Si no está en la imagen base,
    cae al font por defecto de PIL.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _draw_overlay_text(img: Image.Image, text: str, position: str = "bottom",
                       font_size: int = 28, margin: int = 16) -> Image.Image:
    """
    Dibuja un bloque semitransparente con el texto centrado en 'top' o 'bottom'.
    """
    if not text:
        return img

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    # Ajuste de ancho (~84% del ancho) y wrapping
    max_w = int(img.width * 0.84)
    # getlength da mayor consistencia que getsize en fonts recientes
    try:
        avg_char_w = font.getlength("A")
    except Exception:
        # fallback
        avg_char_w = font.getsize("A")[0]

    wrap_chars = max(10, int(max_w / max(1, avg_char_w)))
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=wrap_chars) or [""])

    # Altura del bloque
    line_h = font.size + 6
    block_h = line_h * len(lines) + margin * 2
    y0 = img.height - block_h if position == "bottom" else 0

    # Caja semitransparente
    overlay_color = (0, 0, 0, 140)
    draw.rectangle([0, y0, img.width, y0 + block_h], fill=overlay_color)

    # Texto (blanco con una sombra negra leve)
    y = y0 + margin
    for line in lines:
        try:
            w = font.getlength(line)
        except Exception:
            w = font.getsize(line)[0]
        x = (img.width - w) / 2
        # sombra
        draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0, 200))
        # texto
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 240))
        y += line_h

    return img.convert("RGB")

def frames_from_still(image_path: str, out_dir: Path, total_frames: int,
                      size=(512, 512),
                      overlay_text: bool = False, overlay_text_str: str = "",
                      overlay_position: str = "bottom", overlay_font_size: int = 28):
    """
    Duplica una imagen estática 'total_frames' veces como frame%04d.png.
    Puede superponer texto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    if size:
        img = img.resize(size)

    if overlay_text:
        img = _draw_overlay_text(img, overlay_text_str,
                                 position=overlay_position,
                                 font_size=overlay_font_size)

    for i in range(total_frames):
        img.save(out_dir / f"frame{i:04d}.png")

def frames_from_video(video_path: str, out_dir: Path, fps: int,
                      overlay_text: bool = False, overlay_text_str: str = "",
                      overlay_position: str = "bottom", overlay_font_size: int = 28,
                      size=(512, 512)):
    """
    Extrae frames de un video a frame%04d.png con 'fps' fijo,
    ajusta escala a 'size' manteniendo aspecto (pad), y puede superponer texto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Extrae frames escalados y con pad centrado
    run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
               f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2",
        str(out_dir / "frame%04d.png")
    ])

    if overlay_text:
        for p in sorted(out_dir.glob("frame*.png")):
            img = Image.open(p).convert("RGB")
            img = _draw_overlay_text(img, overlay_text_str,
                                     position=overlay_position,
                                     font_size=overlay_font_size)
            img.save(p)

# ---------------------------
# Generador “por guion” ya existente en tu proyecto
# (no lo tocamos; se asume que crea scene%03d_frame%04d.png en out_dir)
# ---------------------------
def generate_images_from_script(script: str, out_dir: Path,
                                frames_per_scene: int = 8,
                                use_sd: bool = False, internet_ok: bool = True,
                                sd_steps: int = 15, sd_guidance: float = 6.5,
                                sd_width: int = 448, sd_height: int = 448,
                                sd_model_id: str = ""):
    """
    Implementación existente en tu proyecto para generar 'scene*.png' desde el guion.
    Aquí solo dejamos la firma para mantener compatibilidad.
    """
    # NOTA: esta función debe existir en tu repo original.
    # Si tu implementación está en otro archivo, conserva la tuya.
    raise NotImplementedError("generate_images_from_script debe estar implementada en tu proyecto.")
