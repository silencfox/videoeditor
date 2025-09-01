# app/main.py
from __future__ import annotations

import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict

# ===== Config & Generators (existentes) =====
from .config import (
    WORK_DIR,
    SD_MODEL_ID,      # si lo tienes en tu config, se usa como default
    TTS_ENGINE,
    TTS_MODEL,
)

from .generator import images as images_mod       # debes tenerlo (o no usarlo)
from .generator import tts as tts_mod
from .generator import assemble as assemble_mod
from .generator import wav2lip as wav2lip_mod     # opcional si usas lip-sync

# ===== IA Storyboard/Estilos/Animación (nuevos) =====
from .generator import styles as styles_mod
from .generator import storyboard as sb_mod
from .generator import ai_anim as ai_mod


app = FastAPI(title="Script → Video Generator", version="1.0.0")


# =========================
# Utils
# =========================
def _save_upload_to(job_dir: Path, upload: UploadFile, name: Optional[str] = None) -> Path:
    """Guarda un UploadFile en el job_dir y devuelve la ruta."""
    target = job_dir / (name or upload.filename or "upload.bin")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return target


def _ffprobe_duration(media_path: Path) -> float:
    """Obtiene duración en segundos con ffprobe (float)."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path)
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def _images_to_video(glob_expr: str, fps: int, out_video: Path):
    out_video.parent.mkdir(parents=True, exist_ok=True)
    assemble_mod.images_to_video(glob_expr, fps, out_video)


def _mux_audio(video_in: Path, audio_in: Path, out_video: Path):
    out_video.parent.mkdir(parents=True, exist_ok=True)
    assemble_mod.add_audio_to_video(video_in, audio_in, out_video)


# =========================
# Request Model (JSON)
# =========================
class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Permitimos alias "script" para compatibilidad previa
    text: str = Field(..., alias="script")

    audio: bool = True
    voice: Optional[str] = None

    fps: int = 8
    frames_per_scene: int = 8

    # Resolución para generación/animación
    sd_width: int = 448
    sd_height: int = 448

    # Imagen/video base (opcional)
    face_path: Optional[str] = None
    face_is_video: bool = False

    # Wav2Lip (opcional si quisieras usarlo)
    apply_wav2lip: bool = False

    # ====== NUEVO: modos IA ======
    animation_mode: str = "story_to_video"  # "story_to_video" | "none" | otros que ya tengas
    style: str = "anime"                    # "anime" | "comic" | "disney" | "realistic"

    # Modelo base SD (si no, usa env SD_BASE_MODEL o SD_MODEL_ID)
    sd_base_model: Optional[str] = None

    # Parámetros de escenas (keyframe) y animación (img2img)
    sd_key_steps: int = 24
    sd_key_guidance: float = 7.0

    sd_anim_steps: int = 18
    sd_anim_guidance: float = 6.5
    sd_strength_base: float = 0.10
    sd_strength_max: float = 0.22

    sd_seed: Optional[int] = 1234


# =========================
# Core runner
# =========================
def run_generation_core(
    *,
    # Texto & Audio
    text: str,
    audio: bool,
    voice: Optional[str],
    # Salida
    fps: int,
    frames_per_scene: int,
    sd_width: int,
    sd_height: int,
    # Origen visual opcional
    face_path_str: Optional[str],
    face_is_video: bool,
    # Lip-sync opcional
    apply_wav2lip: bool,
    # Modo IA storyboard
    animation_mode: str,
    style: str,
    sd_base_model: Optional[str],
    sd_key_steps: int,
    sd_key_guidance: float,
    sd_anim_steps: int,
    sd_anim_guidance: float,
    sd_strength_base: float,
    sd_strength_max: float,
    sd_seed: Optional[int],
):
    job_id = uuid.uuid4().hex[:8]
    job_dir = Path(WORK_DIR) / job_id
    frames_dir = job_dir / "frames"
    job_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) AUDIO (primero para conocer duración) =====
    audio_file = job_dir / "audio.wav"
    audio_duration = 0.0
    if audio:
        # Generar TTS
        tts_engine = os.environ.get("TTS_ENGINE", TTS_ENGINE)
        tts_voice = voice or TTS_MODEL
        tts_mod.generate_tts(text, audio_file, engine=tts_engine, voice=tts_voice)
        audio_duration = _ffprobe_duration(audio_file)

    # ===== 2) PIPELINE PRINCIPAL por modo =====
    model_id = (
        sd_base_model
        or os.environ.get("SD_BASE_MODEL")
        or (SD_MODEL_ID if "SD_MODEL_ID" in globals() and SD_MODEL_ID else "runwayml/stable-diffusion-v1-5")
    )

    # a) STORY_TO_VIDEO: guion → escenas con estilo → animación modulada por voz
    if animation_mode == "story_to_video":
        parts = sb_mod.split_script(text)
        if not parts:
            parts = [text]

        # asignar tiempos a cada escena en proporción al total de audio
        total_ref = audio_duration if audio_duration > 0 else max(2.0, 2.0 * len(parts))
        durations = sb_mod.allocate_durations(parts, total_ref)

        # curva de energía (0..1) del audio para modular fuerza de animación
        energy = sb_mod.audio_energy_envelope(str(audio_file)) if audio and audio_file.exists() else None

        # generar por escena
        frames_all: List[Path] = []
        for idx, (scene_text, sec) in enumerate(zip(parts, durations)):
            prompt = styles_mod.build_prompt(scene_text, style)
            key_path = frames_dir / f"scene{idx:03d}_key.png"

            # keyframe (txt2img) con estilo
            ai_mod.generate_scene_keyframe(
                prompt=prompt,
                out_path=key_path,
                width=sd_width, height=sd_height,
                steps=sd_key_steps, guidance=sd_key_guidance,
                seed=(sd_seed + idx if sd_seed is not None else None),
                model_id=model_id
            )

            # duración → nº frames
            n_frames = max(1, int(round((sec if sec > 0 else 2.0) * fps)))

            # sub-muestrear energía del audio para esta escena
            if energy is not None and energy.size > 0:
                energy_curve = sb_mod.sample_energy_for_frames(energy, n_frames)
            else:
                energy_curve = [0.5] * n_frames  # movimiento moderado

            scene_dir = frames_dir / f"scene{idx:03d}"
            scene_frames = ai_mod.animate_from_keyframe(
                keyframe_path=key_path,
                out_dir=scene_dir,
                prompt=prompt,
                energy_curve=energy_curve,
                fps=fps,
                width=sd_width, height=sd_height,
                steps=sd_anim_steps, guidance=sd_anim_guidance,
                base_strength=sd_strength_base, max_strength=sd_strength_max,
                seed=(sd_seed + idx if sd_seed is not None else None),
                model_id=model_id
            )
            frames_all.extend(scene_frames)

        # ensamblar video
        used_glob = str(frames_dir / "scene*/*.png")
        raw_video = job_dir / "raw.mp4"
        _images_to_video(used_glob, fps, raw_video)

        final_video = raw_video
        if audio and audio_file.exists():
            _mux_audio(raw_video, audio_file, job_dir / "final.mp4")
            final_video = job_dir / "final.mp4"

        return {"job_id": job_id, "download": f"/download/{job_id}"}

    # b) Fallbacks sencillos con tus módulos previos (opcional)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if face_path_str:
        if face_is_video:
            images_mod.frames_from_video(face_path_str, frames_dir, fps)
        else:
            # si no hay audio, usa ~4s por defecto
            total_frames = max(1, int(round((audio_duration if audio_duration > 0 else 4.0) * fps)))
            images_mod.frames_from_still(face_path_str, frames_dir, total_frames, (sd_width, sd_height))
    else:
        # si no hay imagen base, intenta generar imágenes desde script (si lo tienes implementado)
        # o crea frames vacíos/placeholder (dependiendo de tu proyecto)
        if hasattr(images_mod, "generate_images_from_script"):
            images_mod.generate_images_from_script(
                text,
                frames_dir,
                frames_per_scene=frames_per_scene,
                use_sd=False,
                internet_ok=True,
                sd_steps=18,
                sd_guidance=6.5,
                sd_width=sd_width,
                sd_height=sd_height,
                sd_model_id=model_id
            )
        else:
            raise HTTPException(status_code=500, detail="No hay imagen base y generate_images_from_script no está implementado.")

    raw_video = job_dir / "raw.mp4"
    _images_to_video(str(frames_dir / "*.png"), fps, raw_video)

    final_video = raw_video
    if audio and audio_file.exists():
        if apply_wav2lip:
            # lip-sync sobre el video base (si quieres forzar boca)
            try:
                lip_out = job_dir / "lip.mp4"
                wav2lip_mod.apply_wav2lip(raw_video, audio_file, lip_out)
                _mux_audio(lip_out, audio_file, job_dir / "final.mp4")
            except Exception as e:
                print("Wav2Lip falló, usando mux directo:", e)
                _mux_audio(raw_video, audio_file, job_dir / "final.mp4")
        else:
            _mux_audio(raw_video, audio_file, job_dir / "final.mp4")
        final_video = job_dir / "final.mp4"

    return {"job_id": job_id, "download": f"/download/{job_id}"}


# =========================
# Endpoints
# =========================

@app.post("/generate_json")
async def generate_json(req: GenerateRequest):
    """
    JSON puro: Content-Type: application/json
    Campo principal: "text" (alias: "script").
    """
    try:
        return run_generation_core(
            text=req.text,
            audio=req.audio,
            voice=req.voice,
            fps=req.fps,
            frames_per_scene=req.frames_per_scene,
            sd_width=req.sd_width,
            sd_height=req.sd_height,
            face_path_str=req.face_path,
            face_is_video=req.face_is_video,
            apply_wav2lip=req.apply_wav2lip,
            animation_mode=req.animation_mode,
            style=req.style,
            sd_base_model=req.sd_base_model,
            sd_key_steps=req.sd_key_steps,
            sd_key_guidance=req.sd_key_guidance,
            sd_anim_steps=req.sd_anim_steps,
            sd_anim_guidance=req.sd_anim_guidance,
            sd_strength_base=req.sd_strength_base,
            sd_strength_max=req.sd_strength_max,
            sd_seed=req.sd_seed,
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate")
async def generate(
    # Texto
    text: str = Form(...),
    # Audio
    audio: bool = Form(True),
    voice: Optional[str] = Form(None),
    # Video
    fps: int = Form(8),
    frames_per_scene: int = Form(8),
    sd_width: int = Form(448),
    sd_height: int = Form(448),
    # Imagen/Video base opcional (subida)
    face_upload: Optional[UploadFile] = File(None),
    # Imagen/Video base por ruta (montada en el contenedor)
    face_path: Optional[str] = Form(None),
    face_is_video: bool = Form(False),
    # Lip-sync opcional
    apply_wav2lip: bool = Form(False),
    # IA storyboard
    animation_mode: str = Form("story_to_video"),
    style: str = Form("anime"),
    sd_base_model: Optional[str] = Form(None),
    sd_key_steps: int = Form(24),
    sd_key_guidance: float = Form(7.0),
    sd_anim_steps: int = Form(18),
    sd_anim_guidance: float = Form(6.5),
    sd_strength_base: float = Form(0.10),
    sd_strength_max: float = Form(0.22),
    sd_seed: Optional[int] = Form(1234),
):
    """
    Multipart/form-data: permite subir imagen (face_upload) o indicar una ruta (face_path).
    """
    # Si subieron un archivo, lo guardamos en el WORK_DIR del job dentro del core
    # Nota: para no duplicar lógica, guardamos primero y pasamos la ruta al core.
    # Aquí solo guardamos si hay upload; si no, usaremos face_path tal cual.
    # Guardado real se hace adentro si quisiéramos, pero aquí lo resolveremos simple:
    try:
        # Ejecución: si hay face_upload lo guardamos y pasamos path; si no, pasamos face_path.
        # Para usar el mismo core, haremos un guardado rápido temporal: creamos job y sobreescribimos face_path
        job_id_tmp = uuid.uuid4().hex[:8]
        job_dir_tmp = Path(WORK_DIR) / (job_id_tmp + "_up")
        job_dir_tmp.mkdir(parents=True, exist_ok=True)

        face_path_effective = face_path
        if face_upload is not None:
            saved = _save_upload_to(job_dir_tmp, face_upload)
            face_path_effective = str(saved)

        # Llamar core
        result = run_generation_core(
            text=text,
            audio=audio,
            voice=voice,
            fps=fps,
            frames_per_scene=frames_per_scene,
            sd_width=sd_width,
            sd_height=sd_height,
            face_path_str=face_path_effective,
            face_is_video=face_is_video,
            apply_wav2lip=apply_wav2lip,
            animation_mode=animation_mode,
            style=style,
            sd_base_model=sd_base_model,
            sd_key_steps=sd_key_steps,
            sd_key_guidance=sd_key_guidance,
            sd_anim_steps=sd_anim_steps,
            sd_anim_guidance=sd_anim_guidance,
            sd_strength_base=sd_strength_base,
            sd_strength_max=sd_strength_max,
            sd_seed=sd_seed,
        )

        # Limpieza best-effort de la subida temporal
        try:
            shutil.rmtree(job_dir_tmp, ignore_errors=True)
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{job_id}")
def download(job_id: str):
    job_dir = Path(WORK_DIR) / job_id
    final = job_dir / "final.mp4"
    if final.exists():
        return FileResponse(str(final), media_type="video/mp4", filename=f"{job_id}.mp4")
    raw = job_dir / "raw.mp4"
    if raw.exists():
        return FileResponse(str(raw), media_type="video/mp4", filename=f"{job_id}_raw.mp4")
    return JSONResponse({"error": "video not found"}, status_code=404)
