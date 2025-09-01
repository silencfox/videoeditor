# app/main.py
from __future__ import annotations

import os
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

# ===== Config & Generators (existentes) =====
from .config import (
    WORK_DIR,
    SD_MODEL_ID,      # si lo tienes en tu config, se usa como default
    TTS_ENGINE,
    TTS_MODEL,
)

from .generator import images as images_mod       # si no lo usas, borra esta importación
from .generator import tts as tts_mod
from .generator import assemble as assemble_mod
from .generator import wav2lip as wav2lip_mod     # opcional si usas lip-sync

# ===== IA Storyboard/Estilos/Animación (nuevos) =====
from .generator import styles as styles_mod
from .generator import storyboard as sb_mod
from .generator import ai_anim as ai_mod


app = FastAPI(title="Script → Video Generator", version="1.0.0")


# =========================
# Progreso (helpers)
# =========================
def _progress_path(job_dir: Path) -> Path:
    return job_dir / "progress.json"

def progress_init(job_dir: Path, total: float = 100.0):
    data = {"percent": 0.0, "stage": "starting", "detail": "", "ts": time.time(), "total": total}
    _progress_path(job_dir).write_text(json.dumps(data))

def progress_update(job_dir: Path, percent: float, stage: str, detail: str = ""):
    percent = max(0.0, min(100.0, float(percent)))
    data = {"percent": percent, "stage": stage, "detail": detail, "ts": time.time()}
    _progress_path(job_dir).write_text(json.dumps(data))

def progress_read(job_dir: Path) -> dict:
    p = _progress_path(job_dir)
    if not p.exists():
        return {"percent": 0.0, "stage": "pending", "detail": "", "ts": time.time()}
    return json.loads(p.read_text())


# =========================
# Utils
# =========================
def _save_upload_to(job_dir: Path, upload: UploadFile, name: Optional[str] = None) -> Path:
    target = job_dir / (name or upload.filename or "upload.bin")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return target

def _ffprobe_duration(media_path: Path) -> float:
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
    out_video.parent.mkdir(parents=True, exist_ok=True
    )
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

    # Wav2Lip (opcional)
    apply_wav2lip: bool = False

    # ====== NUEVO: modos IA ======
    animation_mode: str = "story_to_video"  # "story_to_video" | "none"
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
    text: str,
    audio: bool,
    voice: Optional[str],
    fps: int,
    frames_per_scene: int,
    sd_width: int,
    sd_height: int,
    face_path_str: Optional[str],
    face_is_video: bool,
    apply_wav2lip: bool,
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

    progress_init(job_dir)
    progress_update(job_dir, 2, "prepare", "Creando estructura de trabajo")

    # ===== 1) AUDIO (primero para conocer duración) =====
    audio_file = job_dir / "audio.wav"
    audio_duration = 0.0
    if audio:
        progress_update(job_dir, 5, "tts", "Generando voz")
        tts_engine = os.environ.get("TTS_ENGINE", TTS_ENGINE)
        tts_voice = voice or TTS_MODEL
        tts_mod.generate_tts(text, audio_file, engine=tts_engine, voice=tts_voice)
        audio_duration = _ffprobe_duration(audio_file)
        progress_update(job_dir, 10, "tts_done", f"Duración audio: {audio_duration:.2f}s")

    # ===== 2) PIPELINE PRINCIPAL por modo =====
    model_id = (
        sd_base_model
        or os.environ.get("SD_BASE_MODEL")
        or (SD_MODEL_ID if SD_MODEL_ID else "runwayml/stable-diffusion-v1-5")
    )

    # a) STORY_TO_VIDEO
    if animation_mode == "story_to_video":
        progress_update(job_dir, 15, "storyboard", "Segmentando guion y asignando tiempos")
        parts = sb_mod.split_script(text) or [text]
        total_ref = audio_duration if audio_duration > 0 else max(2.0, 2.0 * len(parts))
        durations = sb_mod.allocate_durations(parts, total_ref)
        energy = sb_mod.audio_energy_envelope(str(audio_file)) if audio and audio_file.exists() else None
        progress_update(job_dir, 20, "storyboard_done", f"Escenas: {len(parts)}")

        # Planificación por escenas/frames
        frames_per_scene_planned: List[int] = []
        total_frames_all = 0
        for sec in durations:
            n = max(1, int(round((sec if sec > 0 else 2.0) * fps)))
            frames_per_scene_planned.append(n)
            total_frames_all += n

        # Presupuesto de progreso
        keyframe_budget = 10.0   # %
        anim_budget = 60.0       # %
        progress = 20.0

        # Callback de animación (se define tras variables previas)
        def _scene_progress_cb(scene_idx: int, frame_idx: int, frame_total: int):
            nonlocal progress
            inc = (anim_budget * (1.0 / max(1, total_frames_all)))
            progress = min(95.0, progress + inc)
            progress_update(job_dir, progress, "animating", f"Escena {scene_idx+1}/{len(parts)} frame {frame_idx}/{frame_total}")

        # Generación por escena
        frames_all: List[Path] = []
        for idx, (scene_text, sec) in enumerate(zip(parts, durations)):
            # Keyframe
            progress += keyframe_budget / max(1, len(parts))
            progress_update(job_dir, progress, "keyframe", f"Escena {idx+1}/{len(parts)}: generando keyframe")

            prompt = styles_mod.build_prompt(scene_text, style)
            key_path = frames_dir / f"scene{idx:03d}_key.png"
            ai_mod.generate_scene_keyframe(
                prompt=prompt,
                out_path=key_path,
                width=sd_width, height=sd_height,
                steps=sd_key_steps, guidance=sd_key_guidance,
                seed=(sd_seed + idx if sd_seed is not None else None),
                model_id=model_id
            )

            # Animación
            n_frames = frames_per_scene_planned[idx]
            progress_update(job_dir, progress, "animate_init", f"Escena {idx+1}: {n_frames} frames")

            if energy is not None and getattr(energy, "size", 0) > 0:
                energy_curve = sb_mod.sample_energy_for_frames(energy, n_frames)
            else:
                energy_curve = [0.5] * n_frames

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
                model_id=model_id,
                on_progress=lambda done, total, _idx=idx: _scene_progress_cb(_idx, done, total)
            )
            frames_all.extend(scene_frames)

        # Ensamblado
        used_glob = str(frames_dir / "scene*/*.png")
        raw_video = job_dir / "raw.mp4"
        progress_update(job_dir, 95, "assemble", "Ensamblando video")
        _images_to_video(used_glob, fps, raw_video)

        final_video = raw_video
        if audio and audio_file.exists():
            progress_update(job_dir, 98, "mux", "Empaquetando audio")
            _mux_audio(raw_video, audio_file, job_dir / "final.mp4")
            progress_update(job_dir, 100, "done", "Completado")
            final_video = job_dir / "final.mp4"
        else:
            progress_update(job_dir, 100, "done", "Completado (sin audio)")

        return {"job_id": job_id, "download": f"/download/{job_id}"}

    # b) Fallback simple
    frames_dir.mkdir(parents=True, exist_ok=True)
    if face_path_str:
        if face_is_video:
            images_mod.frames_from_video(face_path_str, frames_dir, fps)
        else:
            total_frames = max(1, int(round((audio_duration if audio_duration > 0 else 4.0) * fps)))
            images_mod.frames_from_still(face_path_str, frames_dir, total_frames, (sd_width, sd_height))
    else:
        if hasattr(images_mod, "generate_images_from_script"):
            images_mod.generate_images_from_script(
                text,  # primer parámetro es el guion
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

    progress_update(job_dir, 100, "done", "Completado (fallback)")
    return {"job_id": job_id, "download": f"/download/{job_id}"}


# =========================
# Endpoints
# =========================

@app.post("/generate_json")
async def generate_json(req: GenerateRequest):
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
    text: str = Form(...),
    audio: bool = Form(True),
    voice: Optional[str] = Form(None),
    fps: int = Form(8),
    frames_per_scene: int = Form(8),
    sd_width: int = Form(448),
    sd_height: int = Form(448),
    face_upload: Optional[UploadFile] = File(None),
    face_path: Optional[str] = Form(None),
    face_is_video: bool = Form(False),
    apply_wav2lip: bool = Form(False),
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
    try:
        job_id_tmp = uuid.uuid4().hex[:8]
        job_dir_tmp = Path(WORK_DIR) / (job_id_tmp + "_up")
        job_dir_tmp.mkdir(parents=True, exist_ok=True)

        face_path_effective = face_path
        if face_upload is not None:
            saved = _save_upload_to(job_dir_tmp, face_upload)
            face_path_effective = str(saved)

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


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    job_dir = Path(WORK_DIR) / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return progress_read(job_dir)
