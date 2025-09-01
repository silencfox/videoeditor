from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import uuid, os, shutil, json, subprocess

from .config import WORK_DIR, SD_MODEL_ID, TTS_ENGINE, TTS_MODEL
from .generator import images as images_mod
from .generator import tts as tts_mod
from .generator import assemble as assemble_mod
# wav2lip es opcional; si no lo usas, puedes comentar esta línea
from .generator import wav2lip as wav2lip_mod  # noqa: F401

app = FastAPI(title="Script -> Video generator (Option B)")

# =========================
# Utilidades
# =========================
def _ensure_dirs(job_dir: Path, frames_dir: Path):
    shutil.rmtree(frames_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ])
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None

def _save_upload_to(job_dir: Path, upload: UploadFile) -> Path:
    target = job_dir / upload.filename
    with open(target, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return target

# =========================
# Modelo JSON
# =========================
class GenerateRequest(BaseModel):
    # Acepta text y/o script (compat)
    text: Optional[str] = None
    script: Optional[str] = None

    audio: bool = True
    voice: Optional[str] = None
    fps: int = 8
    frames_per_scene: int = 8

    # SD / internet (por si usas generate_images_from_script)
    use_sd: bool = False
    internet_ok: bool = True
    sd_steps: int = 15
    sd_guidance: float = 6.5
    sd_width: int = 448
    sd_height: int = 448
    sd_model_id: Optional[str] = None  # si no viene, usamos el de config

    # overlay
    overlay_text: bool = True
    overlay_source: str = "script"      # "script" | "custom" (si implementas custom)
    overlay_position: str = "bottom"
    overlay_font_size: int = 28

    # origen visual
    face_path: Optional[str] = None     # "/app/input/zelda.jpg" o .mp4
    face_is_video: bool = False

    # lip sync opcional (si lo usas)
    apply_wav2lip: bool = False

    @property
    def script_text(self) -> str:
        return (self.text or self.script or "").strip()

# =========================
# Núcleo de generación (Opción B)
# =========================
def run_generation_core(
    *,
    text: str,
    fps: int,
    frames_per_scene: int,
    overlay_text: bool,
    overlay_position: str,
    overlay_font_size: int,
    sd_width: int,
    sd_height: int,
    audio: bool,
    voice: Optional[str],
    use_sd: bool,
    internet_ok: bool,
    sd_steps: int,
    sd_guidance: float,
    sd_model_id: Optional[str],
    face_upload: Optional[UploadFile],
    face_path_str: Optional[str],
    face_is_video: bool,
    apply_wav2lip: bool
):
    job_id = uuid.uuid4().hex[:8]
    job_dir = Path(WORK_DIR) / job_id
    frames_dir = job_dir / "frames"
    _ensure_dirs(job_dir, frames_dir)

    # 1) AUDIO primero (para conocer duración)
    audio_file = job_dir / "audio.wav"
    audio_duration = None
    if audio:
        try:
            tts_engine = os.environ.get("TTS_ENGINE", TTS_ENGINE)
            tts_voice = voice or TTS_MODEL
            tts_mod.generate_tts(text, audio_file, engine=tts_engine, voice=tts_voice)
            audio_duration = _ffprobe_duration(audio_file)
        except Exception as e:
            print("TTS failed (continuo sin audio):", e)
            audio_duration = None

    # 2) FRAMES según origen visual
    used_glob = None

    # Prioridad 1: face subido como archivo (multipart)
    if face_upload is not None:
        face_file = _save_upload_to(job_dir, face_upload)
        if face_is_video:
            images_mod.frames_from_video(
                str(face_file), frames_dir, fps,
                overlay_text=overlay_text,
                overlay_text_str=text,
                overlay_position=overlay_position,
                overlay_font_size=overlay_font_size,
                size=(sd_width, sd_height),
            )
        else:
            total = max(1, int(round(audio_duration * fps))) if audio_duration else max(1, frames_per_scene)
            images_mod.frames_from_still(
                str(face_file), frames_dir, total,
                size=(sd_width, sd_height),
                overlay_text=overlay_text,
                overlay_text_str=text,
                overlay_position=overlay_position,
                overlay_font_size=overlay_font_size,
            )
        used_glob = str(frames_dir / "frame*.png")

    # Prioridad 2: face por ruta (JSON)
    elif face_path_str:
        if face_is_video:
            images_mod.frames_from_video(
                face_path_str, frames_dir, fps,
                overlay_text=overlay_text,
                overlay_text_str=text,
                overlay_position=overlay_position,
                overlay_font_size=overlay_font_size,
                size=(sd_width, sd_height),
            )
        else:
            total = max(1, int(round(audio_duration * fps))) if audio_duration else max(1, frames_per_scene)
            images_mod.frames_from_still(
                face_path_str, frames_dir, total,
                size=(sd_width, sd_height),
                overlay_text=overlay_text,
                overlay_text_str=text,
                overlay_position=overlay_position,
                overlay_font_size=overlay_font_size,
            )
        used_glob = str(frames_dir / "frame*.png")

    # Prioridad 3: generación por guion (si la tienes implementada)
    else:
        # OJO: esta función debe existir en tu proyecto; si no, usa un generador simple
        images_mod.generate_images_from_script(
            text, frames_dir,
            frames_per_scene=frames_per_scene,
            use_sd=use_sd, internet_ok=internet_ok,
            sd_steps=sd_steps, sd_guidance=sd_guidance,
            sd_width=sd_width, sd_height=sd_height,
            sd_model_id=(sd_model_id or SD_MODEL_ID)
        )
        used_glob = str(frames_dir / "scene*.png")

    # 3) Ensamblar RAW
    raw_video = job_dir / "raw.mp4"
    assemble_mod.images_to_video(used_glob, fps, raw_video)
    final_video = raw_video

    # 4) Mezclar audio
    if audio and audio_file.exists():
        try:
            # Si tus frames ya se ajustan a la duración del audio (imagen fija + Option B),
            # no hace falta loop. Si quieres blindaje, puedes usar una función con -stream_loop.
            assemble_mod.add_audio_to_video(raw_video, audio_file, job_dir / "final.mp4")
            final_video = job_dir / "final.mp4"
        except Exception as e:
            print("Mux failed:", e)
            final_video = raw_video

    return {"job_id": job_id, "download": f"/download/{job_id}"}

# =========================
# Endpoints
# =========================

# form-data (con archivo)
@app.post("/generate")
async def generate(
    text: str = Form(...),
    face: Optional[UploadFile] = File(None),
    face_is_video: bool = Form(False),
    audio: bool = Form(True),
    voice: Optional[str] = Form(None),
    fps: int = Form(8),
    frames_per_scene: int = Form(8),
    use_sd: bool = Form(False),
    internet_ok: bool = Form(True),
    sd_steps: int = Form(15),
    sd_guidance: float = Form(6.5),
    sd_width: int = Form(448),
    sd_height: int = Form(448),
    apply_wav2lip: bool = Form(False),
    overlay_text: bool = Form(True),
    overlay_source: str = Form("script"),
    overlay_position: str = Form("bottom"),
    overlay_font_size: int = Form(28)
):
    try:
        return run_generation_core(
            text=text,
            fps=fps, frames_per_scene=frames_per_scene,
            overlay_text=overlay_text, overlay_position=overlay_position, overlay_font_size=overlay_font_size,
            sd_width=sd_width, sd_height=sd_height,
            audio=audio, voice=voice,
            use_sd=use_sd, internet_ok=internet_ok, sd_steps=sd_steps, sd_guidance=sd_guidance, sd_model_id=None,
            face_upload=face, face_path_str=None, face_is_video=face_is_video,
            apply_wav2lip=apply_wav2lip
        )
    except NotImplementedError as nie:
        return JSONResponse({"error": str(nie)}, status_code=400)

# JSON (sin archivo; usa face_path si quieres imagen/video base)
@app.post("/generate_json")
async def generate_json(req: GenerateRequest):
    if not req.script_text:
        return JSONResponse({"error": "Falta 'text' o 'script'."}, status_code=422)
    try:
        return run_generation_core(
            text=req.script_text,
            fps=req.fps, frames_per_scene=req.frames_per_scene,
            overlay_text=req.overlay_text, overlay_position=req.overlay_position, overlay_font_size=req.overlay_font_size,
            sd_width=req.sd_width, sd_height=req.sd_height,
            audio=req.audio, voice=req.voice,
            use_sd=req.use_sd, internet_ok=req.internet_ok,
            sd_steps=req.sd_steps, sd_guidance=req.sd_guidance,
            sd_model_id=(req.sd_model_id or SD_MODEL_ID),
            face_upload=None, face_path_str=req.face_path, face_is_video=req.face_is_video,
            apply_wav2lip=req.apply_wav2lip
        )
    except NotImplementedError as nie:
        # si no tienes implementado generate_images_from_script y no enviaste face/face_path
        return JSONResponse({"error": str(nie), "hint": "Envía 'face_path' (imagen/video) o implementa generate_images_from_script()."}, status_code=400)

# descarga
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
