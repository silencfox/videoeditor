from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import uuid, os

from .config import WORK_DIR, SD_MODEL_ID, TTS_ENGINE, TTS_MODEL
from .generator import images as images_mod
from .generator import tts as tts_mod
from .generator import assemble as assemble_mod
from .generator import wav2lip as wav2lip_mod

app = FastAPI(title="Script -> Video generator")

class GenerateRequest(BaseModel):
    script: str
    audio: bool = True
    voice: str | None = None
    fps: int = 8
    frames_per_scene: int = 8
    use_sd: bool = False
    internet_ok: bool = True
    sd_steps: int = 15
    sd_guidance: float = 6.5
    sd_width: int = 448
    sd_height: int = 448
    apply_wav2lip: bool = False

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex[:8]
    job_dir = Path(WORK_DIR) / job_id
    frames_dir = job_dir / "frames"
    job_dir.mkdir(parents=True, exist_ok=True)

    frames = images_mod.generate_images_from_script(
        req.script,
        frames_dir,
        frames_per_scene=req.frames_per_scene,
        use_sd=req.use_sd,
        internet_ok=req.internet_ok,
        sd_steps=req.sd_steps,
        sd_guidance=req.sd_guidance,
        sd_width=req.sd_width,
        sd_height=req.sd_height,
        sd_model_id=SD_MODEL_ID
    )

    # assemble frames -> raw video
    image_glob = str(frames_dir / "*.png")
    raw_video = job_dir / "raw.mp4"
    assemble_mod.images_to_video(image_glob, req.fps, raw_video)

    final_video = raw_video

    if req.audio:
        audio_file = job_dir / "audio.wav"
        try:
            tts_engine = os.environ.get("TTS_ENGINE", TTS_ENGINE)
            tts_mod.generate_tts(req.script, audio_file, engine=tts_engine, voice=req.voice or TTS_MODEL)
            if req.apply_wav2lip:
                try:
                    wav_out = job_dir / "wav2lip_out.mp4"
                    wav2lip_mod.apply_wav2lip(raw_video, audio_file, wav_out)
                    assemble_mod.add_audio_to_video(wav_out, audio_file, job_dir / "final.mp4")
                except Exception as e:
                    print("Wav2Lip not applied:", e)
                    assemble_mod.add_audio_to_video(raw_video, audio_file, job_dir / "final.mp4")
            else:
                assemble_mod.add_audio_to_video(raw_video, audio_file, job_dir / "final.mp4")
            final_video = job_dir / "final.mp4"
        except Exception as e:
            print("TTS failed:", e)
            final_video = raw_video

    return {"job_id": job_id, "download": f"/download/{job_id}"}

from fastapi.responses import FileResponse, JSONResponse
@app.get("/download/{job_id}")
def download(job_id: str):
    job_dir = Path(WORK_DIR) / job_id
    final = job_dir / "final.mp4"
    if not final.exists():
        # fallback to raw if exists
        raw = job_dir / "raw.mp4"
        if raw.exists():
            return FileResponse(str(raw), media_type="video/mp4", filename=f"{job_id}_raw.mp4")
        return JSONResponse({"error": "video not found"}, status_code=404)
    return FileResponse(str(final), media_type="video/mp4", filename=f"{job_id}.mp4")
