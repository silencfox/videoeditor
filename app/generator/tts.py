import os, subprocess
from pathlib import Path

def _tts_coqui(text: str, out_wav: Path, model_name: str | None):
    try:
        from TTS.api import TTS
        name = model_name or os.environ.get("TTS_MODEL", "tts_models/es/css10/vits")
        tts = TTS(model_name=name, progress_bar=False, gpu=False)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        tts.tts_to_file(text=text, file_path=str(out_wav))
        return out_wav
    except Exception as e:
        raise RuntimeError(f"Coqui TTS failed: {e}")

def _tts_piper(text: str, out_wav: Path, voice_path: str | None):
    voice = voice_path or os.environ.get("TTS_MODEL", "")
    if not voice:
        raise RuntimeError("Piper requires a voice model path (onnx). Set voice in request or TTS_MODEL env.")
    try:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["piper", "-m", voice, "-f", "22050", "-o", str(out_wav)]
        proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="ignore"))
        return out_wav
    except FileNotFoundError:
        raise RuntimeError("piper binary not found in PATH. Install piper or use Coqui.")

def generate_tts(script: str, out_wav: Path, engine: str = "coqui", voice: str | None = None):
    if engine == "piper":
        return _tts_piper(script, out_wav, voice)
    return _tts_coqui(script, out_wav, voice)
