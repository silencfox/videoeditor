import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = BASE_DIR / "work"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = (BASE_DIR / "app" / "static")

WORK_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# External tools/repos
WAV2LIP_PATH = os.environ.get("WAV2LIP_PATH", str(BASE_DIR / "external" / "Wav2Lip"))

# Models / engines
SD_MODEL = os.environ.get("SD_MODEL", "local")  # local or none
SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
TTS_ENGINE = os.environ.get("TTS_ENGINE", "coqui")  # coqui or piper
TTS_MODEL = os.environ.get("TTS_MODEL", "tts_models/es/css10/vits")
