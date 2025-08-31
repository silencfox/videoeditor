#!/usr/bin/env bash
set -euo pipefail
echo "[setup] Precalentando Coqui TTS..."
python - <<'PY'
from TTS.api import TTS
print("Descargando modelo... (primera ejecución)")
tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False, gpu=False)
tts.tts_to_file(text="Prueba de voz en español.", file_path="work/tts_warmup.wav")
print("Listo: work/tts_warmup.wav")
PY
