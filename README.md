# Video Generator (Script → Animated Video)

## Resumen
API en FastAPI que recibe un guion y genera un video animado:
- Genera frames de escenas (Stable Diffusion vía **diffusers** si está disponible; _fallback_ con imágenes simples).
- Audio TTS local (**Coqui TTS** por defecto; opcional **Piper** si está instalado).
- Ensambla el video con **ffmpeg** y permite descargarlo.

> **Sin GPU**: usa `use_sd=false` al principio para obtener resultados rápidos. Con CPU puedes probar SD pero será lento.

---

## Requisitos
- Docker y docker-compose
- (Opcional) GPU NVIDIA si vas a usar modelos pesados (SD/Wav2Lip)
- ffmpeg ya está dentro de la imagen
- (Opcional) Modelos locales (SD / Wav2Lip / voces Piper)

---

## Cómo ejecutar
```bash

## Cómo ejecutar (rápido)
1. Clona/coloca este repo en tu máquina.
2. (Opcional) Clona Wav2Lip:

git clone https://github.com/Rudrabha/Wav2Lip external/Wav2Lip
#dentro de external/Wav2Lip instala requirements si quieres usar lip-sync con GPU/torch

docker build -t silencfox/py38-slim-bullseye:base-os -f .\Dockerfile.base .
docker build -t silencfox/py38-slim-bullseye:torch -f .\Dockerfile.torch .
docker build -t silencfox/py38-slim-bullseye:Wav2Lip -f .\Dockerfile .


docker compose -f .\docker-compose_base.yml build
docker compose -f .\docker-compose_torch.yml build
docker compose up --build

docker-compose build
docker-compose up -d
```


curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Hola, soy Zelda y esta es mi historia",
    "audio": true,
    "use_sd": false,
    "fps": 8,
    "frames_per_scene": 8,
    "face_path": "/app/input/zelda.jpg",
    "face_is_video": false,
    "overlay_text": true,
    "overlay_source": "script",
    "overlay_position": "bottom",
    "overlay_font_size": 28
  }'


curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Narración encima del clip",
    "audio": true,
    "use_sd": false,
    "fps": 8,
    "face_path": "/app/input/base.mp4",
    "face_is_video": true,
    "overlay_text": true,
    "overlay_source": "custom",
    "overlay_custom": "¡Bienvenidos al canal KDvops!",
    "overlay_position": "top",
    "overlay_font_size": 34
  }'



### Probar API
```bash
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d @examples/example_request.json
# Respuesta: { "job_id": "<id>", "download": "/download/<id>" }
curl -O "http://localhost:8000/download/<id>"
```

### Ejemplo de request
Ver `examples/example_request.json`.

---

## Parámetros clave
- `audio`: `true|false` para habilitar TTS.
- `voice`: nombre del modelo Coqui (ej: `tts_models/es/css10/vits`) o deja `null` para usar el de env `TTS_MODEL`.
- `use_sd`: `true|false` para usar Stable Diffusion via diffusers.
- `fps`, `frames_per_scene`: controla duración/suavidad.
- `sd_steps`, `sd_guidance`, `sd_width`, `sd_height`: controles opcionales de SD.

---

## Selección de modelos

### 1) Coqui TTS (recomendado para CPU)
Por defecto se usa el motor `coqui` con la voz de env `TTS_MODEL` (en `docker-compose.yml` viene `tts_models/es/css10/vits`).  
Puedes pasar una voz específica por request con el campo `voice`.

### 2) Piper (opcional, muy ligero en CPU)
Instala binario `piper` en el host y monta su carpeta al contenedor o colócalo en `external/piper`.  
Exporta `TTS_ENGINE=piper` y en la request envía `voice` con la ruta del modelo ONNX de la voz.

### 3) Stable Diffusion (diffusers, CPU o GPU)
- Define `SD_MODEL_ID` (por defecto: `runwayml/stable-diffusion-v1-5`).
- En CPU ajusta parámetros (`sd_steps` bajos: 10–20, `sd_width`/`sd_height` 384 o 448).

> Si no hay SD disponible o falla la carga, el sistema genera frames simples con Pillow (texto sobre fondo).

### 4) Wav2Lip (opcional)
- No está incluido en la imagen. Clona en el host `external/Wav2Lip` y coloca el checkpoint en `external/Wav2Lip/checkpoints/wav2lip_gan.pth`.
- Con CPU es muy lento; úsalo solo si tienes GPU. Si no existe, el sistema hace _fallback_ a añadir el audio sin lip-sync.

---

## Rutas principales
- `POST /generate` → crea trabajo y devuelve `job_id` + URL de descarga
- `GET /download/{job_id}` → devuelve `final.mp4`

---

## Ejemplos

### 1) Flujo rápido (CPU, sin SD)
```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{
  "script":"Escena 1: Juan saluda.\n\nEscena 2: Camina al árbol.",
  "audio": true,
  "use_sd": false,
  "fps": 6,
  "frames_per_scene": 8
}'
```

### 2) Coqui TTS en español
```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{
  "script":"Hola, soy Juan y esta es mi historia.",
  "audio": true,
  "use_sd": false,
  "voice": "tts_models/es/css10/vits"
}'
```

### 3) SD en CPU (lento)
```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{
  "script":"Amanecer en un parque futurista. Un robot despierta.",
  "audio": false,
  "use_sd": true,
  "sd_steps": 15,
  "sd_guidance": 6.5,
  "sd_width": 448,
  "sd_height": 448
}'
```

---

## Notas de rendimiento
- **CPU**: reduce tamaño (`sd_width/height`) y pasos (`sd_steps`). Usa `use_sd=false` para pruebas rápidas.
- **GPU**: corre `docker run --gpus all` o configura en compose la reserva de dispositivos NVIDIA.

---

## Modificaciones del pipeline (paso 2)
### Opción A — Coqui TTS con modelo específico (ES)
1. En `docker-compose.yml` deja:
   ```yaml
   environment:
     - TTS_ENGINE=coqui
     - TTS_MODEL=tts_models/es/css10/vits
   ```
2. En el request, puedes omitir `voice` o sobreescribirlo.
3. El sistema descargará el modelo la primera vez que se use.
4. Para precalentar, ejecuta el script:
   ```bash
   bash scripts/setup_coqui.sh
   ```

### Opción B — Stable Diffusion (diffusers) en CPU
1. En `docker-compose.yml` define:
   ```yaml
   environment:
     - SD_MODEL=local
     - SD_MODEL_ID=runwayml/stable-diffusion-v1-5
   ```
2. En el request activa `use_sd=true` y ajusta `sd_steps`, `sd_width`, `sd_height`.
3. (Opcional) si usas Hugging Face token, exporta `HUGGINGFACE_HUB_TOKEN` y monta `~/.cache/huggingface` como volumen.

> Wav2Lip se deja fuera del Dockerfile. Si en el futuro tienes GPU, clónalo en `external/Wav2Lip` y coloca su checkpoint.

---

## Estructura
```
video-generator/
  app/
    main.py
    config.py
    generator/
      images.py
      tts.py
      assemble.py
      wav2lip.py
      utils.py
    static/
  external/
  models/
  work/
  scripts/
  examples/
    example_request.json
    example_script.txt
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```
