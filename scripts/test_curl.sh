#!/usr/bin/env bash
set -euo pipefail
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{
  "script":"Escena 1: Saludo en el parque.\n\nEscena 2: Caminar hacia el árbol.",
  "audio": true,
  "use_sd": false,
  "fps": 6,
  "frames_per_scene": 8
}'
