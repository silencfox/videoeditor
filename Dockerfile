FROM silencfox/py311-slim:torch

ARG INSTALL_TORCH_CPU=true
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Instala SOLO lo de tu app (torch ya viene en la base)
COPY requirements.txt .

RUN python -m pip install --root-user-action=ignore --upgrade pip && \
    if [ "$INSTALL_TORCH_CPU" = "true" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu --prefer-binary -r requirements.txt; \
    else \
        pip install --root-user-action=ignore --prefer-binary -r requirements.txt; \
    fi

#RUN --mount=type=cache,target=/root/.cache/pip \
    #pip install --prefer-binary \
    #numpy==1.19.5 \
    #opencv-python-headless==4.6.0.66 \
    #librosa==0.8.1 \
    #numba==0.56.4 \
    #soundfile==0.12.1 \
    #tqdm==4.66.4

# Clona e instala Wav2Lip SIN deps
#RUN     

# --- Wav2Lip + checkpoints ---
ARG WAV2LIP_GAN_URL="https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
ARG WAV2LIP_GAN_URL_ALT1="https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip_gan.pth"
ARG WAV2LIP_GAN_URL_ALT2="https://huggingface.co/Non-playing-Character/Wave2lip/resolve/main/wav2lip_gan.pth"
ARG WAV2LIP_STD_URL="https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip.pth"
ARG S3FD_URL="https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
ARG S3FD_URL_ALT="https://huggingface.co/wsj1995/sadTalker/resolve/main/s3fd-619a316812.pth"

RUN set -eux; \
    mkdir -p /app/external; \
    rm -rf /app/external/Wav2Lip; \
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip /app/external/Wav2Lip; \
    mkdir -p /app/external/Wav2Lip/checkpoints /app/external/Wav2Lip/face_detection/detection/sfd; \
    \
    (curl -fL --retry 3 --retry-delay 3 "$WAV2LIP_GAN_URL"    -o /app/external/Wav2Lip/checkpoints/wav2lip_gan.pth \
  || curl -fL --retry 3 --retry-delay 3 "$WAV2LIP_GAN_URL_ALT1" -o /app/external/Wav2Lip/checkpoints/wav2lip_gan.pth \
  || curl -fL --retry 3 --retry-delay 3 "$WAV2LIP_GAN_URL_ALT2" -o /app/external/Wav2Lip/checkpoints/wav2lip_gan.pth \
  || (echo "FALLBACK: usando wav2lip.pth (no-GAN)"; \
      curl -fL --retry 3 --retry-delay 3 "$WAV2LIP_STD_URL" -o /app/external/Wav2Lip/checkpoints/wav2lip.pth)); \
    if [ -f /app/external/Wav2Lip/checkpoints/wav2lip_gan.pth ]; then \
        [ $(stat -c%s /app/external/Wav2Lip/checkpoints/wav2lip_gan.pth) -gt 400000000 ]; \
    else \
        [ $(stat -c%s /app/external/Wav2Lip/checkpoints/wav2lip.pth) -gt 100000000 ]; \
    fi; \
    \
    (curl -fL --retry 3 --retry-delay 3 "$S3FD_URL" -o /app/external/Wav2Lip/face_detection/detection/sfd/s3fd.pth \
  || curl -fL --retry 3 --retry-delay 3 "$S3FD_URL_ALT" -o /app/external/Wav2Lip/face_detection/detection/sfd/s3fd.pth); \
    [ $(stat -c%s /app/external/Wav2Lip/face_detection/detection/sfd/s3fd.pth) -gt 80000000 ];

# 👉 Haz que el código de Wav2Lip sea importable
ENV PYTHONPATH="/app/external/Wav2Lip:${PYTHONPATH}"


# Copia tu código
COPY . /app

RUN rm -rf /root/.cache/pip
RUN mkdir -p work models app/static

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
#CMD ["python","-m","app.main"]