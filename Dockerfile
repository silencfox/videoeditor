FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dentro del Dockerfile, después de instalar dependencias
RUN git clone https://github.com/Rudrabha/Wav2Lip /app/external/Wav2Lip && \
    cd /app/external/Wav2Lip && \
    pip install -r requirements.txt


COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the rest
COPY . /app

# Folders
RUN mkdir -p work models app/static

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
