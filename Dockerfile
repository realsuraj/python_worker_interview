FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# System deps — Python 3.11, ffmpeg (required by faster-whisper to decode
# Chrome's audio/webm;codecs=opus from MediaRecorder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    curl ca-certificates ffmpeg \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 9000

ENV WORKER_HOST=0.0.0.0
ENV WORKER_PORT=9000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS "http://localhost:${WORKER_PORT}/health" || exit 1

CMD ["python", "main.py"]

