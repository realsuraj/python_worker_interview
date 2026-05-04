FROM python:3.11-slim

WORKDIR /app

# System deps — ffmpeg is required by faster-whisper to decode
# Chrome's audio/webm;codecs=opus format from MediaRecorder
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates ffmpeg \
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

