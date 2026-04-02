FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download piper binary
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /usr/local/bin --strip-components=1 piper/piper \
    && chmod +x /usr/local/bin/piper \
    && rm /tmp/piper.tar.gz

# Download piper voice model
RUN mkdir -p /app/models/piper \
    && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
       -O /app/models/piper/voice.onnx \
    && wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
       -O /app/models/piper/voice.onnx.json

# Copy source
COPY . .

# Pre-download HuggingFace DeepSeek model into image
ENV HF_HOME=/app/.hf_cache
RUN python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ignore_patterns=["*.h5","*.msgpack","*.onnx","*.tflite","*.ckpt"],
)
EOF

EXPOSE 8099

ENV WORKER_HOST=0.0.0.0
ENV WORKER_PORT=8099
ENV PIPER_BIN=/usr/local/bin/piper
ENV PIPER_VOICE_MODEL=/app/models/piper/voice.onnx

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8099/health || exit 1

CMD ["python", "main.py"]
