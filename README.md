# Interview AI Worker

FastAPI-based interview worker with RAG question generation, answer evaluation, STT, and TTS.

## Quick Start (Docker)

```bash
git clone <your-repo>
cd ai_worker

docker compose up --build
```

First build takes ~10–15 min (downloads DeepSeek ~1.5GB + piper voice model).  
Subsequent starts are instant — models are cached in a Docker volume.

Worker available at: `http://localhost:8099`

---

## Environment Variables

Override any of these in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `WORKER_PORT` | `8099` | Port to expose |
| `DEEPSEEK_MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | HuggingFace model ID |
| `ENABLE_STT` | `false` | Enable speech-to-text (Whisper) |
| `ENABLE_TTS` | `false` | Enable text-to-speech (Piper) |
| `RAG_FETCH_ONLINE` | `true` | Fetch RAG sources from the web |
| `COUNTER_Q_ENABLED` | `true` | Enable counter/follow-up questions |

Example `.env`:
```env
WORKER_PORT=8099
ENABLE_STT=true
ENABLE_TTS=true
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/interview/questions` | Generate interview questions |
| POST | `/interview/evaluate` | Evaluate candidate answers |
| POST | `/interview/counter-questions` | Generate follow-up questions |
| POST | `/interview/evaluate-with-followup` | Evaluate + follow-up in one call |
| GET/POST | `/rag/sources` | View / update RAG source map |
| POST | `/speech/transcribe` | Transcribe audio (STT) |
| POST | `/speech/synthesize` | Synthesize speech (TTS) |
| WS | `/speech/ws/stt` | Real-time STT via WebSocket |

---

## Local Dev (without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
python main.py
```

---

## What's NOT in Git (downloaded at build time)

| Path | Size | Source |
|---|---|---|
| `.hf_cache/` | ~1.5 GB | HuggingFace (DeepSeek model) |
| `models/piper/` | ~60 MB | HuggingFace (piper voices) |
| `_pydeps/` | small | pip local deps |
