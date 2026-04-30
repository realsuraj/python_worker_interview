# Interview AI Worker

FastAPI-based interview worker with RAG question generation and answer evaluation.

## Quick Start (Docker)

```bash
git clone <your-repo>
cd ai_worker

docker compose up --build
```

First build installs Python dependencies only (no STT/TTS model download).

Worker available at: `http://localhost:9000`

---

## Environment Variables

Override any of these in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `WORKER_PUBLIC_PORT` | `9000` | Host port published by Docker (`host:container`) |
| `WORKER_PORT` | `9000` | Container listen port |
| `ENABLE_LLM` | `false` | Enable external LLM API calls |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL used for task-routed generation |
| `OLLAMA_LIGHT_MODEL` | `qwen2:1.5b` | Default model for lightweight AI tasks such as resume analysis, matching, suggestions, and job drafting |
| `OLLAMA_HEAVY_MODEL` | `mistral:7b-instruct` | Default model for heavy interview flows such as question generation, evaluation, counter-questioning, and voice-turn orchestration |
| `OLLAMA_AUTO_PULL` | `true` | If a selected Ollama model is missing, the worker calls `/api/pull` before falling back |
| `OLLAMA_PRELOAD_ON_STARTUP` | `true` | Pre-check and pre-pull the light/heavy Ollama models during worker startup |
| `OLLAMA_PULL_TIMEOUT_SECONDS` | `900` | Maximum wait time for a model pull request to complete |
| `PY_WORKER_NETWORK_MODE` | `bridge` | GitHub deployment mode for the server container: use `host` on Linux when Ollama is only reachable via `127.0.0.1:11434` on the host |
| `LLM_MODEL_NAME` | `external-api` | Label shown in API responses |
| `LLM_API_URL` | empty | External LLM endpoint URL |
| `LLM_API_KEY` | empty | Bearer token for external LLM endpoint |
| `LLM_API_TIMEOUT_SECONDS` | `45` | Timeout for external LLM calls |
| `AI_FOUNDATION_MIN_SAMPLES` | `300` | Minimum cleaned examples required before exporting a training-ready corpus |
| `AI_FOUNDATION_DATASET_FILE` | `data/ai_foundation_dataset.jsonl` | Cleaned raw capture file for future fine-tuning datasets |
| `AI_FOUNDATION_EXPORT_FILE` | `data/ai_foundation_training_ready.jsonl` | Training-ready export generated after sample threshold is reached |
| `AI_FOUNDATION_STATS_FILE` | `data/ai_foundation_dataset_stats.json` | Aggregate stats for the foundation-model dataset pipeline |
| `ENABLE_STT` | `false` | Browser-side STT mode (backend STT model disabled) |
| `ENABLE_TTS` | `false` | Browser-side TTS mode (backend TTS model disabled) |
| `RAG_FETCH_ONLINE` | `true` | Fetch RAG sources from the web |
| `COUNTER_Q_ENABLED` | `true` | Enable counter/follow-up questions |
| `AUTO_TRAIN_ENABLED` | `true` | Auto-train in idle time |
| `AUTO_TRAIN_IDLE_SECONDS` | `60` | Start auto-train after this idle time (1-minute countdown) |
| `AUTO_TRAIN_MIN_GAP_SECONDS` | `900` | Minimum gap between auto-train runs |
| `AUTO_DISCOVER_PER_DOMAIN` | `120` | URLs discovered per domain per auto-train |
| `TRAIN_FETCH_URL_LIMIT` | `80` | Max URLs fetched per domain per training run |
| `DISCOVERY_COOLDOWN_SECONDS` | `30` | Cooldown between online discovery searches (per domain) |
| `ON_DEMAND_DISCOVERY_ENABLED` | `false` | Enable/disable live discovery during question-start requests (disable for faster start) |
| `INTERVIEW_SESSION_TTL_SECONDS` | `28800` | Pause auto-training while interview session is active (default 8 hours lock) |
| `SEARCH_ENGINES` | `duckduckgo,bing,brave` | Fallback engines for online discovery (auto-prioritized by success rate) |
| `VECTOR_CACHE_ENABLED` | `true` | Store fetched URL cache in vector form on disk |
| `VECTOR_CACHE_DIM` | `256` | Hash-vector dimension for cached lines |
| `VECTOR_TOP_K` | `24` | Top vector-matched lines loaded per URL |
| `URL_TEXT_MEMORY_CAP` | `12` | Max text lines retained in RAM per URL |
| `RL_ENABLED` | `true` | Enable reinforcement learning loop (question reward updates) |
| `RL_RAW_EVENT_RETENTION_DAYS` | `90` | Keep raw RL events only for this many days (daily buckets are retained) |
| `SMALL_MODEL_ENABLED` | `true` | Enable compact ML model for question ranking |
| `SMALL_MODEL_FILE` | `small_question_model.pkl` | Path for trained small model artifact |
| `DATASET_ONLY_MODE` | `true` | For Java interviews, rebuild and serve questions/answers only from the StackOverflow QA dataset, without legacy training-store fallback |
| `STRICT_CANDIDATE_IDENTITY` | `true` | Require candidate identity/session token for uniqueness tracking |
| `RESOURCE_PROFILE` | `medium` | Cost profile: `low`, `medium`, `high` (caps discover/fetch/train usage) |
| `SOURCE_ALLOWLIST_DOMAINS` | empty | Optional comma-separated allowlist domains for source saving |
| `SOURCE_QUALITY_FILE` | `source_quality.json` | URL pass/fail quality tracking store |
| `LEARNING_LOG_FILE` | `learning_log.json` | Learning/training event log store |
| `LEARNING_LOG_MAX_ITEMS` | `5000` | Max retained learning-log entries |
| `LEARNING_POLICY_FILE` | `learning_policy.json` | Adaptive learning state (novelty + cadence) |
| `FAST_LEARN_WEEKS` | `7` | Aggressive learning window before slowing down |
| `NOVEL_URL_REPEAT_DAYS` | `21` | Revisit URL only after this many days (novelty-first) |
| `DAILY_NEW_URL_TARGET` | `20` | Minimum target of new URLs per day before repeats |
| `SEMANTIC_MATCH_ENABLED` | `true` | Concept/meaning-based answer matching |
| `SEMANTIC_MODEL_NAME` | `tfidf-cosine-pkl` | TF-IDF semantic scorer label |

Startup behavior:
- By default, the worker does not run broad web auto-learning or search-engine discovery.
- On worker startup, the small model checks whether it has already learned the default dataset source:
  - `https://huggingface.co/datasets/mteb/stackoverflow-qa`
- If that dataset source is missing, or if the current model still contains older non-dataset sources, the worker rebuilds `small_question_model.pkl` from the StackOverflow QA dataset before Java interview usage continues.
- In dataset-only mode, Java simple-interview questions and ideal answers come only from the dataset-backed question bank and QA bank stored in `small_question_model.pkl`.

Example `.env`:
```env
WORKER_PORT=9000
WORKER_PUBLIC_PORT=9000
ENABLE_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LIGHT_MODEL=qwen2:1.5b
OLLAMA_HEAVY_MODEL=mistral:7b-instruct
OLLAMA_AUTO_PULL=true
OLLAMA_PRELOAD_ON_STARTUP=true
OLLAMA_PULL_TIMEOUT_SECONDS=900
AI_FOUNDATION_MIN_SAMPLES=500
ENABLE_STT=false
ENABLE_TTS=false
```

Model routing policy:

- Lightweight platform tasks default to Qwen 2B through Ollama.
- Interview-grade generation and evaluation tasks default to Mistral 7B through Ollama.
- On startup, the worker can pre-check and pre-pull both model tags so first-request latency stays low.
- Before use, the worker checks Ollama `/api/tags`; if the model is missing and `OLLAMA_AUTO_PULL=true`, it calls `/api/pull` for that exact tag.
- If Ollama is unreachable or the pull fails, the worker keeps external `LLM_API_URL` support as a fallback when configured.

Linux server note:

- If the worker runs in Docker and Ollama runs on the same Linux host bound to `127.0.0.1:11434`, set the GitHub secret `PY_WORKER_NETWORK_MODE=host`.
- If you stay on bridge mode, make sure Ollama listens on a host-reachable address such as `0.0.0.0:11434`.
- Your worker will only auto-pull the exact configured model tags. If `ollama list` does not show `qwen2:1.5b` and `mistral:7b-instruct`, the pull step will try those exact names after connectivity is fixed.

Foundation dataset pipeline:

- Every worker-routed AI request stores a cleaned training example with phone/email masking.
- Raw cleaned captures accumulate in `ai_foundation_dataset.jsonl`.
- Once the sample threshold is met, `POST /ml/foundation-dataset/build` exports a training-ready JSONL corpus for future fine-tuning or distillation work.
- `GET /ml/foundation-dataset/status` reports current counts by task, model, and complexity so you can decide when to train.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ai/infer` | Generic worker-first task inference entrypoint used by the Java backend |
| POST | `/ai/resume/analyze` | Resume parsing, OCR-aware intake, ATS scoring, gap analysis |
| POST | `/ai/candidate/match` | Candidate-job matching, skill gaps, certification guidance |
| POST | `/ai/behavior/analyze` | Behavioral/communication signal analysis |
| POST | `/ai/suggestions` | Candidate preparation and career-path suggestions |
| POST | `/ai/interview/questions` | Structured HR interview pack generation |
| POST | `/ai/job-draft` | Job draft and JD regeneration support |
| POST | `/ai/domain/detect` | Resume/domain inference |
| POST | `/ai/coding/testcases` | Coding testcase generation fallback surface |
| POST | `/ai/coding/solution-bundle` | Coding starter/editorial bundle fallback surface |
| POST | `/ai/voice/turn` | Real-time voice-turn orchestration metadata for validation/counter-questioning |
| POST | `/interview/questions` | Generate interview questions |
| POST | `/interview/evaluate` | Evaluate candidate answers |
| POST | `/interview/counter-questions` | Generate follow-up questions |
| POST | `/interview/evaluate-with-followup` | Evaluate + follow-up in one call |
| GET/POST | `/rag/sources` | View / update RAG source map |
| POST | `/rag/train-once` | One-time market-data training + store questions by domain |
| GET | `/rag/training-store` | Read persisted training store |
| GET | `/rag/learning-log` | Learning/training activity log (`event/domain/limit` filters) |
| GET | `/rag/source-quality` | URL quality report (pass/fail, noisy-source view) |
| GET | `/rag/learning-policy` | Adaptive mode/knowledge/novelty status |
| GET | `/reinforcement/state` | Read reinforcement learning state |
| GET | `/ml/learning-stats` | 1-week / 1-month / 1-year learning stats |
| GET | `/ml/foundation-dataset/status` | Inspect cleaned dataset volume, task mix, model mix, and training readiness |
| POST | `/ml/foundation-dataset/build` | Export the cleaned capture store into a training-ready JSONL corpus once enough samples exist |
| POST | `/ml/train-small-model` | Train compact reusable ML model from RL events |
| POST | `/ml/online-training-dataset` | Estimate or train `small_question_model.pkl` from dataset/source URLs; for Hugging Face datasets, `domain` can be left blank and labels are inferred automatically |
| POST | `/ml/model-source-status` | Check whether the current small model already learned given URLs |
| GET | `/ml/export-model` | Return model export metadata or download `small_question_model.pkl` |
| POST | `/interview/best-answer` | Generate best interview answers from trained store + URLs |
| POST | `/speech/transcribe` | Stub response (backend STT disabled) |
| POST | `/speech/synthesize` | Stub response (backend TTS disabled) |
| WS | `/speech/ws/stt` | Available, returns disabled-mode responses |

---

## Enterprise Worker-First Flow

The platform now supports a worker-first AI routing model:

- Spring Boot sends AI tasks to `/ai/infer` on this worker first.
- The worker handles resume intelligence, job matching, interview generation/evaluation, coding-assist fallbacks, and voice-turn orchestration metadata.
- If the worker is unavailable or returns no usable result, the backend falls back to `HuggingFaceAiService` inference.

This keeps the execution surface centralized in `python_worker_interview` while still preserving a hosted-model fallback path.

Recommended backend properties:

```properties
app.ai.worker.enabled=true
app.ai.worker.url=http://localhost:9000
app.ai.worker.timeout-ms=12000
```

Resume intake notes:

- Searchable PDF and DOCX parsing is supported directly in the worker.
- Image/scanned resume OCR uses PaddleOCR when installed, otherwise falls back to Tesseract if available.
- Browser-side speech remains the active STT/TTS path today; `/ai/voice/turn` returns orchestration metadata for low-latency validation and counter-question prediction.

Scaling direction for this worker:

- Run CPU-first resume/matching tasks on horizontal FastAPI replicas.
- Keep heavier model-backed routes behind a queue or separate GPU deployment tier.
- Cache normalized resume/job features and precomputed interview trees to reduce repeated inference cost.
- Route premium or overflow analysis to larger external models only when the local worker confidence is low.

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

## One-Time Training Example

Train/store 5 questions each for Java, Sales, Business Development, Voice Process, Chat Process, Flutter, React:

```bash
curl -X POST http://localhost:9000/rag/train-once \
  -H "Content-Type: application/json" \
  -d '{
    "domains": ["java", "sales", "business_development", "voice_process", "chat_process", "flutter", "react"],
    "includeDefaults": true,
    "discoverUrls": true,
    "forceRefresh": false,
    "discoveredUrlLimitPerDomain": 120,
    "questionCountPerDomain": 5,
    "role": "Java Developer",
    "department": "Engineering",
    "stream": "Java"
  }'
```

Training output is persisted to `training_store.json` (configurable via `TRAINING_STORE_FILE`).
Questions are not hardcoded; they are mined from fetched web content and optionally synthesized by LLM from that fetched context.
Training emits step logs with `[TRAIN]` prefix (countdown, start, discover, fetch, mine, llm_synthesis, save, done).
Each discovered/fetched URL is actively opened and tracked as pass/fail with reason (saved in `source_quality.json`).
Worker now learns in two phases: fast for first `FAST_LEARN_WEEKS`, then slow mode with novelty-first reinforcement to avoid repeating old links/questions.
Evaluator uses semantic concept matching so answers are graded by meaning, not exact word-by-word text.
Question uniqueness is tracked per candidate in `candidate_question_state.json` (configurable via `CANDIDATE_STATE_FILE`).
If strict identity is enabled, uniqueness requires `candidateId/applicationId/interviewId/sessionToken` fallback to avoid cross-user collisions.
When a candidate exhausts the list, worker auto-learns online and refills up to 50 question/answer pairs, then repeats only if needed.

Enable external LLM later (optional):

```env
ENABLE_LLM=true
LLM_API_URL=https://your-llm-api.example.com/generate
LLM_API_KEY=your-token
LLM_MODEL_NAME=gpt-or-any
```

Generate best answer(s):

```bash
curl -X POST http://localhost:9000/interview/best-answer \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "react",
    "questions": [
      "How do useEffect dependencies influence re-renders and side effects?"
    ],
    "questionUrls": ["https://www.interviewbit.com/react-interview-questions/"]
  }'
```

---

## What's NOT in Git

| Path | Size | Source |
|---|---|---|
| `_pydeps/` | small | pip local deps |

