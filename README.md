# Interview AI Worker

FastAPI-based interview worker with RAG question generation and answer evaluation.

## Quick Start (Docker)

```bash
git clone <your-repo>
cd ai_worker

docker compose up --build
```

First build installs Python dependencies only (no STT/TTS model download).

Worker available at: `http://localhost:8099`

---

## Environment Variables

Override any of these in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `WORKER_PUBLIC_PORT` | `8099` | Host port published by Docker (`host:container`) |
| `WORKER_PORT` | `8099` | Container listen port |
| `ENABLE_LLM` | `false` | Enable external LLM API calls |
| `LLM_MODEL_NAME` | `external-api` | Label shown in API responses |
| `LLM_API_URL` | empty | External LLM endpoint URL |
| `LLM_API_KEY` | empty | Bearer token for external LLM endpoint |
| `LLM_API_TIMEOUT_SECONDS` | `45` | Timeout for external LLM calls |
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
| `SEMANTIC_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Semantic embedding model |

Example `.env`:
```env
WORKER_PORT=8099
WORKER_PUBLIC_PORT=8099
ENABLE_STT=false
ENABLE_TTS=false
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
| POST | `/rag/train-once` | One-time market-data training + store questions by domain |
| GET | `/rag/training-store` | Read persisted training store |
| GET | `/rag/learning-log` | Learning/training activity log (`event/domain/limit` filters) |
| GET | `/rag/source-quality` | URL quality report (pass/fail, noisy-source view) |
| GET | `/rag/learning-policy` | Adaptive mode/knowledge/novelty status |
| GET | `/reinforcement/state` | Read reinforcement learning state |
| GET | `/ml/learning-stats` | 1-week / 1-month / 1-year learning stats |
| POST | `/ml/train-small-model` | Train compact reusable ML model from RL events |
| POST | `/interview/best-answer` | Generate best interview answers from trained store + URLs |
| POST | `/speech/transcribe` | Stub response (backend STT disabled) |
| POST | `/speech/synthesize` | Stub response (backend TTS disabled) |
| WS | `/speech/ws/stt` | Available, returns disabled-mode responses |

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
curl -X POST http://localhost:8099/rag/train-once \
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
curl -X POST http://localhost:8099/interview/best-answer \
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
