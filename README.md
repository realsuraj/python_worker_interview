# Simple Interview AI Worker

Minimal FastAPI worker for mock interview flow.

## Endpoints

- `GET /health`
- `POST /startinterveiw`
- `POST /askquestion`
- `POST /matchanswer`

Only these interview APIs are kept.

## Models Used

- Question + ideal answer generation: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Answer checking: `sentence-transformers/all-MiniLM-L6-v2`

## Flow

1. `startinterveiw`: creates session cache and pre-generates first question+ideal answer.
2. `askquestion`: returns next cached question.
3. `matchanswer`: scores candidate answer against ideal answer with MiniLM.
   - After first match, worker precomputes remaining questions sequentially in background.
   - If answer is weak for a specific concept, worker queues a probing counter-question.
   - Next `askquestion` call returns that counter-question automatically.

## Run with Docker

```bash
docker compose up --build
```

Worker runs on port `9000` by default.

## Environment Variables

- `WORKER_HOST` (default `0.0.0.0`)
- `WORKER_PORT` (default `9000`)
- `DEEPSEEK_MODEL_NAME` (default `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
- `SEMANTIC_MODEL_NAME` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `DEFAULT_TOTAL_QUESTIONS` (default `15`)
- `MAX_TOTAL_QUESTIONS` (default `15`)
- `INTERVIEW_SESSION_TTL_SECONDS` (default `28800`)

## API Examples

### Start

```bash
curl -X POST http://localhost:9000/startinterveiw \
  -H "Content-Type: application/json" \
  -d '{"role":"Java Developer","difficulty":"medium","totalQuestions":8}'
```

### Ask

```bash
curl -X POST http://localhost:9000/askquestion \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"<SESSION_ID>"}'
```

### Match

```bash
curl -X POST http://localhost:9000/matchanswer \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"<SESSION_ID>","questionId":"q1","answer":"..."}'
```
