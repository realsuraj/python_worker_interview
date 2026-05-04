"""
Interview AI Worker  —  v5.0.0
Dual-Engine Architecture:
  • QuestionEngine    → generates initial interview questions (RAG + JD + resume)
  • CounterEngine     → generates follow-up / probing questions from a candidate answer
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import importlib
import json
import logging
import os
import pickle
import random
import re
import sys
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

LOCAL_PY_DEPS = Path(__file__).with_name("_pydeps")
LOCAL_PY_DEPS.mkdir(parents=True, exist_ok=True)
if str(LOCAL_PY_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_PY_DEPS))

from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel

from app.api.enterprise_ai_routes import register_enterprise_ai_routes
from app.api.ml_routes import register_ml_routes
from app.schemas.ml import ModelSourceStatusRequest, OnlineTrainingSetRequest, SmallModelTrainRequest
from app.services.foundation_pipeline import OLLAMA_PRELOAD_ON_STARTUP, append_foundation_example, ensure_ollama_model_available, ollama_model_inventory, preload_ollama_models, resolve_model_profile

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import polars as pl
except Exception:
    pl = None

if os.getenv("DATASET_ONLY_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}:
    HashingVectorizer = None
    SGDRegressor = None
    TfidfVectorizer = None
    cosine_similarity = None
else:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
        from sklearn.linear_model import SGDRegressor
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        HashingVectorizer = None
        SGDRegressor = None
        TfidfVectorizer = None
        cosine_similarity = None

# ── env / config ───────────────────────────────────────────────────────────────
def _env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(key: str, default: int, minimum: int = 0) -> int:
    return max(minimum, int(os.getenv(key, str(default))))

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return float(default)

def _env_port(default: int = 9000) -> int:
    """
    Resolve runtime port with platform compatibility.
    Priority: WORKER_PORT -> PORT -> default
    """
    raw = (
        str(os.getenv("WORKER_PORT", "")).strip()
        or str(os.getenv("PORT", "")).strip()
        or str(default)
    )
    try:
        return int(raw)
    except Exception:
        return default

def _default_worker_path(filename: str) -> Path:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / filename

def _state_file_path(env_key: str, filename: str) -> Path:
    env_path = str(os.getenv(env_key, "")).strip()
    if env_path:
        return Path(env_path)
    legacy = Path(__file__).with_name(filename)
    if legacy.exists():
        return legacy
    return _default_worker_path(filename)

MODEL_NAME              = os.getenv("LLM_MODEL_NAME", "external-api")
ENABLE_LLM              = _env_bool("ENABLE_LLM", "false")
LLM_API_URL             = os.getenv("LLM_API_URL", "").strip()
LLM_API_KEY             = os.getenv("LLM_API_KEY", "").strip()
LLM_API_TIMEOUT_SECONDS = _env_int("LLM_API_TIMEOUT_SECONDS", 45, 5)
LLM_MAX_NEW_TOKENS      = _env_int("LLM_MAX_NEW_TOKENS", 320, 96)
LLM_CONTEXT_LINES       = _env_int("LLM_CONTEXT_LINES", 30, 5)
LLM_POOL_LIMIT          = _env_int("LLM_POOL_LIMIT", 60, 10)
LLM_LINE_WORD_LIMIT     = _env_int("LLM_LINE_WORD_LIMIT", 20, 8)
LLM_USE_FOR_EVALUATION  = _env_bool("LLM_USE_FOR_EVALUATION", "false")
WORKER_HOST             = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT             = _env_port(9000)

RAG_FETCH_ONLINE        = _env_bool("RAG_FETCH_ONLINE", "false")
RAG_SOURCES_FILE        = _state_file_path("RAG_SOURCES_FILE", "rag_sources.json")
RAG_FETCH_TIMEOUT       = _env_int("RAG_FETCH_TIMEOUT_SECONDS", 3, 1)
RAG_MAX_URLS            = _env_int("RAG_MAX_URLS", 12, 1)
RAG_PREFETCH_ON_STARTUP = _env_bool("RAG_PREFETCH_ON_STARTUP", "false")
AUTO_TRAIN_ENABLED      = _env_bool("AUTO_TRAIN_ENABLED", "false")
LEARNING_ALWAYS_ENABLED = _env_bool("LEARNING_ALWAYS_ENABLED", "false")
AUTO_TRAIN_IDLE_SECONDS = _env_int("AUTO_TRAIN_IDLE_SECONDS", 60, 30)
AUTO_TRAIN_MIN_GAP_SEC  = _env_int("AUTO_TRAIN_MIN_GAP_SECONDS", 900, 60)
AUTO_DISCOVER_PER_DOMAIN = _env_int("AUTO_DISCOVER_PER_DOMAIN", 120, 2)
AUTO_TRAIN_DOMAIN_LIST  = [d.strip().lower() for d in os.getenv(
    "AUTO_TRAIN_DOMAINS",
    "java,sales,business_development,voice_process,chat_process,flutter,react"
).split(",") if d.strip()]
TRAIN_FETCH_URL_LIMIT   = _env_int("TRAIN_FETCH_URL_LIMIT", 80, 10)
DISCOVERY_COOLDOWN_SECONDS = _env_int("DISCOVERY_COOLDOWN_SECONDS", 30, 0)
ON_DEMAND_DISCOVERY_ENABLED = _env_bool("ON_DEMAND_DISCOVERY_ENABLED", "false")
INTERVIEW_SESSION_TTL_SECONDS = _env_int("INTERVIEW_SESSION_TTL_SECONDS", 28800, 600)
SEARCH_ENGINES            = [x.strip().lower() for x in os.getenv("SEARCH_ENGINES", "duckduckgo,bing,brave").split(",") if x.strip()]
SEARCH_BLOCK_STATUS_CODES = {403, 429}
VECTOR_CACHE_ENABLED      = _env_bool("VECTOR_CACHE_ENABLED", "true")
VECTOR_CACHE_DIM          = _env_int("VECTOR_CACHE_DIM", 256, 64)
VECTOR_TOP_K              = _env_int("VECTOR_TOP_K", 24, 4)
URL_TEXT_MEMORY_CAP       = _env_int("URL_TEXT_MEMORY_CAP", 12, 4)
VECTOR_CACHE_DIR          = Path(os.getenv("VECTOR_CACHE_DIR", str(_default_worker_path("vector_cache").with_suffix(""))))
VECTOR_CACHE_FILE         = _state_file_path("VECTOR_CACHE_FILE", "vector_cache.json")
VECTOR_CACHE_MAX_BYTES    = _env_int("VECTOR_CACHE_MAX_BYTES", 25 * 1024 * 1024, 1_048_576)
RL_ENABLED                = _env_bool("RL_ENABLED", "true")
SMALL_MODEL_ENABLED       = _env_bool("SMALL_MODEL_ENABLED", "true")
SMALL_MODEL_AUTO_TRAIN_ENABLED = _env_bool("SMALL_MODEL_AUTO_TRAIN_ENABLED", "true")
SMALL_MODEL_AUTO_TRAIN_WINDOW_DAYS = _env_int("SMALL_MODEL_AUTO_TRAIN_WINDOW_DAYS", 365, 7)
SMALL_MODEL_AUTO_TRAIN_MIN_SAMPLES = _env_int("SMALL_MODEL_AUTO_TRAIN_MIN_SAMPLES", 80, 20)
SMALL_MODEL_AUTO_TRAIN_GAP_SECONDS = _env_int("SMALL_MODEL_AUTO_TRAIN_GAP_SECONDS", 21600, 900)
DATASET_ONLY_MODE         = _env_bool("DATASET_ONLY_MODE", "true")
STRICT_CANDIDATE_IDENTITY = _env_bool("STRICT_CANDIDATE_IDENTITY", "true")
RL_RAW_EVENT_RETENTION_DAYS = _env_int("RL_RAW_EVENT_RETENTION_DAYS", 90, 7)
LEARNING_LOG_MAX_ITEMS    = _env_int("LEARNING_LOG_MAX_ITEMS", 5000, 100)
RESOURCE_PROFILE          = os.getenv("RESOURCE_PROFILE", "medium").strip().lower()
SOURCE_ALLOWLIST_DOMAINS  = [d.strip().lower() for d in os.getenv("SOURCE_ALLOWLIST_DOMAINS", "").split(",") if d.strip()]
FAST_LEARN_WEEKS          = _env_int("FAST_LEARN_WEEKS", 7, 1)
NOVEL_URL_REPEAT_DAYS     = _env_int("NOVEL_URL_REPEAT_DAYS", 21, 1)
DAILY_NEW_URL_TARGET      = _env_int("DAILY_NEW_URL_TARGET", 20, 1)
SEMANTIC_MATCH_ENABLED  = _env_bool("SEMANTIC_MATCH_ENABLED", "true")
SEMANTIC_MODEL_NAME     = os.getenv("SEMANTIC_MODEL_NAME", "tfidf-cosine-pkl")
MATCHANSWER_ONLINE_ENABLED = _env_bool("MATCHANSWER_ONLINE_ENABLED", "false")
MATCHANSWER_ONLINE_MAX_SENTENCES = _env_int("MATCHANSWER_ONLINE_MAX_SENTENCES", 1, 0)
MATCHANSWER_ONLINE_MAX_URLS = _env_int("MATCHANSWER_ONLINE_MAX_URLS", 1, 0)

INTERVIEW_QUESTION_COUNT  = _env_int("INTERVIEW_QUESTION_COUNT", 14, 4)
TARGET_WORDS_PER_ANSWER   = _env_int("TARGET_WORDS_PER_ANSWER", 70, 20)
MIN_WORDS_PER_MINUTE      = _env_int("MIN_WORDS_PER_MINUTE", 50, 10)
MAX_ANSWER_WORDS          = _env_int("MAX_ANSWER_WORDS", 220, 80)

# Counter-question engine config
COUNTER_Q_ENABLED         = _env_bool("COUNTER_Q_ENABLED", "true")
COUNTER_Q_PER_ANSWER      = _env_int("COUNTER_Q_PER_ANSWER", 2, 1)
COUNTER_Q_DEPTH_THRESHOLD = _env_int("COUNTER_Q_DEPTH_THRESHOLD", 40, 0)   # score below → probe harder

ENABLE_STT              = _env_bool("ENABLE_STT", "true")
ENABLE_TTS              = _env_bool("ENABLE_TTS", "false")
STT_MODEL_ID            = os.getenv("STT_MODEL_ID", "base").strip() or "base"
STT_PRIORITY_MODE       = os.getenv("STT_PRIORITY_MODE", "worker_first").strip() or "worker_first"
STT_DEVICE              = os.getenv("STT_DEVICE", "cpu").strip() or "cpu"
STT_COMPUTE_TYPE        = os.getenv("STT_COMPUTE_TYPE", "int8").strip() or "int8"
STT_BEAM_SIZE           = _env_int("STT_BEAM_SIZE", 1, 1)
STT_MIN_CHUNK_BYTES     = _env_int("STT_MIN_CHUNK_BYTES", 512, 256)
CORE_JAVA_REFERENCE_URL  = "https://www.interviewbit.com/java-interview-questions/"
TRAINING_STORE_FILE      = _state_file_path("TRAINING_STORE_FILE", "training_store.json")
CANDIDATE_STATE_FILE     = _state_file_path("CANDIDATE_STATE_FILE", "candidate_question_state.json")
REINFORCEMENT_STATE_FILE = _state_file_path("REINFORCEMENT_STATE_FILE", "reinforcement_state.json")
SMALL_MODEL_FILE         = _state_file_path("SMALL_MODEL_FILE", "small_question_model.pkl")
SEMANTIC_SCORER_FILE     = _state_file_path("SEMANTIC_SCORER_FILE", "semantic_scorer.pkl")
SOURCE_QUALITY_FILE      = _state_file_path("SOURCE_QUALITY_FILE", "source_quality.json")
LEARNING_LOG_FILE        = _state_file_path("LEARNING_LOG_FILE", "learning_log.json")
LEARNING_POLICY_FILE     = _state_file_path("LEARNING_POLICY_FILE", "learning_policy.json")
DEFAULT_ONLINE_TRAINING_SOURCES: List[str] = [
    "https://huggingface.co/datasets/mteb/stackoverflow-qa",
]
STACKOVERFLOW_QA_CACHE_DIR = Path(
    os.getenv(
        "STACKOVERFLOW_QA_CACHE_DIR",
        str(Path(__file__).resolve().parent / "data" / "dataset_cache" / "mteb_stackoverflow_qa"),
    )
)
STACKOVERFLOW_QA_REMOTE_FILES: Dict[str, str] = {
    "default": "https://huggingface.co/datasets/mteb/stackoverflow-qa/resolve/main/data/train-00000-of-00001.parquet?download=true",
    "queries": "https://huggingface.co/datasets/mteb/stackoverflow-qa/resolve/main/queries/queries-00000-of-00001.parquet?download=true",
    "corpus": "https://huggingface.co/datasets/mteb/stackoverflow-qa/resolve/main/corpus/corpus-00000-of-00001.parquet?download=true",
}

VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
STACKOVERFLOW_QA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interview-ai-worker")

# ── word sets ──────────────────────────────────────────────────────────────────
STOPWORDS: Set[str] = {
    "the","and","for","with","that","this","from","your","you","into","are","was","were",
    "how","why","what","when","where","which","can","will","would","about","have","has",
    "had","been","their","they","them","his","her","our","ours","its","but","not","all",
    "any","job","role","work","using","used","than","then","there","also","more","most",
    "over","under","very","into","after","before","during","each","per","should","must",
}
GENERIC_TOPIC_STOPWORDS: Set[str] = {
    "description","developer","development","candidate","candidates","interview",
    "question","questions","experience","responsibilities","responsibility",
    "requirement","requirements","role","roles","skills","skill","technology",
    "technologies","detail","details","project","projects","work",
}
NONSENSE_TOKENS: Set[str] = {"asdf","qwerty","zxcv","blah","lorem","ipsum","random","nothing","idk","na"}
PERSONAL_LINE_KEYWORDS: Set[str] = {
    "address","location","village","vill","post","via","dist","district","state","country",
    "pincode","pin code","pin","dob","date of birth","father","mother","marital status","nationality",
}

# ── global state ───────────────────────────────────────────────────────────────
RAG_SOURCE_MAP:       Dict[str, Dict[str, Any]] = {}
URL_CONTENT_CACHE:    Dict[str, List[str]] = {}
URL_CONTENT_INDEX:    Dict[str, List[str]] = {}
TRAINING_STORE_MAP:   Dict[str, Any] = {}
QUESTION_STATE_MAP:   Dict[str, Any] = {}
REINFORCEMENT_STATE_MAP: Dict[str, Any] = {}
SMALL_MODEL_HANDLE: Any = None
WHISPER_HANDLE: Any = None
SOURCE_QUALITY_MAP: Dict[str, Any] = {}
LEARNING_LOG_MAP: Dict[str, Any] = {}
LEARNING_POLICY_MAP: Dict[str, Any] = {}
SEMANTIC_MODEL_HANDLE: Any = None
SEMANTIC_EMBED_CACHE: Dict[str, Any] = {}
LAST_ACTIVITY_TS      = time.time()
LAST_AUTO_TRAIN_TS    = 0.0
LAST_SMALL_MODEL_TRAIN_TS = 0.0
ACTIVE_STT_WS_COUNT   = 0
ACTIVE_INTERVIEW_SESSIONS: Dict[str, int] = {}
SIMPLE_INTERVIEW_SESSIONS: Dict[str, Dict[str, Any]] = {}
LAST_DISCOVERY_TS: Dict[str, float] = {}
SEARCH_PROVIDER_STATS: Dict[str, Dict[str, int]] = {}
STACKOVERFLOW_QA_TABLE_CACHE: Dict[str, List[Dict[str, Any]]] = {}
QA_REFRESH_STATE: Dict[str, Dict[str, Any]] = {}
WORKER_STARTUP_STATUS: Dict[str, Any] = {
    "llmReady": False,
    "sttReady": False, "ttsReady": False,
    "semanticReady": False,
    "vectorCacheEnabled": VECTOR_CACHE_ENABLED,
    "rlEnabled": RL_ENABLED,
    "smallModelEnabled": SMALL_MODEL_ENABLED,
    "smallModelReady": False,
    "resourceProfile": RESOURCE_PROFILE,
    "fastLearnWeeks": FAST_LEARN_WEEKS,
    "llmReason": "not initialized",
    "sttReason": "not initialized", "ttsReason": "not initialized",
    "semanticReason": "not initialized",
    "smallModelReason": "not initialized",
    "ollamaReachable": False,
    "ollamaModels": {},
    "ollamaReason": "not initialized",
}

app = FastAPI(
    title="Interview AI Worker",
    version="5.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

register_enterprise_ai_routes(app)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

class InterviewRequest(BaseModel):
    interviewId: str = ""; applicationId: str = ""; candidateId: str = ""
    candidateName: str = ""; jobId: str = ""; jobTitle: str = ""
    jobDescription: str = ""; department: str = ""; jdSourceUrl: str = ""
    ragSourceUrls: List[str] = []; interviewSourceUrls: List[str] = []
    onlineInterviewLink: str = ""; customPrompts: List[str] = []
    transcript: str = ""; matchScore: int = 0; threshold: int = 70
    candidateProfile: Dict[str, Any] = {}; resume: Dict[str, Any] = {}
    job: Dict[str, Any] = {}; questions: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []; durationSeconds: int = 0
    language: str = "javascript"; roundTag: str = "technical"
    domain: str = ""; domainCategory: str = ""; domainLabel: str = ""


class InterviewQuestionsRequest(BaseModel):
    type: str = "technical"; tag: str = "technical"; roundTag: str = "technical"
    difficulty: str = "medium"; language: str = "javascript"
    interviewId: str = ""; applicationId: str = ""; candidateId: str = ""
    sessionToken: str = ""
    domain: str = ""; domainCategory: str = ""; domainLabel: str = ""
    jobTitle: str = ""; jobDescription: str = ""; department: str = ""
    jdSourceUrl: str = ""; ragSourceUrls: List[str] = []
    interviewSourceUrls: List[str] = []; onlineInterviewLink: str = ""
    customPrompts: List[str] = []; candidateProfile: Dict[str, Any] = {}
    resume: Dict[str, Any] = {}; job: Dict[str, Any] = {}; sources: List[str] = []


class CounterQuestionRequest(BaseModel):
    """Request a follow-up / counter question for a given answer."""
    questionId:       str = ""
    question:         str = ""
    answer:           str = ""
    answerScore:      int = -1          # -1 = not known; triggers auto-eval
    jobTitle:         str = ""
    jobDescription:   str = ""
    language:         str = "javascript"
    domain:           str = ""
    department:       str = ""
    customPrompts:    List[str] = []
    resume:           Dict[str, Any] = {}
    candidateProfile: Dict[str, Any] = {}
    maxQuestions:     int = COUNTER_Q_PER_ANSWER


class SpeechTranscribeRequest(BaseModel):
    audioPath: str = ""; audioBase64: str = ""; contentType: str = ""; language: str = ""


class SpeechSynthesizeRequest(BaseModel):
    text: str = ""; outputPath: str = ""; returnBase64: bool = False; speaker: str = ""


class OneTimeTrainingRequest(BaseModel):
    domains: List[str] = ["java", "sales", "business_development", "voice_process", "chat_process", "flutter", "react"]
    sourceUrls: Dict[str, List[str]] = {}
    includeDefaults: bool = True
    discoverUrls: bool = True
    forceRefresh: bool = False
    discoveredUrlLimitPerDomain: int = 120
    questionCountPerDomain: int = 5
    role: str = ""
    department: str = ""
    stream: str = ""
    runLabel: str = "manual"


class BestAnswerRequest(BaseModel):
    question: str = ""
    questions: List[str] = []
    domain: str = ""
    language: str = "english"
    jobTitle: str = ""
    jobDescription: str = ""
    questionUrls: List[str] = []
    contextUrls: List[str] = []
    candidateAnswer: str = ""
    maxWords: int = 140


class StartInterveiwRequest(BaseModel):
    candidateId: str = ""
    candidateName: str = ""
    role: str = ""
    jobTitle: str = ""
    difficulty: str = "medium"


class AskQuestionRequest(BaseModel):
    sessionId: str = ""


class MatchAnswerRequest(BaseModel):
    sessionId: str = ""
    questionId: str = ""
    answer: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Pure utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_int(v: Any, fb: int = 0) -> int:
    try: return int(v)
    except Exception: return fb

def _safe_float(v: Any, fb: float = 0.0) -> float:
    try: return float(v)
    except Exception: return fb

def _clamp(v: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, v))

def _normalize(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip()).lower()

def _tokenize(v: Any) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-#]{2,}", _normalize(v))
    return [t for t in tokens if t not in STOPWORDS]

def _limit_words(v: Any, n: int = MAX_ANSWER_WORDS) -> str:
    return " ".join(re.findall(r"\S+", str(v or "").strip())[:n])

def _limit_text_words(v: Any, n: int = LLM_LINE_WORD_LIMIT) -> str:
    return " ".join(re.findall(r"\S+", str(v or "").strip())[:n])

def _split_sentences(v: Any) -> List[str]:
    text = re.sub(r"\s+", " ", str(v or "").strip())
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()] if text else []

def _first_non_blank(*vals: Any) -> str:
    for v in vals:
        t = str(v or "").strip()
        if t: return t
    return ""

def _keywords(v: Any, top_n: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for t in _tokenize(v):
        counts[t] = counts.get(t, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:top_n]]

def _overlap_ratio(text: str, kws: List[str]) -> float:
    if not kws: return 0.0
    hay = set(_tokenize(text))
    return sum(1 for k in kws if k in hay) / max(1, len(kws))

def _url_cache_key(url: str) -> str:
    return hashlib.sha1(str(url).encode("utf-8")).hexdigest()

def _normalize_cache_url(url: str) -> str:
    try:
        parsed = urlparse(str(url).strip())
    except Exception:
        return ""
    scheme = (parsed.scheme or "").lower()
    host = (parsed.netloc or "").strip().lower()
    path = parsed.path or "/"
    if scheme not in {"http", "https"} or not host:
        return ""
    # Cache only canonical URL identity (no query/fragment/text payload).
    return f"{scheme}://{host}{path}"

def _vector_cache_path(url: str) -> Path:
    # Legacy per-URL cache path (kept only for migration from older versions).
    normalized = _normalize_cache_url(url)
    if not normalized:
        normalized = str(url).strip()
    return VECTOR_CACHE_DIR / f"{_url_cache_key(normalized)}.json"

def _vector_cache_store_path() -> Path:
    return VECTOR_CACHE_FILE

def _vector_cache_entry_from_raw(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized = _normalize_cache_url(raw.get("url", ""))
    if not normalized:
        return {}
    parsed = urlparse(normalized)
    updated = str(raw.get("updatedAt", "")).strip() or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "url": normalized,
        "host": (parsed.netloc or "").lower(),
        "path": parsed.path or "/",
        "updatedAt": updated,
        "kind": "url_only",
    }

def _vector_cache_load_store() -> Dict[str, Any]:
    p = _vector_cache_store_path()
    if not p.exists():
        return {"version": 1, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": {}}
    if not isinstance(raw, dict):
        return {"version": 1, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": {}}
    entries_raw = raw.get("entries", {})
    if not isinstance(entries_raw, dict):
        entries_raw = {}
    entries: Dict[str, Any] = {}
    for key, val in entries_raw.items():
        rec = _vector_cache_entry_from_raw(val)
        if rec:
            entries[str(key)] = rec
    return {
        "version": 1,
        "updatedAt": str(raw.get("updatedAt", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))),
        "entries": entries,
    }

def _vector_cache_trim_store(store: Dict[str, Any]) -> Dict[str, int]:
    entries_raw = store.get("entries", {})
    entries = entries_raw if isinstance(entries_raw, dict) else {}
    changed = 0
    dropped = 0
    compact: Dict[str, Any] = {}
    seen_urls: Set[str] = set()
    items = list(entries.items())
    # Keep newest entries first while enforcing size.
    items.sort(key=lambda kv: str((kv[1] or {}).get("updatedAt", "")), reverse=True)
    for key, val in items:
        rec = _vector_cache_entry_from_raw(val)
        if not rec:
            dropped += 1
            changed = 1
            continue
        ukey = _normalize(rec.get("url", ""))
        if ukey in seen_urls:
            dropped += 1
            changed = 1
            continue
        seen_urls.add(ukey)
        compact[str(key)] = rec
    store["entries"] = compact
    store["version"] = 1
    store["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Enforce byte cap (default 25MB).
    while True:
        blob = json.dumps(store, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(blob) <= VECTOR_CACHE_MAX_BYTES:
            break
        if not compact:
            break
        # Drop oldest entry.
        oldest_key = min(compact.keys(), key=lambda k: str(compact.get(k, {}).get("updatedAt", "")))
        compact.pop(oldest_key, None)
        store["entries"] = compact
        dropped += 1
        changed = 1
    return {"changed": changed, "dropped": dropped, "kept": len(store.get("entries", {}))}

def _vector_cache_save_store(store: Dict[str, Any]) -> Dict[str, int]:
    stats = _vector_cache_trim_store(store)
    try:
        p = _vector_cache_store_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        stats["bytes"] = p.stat().st_size if p.exists() else 0
    except Exception:
        stats["bytes"] = 0
    return stats

def _hash_vector(text: str, dim: int = VECTOR_CACHE_DIM) -> List[float]:
    vec = [0.0] * max(32, dim)
    toks = _tokenize(text)
    if not toks:
        return vec
    for tok in toks:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % len(vec)
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec

def _dot(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return float(sum(a[i] * b[i] for i in range(n)))

def _store_vector_cache(url: str, lines: List[str]) -> None:
    if not VECTOR_CACHE_ENABLED:
        return
    try:
        normalized = _normalize_cache_url(url)
        if not normalized:
            return
        key = _url_cache_key(normalized)
        store = _vector_cache_load_store()
        store.setdefault("entries", {})
        store["entries"][key] = _vector_cache_entry_from_raw({
            "url": normalized,
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _vector_cache_save_store(store)
    except Exception:
        pass

def _load_vector_cache(url: str) -> Dict[str, Any]:
    if not VECTOR_CACHE_ENABLED:
        return {}
    normalized = _normalize_cache_url(url)
    if not normalized:
        return {}
    try:
        store = _vector_cache_load_store()
        entries = store.get("entries", {})
        if not isinstance(entries, dict):
            return {}
        rec = entries.get(_url_cache_key(normalized), {})
        return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}

def _vector_cache_retrieve(url: str, query: str = "", k: int = VECTOR_TOP_K) -> List[str]:
    # URL-only cache: do not store/replay page text.
    _ = _load_vector_cache(url)
    return []

def _semantic_ready() -> bool:
    return bool(SEMANTIC_MATCH_ENABLED and TfidfVectorizer is not None and cosine_similarity is not None)

def _semantic_seed_corpus() -> List[str]:
    # Keep a small generic corpus so the scorer can still start before any training data exists.
    return [
        "question answer interview technical explanation example impact tradeoff",
        "candidate explains concept implementation debugging testing and optimization",
        "question expected answer candidate answer semantic relevance correctness",
        "experience project challenge solution result learning outcome communication",
        "design scalability reliability maintainability performance and security",
    ]

def _collect_semantic_training_texts() -> List[str]:
    texts: List[str] = []
    store = TRAINING_STORE_MAP if isinstance(TRAINING_STORE_MAP, dict) and TRAINING_STORE_MAP else _load_training_store()
    domains = store.get("domains", {}) if isinstance(store, dict) else {}
    if isinstance(domains, dict):
        for role_key, node in domains.items():
            if not isinstance(node, dict):
                continue
            texts.append(str(role_key or "").strip())
            for q in node.get("questions", []) if isinstance(node.get("questions"), list) else []:
                qt = str(q or "").strip()
                if qt:
                    texts.append(qt)
            qa_bank = node.get("qaBank", [])
            if isinstance(qa_bank, list):
                for item in qa_bank:
                    if not isinstance(item, dict):
                        continue
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    if q:
                        texts.append(q)
                    if a:
                        texts.append(a)
                    if q and a:
                        texts.append(f"Question: {q}\nAnswer: {a}")
    source_map = RAG_SOURCE_MAP if isinstance(RAG_SOURCE_MAP, dict) and RAG_SOURCE_MAP else _load_source_map()
    if isinstance(source_map, dict):
        for role_key, cfg in source_map.items():
            if not isinstance(cfg, dict):
                continue
            texts.append(str(role_key or "").strip())
            keywords = [str(x).strip() for x in cfg.get("keywords", []) if str(x).strip()]
            if keywords:
                texts.append(" ".join(keywords))
    unique: List[str] = []
    seen: Set[str] = set()
    for raw in texts + _semantic_seed_corpus():
        norm = _normalize(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(str(raw).strip())
    return unique

def _build_semantic_scorer() -> Any:
    if not _semantic_ready():
        return None
    corpus = _collect_semantic_training_texts()
    if len(corpus) < 3:
        corpus = _semantic_seed_corpus()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    vectorizer.fit(corpus)
    pack = {
        "vectorizer": vectorizer,
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpusSize": len(corpus),
        "modelName": SEMANTIC_MODEL_NAME,
    }
    try:
        with SEMANTIC_SCORER_FILE.open("wb") as f:
            pickle.dump(pack, f)
    except Exception:
        logger.warning("Unable to persist semantic scorer: %s", SEMANTIC_SCORER_FILE)
    return pack

def _semantic_question_answer_text(question: str, answer: str) -> str:
    return f"Question: {str(question or '').strip()}\nAnswer: {str(answer or '').strip()}"

def _semantic_context_text(question: str, context: str) -> str:
    return f"Question: {str(question or '').strip()}\nContext: {str(context or '').strip()}"

def _get_semantic_model():
    global SEMANTIC_MODEL_HANDLE
    if not _semantic_ready():
        return None
    if SEMANTIC_MODEL_HANDLE is None:
        try:
            if SEMANTIC_SCORER_FILE.exists():
                with SEMANTIC_SCORER_FILE.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict) and loaded.get("vectorizer") is not None:
                    SEMANTIC_MODEL_HANDLE = loaded
            if SEMANTIC_MODEL_HANDLE is None:
                SEMANTIC_MODEL_HANDLE = _build_semantic_scorer()
        except Exception:
            SEMANTIC_MODEL_HANDLE = _build_semantic_scorer()
    return SEMANTIC_MODEL_HANDLE

def _semantic_embedding(text: str):
    if not _semantic_ready():
        return None
    t = _normalize(text)
    if not t:
        return None
    if t in SEMANTIC_EMBED_CACHE:
        return SEMANTIC_EMBED_CACHE[t]
    model = _get_semantic_model()
    vectorizer = model.get("vectorizer") if isinstance(model, dict) else None
    if vectorizer is None:
        return None
    emb = vectorizer.transform([t])
    SEMANTIC_EMBED_CACHE[t] = emb
    if len(SEMANTIC_EMBED_CACHE) > 1200:
        SEMANTIC_EMBED_CACHE.clear()
    return emb

def _semantic_similarity(a: str, b: str) -> float:
    try:
        ea = _semantic_embedding(a)
        eb = _semantic_embedding(b)
        if ea is None or eb is None:
            return 0.0
        return float(max(0.0, min(1.0, cosine_similarity(ea, eb)[0][0])))
    except Exception:
        return 0.0

def _best_semantic_similarity(text: str, candidates: List[str], cap: int = 20) -> float:
    if not candidates:
        return 0.0
    if not _semantic_ready():
        return 0.0
    ranked = sorted(candidates, key=lambda c: _overlap_ratio(str(c), _keywords(text, top_n=12)), reverse=True)
    best = 0.0
    for c in ranked[: max(1, cap)]:
        sim = _semantic_similarity(text, str(c))
        if sim > best:
            best = sim
    return best

def _is_nonsense(text: str) -> bool:
    words = _tokenize(_normalize(text))
    if len(words) < 3: return True
    if any(w in NONSENSE_TOKENS for w in words): return True
    if len(set(words)) / max(1, len(words)) < 0.35 and len(words) >= 6: return True
    if re.search(r"(.)\1{5,}", _normalize(text)): return True
    return False

def _flatten(v: Any, depth: int = 0, max_depth: int = 4) -> List[str]:
    if depth > max_depth or v is None: return []
    if isinstance(v, str):
        t = re.sub(r"\s+", " ", v).strip()
        return [t] if t else []
    if isinstance(v, dict):
        out: List[str] = []
        for k, item in v.items():
            if str(k).lower() in {"id","candidateid","createdat","updatedat"}: continue
            out.extend(_flatten(item, depth+1, max_depth))
        return out
    if isinstance(v, list):
        out = []
        for item in v: out.extend(_flatten(item, depth+1, max_depth))
        return out
    if isinstance(v, (int, float, bool)): return [str(v)]
    return []

def _is_private_line(line: str) -> bool:
    t = str(line or "").strip()
    if not t: return True
    lo = t.lower()
    if "@" in lo and "." in lo: return True
    if re.search(r"(?:\+?\d[\s\-()]*){8,}", t): return True
    if re.search(r"\b\d{6}\b", t): return True
    if lo.count(",") >= 2 and any(k in lo for k in PERSONAL_LINE_KEYWORDS): return True
    if any(lo.startswith(k + ":") for k in PERSONAL_LINE_KEYWORDS): return True
    if any(tok in lo for tok in {"http://","https://","www."}): return True
    if sum(1 for k in PERSONAL_LINE_KEYWORDS if k in lo) >= 2: return True
    return False

def _is_noisy_cache_line(line: str) -> bool:
    t = re.sub(r"\s+", " ", str(line or "")).strip()
    if len(t) < 24:
        return True
    if len(t) > 420:
        return True
    lo = t.lower()
    if _is_private_line(t):
        return True
    if any(p in lo for p in [
        "transcribed by", "subtitles by", "caption by", "cookies", "accept all",
        "sign in", "log in", "copyright", "all rights reserved",
    ]):
        return True
    if re.search(r"(.)\1{7,}", lo):
        return True
    alpha = sum(1 for ch in t if ch.isalpha())
    if alpha < max(8, int(len(t) * 0.35)):
        return True
    return False

def _sanitize_cache_lines(lines: List[Any], max_items: int = 120) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in lines:
        t = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not t:
            continue
        if _is_noisy_cache_line(t):
            continue
        key = _normalize(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max(8, min(max_items, 240)):
            break
    return out

def _cleanup_vector_cache_files() -> Dict[str, int]:
    if not VECTOR_CACHE_ENABLED:
        return {"files": 0, "changed": 0, "kept": 0, "dropped": 0, "bytes": 0}
    store = _vector_cache_load_store()
    migrated = 0
    dropped = 0
    changed = 0
    # Migrate legacy per-URL JSON files into single-file store.
    for legacy in VECTOR_CACHE_DIR.glob("*.json"):
        try:
            raw = json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        rec = _vector_cache_entry_from_raw(raw)
        if rec:
            key = _url_cache_key(rec["url"])
            existing = store.setdefault("entries", {}).get(key, {})
            existing_updated = str(existing.get("updatedAt", ""))
            if not isinstance(existing, dict) or rec.get("updatedAt", "") >= existing_updated:
                store["entries"][key] = rec
                migrated += 1
                changed = 1
        try:
            legacy.unlink()
            dropped += 1
        except Exception:
            pass
    save_stats = _vector_cache_save_store(store)
    changed += _safe_int(save_stats.get("changed"), 0)
    kept = _safe_int(save_stats.get("kept"), 0)
    bytes_size = _safe_int(save_stats.get("bytes"), 0)
    dropped += _safe_int(save_stats.get("dropped"), 0)
    return {
        "files": kept,
        "changed": changed,
        "kept": kept,
        "dropped": dropped,
        "migrated": migrated,
        "bytes": bytes_size,
    }

def _normalize_question(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    text = re.sub(r"^\d+[\).:\-\s]+", "", text)
    text = re.sub(r"^(q\.?|question)\s*\d+[\).:\-\s]+", "", text, flags=re.IGNORECASE)
    if not text: return ""
    if text.endswith("."): text = text[:-1].strip()
    if not text.endswith("?"):
        starters = ("what","why","how","when","where","which","who","explain","describe","difference")
        if text.lower().startswith(starters): text += "?"
        else: return ""
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

def _is_valid_answer_text(answer: str) -> bool:
    text = _limit_words(answer, 220)
    if not text:
        return False
    if _is_nonsense(text):
        return False
    words = re.findall(r"\S+", text)
    if len(words) < 14:
        return False
    if len(_split_sentences(text)) < 2:
        return False
    return True

def _is_valid_qa_pair(question: str, answer: str) -> bool:
    q = _normalize_question(question)
    if not q:
        return False
    if len(q) < 18 or len(q) > 220:
        return False
    return _is_valid_answer_text(answer)

def _is_intro_question(text: str) -> bool:
    q = _normalize(str(text or ""))
    return (
        q.startswith("introduce yourself")
        or "introduce yourself" in q
        or "tell me about yourself" in q
        or "briefly introduce yourself" in q
    )

def _split_numbered_block(text: str) -> List[str]:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw: return []
    matches = list(re.finditer(r"(?:^|\s)(\d{1,3})\.\s+", raw))
    if len(matches) < 2: return [raw]
    parts: List[str] = []
    for idx, m in enumerate(matches):
        end = matches[idx+1].start() if idx+1 < len(matches) else len(raw)
        chunk = raw[m.end():end].strip(" .;:-")
        if chunk: parts.append(chunk)
    return parts or [raw]

def _effective_language(job_title: str, job_desc: str, requested: str) -> str:
    req = _normalize(requested)
    hay = " ".join([_normalize(job_title), _normalize(job_desc)])
    if "java developer" in hay and req == "javascript": return "java"
    if " java " in f" {hay} " and "javascript" not in hay and req == "javascript": return "java"
    if "python" in hay and req == "javascript": return "python"
    if "flutter" in hay and req == "javascript": return "flutter"
    if "react native" in hay and req == "javascript": return "react native"
    return requested or "technology"

def _experience_years_from_payload(payload: Any) -> float:
    def _num(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return -1.0
    cp = payload.candidateProfile if isinstance(getattr(payload, "candidateProfile", {}), dict) else {}
    resume = payload.resume if isinstance(getattr(payload, "resume", {}), dict) else {}
    direct = [
        _num(cp.get("experienceYears")),
        _num(cp.get("yearsOfExperience")),
        _num(cp.get("totalExperience")),
        _num(cp.get("experience")),
        _num(resume.get("experienceYears")),
        _num(resume.get("yearsOfExperience")),
        _num(resume.get("totalExperience")),
    ]
    for x in direct:
        if x >= 0:
            return x
    # Parse labels like fresher / 1-3 / 5+.
    txt = " ".join([
        str(cp.get("experience", "")),
        str(cp.get("experienceLevel", "")),
        str(resume.get("experience", "")),
        str(resume.get("experienceLevel", "")),
    ]).lower()
    if "fresher" in txt:
        return 0.0
    rng = re.search(r"(\d+)\s*[-to]+\s*(\d+)", txt)
    if rng:
        a = _safe_float(rng.group(1), 0.0)
        b = _safe_float(rng.group(2), a)
        return max(0.0, (a + b) / 2.0)
    plus = re.search(r"(\d+)\s*\+", txt)
    if plus:
        return max(0.0, _safe_float(plus.group(1), 0.0))
    return 0.0

def _is_core_java(job_title: str, job_desc: str, language: str, custom_prompts: Optional[List[str]] = None) -> bool:
    hay = " ".join([_normalize(job_title), _normalize(job_desc), _normalize(language),
                    " ".join(_normalize(p) for p in (custom_prompts or []))])
    has_java = "java" in hay
    has_core = "core java" in hay or "core-java" in hay
    excludes = {"javascript","spring boot","springboot","react","node"}
    return has_java and (has_core or not any(e in hay for e in excludes))

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    end = -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return {}
    candidate = raw[start:end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def _ensure_llm_ready(task: str = "general", prompt: str = "", request_payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
    if not ENABLE_LLM or not requests:
        return None
    route = resolve_model_profile(task, prompt=prompt, payload=request_payload)
    if route.get("provider") == "ollama" and route.get("baseUrl"):
        availability = ensure_ollama_model_available(route.get("model", ""))
        if availability.get("ok"):
            if availability.get("pulled"):
                logger.info("Ollama model pulled on demand model=%s", route.get("model"))
            return route
        logger.warning(
            "Ollama model unavailable model=%s reason=%s",
            route.get("model"),
            availability.get("reason", "unknown"),
        )
    if LLM_API_URL:
        return {
            "provider": "external-api",
            "baseUrl": LLM_API_URL,
            "complexity": route.get("complexity", "light"),
            "model": route.get("model", MODEL_NAME),
        }
    return None

def _llm_text(
    prompt: str,
    temperature: float = 0.2,
    max_new_tokens: int = 800,
    task: str = "general",
    request_payload: Optional[Dict[str, Any]] = None,
) -> str:
    route = _ensure_llm_ready(task=task, prompt=prompt, request_payload=request_payload)
    if route is None:
        return ""
    started = time.perf_counter()
    try:
        headers = {"Content-Type": "application/json"}
        endpoint = route.get("baseUrl", "")
        if route.get("provider") == "external-api" and LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        if route.get("provider") == "ollama":
            endpoint = endpoint.rstrip("/") + "/api/generate"
            payload = {
                "model": route.get("model", MODEL_NAME),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                },
            }
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=LLM_API_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            logger.warning("LLM API HTTP %s route=%s model=%s", resp.status_code, route.get("provider"), route.get("model"))
            return ""
        elapsed = round((time.perf_counter() - started) * 1000)
        logger.info("LLM text finished in %sms route=%s model=%s task=%s", elapsed, route.get("provider"), route.get("model"), task)
        data = resp.json() if resp.text.strip() else {}
        if isinstance(data, dict):
            text = str(data.get("response", "")).strip()
            if isinstance(data.get("choices"), list) and data.get("choices"):
                choice = data["choices"][0]
                text = str(choice.get("text", "")).strip() or text
                if not text and isinstance(choice.get("message"), dict):
                    text = str(choice["message"].get("content", "")).strip()
            for key in ("response", "text", "output", "content"):
                if not text and key in data:
                    text = str(data.get(key, "")).strip()
            append_foundation_example(task, prompt, request_payload, {"text": text}, route.get("model", MODEL_NAME), "llm-text")
            return text
    except Exception as ex:
        logger.warning("LLM text failed: %s", ex)
    return ""

def _llm_generate_json(prompt: str, task: str = "general", request_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = _llm_text(
        prompt,
        temperature=0.2,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        task=task,
        request_payload=request_payload,
    )
    if not raw:
        return {}
    parsed = _extract_first_json_object(raw)
    if parsed:
        return parsed
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _normalize_question_mix(questions: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    target_count = max(1, target_count)
    target_core = max(1, int(round(target_count * 0.50)))
    target_logical = max(1, int(round(target_count * 0.30)))
    target_normal = max(1, target_count - target_core - target_logical)
    while target_core + target_logical + target_normal > target_count:
        target_normal = max(0, target_normal - 1)
    while target_core + target_logical + target_normal < target_count:
        target_normal += 1

    buckets = {"core": [], "logical": [], "normal": []}
    seen: Set[str] = set()
    for item in questions:
        q = _normalize_question(item.get("question", ""))
        if not q:
            continue
        key = _normalize(q)
        if key in seen:
            continue
        seen.add(key)
        category = _normalize(item.get("category", "normal"))
        if "core" in category:
            bucket = "core"
        elif "logic" in category:
            bucket = "logical"
        else:
            bucket = "normal"
        buckets[bucket].append({"question": q, "category": bucket})

    final: List[Dict[str, Any]] = []
    def take(bucket: str, n: int) -> None:
        for q in buckets[bucket][:max(0, n)]:
            if len(final) >= target_count:
                break
            final.append(q)

    take("core", target_core)
    take("logical", target_logical)
    take("normal", target_normal)

    # Backfill from any bucket if one category is short.
    merged = buckets["core"] + buckets["logical"] + buckets["normal"]
    for q in merged:
        if len(final) >= target_count:
            break
        if any(_normalize(x["question"]) == _normalize(q["question"]) for x in final):
            continue
        final.append(q)

    return final[:target_count]

def _llm_generate_questions(payload: InterviewQuestionsRequest, snippets: List[str]) -> List[Dict[str, Any]]:
    """Ask the LLM to generate brand-new interview questions from scratch."""
    if not ENABLE_LLM:
        return []
    context_lines = [_limit_text_words(s, LLM_LINE_WORD_LIMIT) for s in snippets[:LLM_CONTEXT_LINES]]
    prompt = (
        "You are a senior technical interviewer. Generate EXACTLY "
        f"{INTERVIEW_QUESTION_COUNT} unique interview questions. Return STRICT JSON only.\n"
        "Distribution: 50% core technical, 30% logical/problem-solving, 20% conceptual.\n"
        "Rules:\n"
        "- Questions must be specific to the job title, stack, and context below.\n"
        "- No generic questions like 'tell me about yourself'.\n"
        "- Each question must end with a '?'.\n"
        "- Questions must be different from each other.\n"
        "Output schema:\n"
        "{\"questions\":[{\"question\":\"...\",\"category\":\"core|logical|conceptual\"}]}\n\n"
        f"Job title: {payload.jobTitle or 'Software Engineer'}\n"
        f"Department: {payload.department or 'Engineering'}\n"
        f"Language/stack: {payload.language or 'javascript'}\n"
        f"Domain: {payload.domainLabel or payload.domain or 'software'}\n"
        f"Job description: {_limit_text_words(payload.jobDescription, 60)}\n\n"
        f"Context:\n{json.dumps(context_lines, ensure_ascii=False)}"
    )
    parsed = _llm_generate_json(
        prompt,
        task="interview_question_generation",
        request_payload={
            "jobTitle": payload.jobTitle,
            "department": payload.department,
            "language": payload.language,
            "domain": payload.domainLabel or payload.domain,
        },
    )
    questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
    if not isinstance(questions, list):
        return []
    return _normalize_question_mix([q for q in questions if isinstance(q, dict)], INTERVIEW_QUESTION_COUNT)

def _llm_select_questions(pool: List[str], payload: InterviewQuestionsRequest, rag_focus: List[str]) -> List[Dict[str, Any]]:
    """
    Ask LLM to pick the best questions from an existing pool.
    Safe behavior:
    - Returns [] when LLM is disabled/unavailable.
    - Never raises on JSON/parse failures.
    """
    if not ENABLE_LLM or _ensure_llm_ready() is None:
        return []
    cleaned_pool = [_normalize_question(q) for q in pool if _normalize_question(q)]
    if not cleaned_pool:
        return []
    prompt = (
        "You are an interview designer. Return STRICT JSON only.\n"
        f"Need total {INTERVIEW_QUESTION_COUNT} interview questions.\n"
        "Choose from the provided candidate pool and avoid duplicates.\n"
        "Prefer role-relevant, practical, and concept-focused questions.\n"
        "Output format:\n"
        "{\"questions\":[{\"question\":\"...\",\"category\":\"core|logical|normal\"}]}\n\n"
        f"Role: {payload.jobTitle or payload.domainLabel or payload.domain or payload.department or 'general'}\n"
        f"Language: {payload.language}\n"
        f"Difficulty: {payload.difficulty}\n"
        f"RAG focus:\n{chr(10).join(rag_focus[: max(6, min(len(rag_focus), 24))])}\n\n"
        f"Candidate pool:\n{chr(10).join(f'- {q}' for q in cleaned_pool[:120])}"
    )
    out = _llm_text(
        prompt,
        temperature=0.2,
        max_new_tokens=800,
        task="interview_question_selection",
        request_payload={
            "jobTitle": payload.jobTitle,
            "language": payload.language,
            "difficulty": payload.difficulty,
        },
    )
    if not out:
        return []
    parsed = _extract_json(out)
    questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
    if not isinstance(questions, list):
        return []
    return _normalize_question_mix([q for q in questions if isinstance(q, dict)], INTERVIEW_QUESTION_COUNT)
    if not candidate_pool:
        return []
    sampled_context = random.sample(snippets, min(len(snippets), LLM_CONTEXT_LINES)) if snippets else []
    sampled_pool = random.sample(candidate_pool, min(len(candidate_pool), LLM_POOL_LIMIT))
    top_context = [_limit_text_words(line, LLM_LINE_WORD_LIMIT) for line in sampled_context]
    pool = [_limit_text_words(line, LLM_LINE_WORD_LIMIT) for line in sampled_pool]
    prompt = (
        "You are an interview designer. Return STRICT JSON only.\n"
        f"Need total {INTERVIEW_QUESTION_COUNT} interview questions.\n"
        "Distribution must be: 50% core technical, 30% logical/problem-solving, 20% normal/fundamental.\n"
        "Rules:\n"
        "- Use only provided context and candidate pool.\n"
        "- No hallucinated technologies.\n"
        "- Questions must be concise and interview-ready.\n"
        "- Prefer practical, role-specific wording.\n"
        "Output schema:\n"
        "{\"questions\":[{\"question\":\"...\",\"category\":\"core|logical|normal\"}]}\n\n"
        f"Job title: {payload.jobTitle}\n"
        f"Department: {payload.department}\n"
        f"Language/stack: {payload.language}\n"
        f"Domain: {payload.domainLabel or payload.domain}\n\n"
        f"Context lines:\n{json.dumps(top_context, ensure_ascii=False)}\n\n"
        f"Candidate pool:\n{json.dumps(pool, ensure_ascii=False)}"
    )
    parsed = _llm_generate_json(
        prompt,
        task="interview_answer_evaluation",
        request_payload={"question": question, "answer": answer},
    )
    questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
    if not isinstance(questions, list):
        return []
    return _normalize_question_mix([q for q in questions if isinstance(q, dict)], INTERVIEW_QUESTION_COUNT)

def _llm_score_answer(question: str, answer: str, context_lines: List[str]) -> Dict[str, Any]:
    prompt = (
        "You are an interview evaluator. Return STRICT JSON only.\n"
        "Score the answer against the question and context.\n"
        "No hallucinations: if unsupported, score low.\n"
        "Output schema:\n"
        "{\"score\":0-100,\"matchPercent\":0-100,\"isCorrect\":true|false,"
        "\"strengths\":[\"...\"],\"gaps\":[\"...\"],\"explanation\":\"...\"}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Context: {json.dumps(context_lines[:40], ensure_ascii=False)}"
    )
    parsed = _llm_generate_json(prompt)
    if not parsed:
        return {}
    score = _clamp(_safe_int(parsed.get("score"), -1))
    match_percent = _clamp(_safe_int(parsed.get("matchPercent"), score))
    is_correct = bool(parsed.get("isCorrect", score >= 60))
    strengths = [str(x).strip() for x in parsed.get("strengths", []) if str(x).strip()] if isinstance(parsed.get("strengths"), list) else []
    gaps = [str(x).strip() for x in parsed.get("gaps", []) if str(x).strip()] if isinstance(parsed.get("gaps"), list) else []
    explanation = str(parsed.get("explanation", "")).strip()
    result = {
        "score": score,
        "matchPercent": match_percent,
        "isCorrect": is_correct,
        "strengths": strengths,
        "gaps": gaps,
        "explanation": explanation,
    }
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAG helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_sources_file() -> None:
    if not RAG_SOURCES_FILE.exists():
        RAG_SOURCES_FILE.write_text(json.dumps({"default":{"sources":[]}}, indent=2), encoding="utf-8")

def _load_source_map() -> Dict[str, Dict[str, Any]]:
    _ensure_sources_file()
    try:
        raw = json.loads(RAG_SOURCES_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"default":{"sources":[]}}
    if not isinstance(raw, dict): return {"default":{"sources":[]}}
    normalized: Dict[str, Dict[str, Any]] = {}
    for role, cfg in raw.items():
        if not isinstance(cfg, dict): continue
        normalized[str(role).strip().lower()] = {
            "sources":  list(dict.fromkeys(s.strip() for s in cfg.get("sources",[]) if str(s).strip())),
            "keywords": list(dict.fromkeys(w.strip().lower() for w in cfg.get("keywords",[]) if str(w).strip())),
        }
    normalized.setdefault("default", {"sources":[],"keywords":[]})
    return normalized

def _default_market_sources() -> Dict[str, Dict[str, Any]]:
    return {
        "java": {
            "keywords": ["java", "jvm", "collections", "multithreading", "oop"],
            "sources": [
                "https://www.interviewbit.com/java-interview-questions/",
                "https://www.geeksforgeeks.org/java-interview-questions/",
                "https://www.javatpoint.com/corejava-interview-questions",
                "https://www.simplilearn.com/tutorials/java-tutorial/java-interview-questions",
                "https://www.turing.com/interview-questions/java",
                "https://www.edureka.co/blog/interview-questions/java-interview-questions/",
                "https://www.digitalocean.com/community/tutorials/java-interview-questions-and-answers",
                "https://www.baeldung.com/java-interview-questions",
            ],
        },
        "sales": {
            "keywords": ["sales", "crm", "pipeline", "negotiation", "closing"],
            "sources": [
                "https://blog.hubspot.com/sales/interview-questions-for-sales-position",
                "https://www.indeed.com/career-advice/interviewing/sales-interview-questions",
                "https://www.salesforce.com/blog/sales-interview-questions/",
                "https://www.zendesk.com/blog/sales-interview-questions/",
                "https://www.glassdoor.com/blog/guide/sales-interview-questions/",
                "https://www.coursera.org/articles/sales-interview-questions",
                "https://www.betterteam.com/sales-interview-questions",
                "https://www.gong.io/blog/sales-interview-questions/",
            ],
        },
        "business_development": {
            "keywords": ["business development", "bdm", "bde", "lead generation", "partnership", "revenue growth", "b2b sales"],
            "sources": [
                "https://www.indeed.com/career-advice/interviewing/business-development-interview-questions",
                "https://www.glassdoor.com/blog/guide/business-development-interview-questions/",
                "https://www.turing.com/interview-questions/business-development",
                "https://www.coursera.org/articles/business-development-manager-interview-questions",
                "https://www.betterteam.com/business-development-manager-interview-questions",
                "https://resources.workable.com/business-development-manager-interview-questions",
            ],
        },
        "voice_process": {
            "keywords": ["voice process", "call center", "customer support", "inbound calls", "outbound calls", "bpo voice"],
            "sources": [
                "https://www.indeed.com/career-advice/interviewing/call-center-interview-questions",
                "https://www.naukri.com/blog/bpo-interview-questions-and-answers/",
                "https://www.simplilearn.com/customer-service-interview-questions-and-answers-article",
                "https://www.glassdoor.com/blog/guide/customer-service-interview-questions/",
                "https://www.foundit.in/career-advice/customer-service-interview-questions/",
            ],
        },
        "chat_process": {
            "keywords": ["chat process", "chat support", "email support", "customer service", "non voice", "bpo chat"],
            "sources": [
                "https://www.indeed.com/career-advice/interviewing/customer-service-interview-questions",
                "https://www.interviewbit.com/customer-service-interview-questions/",
                "https://www.zendesk.com/blog/customer-service-interview-questions/",
                "https://www.glassdoor.com/blog/guide/customer-support-interview-questions/",
                "https://www.freshworks.com/customer-service/help-desk/customer-service-interview-questions-blog/",
            ],
        },
        "flutter": {
            "keywords": ["flutter", "dart", "widget", "state management", "mobile"],
            "sources": [
                "https://www.interviewbit.com/flutter-interview-questions/",
                "https://www.geeksforgeeks.org/flutter-interview-questions-and-answers/",
                "https://www.turing.com/interview-questions/flutter",
                "https://www.simplilearn.com/flutter-interview-questions-article",
                "https://www.javatpoint.com/flutter-interview-questions",
                "https://www.edureka.co/blog/interview-questions/flutter-interview-questions/",
                "https://www.scholarhat.com/tutorial/flutter/flutter-interview-questions-and-answers",
                "https://www.fullstack.cafe/blog/flutter-interview-questions",
            ],
        },
        "react": {
            "keywords": ["react", "hooks", "state", "component", "frontend"],
            "sources": [
                "https://www.interviewbit.com/react-interview-questions/",
                "https://www.geeksforgeeks.org/reactjs-interview-questions/",
                "https://react.dev/learn",
                "https://www.simplilearn.com/tutorials/reactjs-tutorial/reactjs-interview-questions",
                "https://www.javatpoint.com/react-interview-questions",
                "https://www.edureka.co/blog/interview-questions/react-interview-questions/",
                "https://www.turing.com/interview-questions/react-js",
                "https://www.fullstack.cafe/blog/react-interview-questions",
            ],
        },
    }

def _ensure_training_store_file() -> None:
    if not TRAINING_STORE_FILE.exists():
        TRAINING_STORE_FILE.write_text(json.dumps({"trainedAt": "", "domains": {}}, indent=2), encoding="utf-8")

def _load_training_store() -> Dict[str, Any]:
    _ensure_training_store_file()
    try:
        raw = json.loads(TRAINING_STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"trainedAt": "", "domains": {}}
    if not isinstance(raw, dict):
        return {"trainedAt": "", "domains": {}}
    domains = raw.get("domains", {})
    if not isinstance(domains, dict):
        domains = {}
    raw["domains"] = domains
    raw.setdefault("trainedAt", "")
    return raw

def _save_training_store(store: Dict[str, Any]) -> None:
    global TRAINING_STORE_MAP, SEMANTIC_MODEL_HANDLE, SEMANTIC_EMBED_CACHE
    TRAINING_STORE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")
    TRAINING_STORE_MAP = store
    # Rebuild the TF-IDF scorer lazily after training data changes so later answer checks
    # use the freshest question/answer vocabulary without adding work to the save path.
    SEMANTIC_MODEL_HANDLE = None
    SEMANTIC_EMBED_CACHE.clear()
    try:
        if SEMANTIC_SCORER_FILE.exists():
            SEMANTIC_SCORER_FILE.unlink()
    except Exception:
        logger.warning("Unable to refresh semantic scorer cache file: %s", SEMANTIC_SCORER_FILE)

def _sanitize_training_store(store: Dict[str, Any]) -> Dict[str, int]:
    stats = {"domains": 0, "questionsDropped": 0, "qaDropped": 0, "changed": 0}
    if not isinstance(store, dict):
        return stats
    domains = store.get("domains", {})
    if not isinstance(domains, dict):
        store["domains"] = {}
        stats["changed"] = 1
        return stats
    for role_key, node in list(domains.items()):
        if not isinstance(node, dict):
            continue
        stats["domains"] += 1
        changed_local = False
        questions = node.get("questions", [])
        cleaned_questions: List[str] = []
        seen_q: Set[str] = set()
        if isinstance(questions, list):
            for q in questions:
                nq = _normalize_question(q)
                if not nq:
                    stats["questionsDropped"] += 1
                    changed_local = True
                    continue
                k = _normalize(nq)
                if k in seen_q:
                    stats["questionsDropped"] += 1
                    changed_local = True
                    continue
                seen_q.add(k)
                cleaned_questions.append(nq)
        if cleaned_questions != questions:
            node["questions"] = cleaned_questions
            changed_local = True

        qa_bank = node.get("qaBank", [])
        cleaned_qa: List[Dict[str, str]] = []
        seen_qa: Set[str] = set()
        if isinstance(qa_bank, list):
            for item in qa_bank:
                if not isinstance(item, dict):
                    stats["qaDropped"] += 1
                    changed_local = True
                    continue
                q = _normalize_question(item.get("question", ""))
                a = _limit_words(str(item.get("answer", "")).strip(), 180)
                if not _is_valid_qa_pair(q, a):
                    stats["qaDropped"] += 1
                    changed_local = True
                    continue
                k = _normalize(q)
                if k in seen_qa:
                    stats["qaDropped"] += 1
                    changed_local = True
                    continue
                seen_qa.add(k)
                cleaned_qa.append({"question": q, "answer": a})
        if cleaned_qa != qa_bank:
            node["qaBank"] = cleaned_qa
            changed_local = True
        if changed_local:
            domains[role_key] = node
            stats["changed"] += 1
    store["domains"] = domains
    return stats

def _ensure_candidate_state_file() -> None:
    if not CANDIDATE_STATE_FILE.exists():
        CANDIDATE_STATE_FILE.write_text(json.dumps({"byRole": {}}, indent=2), encoding="utf-8")

def _load_candidate_state() -> Dict[str, Any]:
    _ensure_candidate_state_file()
    try:
        raw = json.loads(CANDIDATE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"byRole": {}}
    if not isinstance(raw, dict):
        return {"byRole": {}}
    by_role = raw.get("byRole", {})
    if not isinstance(by_role, dict):
        by_role = {}
    raw["byRole"] = by_role
    return raw

def _save_candidate_state(store: Dict[str, Any]) -> None:
    CANDIDATE_STATE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def _ensure_reinforcement_state_file() -> None:
    if not REINFORCEMENT_STATE_FILE.exists():
        REINFORCEMENT_STATE_FILE.write_text(json.dumps({"roles": {}}, indent=2), encoding="utf-8")

def _load_reinforcement_state() -> Dict[str, Any]:
    _ensure_reinforcement_state_file()
    try:
        raw = json.loads(REINFORCEMENT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"roles": {}}
    if not isinstance(raw, dict):
        return {"roles": {}}
    roles = raw.get("roles", {})
    if not isinstance(roles, dict):
        roles = {}
    raw["roles"] = roles
    return raw

def _save_reinforcement_state(store: Dict[str, Any]) -> None:
    REINFORCEMENT_STATE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def _ensure_source_quality_file() -> None:
    if not SOURCE_QUALITY_FILE.exists():
        SOURCE_QUALITY_FILE.write_text(json.dumps({"urls": {}}, indent=2), encoding="utf-8")

def _load_source_quality() -> Dict[str, Any]:
    _ensure_source_quality_file()
    try:
        raw = json.loads(SOURCE_QUALITY_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"urls": {}}
    if not isinstance(raw, dict):
        return {"urls": {}}
    urls = raw.get("urls", {})
    if not isinstance(urls, dict):
        urls = {}
    raw["urls"] = urls
    return raw

def _save_source_quality(store: Dict[str, Any]) -> None:
    SOURCE_QUALITY_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def _ensure_learning_log_file() -> None:
    if not LEARNING_LOG_FILE.exists():
        LEARNING_LOG_FILE.write_text(json.dumps({"items": []}, indent=2), encoding="utf-8")

def _load_learning_log() -> Dict[str, Any]:
    _ensure_learning_log_file()
    try:
        raw = json.loads(LEARNING_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"items": []}
    if not isinstance(raw, dict):
        return {"items": []}
    items = raw.get("items", [])
    if not isinstance(items, list):
        items = []
    raw["items"] = items
    return raw

def _save_learning_log(store: Dict[str, Any]) -> None:
    LEARNING_LOG_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def _ensure_learning_policy_file() -> None:
    if not LEARNING_POLICY_FILE.exists():
        LEARNING_POLICY_FILE.write_text(json.dumps({"startedAtTs": int(time.time()), "urls": {}, "questions": {}, "daily": {}}, indent=2), encoding="utf-8")

def _load_learning_policy() -> Dict[str, Any]:
    _ensure_learning_policy_file()
    try:
        raw = json.loads(LEARNING_POLICY_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {"startedAtTs": int(time.time()), "urls": {}, "questions": {}, "daily": {}}
    if not isinstance(raw, dict):
        return {"startedAtTs": int(time.time()), "urls": {}, "questions": {}, "daily": {}}
    raw.setdefault("startedAtTs", int(time.time()))
    urls = raw.get("urls", {})
    questions = raw.get("questions", {})
    daily = raw.get("daily", {})
    raw["urls"] = urls if isinstance(urls, dict) else {}
    raw["questions"] = questions if isinstance(questions, dict) else {}
    raw["daily"] = daily if isinstance(daily, dict) else {}
    return raw

def _save_learning_policy(store: Dict[str, Any]) -> None:
    LEARNING_POLICY_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def _today_bucket() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def _learning_policy_compact(store: Dict[str, Any]) -> Dict[str, Any]:
    now_ts = int(time.time())
    url_keep_after = now_ts - (365 * 86400)
    q_keep_after = now_ts - (365 * 86400)
    daily_keep_after = now_ts - (45 * 86400)
    urls = store.get("urls", {})
    if isinstance(urls, dict):
        trimmed_urls = {}
        for k, node in urls.items():
            if not isinstance(node, dict):
                continue
            if _safe_int(node.get("lastSeenTs"), 0) < url_keep_after:
                continue
            trimmed_urls[str(k)] = node
        store["urls"] = trimmed_urls
    questions = store.get("questions", {})
    if isinstance(questions, dict):
        trimmed_q = {}
        for k, node in questions.items():
            if not isinstance(node, dict):
                continue
            if _safe_int(node.get("lastSeenTs"), 0) < q_keep_after:
                continue
            trimmed_q[str(k)] = node
        store["questions"] = trimmed_q
    daily = store.get("daily", {})
    if isinstance(daily, dict):
        trimmed_daily = {}
        for d, node in daily.items():
            try:
                ts = int(time.mktime(time.strptime(d, "%Y-%m-%d")))
            except Exception:
                continue
            if ts < daily_keep_after:
                continue
            if isinstance(node, dict):
                trimmed_daily[d] = node
        store["daily"] = trimmed_daily
    return store

def _learning_knowledge_score() -> float:
    store = _load_training_store()
    domains = store.get("domains", {}) if isinstance(store, dict) else {}
    total_q = 0
    for node in (domains.values() if isinstance(domains, dict) else []):
        if not isinstance(node, dict):
            continue
        total_q += len([q for q in node.get("questions", []) if str(q).strip()])
    q_score = min(55.0, (float(total_q) / 1200.0) * 55.0)
    rstate = _load_reinforcement_state()
    roles = rstate.get("roles", {}) if isinstance(rstate, dict) else {}
    reward_sum = 0.0
    reward_n = 0
    for role_node in (roles.values() if isinstance(roles, dict) else []):
        qstats = role_node.get("questions", {}) if isinstance(role_node, dict) else {}
        if not isinstance(qstats, dict):
            continue
        for stat in qstats.values():
            if not isinstance(stat, dict):
                continue
            reward_sum += _safe_float(stat.get("reward"), 0.0)
            reward_n += 1
    avg_reward = (reward_sum / max(1, reward_n))
    reward_score = max(0.0, min(25.0, (avg_reward + 1.0) * 12.5))
    sq = _load_source_quality()
    urls = sq.get("urls", {}) if isinstance(sq, dict) else {}
    pass_hits = 0
    fail_hits = 0
    for row in (urls.values() if isinstance(urls, dict) else []):
        if not isinstance(row, dict):
            continue
        pass_hits += max(0, _safe_int(row.get("pass"), 0))
        fail_hits += max(0, _safe_int(row.get("fail"), 0))
    total_hits = pass_hits + fail_hits
    quality_ratio = (pass_hits / total_hits) if total_hits > 0 else 0.0
    quality_score = quality_ratio * 20.0
    return max(0.0, min(100.0, round(q_score + reward_score + quality_score, 2)))

def _adaptive_learning_policy() -> Dict[str, Any]:
    global LEARNING_POLICY_MAP
    LEARNING_POLICY_MAP = _load_learning_policy()
    started_ts = _safe_int(LEARNING_POLICY_MAP.get("startedAtTs"), int(time.time()))
    age_days = max(0.0, (time.time() - started_ts) / 86400.0)
    knowledge = _learning_knowledge_score()
    fast_window_days = float(max(1, FAST_LEARN_WEEKS) * 7)
    in_fast_mode = age_days < fast_window_days and knowledge < 82.0
    mode = "fast" if in_fast_mode else "slow"
    base_gap = max(60, AUTO_TRAIN_MIN_GAP_SEC)
    if in_fast_mode:
        gap = max(120, int(base_gap * 0.33))
        discover_mul = 1.5
        fetch_mul = 1.4
        q_mul = 1.3
    else:
        # Slow down as knowledge rises; keep novelty focus.
        gap = max(base_gap, int(base_gap + (knowledge * 12)))
        discover_mul = 0.75 if knowledge < 92 else 0.55
        fetch_mul = 0.70 if knowledge < 92 else 0.50
        q_mul = 0.85 if knowledge < 92 else 0.70
    return {
        "mode": mode,
        "knowledgeScore": knowledge,
        "ageDays": round(age_days, 2),
        "minGapSec": gap,
        "discoverMultiplier": discover_mul,
        "fetchMultiplier": fetch_mul,
        "questionMultiplier": q_mul,
        "preferNovel": True,
        "repeatDays": NOVEL_URL_REPEAT_DAYS,
        "dailyNewUrlTarget": DAILY_NEW_URL_TARGET,
    }

def _mark_learning_url(url: str, domain: str, passed: bool, reason: str = "") -> None:
    global LEARNING_POLICY_MAP
    LEARNING_POLICY_MAP = _load_learning_policy()
    store = LEARNING_POLICY_MAP
    urls = store.setdefault("urls", {})
    day = _today_bucket()
    daily = store.setdefault("daily", {})
    dnode = daily.setdefault(day, {"newUrlHashes": [], "questionHashes": [], "domains": {}})
    if not isinstance(urls, dict):
        urls = {}
    u = str(url).strip()
    if not u:
        return
    h = hashlib.sha1(u.encode("utf-8")).hexdigest()
    node = urls.setdefault(h, {"url": u, "domain": domain, "firstSeenTs": int(time.time()), "lastSeenTs": 0, "pass": 0, "fail": 0, "hits": 0})
    is_new = _safe_int(node.get("hits"), 0) <= 0
    node["lastSeenTs"] = int(time.time())
    node["domain"] = str(domain or node.get("domain", "")).strip()
    node["hits"] = _safe_int(node.get("hits"), 0) + 1
    if passed:
        node["pass"] = _safe_int(node.get("pass"), 0) + 1
    else:
        node["fail"] = _safe_int(node.get("fail"), 0) + 1
    node["lastReason"] = str(reason or "")[:200]
    urls[h] = node
    if is_new and isinstance(dnode, dict):
        hashes = dnode.setdefault("newUrlHashes", [])
        if isinstance(hashes, list) and h not in hashes:
            hashes.append(h)
            dnode["newUrlHashes"] = hashes[-2000:]
    if isinstance(dnode, dict):
        dm = dnode.setdefault("domains", {})
        if isinstance(dm, dict):
            bucket = dm.setdefault(str(domain or "general"), {"pass": 0, "fail": 0})
            if passed:
                bucket["pass"] = _safe_int(bucket.get("pass"), 0) + 1
            else:
                bucket["fail"] = _safe_int(bucket.get("fail"), 0) + 1
            dm[str(domain or "general")] = bucket
            dnode["domains"] = dm
        daily[day] = dnode
    store["urls"] = urls
    store["daily"] = daily
    LEARNING_POLICY_MAP = _learning_policy_compact(store)
    _save_learning_policy(LEARNING_POLICY_MAP)

def _is_url_novel_for_learning(url: str, repeat_days: int) -> bool:
    store = _load_learning_policy()
    urls = store.get("urls", {}) if isinstance(store, dict) else {}
    if not isinstance(urls, dict):
        return True
    h = hashlib.sha1(str(url).strip().encode("utf-8")).hexdigest()
    node = urls.get(h, {})
    if not isinstance(node, dict):
        return True
    last_seen = _safe_int(node.get("lastSeenTs"), 0)
    if last_seen <= 0:
        return True
    return (time.time() - last_seen) >= max(1, repeat_days) * 86400

def _mark_learning_question(domain: str, question: str) -> None:
    global LEARNING_POLICY_MAP
    q = _normalize_question(question)
    if not q:
        return
    LEARNING_POLICY_MAP = _load_learning_policy()
    store = LEARNING_POLICY_MAP
    questions = store.setdefault("questions", {})
    day = _today_bucket()
    daily = store.setdefault("daily", {})
    dnode = daily.setdefault(day, {"newUrlHashes": [], "questionHashes": [], "domains": {}})
    h = hashlib.sha1(_normalize(q).encode("utf-8")).hexdigest()
    node = questions.setdefault(h, {"question": q, "domain": domain, "firstSeenTs": int(time.time()), "lastSeenTs": 0, "hits": 0})
    node["lastSeenTs"] = int(time.time())
    node["domain"] = str(domain or node.get("domain", "")).strip()
    node["hits"] = _safe_int(node.get("hits"), 0) + 1
    questions[h] = node
    if isinstance(dnode, dict):
        hashes = dnode.setdefault("questionHashes", [])
        if isinstance(hashes, list) and h not in hashes:
            hashes.append(h)
            dnode["questionHashes"] = hashes[-4000:]
        daily[day] = dnode
    store["questions"] = questions
    store["daily"] = daily
    LEARNING_POLICY_MAP = _learning_policy_compact(store)
    _save_learning_policy(LEARNING_POLICY_MAP)

def _prioritize_novel_questions(domain: str, questions: List[str], target: int) -> List[str]:
    store = _load_learning_policy()
    qmap = store.get("questions", {}) if isinstance(store, dict) else {}
    seen: Set[str] = set()
    novel: List[str] = []
    known: List[str] = []
    for q in questions:
        nq = _normalize_question(q)
        if not nq:
            continue
        k = _normalize(nq)
        if k in seen:
            continue
        seen.add(k)
        h = hashlib.sha1(k.encode("utf-8")).hexdigest()
        if isinstance(qmap, dict) and isinstance(qmap.get(h), dict):
            known.append(nq)
        else:
            novel.append(nq)
    ordered = [*novel, *known]
    selected = ordered[: max(1, target)]
    for q in selected:
        _mark_learning_question(domain, q)
    return selected

def _append_learning_log(event: str, details: Dict[str, Any]) -> None:
    global LEARNING_LOG_MAP
    LEARNING_LOG_MAP = _load_learning_log()
    items = LEARNING_LOG_MAP.setdefault("items", [])
    if not isinstance(items, list):
        items = []
    payload = {
        "ts": int(time.time()),
        "event": str(event),
        "details": details if isinstance(details, dict) else {},
    }
    items.append(payload)
    if len(items) > LEARNING_LOG_MAX_ITEMS:
        items = items[-LEARNING_LOG_MAX_ITEMS:]
    LEARNING_LOG_MAP["items"] = items
    _save_learning_log(LEARNING_LOG_MAP)

def _record_source_quality(url: str, domain: str, passed: bool, reason: str = "") -> None:
    global SOURCE_QUALITY_MAP
    SOURCE_QUALITY_MAP = _load_source_quality()
    urls = SOURCE_QUALITY_MAP.setdefault("urls", {})
    key = str(url).strip()
    if not key:
        return
    stat = urls.setdefault(key, {"pass": 0, "fail": 0, "domain": domain, "lastStatus": "", "lastReason": "", "lastTs": 0})
    if passed:
        stat["pass"] = max(0, _safe_int(stat.get("pass"), 0)) + 1
        stat["lastStatus"] = "pass"
    else:
        stat["fail"] = max(0, _safe_int(stat.get("fail"), 0)) + 1
        stat["lastStatus"] = "fail"
    stat["domain"] = str(domain or stat.get("domain", "")).strip()
    stat["lastReason"] = str(reason or "").strip()[:300]
    stat["lastTs"] = int(time.time())
    urls[key] = stat
    SOURCE_QUALITY_MAP["urls"] = urls
    _save_source_quality(SOURCE_QUALITY_MAP)

def _resource_limits() -> Dict[str, int]:
    profile = RESOURCE_PROFILE if RESOURCE_PROFILE in {"low", "medium", "high"} else "medium"
    if profile == "low":
        return {"discover": 24, "fetch": 24, "qtarget": 20, "topk": 12}
    if profile == "high":
        return {"discover": 240, "fetch": 120, "qtarget": 50, "topk": 36}
    return {"discover": 120, "fetch": 80, "qtarget": 40, "topk": 24}

def _is_url_allowed(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(str(url))
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False, "invalid_url"
    if not host:
        return False, "missing_host"
    blocked_hosts = {"duckduckgo.com", "search.brave.com", "bing.com"}
    if host in blocked_hosts:
        return False, "search_engine_page"
    blocked_parts = ("feedback", "login", "signup", "signin", "register", "account", "profile", "notification", "auth")
    if any(bp in path for bp in blocked_parts):
        return False, "blocked_path"
    if SOURCE_ALLOWLIST_DOMAINS:
        if not any(host == dom or host.endswith(f".{dom}") for dom in SOURCE_ALLOWLIST_DOMAINS):
            return False, "not_in_allowlist"
    return True, "ok"

def _rl_question_value(role_key: str, question: str) -> float:
    if not RL_ENABLED:
        return 0.0
    roles = REINFORCEMENT_STATE_MAP.get("roles", {}) if isinstance(REINFORCEMENT_STATE_MAP, dict) else {}
    role = roles.get(role_key, {}) if isinstance(roles, dict) else {}
    qstats = role.get("questions", {}) if isinstance(role, dict) else {}
    key = _normalize(question)
    stat = qstats.get(key, {}) if isinstance(qstats, dict) else {}
    shown = max(0, _safe_int(stat.get("shown"), 0))
    reward = _safe_float(stat.get("reward"), 0.0)
    total = 0
    if isinstance(qstats, dict):
        for v in qstats.values():
            if isinstance(v, dict):
                total += max(0, _safe_int(v.get("shown"), 0))
    # UCB-style exploration bonus.
    explore = ((max(1, total) + 1.0) ** 0.5) / ((shown + 1.0) ** 0.5)
    return reward + (0.35 * explore)

def _rl_update_question(role_key: str, question: str, score: int, correct: bool) -> None:
    global REINFORCEMENT_STATE_MAP
    if not RL_ENABLED:
        return
    REINFORCEMENT_STATE_MAP = _load_reinforcement_state()
    roles = REINFORCEMENT_STATE_MAP.setdefault("roles", {})
    role = roles.setdefault(role_key, {"questions": {}, "sources": {}, "questionText": {}, "questionDaily": {}})
    qstats = role.setdefault("questions", {})
    key = _normalize(question)
    question_text = str(question).strip()[:400]
    qtext = role.setdefault("questionText", {})
    qtext[key] = question_text
    role["questionText"] = qtext
    stat = qstats.setdefault(key, {"shown": 0, "good": 0, "bad": 0, "reward": 0.0, "lastScore": 0})
    stat["shown"] = max(0, _safe_int(stat.get("shown"), 0)) + 1
    if correct or score >= 70:
        stat["good"] = max(0, _safe_int(stat.get("good"), 0)) + 1
    else:
        stat["bad"] = max(0, _safe_int(stat.get("bad"), 0)) + 1
    delta = ((_safe_int(score, 0) - 50) / 50.0)
    stat["reward"] = round((_safe_float(stat.get("reward"), 0.0) * 0.9) + (delta * 0.1), 4)
    stat["lastScore"] = _safe_int(score, 0)
    stat["lastUpdated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    qstats[key] = stat
    role["questions"] = qstats
    events = role.setdefault("events", [])
    if isinstance(events, list):
        now_ts = int(time.time())
        day_bucket = time.strftime("%Y-%m-%d", time.gmtime(now_ts))
        events.append({
            "ts": now_ts,
            "role": role_key,
            "question": str(question),
            "score": int(_safe_int(score, 0)),
            "correct": bool(correct),
        })
        qdaily = role.setdefault("questionDaily", {})
        qnode = qdaily.setdefault(key, {})
        b = qnode.setdefault(day_bucket, {"count": 0, "scoreSum": 0.0, "correctSum": 0})
        b["count"] = max(0, _safe_int(b.get("count"), 0)) + 1
        b["scoreSum"] = _safe_float(b.get("scoreSum"), 0.0) + _safe_float(score, 0.0)
        b["correctSum"] = max(0, _safe_int(b.get("correctSum"), 0)) + (1 if correct else 0)
        qnode[day_bucket] = b
        qdaily[key] = qnode
        role["questionDaily"] = qdaily
        # Raw event retention with compaction-ready daily buckets.
        keep_after = now_ts - (RL_RAW_EVENT_RETENTION_DAYS * 86400)
        compacted = [ev for ev in events if isinstance(ev, dict) and _safe_int(ev.get("ts"), 0) >= keep_after]
        if len(compacted) > 120000:
            compacted = compacted[-120000:]
        role["events"] = compacted
    roles[role_key] = role
    REINFORCEMENT_STATE_MAP["roles"] = roles
    _save_reinforcement_state(REINFORCEMENT_STATE_MAP)

def _normalize_source_url(url: str) -> str:
    try:
        parsed = urlparse(str(url or "").strip())
        scheme = (parsed.scheme or "https").lower()
        host = (parsed.netloc or "").lower()
        path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/")
        return f"{scheme}://{host}{path}"
    except Exception:
        return str(url or "").strip()

def _guess_online_source_type(url: str) -> str:
    norm = _normalize_source_url(url)
    if "github.com" in norm:
        return "github"
    if "huggingface.co/datasets/" in norm:
        return "huggingface_dataset"
    return "web"

def _domain_training_keywords(domain: str) -> List[str]:
    key = _normalize(domain)
    if key == "java":
        return [
            "java", "jvm", "jdk", "jre", "spring", "spring boot", "hibernate",
            "maven", "gradle", "servlet", "jdbc", "collections", "multithreading",
            "streams", "lambda", "garbage collection", "exception handling",
        ]
    parts = [p for p in re.split(r"[_\s\-]+", key) if p]
    return parts or [key]

def _matches_domain_keywords(text: str, keywords: List[str]) -> bool:
    hay = str(text or "").lower()
    for raw_kw in keywords:
        kw = str(raw_kw or "").strip().lower()
        if not kw:
            continue
        if " " in kw:
            if kw in hay:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", hay):
            return True
    return False

def _domain_match_score(question: str, answer: str, keywords: List[str]) -> int:
    q_text = str(question or "")
    merged = f"{question or ''} {answer or ''}"
    score = 0
    for raw_kw in keywords:
        kw = str(raw_kw or "").strip().lower()
        if not kw:
            continue
        if _matches_domain_keywords(q_text, [kw]):
            score += 3
        elif _matches_domain_keywords(merged, [kw]):
            score += 1
    return score

AUTO_DOMAIN_HINTS: List[tuple[str, List[str]]] = [
    ("java", ["java", "spring", "spring boot", "hibernate", "jpa", "maven", "gradle", "jvm", "jdk", "jre"]),
    ("angular", ["angular", "angularjs", "rxjs", "ngif", "ngfor", "ngoninit", "typescript"]),
    ("react", ["react", "reactjs", "jsx", "redux", "hooks", "useeffect"]),
    ("javascript", ["javascript", "node", "nodejs", "ecmascript", "js"]),
    ("typescript", ["typescript", "tsconfig", "type narrowing"]),
    ("vue", ["vue", "vuejs", "nuxt"]),
    ("flutter", ["flutter", "dart", "widget tree"]),
    ("python", ["python", "django", "flask", "fastapi", "pandas"]),
    ("sql", ["sql", "mysql", "postgres", "oracle", "database", "jdbc"]),
    ("devops", ["devops", "docker", "kubernetes", "terraform", "jenkins", "ci/cd"]),
    ("sales", ["sales", "closing deals", "prospect", "lead generation", "quota"]),
    ("business_development", ["business development", "bde", "bdm", "pipeline growth"]),
    ("software_engineering", ["software engineering", "software engineer", "mock interview", "interview question"]),
]

def _normalize_label_key(text: str, fallback: str = "general") -> str:
    label = re.sub(r"[^a-z0-9]+", "_", _normalize(text)).strip("_")
    return label or fallback

def _auto_domain_label(text: str, fallback: str = "general") -> str:
    hay = _normalize(text)
    best_label = ""
    best_score = 0
    for label, hints in AUTO_DOMAIN_HINTS:
        score = 0
        for hint in hints:
            if _matches_domain_keywords(hay, [hint]):
                score += 3 if " " in hint else 2
        if score > best_score:
            best_score = score
            best_label = label
    if best_label:
        return _normalize_label_key(best_label, fallback=fallback)
    return _normalize_label_key(fallback, fallback="general")

def _infer_domain_from_source_urls(urls: List[str], fallback: str = "general") -> str:
    parts: List[str] = []
    for url in urls:
        norm_url = _normalize_source_url(url)
        parsed = urlparse(norm_url)
        parts.append(unquote(parsed.path.replace("/", " ")))
    combined = " ".join(parts)
    return _auto_domain_label(combined, fallback=fallback)

def _compact_stackoverflow_question(text: str, max_words: int = 18) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    lower = compact.lower()
    cut = len(compact)
    markers = [
        " i've ", " i have ", " i'm ", " i am ", " i try ", " i tried ", " we ", " my ",
        " when ", " while ", " here ", " stack trace", " error:", " code:", " public ",
        " class ", " @configuration", " @requestmapping",
    ]
    for marker in markers:
        idx = lower.find(marker)
        if 30 <= idx <= 220:
            cut = min(cut, idx)
    compact = compact[:cut].strip(" :-")
    words = re.findall(r"\S+", compact)
    if len(words) > max_words:
        compact = " ".join(words[:max_words]).rstrip(",:;-")
    if len(compact) > 160:
        compact = compact[:160].rsplit(" ", 1)[0].rstrip(",:;-")
    if compact.endswith("."):
        compact = compact[:-1].strip()
    if compact and not compact.endswith("?"):
        compact += "?"
    return compact

def _dataset_only_role(role_key: str) -> bool:
    return bool(DATASET_ONLY_MODE and _normalize(role_key) == "java")

def _default_training_source_set() -> Set[str]:
    return {_normalize_source_url(url) for url in DEFAULT_ONLINE_TRAINING_SOURCES if str(url).strip()}

def _is_huggingface_dataset_training_request(urls: List[str]) -> bool:
    normalized_urls = [_normalize_source_url(url) for url in urls if str(url).strip()]
    return bool(normalized_urls) and all(_guess_online_source_type(url) == "huggingface_dataset" for url in normalized_urls)

def _is_default_dataset_training_request(domain: str, urls: List[str]) -> bool:
    normalized_urls = {_normalize_source_url(url) for url in urls if str(url).strip()}
    return bool(_dataset_only_role(domain) and normalized_urls and normalized_urls.issubset(_default_training_source_set()))

def _small_model_source_registry(model_pack: Dict[str, Any]) -> Dict[str, Any]:
    reg = model_pack.get("sourceRegistry", {}) if isinstance(model_pack, dict) else {}
    return reg if isinstance(reg, dict) else {}

def _source_already_learned(model_pack: Dict[str, Any], url: str) -> Dict[str, Any]:
    registry = _small_model_source_registry(model_pack)
    return registry.get(_normalize_source_url(url), {}) if isinstance(registry, dict) else {}

def _github_repo_parts(url: str) -> tuple[str, str]:
    parsed = urlparse(_normalize_source_url(url))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]

def _fetch_text_url(url: str, timeout_seconds: Optional[int] = None) -> str:
    if not requests:
        return ""
    try:
        resp = requests.get(
            str(url),
            timeout=timeout_seconds or max(5, RAG_FETCH_TIMEOUT * 3),
            headers={
                "User-Agent": "InterviewRAGWorker/5.0",
                "Accept": "application/json, text/plain, text/html;q=0.9, */*;q=0.8",
            },
        )
        if resp.status_code >= 400:
            return ""
        return resp.text or ""
    except Exception:
        return ""

def _fetch_json_url(url: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    text = _fetch_text_url(url, timeout_seconds=timeout_seconds)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

def _github_repo_file_candidates(raw: Any, path_prefix: str = "") -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(raw, dict) and raw.get("type") == "file":
        path = str(raw.get("path", path_prefix)).strip()
        download = str(raw.get("download_url", "")).strip()
        if path and download:
            items.append({"path": path, "download_url": download})
        return items
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).strip()
            path = str(item.get("path", "")).strip()
            if typ == "file":
                download = str(item.get("download_url", "")).strip()
                if path and download:
                    items.append({"path": path, "download_url": download})
    return items

def _is_repo_training_file(path: str) -> bool:
    lo = str(path or "").lower()
    if not lo:
        return False
    preferred_names = ("readme", "question", "answer", "interview", "faq", "guide", "docs", "java")
    preferred_exts = (".md", ".txt", ".rst", ".adoc", ".java")
    return any(name in lo for name in preferred_names) and lo.endswith(preferred_exts)

def _fetch_github_repo_lines(url: str, max_items: int = 120) -> Dict[str, Any]:
    owner, repo = _github_repo_parts(url)
    if not owner or not repo:
        return {"ok": False, "reason": "invalid_github_repo", "lines": [], "meta": {}}
    api = f"https://api.github.com/repos/{owner}/{repo}/contents"
    queue: List[str] = [api]
    scanned = 0
    files: List[Dict[str, str]] = []
    while queue and len(files) < 40 and scanned < 25:
        endpoint = queue.pop(0)
        scanned += 1
        text = _fetch_text_url(endpoint)
        if not text:
            continue
        try:
            raw = json.loads(text)
        except Exception:
            raw = None
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                typ = str(item.get("type", "")).strip()
                path = str(item.get("path", "")).strip()
                if typ == "dir":
                    lo = path.lower()
                    if any(skip in lo for skip in ("/.git", "test", "target", "node_modules")):
                        continue
                    sub = str(item.get("url", "")).strip()
                    if sub:
                        queue.append(sub)
                elif typ == "file":
                    candidate = _github_repo_file_candidates(item)
                    if candidate and _is_repo_training_file(path):
                        files.extend(candidate)
        else:
            files.extend(_github_repo_file_candidates(raw))
    selected = files[: max(4, min(20, max_items // 8))]
    lines: List[str] = []
    for item in selected:
        content = _fetch_text_url(item.get("download_url", ""))
        if not content:
            continue
        ext = str(item.get("path", "")).lower()
        if ext.endswith(".java"):
            extracted = re.findall(r"/\*\*(.*?)\*/|//\s*(.*)", content, flags=re.DOTALL)
            flattened: List[str] = []
            for pair in extracted:
                flattened.extend([x for x in pair if x])
            parts = flattened or content.splitlines()
        else:
            parts = content.splitlines()
        lines.extend(_sanitize_cache_lines(parts, max_items=max_items))
        if len(lines) >= max_items:
            break
    return {
        "ok": bool(lines),
        "reason": ("ok" if lines else "no_repo_lines"),
        "lines": lines[:max_items],
        "meta": {
            "type": "github",
            "owner": owner,
            "repo": repo,
            "scannedEndpoints": scanned,
            "selectedFiles": len(selected),
        },
    }

def _huggingface_dataset_parts(url: str) -> str:
    parsed = urlparse(_normalize_source_url(url))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        return f"{parts[1]}/{parts[2]}"
    return ""

def _hf_rows(dataset_id: str, config: str, split: str = "train", offset: int = 0, length: int = 100) -> List[Dict[str, Any]]:
    rows_url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={quote_plus(dataset_id)}&config={quote_plus(config)}&split={quote_plus(split)}"
        f"&offset={max(0, offset)}&length={max(1, min(length, 200))}"
    )
    data = _fetch_json_url(rows_url, timeout_seconds=max(8, RAG_FETCH_TIMEOUT * 4))
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return rows if isinstance(rows, list) else []

def _row_dict(row: Any) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    inner = row.get("row", None)
    if "row" in row and isinstance(inner, dict):
        return inner
    return row

def _pick_row_value(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text:
            return text
    return ""

def _download_binary_url(url: str, destination: Path, timeout_seconds: int = 90) -> bool:
    if not requests or not str(url or "").strip():
        return False
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=max(20, timeout_seconds)) as resp:
            if resp.status_code != 200:
                logger.warning("Dataset download failed status=%s url=%s", resp.status_code, url)
                return False
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
        if not tmp_path.exists() or tmp_path.stat().st_size < 512:
            return False
        tmp_path.replace(destination)
        return True
    except Exception as ex:
        logger.warning("Dataset download exception url=%s err=%s", url, ex)
        return False
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

def _load_stackoverflow_dataset_records(kind: str) -> List[Dict[str, Any]]:
    cached = STACKOVERFLOW_QA_TABLE_CACHE.get(kind)
    if isinstance(cached, list) and cached:
        return cached
    dataset_url = STACKOVERFLOW_QA_REMOTE_FILES.get(kind, "")
    if not dataset_url or pl is None:
        return []
    dataset_file = STACKOVERFLOW_QA_CACHE_DIR / f"{kind}.parquet"
    if not dataset_file.exists() or dataset_file.stat().st_size < 512:
        if not _download_binary_url(dataset_url, dataset_file):
            return []
    try:
        frame = pl.read_parquet(dataset_file)
        records = frame.to_dicts()
        STACKOVERFLOW_QA_TABLE_CACHE[kind] = records
        return records
    except Exception as ex:
        logger.warning("Unable to read StackOverflow dataset parquet kind=%s path=%s err=%s", kind, dataset_file, ex)
        try:
            dataset_file.unlink()
        except Exception:
            pass
        return []

def _extract_stackoverflow_dataset_knowledge(dataset_id: str, domain: str, max_pairs: int = 80) -> Dict[str, Any]:
    domain_keywords = [kw.lower() for kw in _domain_training_keywords(domain)]
    default_rows = _load_stackoverflow_dataset_records("default") or _hf_rows(dataset_id, "default", "train", 0, max_pairs * 20)
    query_rows = _load_stackoverflow_dataset_records("queries") or _hf_rows(dataset_id, "queries", "train", 0, max_pairs * 30)
    corpus_rows = _load_stackoverflow_dataset_records("corpus") or _hf_rows(dataset_id, "corpus", "train", 0, max_pairs * 30)

    query_map: Dict[str, str] = {}
    for raw in query_rows:
        row = _row_dict(raw)
        qid = _pick_row_value(row, ["query-id", "query_id", "_id", "id"])
        text = _pick_row_value(row, ["text", "query", "body", "question"])
        if qid and text:
            query_map[qid] = text

    corpus_map: Dict[str, str] = {}
    for raw in corpus_rows:
        row = _row_dict(raw)
        cid = _pick_row_value(row, ["corpus-id", "corpus_id", "_id", "id"])
        title = _pick_row_value(row, ["title", "heading"])
        text = _pick_row_value(row, ["text", "body", "answer", "contents"])
        combined = re.sub(r"\s+", " ", " ".join(x for x in [title, text] if x)).strip()
        if cid and combined:
            corpus_map[cid] = combined

    scored_pairs: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for raw in default_rows:
        row = _row_dict(raw)
        qid = _pick_row_value(row, ["query-id", "query_id"])
        cid = _pick_row_value(row, ["corpus-id", "corpus_id"])
        question = query_map.get(qid, "")
        answer = corpus_map.get(cid, "")
        merged = f"{question} {answer}".lower()
        if not question or not answer:
            continue
        if domain_keywords and not _matches_domain_keywords(merged, domain_keywords):
            continue
        if _looks_like_question_dump_line(answer) and not _is_reference_answer_usable(answer):
            continue
        question = _compact_stackoverflow_question(question)
        question = _normalize_question(question) or re.sub(r"\s+", " ", question).strip()
        answer = _limit_words(answer, 180)
        if not question or not answer:
            continue
        key = _normalize(question)
        if key in seen:
            continue
        seen.add(key)
        scored_pairs.append({
            "question": question,
            "answer": answer,
            "score": _domain_match_score(question, answer, domain_keywords),
        })

    scored_pairs.sort(key=lambda item: (int(item.get("score", 0)), len(str(item.get("answer", "")))), reverse=True)
    qa_pairs: List[Dict[str, str]] = [
        {"question": str(item.get("question", "")).strip(), "answer": str(item.get("answer", "")).strip()}
        for item in scored_pairs[:max_pairs]
        if str(item.get("question", "")).strip() and str(item.get("answer", "")).strip()
    ]

    if not qa_pairs:
        # Fallback: use the Java-like query text as question prompts if pairing fails.
        for raw in query_rows:
            row = _row_dict(raw)
            text = _pick_row_value(row, ["text", "query", "body", "question"])
            if not text:
                continue
            if domain_keywords and not _matches_domain_keywords(text, domain_keywords):
                continue
            question = _normalize_question(text) or ""
            if not question:
                continue
            qa_pairs.append({"question": question, "answer": ""})
            if len(qa_pairs) >= max_pairs:
                break

    context_lines: List[str] = []
    for pair in qa_pairs:
        if pair.get("question"):
            context_lines.append(pair["question"])
        if pair.get("answer"):
            context_lines.append(pair["answer"])
    return {
        "qaPairs": qa_pairs,
        "questionBank": [pair["question"] for pair in qa_pairs if pair.get("question")],
        "contextLines": _filter_answer_context_lines(_sanitize_cache_lines(context_lines, max_items=max_pairs * 3), max_items=max_pairs * 3),
    }

def _dataset_answer_text_usable(answer: str) -> bool:
    text = _limit_words(answer, 180)
    if not text:
        return False
    if _looks_like_question_dump_line(text):
        return False
    return len(re.findall(r"\S+", text)) >= 4

def _extract_generic_huggingface_dataset_knowledge(
    dataset_id: str,
    source_url: str,
    requested_domain: str = "",
    max_items: int = 120,
) -> Dict[str, Any]:
    info_url = f"https://datasets-server.huggingface.co/info?dataset={quote_plus(dataset_id)}"
    info = _fetch_json_url(info_url, timeout_seconds=max(8, RAG_FETCH_TIMEOUT * 4))
    raw_configs = info.get("dataset_info", {}) if isinstance(info, dict) else {}
    configs = list(raw_configs.keys()) if isinstance(raw_configs, dict) and raw_configs else ["default"]
    max_row_scan = max(100, min(600, max_items * 6))
    used_config = ""
    used_split = ""
    domain_buckets: Dict[str, Dict[str, Any]] = {}
    combined_questions: List[str] = []
    combined_qas: List[Dict[str, str]] = []
    combined_context: List[str] = []
    seen_questions: Set[str] = set()
    seen_qas: Set[str] = set()

    def _bucket(label: str) -> Dict[str, Any]:
        key = _normalize_label_key(label, fallback="general")
        node = domain_buckets.get(key)
        if not isinstance(node, dict):
            node = {
                "questionBank": [],
                "qaPairs": [],
                "contextLines": [],
                "_seenQuestions": set(),
                "_seenQas": set(),
            }
            domain_buckets[key] = node
        return node

    for config in configs[:4]:
        for split in ("train", "test", "validation"):
            rows: List[Dict[str, Any]] = []
            for offset in range(0, max_row_scan, 100):
                batch = _hf_rows(dataset_id, config, split, offset, min(100, max_row_scan - offset))
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < 100:
                    break
            if not rows:
                continue
            used_config = config
            used_split = split
            for raw in rows:
                row = _row_dict(raw)
                if not row:
                    continue
                instruction = _pick_row_value(row, ["input", "prompt", "instruction", "context", "topic", "category", "classes", "label"])
                explicit_question = _pick_row_value(row, ["question", "question_text", "query", "title"])
                explicit_answer = _pick_row_value(row, ["answer", "expected_answer", "ideal_answer", "best_answer", "solution", "explanation"])
                response = _pick_row_value(row, ["response", "output", "completion", "text"])
                raw_question = ""
                raw_answer = ""
                if explicit_question:
                    raw_question = explicit_question
                    if explicit_answer:
                        raw_answer = explicit_answer
                    elif response and _normalize(response) != _normalize(explicit_question) and not _normalize_question(response):
                        raw_answer = response
                elif response and _normalize_question(_compact_stackoverflow_question(response)):
                    raw_question = response
                    raw_answer = explicit_answer
                elif instruction and response and _normalize_question(_compact_stackoverflow_question(instruction)):
                    raw_question = instruction
                    raw_answer = response
                elif response:
                    raw_question = response
                    raw_answer = explicit_answer
                question = _normalize_question(_compact_stackoverflow_question(raw_question)) or ""
                answer = _limit_words(raw_answer, 180)
                if not question:
                    continue
                label_hint = requested_domain or _pick_row_value(row, ["classes", "class", "topic", "category", "label"])
                label = _normalize_label_key(
                    label_hint or _auto_domain_label(
                        " ".join([instruction, question, answer]),
                        fallback=_infer_domain_from_source_urls([source_url], fallback="software_engineering"),
                    ),
                    fallback="software_engineering",
                )
                bucket = _bucket(label)
                bucket_seen_q = bucket["_seenQuestions"]
                question_key = _normalize(question)
                if question_key not in bucket_seen_q:
                    bucket_seen_q.add(question_key)
                    bucket["questionBank"].append(question)
                if question_key not in seen_questions:
                    seen_questions.add(question_key)
                    combined_questions.append(question)
                if _dataset_answer_text_usable(answer):
                    if question_key not in bucket["_seenQas"]:
                        bucket["_seenQas"].add(question_key)
                        bucket["qaPairs"].append({"question": question, "answer": answer})
                    if question_key not in seen_qas:
                        seen_qas.add(question_key)
                        combined_qas.append({"question": question, "answer": answer})
                for line in [instruction, question, answer]:
                    if str(line or "").strip():
                        bucket["contextLines"].append(str(line).strip())
                        combined_context.append(str(line).strip())
            if combined_questions:
                break
        if combined_questions:
            break

    for label, node in list(domain_buckets.items()):
        node["questionBank"] = [str(q).strip() for q in node.get("questionBank", [])[: max_items * 2] if str(q).strip()]
        node["qaPairs"] = [item for item in node.get("qaPairs", [])[: max_items * 2] if isinstance(item, dict)]
        node["contextLines"] = _sanitize_cache_lines(node.get("contextLines", []), max_items=max_items * 3)
        node.pop("_seenQuestions", None)
        node.pop("_seenQas", None)
        domain_buckets[label] = node

    return {
        "questionBank": combined_questions[: max_items * 2],
        "qaPairs": combined_qas[: max_items * 2],
        "contextLines": _sanitize_cache_lines(combined_context, max_items=max_items * 3),
        "domainBuckets": domain_buckets,
        "config": used_config,
        "split": used_split,
    }

def _fetch_huggingface_dataset_lines(url: str, max_items: int = 120) -> Dict[str, Any]:
    dataset_id = _huggingface_dataset_parts(url)
    if not dataset_id:
        return {"ok": False, "reason": "invalid_huggingface_dataset", "lines": [], "meta": {}}
    if dataset_id == "mteb/stackoverflow-qa":
        dataset_knowledge = _extract_stackoverflow_dataset_knowledge(dataset_id, "java", max_pairs=max(30, min(90, max_items)))
        lines = dataset_knowledge.get("contextLines", [])
        return {
            "ok": bool(lines),
            "reason": ("ok" if lines else "no_dataset_rows"),
            "lines": lines[:max_items],
            "meta": {
                "type": "huggingface_dataset",
                "datasetId": dataset_id,
                "qaPairs": dataset_knowledge.get("qaPairs", []),
                "questionBank": dataset_knowledge.get("questionBank", []),
                "contextLines": dataset_knowledge.get("contextLines", []),
            },
        }
    generic_knowledge = _extract_generic_huggingface_dataset_knowledge(dataset_id, url, max_items=max_items)
    candidate_rows = generic_knowledge.get("contextLines", [])
    if not candidate_rows:
        fallback = _fetch_text_url(url, timeout_seconds=max(8, RAG_FETCH_TIMEOUT * 4))
        candidate_rows = _sanitize_cache_lines(fallback.splitlines(), max_items=max_items) if fallback else []
    return {
        "ok": bool(candidate_rows),
        "reason": ("ok" if candidate_rows else "no_dataset_rows"),
        "lines": candidate_rows[:max_items],
        "meta": {
            "type": "huggingface_dataset",
            "datasetId": dataset_id,
            "config": generic_knowledge.get("config", ""),
            "split": generic_knowledge.get("split", ""),
            "qaPairs": generic_knowledge.get("qaPairs", []),
            "questionBank": generic_knowledge.get("questionBank", []),
            "contextLines": generic_knowledge.get("contextLines", []),
            "domainBuckets": generic_knowledge.get("domainBuckets", {}),
        },
    }

def _fetch_online_training_source(url: str, max_items: int = 120) -> Dict[str, Any]:
    source_type = _guess_online_source_type(url)
    norm_url = _normalize_source_url(url)
    started = time.perf_counter()
    if source_type == "github":
        result = _fetch_github_repo_lines(norm_url, max_items=max_items)
    elif source_type == "huggingface_dataset":
        result = _fetch_huggingface_dataset_lines(norm_url, max_items=max_items)
    else:
        result = {"ok": False, "reason": "no_content", "lines": [], "meta": {"type": source_type}}
        lines = _fetch_url(norm_url) or [_fetch_text_url(norm_url)]
        flattened: List[str] = []
        for item in lines:
            if isinstance(item, list):
                flattened.extend([str(x) for x in item])
            else:
                flattened.append(str(item))
        result["lines"] = _sanitize_cache_lines(flattened, max_items=max_items)
        result["ok"] = bool(result["lines"])
        result["reason"] = "ok" if result["ok"] else "no_web_lines"
    duration_ms = round((time.perf_counter() - started) * 1000)
    lines = result.get("lines", []) if isinstance(result, dict) else []
    clean_lines = _sanitize_cache_lines(lines, max_items=max_items)
    fingerprint = hashlib.sha256("\n".join(clean_lines[:max_items]).encode("utf-8")).hexdigest()[:20] if clean_lines else ""
    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    meta.update({
        "url": norm_url,
        "type": source_type,
        "durationMs": duration_ms,
        "fingerprint": fingerprint,
        "lineCount": len(clean_lines),
    })
    result["lines"] = clean_lines
    result["meta"] = meta
    return result

def _estimate_online_training_source(url: str, model_pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    norm_url = _normalize_source_url(url)
    source_type = _guess_online_source_type(norm_url)
    learned = _source_already_learned(model_pack or {}, norm_url) if isinstance(model_pack, dict) else {}
    already = bool(learned)
    minutes = 2
    if source_type == "github":
        minutes = 4
    elif source_type == "huggingface_dataset":
        minutes = 10
    elif source_type == "web":
        minutes = 2
    return {
        "url": norm_url,
        "type": source_type,
        "alreadyLearned": already,
        "estimatedMinutes": 0 if already else minutes,
        "lastLearnedAt": str(learned.get("learnedAt", "")) if already else "",
        "fingerprint": str(learned.get("fingerprint", "")) if already else "",
    }

def _extract_online_training_samples(domain: str, lines: List[str], max_samples: int = 240) -> Dict[str, Any]:
    cleaned = _sanitize_cache_lines(lines, max_items=max_samples)
    questions = _extract_questions_from_lines(cleaned, min(60, max(10, max_samples // 4)))
    question_keys = {_normalize(q) for q in questions}
    concepts: List[str] = []
    for line in cleaned:
        if _normalize(line) in question_keys:
            continue
        concepts.append(line)
        if len(concepts) >= max(30, max_samples // 3):
            break
    samples: List[Dict[str, Any]] = []
    for q in questions:
        samples.append({"text": f"{domain} [SEP] {q}", "label": 0.95, "kind": "question"})
    for line in concepts:
        samples.append({"text": f"{domain} [SEP] {line}", "label": 0.45, "kind": "concept"})
    if not samples:
        for line in cleaned[:40]:
            samples.append({"text": f"{domain} [SEP] {line}", "label": 0.40, "kind": "fallback"})
    return {
        "samples": samples[:max_samples],
        "questions": questions[:60],
        "concepts": concepts[:60],
        "lineCount": len(cleaned),
    }

def _fit_or_update_small_model(model_pack: Optional[Dict[str, Any]], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if HashingVectorizer is None or SGDRegressor is None:
        return {"ok": False, "reason": "scikit-learn unavailable"}
    X_text = [str(s.get("text", "")).strip() for s in samples if str(s.get("text", "")).strip()]
    y = [float(_safe_float(s.get("label"), 0.0)) for s in samples if str(s.get("text", "")).strip()]
    if not X_text:
        return {"ok": False, "reason": "no_training_samples"}
    pack = model_pack if isinstance(model_pack, dict) else {}
    vectorizer = pack.get("vectorizer")
    model = pack.get("model")
    if vectorizer is None:
        vectorizer = HashingVectorizer(
            n_features=2 ** 14,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )
    if model is None:
        model = SGDRegressor(
            loss="huber",
            alpha=1e-5,
            max_iter=1200,
            tol=1e-3,
            random_state=42,
        )
    X = vectorizer.transform(X_text)
    try:
        if hasattr(model, "partial_fit") and hasattr(model, "coef_"):
            model.partial_fit(X, y)
        else:
            model.fit(X, y)
    except Exception:
        model.fit(X, y)
    pack["vectorizer"] = vectorizer
    pack["model"] = model
    pack["trainedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pack["sampleCount"] = max(len(X_text), _safe_int(pack.get("sampleCount"), 0) + len(X_text))
    pack["trainingMode"] = "online_sources"
    return {"ok": True, "pack": pack, "sampleCount": len(X_text)}

def _small_model_domain_knowledge(model_pack: Optional[Dict[str, Any]], role_key: str) -> Dict[str, Any]:
    pack = model_pack if isinstance(model_pack, dict) else (_load_small_model() or {})
    domains = pack.get("domainKnowledge", {}) if isinstance(pack, dict) else {}
    if not isinstance(domains, dict):
        return {}
    node = domains.get(_normalize(role_key), {})
    return node if isinstance(node, dict) else {}

def _small_model_has_dataset_knowledge(model_pack: Optional[Dict[str, Any]], role_key: str = "java") -> bool:
    pack = model_pack if isinstance(model_pack, dict) else (_load_small_model() or {})
    node = _small_model_domain_knowledge(pack, role_key)
    registry = _small_model_source_registry(pack)
    non_dataset_sources = [
        url for url, meta in registry.items()
        if not (isinstance(meta, dict) and str(meta.get("type", "")).strip() == "huggingface_dataset")
    ] if isinstance(registry, dict) else []
    if non_dataset_sources:
        return False
    return bool(
        isinstance(node, dict)
        and len(node.get("questionBank", [])) >= INTERVIEW_QUESTION_COUNT
        and len(node.get("qaBank", [])) >= 5
    )

def _small_model_question_bank(role_key: str, count: int = INTERVIEW_QUESTION_COUNT) -> List[str]:
    node = _small_model_domain_knowledge(None, role_key)
    questions = node.get("questionBank", []) if isinstance(node, dict) else []
    out: List[str] = []
    seen: Set[str] = set()
    for q in questions if isinstance(questions, list) else []:
        text = _normalize_question(q) or re.sub(r"\s+", " ", str(q or "")).strip()
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= count:
            break
    return out

def _small_model_qa_bank(role_key: str) -> List[Dict[str, str]]:
    node = _small_model_domain_knowledge(None, role_key)
    qa_bank = node.get("qaBank", []) if isinstance(node, dict) else []
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for item in qa_bank if isinstance(qa_bank, list) else []:
        if not isinstance(item, dict):
            continue
        q = _normalize_question(item.get("question", "")) or ""
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        key = _normalize(q)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "answer": _limit_words(a, 180)})
    return out

def _small_model_context_lines(role_key: str, count: int = 60) -> List[str]:
    node = _small_model_domain_knowledge(None, role_key)
    lines = node.get("contextLines", []) if isinstance(node, dict) else []
    return _filter_answer_context_lines([str(x) for x in lines], max_items=count)[:count]

def _merge_small_model_domain_knowledge(
    pack: Dict[str, Any],
    role_key: str,
    question_bank: List[str],
    qa_bank: List[Dict[str, str]],
    context_lines: List[str],
) -> None:
    domains = pack.setdefault("domainKnowledge", {})
    if not isinstance(domains, dict):
        domains = {}
        pack["domainKnowledge"] = domains
    key = _normalize(role_key) or "general"
    existing = domains.get(key, {})
    if not isinstance(existing, dict):
        existing = {}
    merged_questions = list(dict.fromkeys([*(existing.get("questionBank", []) or []), *[str(q).strip() for q in question_bank if str(q).strip()]]))
    existing_qa = existing.get("qaBank", []) if isinstance(existing.get("qaBank", []), list) else []
    merged_qa: List[Dict[str, str]] = []
    seen_qa: Set[str] = set()
    for item in [*existing_qa, *qa_bank]:
        if not isinstance(item, dict):
            continue
        q = _normalize_question(item.get("question", "")) or ""
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        qkey = _normalize(q)
        if qkey in seen_qa:
            continue
        seen_qa.add(qkey)
        merged_qa.append({"question": q, "answer": _limit_words(a, 180)})
    merged_lines = _filter_answer_context_lines(
        [*(existing.get("contextLines", []) or []), *[str(x) for x in context_lines if str(x).strip()]],
        max_items=240,
    )
    existing.update({
        "questionBank": merged_questions[:240],
        "qaBank": merged_qa[:240],
        "contextLines": merged_lines[:240],
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    domains[key] = existing

def _save_small_model_pack(pack: Dict[str, Any]) -> None:
    global SMALL_MODEL_HANDLE, LAST_SMALL_MODEL_TRAIN_TS
    SMALL_MODEL_HANDLE = pack
    with SMALL_MODEL_FILE.open("wb") as f:
        pickle.dump(pack, f)
    LAST_SMALL_MODEL_TRAIN_TS = time.time()

def _current_small_model_sources(model_pack: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    pack = model_pack if isinstance(model_pack, dict) else (_load_small_model() or {})
    registry = _small_model_source_registry(pack)
    rows: List[Dict[str, Any]] = []
    for url, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        row = {"url": url, **meta}
        rows.append(row)
    rows.sort(key=lambda x: str(x.get("learnedAt", "")), reverse=True)
    return rows

def _online_training_status(urls: List[str]) -> Dict[str, Any]:
    pack = _load_small_model() or {}
    items: List[Dict[str, Any]] = []
    learned = 0
    for url in urls:
        rec = _source_already_learned(pack, url)
        items.append({
            "url": _normalize_source_url(url),
            "learned": bool(rec),
            "details": rec if isinstance(rec, dict) else {},
        })
        if rec:
            learned += 1
    return {
        "ok": True,
        "modelPath": str(SMALL_MODEL_FILE),
        "checked": len(items),
        "learnedCount": learned,
        "items": items,
    }

def _ensure_default_small_model_training() -> Dict[str, Any]:
    pack = _load_small_model() or {}
    status = _online_training_status(DEFAULT_ONLINE_TRAINING_SOURCES)
    missing = [item.get("url", "") for item in status.get("items", []) if not item.get("learned")]
    source_registry = _small_model_source_registry(pack)
    foreign_sources = sorted(
        url for url, meta in source_registry.items()
        if not (isinstance(meta, dict) and str(meta.get("type", "")).strip() == "huggingface_dataset")
    )
    has_dataset_knowledge = _small_model_has_dataset_knowledge(pack, "java")
    if not missing and has_dataset_knowledge and not foreign_sources:
        logger.info("[ONLINE_TRAIN][startup] all default sources already learned")
        return {"ok": True, "skipped": True, "reason": "already_learned_defaults", "modelPath": str(SMALL_MODEL_FILE)}
    urls_to_train = missing or list(DEFAULT_ONLINE_TRAINING_SOURCES)
    logger.info(
        "[ONLINE_TRAIN][startup] training default dataset sources count=%s knowledgeReady=%s foreignSources=%s",
        len(urls_to_train), has_dataset_knowledge, len(foreign_sources)
    )
    return _train_small_model_from_online_sources(OnlineTrainingSetRequest(
        domain="java",
        sourceUrls=urls_to_train,
        estimateOnly=False,
        forceRetrain=bool(foreign_sources) or not has_dataset_knowledge,
        maxItemsPerSource=120,
    ))

def _train_small_model_from_online_sources(payload: OnlineTrainingSetRequest) -> Dict[str, Any]:
    global WORKER_STARTUP_STATUS
    if not SMALL_MODEL_ENABLED:
        return {"ok": False, "reason": "small model disabled"}
    requested_domain = _normalize_label_key(payload.domain, fallback="") if str(payload.domain or "").strip() else ""
    urls = list(dict.fromkeys(_normalize_source_url(u) for u in payload.sourceUrls if str(u).strip()))
    domain = requested_domain or _infer_domain_from_source_urls(urls, fallback="software_engineering")
    dataset_only_training = _is_default_dataset_training_request(domain, urls) or _is_huggingface_dataset_training_request(urls)
    if (HashingVectorizer is None or SGDRegressor is None) and not dataset_only_training:
        return {"ok": False, "reason": "scikit-learn unavailable"}
    if not urls:
        return {"ok": False, "reason": "no_source_urls"}
    existing_pack = _load_small_model() or {}
    pack = (
        {"domainKnowledge": {}, "sourceRegistry": {}, "contentMode": "dataset_only"}
        if dataset_only_training and payload.forceRetrain
        else copy.deepcopy(existing_pack)
    )
    estimates = [_estimate_online_training_source(url, model_pack=pack) for url in urls]
    estimate_total = sum(_safe_int(item.get("estimatedMinutes"), 0) for item in estimates)
    if payload.estimateOnly:
        return {
            "ok": True,
            "estimateOnly": True,
            "domain": domain,
            "sourceCount": len(urls),
            "estimatedTotalMinutes": estimate_total,
            "items": estimates,
            "modelPath": str(SMALL_MODEL_FILE),
        }

    started = time.perf_counter()
    source_registry = _small_model_source_registry(pack)
    all_samples: List[Dict[str, Any]] = []
    learned_sources: List[Dict[str, Any]] = []
    skipped_sources: List[Dict[str, Any]] = []
    learned_domains: Set[str] = set()
    merged_question_total = 0
    for item in estimates:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        existing = _source_already_learned(pack, url)
        if existing and not payload.forceRetrain:
            logger.info("[ONLINE_TRAIN][%s] source=%s status=skip_already_learned", domain, url)
            _append_learning_log("online_train_skip", {"domain": domain, "url": url, "reason": "already_learned"})
            skipped_sources.append({"url": url, "reason": "already_learned", "details": existing})
            continue
        logger.info("[ONLINE_TRAIN][%s] source=%s status=fetch_start estimateMinutes=%s", domain, url, item.get("estimatedMinutes", 0))
        fetched = _fetch_online_training_source(url, max_items=max(40, min(payload.maxItemsPerSource, 240)))
        meta = fetched.get("meta", {}) if isinstance(fetched, dict) else {}
        lines = fetched.get("lines", []) if isinstance(fetched, dict) else []
        if not isinstance(meta, dict):
            meta = {}
        if not lines:
            logger.info("[ONLINE_TRAIN][%s] source=%s status=fetch_empty", domain, url)
            _append_learning_log("online_train_fail", {"domain": domain, "url": url, "reason": fetched.get("reason", "fetch_empty")})
            skipped_sources.append({"url": url, "reason": fetched.get("reason", "fetch_empty"), "details": meta})
            continue
        raw_domain_buckets = meta.get("domainBuckets", {}) if isinstance(meta.get("domainBuckets", {}), dict) else {}
        merge_targets: List[tuple[str, Dict[str, Any]]] = []
        if raw_domain_buckets and not requested_domain:
            for label, bucket in raw_domain_buckets.items():
                if not isinstance(bucket, dict):
                    continue
                merge_targets.append((_normalize_label_key(label, fallback=domain), bucket))
        else:
            merge_targets.append((requested_domain or domain, {
                "qaPairs": meta.get("qaPairs", []) if isinstance(meta.get("qaPairs", []), list) else [],
                "questionBank": meta.get("questionBank", []) if isinstance(meta.get("questionBank", []), list) else [],
                "contextLines": meta.get("contextLines", []) if isinstance(meta.get("contextLines", []), list) else lines,
            }))

        sample_count = 0
        source_question_count = 0
        source_domains: List[str] = []
        for target_domain, bucket in merge_targets:
            qa_pairs = bucket.get("qaPairs", []) if isinstance(bucket.get("qaPairs", []), list) else []
            question_bank = bucket.get("questionBank", []) if isinstance(bucket.get("questionBank", []), list) else []
            context_bank = bucket.get("contextLines", []) if isinstance(bucket.get("contextLines", []), list) else lines
            extracted = _extract_online_training_samples(
                target_domain,
                [str(line) for line in context_bank if str(line).strip()] or lines,
                max_samples=max(80, min(payload.maxItemsPerSource * 2, 260)),
            )
            source_question_count += max(len(extracted.get("questions", [])), len(question_bank))
            sample_count += len(extracted.get("samples", []))
            if not dataset_only_training:
                all_samples.extend(extracted.get("samples", []))
            if qa_pairs or question_bank or context_bank:
                _merge_small_model_domain_knowledge(
                    pack,
                    target_domain,
                    [str(q) for q in question_bank if str(q).strip()],
                    [entry for entry in qa_pairs if isinstance(entry, dict)],
                    [str(line) for line in context_bank if str(line).strip()],
                )
            if question_bank or qa_pairs:
                learned_domains.add(target_domain)
                source_domains.append(target_domain)
                merged_question_total += len(question_bank)
        source_rec = {
            "url": url,
            "type": meta.get("type", _guess_online_source_type(url)),
            "domain": (requested_domain or (source_domains[0] if len(source_domains) == 1 else "multi")),
            "domains": sorted(set(source_domains)),
            "fingerprint": meta.get("fingerprint", ""),
            "learnedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lineCount": len(lines),
            "questionCount": source_question_count,
            "sampleCount": sample_count,
            "durationMs": meta.get("durationMs", 0),
        }
        source_registry[_normalize_source_url(url)] = source_rec
        logger.info(
            "[ONLINE_TRAIN][%s] source=%s status=learned lines=%s questions=%s samples=%s",
            domain, url, source_rec["lineCount"], source_rec["questionCount"], source_rec["sampleCount"]
        )
        _append_learning_log("online_train_source", {
            "domain": domain,
            "url": url,
            "type": source_rec["type"],
            "lineCount": source_rec["lineCount"],
            "questionCount": source_rec["questionCount"],
            "sampleCount": source_rec["sampleCount"],
        })
        learned_sources.append(source_rec)
    if not dataset_only_training and not all_samples:
        return {
            "ok": False,
            "reason": "no_new_samples",
            "domain": domain,
            "estimatedTotalMinutes": estimate_total,
            "skippedSources": skipped_sources,
            "modelPath": str(SMALL_MODEL_FILE),
        }
    if dataset_only_training and not merged_question_total:
        return {
            "ok": False,
            "reason": "no_dataset_questions_found",
            "domain": domain,
            "estimatedTotalMinutes": estimate_total,
            "skippedSources": skipped_sources,
            "modelPath": str(SMALL_MODEL_FILE),
        }
    if dataset_only_training:
        pack.pop("vectorizer", None)
        pack.pop("model", None)
        domain_map = pack.get("domainKnowledge", {}) if isinstance(pack.get("domainKnowledge", {}), dict) else {}
        total_questions = sum(len((node.get("questionBank", []) or [])) for node in domain_map.values() if isinstance(node, dict))
        total_answers = sum(len((node.get("qaBank", []) or [])) for node in domain_map.values() if isinstance(node, dict))
        pack["trainedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pack["sampleCount"] = max(total_questions, total_answers, merged_question_total)
        pack["trainingMode"] = "dataset_only"
        pack["contentMode"] = "dataset_only"
        fit_result = {"ok": True, "pack": pack, "sampleCount": pack.get("sampleCount", 0)}
    else:
        fit_result = _fit_or_update_small_model(pack, all_samples)
    if not fit_result.get("ok"):
        return fit_result
    pack = fit_result.get("pack", {}) if isinstance(fit_result.get("pack"), dict) else {}
    pack["sourceRegistry"] = source_registry
    pack["onlineTrainingSummary"] = {
        "domain": requested_domain or "auto",
        "learnedDomains": sorted(learned_domains),
        "lastTrainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lastSourceCount": len(learned_sources),
        "knownSourceCount": len(source_registry),
    }
    _save_small_model_pack(pack)
    WORKER_STARTUP_STATUS["smallModelReady"] = True
    WORKER_STARTUP_STATUS["smallModelReason"] = (
        f"Small model trained online: {fit_result.get('sampleCount', 0)} new samples"
    )
    duration_seconds = round(time.perf_counter() - started, 2)
    _append_learning_log("online_train_done", {
        "domain": domain,
        "sourceCount": len(learned_sources),
        "sampleCount": fit_result.get("sampleCount", 0),
        "durationSeconds": duration_seconds,
    })
    return {
        "ok": True,
        "estimateOnly": False,
        "domain": requested_domain or "auto",
        "learnedDomains": sorted(learned_domains),
        "estimatedTotalMinutes": estimate_total,
        "actualDurationSeconds": duration_seconds,
        "learnedSources": learned_sources,
        "skippedSources": skipped_sources,
        "sampleCount": fit_result.get("sampleCount", 0),
        "knownSourceCount": len(source_registry),
        "modelPath": str(SMALL_MODEL_FILE),
    }

def _load_small_model():
    global SMALL_MODEL_HANDLE
    if not SMALL_MODEL_ENABLED:
        return None
    if SMALL_MODEL_HANDLE is not None:
        return SMALL_MODEL_HANDLE
    if not SMALL_MODEL_FILE.exists():
        return None
    try:
        with SMALL_MODEL_FILE.open("rb") as f:
            SMALL_MODEL_HANDLE = pickle.load(f)
        return SMALL_MODEL_HANDLE
    except Exception:
        return None

def _small_model_predict(role_key: str, question: str) -> float:
    model_pack = _load_small_model()
    if not model_pack or HashingVectorizer is None:
        return 0.0
    try:
        vec = model_pack.get("vectorizer")
        mdl = model_pack.get("model")
        if vec is None or mdl is None:
            return 0.0
        X = vec.transform([f"{role_key} [SEP] {question}"])
        pred = float(mdl.predict(X)[0])
        return max(-1.0, min(1.0, pred))
    except Exception:
        return 0.0

def _train_small_model(window_days: int = 365, min_samples: int = 80) -> Dict[str, Any]:
    global SMALL_MODEL_HANDLE, REINFORCEMENT_STATE_MAP, LAST_SMALL_MODEL_TRAIN_TS
    if DATASET_ONLY_MODE:
        return {"ok": False, "reason": "dataset_only_mode"}
    if not SMALL_MODEL_ENABLED:
        return {"ok": False, "reason": "small model disabled"}
    if HashingVectorizer is None or SGDRegressor is None:
        return {"ok": False, "reason": "scikit-learn unavailable"}
    REINFORCEMENT_STATE_MAP = _load_reinforcement_state()
    roles = REINFORCEMENT_STATE_MAP.get("roles", {}) if isinstance(REINFORCEMENT_STATE_MAP, dict) else {}
    now_ts = int(time.time())
    cutoff = now_ts - max(1, window_days) * 86400
    limits = _resource_limits()
    X_text: List[str] = []
    y: List[float] = []
    for role_key, role_node in roles.items():
        if not isinstance(role_node, dict):
            continue
        events = role_node.get("events", [])
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ts = _safe_int(ev.get("ts"), 0)
            if ts < cutoff:
                continue
            q = str(ev.get("question", "")).strip()
            if not q:
                continue
            score = _safe_int(ev.get("score"), 0)
            reward = max(-1.0, min(1.0, (score - 50) / 50.0))
            X_text.append(f"{role_key} [SEP] {q}")
            y.append(reward)
        # Daily compacted history for long windows (1 month / 1 year) without large raw scans.
        qdaily = role_node.get("questionDaily", {})
        qtext = role_node.get("questionText", {})
        if isinstance(qdaily, dict):
            for qkey, bucket_map in qdaily.items():
                if not isinstance(bucket_map, dict):
                    continue
                q = str(qtext.get(qkey, qkey)).strip()
                if not q:
                    continue
                for day, b in bucket_map.items():
                    try:
                        day_ts = int(time.mktime(time.strptime(day, "%Y-%m-%d")))
                    except Exception:
                        continue
                    if day_ts < cutoff:
                        continue
                    count = max(1, _safe_int(b.get("count"), 1))
                    avg_score = _safe_float(b.get("scoreSum"), 0.0) / max(1, count)
                    reward = max(-1.0, min(1.0, (avg_score - 50.0) / 50.0))
                    repeats = min(3, count)  # weighted but bounded
                    for _ in range(repeats):
                        X_text.append(f"{role_key} [SEP] {q}")
                        y.append(reward)
                    if len(X_text) >= 250000:
                        break
                if len(X_text) >= 250000:
                    break
        if len(X_text) >= 250000:
            break
    if len(X_text) < max(20, min_samples):
        return {"ok": False, "reason": f"not enough samples ({len(X_text)})"}
    # Resource profile cap for fitting time/memory.
    sample_cap = 50000 if RESOURCE_PROFILE == "low" else 120000 if RESOURCE_PROFILE == "medium" else 220000
    if len(X_text) > sample_cap:
        idxs = list(range(len(X_text)))
        random.shuffle(idxs)
        idxs = idxs[:sample_cap]
        X_text = [X_text[i] for i in idxs]
        y = [y[i] for i in idxs]
    vectorizer = HashingVectorizer(
        n_features=2 ** 14,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    X = vectorizer.transform(X_text)
    model = SGDRegressor(
        loss="huber",
        alpha=1e-5,
        max_iter=1200,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X, y)
    existing_pack = _load_small_model() or {}
    SMALL_MODEL_HANDLE = {
        "vectorizer": vectorizer,
        "model": model,
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "windowDays": window_days,
        "sampleCount": len(X_text),
        "sourceRegistry": _small_model_source_registry(existing_pack),
        "onlineTrainingSummary": existing_pack.get("onlineTrainingSummary", {}) if isinstance(existing_pack, dict) else {},
    }
    _save_small_model_pack(SMALL_MODEL_HANDLE)
    return {
        "ok": True,
        "trainedAt": SMALL_MODEL_HANDLE["trainedAt"],
        "windowDays": window_days,
        "sampleCount": len(X_text),
        "resourceProfile": RESOURCE_PROFILE,
        "limits": limits,
        "modelPath": str(SMALL_MODEL_FILE),
    }

def _maybe_auto_train_small_model(trigger: str = "auto", force: bool = False) -> Dict[str, Any]:
    global LAST_SMALL_MODEL_TRAIN_TS, WORKER_STARTUP_STATUS
    if DATASET_ONLY_MODE:
        return {"ok": False, "reason": "dataset_only_mode"}
    if not SMALL_MODEL_AUTO_TRAIN_ENABLED and not force:
        return {"ok": False, "reason": "auto train disabled"}
    now = time.time()
    if not force and LAST_SMALL_MODEL_TRAIN_TS > 0 and (now - LAST_SMALL_MODEL_TRAIN_TS) < SMALL_MODEL_AUTO_TRAIN_GAP_SECONDS:
        return {"ok": False, "reason": "gap_not_elapsed"}
    result = _train_small_model(
        window_days=max(7, SMALL_MODEL_AUTO_TRAIN_WINDOW_DAYS),
        min_samples=max(20, SMALL_MODEL_AUTO_TRAIN_MIN_SAMPLES),
    )
    if result.get("ok"):
        WORKER_STARTUP_STATUS["smallModelReady"] = True
        WORKER_STARTUP_STATUS["smallModelReason"] = (
            f"Small model trained ({trigger}): {result.get('sampleCount', 0)} samples"
        )
        logger.info(
            "[SMALL_MODEL][%s] trained samples=%s path=%s",
            trigger, result.get("sampleCount", 0), result.get("modelPath", str(SMALL_MODEL_FILE))
        )
    else:
        logger.info("[SMALL_MODEL][%s] skipped: %s", trigger, result.get("reason", "unknown"))
    return result

def _learning_stats() -> Dict[str, Any]:
    data = _load_reinforcement_state()
    roles = data.get("roles", {}) if isinstance(data, dict) else {}
    now_ts = int(time.time())
    windows = {"week": 7, "month": 30, "year": 365}
    stats: Dict[str, Any] = {}
    for label, days in windows.items():
        cutoff = now_ts - days * 86400
        count = 0
        score_sum = 0.0
        for role_node in roles.values():
            if not isinstance(role_node, dict):
                continue
            events = role_node.get("events", [])
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if _safe_int(ev.get("ts"), 0) < cutoff:
                    continue
                count += 1
                score_sum += _safe_float(ev.get("score"), 0.0)
        stats[label] = {
            "days": days,
            "samples": count,
            "avgScore": round(score_sum / max(1, count), 2),
        }
    return {
        "ok": True,
        "stats": stats,
        "knowledgeScore": _learning_knowledge_score(),
        "policy": _adaptive_learning_policy(),
        "policyPath": str(LEARNING_POLICY_FILE),
        "smallModelPath": str(SMALL_MODEL_FILE),
    }

def _candidate_key(payload: Any, role_key: str) -> str:
    cid = _first_non_blank(
        getattr(payload, "candidateId", ""),
        getattr(payload, "applicationId", ""),
        getattr(payload, "interviewId", ""),
        getattr(payload, "sessionToken", ""),
    )
    if cid:
        return cid
    cp = getattr(payload, "candidateProfile", {}) if hasattr(payload, "candidateProfile") else {}
    if isinstance(cp, dict):
        derived = _first_non_blank(cp.get("id", ""), cp.get("email", ""), cp.get("name", ""))
        if derived:
            return str(derived).strip()
    if STRICT_CANDIDATE_IDENTITY:
        return ""
    return f"anon-{role_key}-{int(time.time())}"

def _touch_activity() -> None:
    global LAST_ACTIVITY_TS
    LAST_ACTIVITY_TS = time.time()

def _mark_interview_session_active_by_key(session_key: str, ttl_seconds: int = INTERVIEW_SESSION_TTL_SECONDS) -> str:
    global ACTIVE_INTERVIEW_SESSIONS
    key = str(session_key or "").strip()
    if not key:
        key = f"session-anon-{int(time.time())}"
    ACTIVE_INTERVIEW_SESSIONS[key] = int(time.time()) + max(120, int(ttl_seconds))
    return key

def _extend_interview_session(session_key: str, ttl_seconds: int = INTERVIEW_SESSION_TTL_SECONDS) -> None:
    key = str(session_key or "").strip()
    if not key:
        return
    ACTIVE_INTERVIEW_SESSIONS[key] = int(time.time()) + max(120, int(ttl_seconds))

def _mark_interview_session_completed_by_key(session_key: str) -> None:
    global ACTIVE_INTERVIEW_SESSIONS
    key = str(session_key or "").strip()
    if key:
        ACTIVE_INTERVIEW_SESSIONS.pop(key, None)

def _mark_interview_session_active(payload: Any, role_key: str, ttl_seconds: int = INTERVIEW_SESSION_TTL_SECONDS) -> str:
    key = _candidate_key(payload, role_key).strip()
    if not key:
        key = f"session-{role_key}-{int(time.time())}"
    return _mark_interview_session_active_by_key(key, ttl_seconds)

def _mark_interview_session_completed(payload: Any, role_key: str = "") -> None:
    key = _candidate_key(payload, role_key).strip()
    _mark_interview_session_completed_by_key(key)

def _has_active_interview_session() -> bool:
    global ACTIVE_INTERVIEW_SESSIONS
    now_ts = int(time.time())
    alive: Dict[str, int] = {}
    for k, expiry in ACTIVE_INTERVIEW_SESSIONS.items():
        if _safe_int(expiry, 0) > now_ts:
            alive[k] = _safe_int(expiry, 0)
    ACTIVE_INTERVIEW_SESSIONS = alive
    return len(ACTIVE_INTERVIEW_SESSIONS) > 0

def _extract_questions_from_lines(lines: List[str], count: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for line in lines:
        chunks = _split_numbered_block(line) if re.search(r"\b\d+[\).]\s+", line) else [line]
        for chunk in chunks:
            q = _normalize_question(chunk)
            if not q or not q.endswith("?"):
                continue
            if not (18 <= len(q) <= 220):
                continue
            key = _normalize(q)
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
            if len(out) >= count:
                return out
    return out

def _extract_search_result_url(href: str) -> str:
    raw = str(href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.path.startswith("/l/"):
        q = parse_qs(parsed.query).get("uddg", [])
        if q:
            return unquote(q[0])
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return ""

def _provider_search_url(provider: str, query: str, offset: int) -> str:
    q = quote_plus(query)
    if provider == "duckduckgo":
        return f"https://duckduckgo.com/html/?q={q}&s={offset}"
    if provider == "bing":
        first = max(1, offset + 1)
        return f"https://www.bing.com/search?q={q}&first={first}"
    if provider == "brave":
        page = max(1, (offset // 10) + 1)
        return f"https://search.brave.com/search?q={q}&page={page}"
    return ""

def _extract_links_from_search_html(provider: str, html_text: str) -> List[str]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        if not href:
            continue
        resolved = _extract_search_result_url(href)
        if not resolved and href.startswith("/url?q="):
            try:
                qv = parse_qs(urlparse(href).query).get("q", [])
                resolved = unquote(qv[0]) if qv else ""
            except Exception:
                resolved = ""
        if not resolved:
            continue
        # Filter out same-search-engine internal links.
        host = (urlparse(resolved).netloc or "").lower()
        if provider == "duckduckgo" and "duckduckgo.com" in host:
            continue
        if provider == "bing" and "bing.com" in host:
            continue
        if provider == "brave" and "search.brave.com" in host:
            continue
        links.append(resolved)
    return links

def _search_provider_query(provider: str, query: str, offset: int) -> tuple[int, List[str]]:
    if not requests:
        return 0, []
    url = _provider_search_url(provider, query, offset)
    if not url:
        return 0, []
    resp = requests.get(
        url,
        timeout=max(3, RAG_FETCH_TIMEOUT + 2),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 InterviewRAGWorker/5.0"
        },
    )
    if resp.status_code >= 400:
        return resp.status_code, []
    return resp.status_code, _extract_links_from_search_html(provider, resp.text)

def _provider_priority(providers: List[str]) -> List[str]:
    ranked: List[tuple[float, str]] = []
    for p in providers:
        stat = SEARCH_PROVIDER_STATS.get(p, {"success": 0, "fail": 0})
        success = int(stat.get("success", 0))
        fail = int(stat.get("fail", 0))
        score = (success + 1.0) / (success + fail + 2.0)
        ranked.append((score, p))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in ranked]

def _build_discovery_queries(domain: str, role: str = "", department: str = "", stream: str = "") -> List[str]:
    dom = _first_non_blank(domain, "general").strip()
    rl = _first_non_blank(role, dom).strip()
    dep = _first_non_blank(department, "").strip()
    stm = _first_non_blank(stream, dom).strip()
    aliases: Dict[str, List[str]] = {
        "business_development": ["business development", "business development executive", "business development manager", "bde", "bdm"],
        "voice_process": ["voice process", "call center", "customer support voice", "bpo voice", "telecalling"],
        "chat_process": ["chat process", "chat support", "customer support chat", "non voice process", "email support"],
        "sales": ["sales executive", "sales representative", "inside sales", "outside sales"],
    }
    alias_terms = aliases.get(_normalize(dom).replace(" ", "_"), [dom])
    base = [
        f"{term} interview questions"
        for term in alias_terms
    ]
    base += [
        f"{dom} interview questions and answers",
        f"{dom} scenario based interview questions",
        f"{dom} project based interview questions",
        f"{dom} personal interview questions",
        f"{dom} hr interview questions",
        f"{dom} technical interview questions",
        f"{dom} behavioral interview questions",
        f"{dom} experienced interview questions",
        f"{dom} fresher interview questions",
        f"{dom} role play interview questions",
    ]
    contextual = [
        f"{rl} interview questions",
        f"{rl} project interview questions",
        f"{rl} personal and hr interview questions",
        f"{stm} stream interview questions",
    ]
    if dep:
        contextual.extend([
            f"{dep} department interview questions",
            f"{dom} {dep} interview questions",
        ])
    queries = [q.strip() for q in [*base, *contextual] if q.strip()]
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    deduped: List[str] = []
    for q in queries:
        key = _normalize(q)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped

def _extract_candidate_links_from_page(page_url: str, domain: str, limit: int = 40) -> List[str]:
    if not requests or not BeautifulSoup:
        return []
    try:
        resp = requests.get(page_url, timeout=max(3, RAG_FETCH_TIMEOUT + 2), headers={"User-Agent": "InterviewRAGWorker/5.0"})
        if resp.status_code >= 400:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    allowed = ("interview", "question", "questions", "hr", "behavioral", "technical", "project")
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        if not href:
            continue
        if href.startswith("/"):
            base = urlparse(page_url)
            href = f"{base.scheme}://{base.netloc}{href}"
        resolved = _extract_search_result_url(href)
        if not resolved:
            continue
        text = f"{resolved} {_normalize(a.get_text(' ', strip=True))}"
        if domain and domain.lower() not in text and not any(k in text for k in allowed):
            continue
        key = _normalize(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
        if len(out) >= max(5, min(limit, 120)):
            break
    return out

def _discover_market_urls(domain: str, limit: int = 12, role: str = "", department: str = "", stream: str = "") -> List[str]:
    global LAST_DISCOVERY_TS
    if not requests or not BeautifulSoup:
        logger.info("[TRAIN][%s] discover.skip missing requests/bs4", domain)
        return []
    lim = max(2, min(limit, 240))
    query_set = _build_discovery_queries(domain, role, department, stream)
    blocked_hosts = ("youtube.com", "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com")
    urls: List[str] = []
    seen: Set[str] = set()
    provider_blocked: Dict[str, bool] = {p: False for p in SEARCH_ENGINES}
    providers = _provider_priority(SEARCH_ENGINES or ["duckduckgo", "bing", "brave"])
    logger.info("[TRAIN][%s] discover.providers %s", domain, providers)

    for query in query_set:
        if len(urls) >= lim:
            break
        if DISCOVERY_COOLDOWN_SECONDS > 0:
            last_ts = float(LAST_DISCOVERY_TS.get(domain, 0.0))
            wait_for = DISCOVERY_COOLDOWN_SECONDS - (time.time() - last_ts)
            if wait_for > 0:
                logger.info("[TRAIN][%s] discover.cooldown_wait %.1fs before next search", domain, wait_for)
                time.sleep(wait_for)
        logger.info("[TRAIN][%s] discover.query %s", domain, query)
        query_found = 0
        for provider in providers:
            if provider_blocked.get(provider, False):
                logger.info("[TRAIN][%s] discover.provider_skip blocked=%s", domain, provider)
                continue
            try:
                logger.info("[TRAIN][%s] discover.provider_try %s", domain, provider)
                for offset in (0, 30):
                    if len(urls) >= lim:
                        break
                    status, links = _search_provider_query(provider, query, offset)
                    LAST_DISCOVERY_TS[domain] = time.time()
                    if status >= 400:
                        logger.info("[TRAIN][%s] discover.%s_http_%s %s", domain, provider, status, query)
                        stat = SEARCH_PROVIDER_STATS.setdefault(provider, {"success": 0, "fail": 0})
                        stat["fail"] = int(stat.get("fail", 0)) + 1
                        if status in SEARCH_BLOCK_STATUS_CODES:
                            provider_blocked[provider] = True
                            logger.info("[TRAIN][%s] discover.provider_blocked %s -> shifting engine", domain, provider)
                        continue
                    for resolved in links:
                        host = (urlparse(resolved).netloc or "").lower()
                        if any(b in host for b in blocked_hosts):
                            _record_source_quality(resolved, domain, False, "blocked_host")
                            continue
                        ok_url, rsn = _is_url_allowed(resolved)
                        if not ok_url:
                            _record_source_quality(resolved, domain, False, rsn)
                            continue
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        urls.append(resolved)
                        _record_source_quality(resolved, domain, True, f"search_{provider}")
                        query_found += 1
                        if len(urls) >= lim:
                            break
                if query_found > 0:
                    stat = SEARCH_PROVIDER_STATS.setdefault(provider, {"success": 0, "fail": 0})
                    stat["success"] = int(stat.get("success", 0)) + 1
                    logger.info("[TRAIN][%s] discover.provider_ok %s found=%s", domain, provider, query_found)
                    break
            except Exception:
                stat = SEARCH_PROVIDER_STATS.setdefault(provider, {"success": 0, "fail": 0})
                stat["fail"] = int(stat.get("fail", 0)) + 1
                logger.info("[TRAIN][%s] discover.provider_failed %s %s", domain, provider, query)
                continue
        if query_found == 0:
            logger.info("[TRAIN][%s] discover.query_no_results %s", domain, query)
    # Expand by visiting discovered pages and collecting interview-related links from them.
    expanded: List[str] = []
    for seed in urls[: max(8, lim // 2)]:
        children = _extract_candidate_links_from_page(seed, domain=domain, limit=max(6, lim))
        for child in children:
            host = (urlparse(child).netloc or "").lower()
            if any(b in host for b in blocked_hosts):
                _record_source_quality(child, domain, False, "blocked_host")
                continue
            ok_child, child_reason = _is_url_allowed(child)
            if not ok_child:
                _record_source_quality(child, domain, False, child_reason)
                continue
            key = _normalize(child)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(child)
            _record_source_quality(child, domain, True, "expanded_link")
            if len(urls) + len(expanded) >= lim:
                break
        if len(urls) + len(expanded) >= lim:
            break
    if expanded:
        logger.info("[TRAIN][%s] discover.expand extra_urls=%s", domain, len(expanded))
    urls.extend(expanded)
    if not urls:
        # Final fallback: use curated role sources if all search engines are blocked/unusable.
        fallback_sources = (_default_market_sources().get(_normalize(domain).replace(" ", "_"), {}) or {}).get("sources", [])
        if fallback_sources:
            logger.info("[TRAIN][%s] discover.fallback_curated sources=%s", domain, len(fallback_sources))
            urls.extend([str(u).strip() for u in fallback_sources if str(u).strip()])
            extra_from_curated: List[str] = []
            for src in urls[: min(10, len(urls))]:
                extra_from_curated.extend(_extract_candidate_links_from_page(src, domain=domain, limit=20))
            if extra_from_curated:
                logger.info("[TRAIN][%s] discover.fallback_expand extra_urls=%s", domain, len(extra_from_curated))
                urls.extend(extra_from_curated)
    logger.info("[TRAIN][%s] discover.done urls=%s", domain, len(urls[:lim]))
    return urls[:lim]

def _stored_questions_for_role(role_key: str, count: int = INTERVIEW_QUESTION_COUNT) -> List[str]:
    model_questions = _small_model_question_bank(role_key, count=count)
    if model_questions:
        return model_questions[:count]
    if _dataset_only_role(role_key):
        return []
    global TRAINING_STORE_MAP
    domains = TRAINING_STORE_MAP.get("domains", {}) if isinstance(TRAINING_STORE_MAP, dict) else {}
    node = domains.get(role_key, {}) if isinstance(domains, dict) else {}
    questions = node.get("questions", []) if isinstance(node, dict) else []
    cleaned: List[str] = []
    seen: Set[str] = set()
    for q in questions:
        text = _normalize_question(q)
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= count:
            break
    return cleaned

def _stored_qa_bank_for_role(role_key: str) -> List[Dict[str, str]]:
    model_qa = _small_model_qa_bank(role_key)
    if model_qa:
        return model_qa
    if _dataset_only_role(role_key):
        return []
    global TRAINING_STORE_MAP
    domains = TRAINING_STORE_MAP.get("domains", {}) if isinstance(TRAINING_STORE_MAP, dict) else {}
    node = domains.get(role_key, {}) if isinstance(domains, dict) else {}
    qa = node.get("qaBank", []) if isinstance(node, dict) else []
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    if isinstance(qa, list):
        for item in qa:
            if not isinstance(item, dict):
                continue
            q = _normalize_question(item.get("question", ""))
            if not q:
                continue
            ans = str(item.get("answer", "")).strip()
            if not _is_valid_qa_pair(q, ans):
                continue
            key = _normalize(q)
            if key in seen:
                continue
            seen.add(key)
            out.append({"question": q, "answer": ans})
    return out

def _offline_qa_memory_refs(role_key: str, question_text: str, limit: int = 16) -> List[str]:
    global TRAINING_STORE_MAP
    out: List[str] = []
    if not isinstance(TRAINING_STORE_MAP, dict):
        return out
    domains = TRAINING_STORE_MAP.get("domains", {}) if isinstance(TRAINING_STORE_MAP.get("domains", {}), dict) else {}
    node = domains.get(role_key, {}) if isinstance(domains, dict) else {}
    qa_memory = node.get("qaMemory", {}) if isinstance(node, dict) else {}
    if not isinstance(qa_memory, dict):
        return out
    q_key = _normalize(_normalize_question(question_text) or question_text)
    rec = qa_memory.get(q_key, {})
    if not isinstance(rec, dict):
        return out
    ideal = str(rec.get("idealAnswer", "")).strip()
    if ideal:
        out.append(ideal)
    samples = rec.get("samples", [])
    if isinstance(samples, list):
        ranked_samples = sorted(
            [s for s in samples if isinstance(s, dict)],
            key=lambda x: _safe_int(x.get("score"), 0),
            reverse=True,
        )
        for s in ranked_samples:
            a = str(s.get("answer", "")).strip()
            if a:
                out.append(_limit_words(a, 140))
            if len(out) >= max(4, limit):
                break
    return out[: max(4, limit)]

def _learn_offline_qa_memory(
    role_key: str,
    question_text: str,
    ideal_answer: str,
    candidate_answer: str,
    score: int,
    metrics: Dict[str, int],
    sentence_checks: List[Dict[str, Any]],
) -> None:
    global TRAINING_STORE_MAP
    try:
        store = _load_training_store()
        domains = store.setdefault("domains", {})
        if not isinstance(domains, dict):
            domains = {}
            store["domains"] = domains
        node = domains.setdefault(role_key, {})
        if not isinstance(node, dict):
            node = {}
            domains[role_key] = node
        qa_memory = node.setdefault("qaMemory", {})
        if not isinstance(qa_memory, dict):
            qa_memory = {}
            node["qaMemory"] = qa_memory
        q_norm = _normalize_question(question_text) or str(question_text or "").strip()
        q_key = _normalize(q_norm)
        if not q_key:
            return
        rec = qa_memory.setdefault(q_key, {})
        if not isinstance(rec, dict):
            rec = {}
            qa_memory[q_key] = rec
        rec["question"] = q_norm
        rec["idealAnswer"] = str(ideal_answer or "").strip()
        rec["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        samples = rec.get("samples", [])
        if not isinstance(samples, list):
            samples = []
        samples.append({
            "ts": int(time.time()),
            "answer": _limit_words(candidate_answer, 180),
            "score": _clamp(_safe_int(score, 0)),
            "metrics": {
                "overall": _clamp(_safe_int(metrics.get("overall"), 0)),
                "work": _clamp(_safe_int(metrics.get("work"), 0)),
                "fluency": _clamp(_safe_int(metrics.get("fluency"), 0)),
                "relevance": _clamp(_safe_int(metrics.get("relevance"), 0)),
                "confidence": _clamp(_safe_int(metrics.get("confidence"), 0)),
            },
            "sentenceChecks": sentence_checks[:8] if isinstance(sentence_checks, list) else [],
        })
        samples = sorted(
            [s for s in samples if isinstance(s, dict)],
            key=lambda x: (_safe_int(x.get("score"), 0), _safe_int(x.get("ts"), 0)),
            reverse=True,
        )[:60]
        rec["samples"] = samples
        qa_memory[q_key] = rec
        node["qaMemory"] = qa_memory
        domains[role_key] = node
        store["domains"] = domains
        _save_training_store(store)
        TRAINING_STORE_MAP = store
    except Exception:
        logger.exception("[ML][offline] qa_memory_update_failed role=%s", role_key)

def _refresh_role_qa_bank(role_key: str, target_count: int = 50) -> int:
    global TRAINING_STORE_MAP
    target = max(10, min(target_count, 80))
    logger.info("[LEARN][%s] qa_refresh_start target=%s", role_key, target)
    req = OneTimeTrainingRequest(
        domains=[role_key],
        includeDefaults=True,
        discoverUrls=True,
        forceRefresh=False,
        discoveredUrlLimitPerDomain=max(AUTO_DISCOVER_PER_DOMAIN, 80),
        questionCountPerDomain=target,
        role=f"{role_key} role",
        department="interview",
        stream=role_key,
        runLabel=f"qa-refresh-{role_key}",
    )
    rag_train_once(req)
    store = _load_training_store()
    domains = store.get("domains", {})
    node = domains.get(role_key, {}) if isinstance(domains, dict) else {}
    questions = node.get("questions", []) if isinstance(node, dict) else []
    sources = node.get("sources", []) if isinstance(node, dict) else []
    context_lines: List[str] = []
    for src in sources[: max(12, RAG_MAX_URLS)]:
        context_lines.extend(_fetch_url_cached_only(str(src)) or _fetch_url(str(src)))
    qa_bank: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for q_raw in questions[:target]:
        q = _normalize_question(q_raw)
        if not q:
            continue
        key = _normalize(q)
        if key in seen:
            continue
        seen.add(key)
        ans = _build_best_answer(q, context_lines, BestAnswerRequest(domain=role_key)).get("bestAnswer", "")
        ans = _limit_words(ans, 180)
        if _is_valid_qa_pair(q, ans):
            qa_bank.append({"question": q, "answer": ans})
    if isinstance(domains, dict):
        node["qaBank"] = qa_bank
        node["qaBankUpdatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node["questions"] = [item["question"] for item in qa_bank]
        domains[role_key] = node
        store["domains"] = domains
        _save_training_store(store)
        TRAINING_STORE_MAP = store
    logger.info("[LEARN][%s] qa_refresh_done count=%s", role_key, len(qa_bank))
    return len(qa_bank)

def _trigger_role_qa_refresh_async(role_key: str, target_count: int = 50, min_gap_seconds: int = 600) -> bool:
    """Fire-and-forget QA refresh so interview start path stays low-latency."""
    global QA_REFRESH_STATE
    now = time.time()
    state = QA_REFRESH_STATE.get(role_key, {}) if isinstance(QA_REFRESH_STATE.get(role_key), dict) else {}
    in_progress = bool(state.get("inProgress"))
    last_start = _safe_float(state.get("lastStartTs"), 0.0)
    if in_progress:
        logger.info("[LEARN][%s] qa_refresh_skip in_progress", role_key)
        return False
    if now - last_start < max(60, int(min_gap_seconds)):
        logger.info("[LEARN][%s] qa_refresh_skip cooldown=%ss", role_key, int(max(60, int(min_gap_seconds)) - (now - last_start)))
        return False

    QA_REFRESH_STATE[role_key] = {
        "inProgress": True,
        "lastStartTs": now,
        "lastStatus": "started",
        "lastTarget": int(target_count),
    }

    def _runner() -> None:
        global QA_REFRESH_STATE
        try:
            count = _refresh_role_qa_bank(role_key, target_count=target_count)
            QA_REFRESH_STATE[role_key] = {
                "inProgress": False,
                "lastStartTs": now,
                "lastDoneTs": time.time(),
                "lastStatus": "done",
                "lastCount": int(count),
                "lastTarget": int(target_count),
            }
        except Exception as ex:
            logger.exception("[LEARN][%s] qa_refresh_async_failed: %s", role_key, ex)
            QA_REFRESH_STATE[role_key] = {
                "inProgress": False,
                "lastStartTs": now,
                "lastDoneTs": time.time(),
                "lastStatus": f"failed:{type(ex).__name__}",
                "lastTarget": int(target_count),
            }

    threading.Thread(target=_runner, name=f"qa-refresh-{role_key}", daemon=True).start()
    logger.info("[LEARN][%s] qa_refresh_async_started target=%s", role_key, target_count)
    return True

def _select_unique_questions_for_candidate(role_key: str, candidate_key: str, pool: List[str], target: int) -> List[str]:
    if not candidate_key.strip():
        logger.warning("[QUESTION][%s] missing candidate identity; using randomized fallback selection", role_key)
        uniq: List[str] = []
        seen: Set[str] = set()
        for q in pool:
            nq = _normalize_question(q)
            k = _normalize(nq)
            if not nq or k in seen:
                continue
            seen.add(k)
            uniq.append(nq)
        if len(uniq) > 1:
            random.SystemRandom().shuffle(uniq)
        return uniq[:target]
    global QUESTION_STATE_MAP
    QUESTION_STATE_MAP = _load_candidate_state()
    by_role = QUESTION_STATE_MAP.setdefault("byRole", {})
    role_bucket = by_role.setdefault(role_key, {})
    cand = role_bucket.setdefault(candidate_key, {"asked": [], "lastUpdated": ""})
    asked_raw = [str(x).strip() for x in cand.get("asked", []) if str(x).strip()]
    unique_pool: List[str] = []
    seen: Set[str] = set()
    for q in pool:
        nq = _normalize_question(q)
        if not nq:
            continue
        k = _normalize(nq)
        if k in seen:
            continue
        seen.add(k)
        unique_pool.append(nq)
    # Avoid deterministic same-order picks across interviews.
    if len(unique_pool) > 1:
        random.SystemRandom().shuffle(unique_pool)

    valid_keys = {_normalize(q) for q in unique_pool}
    asked = {item for item in asked_raw if item in valid_keys}
    fresh = [q for q in unique_pool if _normalize(q) not in asked]
    if len(fresh) >= target:
        selected = fresh[:target]
    else:
        # Pool exhausted: start async refresh, but never block interview start.
        logger.info("[LEARN][%s][%s] pool_exhausted asked=%s pool=%s -> refresh_qa_50_async", role_key, candidate_key, len(asked), len(unique_pool))
        _trigger_role_qa_refresh_async(role_key, target_count=50)
        refreshed_pool = [item["question"] for item in _stored_qa_bank_for_role(role_key)] or _stored_questions_for_role(role_key, 80)
        if refreshed_pool:
            unique_pool = refreshed_pool
            fresh = [q for q in unique_pool if _normalize(q) not in asked]
        selected = fresh[:target]
        if len(selected) < target:
            # If all lists are completed, allow repeat in round-robin order.
            repeats = [q for q in unique_pool if q not in selected]
            selected.extend(repeats[: max(0, target - len(selected))])

    new_asked = list(asked)
    for q in selected:
        k = _normalize(q)
        if k not in new_asked:
            new_asked.append(k)
    cand["asked"] = new_asked
    cand["lastUpdated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    role_bucket[candidate_key] = cand
    by_role[role_key] = role_bucket
    QUESTION_STATE_MAP["byRole"] = by_role
    _save_candidate_state(QUESTION_STATE_MAP)
    logger.info("[LEARN][%s][%s] unique_selected=%s asked_total=%s", role_key, candidate_key, len(selected), len(new_asked))
    return selected[:target]

def _infer_role_key(job_title: str, domain_label: str, job_desc: str, department: str = "") -> str:
    hay = " ".join([_normalize(job_title), _normalize(domain_label), _normalize(job_desc), _normalize(department)])
    scored: List[tuple[int, str]] = []
    for key, cfg in RAG_SOURCE_MAP.items():
        if key == "default": continue
        kws = [str(k).lower() for k in cfg.get("keywords",[])] or [p for p in key.split("_") if p]
        score = sum(1 for k in kws if k and k in hay)
        if score > 0: scored.append((score, key))
    model_domains = (_load_small_model() or {}).get("domainKnowledge", {})
    if isinstance(model_domains, dict):
        for key in model_domains.keys():
            norm_key = _normalize_label_key(str(key), fallback="")
            if not norm_key:
                continue
            aliases = [norm_key, norm_key.replace("_", " ")]
            score = sum(2 for alias in aliases if alias and alias in hay)
            if score > 0:
                scored.append((score, norm_key))
    if not scored:
        if _matches_domain_keywords(hay, _domain_training_keywords("java")):
            return "java"
        auto_label = _auto_domain_label(hay, fallback="")
        if auto_label:
            return auto_label
        return "default"
    return sorted(scored, key=lambda p: (-p[0], p[1]))[0][1]

def _collect_urls(payload: Any, role_key: str) -> List[str]:
    cfg = RAG_SOURCE_MAP.get(role_key, {})
    default_cfg = RAG_SOURCE_MAP.get("default", {})
    urls: List[str] = []
    for attr in ("sources","interviewSourceUrls","ragSourceUrls"):
        urls.extend(s.strip() for s in getattr(payload, attr, []) if str(s).strip())
    for attr in ("jdSourceUrl","onlineInterviewLink"):
        v = _first_non_blank(getattr(payload, attr, ""))
        if v: urls.append(v)
    urls.extend(s.strip() for s in cfg.get("sources",[]) if str(s).strip())
    urls.extend(s.strip() for s in default_cfg.get("sources",[]) if str(s).strip())
    seen: Set[str] = set(); deduped: List[str] = []
    for u in urls:
        ok, reason = _is_url_allowed(u)
        if not ok:
            _record_source_quality(u, role_key, False, reason)
            _append_learning_log("url_blocked", {"url": u, "role": role_key, "reason": reason})
            continue
        if u in seen: continue
        seen.add(u); deduped.append(u)
    return deduped[:max(4, RAG_MAX_URLS)]

def _fetch_url(url: str) -> List[str]:
    if not RAG_FETCH_ONLINE or not requests or not BeautifulSoup: return []
    if not url.startswith(("http://","https://")): return []
    ok, reason = _is_url_allowed(url)
    if not ok:
        _record_source_quality(url, "", False, reason)
        _append_learning_log("url_blocked", {"url": url, "reason": reason})
        _mark_learning_url(url, "", False, reason)
        return []
    if url in URL_CONTENT_CACHE: return URL_CONTENT_CACHE[url]  # serve from memory cache, no network hit
    vector_cached = _vector_cache_retrieve(url, query="", k=URL_TEXT_MEMORY_CAP)
    if vector_cached:
        URL_CONTENT_CACHE[url] = vector_cached[:URL_TEXT_MEMORY_CAP]
        _record_source_quality(url, "", True, "vector_cache_hit")
        _append_learning_log("url_pass", {"url": url, "reason": "vector_cache_hit", "lines": len(URL_CONTENT_CACHE[url])})
        _mark_learning_url(url, "", True, "vector_cache_hit")
        return URL_CONTENT_CACHE[url]
    lines: List[str] = []
    try:
        resp = requests.get(url, timeout=RAG_FETCH_TIMEOUT, headers={"User-Agent":"InterviewRAGWorker/5.0"})
        if resp.status_code >= 400:
            URL_CONTENT_CACHE[url] = []
            _record_source_quality(url, "", False, f"http_{resp.status_code}")
            _append_learning_log("url_fail", {"url": url, "reason": f"http_{resp.status_code}"})
            _mark_learning_url(url, "", False, f"http_{resp.status_code}")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        for node in soup.find_all(["h1","h2","h3","p","li"]):
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
            if len(text) < 45: continue
            lines.append(text)
            if len(lines) >= 80: break
    except Exception:
        lines = []
        _record_source_quality(url, "", False, "fetch_exception")
        _append_learning_log("url_fail", {"url": url, "reason": "fetch_exception"})
        _mark_learning_url(url, "", False, "fetch_exception")
    if lines:
        lines = _sanitize_cache_lines(lines, max_items=120)
        _store_vector_cache(url, lines)
        _record_source_quality(url, "", True, "fetched")
        _append_learning_log("url_pass", {"url": url, "reason": "fetched", "lines": len(lines)})
        _mark_learning_url(url, "", True, "fetched")
    URL_CONTENT_CACHE[url] = lines[:URL_TEXT_MEMORY_CAP]
    if URL_CONTENT_CACHE[url]:
        _refresh_url_index()
    return URL_CONTENT_CACHE[url]

def _fetch_url_cached_only(url: str, query: str = "") -> List[str]:
    """Return cached content only — never makes a network request."""
    mem = URL_CONTENT_CACHE.get(url, [])
    if mem:
        return mem
    vec = _vector_cache_retrieve(url, query=query, k=VECTOR_TOP_K)
    if vec:
        URL_CONTENT_CACHE[url] = vec[:URL_TEXT_MEMORY_CAP]
        return vec
    return []

def _refresh_url_index() -> None:
    global URL_CONTENT_INDEX
    index: Dict[str, List[str]] = {}
    for url, lines in URL_CONTENT_CACHE.items():
        for line in lines:
            for kw in _keywords(line, top_n=6):
                bucket = index.setdefault(kw, [])
                if line not in bucket:
                    bucket.append(line)
    URL_CONTENT_INDEX = index

def _prefetch_rag_cache() -> None:
    if not RAG_PREFETCH_ON_STARTUP or not RAG_FETCH_ONLINE:
        return
    started = time.perf_counter()
    urls: List[str] = []
    for cfg in RAG_SOURCE_MAP.values():
        if not isinstance(cfg, dict):
            continue
        urls.extend(str(u).strip() for u in cfg.get("sources", []) if str(u).strip())
    seen: Set[str] = set()
    unique_urls: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        unique_urls.append(u)
    for url in unique_urls[: max(RAG_MAX_URLS * 8, 8)]:
        _fetch_url(url)
    _refresh_url_index()
    elapsed = round((time.perf_counter() - started) * 1000)
    logger.info("RAG prefetch complete in %sms (urls=%s, cached=%s)", elapsed, len(unique_urls), len(URL_CONTENT_CACHE))

def _sample_rag_lines(role_key: str, question_texts: List[str], count: int = 20) -> List[str]:
    cfg = RAG_SOURCE_MAP.get(role_key, {})
    hints = [str(k).lower().strip() for k in cfg.get("keywords", []) if str(k).strip()]
    for q in question_texts[:5]:
        hints.extend(_keywords(q, top_n=4))
    picked: List[str] = []
    seen: Set[str] = set()
    for kw in hints:
        for line in URL_CONTENT_INDEX.get(kw, []):
            short = _limit_text_words(line, LLM_LINE_WORD_LIMIT)
            key = _normalize(short)
            if not key or key in seen:
                continue
            seen.add(key)
            picked.append(short)
            if len(picked) >= count:
                return picked
    all_lines = [_limit_text_words(line, LLM_LINE_WORD_LIMIT) for lines in URL_CONTENT_CACHE.values() for line in lines]
    random.shuffle(all_lines)
    for line in all_lines:
        key = _normalize(line)
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(line)
        if len(picked) >= count:
            break
    return picked[:count]

def _looks_like_question_dump_line(text: str) -> bool:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return True
    lo = t.lower()
    if lo.count("?") >= 1:
        return True
    if "model answer" in lo or "interview questions" in lo or "with answers" in lo:
        return True
    if re.search(r"(?:^|\s)\d+\.\s+[A-Za-z]", t):
        return True
    if re.search(r"^[0-9#*_\-\s\W]{0,6}(what|how|why|when|where|which|who|difference between)\b", lo):
        return True
    return False

def _filter_answer_context_lines(lines: List[str], max_items: int = 80) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in lines:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        if _looks_like_question_dump_line(text):
            continue
        key = _normalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out

def _build_snippets(payload: Any) -> List[str]:
    snippets: List[str] = []
    query_hint = " ".join([
        _first_non_blank(getattr(payload, "jobTitle", ""), ""),
        _first_non_blank(getattr(payload, "domainLabel", ""), getattr(payload, "domain", "")),
        _first_non_blank(getattr(payload, "department", ""), ""),
        _first_non_blank(getattr(payload, "jobDescription", ""), ""),
    ]).strip()
    role_text = _first_non_blank(
        getattr(payload, "jobTitle", ""),
        getattr(payload, "domainLabel", ""),
        getattr(payload, "domain", ""),
    )
    dept_text = _first_non_blank(getattr(payload, "department", ""), "engineering")
    stream_text = _first_non_blank(getattr(payload, "domain", ""), getattr(payload, "language", ""))
    role_key = _infer_role_key(
        getattr(payload,"jobTitle",""), getattr(payload,"domainLabel","") or getattr(payload,"domain",""),
        getattr(payload,"jobDescription",""), getattr(payload,"department",""),
    )
    urls = _collect_urls(payload, role_key)
    for u in urls:
        # Use cache if available, only hit network if cache is empty for this URL
        cached = _fetch_url_cached_only(u, query=query_hint)
        if cached:
            snippets.extend(cached)
        else:
            snippets.extend(_fetch_url(u))
    # If online context is thin, discover additional links on demand for this role/domain.
    # Disabled by default to keep interview start latency low.
    if len(snippets) < 80 and RAG_FETCH_ONLINE and ON_DEMAND_DISCOVERY_ENABLED:
        discovered = _discover_market_urls(
            _first_non_blank(role_key, stream_text, "general"),
            limit=min(80, AUTO_DISCOVER_PER_DOMAIN),
            role=role_text,
            department=dept_text,
            stream=stream_text,
        )
        for u in discovered:
            if u not in urls:
                urls.append(u)
            cached = _fetch_url_cached_only(u, query=query_hint)
            extra_lines = cached if cached else _fetch_url(u)
            if extra_lines:
                snippets.extend(extra_lines)
        if discovered:
            logger.info(
                "[LEARN][request][%s] on_demand_discovery links=%s snippets=%s",
                role_key, len(discovered), len(snippets)
            )
    jd = getattr(payload, "jobDescription", "").strip()
    if jd: snippets.extend(s.strip() for s in re.split(r"[\n\r]+", jd) if s.strip())
    snippets.extend(_flatten(getattr(payload,"resume",{})))
    snippets.extend(_flatten(getattr(payload,"candidateProfile",{})))
    snippets.extend(_flatten(getattr(payload,"job",{})))
    dept = getattr(payload,"department","").strip()
    if dept: snippets.append(f"Department: {dept}")
    snippets.extend(str(p).strip() for p in getattr(payload,"customPrompts",[]) if str(p).strip())
    return snippets, role_key, urls

def _merge_source_map_with_market_defaults(source_map: Dict[str, Dict[str, Any]], include_defaults: bool) -> Dict[str, Dict[str, Any]]:
    merged = dict(source_map)
    if not include_defaults:
        return merged
    for role, cfg in _default_market_sources().items():
        current = merged.get(role, {"sources": [], "keywords": []})
        src = list(dict.fromkeys([*(current.get("sources", []) or []), *(cfg.get("sources", []) or [])]))
        kws = list(dict.fromkeys([*(current.get("keywords", []) or []), *(cfg.get("keywords", []) or [])]))
        merged[role] = {"sources": src, "keywords": kws}
    merged.setdefault("default", {"sources": [], "keywords": []})
    return merged

def _build_best_answer(question: str, context_lines: List[str], req: BestAnswerRequest) -> Dict[str, Any]:
    normalized_question = _normalize_question(question)
    if not normalized_question:
        return {"question": question, "bestAnswer": "", "keyPoints": [], "score": 0}

    trimmed_context = _filter_answer_context_lines(
        [_limit_text_words(line, 30) for line in context_lines if str(line).strip()],
        max_items=LLM_CONTEXT_LINES,
    )[:LLM_CONTEXT_LINES]
    if ENABLE_LLM:
        prompt = (
            "You are a senior interview coach.\n"
            "Create the single best realistic interview answer.\n"
            "Return STRICT JSON only.\n"
            "Schema: {\"bestAnswer\":\"...\",\"keyPoints\":[\"...\"],\"confidence\":0-100}\n"
            f"Question: {normalized_question}\n"
            f"Domain: {req.domain or 'general'}\n"
            f"Language/stack: {req.language or 'general'}\n"
            f"Job title: {req.jobTitle or 'Candidate'}\n"
            f"Job description: {_limit_text_words(req.jobDescription, 60)}\n"
            f"Context: {json.dumps(trimmed_context, ensure_ascii=False)}"
        )
        parsed = _llm_generate_json(prompt)
        if parsed:
            answer = _limit_words(parsed.get("bestAnswer", ""), max(40, min(req.maxWords, MAX_ANSWER_WORDS)))
            key_points = [str(x).strip() for x in parsed.get("keyPoints", []) if str(x).strip()] if isinstance(parsed.get("keyPoints"), list) else []
            confidence = _clamp(_safe_int(parsed.get("confidence"), 80))
            if answer:
                result = {
                    "question": normalized_question,
                    "bestAnswer": answer,
                    "keyPoints": key_points[:6],
                    "score": confidence,
                }
                if req.candidateAnswer.strip():
                    compare = _llm_score_answer(normalized_question, req.candidateAnswer, trimmed_context)
                    result["candidateComparison"] = compare
                return result

    q_kws = _keywords(normalized_question, top_n=8)
    ranked: List[tuple[float, str]] = []
    for line in trimmed_context:
        ranked.append((_overlap_ratio(line, q_kws), line))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = [line for _, line in ranked[:4] if line]
    fallback_answer = " ".join(selected).strip()
    if not fallback_answer:
        fallback_answer = (
            f"For {req.domain or 'this role'}, I would answer by explaining fundamentals, implementation trade-offs, "
            "a real project example, and measurable outcome."
        )
    fallback_answer = _limit_words(fallback_answer, max(40, min(req.maxWords, MAX_ANSWER_WORDS)))
    result = {
        "question": normalized_question,
        "bestAnswer": fallback_answer,
        "keyPoints": _keywords(fallback_answer, top_n=5),
        "score": _clamp(int(round(_overlap_ratio(fallback_answer, q_kws) * 100))),
    }
    if req.candidateAnswer.strip():
        result["candidateComparison"] = _llm_score_answer(normalized_question, req.candidateAnswer, trimmed_context)
    return result

def _is_reference_answer_usable(answer: str) -> bool:
    text = _limit_words(str(answer or "").strip(), 180)
    if not text or _is_nonsense(text):
        return False
    if _looks_like_question_dump_line(text):
        return False
    return len(re.findall(r"\S+", text)) >= 6

def _best_qa_bank_answer(role_key: str, question: str) -> str:
    q = _normalize_question(question)
    if not q:
        return ""
    qa_bank = _small_model_qa_bank(role_key)
    if not qa_bank:
        return ""
    q_key = _normalize(q)
    for qa in qa_bank:
        stored_q = _normalize_question(qa.get("question", "")) or ""
        stored_a = str(qa.get("answer", "")).strip()
        if stored_q and _normalize(stored_q) == q_key and _is_reference_answer_usable(stored_a):
            return _limit_words(stored_a, 180)
    q_kws = _keywords(q, top_n=10)
    ranked: List[tuple[float, str]] = []
    for qa in qa_bank:
        stored_q = _normalize_question(qa.get("question", "")) or ""
        stored_a = str(qa.get("answer", "")).strip()
        if not stored_q or not _is_reference_answer_usable(stored_a):
            continue
        lex = _overlap_ratio(stored_q, q_kws)
        sem = _semantic_similarity(
            _semantic_context_text(q, q),
            _semantic_context_text(q, stored_q),
        ) if _semantic_ready() else 0.0
        ranked.append((((0.7 * sem) + (0.3 * lex)), stored_a))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked and ranked[0][0] >= 0.12:
        return _limit_words(ranked[0][1], 180)
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENGINE 1: QuestionEngine
# ══════════════════════════════════════════════════════════════════════════════

class QuestionEngine:
    """Generates a ranked list of interview questions from RAG + JD context."""

    def _extract_candidates(self, lines: List[str], count: int) -> List[str]:
        candidates: List[str] = []
        for line in lines:
            t = line.strip()
            if _is_private_line(t): continue
            if t.endswith("?") and 20 <= len(t) <= 220:
                candidates.append(t)
        if len(candidates) < count:
            for line in lines:
                c = line.strip().rstrip(".")
                if _is_private_line(c) or len(c) < 55 or len(c) > 220 or "?" in c: continue
                candidates.append(f"Can you explain: {c}?")
                if len(candidates) >= count: break
        return candidates[:count]

    def _dynamic_topics(self, job_title: str, language: str, lines: List[str], top_n: int = 6) -> List[str]:
        corpus = " ".join([job_title or "", language or ""] + (lines or []))
        ranked = _keywords(corpus, top_n=top_n * 2)
        title_toks = set(_tokenize(job_title)); lang_toks = set(_tokenize(language))
        topics: List[str] = []
        for tok in ranked:
            if tok.isdigit() or tok in STOPWORDS or tok in GENERIC_TOPIC_STOPWORDS: continue
            if tok in title_toks or tok in lang_toks: continue
            topics.append(tok)
            if len(topics) >= top_n: break
        return topics

    def _build_dynamic(self, job_title: str, language: str, snippets: List[str],
                       department: str = "", custom_prompts: Optional[List[str]] = None,
                       experience_years: float = 0.0,
                       count: int = INTERVIEW_QUESTION_COUNT) -> List[str]:
        title = _first_non_blank(job_title, "this role")
        dept  = _first_non_blank(department, "the department")
        lang  = _first_non_blank(language, "technology")
        prompts = custom_prompts or []
        topics  = self._dynamic_topics(job_title, language, snippets + [department], top_n=max(4, count+1))
        core_java = _is_core_java(job_title, " ".join(snippets), language, prompts)
        exp_label = "fresher" if experience_years < 1 else "mid" if experience_years < 5 else "senior"

        templates = (
            [
                "Explain {topic} and where it appears in {title}.",
                "What mistakes do engineers make with {topic}, and how do you avoid them?",
                "Describe a hands-on {topic} experience and what you learned.",
                "Tell me about a real project problem in {topic}, what broke, and how you fixed it.",
                "What trade-offs arise when choosing {topic} implementations?",
                "How do you debug {topic}-related issues in production?",
            ] if core_java else [
                "How would you design {topic} for {title} using {lang}?",
                "Walk through testing {topic} end-to-end in {title} for {dept}.",
                "Describe one real project problem involving {topic} in {title} and your root-cause + fix.",
                "What trade-offs matter most when implementing {topic} in {title}?",
                "Describe a production incident involving {topic} and your resolution.",
                "How would you scale {topic} in a {title} system?",
            ]
        )
        if exp_label == "fresher":
            templates = [
                "Explain the basics of {topic} for a {title} role.",
                "How would you implement {topic} step-by-step using {lang}?",
                "What common beginner mistakes happen in {topic}, and how would you avoid them?",
                "How do you test and validate {topic} in a small project?",
                "Share a simple project example where you used {topic}.",
                "How would you debug a basic issue related to {topic}?",
            ]
        elif exp_label == "senior":
            templates = [
                "How would you design and scale {topic} architecture for {title}?",
                "What production trade-offs have you made around {topic}, and why?",
                "Describe a high-impact incident involving {topic} and your leadership in resolution.",
                "How do you mentor teams on {topic} standards and reviews?",
                "How do you measure reliability/performance outcomes for {topic} in production?",
                "What would you refactor first in a legacy {topic} stack and how would you de-risk it?",
            ]
        questions: List[str] = []; seen: Set[str] = set()

        for p in prompts:
            c = str(p or "").strip()
            if not c: continue
            if not c.endswith("?"): c += "?"
            key = _normalize(c)
            if key in seen: continue
            seen.add(key); questions.append(c)
            if len(questions) >= count: return questions[:count]

        for i, topic in enumerate(topics):
            q = templates[i % len(templates)].format(topic=topic, title=title, lang=lang, dept=dept)
            key = _normalize(q)
            if key in seen: continue
            seen.add(key); questions.append(q)
            if len(questions) >= count: break

        fallbacks = [
            f"Walk me through your core responsibilities as {title}.",
            f"How do you design, implement, and test features for {title}?",
            f"How do you troubleshoot production issues in {title} workflows?",
            f"How do you balance performance and reliability in {title}?",
        ]
        for fb in fallbacks:
            if len(questions) >= count: break
            key = _normalize(fb)
            if key not in seen: seen.add(key); questions.append(fb)

        while len(questions) < count:
            filler = (f"Discuss a real-world {lang} scenario from {title} including trade-offs."
                      if core_java else f"Describe a {lang} challenge you solved in {title}.")
            key = _normalize(filler)
            questions.append(filler if key not in seen else f"Explain your debugging strategy for {title} with {lang}.")
            seen.add(key)
        return questions[:count]

    def _fetch_core_java_from_web(self, count: int) -> List[str]:
        source_lines = _fetch_url(CORE_JAVA_REFERENCE_URL)
        candidates: List[str] = []; seen: Set[str] = set()
        for line in source_lines:
            for chunk in _split_numbered_block(line):
                cleaned = _normalize_question(chunk)
                if not cleaned or not (18 <= len(cleaned) <= 220): continue
                key = _normalize(cleaned)
                if key in seen: continue
                seen.add(key); candidates.append(cleaned)
                if len(candidates) >= count: break
            if len(candidates) >= count: break
        return candidates[:count]

    def build(self, payload: InterviewQuestionsRequest) -> Dict[str, Any]:
        started = time.perf_counter()
        role_key_hint = _infer_role_key(
            payload.jobTitle,
            payload.domainLabel or payload.domain,
            payload.jobDescription,
            payload.department,
        )
        # Fast path only when pool is large enough to keep interview sets diverse.
        stored_pool = _stored_questions_for_role(role_key_hint, max(INTERVIEW_QUESTION_COUNT * 3, 40))
        if _dataset_only_role(role_key_hint) and len(stored_pool) < INTERVIEW_QUESTION_COUNT:
            _ensure_default_small_model_training()
            stored_pool = _stored_questions_for_role(role_key_hint, max(INTERVIEW_QUESTION_COUNT * 3, 40))
            if len(stored_pool) < INTERVIEW_QUESTION_COUNT:
                raise RuntimeError("Java dataset model is not ready yet")
        min_fast_pool = max(INTERVIEW_QUESTION_COUNT * 2, 28)
        if len(stored_pool) >= min_fast_pool:
            candidate_key = _candidate_key(payload, role_key_hint)
            unique_for_candidate = _select_unique_questions_for_candidate(
                role_key=role_key_hint,
                candidate_key=candidate_key,
                pool=stored_pool,
                target=INTERVIEW_QUESTION_COUNT,
            )
            selected_items = [{"question": q, "category": "trained"} for q in (unique_for_candidate or stored_pool[:INTERVIEW_QUESTION_COUNT])]
            if RL_ENABLED and not _dataset_only_role(role_key_hint) and selected_items:
                selected_items = sorted(
                    selected_items,
                    key=lambda item: (
                        _rl_question_value(role_key_hint, str(item.get("question", ""))) +
                        (0.65 * _small_model_predict(role_key_hint, str(item.get("question", "")))) +
                        (0.05 * random.random())  # exploration jitter to avoid same fixed top-N each interview
                    ),
                    reverse=True,
                )[:INTERVIEW_QUESTION_COUNT]
            role_sources = (RAG_SOURCE_MAP.get(role_key_hint, {}) or {}).get("sources", []) if isinstance(RAG_SOURCE_MAP, dict) else []
            questions = [
                {
                    "questionId": f"q{i}",
                    "question": item.get("question", ""),
                    "category": item.get("category", "trained"),
                    "roleKey": role_key_hint,
                    "referenceLinks": role_sources[: max(2, min(8, len(role_sources)))],
                    "department": payload.department or str(payload.job.get("department", "")),
                }
                for i, item in enumerate(selected_items, 1)
            ]
            total_ms = round((time.perf_counter() - started) * 1000)
            logger.info(
                "QuestionEngine fast-path completed in %sms (role=%s, storedPool=%s, selected=%s)",
                total_ms, role_key_hint, len(stored_pool), len(questions)
            )
            return {
                "questions": questions,
                "provider": "interview-worker-cache",
                "model": MODEL_NAME,
                "roleKey": role_key_hint,
                "sources": role_sources[: max(2, min(8, len(role_sources)))],
                "onlineContextCount": 0,
                "reason": "",
            }
        snippets, role_key, urls = _build_snippets(payload)
        after_snippets = time.perf_counter()
        eff_lang = _effective_language(payload.jobTitle, payload.jobDescription, payload.language)
        core_java = _is_core_java(payload.jobTitle, payload.jobDescription or " ".join(snippets),
                                  eff_lang, payload.customPrompts)
        exp_years = _experience_years_from_payload(payload)

        if core_java:
            question_texts = self._fetch_core_java_from_web(INTERVIEW_QUESTION_COUNT)
            if CORE_JAVA_REFERENCE_URL not in urls: urls.append(CORE_JAVA_REFERENCE_URL)
            if not question_texts:
                question_texts = self._build_dynamic(
                    payload.jobTitle, eff_lang, snippets, payload.department, payload.customPrompts, exp_years)
        else:
            question_texts = self._extract_candidates(snippets, INTERVIEW_QUESTION_COUNT)

        # Deduplicate
        seen: Set[str] = set(); deduped: List[str] = []
        for q in question_texts:
            k = _normalize(q)
            if k not in seen: seen.add(k); deduped.append(q)
        question_texts = deduped

        # Use one-time trained store as a secondary source before synthetic top-up.
        trained_questions = _stored_questions_for_role(role_key, INTERVIEW_QUESTION_COUNT)
        for q in trained_questions:
            k = _normalize(q)
            if k in seen:
                continue
            seen.add(k)
            question_texts.append(q)
            if len(question_texts) >= INTERVIEW_QUESTION_COUNT:
                break

        # Top-up with dynamic templates if RAG pool is thin
        if len(question_texts) < INTERVIEW_QUESTION_COUNT:
            extra = self._build_dynamic(
                payload.jobTitle, eff_lang, snippets, payload.department, payload.customPrompts, exp_years)
            for q in extra:
                k = _normalize(q)
                if k not in seen: seen.add(k); question_texts.append(q)
                if len(question_texts) >= INTERVIEW_QUESTION_COUNT: break

        rag_focus = _sample_rag_lines(role_key, question_texts, count=max(12, LLM_CONTEXT_LINES))

        # Strategy:
        # 1. If RAG pool has enough questions → LLM selects/ranks the best ones
        # 2. If RAG pool is thin (< 6 questions) → LLM generates brand-new unique questions
        # 3. Always top-up with LLM-generated questions if still short after selection
        llm_question_used = False
        pool_is_thin = len(question_texts) < 6

        if pool_is_thin:
            # RAG found nothing useful — LLM generates from scratch
            logger.info("RAG pool thin (%s questions), asking LLM to generate from scratch", len(question_texts))
            llm_generated = _llm_generate_questions(payload, snippets)
            if llm_generated:
                llm_question_used = True
                selected_items = llm_generated
            else:
                selected_items = [{"question": q, "category": "core"} for q in question_texts[:INTERVIEW_QUESTION_COUNT]]
        else:
            # RAG has a pool — LLM selects/ranks the best from it
            llm_selected = _llm_select_questions(question_texts, payload, rag_focus)
            if llm_selected:
                llm_question_used = True
                selected_items = llm_selected
            else:
                selected_items = [{"question": q, "category": "core"} for q in question_texts[:INTERVIEW_QUESTION_COUNT]]

        # Top-up: if still short, ask LLM to generate the remaining questions
        if len(selected_items) < INTERVIEW_QUESTION_COUNT and ENABLE_LLM:
            existing_texts = {_normalize(item.get("question", "")) for item in selected_items}
            needed = INTERVIEW_QUESTION_COUNT - len(selected_items)
            logger.info("Topping up %s questions via LLM generation", needed)
            topup = _llm_generate_questions(payload, snippets)
            for item in topup:
                if len(selected_items) >= INTERVIEW_QUESTION_COUNT:
                    break
                q = _normalize_question(item.get("question", ""))
                if not q or _normalize(q) in existing_texts:
                    continue
                existing_texts.add(_normalize(q))
                selected_items.append(item)
                llm_question_used = True

        after_llm = time.perf_counter()

        # Enforce RAG-pool ratio only when pool was not thin
        if not pool_is_thin:
            rag_pool_norm = {_normalize(q): q for q in question_texts}
            rag_target = max(1, int(round(INTERVIEW_QUESTION_COUNT * 0.8)))
            rag_count = 0
            final_selected: List[Dict[str, Any]] = []
            for item in selected_items:
                q = _normalize_question(item.get("question", ""))
                if not q:
                    continue
                key = _normalize(q)
                if key in rag_pool_norm:
                    rag_count += 1
                final_selected.append({"question": q, "category": item.get("category", "core")})
                if len(final_selected) >= INTERVIEW_QUESTION_COUNT:
                    break
            if rag_count < rag_target:
                for raw in question_texts:
                    if len(final_selected) >= INTERVIEW_QUESTION_COUNT:
                        break
                    q = _normalize_question(raw)
                    if not q:
                        continue
                    key = _normalize(q)
                    if any(_normalize(x["question"]) == key for x in final_selected):
                        continue
                    final_selected.append({"question": q, "category": "core"})
            selected_items = final_selected[:INTERVIEW_QUESTION_COUNT]
        else:
            selected_items = [
                {"question": _normalize_question(item.get("question", "")) or item.get("question", ""),
                 "category": item.get("category", "core")}
                for item in selected_items
            ][:INTERVIEW_QUESTION_COUNT]

        # Candidate-level uniqueness: do not repeat until pool is exhausted.
        candidate_key = _candidate_key(payload, role_key)
        selected_questions = [str(item.get("question", "")).strip() for item in selected_items if str(item.get("question", "")).strip()]
        unique_for_candidate = _select_unique_questions_for_candidate(
            role_key=role_key,
            candidate_key=candidate_key,
            pool=selected_questions,
            target=INTERVIEW_QUESTION_COUNT,
        )
        if unique_for_candidate:
            cat_lookup = {_normalize(str(item.get("question", ""))): str(item.get("category", "core")) for item in selected_items}
            selected_items = [
                {"question": q, "category": cat_lookup.get(_normalize(q), "core")}
                for q in unique_for_candidate
            ]
            logger.info("[QUESTION][%s][%s] unique_mode applied count=%s", role_key, candidate_key, len(selected_items))

        # Reinforcement ranking: prioritize historically better questions while still exploring.
        if RL_ENABLED and not _dataset_only_role(role_key) and selected_items:
            selected_items = sorted(
                selected_items,
                key=lambda item: (
                    _rl_question_value(role_key, str(item.get("question", ""))) +
                    (0.65 * _small_model_predict(role_key, str(item.get("question", "")))) +
                    (0.05 * random.random())  # exploration jitter to avoid same fixed top-N each interview
                ),
                reverse=True,
            )
            logger.info("[RL][%s] question_rerank applied count=%s", role_key, len(selected_items))

        # Interview order diversity: keep role-matched list, but randomize the order each interview.
        if len(selected_items) > 1:
            random.SystemRandom().shuffle(selected_items)

        # Ensure opener exists and stays first for consistent interview flow.
        selected_q_texts = [str(item.get("question", "")).strip() for item in selected_items if str(item.get("question", "")).strip()]
        intro_options = [
            "Introduce yourself.",
            "Give me a quick introduction about your background.",
            "Briefly walk me through your profile and recent work.",
            "Start with a short intro and your most relevant experience.",
        ]
        intro_pick = intro_options[random.SystemRandom().randrange(len(intro_options))]
        if not any(_is_intro_question(q) for q in selected_q_texts):
            selected_items = [{"question": intro_pick, "category": "intro"}, *selected_items]
        else:
            intro_idx = next((i for i, q in enumerate(selected_q_texts) if _is_intro_question(q)), -1)
            if intro_idx > 0:
                intro_item = selected_items[intro_idx]
                selected_items = [intro_item, *selected_items[:intro_idx], *selected_items[intro_idx + 1:]]
        selected_items = selected_items[:INTERVIEW_QUESTION_COUNT]

        questions = [
            {"questionId": f"q{i}", "question": item.get("question", ""), "category": item.get("category", "core"), "roleKey": role_key,
             "referenceLinks": urls, "department": payload.department or str(payload.job.get("department",""))}
            for i, item in enumerate(selected_items, 1)
        ]
        reason = ""
        if not questions:
            if not urls and not payload.jobDescription.strip():
                reason = "No source URLs or job description. Add JD text or links."
            elif not snippets:
                reason = "Could not fetch usable text from configured URLs."
            else:
                reason = "Fetched content did not contain enough relevant material."
        total_ms = round((time.perf_counter() - started) * 1000)
        snippet_ms = round((after_snippets - started) * 1000)
        llm_ms = round((after_llm - after_snippets) * 1000)
        logger.info("QuestionEngine build completed in %sms (snippets=%sms, llm=%sms, urls=%s, snippetsCount=%s)",
                    total_ms, snippet_ms, llm_ms, len(urls), len(snippets))
        return {
            "questions": questions, "provider": ("interview-worker-llm" if llm_question_used else "interview-worker-rag"),
            "model": MODEL_NAME, "roleKey": role_key, "sources": urls,
            "onlineContextCount": len(snippets), "reason": reason,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENGINE 2: CounterEngine
# ══════════════════════════════════════════════════════════════════════════════

class CounterEngine:
    """
    Generates follow-up / probing questions based on a candidate's answer.

    Strategy matrix (driven by answer score):
      ≥ 80  → deepen: ask for architecture/trade-off elaboration
      40–79 → clarify: ask candidate to justify or give a concrete example
      < 40  → probe: expose gaps, ask foundational questions
    """

    # Follow-up template pools — no topic-specific hardcoding
    _DEEP_TEMPLATES = [
        "You mentioned {keyword} — can you walk through the architecture decisions involved?",
        "How would you scale {keyword} in a high-load production environment?",
        "What monitoring and observability would you add around {keyword}?",
        "What are the failure modes of {keyword} and how do you mitigate them?",
        "How does {keyword} interact with the rest of the system in terms of latency and consistency?",
    ]
    _CLARIFY_TEMPLATES = [
        "Can you give a concrete real-world example of how you applied {keyword}?",
        "What specific trade-off led you to choose {keyword} over alternatives?",
        "How would you measure success after implementing {keyword}?",
        "Walk me through the steps you took when {keyword} didn't behave as expected.",
        "What would you do differently with {keyword} knowing what you know now?",
    ]
    _PROBE_TEMPLATES = [
        "It sounds like {keyword} wasn't fully covered — how does {keyword} work fundamentally?",
        "Can you explain {keyword} as if you were teaching it to a junior engineer?",
        "What is the core contract or guarantee that {keyword} provides?",
        "When should you NOT use {keyword}?",
        "How do you debug a broken {keyword} in a production system?",
    ]

    def _pick_templates(self, score: int) -> List[str]:
        if score >= 80:   return self._DEEP_TEMPLATES
        if score >= 40:   return self._CLARIFY_TEMPLATES
        return self._PROBE_TEMPLATES

    def _extract_answer_keywords(self, answer: str, question: str, n: int = 6) -> List[str]:
        combined = f"{question} {answer}"
        kws = _keywords(combined, top_n=n * 2)
        q_kws = set(_keywords(question, top_n=8))
        # Prefer keywords that appear in the answer and are relevant to the question
        prioritized = [k for k in kws if k in set(_tokenize(answer)) and k in q_kws]
        remainder   = [k for k in kws if k not in prioritized]
        return (prioritized + remainder)[:n]

    def _estimate_score(self, answer: str, question: str, snippets: List[str]) -> int:
        """Quick heuristic score when answerScore is not provided."""
        if _is_nonsense(answer): return 5
        q_kws   = _keywords(question, top_n=10)
        src_kws = _keywords(" ".join(snippets), top_n=30)
        q_match  = _overlap_ratio(answer, q_kws[:10])
        src_match = _overlap_ratio(answer, src_kws[:20])
        ratio    = q_match * 0.7 + src_match * 0.3
        words    = len(re.findall(r"\S+", answer))
        length_bonus = min(1.0, words / float(TARGET_WORDS_PER_ANSWER))
        base = int(round(30 + ((ratio - 0.30) / 0.70) * 70)) if ratio >= 0.30 else int(round(ratio * 100))
        return _clamp(int(round(base * (0.5 + 0.5 * length_bonus))))

    def generate(self, req: CounterQuestionRequest) -> Dict[str, Any]:
        if not COUNTER_Q_ENABLED:
            return {"ok": False, "reason": "CounterEngine disabled", "questions": []}

        answer   = _limit_words(req.answer, MAX_ANSWER_WORDS)
        question = req.question.strip()
        max_q    = max(1, min(req.maxQuestions, 5))

        # Build lightweight context
        snippets: List[str] = []
        jd = req.jobDescription.strip()
        if jd: snippets.extend(s.strip() for s in re.split(r"[\n\r]+", jd) if s.strip())
        snippets.extend(_flatten(req.resume))
        snippets.extend(_flatten(req.candidateProfile))

        # Determine score
        score = req.answerScore
        if score < 0:
            score = self._estimate_score(answer, question, snippets)

        templates   = self._pick_templates(score)
        keywords    = self._extract_answer_keywords(answer, question, n=max_q + 2)

        if not keywords:
            # Fallback when answer is too short / empty
            keywords = _keywords(f"{req.jobTitle} {question}", top_n=max_q + 2) or ["the topic"]

        questions: List[str] = []
        seen: Set[str] = set()
        for i, kw in enumerate(keywords):
            if len(questions) >= max_q: break
            tmpl = templates[i % len(templates)]
            q = tmpl.format(keyword=kw)
            k = _normalize(q)
            if k in seen: continue
            seen.add(k); questions.append(q)

        strategy = ("deepen" if score >= 80 else "clarify" if score >= 40 else "probe")
        return {
            "ok": True,
            "strategy": strategy,
            "answerScore": score,
            "questions": [{"questionId": f"cq{i+1}", "question": q} for i, q in enumerate(questions)],
            "keywords": keywords[:max_q],
            "provider": "interview-worker-counter",
            "model": MODEL_NAME,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Evaluator
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(payload: InterviewRequest) -> Dict[str, Any]:
    snippets, role_key, urls = _build_snippets(payload)
    reference = snippets + _split_sentences(payload.jobDescription)
    if payload.department.strip(): reference.append(f"Department: {payload.department.strip()}")
    reference.extend(str(p).strip() for p in payload.customPrompts if str(p).strip())
    source_kws = _keywords("\n".join(reference + [payload.jobTitle or ""]), top_n=40)
    expected_answer_map: Dict[str, str] = {}
    for qa in _stored_qa_bank_for_role(role_key):
        qn = _normalize_question(qa.get("question", ""))
        ans = str(qa.get("answer", "")).strip()
        if not qn or not _is_valid_qa_pair(qn, ans):
            continue
        expected_answer_map[_normalize(qn)] = ans

    answers = payload.answers if isinstance(payload.answers, list) else []
    if not answers:
        return {"overallScore":0,"technicalScore":0,"behavioralScore":5,"communicationScore":8,
                "problemSolvingScore":0,"passed":False,"summary":"No answers submitted.",
                "weakAreas":["No answer evidence","No role alignment"],
                "recommendations":["Submit complete answers tied to job description."],
                "feedback":{"strengths":["Session started"],"improvements":["Provide role-specific answers"]},
                "provider":"interview-worker-rag","model":MODEL_NAME,"roleKey":role_key,"reason":"No answers"}

    q_lookup: Dict[str, str] = {
        str(q.get("questionId","")).strip(): str(q.get("question","")).strip()
        for q in (payload.questions if isinstance(payload.questions, list) else [])
        if q.get("questionId") and q.get("question")
    }

    per_scores: List[int] = []; q_analysis: List[Dict[str, Any]] = []
    nonsense_count = low_speed_count = 0

    llm_used = False
    for idx, item in enumerate(answers):
        raw   = str(item.get("answer","")).strip()
        ans   = _limit_words(raw, MAX_ANSWER_WORDS)
        qid   = str(item.get("questionId","")).strip() or f"q{idx+1}"
        qtext = _first_non_blank(q_lookup.get(qid), str(item.get("question","")).strip())
        q_kws = _keywords(qtext, top_n=10)
        is_intro = _is_intro_question(qtext)

        dur = _safe_float(item.get("durationSeconds"), 0.0)
        if dur <= 0 and _safe_float(item.get("durationMs"), 0.0) > 0:
            dur = _safe_float(item.get("durationMs")) / 1000.0
        if dur <= 0 and payload.durationSeconds > 0:
            dur = payload.durationSeconds / max(1, len(answers))

        sentences = _split_sentences(ans)
        word_count = len(re.findall(r"\S+", ans))
        wpm = 0.0

        if _is_nonsense(ans):
            score = 5; match_ratio = 0.0; nonsense_count += 1
            sent_analysis: List[Dict[str, Any]] = []
            sent_correct_count = 0; all_correct = False
        else:
            src_match_lex = _overlap_ratio(ans, source_kws[:20])
            q_match_lex   = _overlap_ratio(ans, q_kws[:10])
            candidate_sem_text = _semantic_question_answer_text(qtext, ans)
            question_sem_text = _semantic_context_text(qtext, qtext)
            ref_for_sem = [question_sem_text] + [_semantic_context_text(qtext, ref) for ref in reference[:40]]
            q_match_sem   = _semantic_similarity(candidate_sem_text, question_sem_text)
            src_match_sem = _best_semantic_similarity(candidate_sem_text, ref_for_sem, cap=20)
            # Concept-first blend: semantic similarity carries more weight than exact keyword match.
            match_ratio = (
                q_match_lex * 0.30 +
                src_match_lex * 0.15 +
                q_match_sem * 0.35 +
                src_match_sem * 0.20
            )
            logger.info(
                "[EVAL][%s] lexicalQ=%.2f lexicalSrc=%.2f semanticQ=%.2f semanticSrc=%.2f blended=%.2f",
                qid, q_match_lex, src_match_lex, q_match_sem, src_match_sem, match_ratio
            )
            concept_score = _clamp(int(round(match_ratio * 100)))
            # ML-style fast scorer:
            # - Do not expect full answer length.
            # - <30 words can still score around 80 if concept is right.
            # - >40 words with strong concept match can reach 100.
            if word_count < 6:
                score = min(concept_score, 24)
            elif word_count < 30:
                score = _clamp(int(round(concept_score * 0.75 + 20)))
                if concept_score >= 70:
                    score = max(score, 80)
            elif word_count > 40 and concept_score >= 70:
                score = 100
            else:
                score = _clamp(int(round(concept_score * 0.85 + 10)))

            sent_analysis = []
            sent_correct_count = 0
            sentence_refs = [_semantic_context_text(qtext, ref) for ref in reference[:24]]
            for si, s in enumerate(sentences):
                qr_lex = _overlap_ratio(s, q_kws[:10]); rr_lex = _overlap_ratio(s, source_kws[:20])
                sentence_sem_text = _semantic_question_answer_text(qtext, s)
                qr_sem = _semantic_similarity(sentence_sem_text, question_sem_text)
                rr_sem = _best_semantic_similarity(sentence_sem_text, sentence_refs, cap=12)
                sr = (qr_lex * 0.25) + (rr_lex * 0.15) + (qr_sem * 0.40) + (rr_sem * 0.20)
                correct = sr >= 0.34
                if correct: sent_correct_count += 1
                sent_analysis.append({
                    "index": si+1, "sentence": s,
                    "matchPercent": int(round(sr * 100)),
                    "correct": correct, "right": correct, "wrong": not correct,
                    "semanticPercent": int(round(((qr_sem * 0.65) + (rr_sem * 0.35)) * 100)),
                    "lexicalPercent": int(round(((qr_lex * 0.65) + (rr_lex * 0.35)) * 100)),
                })
            all_correct = bool(sentences) and sent_correct_count == len(sentences)
            if dur > 0:
                wpm = (word_count * 60.0) / dur
                if wpm < MIN_WORDS_PER_MINUTE:
                    sf = max(0.4, wpm / float(MIN_WORDS_PER_MINUTE))
                    score = _clamp(int(round(score * sf))); low_speed_count += 1

            # Introduction scoring rule:
            # Evaluate framing/clarity only and grant full marks for coherent intro.
            if is_intro:
                framing_score = 100
                if len(sentences) < 2:
                    framing_score = 92
                if word_count < 12:
                    framing_score = min(framing_score, 88)
                if _is_nonsense(ans):
                    framing_score = 5
                score = _clamp(framing_score)
                match_ratio = 1.0 if score >= 88 else max(match_ratio, 0.8)
                all_correct = score >= 88
                sent_correct_count = len(sentences) if all_correct else sent_correct_count

        llm_eval = {}
        answer_validation: Dict[str, Any] = {}
        if LLM_USE_FOR_EVALUATION and WORKER_STARTUP_STATUS.get("llmReady", False) and ans.strip():
            llm_eval = _llm_score_answer(qtext, ans, reference)
            if llm_eval:
                llm_used = True
                score = _clamp(_safe_int(llm_eval.get("score"), score))
                match_ratio = _clamp(_safe_int(llm_eval.get("matchPercent"), int(round(match_ratio * 100)))) / 100.0
                all_correct = bool(llm_eval.get("isCorrect", all_correct))

        expected_answer = expected_answer_map.get(_normalize(_normalize_question(qtext)))
        if expected_answer and ans.strip():
            lex = _overlap_ratio(ans, _keywords(expected_answer, top_n=20))
            sem = _semantic_similarity(
                _semantic_question_answer_text(qtext, ans),
                _semantic_question_answer_text(qtext, expected_answer),
            )
            validation_ratio = (lex * 0.45) + (sem * 0.55)
            validation_score = _clamp(int(round(validation_ratio * 100)))
            is_validated = validation_score >= 35
            if not is_validated:
                score = _clamp(score - 10)
            answer_validation = {
                "enabled": True,
                "score": validation_score,
                "passed": is_validated,
                "referenceAvailable": True,
            }
        else:
            answer_validation = {"enabled": True, "referenceAvailable": False}

        per_scores.append(score)
        mp = int(round(match_ratio * 100))
        is_correct = bool(llm_eval.get("isCorrect", False)) if llm_eval else (bool(all_correct and score >= 70) if not _is_nonsense(ans) else False)
        q_analysis.append({
            "questionId": qid, "question": qtext, "answer": ans, "fullAnswer": raw,
            "wordLimit": MAX_ANSWER_WORDS, "wordCount": word_count,
            "targetWordCount": TARGET_WORDS_PER_ANSWER, "matchPercent": mp,
            "score": score, "isCorrect": is_correct, "matched": mp >= 30,
            "right": is_correct, "correctnessScore": 1 if is_correct else 0,
            "sentenceAnalysis": sent_analysis if not _is_nonsense(ans) else [],
            "sentenceCount": len(sentences),
            "sentenceCorrectCount": 0 if _is_nonsense(ans) else sent_correct_count,
            "allSentencesCorrect": False if _is_nonsense(ans) else all_correct,
            "wordsPerMinute": round(wpm, 2), "minWordsPerMinuteRule": MIN_WORDS_PER_MINUTE,
            "wpmPenaltyApplied": bool(wpm > 0 and wpm < MIN_WORDS_PER_MINUTE),
            "analysis": (_first_non_blank(
                llm_eval.get("explanation", "") if llm_eval else "",
                "Introduction evaluated on sentence framing and clarity." if is_intro else "",
                "Answer is weakly aligned; needs concrete technical details." if mp < 30 or (_is_nonsense(ans) or not all_correct)
                else "Answer has strong sentence-level topical alignment."
            )),
            "llmStrengths": llm_eval.get("strengths", []) if llm_eval else [],
            "llmGaps": llm_eval.get("gaps", []) if llm_eval else [],
            "answerValidation": answer_validation,
        })

    overall = int(round(sum(per_scores) / max(1, len(per_scores))))
    if nonsense_count >= max(1, len(per_scores) // 2):
        overall = _clamp(overall - 25)

    threshold   = _safe_int(payload.threshold, 70)
    technical   = _clamp(int(round(overall * 1.03)))
    communication = _clamp(int(round(overall * 0.9 + 7)))
    behavioral  = _clamp(int(round((overall + communication) / 2)))
    problem_solving = _clamp(int(round((overall + technical) / 2)))

    improvements: List[str] = []
    if nonsense_count:     improvements.append("Some answers were nonsensical. Use concrete examples.")
    failed_validation = sum(
        1 for qa in q_analysis
        if isinstance(qa.get("answerValidation"), dict)
        and qa["answerValidation"].get("referenceAvailable")
        and not qa["answerValidation"].get("passed", True)
    )
    if failed_validation:
        improvements.append(f"{failed_validation} answers failed reference-answer validation.")
    transcript_lex = _overlap_ratio(payload.transcript, source_kws[:14])
    transcript_sem = _best_semantic_similarity(payload.transcript, reference[:30], cap=15)
    if (transcript_lex * 0.45 + transcript_sem * 0.55) < 0.18:
        improvements.append("Answers are weakly aligned to JD/source material.")
    if low_speed_count:    improvements.append(f"{low_speed_count} answers below {MIN_WORDS_PER_MINUTE} WPM penalized.")
    if technical < threshold: improvements.append("Increase depth: architecture, trade-offs, debugging.")
    if not improvements:   improvements.append("Add quantified impact and deeper implementation detail.")

    strengths = (["Strong relevance to source-linked topics."] if overall >= threshold else []) + \
                (["Responses were reasonably clear."] if communication >= 65 else []) or \
                ["All questions were attempted."]

    if RL_ENABLED:
        try:
            for qa in q_analysis:
                _rl_update_question(
                    role_key=role_key,
                    question=str(qa.get("question", "")),
                    score=_safe_int(qa.get("score"), 0),
                    correct=bool(qa.get("isCorrect", False)),
                )
        except Exception as ex:
            logger.warning("[RL][%s] update_failed: %s", role_key, ex)

    return {
        "overallScore": overall, "technicalScore": technical,
        "behavioralScore": behavioral, "communicationScore": communication,
        "problemSolvingScore": problem_solving, "passed": overall >= threshold,
        "summary": ("Scored by external LLM against fetched context and answer quality."
                    if llm_used else "Scored against concept-level semantic match and RAG/JD relevance."),
        "weakAreas": improvements[:3],
        "recommendations": [
            "Use STAR format with measurable outcomes.",
            "Reference job-specific tools and responsibilities.",
            "Explain technical trade-offs and debugging steps clearly.",
        ],
        "feedback": {"strengths": strengths, "improvements": improvements},
        "provider": ("interview-worker-llm" if llm_used else "interview-worker-rag"), "model": MODEL_NAME,
        "roleKey": role_key, "answerScores": per_scores,
        "questionAnalysis": q_analysis, "sources": urls,
        "evaluationRules": {
            "sentenceBySentence": True,
            "targetWordsPerAnswer": TARGET_WORDS_PER_ANSWER,
            "minimumWordsPerMinute": MIN_WORDS_PER_MINUTE,
            "allSentencesCorrectRule": "All sentences correct + target words met → score = 100",
        },
    }


# ── Engine singletons ─────────────────────────────────────────────────────────
_question_engine = QuestionEngine()
_counter_engine  = CounterEngine()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STT / TTS helpers  (unchanged from v4, consolidated)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_whisper() -> tuple[bool, str]:
    global WHISPER_HANDLE
    if not ENABLE_STT:
        return False, "Worker STT disabled"
    if WhisperModel is None:
        return False, "faster-whisper dependency unavailable"
    if WHISPER_HANDLE is not None:
        return True, f"Whisper ready: {STT_MODEL_ID} ({STT_DEVICE}/{STT_COMPUTE_TYPE})"
    try:
        WHISPER_HANDLE = WhisperModel(
            STT_MODEL_ID,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
        return True, f"Whisper ready: {STT_MODEL_ID} ({STT_DEVICE}/{STT_COMPUTE_TYPE})"
    except Exception as ex:
        WHISPER_HANDLE = None
        return False, f"Whisper init failed: {ex}"

def _ensure_piper() -> tuple[bool, str]:
    return False, "TTS disabled in worker (browser Web Speech API mode; use Google Chrome)"


def _normalize_stt_language(value: str) -> Optional[str]:
    language = str(value or "").strip().lower()
    if not language:
        return None
    if language in {"auto", "detect", "default"}:
        return None
    if "-" in language:
        language = language.split("-", 1)[0]
    return language or None


def _audio_suffix_for_content_type(content_type: str) -> str:
    normalized = str(content_type or "").strip().lower()
    if "webm" in normalized:
        return ".webm"
    if "ogg" in normalized or "opus" in normalized:
        return ".ogg"
    if "wav" in normalized:
        return ".wav"
    if "mpeg" in normalized or "mp3" in normalized:
        return ".mp3"
    if "mp4" in normalized or "aac" in normalized or "m4a" in normalized:
        return ".m4a"
    return ".webm"


def _transcribe_audio_path(audio_path: str, language: str = "") -> Dict[str, Any]:
    ok, reason = _ensure_whisper()
    if not ok or WHISPER_HANDLE is None:
        return {"ok": False, "reason": reason}
    try:
        segments, info = WHISPER_HANDLE.transcribe(
            audio_path,
            language=_normalize_stt_language(language),
            beam_size=STT_BEAM_SIZE,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text = " ".join(str(segment.text or "").strip() for segment in segments).strip()
        return {
            "ok": True,
            "text": text,
            "language": getattr(info, "language", None) or _normalize_stt_language(language) or "",
            "duration": getattr(info, "duration", 0.0) or 0.0,
            "model": STT_MODEL_ID,
            "provider": "python-worker",
        }
    except Exception as ex:
        return {"ok": False, "reason": f"transcription_failed:{ex}"}


def _transcribe_audio_bytes(audio_bytes: bytes, content_type: str = "", language: str = "") -> Dict[str, Any]:
    if not audio_bytes or len(audio_bytes) < STT_MIN_CHUNK_BYTES:
        return {"ok": True, "text": "", "provider": "python-worker", "model": STT_MODEL_ID}
    suffix = _audio_suffix_for_content_type(content_type)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            handle.flush()
            temp_path = handle.name
        return _transcribe_audio_path(temp_path, language)
    except Exception as ex:
        return {"ok": False, "reason": f"transcription_io_failed:{ex}"}
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

def _bootstrap_models() -> None:
    global RAG_SOURCE_MAP, TRAINING_STORE_MAP, QUESTION_STATE_MAP, REINFORCEMENT_STATE_MAP
    global SOURCE_QUALITY_MAP, LEARNING_LOG_MAP, LEARNING_POLICY_MAP, LAST_SMALL_MODEL_TRAIN_TS
    RAG_SOURCE_MAP = _load_source_map()
    TRAINING_STORE_MAP = _load_training_store()
    training_clean = _sanitize_training_store(TRAINING_STORE_MAP)
    if training_clean.get("changed", 0) > 0:
        _save_training_store(TRAINING_STORE_MAP)
    logger.info(
        "TRAINING_STORE cleanup: domains=%s changed=%s questionsDropped=%s qaDropped=%s",
        training_clean.get("domains", 0),
        training_clean.get("changed", 0),
        training_clean.get("questionsDropped", 0),
        training_clean.get("qaDropped", 0),
    )
    QUESTION_STATE_MAP = _load_candidate_state()
    REINFORCEMENT_STATE_MAP = _load_reinforcement_state()
    SOURCE_QUALITY_MAP = _load_source_quality()
    LEARNING_LOG_MAP = _load_learning_log()
    LEARNING_POLICY_MAP = _load_learning_policy()
    cache_clean = _cleanup_vector_cache_files()
    logger.info(
        "VECTOR_CACHE cleanup: files=%s changed=%s kept=%s dropped=%s migrated=%s bytes=%s file=%s",
        cache_clean.get("files", 0),
        cache_clean.get("changed", 0),
        cache_clean.get("kept", 0),
        cache_clean.get("dropped", 0),
        cache_clean.get("migrated", 0),
        cache_clean.get("bytes", 0),
        VECTOR_CACHE_FILE,
    )
    _prefetch_rag_cache()
    llm_ok, llm_reason = _ensure_llm()
    sem_ok, sem_reason = _ensure_semantic_model()
    small_model_loaded = _load_small_model()
    sm_ok = bool(small_model_loaded)
    sm_reason = (f"Small model loaded: {SMALL_MODEL_FILE}" if sm_ok else "Small model not trained yet")
    if SMALL_MODEL_FILE.exists():
        try:
            LAST_SMALL_MODEL_TRAIN_TS = SMALL_MODEL_FILE.stat().st_mtime
        except Exception:
            LAST_SMALL_MODEL_TRAIN_TS = 0.0
    else:
        LAST_SMALL_MODEL_TRAIN_TS = 0.0
    stt_ok, stt_reason = _ensure_whisper()
    tts_ok, tts_reason = _ensure_piper()
    WORKER_STARTUP_STATUS.update(llmReady=llm_ok, sttReady=stt_ok, ttsReady=tts_ok, semanticReady=sem_ok, smallModelReady=sm_ok,
                                  llmReason=llm_reason,
                                  sttReason=stt_reason, ttsReason=tts_reason, semanticReason=sem_reason, smallModelReason=sm_reason)
    logger.info("LLM: %s", llm_reason)
    logger.info("SEMANTIC: %s", sem_reason)
    logger.info("SMALL_MODEL: %s", sm_reason)
    startup_train_result = _ensure_default_small_model_training()
    if startup_train_result.get("ok") and not startup_train_result.get("skipped"):
        logger.info(
            "[ONLINE_TRAIN][startup] defaults trained sources=%s samples=%s",
            len(startup_train_result.get("learnedSources", [])),
            startup_train_result.get("sampleCount", 0),
        )
    elif startup_train_result.get("reason"):
        logger.info("[ONLINE_TRAIN][startup] %s", startup_train_result.get("reason"))
    _maybe_auto_train_small_model(trigger="startup")
    ollama_preload = preload_ollama_models()
    WORKER_STARTUP_STATUS["ollamaReachable"] = bool(ollama_preload.get("ok")) or any(
        bool(item.get("available")) for item in (ollama_preload.get("models", {}) or {}).values() if isinstance(item, dict)
    )
    WORKER_STARTUP_STATUS["ollamaModels"] = ollama_preload.get("models", {})
    WORKER_STARTUP_STATUS["ollamaReason"] = "preloaded" if ollama_preload.get("ok") else "partial_or_unavailable"
    logger.info("OLLAMA preload: %s", json.dumps(ollama_preload, ensure_ascii=True))
    logger.info("STT priority mode: %s", STT_PRIORITY_MODE)
    logger.info("STT: %s", stt_reason); logger.info("TTS: %s", tts_reason)

def _ensure_llm() -> tuple[bool, str]:
    if not ENABLE_LLM:
        return False, "LLM disabled"
    if not requests:
        return False, "requests unavailable"
    inventory = ollama_model_inventory()
    if inventory.get("ok"):
        models = inventory.get("models", []) if isinstance(inventory.get("models"), list) else []
        return True, f"Ollama reachable ({len(models)} models visible)"
    if LLM_API_URL:
        return True, f"External LLM API enabled: {MODEL_NAME}"
    return False, "Neither Ollama nor external LLM API is available"

def _ensure_semantic_model() -> tuple[bool, str]:
    if not SEMANTIC_MATCH_ENABLED:
        return False, "Semantic match disabled"
    if TfidfVectorizer is None or cosine_similarity is None:
        return False, "scikit-learn TF-IDF scorer unavailable"
    try:
        scorer = _get_semantic_model()
        if not isinstance(scorer, dict) or scorer.get("vectorizer") is None:
            return False, "semantic scorer unavailable"
        return True, f"Semantic scorer ready: {SEMANTIC_MODEL_NAME} ({SEMANTIC_SCORER_FILE.name})"
    except Exception as ex:
        return False, f"Semantic scorer failed: {ex}"


# Browser-only voice mode: backend STT/TTS runtime helpers removed.

def _auto_train_loop() -> None:
    global LAST_AUTO_TRAIN_TS, LAST_ACTIVITY_TS
    last_countdown_logged = -1
    while True:
        time.sleep(1)
        if not (AUTO_TRAIN_ENABLED or LEARNING_ALWAYS_ENABLED):
            continue
        now = time.time()
        base_activity = LAST_ACTIVITY_TS if isinstance(LAST_ACTIVITY_TS, (int, float)) else now
        idle_elapsed = int(now - base_activity)
        remaining = max(0, AUTO_TRAIN_IDLE_SECONDS - idle_elapsed)
        # Show visible 1-minute countdown during idle before training starts.
        if remaining > 0:
            should_log = (remaining % 10 == 0) or (remaining <= 10)
            if should_log and remaining != last_countdown_logged:
                logger.info("[TRAIN][auto-idle] countdown %ss to auto-learning start", remaining)
                last_countdown_logged = remaining
            continue
        if ACTIVE_STT_WS_COUNT > 0 or _has_active_interview_session():
            if last_countdown_logged != -2:
                logger.info("[TRAIN][auto-idle] paused active_interview=%s activeSttWs=%s", _has_active_interview_session(), ACTIVE_STT_WS_COUNT)
                last_countdown_logged = -2
            continue
        policy = _adaptive_learning_policy()
        effective_gap = max(60, _safe_int(policy.get("minGapSec"), AUTO_TRAIN_MIN_GAP_SEC))
        if (now - LAST_AUTO_TRAIN_TS) < effective_gap:
            continue
        try:
            last_countdown_logged = -1
            logger.info(
                "[TRAIN][auto-idle] trigger idleSeconds=%s minGapSeconds=%s domains=%s mode=%s knowledge=%.2f",
                AUTO_TRAIN_IDLE_SECONDS, effective_gap, AUTO_TRAIN_DOMAIN_LIST, policy.get("mode", "slow"), _safe_float(policy.get("knowledgeScore"), 0.0)
            )
            discover_target = max(12, int(AUTO_DISCOVER_PER_DOMAIN * _safe_float(policy.get("discoverMultiplier"), 1.0)))
            q_target = max(5, int(min(INTERVIEW_QUESTION_COUNT, 25) * _safe_float(policy.get("questionMultiplier"), 1.0)))
            req = OneTimeTrainingRequest(
                domains=AUTO_TRAIN_DOMAIN_LIST or ["java", "sales", "flutter", "react"],
                includeDefaults=True,
                discoverUrls=True,
                forceRefresh=False,
                discoveredUrlLimitPerDomain=discover_target,
                questionCountPerDomain=max(5, min(q_target, 30)),
                runLabel=f"auto-idle-{policy.get('mode', 'slow')}",
            )
            result = rag_train_once(req)
            LAST_AUTO_TRAIN_TS = now
            _maybe_auto_train_small_model(trigger="auto-idle")
            logger.info(
                "[TRAIN][auto-idle] completed domains=%s durationMs=%s mode=%s knowledge=%.2f",
                req.domains, result.get("durationMs", 0), policy.get("mode", "slow"), _safe_float(policy.get("knowledgeScore"), 0.0)
            )
        except Exception as ex:
            LAST_AUTO_TRAIN_TS = now
            logger.warning("[TRAIN][auto-idle] failed: %s", ex)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FastAPI routes
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_event() -> None:
    global SOURCE_QUALITY_MAP, LEARNING_LOG_MAP, LEARNING_POLICY_MAP
    SOURCE_QUALITY_MAP = _load_source_quality()
    LEARNING_LOG_MAP = _load_learning_log()
    LEARNING_POLICY_MAP = _load_learning_policy()
    policy = _adaptive_learning_policy()
    _append_learning_log("startup", {"resourceProfile": RESOURCE_PROFILE, "strictIdentity": STRICT_CANDIDATE_IDENTITY, "learnMode": policy.get("mode"), "knowledgeScore": policy.get("knowledgeScore")})
    threading.Thread(target=_bootstrap_models, daemon=True).start()
    threading.Thread(target=_auto_train_loop, daemon=True).start()


@app.middleware("http")
async def _track_activity(request: Request, call_next):
    global LAST_ACTIVITY_TS
    LAST_ACTIVITY_TS = time.time()
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, Any]:
    training_domains = sorted(
        (TRAINING_STORE_MAP.get("domains", {}) or {}).keys()
    ) if isinstance(TRAINING_STORE_MAP, dict) else []
    policy = _adaptive_learning_policy()
    live_ollama_inventory = ollama_model_inventory()
    live_ollama_names = set(live_ollama_inventory.get("models", [])) if live_ollama_inventory.get("ok") else set()
    startup_ollama_models = WORKER_STARTUP_STATUS.get("ollamaModels", {})
    configured_model_names = {
        str(resolve_model_profile("candidate_match").get("model", "")).strip(),
        str(resolve_model_profile("interview_evaluation").get("model", "")).strip(),
    }
    configured_model_names = {name for name in configured_model_names if name}
    missing_configured_models = sorted(
        name for name in configured_model_names if name not in live_ollama_names
    )
    ollama_models_ready = all(
        name in live_ollama_names
        for name in configured_model_names
    ) if configured_model_names else True
    deployment_ready = (not ENABLE_LLM) or (not OLLAMA_PRELOAD_ON_STARTUP) or ollama_models_ready
    deployment_reason = "configured_ollama_models_ready" if deployment_ready else (
        "waiting_for_ollama_models:" + ",".join(missing_configured_models)
        if missing_configured_models else
        "waiting_for_ollama_models"
    )
    return {
        "status": "ok", "model": MODEL_NAME, "ready": True,
        "deploymentReady": deployment_ready,
        "deploymentReason": deployment_reason,
        "ragEnabled": True, "onlineRagEnabled": RAG_FETCH_ONLINE,
        "ragRoles": sorted(RAG_SOURCE_MAP.keys()),
        "trainedDomains": training_domains,
        "trainingStorePath": str(TRAINING_STORE_FILE),
        "reinforcementStatePath": str(REINFORCEMENT_STATE_FILE),
        "smallModelPath": str(SMALL_MODEL_FILE),
        "smallModelAutoTrainEnabled": SMALL_MODEL_AUTO_TRAIN_ENABLED,
        "smallModelAutoTrainGapSeconds": SMALL_MODEL_AUTO_TRAIN_GAP_SECONDS,
        "smallModelAutoTrainMinSamples": SMALL_MODEL_AUTO_TRAIN_MIN_SAMPLES,
        "smallModelAutoTrainWindowDays": SMALL_MODEL_AUTO_TRAIN_WINDOW_DAYS,
        "vectorCacheEnabled": VECTOR_CACHE_ENABLED,
        "vectorCacheDir": str(VECTOR_CACHE_DIR),
        "vectorCacheFile": str(VECTOR_CACHE_FILE),
        "vectorCacheMaxBytes": VECTOR_CACHE_MAX_BYTES,
        "urlTextMemoryCap": URL_TEXT_MEMORY_CAP,
        "vectorTopK": VECTOR_TOP_K,
        "questionStatePath": str(CANDIDATE_STATE_FILE),
        "sourceQualityPath": str(SOURCE_QUALITY_FILE),
        "learningLogPath": str(LEARNING_LOG_FILE),
        "learningPolicyPath": str(LEARNING_POLICY_FILE),
        "autoTrainEnabled": AUTO_TRAIN_ENABLED,
        "learningAlwaysEnabled": LEARNING_ALWAYS_ENABLED,
        "autoTrainIdleSeconds": AUTO_TRAIN_IDLE_SECONDS,
        "autoTrainMinGapSeconds": AUTO_TRAIN_MIN_GAP_SEC,
        "effectiveAutoTrainMinGapSeconds": _safe_int(policy.get("minGapSec"), AUTO_TRAIN_MIN_GAP_SEC),
        "onDemandDiscoveryEnabled": ON_DEMAND_DISCOVERY_ENABLED,
        "activeSttWebSockets": ACTIVE_STT_WS_COUNT,
        "activeInterviewSessions": len(ACTIVE_INTERVIEW_SESSIONS),
        "interviewSessionTtlSeconds": INTERVIEW_SESSION_TTL_SECONDS,
        "learningMode": policy.get("mode", "slow"),
        "knowledgeScore": _safe_float(policy.get("knowledgeScore"), 0.0),
        "novelUrlRepeatDays": NOVEL_URL_REPEAT_DAYS,
        "dailyNewUrlTarget": DAILY_NEW_URL_TARGET,
        "strictCandidateIdentity": STRICT_CANDIDATE_IDENTITY,
        "rlRawEventRetentionDays": RL_RAW_EVENT_RETENTION_DAYS,
        "resourceProfile": RESOURCE_PROFILE,
        "sourceAllowlistDomains": SOURCE_ALLOWLIST_DOMAINS,
        "engines": {"question": "QuestionEngine", "counter": "CounterEngine"},
        "counterEnabled": COUNTER_Q_ENABLED, "counterPerAnswer": COUNTER_Q_PER_ANSWER,
        "llmEnabled": ENABLE_LLM,
        "ollamaBaseUrl": str(resolve_model_profile("health").get("baseUrl", "")),
        "ollamaPreloadOnStartup": OLLAMA_PRELOAD_ON_STARTUP,
        "ollamaReachable": bool(live_ollama_inventory.get("ok")),
        "ollamaConfiguredModels": sorted(configured_model_names),
        "ollamaModelsReady": ollama_models_ready,
        "ollamaMissingConfiguredModels": missing_configured_models,
        "ollamaModels": {name: {"available": True} for name in sorted(live_ollama_names)} if live_ollama_inventory.get("ok") else startup_ollama_models,
        "ollamaReason": "live_inventory_ok" if live_ollama_inventory.get("ok") else live_ollama_inventory.get("reason", WORKER_STARTUP_STATUS.get("ollamaReason", "not_initialized")),
        "semanticMatchEnabled": SEMANTIC_MATCH_ENABLED,
        "semanticModel": SEMANTIC_MODEL_NAME,
        "sttEnabled": ENABLE_STT, "ttsEnabled": ENABLE_TTS, "sttModel": STT_MODEL_ID,
        "sttPriorityMode": STT_PRIORITY_MODE,
        **{k: WORKER_STARTUP_STATUS[k] for k in WORKER_STARTUP_STATUS},
        "targetWordsPerAnswer": TARGET_WORDS_PER_ANSWER,
        "minimumWordsPerMinute": MIN_WORDS_PER_MINUTE,
    }


@app.get("/", tags=["system"])
def root_index() -> Dict[str, Any]:
    return {
        "name": "Interview AI Worker API",
        "version": "5.0.0",
        "health": "/health",
        "swagger": "/swagger",
        "openapi": "/openapi.json",
        "redoc": "/redoc",
        "notes": "WebSocket endpoint available at /speech/ws/stt",
    }


def _generate_simple_ideal_answer(question: str, job_title: str, difficulty: str) -> str:
    q = str(question or "").strip()
    if not q:
        return ""
    role_key = _infer_role_key(job_title, "", q, "engineering")
    role_node = _small_model_domain_knowledge(None, role_key)
    best_bank_answer = _best_qa_bank_answer(role_key, q)
    if best_bank_answer:
        return best_bank_answer
    if isinstance(role_node, dict) and len(role_node.get("questionBank", []) or []) > 0 and len(role_node.get("qaBank", []) or []) == 0:
        return ""
    context_lines: List[str] = []
    context_lines.extend(_small_model_context_lines(role_key, count=60))
    if not context_lines and not _dataset_only_role(role_key):
        context_lines.extend(_sample_rag_lines(role_key, [job_title, q, difficulty], count=40))
    for qa in _stored_qa_bank_for_role(role_key)[:24]:
        if not isinstance(qa, dict):
            continue
        qq = str(qa.get("question", "")).strip()
        aa = str(qa.get("answer", "")).strip()
        if qq:
            context_lines.append(qq)
        if aa:
            context_lines.append(aa)
    context_lines = _filter_answer_context_lines(_sanitize_cache_lines(context_lines, max_items=120), max_items=80)
    if context_lines:
        generated = _build_best_answer(
            q,
            context_lines,
            BestAnswerRequest(
                domain=role_key,
                jobTitle=job_title,
                jobDescription="",
                maxWords=140,
            ),
        ).get("bestAnswer", "")
        generated = _limit_words(str(generated or "").strip(), 180)
        if _is_reference_answer_usable(generated):
            return generated
    if _dataset_only_role(role_key):
        return ""
    q_norm = _normalize(q)

    def _looks_like_question_dump(text: str) -> bool:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t:
            return True
        lo = t.lower()
        if lo.count("?") >= 1:
            return True
        if re.search(r"(?:^|\s)\d+\.\s+[A-Za-z]", t):
            return True
        if any(tok in lo for tok in [
            "interview questions",
            "question bank",
            "java oop(object oriented programming) concepts",
            "practice programs",
            "with solutions",
            "top companies",
            "most used language",
        ]):
            return True
        return False

    if "difference between an object-oriented programming language and an object-based programming language" in q_norm:
        return (
            "An object-oriented language supports the full OOP model: classes, objects, inheritance, "
            "polymorphism, abstraction, and encapsulation. An object-based language supports objects and "
            "encapsulation but usually does not provide full inheritance and polymorphism like class-based OOP."
        )
    if "new keyword" in q_norm and ("object is created" in q_norm or "happens internally" in q_norm):
        return (
            "When you use new in Java, the JVM allocates memory for the object on the heap, sets default field values, "
            "runs instance initializers, and then calls the selected constructor. The constructor initializes state and may "
            "invoke super() first for parent initialization. Finally, the object reference is returned and stored in the variable, "
            "while lifecycle cleanup is handled later by garbage collection."
        )
    if "platform independent" in q_norm and "java" in q_norm:
        return (
            "Java is platform independent because Java source code is compiled into bytecode, not native machine code. "
            "That bytecode runs on the JVM, and each operating system has its own JVM implementation. "
            "So the same compiled program can run across Windows, Linux, and macOS without recompiling, with minor "
            "platform-specific behavior handled by the JVM and libraries."
        )
    if "not a pure object oriented" in q_norm and "java" in q_norm:
        return (
            "Java is not considered purely object oriented because it supports primitive data types like int, char, and boolean, "
            "which are not objects. It also includes static members and methods that belong to classes rather than object instances. "
            "So Java is strongly object-oriented in practice, but not purely object-oriented by strict definition."
        )

    answer = (
        f"For {job_title or 'this role'}, explain {q} with a clear definition, core mechanism, "
        "practical example, and trade-offs. Include one short real-world scenario and keep the explanation structured."
    )
    answer = _limit_words(answer, 120)

    question_keywords = _keywords(q, top_n=8)
    overlap = _overlap_ratio(answer, question_keywords) if answer else 0.0
    lower_answer = answer.lower()
    low_signal = (
        len(re.findall(r"\S+", answer)) < 14
        or overlap < 0.16
        or _looks_like_question_dump(answer)
        or lower_answer in {"java. medium interview", "medium interview", "interview"}
        or re.fullmatch(r"[a-z0-9 ._-]{1,40}", lower_answer or "") is not None
    )
    if low_signal:
        if "difference between" in q_norm:
            m = re.search(r"difference between (.+?) and (.+?)(\?|$)", q, flags=re.IGNORECASE)
            if m:
                lhs = str(m.group(1)).strip().rstrip("?.")
                rhs = str(m.group(2)).strip().rstrip("?.")
                answer = (
                    f"{lhs} and {rhs} differ in core capability and scope. "
                    f"{lhs} usually provides stronger abstraction and extensibility, while {rhs} is often simpler and narrower in feature set. "
                    "A good interview answer should compare definition, core concepts, practical use cases, limitations, and one real example."
                )
                return _limit_words(answer, 120)
        return ""
    return answer


@app.post("/startinterveiw", tags=["simple-interview"])
def startinterveiw(payload: StartInterveiwRequest) -> Dict[str, Any]:
    session_id = f"si_{int(time.time() * 1000)}"
    started = time.perf_counter()
    role_text = _first_non_blank(payload.role, payload.jobTitle, "general").strip()
    role_key = _infer_role_key(role_text, "", role_text, "engineering")
    eff_lang = _effective_language(role_text, "", "javascript")
    snippet_seed = [role_text, payload.difficulty, role_key]
    dataset_questions = _small_model_question_bank(role_key, max(INTERVIEW_QUESTION_COUNT * 3, 40))
    if _dataset_only_role(role_key) and len(dataset_questions) < INTERVIEW_QUESTION_COUNT:
        _ensure_default_small_model_training()
        dataset_questions = _small_model_question_bank(role_key, max(INTERVIEW_QUESTION_COUNT * 3, 40))
    snippets = _small_model_context_lines(role_key, count=60) if dataset_questions else _sample_rag_lines(role_key, snippet_seed, count=60)
    if not snippets:
        snippets = _split_sentences(
            f"{role_text}. {payload.difficulty}. core concepts, system design, debugging, testing, performance."
        )

    stored_pool = dataset_questions or _stored_questions_for_role(role_key, max(INTERVIEW_QUESTION_COUNT * 3, 40))
    dynamic_pool = [] if (dataset_questions or _dataset_only_role(role_key)) else _question_engine._build_dynamic(
        role_text,
        eff_lang,
        snippets,
        department="engineering",
        custom_prompts=[],
        experience_years=0.0,
        count=INTERVIEW_QUESTION_COUNT + 6,
    )
    raw_pool = [str(q or "").strip() for q in [*stored_pool, *dynamic_pool] if str(q or "").strip()]
    clean_pool: List[str] = []
    seen: Set[str] = set()
    for raw_q in raw_pool:
        q = _normalize_question(raw_q) or raw_q
        q = q.replace("Â", " ").replace("\u00a0", " ")
        q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        k = _normalize(q)
        if k in seen:
            continue
        seen.add(k)
        clean_pool.append(q)
    if _dataset_only_role(role_key) and not clean_pool:
        raise HTTPException(status_code=503, detail="Java dataset model is not ready yet")
    candidate_key = _first_non_blank(payload.candidateId, payload.candidateName, f"anon-{session_id}")
    unique_questions = _select_unique_questions_for_candidate(
        role_key=role_key,
        candidate_key=candidate_key,
        pool=clean_pool,
        target=INTERVIEW_QUESTION_COUNT,
    )
    if not unique_questions:
        unique_questions = clean_pool[:INTERVIEW_QUESTION_COUNT]
    questions: List[Dict[str, Any]] = []
    for idx, q in enumerate(unique_questions, start=1):
        ideal = _generate_simple_ideal_answer(q, role_text, payload.difficulty)
        questions.append({"questionId": f"q{idx}", "question": q, "idealAnswer": str(ideal).strip()})
    if not questions:
        raise HTTPException(status_code=503, detail="No questions generated")
    SIMPLE_INTERVIEW_SESSIONS[session_id] = {
        "createdAt": int(time.time()),
        "context": {
            "candidateId": payload.candidateId,
            "candidateName": payload.candidateName,
            "role": role_text,
            "jobTitle": _first_non_blank(payload.jobTitle, role_text),
            "difficulty": payload.difficulty,
        },
        "questions": questions,
        "answers": {},
        "nextIndex": 0,
    }
    took_ms = round((time.perf_counter() - started) * 1000)
    logger.info("[simple-start] session=%s role=%s q=%s tookMs=%s", session_id, role_key, len(questions), took_ms)
    return {"ok": True, "sessionId": session_id}


@app.post("/askquestion", tags=["simple-interview"])
def askquestion(payload: AskQuestionRequest) -> Dict[str, Any]:
    session_id = str(payload.sessionId or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")
    session = SIMPLE_INTERVIEW_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    questions = session.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=404, detail="No questions available")
    next_index = _safe_int(session.get("nextIndex"), 0)
    if next_index < 0:
        next_index = 0
    if next_index >= len(questions):
        return {
            "ok": True,
            "sessionId": session_id,
            "completed": True,
            "message": "All questions completed",
        }
    selected = questions[next_index]
    session["nextIndex"] = next_index + 1
    SIMPLE_INTERVIEW_SESSIONS[session_id] = session
    return {
        "ok": True,
        "sessionId": session_id,
        "questionNumber": next_index + 1,
        "totalQuestions": len(questions),
        "questionId": str(selected.get("questionId", "")).strip(),
        "question": str(selected.get("question", "")).strip(),
        "answer": str(selected.get("idealAnswer", "")).strip(),
    }


@app.post("/matchanswer", tags=["simple-interview"])
def matchanswer(payload: MatchAnswerRequest) -> Dict[str, Any]:
    session_id = str(payload.sessionId or "").strip()
    question_id = str(payload.questionId or "").strip()
    answer_text = str(payload.answer or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")
    if not question_id:
        raise HTTPException(status_code=400, detail="questionId is required")
    if not answer_text:
        raise HTTPException(status_code=400, detail="answer is required")

    session = SIMPLE_INTERVIEW_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    questions = session.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    selected = None
    for q in questions:
        if str(q.get("questionId", "")).strip() == question_id:
            selected = q
            break
    if selected is None:
        raise HTTPException(status_code=404, detail="questionId not found")

    question_text = str(selected.get("question", "")).strip()
    ideal_answer = str(selected.get("idealAnswer", "")).strip()
    session_context = session.get("context", {}) if isinstance(session.get("context", {}), dict) else {}
    role_text = _first_non_blank(session_context.get("role"), session_context.get("jobTitle")).strip()
    role_key = _infer_role_key(role_text, "", question_text, "engineering")

    def _looks_like_question_dump(text: str) -> bool:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t:
            return True
        lo = t.lower()
        if lo.count("?") >= 1:
            return True
        if re.search(r"(?:^|\s)\d+\.\s+[A-Za-z]", t):
            return True
        if any(tok in lo for tok in [
            "interview questions", "question bank", "practice programs",
            "with solutions", "top companies", "most used language",
        ]):
            return True
        return False

    def _fast_semantic_similarity(text: str, candidates: List[str], cap: int = 8) -> float:
        # Avoid cold-start model download/load during request path.
        if SEMANTIC_MODEL_HANDLE is None:
            return 0.0
        return _best_semantic_similarity(text, candidates, cap=cap)

    def _collect_local_reference_lines() -> List[str]:
        refs = _sample_rag_lines(role_key, [question_text, ideal_answer], count=24)
        refs = [ln for ln in refs if not _looks_like_question_dump(ln)]
        refs.extend(_offline_qa_memory_refs(role_key, question_text, limit=16))
        if ideal_answer:
            refs.insert(0, ideal_answer)
        return _sanitize_cache_lines(refs, max_items=40)[:40]

    def _split_candidate_sentences(text: str) -> List[str]:
        parts = _split_sentences(text)
        if not parts:
            parts = [text]
        out: List[str] = []
        for p in parts:
            t = re.sub(r"\s+", " ", str(p or "")).strip()
            if len(re.findall(r"\w+", t)) < 4:
                continue
            out.append(t)
        return out[:8]

    def _fetch_sentence_refs_online(sentence: str) -> List[str]:
        # User-required policy: keep answer matching offline by default.
        if not MATCHANSWER_ONLINE_ENABLED or not RAG_FETCH_ONLINE:
            return []
        query = _limit_words(sentence, 18)
        if not query:
            return []
        providers = _provider_priority(SEARCH_ENGINES or ["duckduckgo", "bing", "brave"])
        urls: List[str] = []
        seen_urls: Set[str] = set()
        for provider in providers[:2]:
            try:
                status, links = _search_provider_query(provider, query, 0)
            except Exception:
                continue
            if status >= 400:
                continue
            for u in links:
                ok, _reason = _is_url_allowed(u)
                if not ok:
                    continue
                key = _normalize(u)
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                urls.append(u)
                if len(urls) >= max(1, MATCHANSWER_ONLINE_MAX_URLS):
                    break
            if urls:
                break
        refs: List[str] = []
        for u in urls[: max(1, MATCHANSWER_ONLINE_MAX_URLS)]:
            cached = _fetch_url_cached_only(u, query=query)
            lines = cached if cached else _fetch_url(u)
            for line in lines[:20]:
                t = str(line or "").strip()
                if t and not _looks_like_question_dump(t):
                    refs.append(t)
        return _sanitize_cache_lines(refs, max_items=20)[:20]

    def _sentence_match_score(answer: str, local_refs: List[str]) -> tuple[int, List[Dict[str, Any]], str]:
        if not answer.strip():
            return 0, [], "local"
        sentences = _split_candidate_sentences(answer)
        if not sentences:
            return 0, [], "local"
        details: List[Dict[str, Any]] = []
        unit_scores: List[float] = []
        online_used = False
        online_budget = max(0, MATCHANSWER_ONLINE_MAX_SENTENCES)
        local_sem_refs = [_semantic_context_text(question_text, ref) for ref in local_refs[:30]]
        for idx, sent in enumerate(sentences):
            kws = _keywords(sent, top_n=10)
            local_lex = max([_overlap_ratio(ref, kws) for ref in local_refs[:30]] or [0.0])
            local_sem = _fast_semantic_similarity(_semantic_question_answer_text(question_text, sent), local_sem_refs, cap=10)
            best = max(local_lex, local_sem)
            source = "local"
            if best < 0.42 and online_budget > 0:
                online_budget -= 1
                online_refs = _fetch_sentence_refs_online(sent)
                if online_refs:
                    online_used = True
                    online_lex = max([_overlap_ratio(ref, kws) for ref in online_refs[:20]] or [0.0])
                    online_sem = _fast_semantic_similarity(
                        _semantic_question_answer_text(question_text, sent),
                        [_semantic_context_text(question_text, ref) for ref in online_refs[:20]],
                        cap=8,
                    )
                    online_best = max(online_lex, online_sem)
                    if online_best > best:
                        best = online_best
                        source = "online"
            unit_scores.append(best)
            details.append({
                "index": idx + 1,
                "sentence": sent,
                "score": _clamp(int(round(best * 100))),
                "source": source,
            })
        final_score = _clamp(int(round((sum(unit_scores) / max(1, len(unit_scores))) * 100)))
        return final_score, details, ("online+local" if online_used else "local")

    answer_words = re.findall(r"\S+", answer_text)
    word_count = len(answer_words)
    unique_count = len(set(w.lower() for w in answer_words))
    question_tokens = set(_tokenize(question_text))
    answer_tokens = set(_tokenize(answer_text))
    ideal_tokens = set(_tokenize(ideal_answer))
    q_overlap = len(question_tokens.intersection(answer_tokens))
    i_overlap = len(ideal_tokens.intersection(answer_tokens))
    q_base = len(question_tokens) or 1
    i_base = len(ideal_tokens) or 1
    lexical_relevance = _clamp(int((q_overlap / q_base) * 100))
    lexical_work = _clamp(int((i_overlap / i_base) * 100))
    semantic_ideal = _clamp(int(round(
        _fast_semantic_similarity(
            _semantic_question_answer_text(question_text, answer_text),
            [_semantic_question_answer_text(question_text, ideal_answer)],
            cap=1,
        ) * 100
    ))) if ideal_answer else 0
    reference_lines = _collect_local_reference_lines()
    reference_score, sentence_checks, evaluation_source = _sentence_match_score(answer_text, reference_lines)
    relevance_score = _clamp(max(lexical_relevance, int(reference_score * 0.85)))
    work_score = _clamp(max(lexical_work, semantic_ideal, int(reference_score * 0.92)))

    fluency_score = _clamp(int(min(100, (word_count / 80.0) * 100)))
    if unique_count > 0 and word_count > 0:
        fluency_score = _clamp(int((fluency_score * 0.7) + ((unique_count / word_count) * 100 * 0.3)))
    hedges = {"maybe", "probably", "guess", "not sure", "i think"}
    lower_answer = answer_text.lower()
    hedge_hits = sum(1 for h in hedges if h in lower_answer)
    confidence_score = _clamp(75 - (hedge_hits * 12) + min(20, int(word_count / 20)))

    overall_score = _clamp(int(
        (work_score * 0.40) +
        (relevance_score * 0.25) +
        (fluency_score * 0.20) +
        (confidence_score * 0.15)
    ))

    missing_keywords = [k for k in sorted(question_tokens) if k not in answer_tokens][:8]
    matched_keywords = [k for k in sorted(question_tokens) if k in answer_tokens][:8]
    analysis_parts = [
        f"Matched {len(matched_keywords)} key terms from the question.",
        f"Answer length: {word_count} words.",
    ]
    if missing_keywords:
        analysis_parts.append("Missing focus terms: " + ", ".join(missing_keywords))
    if ideal_answer:
        analysis_parts.append(f"Ideal-answer alignment: {work_score}%")
    analysis_parts.append(f"Semantic alignment: {semantic_ideal}%")
    analysis_parts.append(f"Reference match: {reference_score}% ({evaluation_source})")
    analysis_text = " ".join(analysis_parts).strip()

    feedback = {
        "score": overall_score,
        "isCorrect": overall_score >= 65,
        "matchPercent": work_score,
        "analysis": analysis_text,
    }

    answers_map = session.get("answers", {})
    if not isinstance(answers_map, dict):
        answers_map = {}
        session["answers"] = answers_map
    answers_map[question_id] = {
        "answer": answer_text,
        "metrics": {
            "overall": overall_score,
            "work": work_score,
            "fluency": fluency_score,
            "relevance": relevance_score,
            "confidence": confidence_score,
        },
        "feedback": feedback,
        "answeredAt": int(time.time()),
    }
    SIMPLE_INTERVIEW_SESSIONS[session_id] = session

    first_q = questions[0] if questions else {}
    metrics = {
        "overall": overall_score,
        "work": work_score,
        "fluency": fluency_score,
        "relevance": relevance_score,
        "confidence": confidence_score,
    }

    score_by_qid: Dict[str, Dict[str, Any]] = {}
    totals = {"overall": 0.0, "work": 0.0, "fluency": 0.0, "relevance": 0.0, "confidence": 0.0}
    answered_count = 0
    for q_item in questions:
        qid = str(q_item.get("questionId", "")).strip()
        if not qid:
            continue
        a_item = answers_map.get(qid, {})
        if isinstance(a_item, dict) and a_item:
            m = a_item.get("metrics", {})
            if not isinstance(m, dict):
                m = {}
            m_overall = _safe_int(m.get("overall"), _safe_int(((a_item.get("feedback", {}) or {}).get("score")), 0))
            m_work = _safe_int(m.get("work"), 0)
            m_fluency = _safe_int(m.get("fluency"), 0)
            m_relevance = _safe_int(m.get("relevance"), 0)
            m_confidence = _safe_int(m.get("confidence"), 0)
            score_by_qid[qid] = {
                "overall": m_overall,
                "work": m_work,
                "fluency": m_fluency,
                "relevance": m_relevance,
                "confidence": m_confidence,
                "candidateAnswer": str(a_item.get("answer", "")).strip(),
                "idealAnswer": str(q_item.get("idealAnswer", "")).strip(),
            }
            totals["overall"] += m_overall
            totals["work"] += m_work
            totals["fluency"] += m_fluency
            totals["relevance"] += m_relevance
            totals["confidence"] += m_confidence
            answered_count += 1

    session_metrics = {
        "overall": _clamp(int(round(totals["overall"] / answered_count))) if answered_count else 0,
        "work": _clamp(int(round(totals["work"] / answered_count))) if answered_count else 0,
        "fluency": _clamp(int(round(totals["fluency"] / answered_count))) if answered_count else 0,
        "relevance": _clamp(int(round(totals["relevance"] / answered_count))) if answered_count else 0,
        "confidence": _clamp(int(round(totals["confidence"] / answered_count))) if answered_count else 0,
    }

    question_scores: List[Dict[str, Any]] = []
    question_matrix: List[Dict[str, Any]] = []
    all_answers: List[Dict[str, Any]] = []
    for idx, q_item in enumerate(questions, start=1):
        qid = str(q_item.get("questionId", "")).strip()
        qtxt = str(q_item.get("question", "")).strip()
        qs = score_by_qid.get(qid)
        if not qs:
            continue
        row = {
            "questionId": qid,
            "questionNumber": idx,
            "question": qtxt,
            "answered": True,
            "score": qs.get("overall"),
        }
        question_scores.append(row)
        question_matrix.append({
            "questionId": qid,
            "questionNumber": idx,
            "question": qtxt,
            "overall": qs.get("overall", 0),
            "work": qs.get("work", 0),
            "fluency": qs.get("fluency", 0),
            "relevance": qs.get("relevance", 0),
            "confidence": qs.get("confidence", 0),
            "answered": True,
        })
        all_answers.append({
            "questionId": qid,
            "questionNumber": idx,
            "question": qtxt,
            "candidateAnswer": qs.get("candidateAnswer", ""),
            "idealAnswer": qs.get("idealAnswer", ""),
            "score": qs.get("overall", 0),
        })

    _learn_offline_qa_memory(
        role_key=role_key,
        question_text=question_text,
        ideal_answer=ideal_answer,
        candidate_answer=answer_text,
        score=overall_score,
        metrics=metrics,
        sentence_checks=sentence_checks,
    )
    _rl_update_question(role_key, question_text, overall_score, overall_score >= 65)

    return {
        "ok": True,
        "sessionId": session_id,
        "questionId": question_id,
        "question": question_text,
        "firstQuestion": {
            "questionId": str(first_q.get("questionId", "")).strip(),
            "question": str(first_q.get("question", "")).strip(),
        },
        "metrics": metrics,
        "overallScore": metrics["overall"],
        "workScore": metrics["work"],
        "fluencyScore": metrics["fluency"],
        "relevanceScore": metrics["relevance"],
        "confidenceScore": metrics["confidence"],
        "answeredCount": answered_count,
        "questionScores": question_scores,
        "questionMatrix": question_matrix,
        "sessionMetrics": session_metrics,
        "globalMatrix": session_metrics,
        "sessionOverall": session_metrics["overall"],
        "allAnswers": all_answers,
        "evaluationSource": evaluation_source,
        "sentenceChecks": sentence_checks,
        "feedback": feedback,
    }


@app.get("/rag/sources", include_in_schema=False)
def rag_sources() -> Dict[str, Any]:
    return {"sources": RAG_SOURCE_MAP}


@app.post("/rag/sources", include_in_schema=False)
def update_rag_sources(payload: Dict[str, Any]) -> Dict[str, Any]:
    global RAG_SOURCE_MAP, URL_CONTENT_CACHE, URL_CONTENT_INDEX
    if not isinstance(payload, dict): return {"status": "ignored", "reason": "payload must be object"}
    current = _load_source_map()
    for key, val in payload.items():
        if not isinstance(val, dict): continue
        current[str(key).strip().lower()] = {
            "sources":  list(dict.fromkeys(s.strip() for s in val.get("sources",[]) if str(s).strip())),
            "keywords": list(dict.fromkeys(w.strip().lower() for w in val.get("keywords",[]) if str(w).strip())),
        }
    RAG_SOURCES_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    RAG_SOURCE_MAP = current; URL_CONTENT_CACHE = {}; URL_CONTENT_INDEX = {}
    return {"status": "updated", "roles": sorted(RAG_SOURCE_MAP.keys())}


@app.post("/rag/train-once", include_in_schema=False)
def rag_train_once(payload: OneTimeTrainingRequest) -> Dict[str, Any]:
    global RAG_SOURCE_MAP, URL_CONTENT_CACHE, URL_CONTENT_INDEX, TRAINING_STORE_MAP
    started = time.perf_counter()
    run_label = _first_non_blank(payload.runLabel, "manual")
    adaptive = _adaptive_learning_policy()
    limits = _resource_limits()
    question_target = max(5, min(int(payload.questionCountPerDomain * _safe_float(adaptive.get("questionMultiplier"), 1.0)), limits["qtarget"]))
    discover_limit = max(2, min(int(payload.discoveredUrlLimitPerDomain * _safe_float(adaptive.get("discoverMultiplier"), 1.0)), limits["discover"]))
    requested_domains = [str(d).strip().lower() for d in payload.domains if str(d).strip()]
    domains = requested_domains or ["java", "sales", "business_development", "voice_process", "chat_process", "flutter", "react"]
    logger.info(
        "[TRAIN][%s] start domains=%s qTarget=%s discover=%s discoverLimit=%s includeDefaults=%s forceRefresh=%s role=%s department=%s stream=%s profile=%s mode=%s knowledge=%.2f",
        run_label, domains, question_target, payload.discoverUrls, discover_limit, payload.includeDefaults, payload.forceRefresh,
        payload.role, payload.department, payload.stream, RESOURCE_PROFILE, adaptive.get("mode", "slow"), _safe_float(adaptive.get("knowledgeScore"), 0.0)
    )
    _append_learning_log("train_start", {"runLabel": run_label, "domains": domains, "qTarget": question_target, "discoverLimit": discover_limit, "profile": RESOURCE_PROFILE, "mode": adaptive.get("mode", "slow"), "knowledgeScore": _safe_float(adaptive.get("knowledgeScore"), 0.0)})

    current_sources = _merge_source_map_with_market_defaults(_load_source_map(), payload.includeDefaults)
    defaults = _default_market_sources()
    logger.info("[TRAIN][%s] step=load_sources roles=%s", run_label, sorted(current_sources.keys()))

    if payload.forceRefresh:
        URL_CONTENT_CACHE = {}
        URL_CONTENT_INDEX = {}
        logger.info("[TRAIN][%s] step=cache_reset done", run_label)

    summary: Dict[str, Any] = {}
    trained_domains: Dict[str, Any] = {}
    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    interrupted = False
    interrupt_reason = ""

    def _should_pause_training() -> bool:
        return ACTIVE_STT_WS_COUNT > 0 or _has_active_interview_session()

    for domain in domains:
        if _should_pause_training():
            interrupted = True
            interrupt_reason = "active interview session"
            logger.info("[TRAIN][%s] paused before domain=%s due to active interview", run_label, domain)
            break
        logger.info("[TRAIN][%s][%s] step=domain_start", run_label, domain)
        cfg = current_sources.get(domain, {"sources": [], "keywords": []})
        extra_urls = payload.sourceUrls.get(domain, []) if isinstance(payload.sourceUrls, dict) else []
        role_for_domain = _first_non_blank(payload.role, f"{domain} engineer")
        dept_for_domain = _first_non_blank(payload.department, "engineering")
        stream_for_domain = _first_non_blank(payload.stream, domain)
        logger.info("[TRAIN][%s][%s] step=collect_input_urls extra=%s cfg=%s", run_label, domain, len(extra_urls), len(cfg.get("sources", []) or []))
        discovered_urls = _discover_market_urls(
            domain,
            discover_limit,
            role=role_for_domain,
            department=dept_for_domain,
            stream=stream_for_domain,
        ) if payload.discoverUrls else []
        if payload.discoverUrls:
            logger.info("[TRAIN][%s][%s] step=discover_urls found=%s", run_label, domain, len(discovered_urls))
        merged_urls = list(dict.fromkeys(
            [
                *(cfg.get("sources", []) or []),
                *(defaults.get(domain, {}).get("sources", []) or []),
                *[str(u).strip() for u in extra_urls if str(u).strip()],
                *discovered_urls,
            ]
        ))
        filtered_urls: List[str] = []
        for u in merged_urls:
            ok_url, rsn = _is_url_allowed(u)
            if not ok_url:
                _record_source_quality(u, domain, False, rsn)
                _append_learning_log("url_blocked", {"url": u, "domain": domain, "reason": rsn})
                continue
            filtered_urls.append(u)
        repeat_days = max(1, _safe_int(adaptive.get("repeatDays"), NOVEL_URL_REPEAT_DAYS))
        novel_urls: List[str] = []
        known_urls: List[str] = []
        for u in filtered_urls:
            if _is_url_novel_for_learning(u, repeat_days=repeat_days):
                novel_urls.append(u)
            else:
                known_urls.append(u)
        merged_urls = [*novel_urls, *known_urls]
        logger.info(
            "[TRAIN][%s][%s] step=novelty_split novel=%s known=%s repeatDays=%s",
            run_label, domain, len(novel_urls), len(known_urls), repeat_days
        )
        logger.info("[TRAIN][%s][%s] step=merge_urls total=%s", run_label, domain, len(merged_urls))
        current_sources[domain] = {
            "sources": merged_urls,
            "keywords": list(dict.fromkeys([*(cfg.get("keywords", []) or []), *(defaults.get(domain, {}).get("keywords", []) or []), domain])),
        }

        gathered_lines: List[str] = []
        fetched = 0
        fetch_cap = int(max(10, min(TRAIN_FETCH_URL_LIMIT, limits["fetch"])) * _safe_float(adaptive.get("fetchMultiplier"), 1.0))
        fetch_limit = min(len(merged_urls), max(8, fetch_cap))
        daily_new_target = max(1, _safe_int(adaptive.get("dailyNewUrlTarget"), DAILY_NEW_URL_TARGET))
        fetch_pool = [*novel_urls, *known_urls]
        if len(novel_urls) >= min(fetch_limit, daily_new_target):
            fetch_pool = [*novel_urls, *known_urls]
        logger.info("[TRAIN][%s][%s] step=fetch_urls limit=%s", run_label, domain, fetch_limit)
        for url in fetch_pool[: fetch_limit]:
            if _should_pause_training():
                interrupted = True
                interrupt_reason = "active interview session"
                logger.info("[TRAIN][%s][%s] paused mid-domain due to active interview", run_label, domain)
                break
            lines = _fetch_url(url) if payload.forceRefresh else (_fetch_url_cached_only(url) or _fetch_url(url))
            if lines:
                fetched += 1
                gathered_lines.extend(lines)
                logger.info("[TRAIN][%s][%s] step=fetch_url_ok lines=%s url=%s", run_label, domain, len(lines), url)
                _record_source_quality(url, domain, True, "fetched")
                _append_learning_log("url_pass", {"url": url, "domain": domain, "lines": len(lines)})
                _mark_learning_url(url, domain, True, "fetched")
            else:
                logger.info("[TRAIN][%s][%s] step=fetch_url_empty url=%s", run_label, domain, url)
                _record_source_quality(url, domain, False, "fetch_empty")
                _append_learning_log("url_fail", {"url": url, "domain": domain, "reason": "fetch_empty"})
                _mark_learning_url(url, domain, False, "fetch_empty")
        if interrupted:
            break

        logger.info("[TRAIN][%s][%s] step=mine_questions input_lines=%s", run_label, domain, len(gathered_lines))
        mined_questions = _extract_questions_from_lines(gathered_lines, question_target)
        mined_questions = _prioritize_novel_questions(domain, mined_questions, question_target)
        logger.info("[TRAIN][%s][%s] step=mine_questions_done count=%s", run_label, domain, len(mined_questions))
        if len(mined_questions) < question_target and gathered_lines:
            logger.info("[TRAIN][%s][%s] step=llm_synthesis needed=%s", run_label, domain, question_target - len(mined_questions))
            synth_payload = InterviewQuestionsRequest(
                jobTitle=f"{domain} interview",
                language=domain,
                domain=domain,
                domainLabel=domain,
                jobDescription="\n".join(gathered_lines[:80]),
            )
            llm_items = _llm_generate_questions(synth_payload, gathered_lines)
            seen_q = {_normalize(q) for q in mined_questions}
            for item in llm_items:
                q = _normalize_question(item.get("question", ""))
                if not q:
                    continue
                k = _normalize(q)
                if k in seen_q:
                    continue
                seen_q.add(k)
                mined_questions.append(q)
                if len(mined_questions) >= question_target:
                    break
            logger.info("[TRAIN][%s][%s] step=llm_synthesis_done total=%s", run_label, domain, len(mined_questions))
        mined_questions = _prioritize_novel_questions(domain, mined_questions, question_target)

        qa_bank: List[Dict[str, str]] = []
        for q in mined_questions[:question_target]:
            ans = _limit_words(
                _build_best_answer(
                    q,
                    gathered_lines[:120],
                    BestAnswerRequest(
                        domain=domain,
                        jobTitle=role_for_domain,
                        jobDescription="",
                        maxWords=150,
                    ),
                ).get("bestAnswer", ""),
                180,
            )
            if _is_valid_qa_pair(q, ans):
                qa_bank.append({"question": q, "answer": ans})
        trained_domains[domain] = {
            "trainedAt": trained_at,
            "role": role_for_domain,
            "department": dept_for_domain,
            "stream": stream_for_domain,
            "sources": merged_urls,
            "cachedLineCount": len(gathered_lines),
            "discoveredSourceCount": len(discovered_urls),
            "questions": mined_questions[:question_target],
            "qaBank": qa_bank,
        }
        summary[domain] = {
            "role": role_for_domain,
            "department": dept_for_domain,
            "stream": stream_for_domain,
            "sources": len(merged_urls),
            "discoveredSources": len(discovered_urls),
            "fetchedSources": fetched,
            "cachedLineCount": len(gathered_lines),
            "questionCount": len(mined_questions[:question_target]),
        }
        logger.info(
            "[TRAIN][%s][%s] step=domain_done sources=%s fetched=%s lines=%s questions=%s",
            run_label, domain, len(merged_urls), fetched, len(gathered_lines), len(mined_questions[:question_target])
        )

    if interrupted:
        total_ms = round((time.perf_counter() - started) * 1000)
        _append_learning_log("train_paused_for_interview", {
            "runLabel": run_label,
            "reason": interrupt_reason,
            "durationMs": total_ms,
            "completedDomains": list(summary.keys()),
        })
        return {
            "status": "paused",
            "reason": interrupt_reason or "active interview",
            "trainedAt": trained_at,
            "runLabel": run_label,
            "durationMs": total_ms,
            "domains": summary,
            "storePath": str(TRAINING_STORE_FILE),
        }

    current_sources.setdefault("default", {"sources": [], "keywords": []})
    logger.info("[TRAIN][%s] step=save_rag_sources path=%s", run_label, RAG_SOURCES_FILE)
    RAG_SOURCES_FILE.write_text(json.dumps(current_sources, indent=2), encoding="utf-8")
    RAG_SOURCE_MAP = current_sources
    logger.info("[TRAIN][%s] step=refresh_index", run_label)
    _refresh_url_index()

    logger.info("[TRAIN][%s] step=save_training_store path=%s", run_label, TRAINING_STORE_FILE)
    store = _load_training_store()
    store["trainedAt"] = trained_at
    all_domains = store.get("domains", {})
    if not isinstance(all_domains, dict):
        all_domains = {}
    all_domains.update(trained_domains)
    store["domains"] = all_domains
    _save_training_store(store)
    TRAINING_STORE_MAP = store
    total_ms = round((time.perf_counter() - started) * 1000)
    logger.info("[TRAIN][%s] done domains=%s durationMs=%s", run_label, len(domains), total_ms)
    _append_learning_log("train_done", {"runLabel": run_label, "domains": len(domains), "durationMs": total_ms})

    return {
        "status": "trained",
        "trainedAt": trained_at,
        "runLabel": run_label,
        "durationMs": total_ms,
        "domains": summary,
        "storePath": str(TRAINING_STORE_FILE),
    }


@app.get("/rag/training-store", include_in_schema=False)
def rag_training_store() -> Dict[str, Any]:
    return _load_training_store()


@app.get("/rag/learning-log", include_in_schema=False)
def rag_learning_log(limit: int = 200, event: str = "", domain: str = "") -> Dict[str, Any]:
    data = _load_learning_log()
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        items = []
    limit = max(1, min(int(limit), 2000))
    want_event = _normalize(event)
    want_domain = _normalize(domain)
    filtered: List[Dict[str, Any]] = []
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        ev = _normalize(item.get("event", ""))
        details = item.get("details", {})
        details_domain = _normalize(details.get("domain", "")) if isinstance(details, dict) else ""
        if want_event and ev != want_event:
            continue
        if want_domain and details_domain != want_domain:
            continue
        filtered.append(item)
        if len(filtered) >= limit:
            break
    return {
        "ok": True,
        "path": str(LEARNING_LOG_FILE),
        "maxItems": LEARNING_LOG_MAX_ITEMS,
        "totalItems": len(items),
        "returnedItems": len(filtered),
        "filters": {"event": want_event, "domain": want_domain},
        "items": filtered,
    }


@app.get("/rag/source-quality", include_in_schema=False)
def rag_source_quality(limit: int = 200, domain: str = "", status: str = "all") -> Dict[str, Any]:
    data = _load_source_quality()
    urls = data.get("urls", {}) if isinstance(data, dict) else {}
    if not isinstance(urls, dict):
        urls = {}
    limit = max(1, min(int(limit), 2000))
    want_domain = _normalize(domain)
    want_status = _normalize(status)
    if want_status not in {"all", "pass", "fail"}:
        want_status = "all"

    rows: List[Dict[str, Any]] = []
    total_pass = 0
    total_fail = 0
    for url, raw in urls.items():
        if not isinstance(raw, dict):
            continue
        row_domain = _normalize(raw.get("domain", ""))
        p = max(0, _safe_int(raw.get("pass"), 0))
        f = max(0, _safe_int(raw.get("fail"), 0))
        total = p + f
        total_pass += p
        total_fail += f
        if want_domain and row_domain != want_domain:
            continue
        if want_status == "pass" and p <= 0:
            continue
        if want_status == "fail" and f <= 0:
            continue
        rows.append({
            "url": str(url),
            "domain": row_domain,
            "pass": p,
            "fail": f,
            "totalHits": total,
            "passRate": round((p / total), 4) if total > 0 else 0.0,
            "lastStatus": str(raw.get("lastStatus", "")),
            "lastReason": str(raw.get("lastReason", "")),
            "lastTs": int(_safe_int(raw.get("lastTs"), 0)),
        })

    rows.sort(key=lambda r: (r["fail"], r["totalHits"], -r["passRate"]), reverse=True)
    top = rows[:limit]
    noisy = [r for r in rows if r["fail"] >= 2 and r["fail"] >= r["pass"]][:50]
    return {
        "ok": True,
        "path": str(SOURCE_QUALITY_FILE),
        "filters": {"domain": want_domain, "status": want_status},
        "summary": {
            "urlCount": len(urls),
            "totalPassHits": total_pass,
            "totalFailHits": total_fail,
        },
        "returnedItems": len(top),
        "items": top,
        "noisySources": noisy,
    }


@app.get("/rag/learning-policy", include_in_schema=False)
def rag_learning_policy() -> Dict[str, Any]:
    store = _load_learning_policy()
    policy = _adaptive_learning_policy()
    daily = store.get("daily", {}) if isinstance(store, dict) else {}
    today = _today_bucket()
    today_node = daily.get(today, {}) if isinstance(daily, dict) else {}
    return {
        "ok": True,
        "path": str(LEARNING_POLICY_FILE),
        "policy": policy,
        "startedAtTs": _safe_int(store.get("startedAtTs"), 0) if isinstance(store, dict) else 0,
        "urlKnowledgeCount": len(store.get("urls", {})) if isinstance(store, dict) and isinstance(store.get("urls", {}), dict) else 0,
        "questionKnowledgeCount": len(store.get("questions", {})) if isinstance(store, dict) and isinstance(store.get("questions", {}), dict) else 0,
        "today": today,
        "todayNewUrls": len(today_node.get("newUrlHashes", [])) if isinstance(today_node, dict) and isinstance(today_node.get("newUrlHashes", []), list) else 0,
        "todayQuestions": len(today_node.get("questionHashes", [])) if isinstance(today_node, dict) and isinstance(today_node.get("questionHashes", []), list) else 0,
    }


@app.get("/reinforcement/state", include_in_schema=False)
def reinforcement_state() -> Dict[str, Any]:
    return _load_reinforcement_state()


@app.get("/ml/learning-stats", include_in_schema=False)
def ml_learning_stats() -> Dict[str, Any]:
    return _learning_stats()


register_ml_routes(app, {
    "train_small_model": _train_small_model,
    "train_online_sources": _train_small_model_from_online_sources,
    "online_training_status": _online_training_status,
    "load_small_model": _load_small_model,
    "small_model_file": SMALL_MODEL_FILE,
    "safe_int": _safe_int,
    "worker_startup_status": WORKER_STARTUP_STATUS,
    "small_model_source_registry": _small_model_source_registry,
})


@app.post("/interview/best-answer", include_in_schema=False)
def interview_best_answer(payload: BestAnswerRequest) -> Dict[str, Any]:
    global TRAINING_STORE_MAP
    if not isinstance(TRAINING_STORE_MAP, dict) or not TRAINING_STORE_MAP:
        TRAINING_STORE_MAP = _load_training_store()
    domain = _normalize(payload.domain)
    store_domains = TRAINING_STORE_MAP.get("domains", {}) if isinstance(TRAINING_STORE_MAP, dict) else {}
    domain_store = store_domains.get(domain, {}) if isinstance(store_domains, dict) else {}

    context_lines: List[str] = []
    if payload.jobDescription.strip():
        context_lines.extend(_split_sentences(payload.jobDescription))

    for url in [*payload.questionUrls, *payload.contextUrls]:
        u = str(url).strip()
        if not u:
            continue
        context_lines.extend(_fetch_url_cached_only(u) or _fetch_url(u))

    if isinstance(domain_store, dict):
        context_lines.extend([str(x).strip() for x in domain_store.get("questions", []) if str(x).strip()])
        for src in domain_store.get("sources", []) or []:
            context_lines.extend(_fetch_url_cached_only(str(src)) or [])

    questions = [q for q in [payload.question, *payload.questions] if str(q).strip()]
    if not questions:
        questions = [str(q).strip() for q in domain_store.get("questions", [])[:5]] if isinstance(domain_store, dict) else []
    if not questions:
        return {"ok": False, "reason": "No question provided and no trained questions found for domain."}

    answers = [_build_best_answer(str(q), context_lines, payload) for q in questions]
    return {
        "ok": True,
        "domain": domain or "default",
        "count": len(answers),
        "answers": answers,
        "sourcesUsed": len({*payload.questionUrls, *payload.contextUrls, *(domain_store.get("sources", []) if isinstance(domain_store, dict) else [])}),
    }


# ── Engine 1: question generation ─────────────────────────────────────────────
@app.post("/interview/questions", include_in_schema=False)
def interview_questions(payload: InterviewQuestionsRequest) -> Dict[str, Any]:
    _touch_activity()
    role_key = _infer_role_key(payload.jobTitle, payload.domainLabel or payload.domain, payload.jobDescription, payload.department)
    session_key = _mark_interview_session_active(payload, role_key)
    logger.info("[SESSION] active interview session marked key=%s role=%s", session_key, role_key)
    try:
        result = _question_engine.build(payload)
        append_foundation_example(
            "interview_question_generation",
            payload.jobDescription,
            payload.model_dump(),
            result,
            resolve_model_profile("interview_question_generation", payload=payload.model_dump()).get("model", MODEL_NAME),
            "interview-route",
        )
        return result
    except Exception as ex:
        logger.exception("QuestionEngine build failed; fallback disabled: %s", ex)
        raise HTTPException(status_code=503, detail=f"Interview question generation failed: {type(ex).__name__}")


# ── Engine 2: counter / follow-up questions ───────────────────────────────────
@app.post("/interview/counter-questions", include_in_schema=False)
def interview_counter_questions(payload: CounterQuestionRequest) -> Dict[str, Any]:
    result = _counter_engine.generate(payload)
    append_foundation_example(
        "interview_counter_questions",
        payload.question,
        payload.model_dump(),
        result,
        resolve_model_profile("interview_counter_questions", payload=payload.model_dump()).get("model", MODEL_NAME),
        "interview-route",
    )
    return result


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.post("/interview/evaluate", include_in_schema=False)
def interview_evaluate(payload: InterviewRequest) -> Dict[str, Any]:
    _touch_activity()
    role_key = _infer_role_key(payload.jobTitle, payload.domainLabel or payload.domain, payload.jobDescription, payload.department)
    result = _evaluate(payload)
    append_foundation_example(
        "interview_evaluation",
        payload.jobDescription,
        payload.model_dump(),
        result,
        resolve_model_profile("interview_evaluation", payload=payload.model_dump()).get("model", MODEL_NAME),
        "interview-route",
    )
    _mark_interview_session_completed(payload, role_key)
    logger.info("[SESSION] interview session completed role=%s", role_key)
    return result


# ── Combined: evaluate + auto-generate counter questions ──────────────────────
@app.post("/interview/evaluate-with-followup", include_in_schema=False)
def interview_evaluate_with_followup(payload: InterviewRequest) -> Dict[str, Any]:
    """Evaluate all answers AND generate counter-questions for each answer in one call."""
    _touch_activity()
    role_key = _infer_role_key(payload.jobTitle, payload.domainLabel or payload.domain, payload.jobDescription, payload.department)
    result = _evaluate(payload)
    if not COUNTER_Q_ENABLED:
        result["counterQuestions"] = []
        _mark_interview_session_completed(payload, role_key)
        logger.info("[SESSION] interview session completed role=%s", role_key)
        return result

    counter_map: List[Dict[str, Any]] = []
    for qa in result.get("questionAnalysis", []):
        counter_req = CounterQuestionRequest(
            questionId       = qa.get("questionId",""),
            question         = qa.get("question",""),
            answer           = qa.get("answer",""),
            answerScore      = qa.get("score", -1),
            jobTitle         = payload.jobTitle,
            jobDescription   = payload.jobDescription,
            language         = payload.language,
            domain           = payload.domain,
            department       = payload.department,
            customPrompts    = payload.customPrompts,
            resume           = payload.resume,
            candidateProfile = payload.candidateProfile,
            maxQuestions     = COUNTER_Q_PER_ANSWER,
        )
        counter_result = _counter_engine.generate(counter_req)
        counter_map.append({
            "questionId": qa.get("questionId",""),
            "strategy":   counter_result.get("strategy",""),
            "questions":  counter_result.get("questions",[]),
        })
    result["counterQuestions"] = counter_map
    append_foundation_example(
        "interview_evaluate_with_followup",
        payload.jobDescription,
        payload.model_dump(),
        result,
        resolve_model_profile("interview_evaluate_with_followup", payload=payload.model_dump()).get("model", MODEL_NAME),
        "interview-route",
    )
    _mark_interview_session_completed(payload, role_key)
    logger.info("[SESSION] interview session completed role=%s", role_key)
    return result


# ── STT ───────────────────────────────────────────────────────────────────────
@app.post("/speech/transcribe", include_in_schema=False)
def speech_transcribe(payload: SpeechTranscribeRequest) -> Dict[str, Any]:
    audio_base64 = str(payload.audioBase64 or "").strip()
    audio_path = str(payload.audioPath or "").strip()
    if audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as ex:
            return {"ok": False, "reason": f"invalid_audio_base64:{ex}"}
        return _transcribe_audio_bytes(audio_bytes, payload.contentType, payload.language)
    if audio_path:
        return _transcribe_audio_path(audio_path, payload.language)
    return {"ok": False, "reason": "audioBase64 or audioPath is required"}


@app.websocket("/speech/ws/stt")
async def speech_ws_stt(websocket: WebSocket):
    global ACTIVE_STT_WS_COUNT
    await websocket.accept()
    ACTIVE_STT_WS_COUNT += 1
    generation = 0
    language = ""
    content_type = "audio/webm"
    stt_ok, stt_reason = _ensure_whisper()
    if not stt_ok:
        await websocket.send_json({"type": "ack", "ok": False, "reason": stt_reason})
        await websocket.close()
        ACTIVE_STT_WS_COUNT = max(0, ACTIVE_STT_WS_COUNT - 1)
        return
    await websocket.send_json({
        "type": "ack",
        "ok": True,
        "provider": "python-worker",
        "model": STT_MODEL_ID,
        "sttPriorityMode": STT_PRIORITY_MODE,
    })
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text_payload = message.get("text")
            bytes_payload = message.get("bytes")
            if text_payload is not None:
                try:
                    payload = json.loads(str(text_payload or "{}"))
                except Exception:
                    continue
                event_type = str(payload.get("type") or "").strip().lower()
                if event_type == "ping":
                    await websocket.send_json({"type": "pong", "ok": True})
                    continue
                if event_type == "meta":
                    generation = int(payload.get("generation") or generation or 0)
                    language = str(payload.get("language") or language or "")
                    content_type = str(payload.get("contentType") or content_type or "audio/webm")
                    await websocket.send_json({"type": "ack", "ok": True, "generation": generation})
                    continue
                continue
            if bytes_payload is None:
                continue
            result = await asyncio.to_thread(_transcribe_audio_bytes, bytes(bytes_payload), content_type, language)
            result.setdefault("type", "transcript")
            result.setdefault("generation", generation)
            await websocket.send_json(result)
    except Exception as ex:
        try:
            await websocket.send_json({"type": "error", "ok": False, "reason": str(ex)})
        except Exception:
            pass
    finally:
        ACTIVE_STT_WS_COUNT = max(0, ACTIVE_STT_WS_COUNT - 1)
        try:
            await websocket.close()
        except Exception:
            pass


# ── TTS ───────────────────────────────────────────────────────────────────────
@app.post("/speech/synthesize", include_in_schema=False)
def speech_synthesize(payload: SpeechSynthesizeRequest) -> Dict[str, Any]:
    return {
        "ok": False,
        "reason": "Backend TTS is disabled. This app uses browser Web Speech API. Use Google Chrome for best performance."
    }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)

