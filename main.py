"""
DeepSeek Interview Worker  —  v5.0.0
Dual-Engine Architecture:
  • QuestionEngine    → generates initial interview questions (RAG + JD + resume)
  • CounterEngine     → generates follow-up / probing questions from a candidate answer
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.request import urlretrieve

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# ── env / config ───────────────────────────────────────────────────────────────
def _env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(key: str, default: int, minimum: int = 0) -> int:
    return max(minimum, int(os.getenv(key, str(default))))

MODEL_NAME              = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
WORKER_HOST             = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT             = int(os.getenv("WORKER_PORT", "8099"))

RAG_FETCH_ONLINE        = _env_bool("RAG_FETCH_ONLINE", "true")
RAG_SOURCES_FILE        = Path(os.getenv("RAG_SOURCES_FILE", Path(__file__).with_name("rag_sources.json")))
RAG_FETCH_TIMEOUT       = _env_int("RAG_FETCH_TIMEOUT_SECONDS", 8, 1)
RAG_MAX_URLS            = _env_int("RAG_MAX_URLS", 8, 1)

INTERVIEW_QUESTION_COUNT  = _env_int("INTERVIEW_QUESTION_COUNT", 14, 4)
TARGET_WORDS_PER_ANSWER   = _env_int("TARGET_WORDS_PER_ANSWER", 70, 20)
MIN_WORDS_PER_MINUTE      = _env_int("MIN_WORDS_PER_MINUTE", 50, 10)
MAX_ANSWER_WORDS          = _env_int("MAX_ANSWER_WORDS", 220, 80)

# Counter-question engine config
COUNTER_Q_ENABLED         = _env_bool("COUNTER_Q_ENABLED", "true")
COUNTER_Q_PER_ANSWER      = _env_int("COUNTER_Q_PER_ANSWER", 2, 1)
COUNTER_Q_DEPTH_THRESHOLD = _env_int("COUNTER_Q_DEPTH_THRESHOLD", 40, 0)   # score below → probe harder

ENABLE_STT              = _env_bool("ENABLE_STT")
ENABLE_TTS              = _env_bool("ENABLE_TTS")
STT_MODEL_ID            = os.getenv("STT_MODEL_ID", "openai/whisper-tiny")
STT_LANGUAGE            = os.getenv("STT_LANGUAGE", "en")
PIPER_BIN               = os.getenv("PIPER_BIN", "piper")
PIPER_VOICE_MODEL       = os.getenv("PIPER_VOICE_MODEL", str(Path(__file__).with_name("models") / "piper" / "voice.onnx"))
PIPER_VOICE_URL         = os.getenv("PIPER_VOICE_URL", "")
PIPER_VOICE_CONFIG_URL  = os.getenv("PIPER_VOICE_CONFIG_URL", "")
PIPER_SPEAKER           = os.getenv("PIPER_SPEAKER", "")
AUTO_INSTALL_PIPER      = _env_bool("AUTO_INSTALL_PIPER")
AUTO_INSTALL_PIPER_DEPS = _env_bool("AUTO_INSTALL_PIPER_DEPS")

DEFAULT_PIPER_ONNX_URL   = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
DEFAULT_PIPER_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
CORE_JAVA_REFERENCE_URL  = "https://www.interviewbit.com/java-interview-questions/"

LOCAL_PY_DEPS = Path(__file__).with_name("_pydeps")
LOCAL_PY_DEPS.mkdir(parents=True, exist_ok=True)
if str(LOCAL_PY_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_PY_DEPS))

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("deepseek-worker")

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
SUPPORTED_STT_LANGUAGES: Set[str] = {
    "english","chinese","german","spanish","russian","korean","french","japanese",
}

# ── global state ───────────────────────────────────────────────────────────────
RAG_SOURCE_MAP:       Dict[str, Dict[str, Any]] = {}
URL_CONTENT_CACHE:    Dict[str, List[str]] = {}
STT_PIPELINE_HANDLE   = None
WORKER_STARTUP_STATUS: Dict[str, Any] = {
    "sttReady": False, "ttsReady": False,
    "sttReason": "not initialized", "ttsReason": "not initialized",
}

app = FastAPI(title="DeepSeek Interview Worker", version="5.0.0")

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
    return text

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

def _is_core_java(job_title: str, job_desc: str, language: str, custom_prompts: Optional[List[str]] = None) -> bool:
    hay = " ".join([_normalize(job_title), _normalize(job_desc), _normalize(language),
                    " ".join(_normalize(p) for p in (custom_prompts or []))])
    has_java = "java" in hay
    has_core = "core java" in hay or "core-java" in hay
    excludes = {"javascript","spring boot","springboot","react","node"}
    return has_java and (has_core or not any(e in hay for e in excludes))

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

def _infer_role_key(job_title: str, domain_label: str, job_desc: str, department: str = "") -> str:
    hay = " ".join([_normalize(job_title), _normalize(domain_label), _normalize(job_desc), _normalize(department)])
    scored: List[tuple[int, str]] = []
    for key, cfg in RAG_SOURCE_MAP.items():
        if key == "default": continue
        kws = [str(k).lower() for k in cfg.get("keywords",[])] or [p for p in key.split("_") if p]
        score = sum(1 for k in kws if k and k in hay)
        if score > 0: scored.append((score, key))
    if not scored: return "default"
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
        if u in seen: continue
        seen.add(u); deduped.append(u)
    return deduped[:RAG_MAX_URLS]

def _fetch_url(url: str) -> List[str]:
    if not RAG_FETCH_ONLINE or not requests or not BeautifulSoup: return []
    if not url.startswith(("http://","https://")): return []
    if url in URL_CONTENT_CACHE: return URL_CONTENT_CACHE[url]
    lines: List[str] = []
    try:
        resp = requests.get(url, timeout=RAG_FETCH_TIMEOUT, headers={"User-Agent":"InterviewRAGWorker/5.0"})
        if resp.status_code >= 400: URL_CONTENT_CACHE[url] = []; return []
        soup = BeautifulSoup(resp.text, "html.parser")
        for node in soup.find_all(["h1","h2","h3","p","li"]):
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
            if len(text) < 45: continue
            lines.append(text)
            if len(lines) >= 80: break
    except Exception:
        lines = []
    URL_CONTENT_CACHE[url] = lines
    return lines

def _build_snippets(payload: Any) -> List[str]:
    snippets: List[str] = []
    role_key = _infer_role_key(
        getattr(payload,"jobTitle",""), getattr(payload,"domainLabel","") or getattr(payload,"domain",""),
        getattr(payload,"jobDescription",""), getattr(payload,"department",""),
    )
    urls = _collect_urls(payload, role_key)
    for u in urls: snippets.extend(_fetch_url(u))
    jd = getattr(payload, "jobDescription", "").strip()
    if jd: snippets.extend(s.strip() for s in re.split(r"[\n\r]+", jd) if s.strip())
    snippets.extend(_flatten(getattr(payload,"resume",{})))
    snippets.extend(_flatten(getattr(payload,"candidateProfile",{})))
    snippets.extend(_flatten(getattr(payload,"job",{})))
    dept = getattr(payload,"department","").strip()
    if dept: snippets.append(f"Department: {dept}")
    snippets.extend(str(p).strip() for p in getattr(payload,"customPrompts",[]) if str(p).strip())
    return snippets, role_key, urls

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
                       count: int = INTERVIEW_QUESTION_COUNT) -> List[str]:
        title = _first_non_blank(job_title, "this role")
        dept  = _first_non_blank(department, "the department")
        lang  = _first_non_blank(language, "technology")
        prompts = custom_prompts or []
        topics  = self._dynamic_topics(job_title, language, snippets + [department], top_n=max(4, count+1))
        core_java = _is_core_java(job_title, " ".join(snippets), language, prompts)

        templates = (
            [
                "Explain {topic} and where it appears in {title}.",
                "What mistakes do engineers make with {topic}, and how do you avoid them?",
                "Describe a hands-on {topic} experience and what you learned.",
                "What trade-offs arise when choosing {topic} implementations?",
                "How do you debug {topic}-related issues in production?",
            ] if core_java else [
                "How would you design {topic} for {title} using {lang}?",
                "Walk through testing {topic} end-to-end in {title} for {dept}.",
                "What trade-offs matter most when implementing {topic} in {title}?",
                "Describe a production incident involving {topic} and your resolution.",
                "How would you scale {topic} in a {title} system?",
            ]
        )
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
        snippets, role_key, urls = _build_snippets(payload)
        eff_lang = _effective_language(payload.jobTitle, payload.jobDescription, payload.language)
        core_java = _is_core_java(payload.jobTitle, payload.jobDescription or " ".join(snippets),
                                  eff_lang, payload.customPrompts)

        if core_java:
            question_texts = self._fetch_core_java_from_web(INTERVIEW_QUESTION_COUNT)
            if CORE_JAVA_REFERENCE_URL not in urls: urls.append(CORE_JAVA_REFERENCE_URL)
            if not question_texts:
                question_texts = self._build_dynamic(
                    payload.jobTitle, eff_lang, snippets, payload.department, payload.customPrompts)
        else:
            question_texts = self._extract_candidates(snippets, INTERVIEW_QUESTION_COUNT)

        # Deduplicate
        seen: Set[str] = set(); deduped: List[str] = []
        for q in question_texts:
            k = _normalize(q)
            if k not in seen: seen.add(k); deduped.append(q)
        question_texts = deduped

        # Top-up with dynamic fallback
        if len(question_texts) < INTERVIEW_QUESTION_COUNT:
            extra = self._build_dynamic(
                payload.jobTitle, eff_lang, snippets, payload.department, payload.customPrompts)
            for q in extra:
                k = _normalize(q)
                if k not in seen: seen.add(k); question_texts.append(q)
                if len(question_texts) >= INTERVIEW_QUESTION_COUNT: break

        questions = [
            {"questionId": f"q{i}", "question": q, "roleKey": role_key,
             "referenceLinks": urls, "department": payload.department or str(payload.job.get("department",""))}
            for i, q in enumerate(question_texts, 1)
        ]
        reason = ""
        if not questions:
            if not urls and not payload.jobDescription.strip():
                reason = "No source URLs or job description. Add JD text or links."
            elif not snippets:
                reason = "Could not fetch usable text from configured URLs."
            else:
                reason = "Fetched content did not contain enough relevant material."
        return {
            "questions": questions, "provider": "deepseek-worker-rag",
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
            "provider": "deepseek-worker-counter",
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

    answers = payload.answers if isinstance(payload.answers, list) else []
    if not answers:
        return {"overallScore":0,"technicalScore":0,"behavioralScore":5,"communicationScore":8,
                "problemSolvingScore":0,"passed":False,"summary":"No answers submitted.",
                "weakAreas":["No answer evidence","No role alignment"],
                "recommendations":["Submit complete answers tied to job description."],
                "feedback":{"strengths":["Session started"],"improvements":["Provide role-specific answers"]},
                "provider":"deepseek-worker-rag","model":MODEL_NAME,"roleKey":role_key,"reason":"No answers"}

    q_lookup: Dict[str, str] = {
        str(q.get("questionId","")).strip(): str(q.get("question","")).strip()
        for q in (payload.questions if isinstance(payload.questions, list) else [])
        if q.get("questionId") and q.get("question")
    }

    per_scores: List[int] = []; q_analysis: List[Dict[str, Any]] = []
    nonsense_count = low_speed_count = 0

    for idx, item in enumerate(answers):
        raw   = str(item.get("answer","")).strip()
        ans   = _limit_words(raw, MAX_ANSWER_WORDS)
        qid   = str(item.get("questionId","")).strip() or f"q{idx+1}"
        qtext = _first_non_blank(q_lookup.get(qid), str(item.get("question","")).strip())
        q_kws = _keywords(qtext, top_n=10)

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
            src_match = _overlap_ratio(ans, source_kws[:20])
            q_match   = _overlap_ratio(ans, q_kws[:10])
            match_ratio = q_match * 0.7 + src_match * 0.3
            ans_toks  = _tokenize(ans)
            len_bonus = min(1.0, len(ans_toks) / float(TARGET_WORDS_PER_ANSWER))
            if q_kws and q_match < 0.12:
                base = int(round(q_match * 120.0))
            elif match_ratio < 0.30:
                base = int(round(match_ratio * 100.0))
            else:
                base = int(round(30 + ((match_ratio - 0.30) / 0.70) * 70))
            score = _clamp(int(round(base * (0.5 + 0.5 * len_bonus))))
            if word_count < 6: score = min(score, 24)
            if q_kws and not any(k in ans_toks for k in q_kws[:3]): score = min(score, 28)

            sent_analysis = []
            sent_correct_count = 0
            for si, s in enumerate(sentences):
                qr = _overlap_ratio(s, q_kws[:10]); rr = _overlap_ratio(s, source_kws[:20])
                sr = qr * 0.65 + rr * 0.35; correct = sr >= 0.30
                if correct: sent_correct_count += 1
                sent_analysis.append({
                    "index": si+1, "sentence": s,
                    "matchPercent": int(round(sr * 100)),
                    "correct": correct, "right": correct, "wrong": not correct,
                })
            all_correct = bool(sentences) and sent_correct_count == len(sentences)
            if all_correct and word_count >= TARGET_WORDS_PER_ANSWER:
                score = 100
            else:
                cov = min(1.0, word_count / float(TARGET_WORDS_PER_ANSWER))
                score = _clamp(int(round(score * (0.5 + 0.5 * cov))))
            if dur > 0:
                wpm = (word_count * 60.0) / dur
                if wpm < MIN_WORDS_PER_MINUTE:
                    sf = max(0.4, wpm / float(MIN_WORDS_PER_MINUTE))
                    score = _clamp(int(round(score * sf))); low_speed_count += 1

        per_scores.append(score)
        mp = int(round(match_ratio * 100))
        is_correct = bool(all_correct and score >= 70) if not _is_nonsense(ans) else False
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
            "analysis": ("Answer is weakly aligned; needs concrete technical details."
                         if mp < 30 or (_is_nonsense(ans) or not all_correct)
                         else "Answer has strong sentence-level topical alignment."),
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
    if _overlap_ratio(payload.transcript, source_kws[:14]) < 0.15:
        improvements.append("Answers are weakly aligned to JD/source material.")
    if low_speed_count:    improvements.append(f"{low_speed_count} answers below {MIN_WORDS_PER_MINUTE} WPM penalized.")
    if technical < threshold: improvements.append("Increase depth: architecture, trade-offs, debugging.")
    if not improvements:   improvements.append("Add quantified impact and deeper implementation detail.")

    strengths = (["Strong relevance to source-linked topics."] if overall >= threshold else []) + \
                (["Responses were reasonably clear."] if communication >= 65 else []) or \
                ["All questions were attempted."]

    return {
        "overallScore": overall, "technicalScore": technical,
        "behavioralScore": behavioral, "communicationScore": communication,
        "problemSolvingScore": problem_solving, "passed": overall >= threshold,
        "summary": "Scored against RAG/JD relevance and answer quality.",
        "weakAreas": improvements[:3],
        "recommendations": [
            "Use STAR format with measurable outcomes.",
            "Reference job-specific tools and responsibilities.",
            "Explain technical trade-offs and debugging steps clearly.",
        ],
        "feedback": {"strengths": strengths, "improvements": improvements},
        "provider": "deepseek-worker-rag", "model": MODEL_NAME,
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
    if not ENABLE_STT: return False, "STT disabled"
    if snapshot_download is None: return False, "huggingface_hub unavailable"
    try:
        snapshot_download(repo_id=STT_MODEL_ID,
            allow_patterns=["config.json","generation_config.json","preprocessor_config.json",
                            "tokenizer.json","tokenizer_config.json","special_tokens_map.json",
                            "vocab.json","merges.txt","normalizer.json","*.model",
                            "model.safetensors","pytorch_model.bin"],
            ignore_patterns=["*.h5","*.msgpack","*.onnx","*.ot","*.tflite","*.ckpt"])
        return True, f"Whisper ready: {STT_MODEL_ID}"
    except Exception as ex:
        return False, f"Whisper failed: {ex}"

def _resolve_piper() -> str:
    candidates: List[str] = []
    if PIPER_BIN.strip(): candidates.append(os.path.expandvars(PIPER_BIN.strip()))
    wd = Path(__file__).resolve().parent
    for name in ["piper.exe","piper","bin/piper.exe","bin/piper"]:
        candidates.append(str(wd / name))
    candidates.extend(["piper","piper.exe","piper-tts","piper-tts.exe"])
    seen: Set[str] = set()
    for c in candidates:
        if c.lower() in seen: continue
        seen.add(c.lower())
        if Path(c).exists(): return c
        if shutil.which(c): return c
    return ""

def _pip_install(*packages: str) -> tuple[bool, str]:
    def _run(extra: List[str]) -> int:
        return subprocess.run([sys.executable,"-m","pip","install","--upgrade",*extra,*packages],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=600).returncode
    if _run(["--target", str(LOCAL_PY_DEPS)]) == 0: return True, "installed"
    if _run([]) == 0: return True, "installed (global)"
    return False, "pip install failed"

def _ensure_piper() -> tuple[bool, str]:
    global PIPER_BIN, PIPER_VOICE_MODEL
    if not ENABLE_TTS: return False, "TTS disabled"
    resolved = _resolve_piper()
    if not resolved and importlib.util.find_spec("piper") is None:
        if AUTO_INSTALL_PIPER: _pip_install("piper-tts")
        resolved = _resolve_piper()
    if not resolved and importlib.util.find_spec("piper") is None:
        return False, "Piper not found"
    PIPER_BIN = resolved or "__python_module__:piper"
    model_path = Path(PIPER_VOICE_MODEL)
    if not model_path.exists():
        for found in sorted(model_path.parent.glob("*.onnx")):
            model_path = found; break
    model_url  = _first_non_blank(PIPER_VOICE_URL, DEFAULT_PIPER_ONNX_URL)
    config_url = _first_non_blank(PIPER_VOICE_CONFIG_URL, f"{model_url}.json", DEFAULT_PIPER_CONFIG_URL)
    for path, url in [(model_path, model_url), (Path(str(model_path)+".json"), config_url)]:
        if path.exists() or not url: continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            urlretrieve(url, tmp)
            if tmp.exists() and tmp.stat().st_size > 0: tmp.replace(path)
        except Exception as ex:
            if path == model_path: return False, f"Voice download failed: {ex}"
            logger.warning("Config download failed: %s", ex)
    PIPER_VOICE_MODEL = str(model_path)
    if model_path.exists(): return True, f"Piper ready: {model_path}"
    return False, f"Voice model missing: {model_path}"

def _bootstrap_models() -> None:
    global RAG_SOURCE_MAP
    RAG_SOURCE_MAP = _load_source_map()
    stt_ok, stt_reason = _ensure_whisper()
    tts_ok, tts_reason = _ensure_piper()
    WORKER_STARTUP_STATUS.update(sttReady=stt_ok, ttsReady=tts_ok,
                                  sttReason=stt_reason, ttsReason=tts_reason)
    logger.info("STT: %s", stt_reason); logger.info("TTS: %s", tts_reason)

def _get_stt_pipeline():
    global STT_PIPELINE_HANDLE
    if STT_PIPELINE_HANDLE is None:
        if hf_pipeline is None: raise RuntimeError("transformers unavailable")
        STT_PIPELINE_HANDLE = hf_pipeline("automatic-speech-recognition", model=STT_MODEL_ID)
    return STT_PIPELINE_HANDLE

def _suffix_from_ct(ct: str) -> str:
    lo = str(ct or "").lower()
    if "webm" in lo: return ".webm"
    if "ogg"  in lo: return ".ogg"
    if "mpeg" in lo or "mp3" in lo: return ".mp3"
    if "mp4"  in lo or "m4a" in lo: return ".m4a"
    return ".wav"

def _normalize_stt_lang(v: str) -> str:
    aliases = {"en":"english","en-us":"english","en-gb":"english","en-in":"english",
               "zh":"chinese","zh-cn":"chinese","de":"german","es":"spanish",
               "ru":"russian","ko":"korean","fr":"french","ja":"japanese"}
    raw = str(v or "").strip().lower().replace("_","-")
    normalized = aliases.get(raw, raw)
    return normalized if normalized in SUPPORTED_STT_LANGUAGES else "english"

def _clean_stt(text: str) -> str:
    text = str(text or "").strip()
    if not text: return ""
    for pattern in [r"\[(music|noise|silence|applause|laughs?)\]",
                    r"\((music|noise|silence|applause|laughs?)\)",
                    r"https?://\S+|www\.\S+",
                    r"\b(transcribed by|subtitles by|caption(s)? by)\b"]:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    banned = ["thanks for watching","thank you for watching","subscribe","captions by","see you in the next video"]
    if any(b in text.lower() for b in banned): return ""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens: return ""
    compact: List[str] = []; prev = ""; streak = 0
    for t in tokens:
        streak = streak+1 if t == prev else 1; prev = t
        if streak <= 2: compact.append(t)
    if not compact: return ""
    if len(compact) >= 6:
        trig: Dict[str,int] = {}
        for i in range(len(compact)-2):
            g = f"{compact[i]} {compact[i+1]} {compact[i+2]}"
            trig[g] = trig.get(g,0)+1
        if trig and max(trig.values()) >= 3: return ""
    if len(compact) >= 8 and len(set(compact))/len(compact) < 0.28: return ""
    return " ".join(compact).strip() if len(compact) > 2 else ""

def _run_piper(cmd: List[str], text: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{LOCAL_PY_DEPS}{os.pathsep}{os.environ.get('PYTHONPATH','')}".strip(os.pathsep)}
    return subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, env=env)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FastAPI routes
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_event() -> None:
    _bootstrap_models()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok", "model": MODEL_NAME, "ready": True,
        "ragEnabled": True, "onlineRagEnabled": RAG_FETCH_ONLINE,
        "ragRoles": sorted(RAG_SOURCE_MAP.keys()),
        "engines": {"question": "QuestionEngine", "counter": "CounterEngine"},
        "counterEnabled": COUNTER_Q_ENABLED, "counterPerAnswer": COUNTER_Q_PER_ANSWER,
        "sttEnabled": ENABLE_STT, "ttsEnabled": ENABLE_TTS, "sttModel": STT_MODEL_ID,
        **{k: WORKER_STARTUP_STATUS[k] for k in WORKER_STARTUP_STATUS},
        "targetWordsPerAnswer": TARGET_WORDS_PER_ANSWER,
        "minimumWordsPerMinute": MIN_WORDS_PER_MINUTE,
    }


@app.get("/rag/sources")
def rag_sources() -> Dict[str, Any]:
    return {"sources": RAG_SOURCE_MAP}


@app.post("/rag/sources")
def update_rag_sources(payload: Dict[str, Any]) -> Dict[str, Any]:
    global RAG_SOURCE_MAP, URL_CONTENT_CACHE
    if not isinstance(payload, dict): return {"status": "ignored", "reason": "payload must be object"}
    current = _load_source_map()
    for key, val in payload.items():
        if not isinstance(val, dict): continue
        current[str(key).strip().lower()] = {
            "sources":  list(dict.fromkeys(s.strip() for s in val.get("sources",[]) if str(s).strip())),
            "keywords": list(dict.fromkeys(w.strip().lower() for w in val.get("keywords",[]) if str(w).strip())),
        }
    RAG_SOURCES_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    RAG_SOURCE_MAP = current; URL_CONTENT_CACHE = {}
    return {"status": "updated", "roles": sorted(RAG_SOURCE_MAP.keys())}


# ── Engine 1: question generation ─────────────────────────────────────────────
@app.post("/interview/questions")
def interview_questions(payload: InterviewQuestionsRequest) -> Dict[str, Any]:
    return _question_engine.build(payload)


# ── Engine 2: counter / follow-up questions ───────────────────────────────────
@app.post("/interview/counter-questions")
def interview_counter_questions(payload: CounterQuestionRequest) -> Dict[str, Any]:
    return _counter_engine.generate(payload)


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.post("/interview/evaluate")
def interview_evaluate(payload: InterviewRequest) -> Dict[str, Any]:
    return _evaluate(payload)


# ── Combined: evaluate + auto-generate counter questions ──────────────────────
@app.post("/interview/evaluate-with-followup")
def interview_evaluate_with_followup(payload: InterviewRequest) -> Dict[str, Any]:
    """Evaluate all answers AND generate counter-questions for each answer in one call."""
    result = _evaluate(payload)
    if not COUNTER_Q_ENABLED:
        result["counterQuestions"] = []
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
    return result


# ── STT ───────────────────────────────────────────────────────────────────────
@app.post("/speech/transcribe")
def speech_transcribe(payload: SpeechTranscribeRequest) -> Dict[str, Any]:
    if not ENABLE_STT: return {"ok": False, "reason": "STT disabled"}
    audio_path: Optional[Path] = None; remove_after = False
    try:
        requested = Path(payload.audioPath.strip()) if payload.audioPath.strip() else None
        if requested and requested.exists():
            audio_path = requested
        elif payload.audioBase64.strip():
            raw = base64.b64decode(payload.audioBase64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_from_ct(payload.contentType))
            tmp.write(raw); tmp.flush(); tmp.close()
            audio_path = Path(tmp.name); remove_after = True
        else:
            return {"ok": False, "reason": "Provide audioPath or audioBase64"}
        transcriber = _get_stt_pipeline()
        lang = _normalize_stt_lang(_first_non_blank(payload.language, STT_LANGUAGE))
        result = transcriber(str(audio_path), generate_kwargs={
            "language": lang, "task": "transcribe", "temperature": 0.0, "condition_on_prev_tokens": False})
        raw_text = (result.get("text","") if isinstance(result,dict) else str(result)).strip()
        text = _clean_stt(raw_text)
        return {"ok": True, "text": text, "language": lang, "model": STT_MODEL_ID,
                "hallucinationFiltered": bool(raw_text and not text)}
    except Exception as ex:
        return {"ok": False, "reason": str(ex), "model": STT_MODEL_ID}
    finally:
        if remove_after and audio_path and audio_path.exists():
            try: audio_path.unlink()
            except Exception: pass


@app.websocket("/speech/ws/stt")
async def speech_ws_stt(websocket: WebSocket):
    await websocket.accept()
    if not ENABLE_STT:
        await websocket.send_json({"ok": False, "reason": "STT disabled"}); await websocket.close(); return
    language = _normalize_stt_lang(STT_LANGUAGE); content_type = "audio/webm"; generation = 0
    try:
        transcriber = _get_stt_pipeline()
    except Exception as ex:
        await websocket.send_json({"ok": False, "reason": f"STT init: {ex}"}); await websocket.close(); return
    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"]:
                try:
                    p = json.loads(msg["text"].strip())
                    mt = str(p.get("type","")).strip().lower()
                    if mt in {"config","meta"}:
                        language = _normalize_stt_lang(_first_non_blank(str(p.get("language","")), language))
                        content_type = _first_non_blank(str(p.get("contentType","")), content_type)
                        try: generation = int(p.get("generation", generation))
                        except Exception: pass
                        await websocket.send_json({"ok": True, "type": "ack", "generation": generation})
                    elif mt == "ping":
                        await websocket.send_json({"ok": True, "type": "pong"})
                except Exception: pass
                continue
            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_from_ct(content_type))
                tp = Path(tmp.name)
                try:
                    tmp.write(chunk); tmp.flush(); tmp.close()
                    res = transcriber(str(tp), generate_kwargs={
                        "language": language, "task": "transcribe",
                        "temperature": 0.0, "condition_on_prev_tokens": False})
                    raw = (res.get("text","") if isinstance(res,dict) else str(res)).strip()
                    cleaned = _clean_stt(raw)
                    await websocket.send_json({"ok": True, "text": cleaned, "language": language,
                                               "generation": generation, "hallucinationFiltered": bool(raw and not cleaned)})
                except Exception as ex:
                    await websocket.send_json({"ok": False, "reason": str(ex), "generation": generation})
                finally:
                    try:
                        if tp.exists(): tp.unlink()
                    except Exception: pass
    except WebSocketDisconnect: return
    except Exception:
        try: await websocket.close()
        except Exception: pass


# ── TTS ───────────────────────────────────────────────────────────────────────
@app.post("/speech/synthesize")
def speech_synthesize(payload: SpeechSynthesizeRequest) -> Dict[str, Any]:
    if not ENABLE_TTS: return {"ok": False, "reason": "TTS disabled"}
    text = str(payload.text or "").strip()
    if not text: return {"ok": False, "reason": "text is required"}
    voice = Path(PIPER_VOICE_MODEL)
    if not voice.exists(): return {"ok": False, "reason": f"Voice model missing: {voice}"}
    out = Path(payload.outputPath.strip()) if payload.outputPath.strip() else Path(tempfile.mktemp(suffix=".wav"))
    out.parent.mkdir(parents=True, exist_ok=True)
    speaker = _first_non_blank(payload.speaker, PIPER_SPEAKER)
    spk_args = ["--speaker", speaker] if speaker else []

    candidates: List[List[str]] = []
    if PIPER_BIN and PIPER_BIN != "__python_module__:piper":
        candidates.append([PIPER_BIN, "--model", str(voice), "--output_file", str(out)] + spk_args)
    candidates.append([sys.executable, "-m", "piper", "--model", str(voice), "--output_file", str(out)] + spk_args)

    completed = None; last_cmd = candidates[0]
    for cmd in candidates:
        last_cmd = cmd; completed = _run_piper(cmd, text)
        if completed.returncode == 0: break

    if completed is None: return {"ok": False, "reason": "No Piper candidate", "command": last_cmd}
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        if any(k in stderr for k in ("Traceback","ModuleNotFoundError","ImportError")):
            if AUTO_INSTALL_PIPER: _pip_install("piper-tts")
            if AUTO_INSTALL_PIPER_DEPS: _pip_install("onnxruntime","piper-phonemize","pathvalidate")
            for cmd in candidates:
                last_cmd = cmd; completed = _run_piper(cmd, text)
                if completed.returncode == 0: break
            if completed.returncode != 0:
                return {"ok": False, "reason": completed.stderr.decode("utf-8",errors="ignore")[:3000], "command": last_cmd}
        else:
            return {"ok": False, "reason": stderr[:3000], "command": last_cmd}

    resp: Dict[str, Any] = {"ok": True, "outputPath": str(out), "voiceModel": str(voice)}
    if payload.returnBase64 and out.exists():
        resp["audioBase64"] = base64.b64encode(out.read_bytes()).decode("utf-8")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)