import json
import logging
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("simple-interview-worker")


WORKER_HOST = os.getenv("WORKER_HOST", "0.0.0.0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "9000"))

DEEPSEEK_MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

SESSION_TTL_SECONDS = int(os.getenv("INTERVIEW_SESSION_TTL_SECONDS", "28800"))
DEFAULT_TOTAL_QUESTIONS = int(os.getenv("DEFAULT_TOTAL_QUESTIONS", "15"))
MAX_TOTAL_QUESTIONS = int(os.getenv("MAX_TOTAL_QUESTIONS", "15"))
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "80"))
MIN_NEW_TOKENS = int(os.getenv("LLM_MIN_NEW_TOKENS", "24"))
GEN_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
GEN_TOP_P = float(os.getenv("LLM_TOP_P", "0.8"))
LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", "true").strip().lower() in {"1", "true", "yes", "on"}
LLM_4BIT_COMPUTE_DTYPE = os.getenv("LLM_4BIT_COMPUTE_DTYPE", "float16").strip().lower()
ASKQUESTION_WAIT_MS = int(os.getenv("ASKQUESTION_WAIT_MS", "1500"))
PREFETCH_ON_STARTINTERVIEW = os.getenv("PREFETCH_ON_STARTINTERVIEW", "true").strip().lower() in {"1", "true", "yes", "on"}
QUESTION_REPEAT_RATE = float(os.getenv("QUESTION_REPEAT_RATE", "0.10"))
ROLE_QUESTION_HISTORY_LIMIT = int(os.getenv("ROLE_QUESTION_HISTORY_LIMIT", "1200"))
QUESTION_WORD_LIMIT = int(os.getenv("QUESTION_WORD_LIMIT", "50"))
OUT_OF_CONTEXT_THRESHOLD = float(os.getenv("OUT_OF_CONTEXT_THRESHOLD", "0.60"))
ANSWER_JUDGE_MODE = os.getenv("ANSWER_JUDGE_MODE", "hybrid").strip().lower()
LLM_JUDGE_MAX_NEW_TOKENS = int(os.getenv("LLM_JUDGE_MAX_NEW_TOKENS", "80"))

# 60-core-concept frameworks by interview track.
SHARED_DEV_40 = [
    "problem solving approach",
    "time and space complexity trade-offs",
    "data structures selection",
    "clean code principles",
    "debugging strategy",
    "error handling and resilience",
    "API design fundamentals",
    "REST semantics and status codes",
    "authentication vs authorization",
    "database indexing basics",
    "SQL query optimization",
    "transaction isolation",
    "caching patterns",
    "asynchronous processing basics",
    "concurrency and race conditions",
    "thread safety",
    "unit testing strategy",
    "integration testing strategy",
    "CI/CD workflow",
    "code review best practices",
    "observability and logging",
    "metrics and alerting",
    "security basics in web apps",
    "input validation and sanitization",
    "versioning and backward compatibility",
    "feature flag rollout",
    "performance profiling",
    "memory optimization",
    "system design fundamentals",
    "scalability bottleneck analysis",
    "microservices vs monolith trade-off",
    "message queues and event-driven basics",
    "rate limiting and throttling",
    "fault tolerance patterns",
    "deployment rollback strategy",
    "incident response mindset",
    "requirement clarification to technical tasks",
    "estimation for engineering tasks",
    "API contract validation",
    "schema evolution strategy",
]

JAVA_EXTRA_20 = [
    "JVM memory model",
    "garbage collection tuning basics",
    "Java collections internals",
    "equals and hashCode contract",
    "immutability in Java",
    "Java streams best practices",
    "optional usage patterns",
    "exception hierarchy design",
    "synchronization primitives",
    "CompletableFuture patterns",
    "Spring dependency injection",
    "Spring Boot auto-configuration",
    "Spring transactional boundaries",
    "Spring Security basics",
    "JPA entity mapping pitfalls",
    "N+1 query prevention",
    "connection pool tuning",
    "Maven or Gradle build hygiene",
    "Java profiling tools usage",
    "JDK upgrade and compatibility",
]

PYTHON_EXTRA_20 = [
    "Python data model basics",
    "list vs tuple vs set choices",
    "dictionary performance characteristics",
    "comprehension readability and cost",
    "iterators and generators",
    "context managers",
    "exception handling patterns",
    "type hints and static analysis",
    "packaging and virtual environments",
    "pip dependency pinning",
    "asyncio fundamentals",
    "multi-threading vs multiprocessing",
    "GIL implications",
    "FastAPI request lifecycle",
    "Pydantic validation design",
    "ORM usage patterns in Python",
    "pytest strategy",
    "Python logging and tracing",
    "memory leaks in long-running services",
    "Python performance profiling tools",
]

JS_EXTRA_20 = [
    "JavaScript event loop",
    "call stack and microtasks",
    "closures and lexical scope",
    "prototype and inheritance model",
    "async await error handling",
    "promise chaining pitfalls",
    "module system and bundling",
    "state management patterns",
    "React rendering lifecycle",
    "React hooks correctness",
    "frontend performance optimization",
    "web security basics",
    "XSS and CSRF defenses",
    "Node.js stream processing",
    "Node.js clustering strategy",
    "Express middleware design",
    "API contract typing",
    "browser caching strategy",
    "web accessibility fundamentals",
    "frontend testing pyramid",
]

REACT_EXTRA_20 = [
    "react component architecture",
    "props and state design",
    "react rendering lifecycle",
    "react hooks correctness",
    "useEffect dependency pitfalls",
    "state management with context and redux",
    "memoization with useMemo and useCallback",
    "performance optimization in large lists",
    "error boundaries in react apps",
    "form handling and validation patterns",
    "routing with react router",
    "api integration and loading states",
    "optimistic UI updates",
    "frontend caching strategy",
    "accessibility in react components",
    "component testing with react testing library",
    "code splitting and lazy loading",
    "bundle optimization basics",
    "security basics in frontend",
    "production debugging and monitoring",
]

NODE_EXTRA_20 = [
    "node event loop internals",
    "async await and promise error handling",
    "express middleware architecture",
    "request validation and sanitization",
    "authentication and authorization in node apis",
    "database query optimization in backend services",
    "rate limiting and abuse protection",
    "caching in node services",
    "queue-based background jobs",
    "idempotency in api design",
    "logging and observability in node",
    "graceful shutdown and process management",
    "worker threads and clustering trade-offs",
    "stream processing patterns",
    "websocket scaling considerations",
    "microservice communication patterns",
    "testing strategy for node services",
    "deployment and rollback strategy",
    "security vulnerabilities in node ecosystem",
    "performance profiling and memory leak debugging",
]

FLUTTER_EXTRA_20 = [
    "flutter widget lifecycle",
    "build method optimization",
    "state management strategy in flutter",
    "provider vs riverpod vs bloc trade-offs",
    "flutter navigation architecture",
    "go_router or navigator 2 usage",
    "async operations in dart",
    "isolates usage for heavy work",
    "rendering performance and jank reduction",
    "devtools profiling workflow",
    "layout constraints and responsive ui",
    "custom painter and animation basics",
    "platform channels integration",
    "offline-first data sync in flutter",
    "local storage choices in flutter apps",
    "network layer design in flutter",
    "error handling and retry in mobile flows",
    "flutter testing strategy",
    "release build optimization",
    "play store or app store deployment readiness",
]

TESTING_60 = [
    "test strategy and scope",
    "risk-based test planning",
    "test case design techniques",
    "boundary value analysis",
    "equivalence partitioning",
    "decision table testing",
    "state transition testing",
    "exploratory testing approach",
    "regression suite management",
    "smoke vs sanity tests",
    "functional vs non-functional testing",
    "performance test planning",
    "load vs stress vs soak testing",
    "API testing strategy",
    "contract testing basics",
    "UI automation reliability",
    "flaky test root causes",
    "test data management",
    "environment management",
    "defect lifecycle handling",
    "severity vs priority",
    "bug report quality",
    "root cause analysis for defects",
    "shift-left testing",
    "shift-right testing",
    "CI test pipeline optimization",
    "parallel test execution",
    "test coverage metrics",
    "code coverage limits",
    "mutation testing basics",
    "mocking and stubbing decisions",
    "service virtualization",
    "security testing basics",
    "OWASP testing essentials",
    "accessibility testing",
    "cross-browser testing",
    "mobile testing strategy",
    "usability validation",
    "database testing basics",
    "data integrity validation",
    "ETL test scenarios",
    "integration test isolation",
    "observability for QA",
    "log-driven debugging",
    "release quality gates",
    "test estimation and planning",
    "stakeholder communication in QA",
    "quality KPIs",
    "triage meeting effectiveness",
    "automation ROI analysis",
    "SDET coding practices",
    "maintainable test framework design",
    "BDD vs TDD in practice",
    "test documentation quality",
    "production validation checks",
    "incident reproduction strategy",
    "post-release validation",
    "customer-reported bug handling",
    "continuous improvement in QA",
    "mentoring junior testers",
]

DESKTOP_SUPPORT_60 = [
    "ticket triage prioritization",
    "SLA and escalation handling",
    "hardware diagnostics basics",
    "OS troubleshooting workflow",
    "Windows user profile issues",
    "printer troubleshooting",
    "network connectivity diagnostics",
    "DNS and DHCP basics",
    "VPN troubleshooting",
    "email client issue resolution",
    "MFA and login support",
    "password reset security checks",
    "active directory basics",
    "group policy troubleshooting",
    "software installation policies",
    "endpoint security tooling",
    "antivirus incident handling",
    "malware containment basics",
    "remote support best practices",
    "service desk communication",
    "customer empathy in support",
    "root cause capture in tickets",
    "knowledge base documentation",
    "asset inventory tracking",
    "device provisioning workflow",
    "onboarding IT checklist",
    "offboarding access revocation",
    "backup and restore basics",
    "disk space management",
    "system performance troubleshooting",
    "startup issue diagnosis",
    "blue screen response basics",
    "driver conflict resolution",
    "browser and cache issues",
    "certificate trust problems",
    "proxy and firewall checks",
    "collaboration tools support",
    "audio and video call issues",
    "USB and peripheral issues",
    "mobile device support basics",
    "patch management process",
    "change management in IT support",
    "incident vs service request",
    "problem management basics",
    "major incident coordination",
    "vendor coordination process",
    "compliance awareness in support",
    "data privacy in support operations",
    "phishing awareness response",
    "reporting and dashboard metrics",
    "queue management techniques",
    "time management under ticket load",
    "handoff quality across shifts",
    "documentation standardization",
    "self-service enablement",
    "automation opportunities in support",
    "support quality audit",
    "post-incident review actions",
    "user training and adoption support",
    "continuous improvement in service desk",
]

SALES_60 = [
    "lead qualification methodology",
    "ICP and buyer persona clarity",
    "prospecting strategy",
    "cold outreach messaging",
    "discovery call structure",
    "active listening in sales",
    "pain point identification",
    "business impact quantification",
    "value proposition articulation",
    "competitive positioning",
    "objection handling framework",
    "pricing negotiation fundamentals",
    "closing techniques",
    "pipeline management hygiene",
    "forecast accuracy practices",
    "CRM discipline and notes quality",
    "stakeholder mapping",
    "multi-threading accounts",
    "account planning",
    "territory planning",
    "sales cadence optimization",
    "follow-up strategy",
    "proposal quality",
    "demo storytelling",
    "solution selling basics",
    "consultative selling",
    "SPIN selling basics",
    "MEDDICC basics",
    "BANT usage in context",
    "handling no-decision outcomes",
    "renewal and upsell motion",
    "cross-sell qualification",
    "customer success collaboration",
    "pre-sales and sales alignment",
    "marketing and sales handoff",
    "deal risk management",
    "procurement process navigation",
    "legal and compliance checkpoints",
    "RFP response strategy",
    "time management in sales cycles",
    "quota planning",
    "win-loss analysis",
    "data-driven sales coaching",
    "call review and improvement",
    "email and call metrics interpretation",
    "personal branding for sales",
    "relationship building",
    "enterprise sales cycle understanding",
    "SMB vs enterprise selling differences",
    "channel and partner sales basics",
    "regional market adaptation",
    "selling in India market context",
    "handling price-sensitive buyers",
    "trust building with first-time buyers",
    "ethical selling practices",
    "customer retention mindset",
    "post-sale handover quality",
    "sales presentation clarity",
    "resilience under rejection",
    "continuous learning in sales",
]

GENERAL_PRO_60 = [
    "communication clarity",
    "structured problem solving",
    "stakeholder management",
    "prioritization under pressure",
    "time management",
    "conflict resolution",
    "decision-making framework",
    "ownership mindset",
    "team collaboration",
    "cross-functional alignment",
    "goal setting and tracking",
    "execution discipline",
    "risk identification",
    "escalation judgment",
    "process improvement",
    "customer-centric thinking",
    "quality mindset",
    "adaptability and learning",
    "feedback handling",
    "coaching and mentoring",
    "leadership without authority",
    "data-informed decisions",
    "documentation discipline",
    "meeting effectiveness",
    "presentation skills",
    "negotiation basics",
    "resource management",
    "planning and estimation",
    "root cause analysis",
    "continuous improvement loop",
    "ethical judgment",
    "compliance awareness",
    "accountability culture",
    "change management",
    "project lifecycle understanding",
    "KPI selection",
    "outcome orientation",
    "customer issue handling",
    "vendor management basics",
    "cost consciousness",
    "quality vs speed trade-off",
    "problem decomposition",
    "critical thinking",
    "working with ambiguity",
    "career motivation clarity",
    "role-specific domain learning",
    "interview storytelling",
    "STAR method communication",
    "failure and recovery examples",
    "achievement articulation",
    "initiative and innovation",
    "collaboration in distributed teams",
    "tooling and process adoption",
    "professional integrity",
    "long-term ownership",
    "resilience in setbacks",
    "focus and deep work",
    "handover and transition quality",
    "knowledge sharing",
    "self-review and growth plan",
]

MANAGEMENT_60 = [
    "team goal setting",
    "performance management",
    "delegation strategy",
    "prioritization across teams",
    "resource planning",
    "hiring and interviewing",
    "onboarding effectiveness",
    "coaching and mentoring",
    "feedback framework",
    "conflict resolution",
    "cross-functional alignment",
    "stakeholder communication",
    "meeting cadence design",
    "roadmap planning",
    "execution tracking",
    "risk management",
    "escalation handling",
    "incident leadership",
    "delivery predictability",
    "quality culture",
    "process improvement",
    "team morale management",
    "retention strategy",
    "career growth plans",
    "succession planning",
    "budget awareness",
    "vendor coordination",
    "compliance governance",
    "change management",
    "organizational influence",
    "decision-making transparency",
    "data-driven management",
    "KPI selection",
    "OKR planning",
    "operational review rhythm",
    "accountability systems",
    "project portfolio balancing",
    "scope control",
    "trade-off communication",
    "customer escalation handling",
    "service-level management",
    "policy implementation",
    "management under ambiguity",
    "remote team leadership",
    "knowledge sharing culture",
    "documentation discipline",
    "dependency management",
    "recruitment funnel quality",
    "managerial ethics",
    "fairness and bias mitigation",
    "team restructuring planning",
    "high performer growth",
    "low performer recovery",
    "manager stakeholder mapping",
    "business context understanding",
    "problem-solving leadership",
    "continuous improvement loops",
    "post-mortem effectiveness",
    "strategic communication",
    "leadership maturity",
]

LEADERSHIP_EXEC_60 = [
    "vision and strategy articulation",
    "business model clarity",
    "market positioning",
    "long-term roadmap",
    "P&L understanding",
    "unit economics awareness",
    "capital allocation decisions",
    "growth strategy",
    "go-to-market alignment",
    "organizational design",
    "executive hiring",
    "leadership team building",
    "board communication",
    "investor communication",
    "risk governance",
    "compliance and policy leadership",
    "cybersecurity governance",
    "data privacy governance",
    "enterprise architecture decisions",
    "technology investment priorities",
    "innovation strategy",
    "portfolio prioritization",
    "M&A integration thinking",
    "partnership strategy",
    "customer trust strategy",
    "brand stewardship",
    "operational excellence model",
    "scaling culture",
    "performance culture",
    "decision speed vs quality trade-off",
    "cross-functional leadership",
    "global market adaptation",
    "India market strategy awareness",
    "competitive intelligence use",
    "crisis leadership",
    "major incident response",
    "restructuring strategy",
    "cost optimization leadership",
    "quality and reliability governance",
    "talent retention at scale",
    "succession at executive level",
    "ethics and integrity leadership",
    "stakeholder expectation management",
    "enterprise KPI framework",
    "data-driven executive reviews",
    "change narrative building",
    "enterprise communication clarity",
    "execution governance",
    "portfolio risk balancing",
    "legal and regulatory coordination",
    "public communication readiness",
    "operating cadence design",
    "leadership accountability",
    "strategic scenario planning",
    "resilience planning",
    "culture transformation",
    "digital transformation execution",
    "customer-centric strategy",
    "long-term value creation",
    "founder or executive mindset",
]


app = FastAPI(
    title="Simple Interview Worker",
    version="3.0.0",
    description="Minimal interview worker: startinterveiw, askquestion, matchanswer",
)

cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
cors_origins = [x.strip() for x in cors_origins_raw.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in cors_origins else cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartInterveiwRequest(BaseModel):
    candidateId: Optional[str] = ""
    candidateName: Optional[str] = ""
    role: Optional[str] = ""
    jobTitle: Optional[str] = ""
    difficulty: Optional[str] = "medium"
    totalQuestions: Optional[int] = Field(default=None, ge=1, le=MAX_TOTAL_QUESTIONS)


class AskQuestionRequest(BaseModel):
    sessionId: str = Field(..., min_length=4)


class MatchAnswerRequest(BaseModel):
    sessionId: str = Field(..., min_length=4)
    questionId: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


@dataclass
class InterviewTurn:
    question_id: str
    question: str
    ideal_answer: str = ""
    concept: str = ""
    asked: bool = False
    answered: bool = False


@dataclass
class InterviewSession:
    session_id: str
    created_at: float
    updated_at: float
    role: str
    difficulty: str
    total_questions: int
    turns: List[InterviewTurn] = field(default_factory=list)
    asked_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


SESSIONS: Dict[str, InterviewSession] = {}
SESSIONS_LOCK = threading.Lock()
GENERATION_FUTURES: Dict[str, Future] = {}
ROLE_QUESTION_HISTORY: Dict[str, List[str]] = {}
ROLE_QUESTION_HISTORY_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("GENERATION_WORKERS", "4")))


_QA_MODEL_LOCK = threading.Lock()
_QA_TOKENIZER = None
_QA_MODEL = None
_QA_MODEL_ERROR: Optional[str] = None

_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER = None
_EMBEDDER_ERROR: Optional[str] = None


def _resolve_4bit_compute_dtype() -> torch.dtype:
    # Keep dtype configurable while mapping only safe torch dtypes.
    # Unsupported values gracefully fall back to float16.
    if LLM_4BIT_COMPUTE_DTYPE in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if LLM_4BIT_COMPUTE_DTYPE in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


def _now() -> float:
    return time.time()


def _normalize_difficulty(value: str) -> str:
    raw = str(value or "medium").strip().lower()
    if raw in {"easy", "medium", "hard"}:
        return raw
    return "medium"


def _resolve_role(payload: StartInterveiwRequest) -> str:
    return (
        str(payload.role or "").strip()
        or str(payload.jobTitle or "").strip()
        or "General"
    )


def _cleanup_expired_sessions() -> None:
    cutoff = _now() - SESSION_TTL_SECONDS
    stale_ids: List[str] = []
    with SESSIONS_LOCK:
        for sid, session in SESSIONS.items():
            if session.updated_at < cutoff:
                stale_ids.append(sid)
        for sid in stale_ids:
            SESSIONS.pop(sid, None)
            fut = GENERATION_FUTURES.pop(sid, None)
            if fut and not fut.done():
                fut.cancel()
    if stale_ids:
        logger.info("Removed %d expired interview sessions", len(stale_ids))


def _get_session_or_404(session_id: str) -> InterviewSession:
    _cleanup_expired_sessions()
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        session.updated_at = _now()
        return session


def _load_qa_model() -> Optional[Any]:
    global _QA_MODEL, _QA_TOKENIZER, _QA_MODEL_ERROR
    if _QA_MODEL is not None and _QA_TOKENIZER is not None:
        return _QA_MODEL
    if _QA_MODEL_ERROR:
        return None

    with _QA_MODEL_LOCK:
        if _QA_MODEL is not None and _QA_TOKENIZER is not None:
            return _QA_MODEL
        if _QA_MODEL_ERROR:
            return None
        try:
            logger.info("Loading question model: %s", DEEPSEEK_MODEL_NAME)
            _QA_TOKENIZER = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_NAME, trust_remote_code=True)
            # First try 4-bit quantized load for lower memory and better throughput.
            # If quantization runtime support is missing, we log and fallback to normal load.
            if LLM_LOAD_IN_4BIT:
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=_resolve_4bit_compute_dtype(),
                    )
                    _QA_MODEL = AutoModelForCausalLM.from_pretrained(
                        DEEPSEEK_MODEL_NAME,
                        trust_remote_code=True,
                        quantization_config=bnb_cfg,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    logger.info("Question model loaded with 4-bit quantization")
                except Exception as quant_exc:
                    logger.warning(
                        "4-bit load failed (%s). Falling back to non-quantized load.",
                        quant_exc,
                    )
                    _QA_MODEL = AutoModelForCausalLM.from_pretrained(
                        DEEPSEEK_MODEL_NAME,
                        trust_remote_code=True,
                    )
            else:
                _QA_MODEL = AutoModelForCausalLM.from_pretrained(
                    DEEPSEEK_MODEL_NAME,
                    trust_remote_code=True,
                )
            _QA_MODEL.eval()
            logger.info("Question model loaded")
            return _QA_MODEL
        except Exception as exc:
            _QA_MODEL_ERROR = str(exc)
            logger.exception("Failed to load question model, fallback mode active")
            return None


def _load_embedder() -> Optional[SentenceTransformer]:
    global _EMBEDDER, _EMBEDDER_ERROR
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_ERROR:
        return None

    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _EMBEDDER_ERROR:
            return None
        try:
            logger.info("Loading semantic model: %s", SEMANTIC_MODEL_NAME)
            _EMBEDDER = SentenceTransformer(SEMANTIC_MODEL_NAME)
            logger.info("Semantic model loaded")
            return _EMBEDDER
        except Exception as exc:
            _EMBEDDER_ERROR = str(exc)
            logger.exception("Failed to load semantic model, lexical fallback active")
            return None


def _extract_json_obj(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    raw = str(text).strip()

    # 1) Fast path: payload is already a JSON object.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Robust path: scan balanced JSON object candidates and parse each.
    candidates: List[Dict[str, Any]] = []
    for start in [i for i, ch in enumerate(raw) if ch == "{"]:
        depth = 0
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = raw[start : idx + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except Exception:
                        pass
                    break

    if not candidates:
        return {}

    # Prefer objects that contain expected generation fields and real content.
    for obj in candidates:
        if "question" in obj:
            q = str(obj.get("question", "")).strip()
            if q in {"...", ".", ""}:
                continue
            if len(q) < 8:
                continue
            return obj
    return candidates[0]


def _sanitize_generation_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Handle common wrappers like markdown JSON fences.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def _extract_labeled_fields(text: str) -> Dict[str, str]:
    raw = str(text or "")
    # Fallback parser for models that fail strict JSON but produce labeled text.
    q_match = re.search(r"(?im)^\s*question\s*[:\-]\s*(.+)$", raw)
    c_match = re.search(r"(?im)^\s*concept\s*[:\-]\s*(.+)$", raw)
    return {
        "question": q_match.group(1).strip() if q_match else "",
        "concept": c_match.group(1).strip() if c_match else "",
    }


def _is_placeholder_text(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in {"...", ".", "-", "_", "n/a", "na", "none"}


def _resolve_interview_track(role: str) -> str:
    raw = str(role or "").strip().lower()
    if any(k in raw for k in ["java", "spring", "jvm"]):
        return "java_dev"
    if any(k in raw for k in ["flutter", "dart", "mobile developer", "android developer", "ios developer", "react native"]):
        return "flutter_dev"
    if any(k in raw for k in ["react", "frontend", "front-end", "ui developer"]):
        return "react_dev"
    if any(k in raw for k in ["node", "nodejs", "node.js", "express", "backend developer", "back-end"]):
        return "node_dev"
    if any(k in raw for k in ["sales", "business development", "account executive", "inside sales", "bde"]):
        return "sales"
    if any(k in raw for k in ["business executive", "business operations", "operations executive", "business analyst", "business manager", "executive"]):
        return "business_exec"
    if any(k in raw for k in ["developer", "engineer", "software"]):
        return "tech_general"
    return "business_exec"


def _concept_pool_for_track(track: str) -> List[str]:
    if track == "java_dev":
        return [*SHARED_DEV_40, *JAVA_EXTRA_20]
    if track == "flutter_dev":
        return [*SHARED_DEV_40, *FLUTTER_EXTRA_20]
    if track == "react_dev":
        return [*SHARED_DEV_40, *REACT_EXTRA_20]
    if track == "node_dev":
        return [*SHARED_DEV_40, *NODE_EXTRA_20]
    if track == "tech_general":
        return [*SHARED_DEV_40, *JAVA_EXTRA_20]
    if track == "sales":
        return SALES_60
    if track == "business_exec":
        return GENERAL_PRO_60
    return [*SHARED_DEV_40, *JAVA_EXTRA_20]


def _difficulty_guidance(difficulty: str) -> str:
    if difficulty == "easy":
        return "Ask fundamentals and practical basics; avoid deep edge-case traps."
    if difficulty == "hard":
        return "Ask architecture or high-pressure scenario questions with trade-offs and failure handling."
    return "Ask practical mid-level scenario questions with implementation detail and decision trade-offs."


def _track_interview_style(track: str) -> str:
    if track.endswith("_dev") or track == "tech_general":
        return (
            "Developer interview style: concept-first, real scenario, constraints, debugging angle, and trade-off follow-up."
        )
    if track == "sales":
        return "Sales interview style: discovery quality, value articulation, objection handling, and pipeline execution."
    return "Business executive interview style: ownership, execution, prioritization, and measurable outcomes."


def _role_bucket_key(role: str, track: str) -> str:
    role_key = re.sub(r"\s+", " ", str(role or "").strip().lower())
    return f"{track}|{role_key}"


def _remember_question_for_role(bucket: str, question: str) -> None:
    q = str(question or "").strip()
    if not q:
        return
    with ROLE_QUESTION_HISTORY_LOCK:
        history = ROLE_QUESTION_HISTORY.setdefault(bucket, [])
        history.append(q)
        if len(history) > ROLE_QUESTION_HISTORY_LIMIT:
            ROLE_QUESTION_HISTORY[bucket] = history[-ROLE_QUESTION_HISTORY_LIMIT:]


def _pick_repeat_question_for_role(bucket: str, session_existing_keys: set[str]) -> str:
    with ROLE_QUESTION_HISTORY_LOCK:
        history = list(ROLE_QUESTION_HISTORY.get(bucket, []))
    if not history:
        return ""
    candidates = [q for q in history if _question_key(q) not in session_existing_keys]
    if not candidates:
        return ""
    return random.choice(candidates)


def _apply_repeat_policy(
    role: str,
    track: str,
    turn_number: int,
    question: str,
    session_existing_keys: set[str],
) -> str:
    bucket = _role_bucket_key(role, track)
    allow_repeat = random.random() < max(0.0, min(1.0, QUESTION_REPEAT_RATE))

    selected = str(question or "").strip()
    if allow_repeat:
        repeated = _pick_repeat_question_for_role(bucket, session_existing_keys)
        if repeated:
            selected = repeated

    selected = _ensure_unique_question_text(selected, turn_number, session_existing_keys)
    _remember_question_for_role(bucket, selected)
    return selected


def _select_concept_for_turn(session: InterviewSession, turn_number: int, track: str) -> str:
    pool = _concept_pool_for_track(track)
    used = {_question_key(t.concept) for t in session.turns if str(t.concept or "").strip()}
    base_idx = max(0, turn_number - 2) % max(1, len(pool))
    for offset in range(len(pool)):
        candidate = pool[(base_idx + offset) % len(pool)]
        if _question_key(candidate) not in used:
            return candidate
    return pool[base_idx]


def _generate_turn_with_llm(session: InterviewSession, turn_number: int, previous_questions: Optional[List[str]] = None) -> Dict[str, str]:
    model = _load_qa_model()
    track = _resolve_interview_track(session.role)
    target_concept = _select_concept_for_turn(session, turn_number, track)
    if model is None or _QA_TOKENIZER is None:
        raise RuntimeError("Question model is not available")

    if previous_questions is None:
        previous_questions = [t.question for t in session.turns]
    previous_concepts = [t.concept for t in session.turns if str(t.concept or "").strip()]
    previous_questions = previous_questions[-12:]
    previous_concepts = previous_concepts[-20:]
    # Keep prompt compact to reduce token processing latency per askquestion.
    # Avoid embedding a literal JSON object example in prompt text, because
    # model echo can make parsers accidentally pick that schema object.
    prompt = (
        "Return ONLY a strict JSON object with keys: question, concept.\n"
        f"Role={session.role}; Track={track}; Difficulty={session.difficulty}; Turn={turn_number}/{session.total_questions}\n"
        f"RequiredConcept={target_concept}\n"
        f"QuestionWordLimit={QUESTION_WORD_LIMIT}\n"
        "Rules: one practical interview question; technical-core for tech roles; avoid repeats from prior context.\n"
        f"PriorQuestions={previous_questions[-6:]}\n"
        f"PriorConcepts={previous_concepts[-10:]}"
    )

    tokenizer = _QA_TOKENIZER
    effective_max_new_tokens = MAX_NEW_TOKENS
    question = ""
    concept = target_concept
    for attempt in range(3):
        # Retry prompts are progressively stricter to reduce chain-of-thought
        # spill and force JSON payload compliance.
        if attempt == 0:
            prompt_current = prompt
        elif attempt == 1:
            prompt_current = (
                f"Output JSON only. No explanations. "
                f'Keys: "question","concept". '
                f"Role={session.role}; Concept={target_concept}; "
                f"Question<= {QUESTION_WORD_LIMIT} words."
            )
        else:
            prompt_current = (
                f'No reasoning. No preface. Start with "{{" and end with "}}". '
                f'{{"question":"<max {QUESTION_WORD_LIMIT} words>","concept":"{target_concept}"}}'
            )

        # Prefer chat template when supported by model/tokenizer.
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": "You are an interview question generator. Reply with JSON only. Do not include reasoning.",
                },
                {"role": "user", "content": prompt_current},
            ]
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(inputs, torch.Tensor):
                    inputs = {"input_ids": inputs}
            except Exception:
                inputs = tokenizer(prompt_current, return_tensors="pt")
        else:
            inputs = tokenizer(prompt_current, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                min_new_tokens=max(1, MIN_NEW_TOKENS),
                max_new_tokens=effective_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Parse only generated continuation; full decode includes prompt and
        # often breaks JSON extraction for strict payload parsing.
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        decoded = _sanitize_generation_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
        parsed = _extract_json_obj(decoded)
        if not parsed:
            # Fallback parser for non-JSON but labeled model output.
            parsed = _extract_labeled_fields(decoded)
        # Enforce output caps for predictable short responses.
        question = _limit_words(str(parsed.get("question", "")).strip().strip('"').strip("'"), QUESTION_WORD_LIMIT)
        parsed_concept = str(parsed.get("concept", "")).strip()
        if parsed_concept:
            concept = parsed_concept
        # Reject placeholders and near-empty text to prevent "..." style output.
        q_words = len(question.split())
        if (
            question
            and not _is_placeholder_text(question)
            and q_words >= 5
        ):
            break
        logger.warning(
            "Invalid generation payload attempt=%s turn=%s role=%s raw=%s",
            attempt + 1,
            turn_number,
            session.role,
            decoded[:300],
        )
        question = ""
    if not question:
        raise RuntimeError("LLM did not return a valid question payload")

    return {
        "question": question,
        "concept": concept,
    }


def _ensure_generated_until(session: InterviewSession, target_count: int) -> None:
    while len(session.turns) < min(target_count, session.total_questions):
        turn_number = len(session.turns) + 1
        qa = _generate_turn_with_llm(session, turn_number)
        existing_questions = {_question_key(t.question) for t in session.turns}
        qa["question"] = _apply_repeat_policy(
            role=session.role,
            track=_resolve_interview_track(session.role),
            turn_number=turn_number,
            question=qa.get("question", ""),
            session_existing_keys=existing_questions,
        )
        turn = InterviewTurn(
            question_id=f"q{turn_number}",
            question=qa["question"],
            concept=str(qa.get("concept", "")),
        )
        session.turns.append(turn)


def _precompute_remaining(session_id: str) -> None:
    try:
        session = _get_session_or_404(session_id)
        while True:
            with session.lock:
                target_count = session.total_questions
                current_count = len(session.turns)
                if current_count >= target_count:
                    session.updated_at = _now()
                    break
                turn_number = current_count + 1
                previous_questions = [t.question for t in session.turns]

            qa = _generate_turn_with_llm(
                session=session,
                turn_number=turn_number,
                previous_questions=previous_questions,
            )

            with session.lock:
                if len(session.turns) < turn_number:
                    existing_questions = {_question_key(t.question) for t in session.turns}
                    qtext = _apply_repeat_policy(
                        role=session.role,
                        track=_resolve_interview_track(session.role),
                        turn_number=turn_number,
                        question=qa.get("question", ""),
                        session_existing_keys=existing_questions,
                    )
                    session.turns.append(
                        InterviewTurn(
                            question_id=f"q{turn_number}",
                            question=qtext,
                            concept=str(qa.get("concept", "")),
                        )
                    )
                session.updated_at = _now()
        logger.info("Precomputed all questions for session=%s", session_id)
    except Exception:
        logger.exception("Failed precompute for session=%s", session_id)


def _schedule_remaining_generation(session_id: str) -> None:
    # This scheduler ensures only one active pre-generation job per session.
    # It avoids duplicate heavy generation work when multiple requests hit quickly.
    with SESSIONS_LOCK:
        existing = GENERATION_FUTURES.get(session_id)
        if existing and not existing.done():
            return
        GENERATION_FUTURES[session_id] = EXECUTOR.submit(_precompute_remaining, session_id)


def _wait_for_precomputed_turn(session: InterviewSession, target_count: int, wait_ms: int) -> bool:
    # Wait briefly for background pre-generation so askquestion can return from cache.
    # Keeping this wait bounded prevents long tail latency in request handling.
    deadline = time.time() + (max(0, wait_ms) / 1000.0)
    while time.time() < deadline:
        with session.lock:
            if len(session.turns) >= target_count:
                return True
        time.sleep(0.05)
    with session.lock:
        return len(session.turns) >= target_count


def _question_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _limit_words(text: str, limit: int) -> str:
    # Enforces strict word budgets after JSON parsing.
    # This guarantees payload size caps even when model output is verbose.
    words = re.findall(r"\S+", str(text or "").strip())
    if limit <= 0:
        return ""
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).strip()


def _ensure_unique_question_text(question: str, turn_number: int, existing_keys: set[str]) -> str:
    q = str(question or "").strip()
    if not q:
        q = f"Question {turn_number}: Explain your approach for a practical {turn_number} scenario in this role."
    key = _question_key(q)
    if key not in existing_keys:
        return q

    # Guarantee uniqueness in-session even if model/fallback repeats.
    unique_q = f"{q} [Q{turn_number}]"
    if _question_key(unique_q) not in existing_keys:
        return unique_q
    return f"{q} [Q{turn_number}-{int(time.time() * 1000) % 100000}]"


def _bounded_similarity_from_cos(cos_value: float) -> float:
    return max(0.0, min(1.0, (float(cos_value) + 1.0) / 2.0))


def _sentence_relevance_score(sentence: str, question: str, embedder: Optional[SentenceTransformer]) -> float:
    if not sentence.strip():
        return 0.0
    if embedder is None:
        s_words = set(re.findall(r"\w+", sentence.lower()))
        q_words = set(re.findall(r"\w+", question.lower()))
        return len(s_words & q_words) / max(1, len(q_words))
    vectors = embedder.encode([sentence, question], convert_to_tensor=True)
    return _bounded_similarity_from_cos(float(util.cos_sim(vectors[0], vectors[1]).item()))


def _compact_answer_contexts(answer: str, question: str, concept: str) -> Dict[str, Any]:
    # Long answers are reduced to the most relevant sentence-level contexts so
    # the LLM judge receives compact, targeted evidence instead of junk text.
    answer = str(answer or "").strip()
    sentences = _split_sentences(answer)
    total_words = len(answer.split())
    if total_words <= 15 or len(sentences) <= 1:
        compact = _limit_words(answer, 30)
        return {"compactAnswer": compact, "selectedContexts": [compact] if compact else [], "truncated": False}

    embedder = _load_embedder()
    question_focus = f"{question} {concept}".strip()
    ranked = sorted(
        (
            {
                "sentence": s,
                "score": _sentence_relevance_score(s, question_focus, embedder),
            }
            for s in sentences
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected: List[str] = []
    word_budget = 36
    used_words = 0
    for item in ranked[:4]:
        sentence = str(item["sentence"]).strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())
        if selected and used_words + sentence_words > word_budget:
            continue
        selected.append(sentence)
        used_words += sentence_words
        if len(selected) >= 3:
            break
    compact = _limit_words(" ".join(selected) if selected else answer, word_budget)
    compact_sentences = [s for s in _split_sentences(compact) if s]
    return {
        "compactAnswer": compact,
        "selectedContexts": compact_sentences,
        "truncated": compact.strip() != answer.strip(),
    }


def _analyze_answer(answer: str, question: str, concept: str) -> Dict[str, Any]:
    embedder = _load_embedder()
    question_focus = f"{question} {concept}".strip()
    if embedder is None:
        # Fallback path when semantic model is unavailable. This is less accurate
        # than embedding-based semantic checks but keeps API functional.
        a_words = set(re.findall(r"\w+", answer.lower()))
        q_words = set(re.findall(r"\w+", question_focus.lower()))
        q_overlap = len(a_words & q_words)
        q_denom = max(1, len(q_words))
        relevance_similarity = q_overlap / q_denom
    else:
        vectors = embedder.encode([answer, question_focus], convert_to_tensor=True)
        answer_vec = vectors[0]
        question_vec = vectors[1]
        relevance_similarity = _bounded_similarity_from_cos(float(util.cos_sim(answer_vec, question_vec).item()))

    out_of_context = relevance_similarity < OUT_OF_CONTEXT_THRESHOLD
    if out_of_context:
        score = 0
    else:
        score = int(round(relevance_similarity * 100))
    return {
        "similarity": round(relevance_similarity, 4),
        "relevanceSimilarity": round(relevance_similarity, 4),
        "outOfContext": out_of_context,
        "score": max(0, min(100, score)),
    }


def _judge_answer_with_llm(question: str, concept: str, answer: str) -> Dict[str, Any]:
    # LLM judge provides context-aware right/wrong classification, independent
    # of word overlap. This directly addresses out-of-context false positives.
    model = _load_qa_model()
    tokenizer = _QA_TOKENIZER
    if model is None or tokenizer is None:
        return {"available": False, "verdict": "unknown", "score": 0, "reason": "LLM judge unavailable"}

    prompt = (
        "You are an interview answer evaluator.\n"
        "Return ONLY strict JSON with keys: verdict, score, reason.\n"
        "verdict must be one of: correct, partial, wrong, out_of_context.\n"
        "score must be integer 0..100.\n"
        "Rules:\n"
        "- If answer is out of context to question, verdict=out_of_context and score=0.\n"
        "- If answer is contextually wrong, verdict=wrong and score=0.\n"
        "- If answer is partially right, verdict=partial with 1..79.\n"
        "- If answer is correct and relevant, verdict=correct with 80..100.\n"
        f"Question: {question}\n"
        f"CoreConcept: {concept}\n"
        f"CandidateAnswer: {answer}\n"
    )

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "Evaluate answer correctness and context. Output JSON only."},
                {"role": "user", "content": prompt},
            ]
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(inputs, torch.Tensor):
                    inputs = {"input_ids": inputs}
            except Exception:
                inputs = tokenizer(prompt, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=LLM_JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        decoded = _sanitize_generation_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
        parsed = _extract_json_obj(decoded)
        if not parsed:
            return {"available": False, "verdict": "unknown", "score": 0, "reason": "Invalid LLM judge payload"}

        verdict = str(parsed.get("verdict", "unknown")).strip().lower()
        score = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", "")).strip()
        if verdict not in {"correct", "partial", "wrong", "out_of_context"}:
            verdict = "unknown"
        score = max(0, min(100, score))
        return {"available": True, "verdict": verdict, "score": score, "reason": reason}
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return {"available": False, "verdict": "unknown", "score": 0, "reason": "LLM judge exception"}


def _build_feedback(
    score: int,
    answer: str,
    out_of_context: bool,
    relevance_similarity: float,
    compact_answer: str,
    judge_reason: str,
) -> str:
    if out_of_context:
        return (
            "Out of context. The answer is not relevant to the asked question context, so score is 0. "
            f"Question relevance similarity: {round(relevance_similarity, 4)}. "
            f"Your answer length: {len(answer.split())} words. "
            f"Judged context length: {len(compact_answer.split())} words."
        )
    if score >= 85:
        level = "Excellent"
        advice = "Strong coverage and relevance. Add one metric/example to make it even better."
    elif score >= 65:
        level = "Good"
        advice = "Solid direction. Add clearer structure and more role-specific detail."
    elif score >= 40:
        level = "Fair"
        advice = "Partially correct. Cover missing steps, trade-offs, and practical examples."
    else:
        level = "Needs Improvement"
        advice = "Answer is too generic or off-target. Reframe using a concrete end-to-end approach."

    return (
        f"{level}. {advice} "
        f"Your answer length: {len(answer.split())} words. "
        f"Judged context length: {len(compact_answer.split())} words. "
        f"Judge note: {judge_reason or 'No extra note.'}"
    )


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())]
    parts = [p for p in parts if p]
    return parts or [str(text or "").strip()]


def _warm_models_on_startup() -> None:
    started = time.perf_counter()
    logger.info("Model warmup starting: question=%s semantic=%s", DEEPSEEK_MODEL_NAME, SEMANTIC_MODEL_NAME)

    qa_ok = _load_qa_model() is not None
    sem_ok = _load_embedder() is not None

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "Model warmup done in %sms (questionModelLoaded=%s, semanticModelLoaded=%s)",
        elapsed_ms,
        qa_ok,
        sem_ok,
    )


@app.on_event("startup")
def _startup() -> None:
    logger.info("Simple interview worker booting on %s:%s", WORKER_HOST, WORKER_PORT)
    _warm_models_on_startup()


@app.get("/health")
def health() -> Dict[str, Any]:
    _cleanup_expired_sessions()
    with SESSIONS_LOCK:
        active_sessions = len(SESSIONS)
    return {
        "ok": True,
        "service": "simple-interview-worker",
        "activeSessions": active_sessions,
        "models": {
            "questionGenerator": DEEPSEEK_MODEL_NAME,
            "answerMatcher": SEMANTIC_MODEL_NAME,
        },
    }


@app.post("/startinterveiw", tags=["simple-interview"])
def startinterveiw(payload: StartInterveiwRequest) -> Dict[str, Any]:
    _cleanup_expired_sessions()

    total_questions = payload.totalQuestions or DEFAULT_TOTAL_QUESTIONS
    # Keep at least 2 so interview never ends immediately after q1 intro.
    total_questions = max(2, min(total_questions, MAX_TOTAL_QUESTIONS))

    session_id = str(uuid.uuid4())
    session = InterviewSession(
        session_id=session_id,
        created_at=_now(),
        updated_at=_now(),
        role=_resolve_role(payload),
        difficulty=_normalize_difficulty(payload.difficulty or "medium"),
        total_questions=total_questions,
    )

    with SESSIONS_LOCK:
        SESSIONS[session_id] = session
    if PREFETCH_ON_STARTINTERVIEW:
        _schedule_remaining_generation(session_id)

    logger.info(
        "Started session=%s role=%s difficulty=%s totalQuestions=%s prefetch=%s",
        session_id,
        session.role,
        session.difficulty,
        total_questions,
        PREFETCH_ON_STARTINTERVIEW,
    )

    return {
        "sessionId": session_id,
        "mockInterviewId": session_id,
        "status": "started",
        "totalQuestions": total_questions,
        "cachedQuestions": 0,
        "message": "Interview session created.",
    }


@app.post("/askquestion", tags=["simple-interview"])
def askquestion(payload: AskQuestionRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    session = _get_session_or_404(payload.sessionId)

    with session.lock:
        pending_normal = next((t for t in session.turns if t.asked and not t.answered), None)
        if pending_normal is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Answer pending for questionId={pending_normal.question_id}",
            )

        if session.asked_index >= session.total_questions:
            closing_text = "Interview completed. Thank you. Please submit your interview."
            return {
                "sessionId": session.session_id,
                "completed": True,
                "question": closing_text,
                "questionId": "completed",
                "remaining": 0,
            }

    target_count = session.asked_index + 1
    with session.lock:
        need_generation = len(session.turns) < target_count

    if need_generation:
        if PREFETCH_ON_STARTINTERVIEW:
            _schedule_remaining_generation(session.session_id)
            ready = _wait_for_precomputed_turn(session, target_count, ASKQUESTION_WAIT_MS)
            if not ready:
                # Cache-first mode timed out. Try a direct on-demand generation once
                # to avoid repeated 202 loops when background precompute keeps failing.
                with session.lock:
                    if len(session.turns) < target_count:
                        try:
                            qa = _generate_turn_with_llm(
                                session,
                                target_count,
                                previous_questions=[t.question for t in session.turns],
                            )
                        except Exception:
                            raise HTTPException(
                                status_code=202,
                                detail="Question is still generating. Retry /askquestion shortly.",
                            )
                        existing_questions = {_question_key(t.question) for t in session.turns}
                        qtext = _apply_repeat_policy(
                            role=session.role,
                            track=_resolve_interview_track(session.role),
                            turn_number=target_count,
                            question=qa.get("question", ""),
                            session_existing_keys=existing_questions,
                        )
                        session.turns.append(
                            InterviewTurn(
                                question_id=f"q{target_count}",
                                question=qtext,
                                concept=str(qa.get("concept", "")),
                            )
                        )
                        session.updated_at = _now()
        else:
            # Backward-compatible fallback: synchronous generation when prefetch is disabled.
            with session.lock:
                if len(session.turns) < target_count:
                    try:
                        qa = _generate_turn_with_llm(
                            session,
                            target_count,
                            previous_questions=[t.question for t in session.turns],
                        )
                    except Exception as exc:
                        raise HTTPException(status_code=503, detail=f"Question generation unavailable: {exc}")
                    existing_questions = {_question_key(t.question) for t in session.turns}
                    qtext = _apply_repeat_policy(
                        role=session.role,
                        track=_resolve_interview_track(session.role),
                        turn_number=target_count,
                        question=qa.get("question", ""),
                        session_existing_keys=existing_questions,
                    )
                    session.turns.append(
                        InterviewTurn(
                            question_id=f"q{target_count}",
                            question=qtext,
                            concept=str(qa.get("concept", "")),
                        )
                    )
                    session.updated_at = _now()

    with session.lock:
        turn = session.turns[session.asked_index]
        if not str(turn.question or "").strip():
            try:
                qa = _generate_turn_with_llm(session, session.asked_index + 1, previous_questions=[t.question for t in session.turns])
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Question generation unavailable: {exc}")
            existing_questions = {_question_key(t.question) for t in session.turns}
            turn.question = _apply_repeat_policy(
                role=session.role,
                track=_resolve_interview_track(session.role),
                turn_number=session.asked_index + 1,
                question=qa.get("question", ""),
                session_existing_keys=existing_questions,
            )
            if not str(turn.concept or "").strip():
                turn.concept = str(qa.get("concept", ""))
        turn.asked = True
        session.asked_index += 1
        session.updated_at = _now()

        remaining = max(0, session.total_questions - session.asked_index)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "askquestion served session=%s questionId=%s elapsedMs=%s remaining=%s",
            session.session_id,
            turn.question_id,
            elapsed_ms,
            remaining,
        )
        return {
            "sessionId": session.session_id,
            "completed": False,
            "questionId": turn.question_id,
            "question": turn.question,
            "remaining": remaining,
        }


@app.post("/matchanswer", tags=["simple-interview"])
def matchanswer(payload: MatchAnswerRequest) -> Dict[str, Any]:
    session = _get_session_or_404(payload.sessionId)

    with session.lock:
        matched: Optional[InterviewTurn] = None
        question_for_scoring = ""
        concept_for_scoring = ""

        for turn in session.turns:
            if turn.question_id == payload.questionId:
                matched = turn
                break
        if matched is None:
            raise HTTPException(status_code=404, detail="Question not found for this session")
        matched.answered = True
        question_for_scoring = matched.question
        concept_for_scoring = matched.concept
        session.updated_at = _now()

    answer = str(payload.answer or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Answer is required")

    compact = _compact_answer_contexts(
        answer=answer,
        question=question_for_scoring,
        concept=concept_for_scoring,
    )
    compact_answer = str(compact.get("compactAnswer", "")).strip()
    analysis = _analyze_answer(
        answer=compact_answer or answer,
        question=question_for_scoring,
        concept=concept_for_scoring,
    )
    score = int(analysis["score"])
    judge = {"available": False, "verdict": "unknown", "score": 0, "reason": ""}
    if ANSWER_JUDGE_MODE in {"llm", "hybrid"}:
        judge = _judge_answer_with_llm(
            question=question_for_scoring,
            concept=concept_for_scoring,
            answer=compact_answer or answer,
        )
        if judge.get("available"):
            verdict = str(judge.get("verdict", "unknown"))
            llm_score = int(judge.get("score", 0))
            if verdict in {"wrong", "out_of_context"}:
                # Requested behavior: contextually wrong/out-of-context answers get zero.
                score = 0
                analysis["outOfContext"] = verdict == "out_of_context" or bool(analysis.get("outOfContext"))
            elif verdict in {"partial", "correct"}:
                if ANSWER_JUDGE_MODE == "llm":
                    score = llm_score
                else:
                    score = int(round((0.6 * llm_score) + (0.4 * score)))
                score = max(0, min(100, score))

    feedback = _build_feedback(
        score=score,
        answer=answer,
        out_of_context=bool(analysis.get("outOfContext", False)),
        relevance_similarity=float(analysis.get("relevanceSimilarity", 0.0)),
        compact_answer=compact_answer or answer,
        judge_reason=str(judge.get("reason", "")),
    )

    return {
        "sessionId": session.session_id,
        "questionId": payload.questionId,
        "score": score,
        "overallScore": score,
        "similarity": analysis["similarity"],
        "relevanceSimilarity": analysis["relevanceSimilarity"],
        "outOfContext": analysis["outOfContext"],
        "judgeMode": ANSWER_JUDGE_MODE,
        "llmJudge": judge,
        "evaluatedAnswer": compact_answer or answer,
        "selectedContexts": compact.get("selectedContexts", []),
        "answerTruncatedForJudge": bool(compact.get("truncated", False)),
        "feedback": feedback,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)
