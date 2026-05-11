from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
from app.services.foundation_pipeline import append_foundation_example, resolve_model_profile

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import edge_tts as _edge_tts_mod
    _edge_tts_communicate = _edge_tts_mod.Communicate
except Exception:
    _edge_tts_communicate = None


router = APIRouter(tags=["enterprise-ai"])
_OCR_ENGINE = None
logger = logging.getLogger("interview-ai-worker.enterprise")
_WORKER_LLM_ENABLED = os.getenv("ENABLE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
_WORKER_STT_ENABLED = os.getenv("ENABLE_STT", "true").strip().lower() in {"1", "true", "yes", "on"}
_TTS_ENABLED = os.getenv("ENABLE_TTS", "false").strip().lower() in {"1", "true", "yes", "on"}
_INTERVIEW_AUDIO_OUTPUT_DIR = os.getenv("INTERVIEW_AUDIO_OUTPUT_DIR", "generated_audio").strip() or "generated_audio"
_INTERVIEW_AUDIO_URL_PREFIX = (os.getenv("INTERVIEW_AUDIO_URL_PREFIX", "/audio/interview").strip() or "/audio/interview").rstrip("/")
_VOICE_PROFILE_TO_EDGE_VOICE: Dict[str, str] = {
    "hr_friendly_indian_en": "en-IN-NeerjaNeural",
    "hr_friendly_indian_male": "en-IN-AaravNeural",
    "indian_professional_female": "en-IN-PriyaNeural",
    "indian_expressive_female": "en-IN-AnanyaNeural",
    "global_professional_female": "en-US-AriaNeural",
    "global_professional_male": "en-US-GuyNeural",
    "uk_professional_female": "en-GB-SoniaNeural",
    "uk_professional_male": "en-GB-RyanNeural",
}

SKILL_GROUPS: Dict[str, List[str]] = {
    "backend": ["java", "spring", "spring boot", "python", "django", "flask", "node", "express", "microservices", "rest api", "sql", "postgresql", "mysql", "docker", "kubernetes"],
    "frontend": ["react", "next.js", "javascript", "typescript", "redux", "html", "css", "vite"],
    "data": ["sql", "spark", "airflow", "etl", "analytics", "mongodb", "redis", "kafka"],
    "ai": ["machine learning", "deep learning", "nlp", "llm", "hugging face", "pytorch", "tensorflow", "rag", "faiss", "qdrant", "vector search"],
    "business": ["sales", "business development", "crm", "lead generation", "negotiation", "stakeholder management"],
}

ROLE_BASELINES: Dict[str, List[str]] = {
    "backend": ["java", "spring boot", "sql", "rest api", "microservices", "docker"],
    "frontend": ["react", "javascript", "typescript", "css", "html"],
    "fullstack": ["react", "javascript", "typescript", "java", "spring boot", "sql", "docker"],
    "data": ["sql", "python", "etl", "analytics", "airflow"],
    "ai": ["python", "machine learning", "nlp", "llm", "rag", "vector search"],
    "business": ["sales", "crm", "lead generation", "pipeline management", "negotiation"],
}

CERTIFICATIONS: Dict[str, List[str]] = {
    "backend": ["Oracle Java SE", "AWS Developer Associate", "Azure Developer Associate"],
    "frontend": ["Meta Front-End Developer", "JavaScript Algorithms and Data Structures"],
    "data": ["Google Data Analytics", "Databricks Data Engineer Associate"],
    "ai": ["Azure AI Engineer Associate", "AWS Machine Learning Specialty", "Hugging Face Course"],
    "business": ["HubSpot Inbound Sales", "Salesforce Administrator"],
}

CAREER_PATHS: Dict[str, List[str]] = {
    "backend": ["Backend Engineer", "Senior Backend Engineer", "Platform Engineer", "Engineering Manager"],
    "frontend": ["Frontend Engineer", "Senior Frontend Engineer", "UI Architect", "Product Engineer"],
    "fullstack": ["Full-Stack Engineer", "Senior Product Engineer", "Tech Lead", "Engineering Manager"],
    "data": ["Data Analyst", "Analytics Engineer", "Data Engineer", "Data Platform Lead"],
    "ai": ["ML Engineer", "Applied AI Engineer", "AI Platform Engineer", "AI Architect"],
    "business": ["BD Executive", "Account Executive", "Enterprise Sales Manager", "Revenue Leader"],
}


class EnterpriseInferRequest(BaseModel):
    task: str
    model: str = ""
    prompt: str = ""
    maxTokens: int = 512
    request: Dict[str, Any] = Field(default_factory=dict)


def register_enterprise_ai_routes(app: FastAPI) -> None:
    app.include_router(router)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return re.sub("\\s+", " ", str(value or "")).strip()


def _tokens(value: Any) -> List[str]:
    return [token.lower() for token in re.findall("[A-Za-z][A-Za-z0-9+#./-]{1,}", _text(value))]


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        cleaned = _text(item)
        if not cleaned:
            continue
        marker = cleaned.lower()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(cleaned)
    return ordered


def _extract_embedded_request(prompt: str) -> Dict[str, Any]:
    match = re.search("Input JSON:\\s*(\\{.*\\})\\s*$", prompt or "", re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _role_family(payload: Dict[str, Any]) -> str:
    combined = " ".join([
        _text(payload.get("jobTitle")),
        _text(payload.get("title")),
        _text(payload.get("department")),
        _text(payload.get("domain")),
        _text(payload.get("domainLabel")),
        _text(payload.get("jobDescription")),
    ]).lower()
    if any(token in combined for token in ["machine learning", "artificial intelligence", "llm", "nlp", "ai engineer"]):
        return "ai"
    if any(token in combined for token in ["data engineer", "data analyst", "analytics", "etl"]):
        return "data"
    if any(token in combined for token in ["frontend", "react", "ui", "ux", "javascript", "typescript"]):
        return "frontend"
    if any(token in combined for token in ["full stack", "fullstack"]):
        return "fullstack"
    if any(token in combined for token in ["sales", "business development", "account executive", "recruiter"]):
        return "business"
    return "backend"


def _extract_skills(text_value: str) -> List[str]:
    lower = text_value.lower()
    hits: List[str] = []
    for group in SKILL_GROUPS.values():
        for skill in group:
            if skill.lower() in lower:
                hits.append(skill)
    return _unique(hits)


def _experience_years(text_value: str, payload: Dict[str, Any]) -> float:
    explicit = payload.get("experienceYears")
    try:
        if explicit is not None and str(explicit).strip() != "":
            return max(float(explicit), 0.0)
    except Exception:
        pass
    matches = [float(value) for value in re.findall("(\\d+(?:\\.\\d+)?)\\+? years", text_value, re.I)]
    return max(matches) if matches else 0.0


def _education_items(text_value: str) -> List[str]:
    parts = re.split("[\\n.;]", text_value)
    hits = [part.strip() for part in parts if re.search("bachelor|master|mba|phd|diploma|university|college|b\\.?tech|m\\.?tech", part, re.I)]
    return _unique(hits[:5])


def _certification_items(text_value: str) -> List[str]:
    patterns = [
        "aws certified[^\\r\\n]*",
        "azure[^\\r\\n]*associate",
        "oracle[^\\r\\n]*java[^\\r\\n]*",
        "certified[^\\r\\n]*",
    ]
    hits: List[str] = []
    for pattern in patterns:
        hits.extend(re.findall(pattern, text_value, re.I))
    return _unique(hits[:5])


def _contact_details(text_value: str) -> Dict[str, str]:
    email_match = re.search("[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", text_value, re.I)
    phone_match = re.search("(?:\\+?\\d{1,3}[\\s-]?)?(?:\\(?\\d{3}\\)?[\\s-]?)?\\d{3}[\\s-]?\\d{4}", text_value)
    linkedin_match = re.search("(?:https?://)?(?:www\\.)?linkedin\\.com/in/[A-Za-z0-9\\-_/]+", text_value, re.I)
    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "linkedin": linkedin_match.group(0) if linkedin_match else "",
    }


def _decode_file(encoded: str) -> bytes:
    cleaned = encoded.split(",", 1)[-1].strip()
    return base64.b64decode(cleaned)


def _pdf_text(data: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _docx_text(data: bytes) -> str:
    if Document is None:
        return ""
    try:
        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)
    except Exception:
        return ""


def _image_ocr_text(data: bytes) -> Dict[str, str]:
    if Image is None:
        return {"text": "", "engine": "unavailable"}
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        return {"text": "", "engine": "invalid-image"}

    global _OCR_ENGINE
    if PaddleOCR is not None:
        try:
            if _OCR_ENGINE is None:
                _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            blocks = _OCR_ENGINE.ocr(image, cls=True) or []
            lines = [entry[1][0] for block in blocks for entry in (block or []) if entry and len(entry) > 1 and entry[1]]
            return {"text": "\n".join(lines), "engine": "paddleocr"}
        except Exception:
            pass
    if pytesseract is not None:
        try:
            return {"text": pytesseract.image_to_string(image), "engine": "pytesseract"}
        except Exception:
            pass
    return {"text": "", "engine": "ocr-unavailable"}


def _resume_intake(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    text_parts: List[str] = []
    for key in ["resumeText", "text", "transcript", "jobDescription"]:
        value = payload.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    for key in ["resume", "candidateProfile", "profile", "job"]:
        value = payload.get(key)
        if isinstance(value, dict):
            text_parts.extend(str(item) for item in value.values() if isinstance(item, (str, int, float)))
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))

    text_value = _text(" ".join(text_parts))
    extracted_from_file = False
    ocr_engine = "none"
    encoded_file = payload.get("fileBase64") or payload.get("resumeBase64") or payload.get("documentBase64")
    file_name = _text(payload.get("fileName") or payload.get("resumeFileName") or payload.get("documentName")).lower()
    mime_type = _text(payload.get("mimeType") or payload.get("contentType")).lower()
    if encoded_file:
        try:
            data = _decode_file(str(encoded_file))
            if file_name.endswith(".pdf") or mime_type == "application/pdf":
                text_value = _text(_pdf_text(data)) or text_value
            elif file_name.endswith(".docx") or mime_type.endswith("wordprocessingml.document"):
                text_value = _text(_docx_text(data)) or text_value
            elif file_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) or mime_type.startswith("image/"):
                ocr_result = _image_ocr_text(data)
                text_value = _text(ocr_result.get("text")) or text_value
                ocr_engine = ocr_result.get("engine") or "none"
            extracted_from_file = True
        except Exception:
            extracted_from_file = True
    if not text_value:
        text_value = _text(prompt)
    return {"text": text_value, "extractedFromFile": extracted_from_file, "ocrEngine": ocr_engine}


def _overlap(left: List[str], right: List[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(item.lower() for item in left)
    right_set = set(item.lower() for item in right)
    return len(left_set & right_set) / max(1, len(right_set))


def _resume_analysis(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    intake = _resume_intake(payload, prompt)
    resume_text = intake["text"]
    family = _role_family(payload)
    expected = ROLE_BASELINES.get(family, ROLE_BASELINES["backend"])
    skills = _extract_skills(resume_text)
    experience_years = _experience_years(resume_text, payload)
    matched = [skill for skill in expected if skill.lower() in {item.lower() for item in skills}]
    missing = [skill for skill in expected if skill.lower() not in {item.lower() for item in skills}]
    ats_score = min(98, 35 + len(matched) * 8 + min(int(experience_years * 6), 24) + (8 if _education_items(resume_text) else 0) + (6 if _contact_details(resume_text).get("email") else 0))
    compatibility = min(98, int(40 + _overlap(skills, expected) * 45 + min(experience_years * 3, 15)))
    improvements = [
        {"title": "Close critical skill gaps", "description": f"Add evidence for missing skills such as {', '.join(missing[:4]) or 'role-specific tooling'}.", "impact": "high"},
        {"title": "Quantify achievements", "description": "Turn responsibilities into outcomes with metrics such as scale, revenue, quality, or latency.", "impact": "medium"},
        {"title": "Keep ATS-friendly structure", "description": "Use clear sections for summary, skills, experience, projects, education, and certifications.", "impact": "medium"},
    ]
    return {
        "overallScore": ats_score,
        "score": ats_score,
        "atsScore": ats_score,
        "categories": [
            {"label": "Keyword Optimisation", "score": min(100, 30 + len(matched) * 10), "status": "good" if len(matched) >= max(2, len(expected) // 2) else "improve"},
            {"label": "Experience Depth", "score": min(100, 25 + int(experience_years * 12)), "status": "good" if experience_years >= 3 else "improve"},
            {"label": "Formatting & Structure", "score": 78 if resume_text else 20, "status": "good" if resume_text else "improve"},
            {"label": "Profile Completeness", "score": min(100, 35 + len(skills) * 4 + (8 if _education_items(resume_text) else 0) + (6 if _certification_items(resume_text) else 0)), "status": "good" if len(skills) >= 6 else "improve"},
        ],
        "strengths": _unique([
            f"Resume shows practical exposure to {', '.join(matched[:4])}." if matched else "Resume provides enough context for role-level analysis.",
            f"Detected {experience_years:.1f} years of experience signal across the profile." if experience_years else "Experience duration should be made more explicit.",
            "Contact details are visible in recruiter-friendly plain text." if _contact_details(resume_text).get("email") else "Contact details should be clearer at the top of the resume.",
        ])[:4],
        "improvements": improvements,
        "suggestions": [{"type": "critical" if index == 0 else "recommended", "title": item["title"], "description": item["description"], "impact": item["impact"]} for index, item in enumerate(improvements)],
        "formattingIssues": [
            {"title": "ATS-safe layout", "description": "Avoid tables, text boxes, and image-only formatting when exporting resumes."},
            {"title": "Scanned resume fallback", "description": "Prefer searchable PDF or DOCX over scanned-only files whenever possible."},
        ],
        "keywords": [{"word": skill, "present": skill in matched} for skill in expected],
        "jobMatch": {"score": compatibility, "roleFamily": family},
        "missingSkills": missing,
        "jobCompatibilityScore": compatibility,
        "candidateProfileSummary": _text(f"Candidate is strongest in {family} work, with {experience_years:.1f} years of experience signal and next-step gaps in {', '.join(missing[:4]) or 'resume tailoring'}."),
        "parsedProfile": {
            "skills": skills,
            "education": _education_items(resume_text),
            "experienceYears": experience_years,
            "certifications": _certification_items(resume_text),
            "contact": _contact_details(resume_text),
            "domainExpertise": family,
        },
        "ocr": {"performed": bool(intake["extractedFromFile"]), "engine": intake["ocrEngine"]},
        "provider": "python-worker",
    }


def _candidate_match(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    analysis = _resume_analysis(payload, prompt)
    family = _role_family(payload)
    matched_skills = [item["word"] for item in analysis["keywords"] if item.get("present")]
    required = ROLE_BASELINES.get(family, ROLE_BASELINES["backend"])
    overlap_score = int(_overlap(matched_skills, required) * 100)
    match_score = min(98, int(35 + overlap_score * 0.45 + min(float(analysis["parsedProfile"]["experienceYears"]) * 5, 18)))
    return {
        "matchScore": match_score,
        "skillSimilarity": overlap_score,
        "experienceAlignment": min(100, int(float(analysis["parsedProfile"]["experienceYears"]) * 14 + 20)),
        "matchingJobs": [{"title": payload.get("jobTitle") or "Matched role", "score": match_score, "reason": "Skill and experience alignment derived from resume evidence."}],
        "missingSkills": analysis["missingSkills"],
        "skillGapRecommendations": [f"Add project or production evidence for {skill}." for skill in analysis["missingSkills"][:4]],
        "suggestedCertifications": CERTIFICATIONS.get(family, CERTIFICATIONS["backend"])[:3],
        "careerPathInsights": CAREER_PATHS.get(family, CAREER_PATHS["backend"]),
        "provider": "python-worker",
    }


def _behavior_analysis(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    transcript = _text(payload.get("answer") or payload.get("transcript") or payload.get("text") or prompt)
    words = _tokens(transcript)
    filler_count = len(re.findall("\\b(um|uh|like|actually|basically)\\b", transcript, re.I))
    return {
        "confidence": "high" if len(words) > 90 else "medium" if len(words) > 45 else "low",
        "communicationClarity": max(20, min(95, 78 - filler_count * 4 + min(len(words) // 12, 12))),
        "signals": {
            "ownership": bool(re.search("\\b(I led|I owned|I delivered|I implemented)\\b", transcript, re.I)),
            "collaboration": bool(re.search("\\b(team|cross-functional|stakeholder|collaborat)\\b", transcript, re.I)),
            "metrics": bool(re.search("\\b\\d+%|\\d+x|\\d+ users|\\d+ ms\\b", transcript, re.I)),
        },
        "observations": _unique([
            "Candidate frames answers with ownership." if re.search("\\bI\\b", transcript) else "Encourage first-person ownership in responses.",
            "Examples include measurable outcomes." if re.search("\\b\\d+%|\\d+x|\\d+ users|\\d+ ms\\b", transcript, re.I) else "Add measurable impact to improve credibility.",
            "Answer length is sufficient for reliable evaluation." if len(words) > 45 else "Probe for more context, action, and result depth.",
        ]),
        "provider": "python-worker",
    }


def _suggestions(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    family = _role_family(payload)
    return {
        "suggestions": [
            f"Prepare STAR stories tied to {family} delivery and stakeholder impact.",
            f"Strengthen evidence around {', '.join(ROLE_BASELINES.get(family, [])[:3])}.",
            "Record mock answers and tighten them to a problem-action-impact structure.",
        ],
        "recommendedCertifications": CERTIFICATIONS.get(family, CERTIFICATIONS["backend"])[:3],
        "careerPathInsights": CAREER_PATHS.get(family, CAREER_PATHS["backend"]),
        "provider": "python-worker",
    }


def _question_bank(payload: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    family = _role_family(payload)
    # Use pre-resolved skills list sent by Java scheduler (from candidate resume + job) when available
    pre_resolved = [s for s in (payload.get("skills") or []) if isinstance(s, str) and s.strip()]
    extracted = _extract_skills(_text(payload.get("jobDescription")) + " " + _text(payload.get("resume")))[:4]
    focus = pre_resolved[:4] or extracted or ROLE_BASELINES.get(family, ["problem solving", "delivery"])
    role = _text(payload.get("role") or payload.get("jobTitle") or family.title()) or family.title()
    seniority = _text(payload.get("seniority") or payload.get("experienceLevel") or "mid") or "mid"
    question_templates = [
        lambda fa: f"Describe a production situation where you used {fa} to improve outcomes for the {role} team.",
        lambda fa: f"What trade-offs would you consider when using {fa} in a {seniority} {role} role?",
        lambda fa: f"How have you applied {fa} to solve a scalability or performance challenge?",
        lambda fa: f"Walk me through how you would debug a critical issue involving {fa} in a live system.",
        lambda fa: f"What is your approach to testing and ensuring reliability when working with {fa}?",
        lambda fa: f"How do you stay current with best practices for {fa}, and what did you change in your last project?",
    ]
    questions: List[Dict[str, Any]] = []
    for index in range(count):
        focus_area = focus[index % len(focus)]
        question = question_templates[index % len(question_templates)](focus_area)
        questions.append({
            "question": question,
            "answer": (
                f"A strong answer explains context (why {focus_area} was chosen), the specific decision made, "
                f"measurable impact (performance, reliability, velocity), and what the candidate would improve next."
            ),
            "focus": focus_area,
            "difficulty": "medium" if seniority in {"junior", "mid"} else "high",
            "competency": "technical" if family != "business" else "business",
        })
    return questions


def _hr_interview_questions(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    requested = int(payload.get("questionCount") or payload.get("requestedQuestionCount") or 5)
    questions = _question_bank(payload, max(1, min(requested, 12)))
    role = _text(payload.get("role") or payload.get("jobTitle") or "Candidate") or "Candidate"
    seniority = _text(payload.get("seniority") or "mid") or "mid"
    return {
        "title": f"{role} interview pack",
        "role": role,
        "seniority": seniority,
        "summary": f"Structured {len(questions)}-question pack generated for {role} with {seniority} focus.",
        "notes": "Questions balance role depth, trade-offs, and behavioral evidence.",
        "questions": questions,
    }


def _mock_interview_questions(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    count = int(payload.get("questionCount") or 5)
    questions = [{"question": item["question"]} for item in _question_bank(payload, max(1, min(count, 8)))]
    if questions:
        questions[0] = {"question": "Introduce yourself."}
    return {"questions": questions}


def _hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _stable_embedding(value: str, size: int = 384) -> List[float]:
    base = (value or " ").encode("utf-8")
    digest = hashlib.sha256(base).digest()
    values: List[float] = []
    while len(values) < size:
        digest = hashlib.sha256(digest + base).digest()
        for index in range(0, len(digest), 2):
            chunk = digest[index:index + 2]
            number = int.from_bytes(chunk, "big", signed=False)
            normalized = round((number / 65535.0) * 2.0 - 1.0, 6)
            values.append(normalized)
            if len(values) >= size:
                break
    return values


def _question_key(payload: Dict[str, Any], sequence_no: int) -> str:
    scope = _text(payload.get("applicationId") or payload.get("jobId") or payload.get("candidateId") or "pack")
    slug = re.sub(r"[^a-z0-9]+", "-", scope.lower()).strip("-") or "pack"
    return f"{slug}-q{sequence_no}"


def _cosine_similarity_score(vec_a: List[float], vec_b: List[float]) -> float:
    """Pure-Python cosine similarity — no sklearn required."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def _build_counter_questions(
    question_key: str,
    question_text: str,
    expected_answer: str,
    focus_area: str,
    voice_profile: str,
    question_index: int,
) -> List[Dict[str, Any]]:
    """Pre-generate 2 cosine-similarity-ranked counter questions for each main question."""
    probes = [
        (
            "counter-probe-1",
            f"Can you give me a concrete example from your own experience where you applied {focus_area} in a real project?",
            "curious",
        ),
        (
            "counter-probe-2",
            f"What trade-offs or limitations did you face when working with {focus_area}, and how did you resolve them?",
            "analytical",
        ),
    ]
    q_embedding = _stable_embedding(question_text)
    a_embedding = _stable_embedding(expected_answer)

    counter_questions: List[Dict[str, Any]] = []
    for cq_suffix, cq_text, emotion in probes:
        cq_key = f"{question_key}-{cq_suffix}"
        cq_embedding = _stable_embedding(cq_text)
        sim_to_q = _cosine_similarity_score(cq_embedding, q_embedding)
        sim_to_a = _cosine_similarity_score(cq_embedding, a_embedding)
        combined_sim = round((sim_to_q + sim_to_a) / 2, 4)

        print(
            f"[SCHEDULER][COUNTER-Q #{question_index}] '{cq_text[:90]}'"
            f" | cosine_sim={combined_sim:.4f} | voice={voice_profile} | emotion={emotion}"
        )
        logger.info(
            "[INTERVIEW-PACK][COUNTER-Q] question=%s | key=%s | cosine_sim=%.4f | emotion=%s | text='%s'",
            question_key, cq_key, combined_sim, emotion, cq_text[:90],
        )

        cq_audio = _audio_asset(cq_key, cq_text, "counter_question", voice_profile, emotion)
        counter_questions.append({
            "questionKey": cq_key,
            "parentKey": question_key,
            "text": cq_text,
            "emotion": emotion,
            "cosineSimilarity": combined_sim,
            "triggerThreshold": 0.50,
            "audio": cq_audio,
            "embedding": cq_embedding,
        })

    # Best-ranked probe first (highest combined cosine similarity)
    counter_questions.sort(key=lambda x: x["cosineSimilarity"], reverse=True)
    return counter_questions


def _duration_ms(text_value: str, minimum: int = 1800) -> int:
    word_count = max(1, len(_tokens(text_value)))
    return max(minimum, min(9000, word_count * 420))


def _edge_voice_for_profile(voice_profile: str) -> str:
    normalized = _text(voice_profile).lower().replace("-", "_").replace(" ", "_")
    if normalized in _VOICE_PROFILE_TO_EDGE_VOICE:
        return _VOICE_PROFILE_TO_EDGE_VOICE[normalized]
    if voice_profile in _VOICE_PROFILE_TO_EDGE_VOICE.values():
        return voice_profile
    return _VOICE_PROFILE_TO_EDGE_VOICE["hr_friendly_indian_en"]


def _audio_destination(audio_key: str, voice_profile: str) -> str:
    return os.path.join(_INTERVIEW_AUDIO_OUTPUT_DIR, voice_profile, f"{audio_key}.mp3")


async def _synthesize_audio_file(text_value: str, voice_name: str, output_path: str) -> bool:
    if _edge_tts_communicate is None:
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[TTS][SYNTHESIZE] voice={voice_name} | path={output_path} | text='{text_value[:80]}...'")
    logger.info("[TTS][SYNTHESIZE] voice=%s | path=%s | text='%s'", voice_name, output_path, text_value[:80])
    communicate = _edge_tts_communicate(text_value, voice_name)
    await communicate.save(output_path)
    success = os.path.exists(output_path) and os.path.getsize(output_path) > 512
    size_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if success:
        print(f"[TTS][DONE] Audio saved | path={output_path} | size={size_bytes} bytes")
        logger.info("[TTS][DONE] Audio saved | path=%s | size=%d bytes", output_path, size_bytes)
    else:
        print(f"[TTS][WARN] Audio file missing or too small | path={output_path} | size={size_bytes} bytes")
        logger.warning("[TTS][WARN] Audio file missing or too small | path=%s | size=%d bytes", output_path, size_bytes)
    return success


def _ensure_audio_file(audio_key: str, text_value: str, voice_profile: str) -> Dict[str, Any]:
    if not _TTS_ENABLED:
        print(f"[TTS][SKIP] ENABLE_TTS=false — skipping audio for key={audio_key}")
        return {"enabled": False, "generated": False, "reason": "ENABLE_TTS is false"}
    if _edge_tts_communicate is None:
        print(f"[TTS][SKIP] edge-tts not installed — skipping audio for key={audio_key}")
        return {"enabled": False, "generated": False, "reason": "edge-tts dependency unavailable"}
    output_path = _audio_destination(audio_key, voice_profile)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 512:
        print(f"[TTS][CACHE] Audio already exists | key={audio_key} | path={output_path} | size={os.path.getsize(output_path)} bytes")
        logger.info("[TTS][CACHE] Audio already exists | key=%s | path=%s", audio_key, output_path)
        return {
            "enabled": True,
            "generated": True,
            "reason": "audio already exists",
            "localPath": output_path.replace("\\", "/"),
            "sizeBytes": os.path.getsize(output_path),
        }
    print(f"[TTS][START] Generating audio | key={audio_key} | voice={_edge_voice_for_profile(voice_profile)} | text='{text_value[:80]}...'")
    logger.info("[TTS][START] Generating audio | key=%s | voice=%s | text='%s'", audio_key, _edge_voice_for_profile(voice_profile), text_value[:80])
    try:
        created = asyncio.run(_synthesize_audio_file(text_value, _edge_voice_for_profile(voice_profile), output_path))
        return {
            "enabled": True,
            "generated": created,
            "reason": "audio synthesized" if created else "edge-tts did not produce a valid file",
            "localPath": output_path.replace("\\", "/"),
            "sizeBytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
        }
    except Exception as ex:
        print(f"[TTS][ERROR] Audio synthesis failed | key={audio_key} | error={ex}")
        logger.error("[TTS][ERROR] Audio synthesis failed | key=%s | error=%s", audio_key, ex)
        return {
            "enabled": True,
            "generated": False,
            "reason": str(ex),
            "localPath": output_path.replace("\\", "/"),
        }


def _audio_asset(audio_key: str, text_value: str, category: str, voice_profile: str, emotion_tone: str = "neutral") -> Dict[str, Any]:
    checksum = _hash_text(f"{audio_key}:{text_value}:{voice_profile}")
    synthesis = _ensure_audio_file(audio_key, text_value, voice_profile)
    base_path = os.path.join(_INTERVIEW_AUDIO_OUTPUT_DIR, voice_profile)
    file_name = f"{audio_key}.mp3"
    local_path = synthesis.get("localPath") if isinstance(synthesis.get("localPath"), str) else os.path.join(base_path, file_name).replace("\\", "/")
    return {
        "audioKey": audio_key,
        "category": category,
        "voiceProfile": voice_profile,
        "emotionTone": emotion_tone,
        "text": text_value,
        "localPath": local_path,
        "staticUrl": f"{_INTERVIEW_AUDIO_URL_PREFIX}/{voice_profile}/{file_name}",
        "durationMs": _duration_ms(text_value),
        "checksumSha256": checksum,
        "metadata": {
            "generated": bool(synthesis.get("generated")),
            "engine": "edge-tts" if bool(synthesis.get("generated")) else "manifest-only",
            "ttsEnabled": bool(synthesis.get("enabled")),
            "reason": _text(synthesis.get("reason")),
            "edgeVoice": _edge_voice_for_profile(voice_profile),
            "sizeBytes": synthesis.get("sizeBytes") or 0,
            "format": "mp3",
        },
    }


def _transition_bundle(question_key: str, voice_profile: str) -> Dict[str, Dict[str, Any]]:
    return {
        "positive": _audio_asset(f"{question_key}-positive", "Good answer. Let us move to the next question.", "transition", voice_profile, "encouraging"),
        "neutral": _audio_asset(f"{question_key}-neutral", "Thank you. Here is the next question.", "transition", voice_profile, "neutral"),
        "corrective": _audio_asset(f"{question_key}-corrective", "Take a moment and focus on the core trade-off in your answer.", "transition", voice_profile, "supportive"),
    }


def _mandatory_keywords(focus_area: str, payload: Dict[str, Any]) -> List[str]:
    keywords = [focus_area]
    keywords.extend(_extract_skills(" ".join([
        _text(payload.get("jobDescription")),
        _text(payload.get("title") or payload.get("jobTitle")),
        _text(payload.get("resumeText") or payload.get("resume")),
    ]))[:4])
    return _unique(keywords)[:5]


def _optional_keywords(focus_area: str, payload: Dict[str, Any]) -> List[str]:
    family = _role_family(payload)
    baseline = ROLE_BASELINES.get(family, ROLE_BASELINES["backend"])
    extras = [item for item in baseline if item.lower() != focus_area.lower()]
    extras.extend(_extract_skills(_text(payload.get("jobDescription")))[:4])
    return _unique(extras)[:6]


def _rubric(question_type: str, focus_area: str, difficulty: str) -> Dict[str, Any]:
    return {
        "questionType": question_type,
        "difficulty": difficulty,
        "dimensions": [
            {"name": "relevance", "weight": 0.35, "expectation": f"Answer stays anchored to {focus_area}."},
            {"name": "specificity", "weight": 0.30, "expectation": "Answer includes concrete implementation details or examples."},
            {"name": "impact", "weight": 0.20, "expectation": "Answer explains measurable outcome, trade-off, or learning."},
            {"name": "communication", "weight": 0.15, "expectation": "Answer is structured and easy to follow."},
        ],
        "passSignals": [
            f"Names the role of {focus_area} in a real project or production decision.",
            "Connects action to outcome using metrics, reliability, user impact, or delivery speed.",
            "Explains a trade-off, limitation, or lesson learned.",
        ],
        "failSignals": [
            "Response stays generic and does not mention an actual example.",
            "Important technical or business constraints are missing.",
        ],
    }


def _adaptive_mappings(question_keys: List[str], voice_profile: str) -> List[Dict[str, Any]]:
    mappings: List[Dict[str, Any]] = []
    for index in range(len(question_keys) - 1):
        transition_key = f"{question_keys[index]}-neutral"
        mappings.append({
            "fromQuestionKey": question_keys[index],
            "toQuestionKey": question_keys[index + 1],
            "decisionBand": "average",
            "priority": (index + 1) * 10,
            "transitionAudioKey": transition_key,
            "conditions": {
                "minScore": 0.45,
                "maxScore": 0.79,
                "voiceProfile": voice_profile,
            },
        })
    return mappings


def _interview_pack_generation(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    requested_count = int(payload.get("questionCount") or payload.get("requestedQuestionCount") or 8)
    count = max(3, min(requested_count, 10))
    base_questions = _question_bank(payload, count)
    voice_profile = _text(payload.get("activeVoiceProfile") or "hr_friendly_indian_en") or "hr_friendly_indian_en"
    round_tag = _text(payload.get("roundTag") or payload.get("tag") or "technical") or "technical"
    role = _text(payload.get("jobTitle") or payload.get("title") or "Candidate") or "Candidate"
    family = _role_family(payload)
    candidate_id = _text(payload.get("candidateId") or "")
    job_id = _text(payload.get("jobId") or "")
    application_id = _text(payload.get("applicationId") or "")
    trigger = _text(payload.get("trigger") or "scheduler")

    print(
        f"\n{'='*70}\n"
        f"[SCHEDULER] *** INTERVIEW PACK GENERATION STARTED ***\n"
        f"[SCHEDULER] jobId={job_id} | applicationId={application_id} | candidateId={candidate_id}\n"
        f"[SCHEDULER] role='{role}' | voice={voice_profile} | roundTag={round_tag}\n"
        f"[SCHEDULER] questions={count} | trigger={trigger} | ttsEnabled={_TTS_ENABLED}\n"
        f"{'='*70}"
    )
    logger.info(
        "[INTERVIEW-PACK][START] jobId=%s applicationId=%s candidateId=%s role='%s' voice=%s questions=%d trigger=%s ttsEnabled=%s",
        job_id, application_id, candidate_id, role, voice_profile, count, trigger, _TTS_ENABLED,
    )

    questions: List[Dict[str, Any]] = []
    transition_audios: List[Dict[str, Any]] = []
    question_keys: List[str] = []

    for index, item in enumerate(base_questions, start=1):
        question_key = _question_key(payload, index)
        question_keys.append(question_key)
        question_text = _text(item.get("question"))
        expected_answer = _text(item.get("answer"))
        focus_area = _text(item.get("focus") or ROLE_BASELINES.get(family, ["problem solving"])[0]) or "problem solving"
        mandatory_keywords = _mandatory_keywords(focus_area, payload)
        optional_keywords = _optional_keywords(focus_area, payload)

        print(f"\n[SCHEDULER][Q#{index:02d}] focus='{focus_area}' | key={question_key}")
        print(f"[SCHEDULER][Q#{index:02d}] QUESTION : {question_text}")
        print(f"[SCHEDULER][Q#{index:02d}] ANSWER   : {expected_answer}")
        logger.info(
            "[INTERVIEW-PACK][QUESTION #%d] key=%s | focus='%s' | text='%s' | answer='%s'",
            index, question_key, focus_area, question_text[:120], expected_answer[:120],
        )

        # Transition audio clips (Indian woman voice, different emotion tones)
        transitions = _transition_bundle(question_key, voice_profile)
        transition_audios.extend(list(transitions.values()))

        # Question audio (Indian woman voice, professional tone)
        print(f"[SCHEDULER][Q#{index:02d}][AUDIO] Generating QUESTION audio | voice={voice_profile} | emotion=professional")
        logger.info("[INTERVIEW-PACK][AUDIO-Q] Generating question audio | key=%s | voice=%s", question_key, voice_profile)
        question_audio = _audio_asset(
            f"{question_key}-question",
            question_text,
            "question",
            voice_profile,
            "professional",
        )
        q_audio_generated = question_audio.get("metadata", {}).get("generated", False)
        print(f"[SCHEDULER][Q#{index:02d}][AUDIO] QUESTION audio {'✓ generated' if q_audio_generated else '✗ skipped/failed'} | url={question_audio.get('staticUrl', '')}")

        # Answer audio (Indian woman voice, warm tone — for review/model answer playback)
        print(f"[SCHEDULER][Q#{index:02d}][AUDIO] Generating ANSWER audio | voice={voice_profile} | emotion=warm")
        logger.info("[INTERVIEW-PACK][AUDIO-A] Generating answer audio | key=%s | voice=%s", question_key, voice_profile)
        answer_audio = _audio_asset(
            f"{question_key}-answer",
            expected_answer,
            "answer",
            voice_profile,
            "warm",
        )
        a_audio_generated = answer_audio.get("metadata", {}).get("generated", False)
        print(f"[SCHEDULER][Q#{index:02d}][AUDIO] ANSWER audio {'✓ generated' if a_audio_generated else '✗ skipped/failed'} | url={answer_audio.get('staticUrl', '')}")

        # Counter questions — pre-generated with cosine similarity ranking (Indian woman voice, curious/analytical tone)
        print(f"[SCHEDULER][Q#{index:02d}][COUNTER-Q] Pre-generating 2 counter questions via cosine similarity ...")
        counter_qs = _build_counter_questions(
            question_key, question_text, expected_answer, focus_area, voice_profile, index
        )

        questions.append({
            "questionKey": question_key,
            "sequenceNo": index,
            "roundTag": round_tag,
            "questionType": "intro" if index == 1 else "main",
            "difficulty": _text(item.get("difficulty") or payload.get("difficulty") or "medium") or "medium",
            "topic": focus_area,
            "subtopic": role,
            "question": question_text,
            "questionText": question_text,
            "expectedAnswer": expected_answer,
            "idealAnswerSummary": expected_answer,
            "mandatoryKeywords": mandatory_keywords,
            "optionalKeywords": optional_keywords,
            "keywords": _unique(mandatory_keywords + optional_keywords)[:8],
            "evaluationRubric": _rubric(_text(item.get("competency") or "technical"), focus_area, _text(item.get("difficulty") or "medium") or "medium"),
            "audioManifest": {
                "question": question_audio,
                "answer": answer_audio,
                "transitions": transitions,
                "counterQuestions": {cq["questionKey"]: cq["audio"] for cq in counter_qs},
            },
            "metadata": {
                "role": role,
                "roleFamily": family,
                "focus": focus_area,
                "trigger": trigger,
                "language": _text(payload.get("language") or "en") or "en",
                "followups": {cq["questionKey"]: {"text": cq["text"], "emotion": cq["emotion"], "cosineSimilarity": cq["cosineSimilarity"], "triggerThreshold": cq["triggerThreshold"]} for cq in counter_qs},
            },
            "embeddings": {
                "question": _stable_embedding(question_text),
                "expectedAnswer": _stable_embedding(expected_answer),
                "counterQuestions": [{"key": cq["questionKey"], "embedding": cq["embedding"], "cosineSimilarity": cq["cosineSimilarity"]} for cq in counter_qs],
            },
        })
        print(f"[SCHEDULER][Q#{index:02d}] ✓ Complete — {len(counter_qs)} counter question(s) pre-generated with cosine similarity")

    adaptive_mappings = _adaptive_mappings(question_keys, voice_profile)
    pack_version = int(datetime.now(timezone.utc).timestamp())
    pack_hash = _hash_text(json.dumps(questions, ensure_ascii=True, sort_keys=True))

    print(
        f"\n{'='*70}\n"
        f"[SCHEDULER] *** INTERVIEW PACK GENERATION COMPLETE ***\n"
        f"[SCHEDULER] jobId={job_id} | applicationId={application_id} | role='{role}'\n"
        f"[SCHEDULER] totalQuestions={len(questions)} | packVersion={pack_version}\n"
        f"[SCHEDULER] voice={voice_profile} | ttsEnabled={_TTS_ENABLED}\n"
        f"{'='*70}\n"
    )
    logger.info(
        "[INTERVIEW-PACK][DONE] jobId=%s applicationId=%s role='%s' totalQuestions=%d packVersion=%d voice=%s",
        job_id, application_id, role, len(questions), pack_version, voice_profile,
    )

    return {
        "title": f"{role} interview pack",
        "provider": "python-worker",
        "questionPackVersion": pack_version,
        "packHash": pack_hash,
        "voiceProfile": voice_profile,
        "roundTag": round_tag,
        "questions": questions,
        "adaptiveMappings": adaptive_mappings,
        "transitionAudios": transition_audios,
        "summary": f"Generated {len(questions)} interview questions with audio (voice={voice_profile}) and counter questions for {role}.",
    }


def _job_draft(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    description_only = bool(payload.get("descriptionOnly") or payload.get("jdOnly"))
    role = _text(payload.get("title") or payload.get("jobTitle") or "Software Engineer") or "Software Engineer"
    department = _text(payload.get("department") or "Engineering") or "Engineering"
    location = _text(payload.get("location") or "Remote") or "Remote"
    family = _role_family(payload)
    skills = ROLE_BASELINES.get(family, ROLE_BASELINES["backend"])
    description = _text(
        f"We are hiring a {role} to join our {department} team. This person will own production delivery, collaborate across product and operations, and turn business requirements into reliable outcomes at scale. The role requires hands-on execution, clear communication, and ownership of quality, performance, and stakeholder alignment. You will work with {', '.join(skills[:4])}, help shape delivery standards, and drive measurable impact through better systems, faster iteration, and disciplined execution."
    )
    if description_only:
        return {"description": description}
    return {
        "title": role,
        "description": description,
        "department": department,
        "location": location,
        "type": "full-time",
        "remote": location.lower() == "remote",
        "requirements": [f"Hands-on experience with {skill}." for skill in skills[:5]],
        "skills": skills[:20],
        "responsibilities": [
            "Own delivery of production-ready solutions with measurable business impact.",
            "Collaborate with cross-functional stakeholders to refine requirements and execution plans.",
            "Drive quality, reliability, and observability across releases.",
            "Use metrics to improve velocity, stability, and customer outcomes.",
        ],
        "preferredQualifications": [
            "Strong written and verbal communication.",
            "Experience operating in fast-moving product environments.",
            "Ability to mentor peers and improve delivery standards.",
        ],
        "interviewAsks": [
            "Walk us through a high-impact project you delivered.",
            "Describe a difficult trade-off you made under time pressure.",
            "How do you ensure quality in production delivery?",
            "Tell us about a time you influenced stakeholders without authority.",
            "What signals tell you a project is off track?",
        ],
        "experience": {"min": 2, "max": 6},
        "salary": {"min": 1200000, "max": 2600000, "currency": "INR"},
        "deadline": "",
        "companyName": _text(payload.get("companyName") or "Talent AI") or "Talent AI",
        "companyDescription": "Enterprise hiring platform focused on end-to-end recruitment intelligence.",
        "companyWebsite": _text(payload.get("companyWebsite")),
        "educationRequirement": "graduate",
        "interviewRounds": "Technical screening, domain interview, hiring manager, HR discussion",
        "industryType": "technology" if family in {"backend", "frontend", "fullstack", "data", "ai"} else family,
        "jdSourceUrl": "",
        "ragSourceUrls": [],
        "interviewSourceUrls": [],
        "status": "active",
    }


def _domain_detection(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    family = _role_family(payload)
    domain = "business" if family == "business" else "technology"
    subdomain_map = {
        "backend": "software_engineering",
        "frontend": "software_engineering",
        "fullstack": "software_engineering",
        "data": "data_science",
        "ai": "ai_ml",
        "business": "business_development",
    }
    return {
        "domain": domain,
        "subdomain": subdomain_map.get(family, "other"),
        "confidence": "high",
        "reasoning": f"Detected strongest alignment with {family} based on role keywords and skill signals.",
    }


def _coding_testcases(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    existing = payload.get("testCases")
    if isinstance(existing, list) and existing:
        return {"testCases": existing}
    title = _text(payload.get("title") or payload.get("prompt") or "problem") or "problem"
    return {
        "testCases": [
            {"input": "simple happy-path input", "output": "expected happy-path output", "public": True},
            {"input": "edge case input", "output": "expected edge case output", "public": True},
            {"input": "minimum boundary input", "output": "expected minimum boundary output", "public": False},
            {"input": "maximum boundary input", "output": "expected maximum boundary output", "public": False},
            {"input": f"unusual input for {title}", "output": "handled safely", "public": False},
        ]
    }


def _coding_solution_bundle(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    snippets = payload.get("codeSnippets") if isinstance(payload.get("codeSnippets"), dict) else {}
    return {
        "starterCode": {
            "java": snippets.get("java") or "class Solution {\n    public int solve(int[] input) {\n        // TODO implement\n        return 0;\n    }\n}",
            "javascript": snippets.get("javascript") or "function solve(input) {\n  // TODO implement\n  return 0;\n}",
            "python": snippets.get("python") or "def solve(input):\n    # TODO implement\n    return 0\n",
        },
        "solutions": {},
        "editorial": "Start with a correct baseline, add boundary tests, then optimize the critical path.",
        "hints": "Break the problem into parsing, core transformation, and edge-case handling before optimizing.",
    }


def _interview_evaluation(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    answers = payload.get("answers") if isinstance(payload.get("answers"), list) else []
    questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    analysis: List[Dict[str, Any]] = []
    scores: List[int] = []
    for index, answer_item in enumerate(answers or [{}]):
        answer_text = _text(answer_item.get("answer") if isinstance(answer_item, dict) else answer_item)
        question_text = ""
        if index < len(questions):
            question_candidate = questions[index]
            question_text = _text(question_candidate.get("question") if isinstance(question_candidate, dict) else question_candidate)
        score = min(95, max(18, int(30 + _overlap(_tokens(answer_text), _tokens(question_text)) * 45 + min(len(_tokens(answer_text)) // 4, 20))))
        scores.append(score)
        analysis.append({
            "questionId": f"q-{index + 1}",
            "question": question_text,
            "answer": answer_text,
            "score": score,
            "assessment": "Strong evidence and specificity." if score >= 75 else "This answer can be stronger with more concrete examples, metrics, or outcomes.",
        })
    overall = int(sum(scores) / max(1, len(scores))) if scores else 42
    return {
        "score": overall,
        "summary": "Interview evaluation completed with worker-side scoring.",
        "strengths": ["Candidate provides structured answers."] if overall >= 70 else ["Candidate shows baseline domain familiarity."],
        "improvements": ["Use more metrics, concrete examples, and trade-off reasoning in each response."],
        "questionAnalysis": analysis,
    }


def _mock_turn_evaluation(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    # Support both flat keys (question/answer) and Java's nested format
    # (currentQuestion.question / candidateAnswer).
    current_q = payload.get("currentQuestion") or {}
    if not isinstance(current_q, dict):
        current_q = {}
    question_text = _text(
        payload.get("question") or
        payload.get("questionText") or
        current_q.get("question", "")
    )
    answer_text = _text(
        payload.get("answer") or
        payload.get("answerText") or
        payload.get("candidateAnswer", "")
    )
    score = min(95, max(15, int(28 + _overlap(_tokens(answer_text), _tokens(question_text)) * 50 + min(len(_tokens(answer_text)) // 5, 18))))
    probe = score < 75
    # Generate a contextual counter question based on the actual interview question.
    if probe:
        q_focus = (question_text[:120] if question_text else "your previous answer").strip()
        if score < 50:
            counter_q = f"Let us look at that once more. Can you explain {q_focus} with a practical example?"
        else:
            counter_q = f"Good start. Can you go deeper on {q_focus} and include one real-world trade-off or limitation?"
    else:
        counter_q = ""
    return {
        "accuracyScore": score,
        "accuracyLevel": "high" if score >= 75 else "medium" if score >= 50 else "low",
        "shouldAskCounterQuestion": probe,
        "assessment": "Answer addresses the question clearly." if score >= 75 else "This answer is on the right track and can be stronger with more precise evidence.",
        "strengths": ["Relevant concepts detected in the answer."] if score >= 60 else [],
        "gaps": ["Add one concrete example, technical detail, or measurable outcome."] if probe else [],
        "counterQuestion": counter_q,
    }


def _realtime_performance(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    answer_text = _text(payload.get("answer") or payload.get("transcript") or prompt)
    words = len(_tokens(answer_text))
    return {
        "latencyBudgetMs": 500,
        "currentAnswerWordCount": words,
        "signal": "strong" if words > 50 else "developing",
        "observation": "Response is on track." if words > 50 else "Probe for more detail before final scoring.",
    }


def _keyword_detection(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    answer_text = _text(payload.get("answer") or payload.get("text") or prompt).lower()
    keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
    coverage = [{"keyword": str(item).strip(), "present": str(item).strip().lower() in answer_text} for item in keywords if str(item).strip()]
    return {"coverage": coverage, "coveredCount": sum(1 for item in coverage if item["present"]), "totalCount": len(coverage)}


def _session_summary(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    evaluation = _interview_evaluation(payload, prompt)
    return {
        "summary": f"Session closed with score {evaluation['score']} and {len(evaluation['questionAnalysis'])} evaluated turns.",
        "strengths": evaluation["strengths"],
        "gaps": evaluation["improvements"],
        "nextSteps": ["Retest after improving metrics-driven and trade-off-based answers."],
    }


def _voice_turn(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    evaluation = _mock_turn_evaluation(payload, prompt)
    return {
        "partialTranscript": _text(payload.get("transcript") or payload.get("answer") or prompt),
        "vad": {"speechDetected": bool(_text(payload.get("transcript") or payload.get("answer") or prompt)), "engine": "text-signal-fallback"},
        "validation": evaluation,
        "counterQuestionPrediction": evaluation.get("counterQuestion"),
        "tts": {"enabled": False, "reason": "Browser speech mode remains active; worker returns orchestration metadata only."},
        "latencyTargets": {"partialSttMs": 500, "validationMs": 500, "counterQuestionMs": 700, "ttsMs": 300},
    }


def _audio_batch_synthesis(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """Batch edge-tts audio synthesis.

    Called by the Java scheduler after it generates Java-fallback questions so that
    audio files are pre-generated without delay. Each item in ``payload["items"]``
    looks like::

        {"key": "qKey-question", "text": "...", "category": "question",
         "voiceProfile": "hr_friendly_indian_en", "emotion": "professional"}

    Returns::

        {"audioItems": {"qKey-question": <audioAsset>, ...}, "count": N}
    """
    items = payload.get("items") or []
    default_voice = _text(payload.get("voiceProfile") or "hr_friendly_indian_en") or "hr_friendly_indian_en"

    if not items:
        print("[AUDIO-BATCH] No items received — nothing to synthesize")
        return {"audioItems": {}, "provider": "python-worker", "count": 0}

    print(
        f"\n{'='*60}\n"
        f"[AUDIO-BATCH] *** BATCH AUDIO SYNTHESIS STARTED ***\n"
        f"[AUDIO-BATCH] items={len(items)} | defaultVoice={default_voice} | ttsEnabled={_TTS_ENABLED}\n"
        f"{'='*60}"
    )
    logger.info("[AUDIO-BATCH][START] items=%d | voice=%s | ttsEnabled=%s", len(items), default_voice, _TTS_ENABLED)

    audio_items: Dict[str, Any] = {}
    for item in items:
        key = _text(item.get("key"))
        text = _text(item.get("text"))
        category = _text(item.get("category") or "question")
        voice = _text(item.get("voiceProfile") or default_voice) or default_voice
        emotion = _text(item.get("emotion") or "neutral") or "neutral"

        if not key or not text:
            continue

        print(f"[AUDIO-BATCH] Synthesizing | key={key} | category={category} | voice={voice} | emotion={emotion} | text='{text[:70]}...'")
        logger.info("[AUDIO-BATCH] Synthesizing | key=%s | category=%s | voice=%s", key, category, voice)

        asset = _audio_asset(key, text, category, voice, emotion)
        audio_items[key] = asset

        generated = asset.get("metadata", {}).get("generated", False)
        print(f"[AUDIO-BATCH] {'✓ generated' if generated else '✗ failed/skipped'} | key={key} | url={asset.get('staticUrl', '')}")

    print(
        f"\n[AUDIO-BATCH] *** BATCH AUDIO SYNTHESIS COMPLETE ***\n"
        f"[AUDIO-BATCH] processed={len(audio_items)}/{len(items)} | voice={default_voice}\n"
        f"{'='*60}\n"
    )
    logger.info("[AUDIO-BATCH][DONE] processed=%d/%d | voice=%s", len(audio_items), len(items), default_voice)
    return {
        "audioItems": audio_items,
        "provider": "python-worker",
        "count": len(audio_items),
        "voiceProfile": default_voice,
    }


TASK_BUILDERS = {
    "resume_analysis": _resume_analysis,
    "candidate_match": _candidate_match,
    "behavior_analysis": _behavior_analysis,
    "candidate_suggestions": _suggestions,
    "hr_interview_questions": _hr_interview_questions,
    "job_draft_generation": _job_draft,
    "domain_detection": _domain_detection,
    "coding_testcases": _coding_testcases,
    "coding_solution_bundle": _coding_solution_bundle,
    "interview_evaluation": _interview_evaluation,
    "mock_interview_turn_evaluation": _mock_turn_evaluation,
    "mock_interview_questions": _mock_interview_questions,
    "interview_pack_generation": _interview_pack_generation,
    "audio_batch_synthesis": _audio_batch_synthesis,
    "realtime_performance": _realtime_performance,
    "keyword_detection": _keyword_detection,
    "session_summary": _session_summary,
    "voice_turn_orchestration": _voice_turn,
}


def _dispatch(task: str, payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    builder = TASK_BUILDERS.get(task)
    if builder is None:
        logger.warning("[ENTERPRISE-AI] task=%s route=enterprise builder=missing llmEnabled=%s sttEnabled=%s ttsEnabled=%s", task, _WORKER_LLM_ENABLED, _WORKER_STT_ENABLED, _TTS_ENABLED)
        return {"summary": f"No specialized worker builder found for task '{task}'.", "provider": "python-worker", "promptEcho": _text(prompt)[:800]}
    logger.info("[ENTERPRISE-AI] task=%s route=enterprise builder=hit llmEnabled=%s sttEnabled=%s ttsEnabled=%s payloadKeys=%s", task, _WORKER_LLM_ENABLED, _WORKER_STT_ENABLED, _TTS_ENABLED, sorted((payload or {}).keys()))
    result = builder(payload, prompt)
    result.setdefault("provider", "python-worker")
    result.setdefault("task", task)
    return result


def _infer_result(task: str, model: str, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    selected_model = resolve_model_profile(task, requested_model=model, prompt=prompt, payload=payload)
    structured = _dispatch(task, payload, prompt)
    logger.info("[ENTERPRISE-AI] task=%s selectedModel=%s complexity=%s provider=%s textLike=%s", task, selected_model.get("model", model or "enterprise-heuristic-worker"), selected_model.get("complexity", "light"), structured.get("provider", "python-worker"), bool(structured))
    append_foundation_example(task, prompt, payload, structured, selected_model.get("model", model or "enterprise-heuristic-worker"), "enterprise-worker")
    return {
        "provider": "python-worker",
        "task": task,
        "model": selected_model.get("model", model or "enterprise-heuristic-worker"),
        "complexity": selected_model.get("complexity", "light"),
        "text": json.dumps(structured, ensure_ascii=True),
        "raw": structured,
        "structured": structured,
        "generatedAt": _now_iso(),
    }


@router.post("/ai/infer", include_in_schema=False)
def ai_infer(body: EnterpriseInferRequest) -> Dict[str, Any]:
    payload = dict(body.request or {}) or _extract_embedded_request(body.prompt)
    logger.info("[ENTERPRISE-AI] endpoint=/ai/infer task=%s requestedModel=%s maxTokens=%s", body.task, body.model, body.maxTokens)
    return _infer_result(body.task, body.model, body.prompt, payload)


@router.post("/ai/resume/analyze", include_in_schema=False)
def ai_resume_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _resume_analysis(payload, "")


@router.post("/ai/candidate/match", include_in_schema=False)
def ai_candidate_match(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _candidate_match(payload, "")


@router.post("/ai/behavior/analyze", include_in_schema=False)
def ai_behavior_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _behavior_analysis(payload, "")


@router.post("/ai/suggestions", include_in_schema=False)
def ai_suggestions(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _suggestions(payload, "")


@router.post("/ai/interview/questions", include_in_schema=False)
def ai_interview_questions(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _hr_interview_questions(payload, "")


@router.post("/ai/job-draft", include_in_schema=False)
def ai_job_draft(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _job_draft(payload, "")


@router.post("/ai/domain/detect", include_in_schema=False)
def ai_domain_detect(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _domain_detection(payload, "")


@router.post("/ai/coding/testcases", include_in_schema=False)
def ai_coding_testcases(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _coding_testcases(payload, "")


@router.post("/ai/coding/solution-bundle", include_in_schema=False)
def ai_coding_solution_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _coding_solution_bundle(payload, "")


@router.post("/ai/voice/turn", include_in_schema=False)
def ai_voice_turn(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _voice_turn(payload, "")
