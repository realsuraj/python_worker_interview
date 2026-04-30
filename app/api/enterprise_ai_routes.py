from __future__ import annotations

import base64
import io
import json
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


router = APIRouter(tags=["enterprise-ai"])
_OCR_ENGINE = None

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
    focus = _extract_skills(_text(payload.get("jobDescription")) + " " + _text(payload.get("resume")))[:4] or ROLE_BASELINES.get(family, ["problem solving", "delivery"])
    role = _text(payload.get("role") or payload.get("jobTitle") or family.title()) or family.title()
    seniority = _text(payload.get("seniority") or payload.get("experienceLevel") or "mid") or "mid"
    questions: List[Dict[str, Any]] = []
    for index in range(count):
        focus_area = focus[index % len(focus)]
        question = f"Describe a production situation where you used {focus_area} to improve outcomes for the {role} team." if index % 2 == 0 else f"What trade-offs would you consider when using {focus_area} in a {seniority} {role} role?"
        questions.append({
            "question": question,
            "answer": f"A strong answer should explain context, why {focus_area} mattered, the decision made, measurable impact, and what the candidate would improve next.",
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
            "assessment": "Strong evidence and specificity." if score >= 75 else "Answer needs more depth, metrics, or concrete examples.",
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
    question_text = _text(payload.get("question") or payload.get("questionText"))
    answer_text = _text(payload.get("answer") or payload.get("answerText"))
    score = min(95, max(15, int(28 + _overlap(_tokens(answer_text), _tokens(question_text)) * 50 + min(len(_tokens(answer_text)) // 5, 18))))
    probe = score < 75
    return {
        "accuracyScore": score,
        "accuracyLevel": "high" if score >= 75 else "medium" if score >= 50 else "low",
        "shouldAskCounterQuestion": probe,
        "assessment": "Answer addresses the question clearly." if score >= 75 else "Answer is partially correct but needs more precise evidence.",
        "strengths": ["Relevant concepts detected in the answer."] if score >= 60 else [],
        "gaps": ["Add missing technical depth and concrete evidence."] if probe else [],
        "counterQuestion": "What would you change if this had to operate at 10x scale in production?" if probe else "",
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
    "realtime_performance": _realtime_performance,
    "keyword_detection": _keyword_detection,
    "session_summary": _session_summary,
    "voice_turn_orchestration": _voice_turn,
}


def _dispatch(task: str, payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    builder = TASK_BUILDERS.get(task)
    if builder is None:
        return {"summary": f"No specialized worker builder found for task '{task}'.", "provider": "python-worker", "promptEcho": _text(prompt)[:800]}
    result = builder(payload, prompt)
    result.setdefault("provider", "python-worker")
    result.setdefault("task", task)
    return result


def _infer_result(task: str, model: str, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    selected_model = resolve_model_profile(task, requested_model=model, prompt=prompt, payload=payload)
    structured = _dispatch(task, payload, prompt)
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
