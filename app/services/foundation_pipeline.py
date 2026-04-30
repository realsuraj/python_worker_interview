from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


def _worker_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _path_from_env(env_key: str, filename: str) -> Path:
    raw = str(os.getenv(env_key, "")).strip()
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return _worker_data_dir() / filename


OLLAMA_BASE_URL = str(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip().rstrip("/")
OLLAMA_LIGHT_MODEL = str(os.getenv("OLLAMA_LIGHT_MODEL", "qwen2:1.5b")).strip() or "qwen2:1.5b"
OLLAMA_HEAVY_MODEL = str(os.getenv("OLLAMA_HEAVY_MODEL", "mistral:7b-instruct")).strip() or "mistral:7b-instruct"
OLLAMA_AUTO_PULL = str(os.getenv("OLLAMA_AUTO_PULL", "true")).strip().lower() in {"1", "true", "yes", "on"}
OLLAMA_PRELOAD_ON_STARTUP = str(os.getenv("OLLAMA_PRELOAD_ON_STARTUP", "true")).strip().lower() in {"1", "true", "yes", "on"}
OLLAMA_PULL_TIMEOUT_SECONDS = max(15, int(str(os.getenv("OLLAMA_PULL_TIMEOUT_SECONDS", "900")).strip() or "900"))
AI_FOUNDATION_DATASET_FILE = _path_from_env("AI_FOUNDATION_DATASET_FILE", "ai_foundation_dataset.jsonl")
AI_FOUNDATION_EXPORT_FILE = _path_from_env("AI_FOUNDATION_EXPORT_FILE", "ai_foundation_training_ready.jsonl")
AI_FOUNDATION_STATS_FILE = _path_from_env("AI_FOUNDATION_STATS_FILE", "ai_foundation_dataset_stats.json")
AI_FOUNDATION_MIN_SAMPLES = max(50, int(str(os.getenv("AI_FOUNDATION_MIN_SAMPLES", "300")).strip() or "300"))
AI_FOUNDATION_MAX_TEXT_CHARS = max(400, int(str(os.getenv("AI_FOUNDATION_MAX_TEXT_CHARS", "4000")).strip() or "4000"))

_HEAVY_TASK_HINTS = {
    "interview",
    "evaluation",
    "counter",
    "voice",
    "session_summary",
    "answer_score",
    "question_generation",
    "question_selection",
    "mock_turn",
}
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}")
_WS_RE = re.compile(r"\s+")


def task_complexity(task: str, prompt: str = "", payload: Optional[Dict[str, Any]] = None) -> str:
    hay = " ".join([
        str(task or "").strip().lower(),
        str(prompt or "")[:300].lower(),
        json.dumps(payload or {}, ensure_ascii=True)[:600].lower(),
    ])
    return "heavy" if any(hint in hay for hint in _HEAVY_TASK_HINTS) else "light"


def resolve_model_profile(
    task: str,
    requested_model: str = "",
    prompt: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    requested = str(requested_model or "").strip()
    complexity = task_complexity(task, prompt=prompt, payload=payload)
    model_name = requested or (OLLAMA_HEAVY_MODEL if complexity == "heavy" else OLLAMA_LIGHT_MODEL)
    return {
        "provider": "ollama",
        "baseUrl": OLLAMA_BASE_URL,
        "complexity": complexity,
        "model": model_name,
    }


def _ollama_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(
        url=f"{OLLAMA_BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="ignore").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def ollama_model_inventory() -> Dict[str, Any]:
    try:
        data = _ollama_json("GET", "/api/tags", timeout=20)
    except urllib_error.HTTPError as ex:
        return {
            "ok": False,
            "models": [],
            "reason": f"ollama_http_error:{ex.code}",
        }
    except urllib_error.URLError as ex:
        return {
            "ok": False,
            "models": [],
            "reason": f"ollama_url_error:{ex.reason}",
        }
    except TimeoutError:
        return {"ok": False, "models": [], "reason": "ollama_timeout"}
    except json.JSONDecodeError:
        return {"ok": False, "models": [], "reason": "ollama_invalid_json"}
    models = data.get("models", []) if isinstance(data, dict) else []
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict) and str(item.get("name", "")).strip()]
    return {"ok": True, "models": names}


def ensure_ollama_model_available(model_name: str) -> Dict[str, Any]:
    name = str(model_name or "").strip()
    if not name:
        return {"ok": False, "reason": "missing_model_name"}
    inventory = ollama_model_inventory()
    if not inventory.get("ok"):
        return {"ok": False, "reason": inventory.get("reason", "ollama_unreachable"), "model": name}
    installed = set(inventory.get("models", []))
    if name in installed:
        return {"ok": True, "model": name, "pulled": False, "available": True}
    if not OLLAMA_AUTO_PULL:
        return {"ok": False, "model": name, "pulled": False, "available": False, "reason": "model_missing_auto_pull_disabled"}
    try:
        pull_result = _ollama_json(
            "POST",
            "/api/pull",
            payload={"name": name, "stream": False},
            timeout=OLLAMA_PULL_TIMEOUT_SECONDS,
        )
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError):
        return {"ok": False, "model": name, "pulled": False, "available": False, "reason": "pull_failed"}
    refreshed = ollama_model_inventory()
    refreshed_models = set(refreshed.get("models", [])) if refreshed.get("ok") else set()
    available = name in refreshed_models
    return {
        "ok": available,
        "model": name,
        "pulled": True,
        "available": available,
        "reason": "pulled" if available else str(pull_result.get("status", "pull_finished_missing")),
    }


def preload_ollama_models() -> Dict[str, Any]:
    if not OLLAMA_PRELOAD_ON_STARTUP:
        return {
            "ok": True,
            "autoPull": OLLAMA_AUTO_PULL,
            "preloadOnStartup": False,
            "models": {},
        }
    targets = [OLLAMA_LIGHT_MODEL, OLLAMA_HEAVY_MODEL]
    results: Dict[str, Any] = {}
    for model_name in targets:
        results[model_name] = ensure_ollama_model_available(model_name)
    return {
        "ok": all(bool(result.get("ok")) for result in results.values()),
        "autoPull": OLLAMA_AUTO_PULL,
        "preloadOnStartup": OLLAMA_PRELOAD_ON_STARTUP,
        "models": results,
    }


def _sanitize_text(value: Any) -> str:
    text = _WS_RE.sub(" ", str(value or "")).strip()
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text[:AI_FOUNDATION_MAX_TEXT_CHARS]


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            cleaned[str(key)[:120]] = sanitize_payload(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value[:40]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _sanitize_text(value) if not isinstance(value, bool) else value
    return _sanitize_text(value)


def _ensure_dataset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def append_foundation_example(
    task: str,
    prompt: str,
    request_payload: Optional[Dict[str, Any]],
    response_payload: Optional[Dict[str, Any]],
    model_name: str,
    source: str,
) -> Dict[str, Any]:
    record = {
        "ts": int(time.time()),
        "task": str(task or "general").strip() or "general",
        "complexity": task_complexity(task, prompt=prompt, payload=request_payload),
        "model": str(model_name or "").strip(),
        "source": str(source or "worker").strip() or "worker",
        "prompt": _sanitize_text(prompt),
        "request": sanitize_payload(request_payload or {}),
        "response": sanitize_payload(response_payload or {}),
    }
    _ensure_dataset_file(AI_FOUNDATION_DATASET_FILE)
    with AI_FOUNDATION_DATASET_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return {"ok": True, "path": str(AI_FOUNDATION_DATASET_FILE), "task": record["task"]}


def _load_dataset_records() -> List[Dict[str, Any]]:
    _ensure_dataset_file(AI_FOUNDATION_DATASET_FILE)
    records: List[Dict[str, Any]] = []
    for raw_line in AI_FOUNDATION_DATASET_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def foundation_dataset_status() -> Dict[str, Any]:
    records = _load_dataset_records()
    by_task: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_complexity: Dict[str, int] = {"light": 0, "heavy": 0}
    for record in records:
        task = str(record.get("task", "general")).strip() or "general"
        model = str(record.get("model", "unknown")).strip() or "unknown"
        complexity = str(record.get("complexity", "light")).strip() or "light"
        by_task[task] = by_task.get(task, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        by_complexity[complexity] = by_complexity.get(complexity, 0) + 1
    return {
        "ok": True,
        "datasetPath": str(AI_FOUNDATION_DATASET_FILE),
        "exportPath": str(AI_FOUNDATION_EXPORT_FILE),
        "statsPath": str(AI_FOUNDATION_STATS_FILE),
        "recordCount": len(records),
        "minSamplesForTraining": AI_FOUNDATION_MIN_SAMPLES,
        "readyForTraining": len(records) >= AI_FOUNDATION_MIN_SAMPLES,
        "ollamaAutoPull": OLLAMA_AUTO_PULL,
        "models": by_model,
        "tasks": by_task,
        "complexity": by_complexity,
    }


def build_foundation_training_corpus(min_samples: int = 0) -> Dict[str, Any]:
    records = _load_dataset_records()
    required = max(1, int(min_samples or AI_FOUNDATION_MIN_SAMPLES))
    if len(records) < required:
        status = foundation_dataset_status()
        status.update({
            "ok": False,
            "reason": "insufficient_samples",
            "requiredSamples": required,
            "missingSamples": max(0, required - len(records)),
        })
        return status

    export_rows: List[str] = []
    for record in records:
        task = str(record.get("task", "general")).strip() or "general"
        prompt = str(record.get("prompt", "")).strip()
        cleaned_request = record.get("request", {}) if isinstance(record.get("request"), dict) else {}
        cleaned_response = record.get("response", {}) if isinstance(record.get("response"), dict) else {}
        export_rows.append(json.dumps({
            "instruction": f"Complete the recruitment AI task '{task}' using the cleaned input.",
            "input": {
                "prompt": prompt,
                "request": cleaned_request,
            },
            "output": cleaned_response,
            "metadata": {
                "task": task,
                "model": str(record.get("model", "")).strip(),
                "complexity": str(record.get("complexity", "light")).strip(),
                "source": str(record.get("source", "worker")).strip(),
                "capturedAt": int(record.get("ts", 0)),
            },
        }, ensure_ascii=True))

    _ensure_dataset_file(AI_FOUNDATION_EXPORT_FILE)
    AI_FOUNDATION_EXPORT_FILE.write_text("\n".join(export_rows) + ("\n" if export_rows else ""), encoding="utf-8")
    stats = foundation_dataset_status()
    stats.update({
        "ok": True,
        "requiredSamples": required,
        "exportedSamples": len(export_rows),
        "trainingReadyPath": str(AI_FOUNDATION_EXPORT_FILE),
        "generatedAt": int(time.time()),
    })
    AI_FOUNDATION_STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats