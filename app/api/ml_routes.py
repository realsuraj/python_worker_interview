from __future__ import annotations

from typing import Any, Callable, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.schemas.ml import ModelSourceStatusRequest, OnlineTrainingSetRequest, SmallModelTrainRequest
from app.services.foundation_pipeline import build_foundation_training_corpus, foundation_dataset_status


def register_ml_routes(app: FastAPI, deps: Dict[str, Callable[..., Any] | Any]) -> None:
    train_small_model = deps["train_small_model"]
    train_online_sources = deps["train_online_sources"]
    online_training_status = deps["online_training_status"]
    load_small_model = deps["load_small_model"]
    small_model_file = deps["small_model_file"]
    safe_int = deps["safe_int"]
    worker_startup_status = deps["worker_startup_status"]
    source_registry = deps["small_model_source_registry"]

    @app.post("/ml/train-small-model", include_in_schema=False)
    def ml_train_small_model(payload: SmallModelTrainRequest) -> Dict[str, Any]:
        result = train_small_model(window_days=max(1, payload.windowDays), min_samples=max(20, payload.minSamples))
        if result.get("ok"):
            worker_startup_status["smallModelReady"] = True
            worker_startup_status["smallModelReason"] = f"Small model trained: {result.get('sampleCount', 0)} samples"
        return result

    @app.post("/ml/online-training-dataset", tags=["ml"])
    def ml_post_online_training_set(payload: OnlineTrainingSetRequest) -> Dict[str, Any]:
        return train_online_sources(payload)

    @app.post("/ml/model-source-status", tags=["ml"])
    def ml_model_source_status(payload: ModelSourceStatusRequest) -> Dict[str, Any]:
        return online_training_status(payload.sourceUrls if isinstance(payload.sourceUrls, list) else [])

    @app.get("/ml/export-model", tags=["ml"])
    def ml_export_model(download: bool = False) -> Any:
        if not small_model_file.exists():
            raise HTTPException(status_code=404, detail="small_question_model.pkl not found")
        if download:
            return FileResponse(
                path=str(small_model_file),
                filename=small_model_file.name,
                media_type="application/octet-stream",
            )
        pack = load_small_model() or {}
        return {
            "ok": True,
            "modelPath": str(small_model_file),
            "downloadUrl": "/ml/export-model?download=true",
            "trainedAt": str(pack.get("trainedAt", "")) if isinstance(pack, dict) else "",
            "sampleCount": safe_int(pack.get("sampleCount"), 0) if isinstance(pack, dict) else 0,
            "sourceCount": len(source_registry(pack if isinstance(pack, dict) else {})),
        }

    @app.get("/ml/foundation-dataset/status", tags=["ml"])
    def ml_foundation_dataset_status() -> Dict[str, Any]:
        return foundation_dataset_status()

    @app.post("/ml/foundation-dataset/build", tags=["ml"])
    def ml_build_foundation_dataset(minSamples: int = 0) -> Dict[str, Any]:
        return build_foundation_training_corpus(minSamples)
