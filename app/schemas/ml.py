from __future__ import annotations

from typing import List

from pydantic import BaseModel


class SmallModelTrainRequest(BaseModel):
    windowDays: int = 365
    minSamples: int = 80


class OnlineTrainingSetRequest(BaseModel):
    domain: str = ""
    sourceUrls: List[str] = []
    estimateOnly: bool = True
    forceRetrain: bool = False
    maxItemsPerSource: int = 120


class ModelSourceStatusRequest(BaseModel):
    sourceUrls: List[str] = []
