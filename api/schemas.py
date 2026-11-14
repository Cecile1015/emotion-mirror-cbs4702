"""Typed HTTP contracts. Label order follows src.config, as in Steps 1–7."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelId = Literal["bow_nb", "tfidf_lr", "improved_lr", "bilstm", "bert", "llm"]
MODEL_IDS = ("bow_nb", "tfidf_lr", "improved_lr", "bilstm", "bert")
ALL_MODEL_IDS = (*MODEL_IDS, "llm")
MAX_CHARACTERS = 2000


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_CHARACTERS)
    models: list[ModelId] = Field(default_factory=lambda: ["tfidf_lr", "bert"], min_length=1, max_length=6)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value or not re.search(r"[A-Za-z]", value):
            raise ValueError("Enter English text containing at least one Latin letter.")
        if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
            raise ValueError("Control characters are not supported.")
        return value

    @field_validator("models")
    @classmethod
    def unique_models(cls, value: list[ModelId]) -> list[ModelId]:
        if len(set(value)) != len(value):
            raise ValueError("Choose each model only once.")
        return value


class Prediction(BaseModel):
    model_id: ModelId
    label: str
    confidence: float | None
    probabilities: dict[str, float] | None
    temperature: float | None
    calibrated: bool
    elapsed_ms: float
    device: str
    provider: str = "local"
    served_model: str | None = None
    prompt_id: str | None = None
    n_shots: int | None = None
    parsing_status: str | None = None
    usage: dict[str, int] | None = None
    decoding_temperature: float | None = None
    truncated: bool = False
    input_tokens: int | None = None
    token_limit: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ModelError(BaseModel):
    model_id: ModelId
    code: Literal["model_unavailable", "prediction_failed"]
    message: str


class PredictResponse(BaseModel):
    text: str
    labels: list[str]
    results: list[Prediction]
    errors: list[ModelError]
    all_agree: bool | None
