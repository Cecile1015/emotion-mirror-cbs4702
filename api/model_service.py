"""Load the selected saved models once; apply validation temperatures once.

No fitting or test-label-based selection happens here. Local models reuse the
earlier experiments; explicit llm requests use the selected DeepSeek prompt.
Failed optional models do not disable the others.
"""

from __future__ import annotations

import json
import logging
import threading
from time import perf_counter

import numpy as np

from src.config import BERT_DIR, LABELS, METRICS_DIR, MODELS_DIR, RESULTS_DIR
from src.predictors import MODEL_INFO, load_predictor
from .schemas import ALL_MODEL_IDS, MODEL_IDS, ModelError, Prediction, PredictResponse

logger = logging.getLogger(__name__)
CALIBRATION_FILES = {
    "tfidf_lr": "temperature_improved_classical.json",
    "improved_lr": "temperature_improved_classical.json",
    "bilstm": "temperature_bilstm.json",
    "bert": "temperature_bert.json",
}
RATIONALE = {
    "bow_nb": "Count-based generative baseline; illustrates the baseline gap.",
    "tfidf_lr": "Strong, inexpensive sparse baseline and default comparator.",
    "improved_lr": "Controlled classical improvement: sublinear TF and class weights.",
    "bilstm": "Word-order-sensitive GloVe sequence model; competitive with BERT.",
    "bert": "Contextual encoder selected using validation macro-F1.",
}


def calibrated_probabilities(raw: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T), equivalent to logit scaling up to round-off.

    The predictor contract is *raw* probabilities. T was fitted on validation
    data previously; it is not a user-adjustable confidence slider.
    """
    p = np.asarray(raw, dtype=np.float64)
    if p.shape != (1, len(LABELS)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probability matrix from model.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Model probabilities do not sum to one.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite.")
    z = np.log(np.clip(p, np.finfo(np.float64).tiny, 1.0)) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


class ModelService:
    def __init__(self, model_ids=ALL_MODEL_IDS, device: str = "cpu"):
        self.model_ids = tuple(model_ids)
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids) or set(self.model_ids) - set(ALL_MODEL_IDS):
            raise ValueError(f"EMOTION_API_MODELS must contain unique IDs from {ALL_MODEL_IDS}")
        self.remote = None
        self.device = device
        self.predictors = {}
        self.temperatures = {}
        self.load_errors = {}
        self.bert_tokenizer = None
        # Existing torch inference helpers toggle eval/train mode; serialize
        # model calls to avoid races and excessive CPU memory use.
        self.lock = threading.Lock()
        self.started = False

    def load(self) -> None:
        """Called once by FastAPI lifespan, never once per HTTP request."""
        with self.lock:
            if self.started:
                return
            for model_id in self.model_ids:
                try:
                    if model_id == "llm":
                        from .deepseek_service import DeepSeekService
                        self.remote = DeepSeekService()
                        if not self.remote.online:
                            self.load_errors[model_id] = "Set DEEPSEEK_API_KEY in the server environment or project .env."
                        continue
                    t = 1.0
                    if model_id in CALIBRATION_FILES:
                        values = json.loads((METRICS_DIR / CALIBRATION_FILES[model_id]).read_text())
                        t = float(values[model_id])  # missing calibration is an explicit failure
                        if not np.isfinite(t) or t <= 0:
                            raise ValueError("Invalid saved temperature")
                    if model_id == "bert":
                        # Serving stays offline: prepare artifacts explicitly before starting.
                        from src.bert_infer import has_weights
                        if not has_weights(BERT_DIR):
                            raise FileNotFoundError("Prepare local BERT weights before starting the API")
                        from transformers import BertTokenizer
                        self.bert_tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)
                    predictor = load_predictor(model_id, device=self.device)
                    self.predictors[model_id] = predictor
                    self.temperatures[model_id] = t
                except Exception:
                    logger.exception("Failed to load model %s", model_id)
                    self.load_errors[model_id] = "Model artifacts or dependencies unavailable; see server log and docs/API.md."
            self.started = True

    def health(self) -> dict:
        local_ids = [m for m in self.model_ids if m != "llm"]
        remote_ready = self.remote is not None and self.remote.online
        ready = self.started and all(m in self.predictors for m in local_ids) and (bool(local_ids) or remote_ready)
        return {"status": "ready" if ready else "degraded" if self.predictors else "unavailable",
                "ready": ready, "available_models": list(self.predictors) + (["llm"] if remote_ready else []),
                "remote_configured": remote_ready, "remote_connectivity_checked": False, "errors": self.load_errors}

    def registry(self) -> dict:
        # Evaluation data are metadata only; never used to alter predictions.
        path = RESULTS_DIR / "app" / "ledger.json"
        records = {}
        if path.exists():
            for row in json.loads(path.read_text())["test"]:
                records[row["id"]] = {k: row[k] for k in ("accuracy", "macro_f1", "val_macro_f1")}
        registry = {"labels": LABELS, "default_models": ["tfidf_lr", "bert"],
                "models": [{"id": m, "name": MODEL_INFO[m]["name"],
                            "role": "baseline" if m in ("bow_nb", "tfidf_lr") else "improved",
                            "rationale": RATIONALE[m], "enabled": m in self.model_ids,
                            "available": m in self.predictors,
                            "temperature": self.temperatures.get(m),
                            "test_metrics": records.get(m), "evaluation_texts": 2000,
                            "error": self.load_errors.get(m)} for m in MODEL_IDS]}
        subset = json.loads(path.read_text()).get("subset300", []) if path.exists() else []
        for row in registry["models"]:
            row["subset300_metrics"] = next((x for x in subset if x["id"] == row["id"]), None)
            row["supports_probabilities"] = True
        registry["models"].append({
            "id": "llm", "name": "DeepSeek few-shot (18 demonstrations)", "role": "llm",
            "rationale": "Selected few-shot prompt from notebook 07; remote decoder-only comparator.",
            "enabled": "llm" in self.model_ids,
            "available": self.remote is not None and self.remote.online,
            "test_metrics": None, "evaluation_texts": 300,
            "subset300_metrics": next((x for x in subset if x["id"] == "llm"), None),
            "temperature": None, "supports_probabilities": False, "requires_network": True,
            "prompt_id": self.remote.prompt.prompt_id if self.remote else None,
            "n_shots": self.remote.prompt.n_shots if self.remote else 18,
            "availability_note": "Configured credentials do not establish provider connectivity or credit balance.",
            "error": self.load_errors.get("llm"),
        })
        return registry

    def predict(self, text: str, model_ids: list[str]) -> PredictResponse:
        results, errors = [], []
        with self.lock:
            for model_id in model_ids:
                if model_id == "llm":
                    continue
                predictor = self.predictors.get(model_id)
                if predictor is None:
                    errors.append(ModelError(model_id=model_id, code="model_unavailable",
                                             message="Model is not loaded. Check /models and /health."))
                    continue
                try:
                    start = perf_counter()
                    count, limit = None, None
                    warnings = []
                    if model_id == "bert":
                        from src.bert_infer import MAX_LENGTH
                        count = len(self.bert_tokenizer.encode(text, add_special_tokens=True, truncation=False))
                        limit = MAX_LENGTH
                    elif model_id == "bilstm":
                        from src.bilstm import MAX_LEN
                        from src.data import tokenize
                        count, limit = len(tokenize(text)), MAX_LEN
                    truncated = count is not None and count > limit
                    if truncated:
                        warnings.append(f"Input truncated to {limit} model tokens; models may see different amounts of text.")
                    raw = predictor.predict_proba([text])
                    t = self.temperatures[model_id]
                    p = calibrated_probabilities(raw, t)[0]
                    results.append(Prediction(
                        model_id=model_id, label=LABELS[int(p.argmax())], confidence=float(p.max()),
                        probabilities=dict(zip(LABELS, map(float, p))), temperature=t,
                        calibrated=model_id in CALIBRATION_FILES,
                        elapsed_ms=round((perf_counter() - start) * 1000, 3), device=predictor.device,
                        truncated=truncated, input_tokens=count, token_limit=limit, warnings=warnings))
                except Exception:
                    logger.exception("Prediction failed for %s", model_id)
                    errors.append(ModelError(model_id=model_id, code="prediction_failed",
                                             message="Inference failed; see the server log."))
        # Network waiting is outside the local-model lock: an LLM timeout must
        # not hold up another request for local inference.
        if "llm" in model_ids:
            if self.remote is None or not self.remote.online:
                errors.append(ModelError(model_id="llm", code="model_unavailable",
                                         message="DeepSeek is not configured; set the server API key."))
            else:
                try:
                    results.append(self.remote.predict(text))
                except Exception:
                    logger.warning("DeepSeek prediction failed (upstream details withheld).")
                    errors.append(ModelError(model_id="llm", code="prediction_failed",
                                             message="DeepSeek unavailable, timed out, or returned invalid output. Check server credentials, credit and connectivity."))
        results.sort(key=lambda result: model_ids.index(result.model_id))
        agreement = len({r.label for r in results}) == 1 if len(results) >= 2 and not errors else None
        return PredictResponse(text=text, labels=LABELS, results=results, errors=errors, all_agree=agreement)

    def close(self) -> None:
        with self.lock:
            if self.remote is not None:
                self.remote.close()
                self.remote = None
            self.predictors.clear()
            self.temperatures.clear()
            self.load_errors.clear()
            self.bert_tokenizer = None
            self.started = False
