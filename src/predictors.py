"""One interface for all trained models: raw strings in, class probabilities out.

Used by 08_error_analysis (behavioural tests, word-level explanations) and by the
FastAPI emotion service. Every predictor has ``predict_proba(texts) -> (n, 6)``
with columns in the order of ``src.config.LABELS``.

The role names follow the rest of the project: ``glove_lr`` = GloVe Wikipedia
300d + LR, ``bilstm`` = BiLSTM on fine-tuned GloVe, ``bert`` = fine-tuned
bert-base-uncased.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

import joblib
import numpy as np

from .config import DATA_DIR, METRICS_DIR, MODELS_DIR

CLASSICAL_DIR = MODELS_DIR / "classical"

# Display information for every model of the representation ladder (Steps 2-6).
MODEL_INFO = {
    "bow_nb": {"name": "BoW + Multinomial NB", "step": "Step 2", "lecture": "L5-L6",
               "representation": "word counts", "pretrained": "no", "word_order": "no", "contextual": "no"},
    "bow_lr": {"name": "BoW + LR", "step": "Step 2", "lecture": "L5-L6",
               "representation": "word counts", "pretrained": "no", "word_order": "no", "contextual": "no"},
    "tfidf_lr": {"name": "TF-IDF + LR", "step": "Step 2", "lecture": "L5-L6",
                 "representation": "TF-IDF", "pretrained": "no", "word_order": "no", "contextual": "no"},
    "improved_lr": {"name": "TF-IDF (sublinear) + LR, class weights", "step": "Step 3", "lecture": "L5-L6",
                    "representation": "TF-IDF, sublinear tf", "pretrained": "no", "word_order": "no",
                    "contextual": "no"},
    "glove_lr": {"name": "GloVe mean + LR", "step": "Step 4", "lecture": "L6",
                 "representation": "static word vectors, averaged", "pretrained": "yes (static)",
                 "word_order": "no", "contextual": "no"},
    "bilstm": {"name": "BiLSTM (fine-tuned GloVe)", "step": "Step 4", "lecture": "L7",
               "representation": "static word vectors, fine-tuned", "pretrained": "yes (adapted)",
               "word_order": "yes", "contextual": "no"},
    "bert": {"name": "BERT fine-tuned", "step": "Step 5", "lecture": "L8-L9",
             "representation": "contextual WordPiece embeddings", "pretrained": "yes (contextual)",
             "word_order": "yes", "contextual": "yes"},
    "llm": {"name": "DeepSeek few-shot", "step": "Step 6", "lecture": "L9",
            "representation": "prompted decoder-only LLM", "pretrained": "yes (contextual)",
            "word_order": "yes", "contextual": "yes"},
}
LADDER = ["bow_nb", "bow_lr", "tfidf_lr", "improved_lr", "glove_lr", "bilstm", "bert"]


def temperatures() -> dict[str, float]:
    """Temperatures fitted on the validation split (Steps 3-5); 1.0 for models without one."""
    out = {m: 1.0 for m in LADDER}
    for name in ("temperature_improved_classical.json", "temperature_bilstm.json", "temperature_bert.json"):
        path = METRICS_DIR / name
        if path.exists():
            values = json.loads(path.read_text())
            out.update({m: float(t) for m, t in values.items() if m in out})
    return out


@dataclass
class Predictor:
    """A loaded model with a uniform ``predict_proba`` (raw softmax probabilities)."""

    model_id: str
    predict_proba: Callable[[Sequence[str]], np.ndarray]
    device: str = "cpu"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(list(texts)).argmax(axis=1)


def load_predictor(model_id: str, device: str | None = None) -> Predictor:
    """Load one model of ``LADDER`` (FileNotFoundError if the GloVe cache of ``glove_lr`` is missing)."""
    if model_id in ("bow_nb", "bow_lr", "tfidf_lr", "improved_lr", "glove_lr"):
        if model_id == "glove_lr" and not (DATA_DIR / "embeddings" / "wiki300.vectors.npy").exists():
            raise FileNotFoundError("GloVe cache data/embeddings/wiki300.* not present (created by 04_static_embeddings)")
        pipeline = joblib.load(CLASSICAL_DIR / f"{model_id}.joblib")
        return Predictor(model_id, lambda texts: pipeline.predict_proba(list(texts)))
    if model_id == "bilstm":
        from .bilstm import get_device, load_checkpoint, predict_proba

        dev = get_device(device)
        model, vocab, _ = load_checkpoint(MODELS_DIR / "bilstm" / "bilstm.pt", dev)
        return Predictor(model_id, lambda texts: predict_proba(model, vocab, list(texts), dev), str(dev))
    if model_id == "bert":
        from .bert_infer import BERT_DIR, BertEmotionClassifier

        # Uses models/bert/ or, if the weights are missing, downloads them from the Hugging Face Hub.
        clf = BertEmotionClassifier(BERT_DIR, device=device, temperature=1.0)
        return Predictor(model_id, lambda texts: clf.predict_proba(list(texts)), str(clf.device))
    raise ValueError(f"Unknown model id '{model_id}'. Choose from {LADDER}.")


__all__ = ["CLASSICAL_DIR", "LADDER", "MODEL_INFO", "Predictor", "load_predictor", "temperatures"]
