"""Inference with the fine-tuned BERT emotion classifier (Lectures 8-9).

The module is used by ``06_bert.ipynb`` (evaluation, attention figures) and by the
FastAPI emotion service. It runs on the CPU or the Apple-silicon GPU (MPS) of a
Mac; no CUDA device is required.

* ``ensure_local_model``     uses ``models/bert/`` if the weights are present,
                             otherwise downloads them from the Hugging Face Hub.
* ``save_model`` / ``load_model``  ``BertTokenizer`` + ``BertForSequenceClassification``.
* ``predict_logits``         batched forward pass over raw strings.
* ``softmax_with_temperature``  calibrated class probabilities (temperature scaling).
* ``attention_rollout``      token-to-token attention propagated through all layers.
* ``merge_wordpieces``       aggregates WordPiece scores to whole words.
* ``BertEmotionClassifier``  the high-level interface: ``predict``, ``predict_proba``
                             and ``explain`` (self-attention weights of every layer and head).

Command-line use from the project root::

    python -m src.bert_infer "i feel so lonely without you" "i was shocked by the news"

Self-attention (Lecture 8). In each of the 12 layers and each of the 12 heads,
token *i* builds a query q_i, every token *j* a key k_j and a value v_j, and the
new representation of *i* is the weighted sum of the values with weights

    alpha_ij = softmax_j( q_i . k_j / sqrt(d_k) ),     d_k = 768 / 12 = 64.

``explain`` returns these alpha_ij. They show where the model *routes*
information, which is not the same as which words *cause* the prediction
(Jain & Wallace, 2019); attention rollout (Abnar & Zuidema, 2020) is offered as a
less layer-specific summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from .config import BERT_DIR, BERT_HUB_REPO, LABELS

# Longest training text = 87 WordPiece tokens including [CLS] and [SEP] (06_bert §3),
# so 128 never truncates a dataset text; longer user inputs are truncated.
MAX_LENGTH = 128
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")
INFO_FILE = "training_info.json"
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


# ---------------------------------------------------------------------------
# Device and model files
# ---------------------------------------------------------------------------
def get_device(prefer: str | None = None) -> torch.device:
    """CUDA if available (Colab), then the Apple-silicon GPU (MPS), then the CPU."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def has_weights(model_dir: Path) -> bool:
    return any((Path(model_dir) / name).exists() for name in WEIGHT_FILES)


def ensure_local_model(model_dir: Path | str = BERT_DIR, repo_id: str = BERT_HUB_REPO) -> Path:
    """Return a folder that contains the fine-tuned model.

    If ``model_dir`` has no weight file, the repository ``repo_id`` is downloaded
    from the Hugging Face Hub into ``model_dir`` (a private repository requires
    ``hf auth login`` or the ``HF_TOKEN`` environment variable first).
    """
    model_dir = Path(model_dir)
    if has_weights(model_dir):
        return model_dir
    from huggingface_hub import snapshot_download

    print(f"No weights in {model_dir}; downloading {repo_id} from the Hugging Face Hub ...")
    snapshot_download(repo_id=repo_id, local_dir=str(model_dir))
    if not has_weights(model_dir):
        raise FileNotFoundError(f"{repo_id} does not contain {' or '.join(WEIGHT_FILES)}")
    return model_dir


def read_training_info(model_dir: Path | str) -> dict:
    """Hyper-parameters, selection results and the fitted temperature (written by 06_bert)."""
    path = Path(model_dir) / INFO_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    model_dir: Path | str = BERT_DIR,
    training_info: dict | None = None,
) -> Path:
    """Save weights, configuration (with id2label) and tokenizer to ``model_dir``.

    transformers 5 stores the tokenizer only as ``tokenizer.json``; ``vocab.txt``
    (one WordPiece per line, line number = token id) is written as well so that
    older transformers versions can load the folder too.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    vocab_file = model_dir / "vocab.txt"
    if not vocab_file.exists():
        vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
        vocab_file.write_text("".join(f"{token}\n" for token, _ in vocab), encoding="utf-8")
    if training_info is not None:
        (model_dir / INFO_FILE).write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    return model_dir


def load_model(
    model_dir: Path | str = BERT_DIR,
    device: torch.device | None = None,
    repo_id: str = BERT_HUB_REPO,
) -> tuple[BertForSequenceClassification, BertTokenizer, dict]:
    """Load the tokenizer and the classifier in evaluation mode.

    ``attn_implementation="eager"`` computes the attention matrix explicitly, which
    is required to return the attention weights (the fused SDPA kernel does not
    expose them). For single sentences the speed difference is negligible.
    """
    model_dir = ensure_local_model(model_dir, repo_id)
    device = device or get_device()
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir), attn_implementation="eager")
    model.to(device).eval()
    return model, tokenizer, read_training_info(model_dir)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    texts: Sequence[str],
    device: torch.device | None = None,
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Return the (n, 6) matrix of logits z for raw input strings.

    Texts are processed in order of length so that each mini-batch is padded to a
    similar length (less wasted computation); the output keeps the input order.
    The model's training/evaluation mode is restored afterwards.
    """
    texts = list(texts)
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    order = np.argsort([len(t.split()) for t in texts], kind="stable")
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        logits[idx] = model(**enc).logits.float().cpu().numpy()
    model.train(was_training)
    return logits


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """p_c = exp(z_c / T) / sum_k exp(z_k / T).

    T > 1 softens over-confident probabilities; the arg-max (the predicted label)
    is unchanged for every T > 0.
    """
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Attention analysis (Lecture 8)
# ---------------------------------------------------------------------------
def attention_rollout(attentions: np.ndarray, residual_weight: float = 0.5) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020).

    ``attentions`` has shape (layers, heads, seq, seq). Each layer's head-averaged
    attention A_l is mixed with the identity to account for the residual
    connection, A'_l = w I + (1 - w) A_l, re-normalised row-wise, and the layers
    are multiplied: R = A'_L ... A'_2 A'_1. Row i of R estimates how much each
    input token contributes to the final representation of token i.
    """
    attentions = np.asarray(attentions, dtype=np.float64)
    n = attentions.shape[-1]
    rollout = np.eye(n)
    for layer in attentions:
        a = residual_weight * np.eye(n) + (1.0 - residual_weight) * layer.mean(axis=0)
        a /= a.sum(axis=-1, keepdims=True)
        rollout = a @ rollout
    return rollout


def merge_wordpieces(tokens: Sequence[str], scores: Sequence[float]) -> tuple[list[str], np.ndarray]:
    """Join WordPiece tokens ("for", "##lor", "##n" -> "forlorn") and sum their scores.

    Attention weights are probability mass, so the mass of a word is the sum over
    its pieces. Special tokens ([CLS], [SEP], [PAD]) are dropped.
    """
    words: list[str] = []
    word_scores: list[float] = []
    for token, score in zip(tokens, scores):
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("##") and words:
            words[-1] += token[2:]
            word_scores[-1] += float(score)
        else:
            words.append(token)
            word_scores.append(float(score))
    return words, np.asarray(word_scores)


@dataclass
class Explanation:
    """Prediction and self-attention weights for one input text."""

    text: str
    label: str
    probabilities: dict[str, float]
    tokens: list[str]                      # WordPiece tokens including [CLS] and [SEP]
    attentions: np.ndarray                 # (layers, heads, seq, seq) attention weights alpha_ij
    cls_attention: np.ndarray              # last layer, mean over heads, row of [CLS]
    rollout: np.ndarray                    # attention rollout, row of [CLS]
    words: list[str] = field(default_factory=list)
    word_cls_attention: np.ndarray = field(default_factory=lambda: np.zeros(0))
    word_rollout: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def top_words(self, k: int = 5, by: str = "rollout") -> list[tuple[str, float]]:
        scores = self.word_rollout if by == "rollout" else self.word_cls_attention
        order = np.argsort(-scores)[:k]
        return [(self.words[i], float(scores[i])) for i in order]

    def to_dict(self, include_matrices: bool = False) -> dict:
        """JSON-serialisable form for the web service (the full tensor is optional)."""
        out = {
            "text": self.text,
            "label": self.label,
            "probabilities": self.probabilities,
            "tokens": self.tokens,
            "cls_attention": self.cls_attention.round(4).tolist(),
            "rollout": self.rollout.round(4).tolist(),
            "words": self.words,
            "word_cls_attention": self.word_cls_attention.round(4).tolist(),
            "word_rollout": self.word_rollout.round(4).tolist(),
        }
        if include_matrices:
            out["attentions"] = self.attentions.round(4).tolist()
        return out


class BertEmotionClassifier:
    """Fine-tuned ``bert-base-uncased`` with calibrated probabilities and attention output.

    >>> clf = BertEmotionClassifier()                      # models/bert/ (or the Hub)
    >>> clf.predict("i feel so lonely without you")["label"]
    'sadness'
    """

    def __init__(
        self,
        model_dir: Path | str = BERT_DIR,
        device: torch.device | str | None = None,
        temperature: float | None = None,
        repo_id: str = BERT_HUB_REPO,
    ):
        device = torch.device(device) if isinstance(device, str) else device
        self.model, self.tokenizer, self.info = load_model(model_dir, device, repo_id)
        self.device = next(self.model.parameters()).device
        # Temperature fitted on the validation split in 06_bert (1.0 = raw softmax).
        self.temperature = float(temperature if temperature is not None else self.info.get("temperature", 1.0))
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if self.labels != LABELS:
            raise ValueError(f"Label order of the checkpoint {self.labels} differs from src.config.LABELS")

    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        logits = predict_logits(self.model, self.tokenizer, texts, self.device, batch_size)
        return softmax_with_temperature(logits, self.temperature)

    def predict(self, text: str) -> dict:
        proba = self.predict_proba([text])[0]
        return {
            "label": self.labels[int(proba.argmax())],
            "confidence": float(proba.max()),
            "probabilities": {name: float(p) for name, p in zip(self.labels, proba)},
        }

    @torch.no_grad()
    def explain(self, text: str) -> Explanation:
        """Predict one text and return the attention weights of all layers and heads."""
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_attentions=True)
        proba = softmax_with_temperature(out.logits.float().cpu().numpy(), self.temperature)[0]
        # out.attentions: one (batch, heads, seq, seq) tensor per layer -> (layers, heads, seq, seq)
        attentions = torch.stack(out.attentions).squeeze(1).float().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        cls_attention = attentions[-1].mean(axis=0)[0]
        rollout = attention_rollout(attentions)[0]
        words, word_cls = merge_wordpieces(tokens, cls_attention)
        _, word_roll = merge_wordpieces(tokens, rollout)
        return Explanation(
            text=text,
            label=self.labels[int(proba.argmax())],
            probabilities={name: float(p) for name, p in zip(self.labels, proba)},
            tokens=tokens,
            attentions=attentions,
            cls_attention=cls_attention,
            rollout=rollout,
            words=words,
            word_cls_attention=word_cls,
            word_rollout=word_roll,
        )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify the emotion of English texts with the fine-tuned BERT model.")
    parser.add_argument("texts", nargs="+", help="one or more input texts")
    parser.add_argument("--model-dir", default=str(BERT_DIR))
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    parser.add_argument("--top", type=int, default=5, help="number of words to list by attention rollout")
    args = parser.parse_args(argv)

    clf = BertEmotionClassifier(args.model_dir, device=args.device)
    print(f"model: {args.model_dir} | device: {clf.device} | temperature: {clf.temperature:.3f}")
    for text in args.texts:
        exp = clf.explain(text)
        probs = "  ".join(f"{name} {p:.3f}" for name, p in exp.probabilities.items())
        words = ", ".join(f"{w} ({s:.2f})" for w, s in exp.top_words(args.top))
        print(f"\n{text}\n  -> {exp.label.upper()}  [{probs}]\n  attention rollout from [CLS]: {words}")


if __name__ == "__main__":
    main()
