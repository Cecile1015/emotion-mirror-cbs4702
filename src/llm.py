"""Emotion classification with a decoder-only large language model (Step 6).

The model is reached through the DeepSeek API, which follows the OpenAI Chat
Completions format. Classification is cast as conditional text generation: the
prompt (an instruction, optional labelled demonstrations and the input text) is
the left context, and the model generates a short JSON object
``{"label": "<emotion>"}`` token by token.

* ``PromptSpec``            an instruction plus optional few-shot demonstrations;
                            builds the chat messages for one input text.
* ``parse_label``           maps a raw completion to one of the six labels and
                            records how much repair was needed (validity status).
* ``label_distribution``    class probabilities from the token log-probabilities
                            at the position where the label is generated
                            (``is_degenerate`` flags the one-hot values that the
                            service returns at temperature 0).
* ``DeepSeekClassifier``    cached, concurrent API calls with retries, a corrective
                            follow-up for invalid answers and token-usage logging.
* ``estimate_cost``         converts the logged token usage into US dollars.

Every response is appended to a JSON-lines cache in ``results/llm/cache/``, keyed
by a hash of the request. A re-run with the same prompts therefore reads the
responses from the cache: it costs nothing, needs no API key and reproduces the
same numbers even though the remote model is not fully deterministic.

The API key is read from the environment or from a ``.env`` file in the project
root (``DEEPSEEK_API_KEY``); the key is never written to the cache or to any
result file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import LABEL2ID, LABELS, RESULTS_DIR, ROOT

LLM_DIR = RESULTS_DIR / "llm"
CACHE_DIR = LLM_DIR / "cache"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# List prices in US dollars per one million tokens (peak hours), as published on the
# DeepSeek pricing page; off-peak prices are lower. Input tokens that DeepSeek finds
# in its prefix cache ("cache hit") are billed at a much lower rate than new input
# tokens ("cache miss"). Update these values if the pricing page changes.
PRICE_USD_PER_MILLION = {
    "cache_hit_input": 0.006,
    "cache_miss_input": 0.30,
    "output": 1.20,
}

# Status of a parsed answer, from the strictest to the most lenient.
STATUS_ORDER = ["valid", "normalised", "repaired", "invalid", "api_error"]
INVALID_ID = -1  # label id for answers that could not be mapped to an emotion (always scored as wrong)

# Close variants of the label names that the parser maps to the canonical label.
# The list is fixed before any evaluation; how often it is used is reported.
SYNONYMS = {
    "sad": "sadness", "sorrow": "sadness", "grief": "sadness", "unhappiness": "sadness",
    "happy": "joy", "happiness": "joy", "joyful": "joy", "joyous": "joy",
    "loving": "love", "affection": "love", "romance": "love",
    "angry": "anger", "rage": "anger", "annoyance": "anger", "irritation": "anger",
    "afraid": "fear", "scared": "fear", "fearful": "fear", "anxiety": "fear", "anxious": "fear",
    "surprised": "surprise", "shock": "surprise", "shocked": "surprise", "amazement": "surprise",
}
_LABEL_RE = re.compile(r"\b(" + "|".join(sorted(set(LABELS) | set(SYNONYMS), key=len, reverse=True)) + r")\b")
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_LABEL_VALUE_RE = re.compile(r'"label"\s*:\s*"')

CORRECTION_MESSAGE = (
    "Your previous reply could not be read. Reply only with a JSON object of the form "
    '{"label": "<label>"}, where <label> is exactly one of: ' + ", ".join(LABELS) + "."
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def format_user_turn(text: str) -> str:
    """The user message that carries one input text."""
    return f"Text: {text}"


def format_answer(label: str) -> str:
    """The target completion for one label, used for demonstrations."""
    return json.dumps({"label": label})


@dataclass(frozen=True)
class PromptSpec:
    """An instruction (system message) and optional labelled demonstrations.

    Demonstrations are given as earlier conversation turns: the user sends a
    training text and the assistant answers with its gold label in the target
    JSON format. The model therefore sees the input format, the output format
    and the label convention of the dataset before the test text.
    """

    prompt_id: str
    system: str
    demos: tuple[tuple[str, str], ...] = ()
    description: str = ""

    @property
    def n_shots(self) -> int:
        return len(self.demos)

    def messages(self, text: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        for demo_text, demo_label in self.demos:
            msgs.append({"role": "user", "content": format_user_turn(demo_text)})
            msgs.append({"role": "assistant", "content": format_answer(demo_label)})
        msgs.append({"role": "user", "content": format_user_turn(text)})
        return msgs

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "description": self.description,
            "system": self.system,
            "demos": [{"text": t, "label": y} for t, y in self.demos],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptSpec":
        return cls(
            prompt_id=d["prompt_id"],
            system=d["system"],
            demos=tuple((x["text"], x["label"]) for x in d.get("demos", [])),
            description=d.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
@dataclass
class ParseResult:
    label: str | None  # canonical label, or None if the answer is unusable
    status: str  # "valid", "normalised" or "invalid"
    detail: str = ""


def _canonical(value: str) -> tuple[str | None, bool]:
    """Return (label, exact) for a candidate label string."""
    if value in LABELS:
        return value, True
    cleaned = re.sub(r"[^a-z]", "", value.lower())
    if cleaned in LABELS:
        return cleaned, False
    if cleaned in SYNONYMS:
        return SYNONYMS[cleaned], False
    return None, False


def parse_label(raw: str | None) -> ParseResult:
    """Map a raw completion to one of the six labels.

    * ``valid``       a JSON object whose "label" field is exactly one of the six labels;
    * ``normalised``  the label could be recovered by a fixed rule: different case or
                      punctuation, a listed synonym (e.g. "happiness" -> joy), another
                      key name, or a single label word in non-JSON text;
    * ``invalid``     empty output, no label, or more than one candidate label.
    """
    if raw is None or not raw.strip():
        return ParseResult(None, "invalid", "empty output")
    obj, whole_json = None, True
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        whole_json = False  # text around the JSON object, or a broken object
        match = _JSON_OBJECT_RE.search(raw)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                obj = None
    if isinstance(obj, dict):
        if "label" in obj and isinstance(obj["label"], str):
            label, exact = _canonical(obj["label"])
            if label is not None:
                is_clean_json = exact and whole_json
                if is_clean_json:
                    return ParseResult(label, "valid")
                detail = "text around the JSON object" if not whole_json else f"label field '{obj['label']}'"
                return ParseResult(label, "normalised", detail)
        # Another key (e.g. "emotion") or a non-string value: look at all string values.
        values = {v for v in obj.values() if isinstance(v, str)}
        labels = {lab for lab in (_canonical(v)[0] for v in values) if lab is not None}
        if len(labels) == 1:
            return ParseResult(labels.pop(), "normalised", "label found under another key")
        return ParseResult(None, "invalid", "JSON without a usable label")
    # No JSON object: accept the answer only if it names exactly one emotion.
    found = {SYNONYMS.get(m, m) for m in _LABEL_RE.findall(raw.lower())}
    if len(found) == 1:
        return ParseResult(found.pop(), "normalised", "label word in non-JSON text")
    return ParseResult(None, "invalid", "no JSON object" if not found else f"several labels: {sorted(found)}")


# ---------------------------------------------------------------------------
# Token probabilities
# ---------------------------------------------------------------------------
def _clean_piece(token: str) -> str:
    return token.strip().strip('"').strip().lower()


def _label_for_token(token: str, piece_map: dict[str, str] | None = None) -> str | None:
    """The label that a candidate token begins.

    A piece that begins several labels (e.g. "s" for sadness and surprise) is
    resolved with ``piece_map``: the first pieces that the model actually
    generated for each label in the same run (e.g. "s" -> sadness when
    "sadness" is tokenised as "s" + "adness" and "surprise" as "sur" + "prise").
    """
    piece = _clean_piece(token)
    if not piece:
        return None
    if piece_map and piece in piece_map:
        return piece_map[piece]
    matches = [lab for lab in LABELS if lab.startswith(piece) or piece.startswith(lab)]
    return matches[0] if len(matches) == 1 else None


def label_token_logprobs(raw: str, tokens: list[dict]) -> list[list] | None:
    """Top log-probabilities at the first generated token of the label value.

    ``tokens`` is the list of generated tokens with their ``top_logprobs``. The
    token that covers the first character after ``"label": "`` is located by
    accumulating the token strings, which reproduce the generated text.
    """
    match = _LABEL_VALUE_RE.search(raw or "")
    if match is None or not tokens:
        return None
    start, offset = match.end(), 0
    for tok in tokens:
        end = offset + len(tok["token"])
        if offset <= start < end:
            return tok["top"]
        offset = end
    return None


def is_degenerate(top: list[list] | None, floor: float = -9000.0) -> bool:
    """True if only the generated token has a finite log-probability.

    With temperature 0 the service reports log-probability 0 for the chosen
    token and -9999 for every alternative, i.e. a one-hot distribution that
    carries no information about the model's uncertainty.
    """
    if not top:
        return False
    best = max(lp for _, lp in top)
    return best > -1e-6 and all(lp <= floor for _, lp in top if lp != best)


def first_piece_map(records: Iterable[dict]) -> dict[str, str]:
    """Map each generated first piece to its label (e.g. {"s": "sadness", "sur": "surprise"})."""
    votes: dict[str, dict[str, int]] = {}
    for rec in records:
        top, label = rec.get("label_top_logprobs"), rec.get("label")
        if not top or label is None:
            continue
        generated = _clean_piece(max(top, key=lambda t: t[1])[0])
        if generated:
            votes.setdefault(generated, {}).setdefault(label, 0)
            votes[generated][label] += 1
    return {piece: max(v, key=v.get) for piece, v in votes.items()}


def label_distribution(top: list[list] | None, piece_map: dict[str, str] | None = None) -> tuple[np.ndarray | None, float]:
    """Class probabilities from the top log-probabilities at the label position.

    Each candidate token is assigned to the label it begins (e.g. the token
    "sur" to surprise); the probabilities of the six labels are renormalised.
    Returns the distribution and the probability mass that the six labels cover
    before renormalisation (close to 1 when the model is certain of the format).
    """
    if not top:
        return None, 0.0
    proba = np.zeros(len(LABELS))
    for token, logprob in top:
        label = _label_for_token(token, piece_map)
        if label is not None:
            proba[LABEL2ID[label]] += math.exp(logprob)
    mass = float(proba.sum())
    if mass <= 0:
        return None, 0.0
    return proba / mass, mass


# ---------------------------------------------------------------------------
# API client with cache
# ---------------------------------------------------------------------------
def load_env(path: Path | None = None) -> None:
    """Read ``KEY=value`` lines from the project's .env file into the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; the environment may already be set
        return
    load_dotenv(path or ROOT / ".env", override=False)


def _usage_dict(usage) -> dict:
    """Token counts of one response, including DeepSeek's prefix-cache fields."""
    data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
    prompt = int(data.get("prompt_tokens") or 0)
    hit = data.get("prompt_cache_hit_tokens")
    miss = data.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        details = data.get("prompt_tokens_details") or {}
        hit = details.get("cached_tokens") or 0
    hit = int(hit or 0)
    miss = int(miss if miss is not None else prompt - hit)
    return {
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": int(data.get("completion_tokens") or 0),
    }


@dataclass
class DeepSeekClassifier:
    """Zero-/few-shot emotion classification through the DeepSeek Chat Completions API.

    Decoding settings: non-thinking mode (no hidden reasoning tokens), temperature
    0 (always the most probable next token, as far as the service allows), JSON
    output mode and a small output budget, since the answer is about 8 tokens.
    """

    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 32
    request_logprobs: bool = True
    top_logprobs: int = 20
    max_workers: int = 8
    timeout: float = 60.0
    max_retries: int = 5
    cache_dir: Path = CACHE_DIR
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self):
        load_env()
        self.model = self.model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = self.base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY") or None
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.client = None
        if self.api_key:
            from openai import OpenAI

            # The SDK retries rate-limit (429), server (5xx) and connection errors
            # with exponential back-off, up to max_retries times.
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                 timeout=self.timeout, max_retries=self.max_retries)

    # -- configuration -------------------------------------------------------
    @property
    def online(self) -> bool:
        return self.client is not None

    def decoding_params(self) -> dict:
        """Request parameters that can change the answer (part of the cache key)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }

    def cache_key(self, messages: list[dict], repeat: int = 0) -> str:
        payload = json.dumps({"params": self.decoding_params(), "messages": messages, "repeat": repeat},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- one request ---------------------------------------------------------
    def _request(self, messages: list[dict]) -> dict:
        """Send one Chat Completions request and return a compact, JSON-serialisable record."""
        params = self.decoding_params()
        kwargs = {
            "model": params["model"],
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "response_format": params["response_format"],
            "extra_body": {"thinking": params["thinking"]},
        }
        if self.request_logprobs:
            kwargs.update(logprobs=True, top_logprobs=self.top_logprobs)
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # the service may not offer log-probabilities for this model
            if self.request_logprobs and "logprob" in str(exc).lower():
                self.request_logprobs = False
                kwargs.pop("logprobs"), kwargs.pop("top_logprobs")
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        latency = time.perf_counter() - t0
        choice = response.choices[0]
        content = choice.message.content or ""
        tokens = None
        if getattr(choice, "logprobs", None) is not None and choice.logprobs.content:
            tokens = [
                {"token": t.token, "top": [[c.token, c.logprob] for c in (t.top_logprobs or [])]}
                for t in choice.logprobs.content
            ]
        return {
            "content": content,
            "finish_reason": choice.finish_reason,
            "served_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "latency_s": latency,
            **_usage_dict(response.usage),
            "label_top_logprobs": label_token_logprobs(content, tokens) if tokens else None,
        }

    def classify_messages(self, messages: list[dict]) -> dict:
        """Classify one text; send one corrective follow-up if the first answer is unusable."""
        attempts = []
        try:
            first = self._request(messages)
            attempts.append(first)
            parsed = parse_label(first["content"])
            status, used = parsed.status, first
            if parsed.label is None:
                follow_up = messages + [
                    {"role": "assistant", "content": first["content"]},
                    {"role": "user", "content": CORRECTION_MESSAGE},
                ]
                second = self._request(follow_up)
                attempts.append(second)
                parsed = parse_label(second["content"])
                status = "repaired" if parsed.label is not None else "invalid"
                used = second
            error = None
        except Exception as exc:  # e.g. a request refused by the content filter
            parsed, status, used = ParseResult(None, "api_error"), "api_error", None
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        return {
            "label": parsed.label,
            "status": status,
            "detail": parsed.detail if error is None else error,
            "raw": used["content"] if used else None,
            "label_top_logprobs": used["label_top_logprobs"] if used else None,
            "served_model": attempts[-1]["served_model"] if attempts else None,
            "system_fingerprint": attempts[-1]["system_fingerprint"] if attempts else None,
            "n_requests": len(attempts),
            "latency_s": sum(a["latency_s"] for a in attempts),
            "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
            "cache_hit_tokens": sum(a["cache_hit_tokens"] for a in attempts),
            "cache_miss_tokens": sum(a["cache_miss_tokens"] for a in attempts),
            "completion_tokens": sum(a["completion_tokens"] for a in attempts),
            "first_raw": attempts[0]["content"] if attempts else None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # -- many texts, with cache ---------------------------------------------
    def _cache_path(self, run_name: str) -> Path:
        return self.cache_dir / f"{run_name}.jsonl"

    def load_cache(self, run_name: str) -> dict[str, dict]:
        path = self._cache_path(run_name)
        if not path.exists():
            return {}
        records = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["key"]] = rec
        return records

    def run(
        self,
        prompt: PromptSpec,
        texts: Sequence[str],
        ids: Sequence[int],
        run_name: str,
        repeat: int = 0,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Classify ``texts`` with ``prompt``; cached answers are reused, missing ones requested.

        Returns one row per text, in input order, with the predicted label id
        (``INVALID_ID`` for unusable answers), the validity status, the raw answer,
        the token usage, the latency and, if available, the label probabilities.
        """
        texts, ids = list(texts), [int(i) for i in ids]
        keys = [self.cache_key(prompt.messages(t), repeat) for t in texts]
        cache = self.load_cache(run_name)
        todo = [j for j, k in enumerate(keys) if k not in cache]
        if todo and not self.online:
            raise RuntimeError(
                f"{len(todo)} of {len(texts)} responses for run '{run_name}' are not in the cache "
                "and no DEEPSEEK_API_KEY is set."
            )
        if todo:
            path = self._cache_path(run_name)
            bar = None
            if progress:
                from tqdm.auto import tqdm

                bar = tqdm(total=len(todo), desc=run_name)
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool, open(path, "a", encoding="utf-8") as fh:
                futures = {pool.submit(self.classify_messages, prompt.messages(texts[j])): j for j in todo}
                for future in as_completed(futures):
                    j = futures[future]
                    rec = {"key": keys[j], "run": run_name, "prompt_id": prompt.prompt_id, "repeat": repeat,
                           "idx": ids[j], "text": texts[j], **future.result()}
                    with self._lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                    cache[keys[j]] = rec
                    if bar is not None:
                        bar.update(1)
            if bar is not None:
                bar.close()
        return self._to_frame([cache[k] for k in keys])

    @staticmethod
    def _to_frame(records: list[dict]) -> pd.DataFrame:
        rows = []
        piece_map = first_piece_map(records)
        for rec in records:
            proba, mass = label_distribution(rec.get("label_top_logprobs"), piece_map)
            row = {
                "idx": rec["idx"],
                "text": rec["text"],
                "pred": LABEL2ID[rec["label"]] if rec["label"] is not None else INVALID_ID,
                "pred_label": rec["label"] or "invalid",
                "status": rec["status"],
                "detail": rec["detail"],
                "raw": rec["raw"],
                "n_requests": rec["n_requests"],
                "latency_s": rec["latency_s"],
                "prompt_tokens": rec["prompt_tokens"],
                "cache_hit_tokens": rec["cache_hit_tokens"],
                "cache_miss_tokens": rec["cache_miss_tokens"],
                "completion_tokens": rec["completion_tokens"],
                "served_model": rec["served_model"],
                "label_mass": mass,
                "degenerate_logprobs": is_degenerate(rec.get("label_top_logprobs")),
            }
            for i, name in enumerate(LABELS):
                row[f"p_{name}"] = proba[i] if proba is not None else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probabilities, cost and summaries
# ---------------------------------------------------------------------------
def probability_matrix(frame: pd.DataFrame) -> np.ndarray | None:
    """(n, 6) class probabilities of a run, or None if log-probabilities are missing.

    Answers without probabilities (invalid answers, or answers whose label token
    was not found) receive a uniform distribution, so they are maximally uncertain.
    """
    cols = [f"p_{name}" for name in LABELS]
    proba = frame[cols].to_numpy(dtype=float, copy=True)
    has = ~np.isnan(proba).any(axis=1)
    if not has.any():
        return None
    proba[~has] = 1.0 / len(LABELS)
    return proba


def estimate_cost(frame: pd.DataFrame, prices: dict = PRICE_USD_PER_MILLION) -> float:
    """Estimated cost in US dollars of all requests in a run (list prices)."""
    return float(
        frame["cache_hit_tokens"].sum() * prices["cache_hit_input"]
        + frame["cache_miss_tokens"].sum() * prices["cache_miss_input"]
        + frame["completion_tokens"].sum() * prices["output"]
    ) / 1e6


def run_summary(frame: pd.DataFrame, prices: dict = PRICE_USD_PER_MILLION) -> dict:
    """Validity counts, token usage, cost and latency of one run."""
    n = len(frame)
    counts = frame["status"].value_counts()
    prompt = frame["prompt_tokens"].sum()
    return {
        "texts": n,
        "requests": int(frame["n_requests"].sum()),
        **{f"n_{s}": int(counts.get(s, 0)) for s in STATUS_ORDER},
        "input tokens / text": prompt / n,
        "cache-hit share": frame["cache_hit_tokens"].sum() / max(prompt, 1),
        "output tokens / text": frame["completion_tokens"].sum() / n,
        "cost (USD)": estimate_cost(frame, prices),
        "USD per 1,000 texts": estimate_cost(frame, prices) / n * 1000,
        "median latency (s)": float(frame["latency_s"].median()),
        "p95 latency (s)": float(frame["latency_s"].quantile(0.95)),
    }


def save_prompt(prompt: PromptSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompt.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompt(path: Path) -> PromptSpec:
    return PromptSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__: Iterable[str] = [
    "CACHE_DIR", "DEFAULT_MODEL", "INVALID_ID", "LLM_DIR", "PRICE_USD_PER_MILLION", "STATUS_ORDER",
    "DeepSeekClassifier", "ParseResult", "PromptSpec", "estimate_cost", "format_answer", "format_user_turn",
    "first_piece_map", "is_degenerate", "label_distribution", "label_token_logprobs", "load_prompt", "parse_label", "probability_matrix",
    "run_summary", "save_prompt",
]
