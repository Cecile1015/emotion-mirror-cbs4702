"""Live inference using the exact few-shot prompt selected in notebook 07.

No user text is written into the experimental JSONL caches. Remote requests
are made only when the caller explicitly includes `llm` in /predict.
"""
from __future__ import annotations

import json
from time import perf_counter

from src.config import LABELS
from src.llm import DeepSeekClassifier, LLM_DIR, load_prompt
from .schemas import Prediction


class DeepSeekService:
    def __init__(self):
        selection = json.loads((LLM_DIR / "selection.json").read_text())
        self.prompt = load_prompt(LLM_DIR / "prompt_llm.json")
        if self.prompt.prompt_id != selection["selected_prompts"]["llm_fewshot"]:
            raise ValueError("Saved prompt does not match the selected few-shot experiment")
        params = selection["decoding"]
        self.classifier = DeepSeekClassifier(
            model=selection["requested_model"], temperature=params["temperature"],
            max_tokens=params["max_tokens"], request_logprobs=False,
            timeout=15.0, max_retries=0, max_workers=1,
        )

    @property
    def online(self):
        return self.classifier.online

    def predict(self, text):
        from .public_guard import remote_budget
        remote_budget.reserve()
        start = perf_counter()
        # Reuse the established parser and one corrective follow-up. Do not
        # call run(): it persists input texts in the research cache.
        result = self.classifier.classify_messages(self.prompt.messages(text))
        if result["status"] in ("api_error", "invalid") or result["label"] not in LABELS:
            # Upstream error text may contain request data; never echo it.
            raise RuntimeError("DeepSeek request failed or returned an unusable label")
        return Prediction(
            model_id="llm", label=result["label"], confidence=None, probabilities=None,
            temperature=None, calibrated=False, device="remote", provider="deepseek",
            served_model=result.get("served_model"), prompt_id=self.prompt.prompt_id,
            n_shots=self.prompt.n_shots, parsing_status=result["status"],
            decoding_temperature=self.classifier.temperature,
            usage={key: int(result.get(key) or 0) for key in
                   ("prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens", "n_requests")},
            elapsed_ms=round((perf_counter() - start) * 1000, 3),
            warnings=["No validated six-class probability distribution is available; confidence is not reported.",
                      "Selected text is sent to DeepSeek and billed to the server API account."],
        )

    def close(self):
        if self.classifier.client is not None:
            self.classifier.client.close()
