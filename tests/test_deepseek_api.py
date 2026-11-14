"""Offline checks for optional remote comparison; no paid calls in pytest."""
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.deepseek_service import DeepSeekService
from api.main import create_app
from api.model_service import ModelService
from src.llm import PromptSpec
from src.predictors import Predictor


def adapter(record):
    remote = object.__new__(DeepSeekService)
    remote.prompt = PromptSpec("test-prompt", "Return JSON", ())
    remote.classifier = SimpleNamespace(online=True, temperature=0.0, client=None,
        classify_messages=lambda messages: record)
    return remote


def service(remote=None):
    s = ModelService(["tfidf_lr", "llm"])
    s.started = True
    s.remote = remote
    s.predictors["tfidf_lr"] = Predictor("tfidf_lr", lambda texts: np.array([[.1, .5, .1, .1, .1, .1]]))
    s.temperatures["tfidf_lr"] = 1
    return s


def test_remote_label_not_fake_probability():
    remote = adapter({"status": "valid", "label": "sadness", "served_model": "test-provider", "prompt_tokens": 100})
    with TestClient(create_app(service(remote))) as c:
        result = c.post("/predict", json={"text": "i do not feel happy", "models": ["llm", "tfidf_lr"]}).json()
        assert [r["model_id"] for r in result["results"]] == ["llm", "tfidf_lr"]
        llm = result["results"][0]
        assert llm["confidence"] is None and llm["probabilities"] is None and llm["temperature"] is None
        assert llm["decoding_temperature"] == 0 and llm["usage"]["prompt_tokens"] == 100
        assert result["all_agree"] is False
        registry = c.get("/models").json()["models"]
        assert registry[-1]["evaluation_texts"] == 300 and registry[-1]["test_metrics"] is None
        assert registry[-1]["subset300_metrics"]["id"] == "llm"
        assert registry[1]["subset300_metrics"]["id"] == "tfidf_lr"


def test_missing_key_preserves_local_predictions():
    with TestClient(create_app(service())) as c:
        assert c.get("/health").status_code == 200
        result = c.post("/predict", json={"text": "hello", "models": ["tfidf_lr", "llm"]}).json()
        assert len(result["results"]) == 1 and result["errors"][0]["code"] == "model_unavailable"
        assert result["all_agree"] is None
        assert c.post("/predict", json={"text": "hello", "models": ["llm"]}).status_code == 503


@pytest.mark.parametrize("record", [
    {"status": "invalid", "label": None},
    {"status": "api_error", "label": None, "detail": "secret-upstream-value"},
    {"status": "valid", "label": "neutral"},
])
def test_provider_failure_is_partial_and_sanitized(record):
    with TestClient(create_app(service(adapter(record)))) as c:
        response = c.post("/predict", json={"text": "hello", "models": ["tfidf_lr", "llm"]})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
        assert response.json()["errors"][0]["code"] == "prediction_failed"
        assert "secret-upstream-value" not in response.text


def test_timeout_does_not_discard_local_result():
    remote = adapter({})
    def timeout(messages):
        raise TimeoutError("secret-upstream-value")
    remote.classifier.classify_messages = timeout
    with TestClient(create_app(service(remote))) as c:
        response = c.post("/predict", json={"text": "hello", "models": ["tfidf_lr", "llm"]})
        assert response.status_code == 200 and response.json()["results"][0]["model_id"] == "tfidf_lr"
        assert "secret-upstream-value" not in response.text


def test_keyless_adapter_and_selected_prompt(monkeypatch):
    import src.llm
    monkeypatch.setattr(src.llm, "load_env", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    remote = DeepSeekService()
    assert not remote.online
    assert remote.prompt.prompt_id == "fs_random_k3" and remote.prompt.n_shots == 18
    assert remote.classifier.model == "deepseek-flash"
    assert remote.classifier.max_retries == 0 and remote.classifier.timeout == 15
    assert remote.classifier.temperature == 0 and remote.classifier.max_tokens == 32
    remote.close()
