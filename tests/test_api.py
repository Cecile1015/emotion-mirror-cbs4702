"""HTTP contract/failure tests plus opt-in integration with the saved models."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.model_service import ModelService, calibrated_probabilities
from api.schemas import MODEL_IDS
from src.config import LABELS
from src.predictors import Predictor, load_predictor


@pytest.fixture
def client():
    service = ModelService(["tfidf_lr", "bert"])
    service.started = True
    service.predictors["tfidf_lr"] = Predictor("tfidf_lr", lambda texts: np.array([[.1, .5, .1, .1, .1, .1]]))
    service.temperatures["tfidf_lr"] = 2.0
    service.load_errors["bert"] = "Missing artifacts"
    with TestClient(create_app(service)) as c:
        yield c


@pytest.mark.parametrize("payload", [
    {}, {"text": "   "}, {"text": 42}, {"text": "?!123"}, {"text": "中文"},
    {"text": "hello\x00"}, {"text": "x" * 2001}, {"text": "hello", "models": []},
    {"text": "hello", "models": ["unknown"]}, {"text": "hello", "models": ["bert", "bert"]},
    {"text": "hello", "unknown": True},
])
def test_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_partial_result_and_health(client):
    response = client.post("/predict", json={"text": "  I feel happy  "})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "I feel happy" and data["all_agree"] is None
    assert data["errors"][0]["model_id"] == "bert"
    result = data["results"][0]
    expected = np.sqrt([.1, .5, .1, .1, .1, .1])
    expected /= expected.sum()
    np.testing.assert_allclose(list(result["probabilities"].values()), expected)
    assert result["calibrated"] and result["confidence"] < .5
    assert client.get("/health").status_code == 503
    assert client.get("/models").json()["models"][-1]["available"] is False
    assert client.post("/predict", json={"text": "hello", "models": ["bert"]}).status_code == 503


def test_cors_and_schema(client):
    headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
    assert client.options("/predict", headers=headers).headers["access-control-allow-origin"] == headers["Origin"]
    headers["Origin"] = "https://unrelated.example"
    assert "access-control-allow-origin" not in client.options("/predict", headers=headers).headers
    assert "/predict" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("raw,t", [([[float("nan")] * 6], 1), ([[0] * 6], 1), ([[1/6] * 6], 0)])
def test_invalid_probabilities(raw, t):
    with pytest.raises(ValueError):
        calibrated_probabilities(raw, t)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("EMOTION_RUN_MODEL_TESTS") != "1", reason="Set EMOTION_RUN_MODEL_TESTS=1 for local artifacts")
def test_real_saved_models():
    service = ModelService(device="cpu")
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["ready"], service.load_errors
        identities = {m: id(p) for m, p in service.predictors.items()}
        service.load()
        assert identities == {m: id(p) for m, p in service.predictors.items()}
        for text in ["i feel happy today", "i feel lonely today", "i do not feel happy today"]:
            response = client.post("/predict", json={"text": text, "models": list(MODEL_IDS)})
            assert response.status_code == 200 and not response.json()["errors"]
            for r in response.json()["results"]:
                # Independent reload checks that serving does not replace weights,
                # alter preprocessing or apply a second calibration.
                raw = load_predictor(r["model_id"], device="cpu").predict_proba([text])[0]
                expected = np.maximum(raw.astype(float), np.finfo(float).tiny) ** (1 / r["temperature"])
                expected /= expected.sum()
                np.testing.assert_allclose(list(r["probabilities"].values()), expected, atol=2e-6)
                assert r["label"] == LABELS[int(raw.argmax())]
        long = client.post("/predict", json={"text": "happy " * 180, "models": ["tfidf_lr", "bilstm", "bert"]}).json()
        assert not long["errors"]
        assert [r["truncated"] for r in long["results"]] == [False, True, True]
        assert [r["token_limit"] for r in long["results"]] == [None, 66, 128]
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: client.post("/predict", json={"text": "i feel happy"}), range(3)))
        assert all(r.status_code == 200 and not r.json()["errors"] for r in responses)
    assert not service.predictors
