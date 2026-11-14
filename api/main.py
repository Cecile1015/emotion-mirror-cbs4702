"""Step 8 · Emotion Mirror API.

Run from the project root: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .model_service import ModelService
from .schemas import ALL_MODEL_IDS, PredictRequest, PredictResponse


def create_app(service: ModelService | None = None) -> FastAPI:
    ids = os.environ.get("EMOTION_API_MODELS", ",".join(ALL_MODEL_IDS)).split(",")
    service = service if service is not None else ModelService(
        tuple(m.strip() for m in ids), device=os.environ.get("EMOTION_API_DEVICE", "cpu"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(service.load)
        app.state.models = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="Emotion Mirror API", version="1.1.0", lifespan=lifespan,
                  description="Compare saved English text classifiers from CBS4702 Steps 2–5. "
                              "Optional llm requests use the selected DeepSeek prompt. Confidence is not emotion intensity.")
    origins = [s.strip() for s in os.environ.get(
        "EMOTION_API_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if s.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/health")
    def health(response: Response):
        status = service.health()
        if not status["ready"]:
            response.status_code = 503
        return status

    @app.get("/models")
    def models():
        return service.registry()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, response: Response):
        result = service.predict(request.text, request.models)
        if not result.results:
            response.status_code = 503
        return result

    return app


app = create_app()
