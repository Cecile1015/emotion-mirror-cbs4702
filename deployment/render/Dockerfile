# The same React interface, built once; Python serves the static assets and APIs.
FROM node:24.22.1-bookworm-slim AS website
WORKDIR /web
ENV NEXT_TELEMETRY_DISABLED=1
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN npm install -g pnpm@11.25.0 && pnpm install --frozen-lockfile
COPY web/ ./
COPY deployment/render/next-export.config.mjs ./next.config.mjs
# Python implements the same-origin APIs in the final image.
RUN rm -rf app/api tests && pnpm build

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 \
    EMOTION_CPU_THREADS=2 EMOTION_DEEPSEEK_SESSION_CALL_LIMIT=100
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements-models.txt requirements-api.txt requirements-speech.txt requirements-multimodal.txt ./
# Explicit CPU wheel prevents several GB of unused CUDA dependencies.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt -r requirements-speech.txt -r requirements-multimodal.txt
COPY deployment/render/model_downloads.json deployment/render/fetch_models.py ./deployment/render/
RUN python deployment/render/fetch_models.py --root /app
COPY api/ ./api/
COPY src/ ./src/
COPY models/classical/ ./models/classical/
COPY models/bert/training_info.json ./models/bert/training_info.json
COPY models/bilstm/bilstm.pt ./models/bilstm/bilstm.pt
COPY results/metrics/ ./results/metrics/
COPY results/app/ledger.json ./results/app/ledger.json
COPY results/llm/selection.json results/llm/prompt_llm.json ./results/llm/
COPY web/data/ ./web/data/
COPY --from=website /web/out ./web/out/
COPY web/licenses/ ./web/licenses/
COPY web/THIRD_PARTY_NOTICES.md ./web/THIRD_PARTY_NOTICES.md
COPY deployment/render/start.py ./start.py
RUN useradd --create-home --uid 10001 mirror && chown -R mirror:mirror /app
USER mirror
EXPOSE 10000
CMD ["python", "start.py"]
