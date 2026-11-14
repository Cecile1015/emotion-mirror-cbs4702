# Render deployment

Prepared 2026-11-14. Public deployment is pending approval of paid compute, publication of the deployment code to GitHub, and configuration of the DeepSeek secret. A prepared browser form is not a deployed service.

## Architecture and resource decision

A Docker multi-stage build exports the Next.js interface and serves it with FastAPI from one HTTPS origin. Five saved text models, Whisper tiny.en, the expression model and the acoustic model run on CPU. DeepSeek remains a server-side provider call. The original local launcher is retained.

The fresh-process memory experiment (`reproducibility/render_memory_profile.json`) measured a peak of 2268.2 MiB on macOS ARM with two CPU threads, including paired requests. This exceeds 2 GiB. The candidate Render plan is **2c-4g: 2 CPUs, 4 GB RAM, US$85/month**, as displayed in the dashboard on 2026-11-14. This is an estimate for choosing capacity, not a guarantee of Linux memory consumption or latency. Verify both after deploying. Render's 512 MB free plan cannot support this complete model configuration. Do not silently remove models to fit it.

Price source: https://render.com/pricing . The compute price excludes DeepSeek usage and any applicable taxes, bandwidth or additional Render charges. No paid service was created during preparation.

## Deployment settings

Use the connected GitHub repository `Cecile1015/emotion-mirror-cbs4702` after publishing the new deployment files.

| Setting | Value |
|---|---|
| Service | emotion-mirror-cbs4702 |
| Language | Docker |
| Branch | main, after reviewing and pushing the prepared changes |
| Region | Singapore |
| Root directory | blank |
| Dockerfile | ./deployment/render/Dockerfile |
| Docker build context | . |
| Docker command | blank; use the image CMD |
| Health check | /healthz |
| Compute | 2c-4g, one instance |
| Auto-deploy | Off |
| Persistent disk | None |

`render.yaml` records equivalent settings for a Blueprint. Creating a Blueprint also creates billable resources; review before applying it.

Set `DEEPSEEK_API_KEY` privately in the new service's Environment settings. Do not commit `.env`, paste the key into a notebook, or include it in a screenshot. Copying the local key to Render requires the account owner's approval. Set `EMOTION_DEEPSEEK_SESSION_CALL_LIMIT=100` and `EMOTION_CPU_THREADS=2`.

The DeepSeek allowance is per process lifetime: each comparison reserves two calls for the initial request and one possible parser repair. Unused reservations are not refunded. Restarting resets this counter; it is **not a monthly spending cap**. The public middleware also limits request sizes, overall/media request rates and concurrent requests. Keep one worker and one instance so these in-memory controls remain shared. These measures do not provide account-level authentication or a provider billing cap.

## Reproducible build inputs

The Docker context is allowlisted in `.dockerignore`; it excludes credentials, virtual environments, raw recordings, datasets, notebooks and build caches. Selected classical pipelines and `bilstm.pt` are included from the project. BERT, Whisper and bonus-model files are fetched from public, pinned revisions in `deployment/render/model_downloads.json`, and checked against byte counts and SHA-256 before use. No Hugging Face access token is needed for these public artifacts.

The original local BERT `training_info.json` is retained because its Hub copy differs. Trained weights were not replaced or retrained. CPU-only PyTorch avoids downloading unnecessary CUDA libraries. The runtime runs as an unprivileged user and binds to Render's `PORT` (default 10000).

## Local verification and limits

- Static Next.js export completed with `scripts/build_render.py`.
- All 40 Python tests passed (39 in the default suite and the separately enabled saved-model integration test). See `reproducibility/render_validation.json`.
- Hosted contract tests cover readiness, static-file isolation, oversized requests and rejection of mismatched model hashes.
- Public-fixture integration passed: audio transcription, TF-IDF and BERT predictions, expression and acoustic predictions. See `reproducibility/render_local_smoke.json`. This test makes no paid DeepSeek request.
- All 19 remote model files matched the pinned manifest. See `reproducibility/render_remote_artifacts.json`.
- Linux x86-64 / Python 3.11 binary dependency resolution succeeded, including CPU PyTorch. See `reproducibility/render_linux_dependencies.json`.
- Docker is not installed on the development Mac, so the actual Linux image has **not** been built or run locally. Render's first build and production tests remain required.
- Live microphone/camera permissions on the group's devices remain a manual acceptance test. Automated fixtures do not establish real-world emotion accuracy. See `MULTIMODAL.md` for the independent evaluation and limitations.

For local hosted-mode verification, from the project root:

```bash
.venv/bin/python scripts/build_render.py
.venv/bin/python -m uvicorn api.hosted:create_hosted_app --factory --host 127.0.0.1 --port 10000
```

Open http://127.0.0.1:10000 while that terminal is running. In a separate terminal, `scripts/check_space.py` accepts a base URL for integration with the downloaded public CREMA-D fixtures. It retains its original Hugging Face-oriented filename; the HTTP contract is the same.

## Release acceptance checklist

- [x] Prepare Docker, Render settings, pinned downloads and same-origin routes.
- [x] Complete local resource measurement, static build and fixture integration.
- [ ] Review the paid plan and approve its recurring charge.
- [ ] Review and push the deployment files to GitHub; keep credentials and caches excluded.
- [ ] Add the DeepSeek key privately to this Render service with the owner's approval.
- [ ] Create the service and inspect successful Docker build/start logs.
- [ ] Verify HTTPS `/healthz`, `/api/models`, static assets and all selected prediction channels.
- [ ] Check real DeepSeek output with one explicitly selected comparison.
- [ ] Verify Origin handling, input limits, partial-failure display, memory and latency on Linux.
- [ ] Test microphone/camera on desktop and phone, then verify the site still works with the development Mac shut down.
- [ ] Record the actual public URL and release commit; stop or delete paid compute when no longer required.
