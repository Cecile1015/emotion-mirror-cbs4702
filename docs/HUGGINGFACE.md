# Hugging Face Docker deployment

The selected target is a public Docker Space owned by `Cecile1015`, using `cpu-basic` (2 vCPU, 16 GB). Under the hosting policy verified on 2026-11-14, personal PRO membership costs US$9/month and enables this configuration without hourly hardware charges. DeepSeek calls are billed separately by DeepSeek. CPU Basic can sleep after inactivity; it is not an always-on uptime guarantee.

## Deployment and updates

From the project root, after reviewing the files and activating PRO:

```bash
.venv/bin/python scripts/prepare_space.py
.venv/bin/python scripts/deploy_space.py
```

Preparation creates an allowlisted `.hf-deploy/` source bundle. It does not upload secrets, notebook outputs, datasets, browser data or caches. Large public model artifacts are fetched during the Docker build from pinned revisions and verified with SHA-256; only the selected classical pipelines and BiLSTM checkpoint are uploaded. The same Docker runtime used in the Render preparation serves the full React website and all APIs on port 10000.

The deploy script uses the existing Hugging Face login. It requires PRO and requests CPU Basic only; it never purchases a subscription, requests paid hardware or transfers the DeepSeek key. It refuses to overwrite a pre-existing Space unless the local deployment record identifies it as this project's previously created target.

GitHub and Spaces are separate repositories. Pushing GitHub alone does not deploy this Space. Re-run the two commands from the updated project to release changes. This initial Docker adaptation adds local deployment-script and metadata changes after GitHub commit `5ca403aab7c15b035cb2e88994835c40a5dbc606`; push these source changes to GitHub separately for reproducibility. No GitHub push is performed by the deployment script.

## Owner-managed DeepSeek secret

Once the Space exists, open its Settings → Variables and secrets → New secret. Name it `DEEPSEEK_API_KEY` and enter the existing project key privately. Never put it into a public Variable or upload `.env`. Until it is configured, the website will report DeepSeek as unavailable while the five local text models remain usable. Restart the Space after changing the secret if it has not restarted automatically.

The public demo reserves up to 100 upstream calls per process lifetime, two reservations per selected DeepSeek comparison to allow one parser repair. Restarts reset the counter; this is not a monthly billing limit. Request size, shared request-rate and concurrency limits are active.

## Verification

Use the direct `.hf.space` URL for browser microphone/camera access. The Space wrapper and direct URL show the same website, but browser permission behavior may differ in an iframe.

- `/healthz`: local text models loaded and required media artifact files present; it does not make a paid provider request.
- `/api/models`: five local text models plus optional DeepSeek availability.
- `scripts/check_space.py <base-url>`: public fixture transcription, two text models and both optional bonus models. No paid DeepSeek request.
- Manually test device microphone/camera permission, editing the transcript, single-channel bonus selection, and partial failures.
- Confirm one real DeepSeek comparison after the owner configures the secret.

Keep native label sets separate. The external bonus-model validation is exploratory and must not be presented as reliable hidden-emotion inference; see `MULTIMODAL.md`.

Current release status is recorded in `docs/reproducibility/space_deployment.json` after a Space is created. A prepared local bundle is not a live public deployment.

## Verified release — 2026-11-14

- Direct application: https://cecile1015-emotion-mirror-cbs4702.hf.space
- Space: https://huggingface.co/spaces/Cecile1015/emotion-mirror-cbs4702
- Space source commit: `a0bd03a5a25938e3c1f64a0c4855262817d6fdb0`.
- Actual Docker build and startup succeeded on CPU Basic. All six text models were available after the owner configured the Secret. No paid hardware was requested.
- Five local classifiers and real DeepSeek each returned a label through the public API. A separate Chrome six-model submission displayed all six result cards, retaining N/A confidence for DeepSeek.
- A public CREMA-D fixture passed cloud Whisper transcription plus expression and acoustic inference. The two bonus channels produced different native-label outputs, which is expected and not a failure of the comparison design.
- Browser Origin acceptance, foreign-Origin rejection, oversized-body rejection, hidden model/secret paths and readiness passed. Refer to `reproducibility/hf_public_validation.json`, `hf_six_model_validation.json` and `space_public_smoke.json`.
- Real user microphone/camera capture, a physical phone test and a test after shutting down the development computer remain manual checks. This release does not claim those checks were performed. Linux RAM usage was not directly measured; the prior resource profile is from macOS.
- GitHub remains at the commit the user pushed. The Docker adaptation and these release records are local changes that the user can review and push separately. Updating GitHub alone does not automatically update this Space.
