# 10 · Voice-first interactive model comparison

## Purpose and scope

Emotion Mirror provides a working local interface to the six selected emotion
classifiers. It adapts BooDarling’s burgundy wallpaper, gold mirror frame,
mirrored camera and typography; attribution and the original MIT license are
retained in `web/THIRD_PARTY_NOTICES.md` and `web/licenses/`.

The principal interaction is **record → stop and transcribe → review/edit →
compare**. The default transcription engine is local Whisper tiny.en. An optional
browser live-caption mode is also provided. Facial expressions and vocal pitch
are not used to infer emotion. Their comparison remains a separate bonus stage.

## Run on the prepared Mac

Double-click **Start Emotion Mirror.command** in the project folder. Alternatively:

```bash
.venv/bin/python scripts/run_website.py
```

The launcher resolves paths relative to its own file, discovers a system Node.js
installation or the existing Codex-bundled runtime, starts the required services,
and reuses an already running emotion API. Open **http://127.0.0.1:3000** in Chrome.
Keep the terminal open. Ctrl+C stops only processes started by the launcher.
Stop any previously running website on port 3000 before using the launcher.

- Website: `127.0.0.1:3000` (Next.js)
- Emotion models: `127.0.0.1:8000` (existing FastAPI service)
- Local transcription: `127.0.0.1:8001` (separate FastAPI sidecar)

The app is bound to loopback for local development. These addresses are not
public deployment URLs and do not work from another device’s own localhost.

## Reproduce on another machine

Use Python 3.11 and Node.js 22+; install the existing project/API dependencies and
restore the emotion-model artifacts as described in `API.md`. Then:

```bash
.venv/bin/python -m pip install -r requirements-speech.txt -c requirements-models.txt -c requirements-api.txt
.venv/bin/python scripts/prepare_speech.py
cd web
npx pnpm@11.25.0 install --frozen-lockfile
npx pnpm@11.25.0 build
cd ..
.venv/bin/python scripts/run_website.py --production
```

The speech preparation script downloads a pinned public checkpoint once and
records its SHA-256 hashes. Subsequent transcription uses local files only. The
speech dependencies are additions; the original emotion-model library versions
are constrained during installation. Check platform wheel availability on a
new OS. Large speech/model files and dependency caches must not be committed.

Server-side settings in `web/.env.local` (see `.env.example`):

```text
EMOTION_API_URL=http://127.0.0.1:8000
EMOTION_SPEECH_URL=http://127.0.0.1:8001
```

DeepSeek’s key remains in the project-root `.env` used by the Python emotion
service. It is never sent to the browser or stored in frontend environment files.

## Architecture and data flow

```text
Browser microphone → MediaRecorder → Next.js /api/transcribe
    → local FastAPI speech sidecar → Whisper tiny.en → editable English transcript
User confirms text + selected models → Next.js /api/predict
    → existing FastAPI emotion API → saved local models / optional DeepSeek
    → label cards, calibrated local probabilities, explicit per-model errors
```

The browser’s alternative SpeechRecognition mode produces live captions but may
contact the browser provider’s transcription server. This route can fail even
when the emotion API works. The default MediaRecorder/Whisper route avoids that
external speech-service dependency. Microphone permission still requires a real
user gesture and browser/OS approval.

Recordings stop at 60 seconds; the speech endpoint accepts at most 8 MB and
bounds decoded audio to 62 seconds to allow encoder padding. Silence, malformed
audio, absent models and concurrent transcription produce explicit errors.
Audio remains in memory and is not stored by the application. A public deployment
would transmit audio to the hosted speech server; it must disclose that boundary.
Transcription is not automatically forwarded to the emotion classifiers.

The camera is optional. Frames stay in the browser and are not uploaded. Tracks
are released when switched off, when the view is left or the page is hidden.
Pending permissions can be cancelled; late streams are released. In-progress
local recording/transcription is cancelled when the page is hidden. Draft text is
kept in React memory when navigating between app sections, and is not saved
across a full page reload. Browser mode preserves final and interim text for review.

## Model and evidence presentation

The Mirror defaults to TF-IDF + LR and BERT. Users can select BoW + NB, improved
LR, BiLSTM and DeepSeek (`llm`). DeepSeek selection explicitly discloses external
text submission and project API credit use. Its confidence is N/A; no fabricated
probability distribution is displayed. Failures preserve successful predictions,
and edited inputs clearly mark prior results as stale. The prediction proxy makes
no automatic paid retries.

The Ledger reads the saved research results. All six models can be compared on
the shared 300-text subset; DeepSeek is marked unevaluated on the full 2,000-text
set. Per-class F1 and confusion matrices explicitly refer to the full local-model
test set. Crypt of Errors supports search/category filtering and transferring a
saved example into the editable mirror input without automatically predicting.

Research JSON files are copied into `web/data` for a self-contained website build.
Refresh them after intentional changes to research results with
`cd web && node scripts/sync-evidence.mjs`. The website never edits those results.

## Validation and limits

Run the frontend unit checks with `cd web && pnpm test`, type checks with
`pnpm typecheck`, and production compilation with `pnpm build`.
Browser integration scenarios are in `web/tests/browser/`; `pnpm test:e2e`
requires a running local website/API and an accessible Chrome installation.
`CHROME_PATH` overrides the documented macOS Chrome binary. Speech/camera
failure scenarios in that suite use test doubles, not production fallbacks.
They do not establish real microphone quality.

Python input checks run with `EMOTION_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q`.
The original notebook 09 records the earlier API milestone; validation for this
website milestone is recorded separately in `docs/reproducibility/website_validation.json`.

An 11-second public English speech fixture from OpenAI Whisper’s tests verifies actual local transcription and a submitted
text verifies real emotion inference. Neither check estimates recognition
accuracy across accents, room noise or microphones. Each group member should
perform the manual microphone/camera checks below before presenting.

- [ ] Allow microphone access; speak English, stop, inspect and correct the transcript.
- [ ] Compare the corrected transcript with local models and optionally DeepSeek.
- [ ] Deny microphone/camera permission and confirm manual typing remains usable.
- [ ] Try silence/noise; confirm there is an understandable error or correctable transcript.
- [ ] Turn camera on/off and leave the page; confirm the hardware indicator turns off.
- [ ] Check Chrome and Safari/Edge on the intended presentation device.

Public hosting, authentication, persistent rate limits, HTTPS microphone permissions
and load testing remain deployment work. A static frontend alone cannot host the
Python emotion or speech models. Keep both server URLs server-side and deploy
those services with the required model files before claiming an operational
public website.

## References

- BooDarling source and license: https://github.com/Cecile1015/BooDarling_CBS4702
- MediaRecorder: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder
- SpeechRecognition: https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition
- getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- Faster Whisper: https://github.com/SYSTRAN/faster-whisper
- Pinned English model: https://huggingface.co/Systran/faster-whisper-tiny.en
- Next.js route handlers: https://nextjs.org/docs/app/getting-started/route-handlers


## Optional Mind-reading extension (2026-11-13)

The new bonus control is **off by default**. Select expression and/or acoustic analysis, turn on the camera if expression is selected, and make a fresh Whisper recording (20 seconds maximum). Review the transcript, then Compare. Merely enabling the camera still does not run facial inference. See `MULTIMODAL.md` for actual validation results and limitations.

Install `requirements-multimodal.txt` and run `scripts/prepare_multimodal.py` once. Stop the previous launcher with Ctrl+C before starting the updated `Start Emotion Mirror.command`; already-running Python processes retain old code.

The hosted version uses the same React interface as a static export, with same-origin Python API endpoints and a Gradio service inspector. Run `scripts/prepare_space.py` to build the allowlisted upload directory; `scripts/deploy_space.py` uploads only that directory using the existing Hugging Face CLI login. No GitHub push is performed.
