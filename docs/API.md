# Step 8 · From Saved Models to an HTTP API

## 1. Purpose and model selection

Steps 1–7 established the data splits, model comparisons and error analysis.
This step exposes **existing** models to the future website. It does not tune
models, fit temperatures or alter test data. Explicit `llm` requests call the
DeepSeek configuration selected in notebook 07.

| ID | Role in the interactive comparison | Test macro-F1 (2,000 texts) |
|---|---|---:|
| `bow_nb` | Generative count baseline; clear starting point | 0.6949 |
| `tfidf_lr` | Strong, fast sparse baseline; default comparator | 0.8320 |
| `improved_lr` | Sublinear TF and class weighting | 0.8429 |
| `bilstm` | Word order with fine-tuned GloVe embeddings | 0.8939 |
| `bert` | Contextual encoder selected by validation macro-F1 | 0.8962 |
| `llm` | Selected DeepSeek few-shot model, 18 demonstrations | N/A — evaluated on 300 texts |

The deployed **set** illustrates distinct experiments already completed; no new
hyperparameter selection is made here. BERT and BiLSTM were not significantly
different in Step 7. BoW + LR largely overlaps the sparse LR comparisons, while
averaged GloVe has a large external cache and weaker results. They remain in the
offline ledger. DeepSeek is available as an optional paid comparator, with
its results compared only on the common 300-text subset. `/models` returns
`subset300_metrics` for all six models; use those fields for a common-population
score comparison. DeepSeek has no 2,000-text test score.

## 2. Preserve the analysis

`scripts/freeze_artifacts.py` creates `local_backups/<UTC timestamp>/` with:

- `artifacts.tar.gz`: code, notebooks, all saved model checkpoints, evaluation
  files, handcrafted data and cached dataset splits;
- `manifest.json`: SHA-256 of every included file and the verified archive;
- `environment.txt`: installed package versions, without environment values.

The archive is verified by reading each archived file and hashing its bytes.
It excludes `.env`, `.git`, virtual environments and GloVe download caches.
Do not commit `local_backups/`. Copy the archive to independent storage yourself.
The pre-API manifest is historical: later edits to README/docs intentionally
make those files differ. Use `--verify` to identify changes, not overwrite them.

```bash
python scripts/freeze_artifacts.py
python scripts/freeze_artifacts.py --verify local_backups/<timestamp>/manifest.json
```

## 3. Install and start locally

Use Python 3.11 and the existing `.venv` on this Mac:

```bash
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

On a fresh machine, install `requirements.txt` first, then
`requirements-test.txt`. The tested macOS environment inventory is recorded in
`docs/reproducibility/`; it is a reproducibility record, not a claim that every
platform-specific package works on Linux. Check the cloud environment separately.

Restore local models from the snapshot, or prepare BERT explicitly before
serving: `hf download Cecile1015/emotion-mirror-bert --local-dir models/bert`.
For exact reproduction use the archived weights and verify their hashes. A
download of the Hub's latest revision alone does not guarantee the same bytes.
Do not expose arbitrary user-supplied joblib or torch checkpoints to the loader.

Required serving artifacts: three classical pipelines (`bow_nb`, `tfidf_lr`,
`improved_lr`), `models/bilstm/bilstm.pt`, BERT weights/config/tokenizer and the
three saved temperature JSON files in `results/metrics/`. BiLSTM's checkpoint
already contains its embeddings: it does not need the GloVe cache at runtime.

Visit `http://127.0.0.1:8000/docs`, expand POST `/predict`, choose **Try it out**,
enter a request and choose **Execute**. Swagger UI may need network to load its
documentation assets; local-model inference works offline after setup. DeepSeek
requires a network connection and a funded server-side API account.

Optional API settings are read from the shell environment. The DeepSeek adapter
also loads credentials and its base URL from the ignored project `.env` file
without overriding existing environment values. Put the key in `DEEPSEEK_API_KEY`;
never put it in browser code or a `NEXT_PUBLIC_` variable. The serving model name
is fixed by `results/llm/selection.json`, not by `DEEPSEEK_MODEL`.

| Variable | Default | Meaning |
|---|---|---|
| `EMOTION_API_MODELS` | all six IDs above | Comma-separated enabled models |
| `EMOTION_API_DEVICE` | `cpu` | `cpu`, or explicitly tested `mps`/`cuda` |
| `EMOTION_API_ORIGINS` | localhost:3000 and 127.0.0.1:3000 (HTTP) | Comma-separated exact frontend origins |

An intentionally reduced service can use `EMOTION_API_MODELS=tfidf_lr,bert`.
If BERT is not installed, use `bow_nb,tfidf_lr,improved_lr,bilstm` and request
available IDs explicitly. The recommended default request still compares
TF-IDF and BERT, so missing default models appear as explicit errors.

## 4. Request and response

```json
{"text":"i do not feel happy today","models":["tfidf_lr","bert","llm"]}
```

The response contains `text`, fixed-order `labels`, `results`, `errors` and
`all_agree`. Each result contains its `model_id`, `label`, `confidence`, six
`probabilities`, calibration `temperature`, `calibrated`, `elapsed_ms`, `device`,
`truncated`, `input_tokens`, `token_limit` and `warnings`.

For `llm`, `confidence`, `probabilities` and calibration `temperature` are
`null`. Render these as **N/A**; do not draw a probability bar or substitute a
one-hot vector. Additional fields identify `provider`, `served_model`,
`prompt_id`, `n_shots`, `parsing_status`, `usage` (including token counts and
request count), and `decoding_temperature`. Decoding temperature 0 is distinct
from a fitted calibration temperature. Local predictions keep their original
six-class probability distributions. Omitting `models` still selects only
TF-IDF + LR and BERT; startup and registry requests never make paid calls.

`all_agree` is null for one result or any partial failure, so a missing model
cannot be mistaken for agreement. If some requested models fail, HTTP 200
returns successful results **and** errors; the UI must display both. If none
succeeds, HTTP 503 returns the same response shape. Request validation is 422.
`GET /health` is 200 when all configured **local** models are ready. Optional
DeepSeek credentials are reported with `remote_configured`;
`remote_connectivity_checked=false` makes clear that this is not a credit or
connection test. A remote-only service requires a configured client for readiness.
`/models` distinguishes disabled models from enabled-but-unavailable models;
DeepSeek availability means credentials are configured, not a guarantee that
the next request succeeds.

## 5. Preprocessing and calibration

The service trims outer whitespace and otherwise preserves text. Each saved
pipeline/tokenizer retains its original preprocessing. The input contract is
English text, 1–2,000 characters, with at least one Latin letter. This lightweight
check is not a language detector; multilingual inputs are not validated by the
project's English evaluation.

The existing `load_predictor` returns **raw** probabilities (BERT is explicitly
loaded with temperature 1). The API applies `softmax(log(p)/T)` once, using only
the validation-fitted temperature files. This is mathematically equivalent to
logit temperature scaling up to floating-point precision; extremely small raw
probabilities can underflow and cannot be reconstructed exactly. Positive
temperatures preserve the winning label. NB uses T=1 and is marked uncalibrated.
Missing calibration files make the affected model unavailable instead of
silently replacing T with 1.

BiLSTM retains 66 word tokens and BERT 128 WordPiece tokens including special
tokens. The API reports actual model-specific counts and truncation; classical
models use the full accepted input. This paragraph describes local models;
DeepSeek token usage includes its instruction and 18 demonstrations. Long-input results may therefore reflect
different visible spans. Confidence is neither emotion intensity nor evidence
of a user's true psychological state.

## 6. Validation and limitations

```bash
python -m pytest -q
EMOTION_RUN_MODEL_TESTS=1 python -m pytest -q
```

The fast tests inject a small predictor to check validation, partial failures,
health, CORS and a known temperature-scaling example. The opt-in integration
test loads all five real models on CPU, compares API probabilities and labels
against independently loaded predictors, checks both neural truncation limits,
confirms loading is idempotent and sends concurrent requests. Mock responses
exist only in tests, never as a production fallback. The DeepSeek test module
checks absent credentials, timeouts, invalid output, probability nullability
and partial-result preservation. All automated tests run without paid calls.

The submitted notebook additionally records one opt-in live provider check in
`docs/reproducibility/deepseek_live_validation.json`. To repeat it, set
`EMOTION_RUN_LIVE_LLM=1` before starting the notebook kernel. It consumes API
credit. Default notebook reruns skip this live check.

Run `notebooks/09_model_api.ipynb` in the existing project kernel for a guided,
executed walkthrough. Its TestClient sends in-process HTTP requests; starting
Uvicorn separately is still needed for the future website.

The current API is a local-development service. It serializes local inference to
protect existing model state and limit simultaneous neural work. Remote waiting
is outside this local lock. Measured
request timings are demonstration timings, not a controlled model benchmark.
Public authentication, rate limits, proxy body-size limits, timeout/load tests,
HTTPS hosting and face/audio routes belong to the deployment/bonus stages.

## 7. DeepSeek serving details

The adapter reuses `src/llm.py`, `results/llm/selection.json` and
`results/llm/prompt_llm.json`: `deepseek-flash`, `fs_random_k3`, 18 training
examples, decoding temperature 0, 32 output tokens, JSON output and disabled
thinking. It uses the original parser and permits one corrective follow-up.
Log-probabilities are not requested because no validated six-class probability
distribution is available for this configuration. The SDK has no automatic
retries and uses a 15-second network-operation timeout, not an end-to-end SLA.

A DeepSeek failure returns a per-model error while preserving successful local
results. The comparison HTTP response waits for remote completion or failure;
this endpoint does not stream local results first. Display a loading state and
the returned errors. Do not silently retry paid requests in the browser.

Only text and the saved prompt are sent to DeepSeek. The adapter does not use
the research JSONL cache for visitor inputs. The frontend should disclose the
remote submission when `llm` is selected. Actual provider usage is returned;
cost is not estimated using stale experimental pricing. Public rate limiting
and server-side access controls remain deployment tasks.

After updating these files, stop an older Uvicorn process with Ctrl+C in its
terminal and start it again. An already running process without reload keeps
the old imported code.

## 8. Next stage

Build the Next.js mirror interface and call this contract from its server
routes. Static evaluation pages should read existing `results/app/*.json`.
Include `llm` in the model selector and compare its label beside local models,
with N/A for its confidence and the shared 300-text evaluation table.
Preserve manual text input alongside speech transcription. Do not send raw
images or audio to `/predict`: face and acoustic models need separate inputs,
labels, evaluation and endpoints.

References: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/),
[DeepSeek chat completions](https://api-docs.deepseek.com/api/create-chat-completion/),
[DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/).
