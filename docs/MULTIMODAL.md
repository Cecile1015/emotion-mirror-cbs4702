# Optional Multimodal Comparison

## Research purpose

The optional **Mind-reading / 读心术** control compares three sources of evidence from one utterance: reviewed words, visible facial expression, and vocal delivery. The name is a visual metaphor, not a claim that a model can infer a person's hidden mental state. This extension does not retrain or alter the text classifiers from the earlier experiments.

## Independent model selection

| Channel | Frozen model | Native output | Selection rationale |
|---|---|---|---|
| Expression | [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) | angry, disgust, fear, happy, neutral, sad, surprise | A FER2013-trained ViT with a documented label inventory, Apache-2.0 license, safetensors weights and CPU inference. |
| Acoustic | [superb/wav2vec2-base-superb-er](https://huggingface.co/superb/wav2vec2-base-superb-er) | neu, hap, ang, sad | A documented SUPERB/IEMOCAP English speech classifier, Apache-2.0 license, 16 kHz waveform input and CPU inference. |

These are practical selections for an exploratory demonstration, not winners of a comprehensive model search. Model revisions and SHA-256 digests are fixed in `docs/reproducibility/multimodal_models.json`. No remote model code is executed. Original model cards accompany the downloaded weights.

Expression inference uses OpenCV's frontal-face Haar detector, crops a single face with a 10% margin, applies the model's saved PIL preprocessing, and averages native softmax scores over up to three valid frames. A frame without a face is omitted; no valid frame or any detected multiple-face frame produces an explicit abstention. Detection is imperfect and does not establish identity or guarantee that every additional person is detected.

Acoustic inference decodes to mono float32 at 16 kHz and accepts 1–20 seconds (one second of encoder tolerance). Contiguous chunks of at most eight seconds bound attention memory; chunk scores are duration-weighted. Signal checks reject silence and insufficient voiced content. Median fundamental frequency is estimated using autocorrelation over 40 ms windows with 20 ms steps, a 70–400 Hz range and a 0.6 periodicity threshold; it is omitted when fewer than ten windows pass. RMS dBFS and the voiced-window fraction describe recording quality. These descriptors do not independently determine emotion.

All bonus scores are **uncalibrated**. The text model temperatures do not apply to these models. Native label inventories differ, and there is no fusion model, shared confidence scale, or cross-channel “true emotion” score.

## Fixed external smoke evaluation

The protocol was fixed before inference: CREMA-D actors 1001–1006, sentence DFA, unspecified intensity XX. The expression model was tested on all six represented acted classes (36 clips); the acoustic model on its four supported classes (24 clips). Three frames at the quarter, midpoint and three-quarter positions were extracted per video. Filenames supply the intended acting label; these are **not** the crowd-perceived majority labels or evidence of an actor's inner feelings.

| Channel | Attempted | Predictions | Correct / all attempted | Accuracy on predicted | Macro-F1 on predicted, represented classes |
|---|---:|---:|---:|---:|---:|
| Expression | 36 | 35 | 16/36 (44.4%) | 45.7% | 0.3937 |
| Acoustic | 24 | 23 | 10/24 (41.7%) | 43.5% | 0.3422 |

Expression abstained once because the detector reported multiple faces; acoustic abstained once for insufficient voiced signal. Failures count as incorrect in the all-attempted accuracy. All predictions and failures are retained in `docs/reproducibility/multimodal_validation.json`. The expression confusion matrix includes a surprise column even though CREMA-D has no surprise reference examples. Macro-F1 is explicitly restricted to represented reference classes.

These results expose substantial domain mismatch. For example, this acoustic checkpoint frequently predicts anger for happy and neutral clips, and the expression model struggles with fear and sadness. The website must therefore present the bonus as an unreliable, exploratory comparison. The sample has only six actors and one fixed sentence and cannot establish population accuracy, fairness, mobile-device performance or spontaneous-speech reliability. No threshold, prompt, weight or example was selected to improve this score after observing the results.

The source is [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), revision `1658cd342dff90010aa843eaeebd53610a08b1dc`, ODbL-1.0 database / DbCL-1.0 individual contents. Cite Cao et al. (2014), *IEEE Transactions on Affective Computing*, 5(4), 377–390, DOI 10.1109/TAFFC.2014.2336244. Raw validation fixtures remain in the ignored local directory `data/bonus_validation`; they are not bundled into the public website. The dataset authors request registration using the form linked in their repository.

## Capture, consent and failure handling

The bonus is off by default. Expression and acoustic channels can be selected independently. Enabling it selects the Whisper recording path and caps a recording at 20 seconds. A fresh recording replaces the previous transcript so that text, audio and snapshots concern one utterance. At most three snapshots are retained from that recording. Editing the reviewed transcript does not alter the recorded voice or expression.

Whisper receives audio when recording stops. Only clicking Compare sends selected media for bonus inference; an unselected channel's payload is rejected by the API. Merely turning on the mirror camera does not enable image inference. Turning the bonus off, clearing the recording or leaving the Mirror releases retained media; changing browser tabs stops capture. Media are processed in memory and not saved by the application. On a public deployment, media travel over HTTPS to the project server. Optional DeepSeek receives only the confirmed text, not images or audio.

Text and bonus requests are independent; one failed channel does not remove successful text or other-channel results. Input bounds, malformed images, silence, no face, multiple detected faces, busy service and missing model files have explicit failure paths.

## Reproduction

From the project directory:

```bash
.venv/bin/python -m pip install -r requirements-multimodal.txt
.venv/bin/python scripts/prepare_multimodal.py
.venv/bin/python scripts/validate_multimodal.py
.venv/bin/python -m pytest tests/test_multimodal.py tests/test_speech_api.py -q
```

Restart the website launcher after installing these services. Existing running Python processes will not pick up changed code automatically. The local launcher remains `Start Emotion Mirror.command`.

Public deployment uses a static export of the same React UI with FastAPI routes and an actual Gradio service inspector at `/service`. The deployment subset is built by `scripts/prepare_space.py`; it excludes `.env`, datasets, notebooks, dependency caches and backups. Hugging Face credentials are used only by the uploader; the running Space does not receive the deployment token. DeepSeek belongs in a Space Secret.


## Public deployment

The user has activated Hugging Face PRO. The complete Docker application is deployed on CPU Basic at https://cecile1015-emotion-mirror-cbs4702.hf.space ; it runs independently of the development computer. The DeepSeek key is configured as a Space Secret. Deployment revisions and public integration checks are recorded in `reproducibility/space_deployment.json` and `reproducibility/space_public_smoke.json`. Hardware was not upgraded for the detector repair.

## 2026-11-14 detector repair

The original evaluation above describes the Haar-based pipeline. The deployed pipeline now uses OpenCV YuNet 2023mar (MIT), confidence threshold 0.9, NMS threshold 0.3 and maximum detection image dimension 640. No expression or acoustic classifier weights changed. The detector artifact and license are pinned and SHA-256 checked in the build manifest.

Ambiguous multi-region snapshots are skipped rather than aborting the entire recording. If none remain, the response explains that detected regions do not establish a person count and may reflect background/reflection false alarms. Distinct detected faces are not silently replaced by the largest face. The result reports valid and excluded snapshot offsets; it does not track identity across snapshots.

Regression checks found one face in a public single-person frame, two faces in a side-by-side composite of that frame and zero in a blank image. Re-evaluation on the original 36 CREMA-D clips returned predictions for all 36, compared with 35 previously, but expression correctness remained 16/36. Acoustic results remained 10/24 with one abstention. This is reuse of the same small convenience sample after a repair, not an independent held-out accuracy study. Original results remain in `reproducibility/multimodal_validation.json`; revised results are in `reproducibility/multimodal_yunet_validation.json`.

Detector confidence concerns whether a region resembles a face. Classifier softmax scores concern expression labels and are uncalibrated. Neither establishes an inner emotional state. Lighting, pose, motion and a mismatch between FER2013 training images and webcam video can affect classification. User-device retesting is still needed; the reported single-person false alarm cannot be reconstructed without its original frame.
