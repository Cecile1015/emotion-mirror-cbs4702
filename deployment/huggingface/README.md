---
title: Emotion Mirror CBS4702
emoji: 🪞
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: 3.12
app_file: app.py
pinned: false
---
# Emotion Mirror · CBS4702

Speak, review the transcript and compare five saved text classifiers and the selected DeepSeek few-shot prompt. An optional **Mind-reading** control independently enables facial-expression and acoustic models. It does **not** reveal hidden feelings.

Open this Space's **direct `.hf.space` URL** for phone/browser microphone and camera permissions. The BooDarling-inspired React interface is served at `/`; the Gradio service inspector is at `/service`.

- Text: BoW + NB, TF-IDF + LR, improved LR, BiLSTM, BERT, optional DeepSeek.
- Transcription: Whisper tiny.en on this server (English).
- Expression: `trpakov/vit-face-expression` (seven native labels).
- Acoustic: `superb/wav2vec2-base-superb-er` (four native labels), with descriptive pitch/energy.
- Media are processed in memory; no recordings or snapshots are saved by the application. Up to three selected snapshots and a 20-second recording can be submitted for bonus inference.
- Local models run on CPU. Cold starts and shared limits may cause waiting; text entry remains available.
- DeepSeek credentials belong in **Space Secrets**, never source files. The demo reserves at most 100 upstream calls per process lifetime (two per comparison, including possible repair); restart resets this allowance. This is not a monetary billing cap.
- Requests: at most 20/minute across this shared instance, including at most 6 media requests, two in flight, bounded upload sizes. No visitor text, media or IP addresses are stored for these counters.

## Evidence and limitations

See `docs/MULTIMODAL.md` and `docs/reproducibility/multimodal_validation.json` for the fixed external smoke evaluation. These bonus results are not comparable with the six-class text benchmark. No fusion or cross-channel confidence ranking is performed.

See `web/licenses/BooDarling-MIT.txt` and `web/THIRD_PARTY_NOTICES.md` for visual design attribution. Bonus models are Apache-2.0; Whisper is MIT. Validation fixtures derive from CREMA-D (ODbL-1.0 / DbCL-1.0) and are not redistributed here. Third-party model licenses and original cards are retained with weights.

Source is a deployment subset of the course repository. No training dataset, notebook outputs, `.env`, browser data or local research backups are uploaded.
