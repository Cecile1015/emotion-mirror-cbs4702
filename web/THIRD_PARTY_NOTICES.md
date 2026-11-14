# Design and resource attribution

Emotion Mirror adapts the visual language and camera lifecycle of **Boo, Darling!**,
https://github.com/Cecile1015/BooDarling_CBS4702, revision
`85b38186fb9cb55dc1c697ebf6c0136376f75ac9`.

Copyright (c) 2026 Kevin Dai. MIT License. The complete notice is retained in
`licenses/BooDarling-MIT.txt`. Adapted elements: burgundy palette, patterned
wallpaper, gold mirror frame, mirrored video, glass reflections, and the
camera hook. The emotion comparison, recording workflow, speech sidecar,
research views and server proxy are new project-specific implementations.
No costume-reference photos, ambient music, identity judgments or mock scores
from the source application are included.

Pirata One and Almendra are served locally, downloaded from Google Fonts’
`ofl/pirataone` and `ofl/almendra` directories. Their SIL Open Font License notices
are in `licenses/`; no font request is made to Google at runtime.

Whisper tiny.en is used only for speech transcription, not emotion prediction.
The converted model is Systran/faster-whisper-tiny.en (MIT), pinned in
`docs/reproducibility/speech_model.json` in the parent project. It is downloaded
separately into the ignored `models/speech/` directory. Original model:
https://huggingface.co/openai/whisper-tiny.en .

The cover also adapts the original operator gate, dark vignette and RGB-split title; the visible name is Boo, Darlings: emotion mirror. The original MIT copyright notice remains in licenses/BooDarling-MIT.txt.

Face detection uses OpenCV YuNet 2023mar (MIT), pinned in the deployment model manifest. Its LICENSE and model README are downloaded with the detector. This replaces Haar detection; the ViT expression classifier is unchanged.
