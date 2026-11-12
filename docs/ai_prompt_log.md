# AI-Assisted Development Log

The assignment asks which AI tools were used, representative prompts, and what
was checked or modified. One entry is added per working session; the report
summarises this log.

| Date | Step | Tool | Representative prompt | What was checked or modified |
|---|---|---|---|---|
| 2026-11-11 | 1 · Setup and EDA | Claude | "Design the project structure, shared evaluation module and an exploratory analysis notebook for the dair-ai/emotion dataset, aligned with Lectures 5–9." | |
| 2026-11-12 | 3 · Improvement 1 (n-gram TF-IDF + LR) | Claude | "Build 03_improved_classical: word (1,2) + char (2,5) TF-IDF with FeatureUnion + LR, tuned with PredefinedSplit; one-variable ablations for char n-grams, class weights and bigrams with McNemar and paired bootstrap; link the discussion to FastText subwords and the EDA unseen-word rate." | |
| 2026-11-12 | 4 · Improvement 2 (GloVe + LR, BiLSTM) | Claude | "Build 04_static_embeddings (GloVe Twitter-200 vs Wiki-300: download, coverage, mean sentence vectors + LR, controlled comparison) and src/bilstm.py + 05_bilstm (GloVe-initialised BiLSTM in the style of the Lecture 7 nn.LSTM example, MPS training, early stopping on validation macro-F1, frozen vs fine-tuned embeddings), with test evaluation, significance tests and an explanation linked to Lectures 6–7." | |
