---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: bert-base-uncased
datasets:
- dair-ai/emotion
metrics:
- accuracy
- f1
---

# Emotion Mirror · BERT (CBS4702 Final Project)

`bert-base-uncased` fine-tuned on [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (16,000 training texts)
to classify short English texts into six emotions: sadness, joy, love, anger, fear, surprise.
Code and full evaluation: https://github.com/Cecile1015/emotion-mirror-cbs4702 (`notebooks/06_bert.ipynb`).

## Training

Full fine-tuning, `[CLS]` -> pooler -> dropout -> linear; class-weighted cross-entropy; AdamW, learning rate 5e-05
(selected from 2e-05, 3e-05, 5e-05 on validation macro-F1), 10% linear warm-up, batch size 32,
4 epochs; checkpoint selected by validation macro-F1 (epoch 3.0); seed 42.

## Results (official test split, 2,000 texts)

| Metric | Value |
|---|---|
| Accuracy | 0.930 |
| Macro-F1 | 0.896 |

Per-class F1: sadness 0.969, joy 0.944, love 0.848, anger 0.941, fear 0.893, surprise 0.783.

Probabilities are calibrated by temperature scaling, T = 1.320 (stored in `training_info.json`;
apply `softmax(logits / T)`).

## Limitations

The dataset consists of English Twitter messages built around templates such as "i feel ...". The model recognises the
emotion *expressed* in a text, not the writer's actual emotional state, and it is a course project, not a clinical or
psychological instrument.
