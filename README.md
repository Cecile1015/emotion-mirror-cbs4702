# Emotion Mirror — Emotion Classification of Short English Texts

CBS4702 Final Project. Models are trained and compared on the
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset
(six classes: sadness, joy, love, anger, fear, surprise), and the results are
made accessible through an interactive web application.

> This README grows with the project. The sections below cover the steps completed so far.

## Repository layout

```
emotion-mirror-cbs4702/
├── data/                 cached dataset splits and GloVe vectors (downloaded automatically, not committed)
├── notebooks/            one notebook per project stage
│   ├── 01_eda.ipynb      exploratory data analysis
│   ├── 02_baselines.ipynb  bag-of-words baselines (Naive Bayes, logistic regression, TF-IDF)
│   ├── 03_improved_classical.ipynb  word + character n-gram TF-IDF, class weights, ablation study
│   ├── 04_static_embeddings.ipynb   GloVe coverage, averaged word vectors + LR, Twitter vs Wikipedia GloVe
│   ├── 05_bilstm.ipynb              bidirectional LSTM on GloVe, frozen vs fine-tuned embeddings
│   ├── 06_bert.ipynb                fine-tuned bert-base-uncased (Colab GPU), frozen-BERT control, attention
│   └── 07_llm_deepseek.ipynb        zero-/few-shot DeepSeek LLM vs all trained models on a 300-text test subset
├── src/                  code shared by all notebooks and the web service
│   ├── config.py         paths, label set, colours, random seed
│   ├── data.py           dataset loading and caching
│   ├── evaluate.py       metrics, summary tables, confusion matrices, significance tests
│   ├── slices.py         diagnostic subsets (unseen words, negation, 'feel' template)
│   ├── embeddings.py     GloVe download/cache and the averaged-vector transformer
│   ├── bilstm.py         BiLSTM vocabulary, dataset, model, training loop, checkpoints
│   ├── bert_infer.py     BERT loading (local or Hugging Face Hub), prediction, attention weights
│   └── llm.py            DeepSeek prompts, output validation, cached API calls, token cost
├── models/               saved model files
│   ├── classical/        fitted scikit-learn pipelines (*.joblib, vectoriser + classifier)
│   ├── bilstm/           BiLSTM checkpoints (only bilstm.pt, the selected model, is committed)
│   └── bert/             fine-tuned BERT (weights on the Hugging Face Hub; model card and training info committed)
├── results/
│   ├── figures/          figures used in the report and the web app
│   ├── llm/              evaluation subsets, selected prompts and the cache of every DeepSeek response
│   ├── metrics/          one JSON file per model and split
│   └── predictions/      one CSV file per model and split (used for error analysis)
├── docs/                 AI-assistance log and screenshots
└── report/               written report
```

## Setup (macOS)

```bash
cd emotion-mirror-cbs4702
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name emotion-mirror --display-name "Python (emotion-mirror)"
```

Open the notebooks in VS Code or JupyterLab (`jupyter lab`) and select the
**Python (emotion-mirror)** kernel. The first run of any notebook downloads
the dataset into `data/`; later runs work offline.

`04_static_embeddings.ipynb` downloads two pre-trained GloVe files on its first
run (about 1.2 GB in total) into `data/embeddings/` and converts them to a
NumPy cache (about 0.8 GB); this takes a few minutes once. The GloVe-based
models (`glove_*_lr`, `bilstm`) load the vectors from this cache.

`05_bilstm.ipynb` trains on the Apple-silicon GPU (`mps`) when it is available
(CUDA on Colab, otherwise the CPU). It trains six models (two settings × three
seeds); with `RETRAIN = False`, a re-run loads the saved checkpoints instead.

### Fine-tuned BERT (`06_bert.ipynb`)

BERT is fine-tuned on a Google Colab T4 GPU (about 30–40 minutes for the whole
notebook). Before opening it in Colab, push the current `src/` to GitHub, because
Colab clones the repository. The notebook mounts Google Drive to keep finished
training runs (a disconnected runtime resumes from them), uploads the model to the
Hugging Face Hub (§14) and downloads `bert_results.zip` with all metrics,
predictions and figures (§15). The upload needs a Hugging Face access token of
type *Write* (huggingface.co → Settings → Access Tokens), stored in Colab under
*Secrets* as `HF_TOKEN` with notebook access switched on.

On the Mac, unzip the results in the project root and fetch the weights (about 440 MB):

```bash
unzip -o ~/Downloads/bert_results.zip
hf download Cecile1015/emotion-mirror-bert --local-dir models/bert   # or let src/bert_infer.py download them
python -m src.bert_infer "i feel so lonely without you" "i was shocked by the news"
```

`src/bert_infer.py` runs on the CPU or the Apple-silicon GPU and returns
temperature-calibrated probabilities and the self-attention weights of every layer
and head (`BertEmotionClassifier.predict` / `.explain`). The Hub repository name is
`BERT_HUB_REPO` in `src/config.py`.

### DeepSeek LLM (`07_llm_deepseek.ipynb`)

The notebook calls the DeepSeek API, which needs an API key (platform.deepseek.com → API keys;
the account needs a small prepaid balance). Store the key in a file `.env` in the project root;
`.env` is listed in `.gitignore`, so the key is never committed:

```bash
cp .env.example .env      # then replace sk-your-key-here with your key
```

All runs together make about 3,300 short requests (seven prompts on 300 validation texts, a repeat
of the two selected prompts, and the two final prompts on 300 test texts); at DeepSeek's list prices
this costs well under one US dollar and takes about 10–15 minutes with 8 parallel requests. Check
the current prices on the DeepSeek pricing page and update `PRICE_USD_PER_MILLION` in `src/llm.py`
if they have changed. Every response is stored in `results/llm/cache/`; commit this folder, so that
the notebook can be re-run by anyone, without a key and at no cost, with identical results.

## Running on Google Colab

Every notebook starts with a setup cell that detects Colab, clones this
repository into `/content/emotion-mirror-cbs4702` and adds it to the Python
path, so the same notebook runs unchanged in both environments. Open a
notebook directly from GitHub:

| Notebook | Colab |
|---|---|
| `01_eda.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/01_eda.ipynb) |
| `02_baselines.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/02_baselines.ipynb) |
| `03_improved_classical.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/03_improved_classical.ipynb) |
| `04_static_embeddings.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/04_static_embeddings.ipynb) |
| `05_bilstm.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/05_bilstm.ipynb) |
| `06_bert.ipynb` (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/06_bert.ipynb) |
| `07_llm_deepseek.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cecile1015/emotion-mirror-cbs4702/blob/main/notebooks/07_llm_deepseek.ipynb) |

The repository must be public for Colab to clone it without credentials.
Files written on Colab (figures, metrics, models) are lost when the runtime
disconnects; download them or commit them back to the repository.

## Results so far

Test split (2,000 texts); hyper-parameters selected by validation macro-F1.

| Model | Features | Classifier | Accuracy | Macro-F1 |
|---|---|---|---|---|
| `bow_nb` | word counts | Multinomial Naive Bayes (alpha = 0.3) | 0.802 | 0.695 |
| `bow_lr` | word counts | Softmax logistic regression (C = 10) | 0.883 | 0.831 |
| `tfidf_lr` | TF-IDF | Softmax logistic regression (C = 30) | 0.882 | 0.832 |
| `ngram_cnb` | TF-IDF word 1–2 + char 2–5 grams | Complement Naive Bayes | 0.872 | 0.824 |
| `ngram_lr` | TF-IDF word 1–2 + char 2–5 grams | Softmax LR, class weights (C = 30) | 0.880 | 0.828 |
| `improved_lr` | TF-IDF unigrams (sublinear tf) | Softmax LR, class weights (C = 3) | 0.885 | 0.843 |
| `glove_twitter_lr` | mean of GloVe Twitter vectors (200-d) | Softmax LR (C = 0.1) | 0.655 | 0.567 |
| `glove_wiki_lr` | mean of GloVe Wikipedia + Gigaword vectors (300-d) | Softmax LR (C = 3) | 0.674 | 0.601 |
| `bilstm_frozen` | GloVe 300-d, frozen | BiLSTM (H = 128) + linear, class weights | 0.922 | 0.888 |
| `bilstm_finetuned` | GloVe 300-d, fine-tuned | BiLSTM (H = 128) + linear, class weights | 0.932 | 0.894 |
| `bert_frozen` | BERT-base, frozen, pooled `[CLS]` | linear layer only, class weights | 0.483 | 0.418 |
| `bert_frozen_mean` | BERT-base, frozen, mean of token vectors | linear layer only, class weights | 0.608 | 0.530 |
| `bert_finetuned` | BERT-base, fine-tuned (lr = 5e-5) | `[CLS]` → pooler → linear, class weights | 0.930 | 0.896 |

Logistic regression is significantly better than Naive Bayes on the same features
(McNemar p < 1e-19); counts vs. TF-IDF makes no significant difference in accuracy
(p = 0.81), but TF-IDF + LR gives better-calibrated probabilities.
See `notebooks/02_baselines.ipynb` §7 for the discussion.

Improvement 1 (`03_improved_classical.ipynb`) tunes a word + character n-gram
model with `PredefinedSplit` and runs a 2³ ablation (word bigrams × character
n-grams × class weights) on the validation split. Class weights are the only
component that helps consistently; the selected model (`improved_lr`) is the
unigram model with class weights. No representation change is significant on
the test set, but character n-grams halve the loss of macro-F1 when the test
texts contain spelling mistakes (stress test, §7.3).

Improvement 2 replaces sparse features with pre-trained GloVe vectors.
Averaged into one vector per text (`04_static_embeddings.ipynb`), they are far
weaker than the sparse models (test macro-F1 0.601 vs 0.843 for `improved_lr`; McNemar p < 1e-67):
distributional vectors place opposite emotions close together, and the average
dilutes the emotion words in long texts. The same vectors read by a
bidirectional LSTM (`05_bilstm.ipynb`) give the best model so far, significantly
better than `improved_lr` (Δ macro-F1 = +0.051 [+0.036, +0.068], McNemar
p ≈ 2e-13). Fine-tuning the embeddings adds +0.006 macro-F1, which is not
significant (p = 0.065), but gives more stable results across seeds; shuffling
the words of the test texts costs the BiLSTM 0.09–0.15 macro-F1, so it relies
on word order. Numbers are from the seed-42 models; see `05_bilstm.ipynb` for
the variance across three seeds.

Saved models, which accept raw strings:

* `models/classical/`: `bow_nb`, `bow_lr`, `tfidf_lr`, `improved_lr`, `ngram_lr`
  (13 MB, the typo-robust full model), `glove_twitter_lr`, `glove_wiki_lr` and
  `glove_lr` (the selected variant; these need the GloVe cache in `data/embeddings/`).
* `models/bilstm/bilstm.pt` (33 MB): the selected BiLSTM; load it with
  `src.bilstm.load_checkpoint` and apply the temperature in
  `results/metrics/temperature_bilstm.json` to its probabilities.
* `models/bert/` (Hugging Face Hub): the fine-tuned BERT; load it with
  `src.bert_infer.BertEmotionClassifier`, which applies the temperature stored in
  `models/bert/training_info.json`.

Improvement 3 (`06_bert.ipynb`) fine-tunes `bert-base-uncased` with a linear
classifier on the `[CLS]` vector; the learning rate (5e-5) and the
checkpoint (epoch 3) were selected by validation macro-F1. BERT is significantly
better than `improved_lr` (Δ macro-F1 = +0.053 [+0.035, +0.072], McNemar
p ≈ 3e-11) but statistically tied with the BiLSTM (Δ = +0.002 [−0.013, +0.017],
p = 0.85; seed means 0.894 vs 0.891). Fine-tuning the encoder is essential: a
frozen BERT whose linear layer alone is trained reaches only 0.418 macro-F1
(Δ = +0.478 for full fine-tuning). BERT gives the best-calibrated probabilities
of all models (test log-loss 0.149 after temperature scaling) and is published
on the Hugging Face Hub as
[`Cecile1015/emotion-mirror-bert`](https://huggingface.co/Cecile1015/emotion-mirror-bert).

Step 6 (`07_llm_deepseek.ipynb`) prompts a decoder-only LLM (DeepSeek, temperature 0,
JSON output) without any training: zero-shot with an instruction, and few-shot with labelled
training texts as demonstrations. Prompts are developed on a stratified 300-text validation
subset; all trained models are re-scored on the same stratified 300-text test subset and
compared with McNemar's test and a paired bootstrap. On the 300 test texts, DeepSeek reaches
macro-F1 0.662 zero-shot and 0.713 with 18 demonstrations (Δ = +0.051, Holm p = 0.003), far below
fine-tuned BERT on the same texts (0.925; Δ = −0.212 [−0.284, −0.145], p ≈ 8e-11) and also below
TF-IDF + LR (0.836). The prompt that describes the dataset's labelling convention (the named feeling
word decides) helps far more than everyday definitions of the emotions, and the LLM loses most on
negated texts, where that convention and the literal meaning diverge. At temperature 0 the API
returns one-hot log-probabilities, so the LLM gives no usable confidence. All API runs together cost
about US$0.24.

## Reproducibility

* All randomness is controlled by `SEED = 42` in `src/config.py`.
* Every model is scored by `src/evaluate.py`, which writes its metrics to
  `results/metrics/<model>__<split>.json` and its predictions to
  `results/predictions/<model>__<split>.csv`.
* The training split is used for fitting, the validation split for model
  selection, and the test split only for the final evaluation.

## References

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for
Word Representation. *Proceedings of EMNLP 2014*, 1532–1543.

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
*Neural Computation*, 9(8), 1735–1780.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding. *Proceedings of
NAACL-HLT 2019*, 4171–4186.

Brown, T. B., et al. (2020). Language Models are Few-Shot Learners.
*Advances in Neural Information Processing Systems 33*, 1877–1901.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of
Modern Neural Networks. *Proceedings of ICML 2017*, 1321–1330.

## Data source

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).
CARER: Contextualized Affect Representations for Emotion Recognition.
*Proceedings of EMNLP 2018*, 3687–3697.
