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
│   └── 05_bilstm.ipynb              bidirectional LSTM on GloVe, frozen vs fine-tuned embeddings
├── src/                  code shared by all notebooks and the web service
│   ├── config.py         paths, label set, colours, random seed
│   ├── data.py           dataset loading and caching
│   ├── evaluate.py       metrics, summary tables, confusion matrices, significance tests
│   ├── slices.py         diagnostic subsets (unseen words, negation, 'feel' template)
│   ├── embeddings.py     GloVe download/cache and the averaged-vector transformer
│   └── bilstm.py         BiLSTM vocabulary, dataset, model, training loop, checkpoints
├── models/               saved model files
│   ├── classical/        fitted scikit-learn pipelines (*.joblib, vectoriser + classifier)
│   └── bilstm/           BiLSTM checkpoints (only bilstm.pt, the selected model, is committed)
├── results/
│   ├── figures/          figures used in the report and the web app
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

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of
Modern Neural Networks. *Proceedings of ICML 2017*, 1321–1330.

## Data source

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).
CARER: Contextualized Affect Representations for Emotion Recognition.
*Proceedings of EMNLP 2018*, 3687–3697.
