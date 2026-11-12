# Emotion Mirror — Emotion Classification of Short English Texts

CBS4702 Final Project. Models are trained and compared on the
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset
(six classes: sadness, joy, love, anger, fear, surprise), and the results are
made accessible through an interactive web application.

> This README grows with the project. The sections below cover the steps completed so far.

## Repository layout

```
emotion-mirror-cbs4702/
├── data/                 cached dataset splits (downloaded automatically, not committed)
├── notebooks/            one notebook per project stage
│   ├── 01_eda.ipynb      exploratory data analysis
│   ├── 02_baselines.ipynb  bag-of-words baselines (Naive Bayes, logistic regression, TF-IDF)
│   └── 03_improved_classical.ipynb  word + character n-gram TF-IDF, class weights, ablation study
├── src/                  code shared by all notebooks and the web service
│   ├── config.py         paths, label set, colours, random seed
│   ├── data.py           dataset loading and caching
│   ├── evaluate.py       metrics, summary tables, confusion matrices, significance tests
│   └── slices.py         diagnostic subsets (unseen words, negation, 'feel' template)
├── models/               saved model files
│   └── classical/        fitted scikit-learn pipelines (*.joblib, vectoriser + classifier)
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

Saved pipelines (`models/classical/`, accept raw strings): `bow_nb`, `bow_lr`,
`tfidf_lr`, `improved_lr` and `ngram_lr` (13 MB, the typo-robust full model).

## Reproducibility

* All randomness is controlled by `SEED = 42` in `src/config.py`.
* Every model is scored by `src/evaluate.py`, which writes its metrics to
  `results/metrics/<model>__<split>.json` and its predictions to
  `results/predictions/<model>__<split>.csv`.
* The training split is used for fitting, the validation split for model
  selection, and the test split only for the final evaluation.

## Data source

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).
CARER: Contextualized Affect Representations for Emotion Recognition.
*Proceedings of EMNLP 2018*, 3687–3697.
