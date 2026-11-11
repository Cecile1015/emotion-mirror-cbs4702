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
│   └── 01_eda.ipynb      exploratory data analysis
├── src/                  code shared by all notebooks and the web service
│   ├── config.py         paths, label set, colours, random seed
│   ├── data.py           dataset loading and caching
│   └── evaluate.py       metrics, summary tables, confusion matrices, significance tests
├── models/               saved model files
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

The repository must be public for Colab to clone it without credentials.
Files written on Colab (figures, metrics, models) are lost when the runtime
disconnects; download them or commit them back to the repository.

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
