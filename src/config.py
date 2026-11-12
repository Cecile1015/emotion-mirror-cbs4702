"""Project-wide configuration: paths, label inventory, colours and random seed.

Every notebook and the web service import from this module so that all models
share the same label order, data splits and output locations.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Dataset (dair-ai/emotion, "split" configuration)
# ---------------------------------------------------------------------------
HF_DATASET_ROOT = "hf://datasets/dair-ai/emotion/"
SPLIT_FILES = {
    "train": "split/train-00000-of-00001.parquet",
    "validation": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}
SPLITS = tuple(SPLIT_FILES)

# Label order follows the dataset card: the integer id is the list index.
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
NUM_LABELS = len(LABELS)

# One fixed colour per emotion, reused in every figure and in the web app.
EMOTION_COLORS = {
    "sadness": "#3B6BA5",
    "joy": "#E8A33D",
    "love": "#D45087",
    "anger": "#C0392B",
    "fear": "#6A4C93",
    "surprise": "#2A9D8F",
}

# ---------------------------------------------------------------------------
# Fine-tuned BERT (Step 5)
# ---------------------------------------------------------------------------
# The model is fine-tuned on Google Colab and saved to models/bert/. Its weights
# (about 440 MB) exceed the GitHub file limit, so they are backed up on the
# Hugging Face Hub; src/bert_infer.py downloads them when the local copy is missing.
# Replace the user name with your own Hugging Face account, or set the
# EMOTION_BERT_REPO environment variable.
BERT_BASE_MODEL = "bert-base-uncased"
BERT_DIR = MODELS_DIR / "bert"
BERT_HUB_REPO = os.environ.get("EMOTION_BERT_REPO", "Cecile1015/emotion-mirror-bert")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix the random state of Python, NumPy and (if installed) PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for path in (DATA_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR, PRED_DIR):
        path.mkdir(parents=True, exist_ok=True)
