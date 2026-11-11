"""Data loading for the dair-ai/emotion dataset.

The three official splits are downloaded once from the Hugging Face Hub and
cached as local parquet files in ``data/`` so that every later run is offline
and byte-for-byte identical.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import DATA_DIR, HF_DATASET_ROOT, ID2LABEL, NUM_LABELS, SPLIT_FILES, SPLITS

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _validate(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Check the schema expected by every downstream step."""
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{split}: missing columns {missing}")
    if df["text"].isna().any():
        raise ValueError(f"{split}: contains missing texts")
    bad = ~df["label"].between(0, NUM_LABELS - 1)
    if bad.any():
        raise ValueError(f"{split}: {int(bad.sum())} labels outside 0..{NUM_LABELS - 1}")
    return df


def load_split(split: str, refresh: bool = False) -> pd.DataFrame:
    """Return one split with columns ``text``, ``label`` and ``label_name``.

    The local cache is used when present; ``refresh=True`` forces a new download.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{split}'. Choose from {SPLITS}.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{split}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
    else:
        df = pd.read_parquet(HF_DATASET_ROOT + SPLIT_FILES[split])
        df.to_parquet(cache, index=False)
    df = _validate(df[["text", "label"]].copy(), split)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(ID2LABEL)
    return df.reset_index(drop=True)


def load_splits(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return the train, validation and test splits as a dictionary."""
    return {split: load_split(split, refresh=refresh) for split in SPLITS}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer used for descriptive statistics.

    The dataset is already lower-cased with punctuation removed, so a simple
    regular expression is sufficient; contractions such as "don't" are kept
    as single tokens. Model-specific tokenizers (CountVectorizer, WordPiece)
    are defined in the modelling notebooks.
    """
    return _TOKEN_RE.findall(text.lower())
