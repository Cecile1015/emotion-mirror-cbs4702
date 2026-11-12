"""Behavioural tests in the style of CheckList (Ribeiro et al., 2020).

The test suite (``data/handcrafted/behavioural_tests.csv``) contains short
hand-written sentences, grouped by linguistic capability:

* negation   - an emotion word under negation ("i don t feel happy today") next to
               its affirmative counterpart, and negation words that do not reverse
               the emotion ("i can t stop smiling").
* sarcasm    - positive words used to express a negative attitude.
* intensity  - the same sentence with a downtoner, without modifier and with an intensifier.
* mixed      - two emotions in one sentence.

Three test types are scored:

* MFT (minimum functionality test): the predicted label is one of the expected labels.
* INV (invariance): within an intensity group the predicted label does not change.
* DIR (directional expectation): a perturbation moves the probability of a class in a
  known direction - negation lowers the probability of the affirmed emotion by at least
  ``DIR_MARGIN``; an intensifier does not give the emotion a lower probability than a
  downtoner. Mixed sentences are also scored by whether both emotions are in the top two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, LABEL2ID, LABELS

SUITE_PATH = DATA_DIR / "handcrafted" / "behavioural_tests.csv"
CAPABILITIES = ["negation", "sarcasm", "intensity", "mixed"]
DIR_MARGIN = 0.10


def load_suite(path=SUITE_PATH) -> pd.DataFrame:
    """Read the test suite; ``expected`` becomes a list of acceptable labels."""
    suite = pd.read_csv(path, dtype=str, keep_default_na=False)
    suite["expected"] = suite["expected"].str.split("|")
    bad = {lab for labels in suite["expected"] for lab in labels} - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels in the test suite: {bad}")
    return suite


def score_items(suite: pd.DataFrame, pred: np.ndarray, proba: np.ndarray | None, model: str) -> pd.DataFrame:
    """One row per test sentence: prediction, probabilities and the MFT result."""
    out = suite[["test_id", "capability", "group", "variant", "text"]].copy()
    out.insert(0, "model", model)
    out["expected"] = suite["expected"].str.join("|")
    pred = np.asarray(pred)
    out["pred_label"] = [LABELS[i] if i >= 0 else "invalid" for i in pred]
    out["mft_pass"] = [p in exp for p, exp in zip(out["pred_label"], suite["expected"])]
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        for i, name in enumerate(LABELS):
            out[f"p_{name}"] = proba[:, i]
        out["confidence"] = proba.max(axis=1)
        top2 = np.argsort(-proba, axis=1)[:, :2]
        out["top2"] = [f"{LABELS[a]}|{LABELS[b]}" for a, b in top2]
    return out


def group_tests(items: pd.DataFrame) -> pd.DataFrame:
    """INV and DIR results per test group (and top-2 for mixed sentences) for one model."""
    rows = []
    has_proba = "p_sadness" in items.columns
    model = items["model"].iloc[0]
    for group, g in items.groupby("group", sort=False):
        capability = g["capability"].iloc[0]
        if capability == "negation" and {"affirmative", "negated"} <= set(g["variant"]):
            aff, neg = g[g.variant == "affirmative"].iloc[0], g[g.variant == "negated"].iloc[0]
            label = aff["expected"].split("|")[0]
            passed = (aff[f"p_{label}"] - neg[f"p_{label}"] >= DIR_MARGIN) if has_proba else np.nan
            rows.append({"model": model, "capability": capability, "test": "DIR: negation lowers P(affirmed emotion)",
                         "group": group, "pass": passed})
        elif capability == "intensity":
            rows.append({"model": model, "capability": capability, "test": "INV: same label across intensities",
                         "group": group, "pass": g["pred_label"].nunique() == 1})
            label = g["expected"].iloc[0].split("|")[0]
            low = g[g.variant == "downtoned"].iloc[0]
            high = g[g.variant == "intensified"].iloc[0]
            passed = (high[f"p_{label}"] >= low[f"p_{label}"]) if has_proba else np.nan
            rows.append({"model": model, "capability": capability, "test": "DIR: intensifier does not lower P(emotion)",
                         "group": group, "pass": passed})
        elif capability == "mixed":
            for _, row in g.iterrows():
                expected = set(row["expected"].split("|"))
                passed = expected <= set(row["top2"].split("|")) if has_proba else np.nan
                rows.append({"model": model, "capability": capability, "test": "both emotions in the top two",
                             "group": row["test_id"], "pass": passed})
    return pd.DataFrame(rows)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate k / n (better than the normal interval for small n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def expected_ids(suite: pd.DataFrame) -> list[int]:
    """The first expected label of every sentence as an id (for confusion-style tables)."""
    return [LABEL2ID[e[0]] for e in suite["expected"]]


__all__ = ["CAPABILITIES", "DIR_MARGIN", "SUITE_PATH", "expected_ids", "group_tests", "load_suite",
           "score_items", "wilson_interval"]
