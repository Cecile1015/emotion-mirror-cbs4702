"""Shared evaluation utilities.

Every model in the project is scored by the same functions in this module, so
that the summary table compares like with like:

* ``evaluate``          accuracy, macro-/weighted-F1, per-class P/R/F1 and the
                        confusion matrix; results are written to
                        ``results/metrics/<model>__<split>.json`` and the
                        per-example predictions to ``results/predictions/``.
* ``summary_table``     collects the saved metrics of several models.
* ``mcnemar_test``      paired significance test on per-example correctness.
* ``paired_bootstrap``  confidence interval for the difference in a metric
                        between two models evaluated on the same examples.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import EMOTION_COLORS, FIG_DIR, LABELS, METRICS_DIR, NUM_LABELS, PRED_DIR

LABEL_IDS = list(range(NUM_LABELS))


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Accuracy, macro-F1, weighted-F1, per-class scores and confusion matrix.

    Macro-F1 averages the six per-class F1 scores with equal weight, so the
    minority classes (love, surprise) count as much as the majority classes;
    it is therefore the primary model-selection metric of the project.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)
        ),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in LABEL_IDS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
    }


def evaluate(
    model_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    split: str = "test",
    y_proba: np.ndarray | None = None,
    texts: Sequence[str] | None = None,
    notes: str | None = None,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Score one model on one split and (optionally) save the results.

    Parameters
    ----------
    model_name : short identifier used in file names and tables, e.g. "tfidf_lr".
    y_true, y_pred : gold and predicted label ids.
    split : "validation" or "test" (or a named subset such as "test_subset300").
    y_proba : optional (n, 6) array of class probabilities, saved with the predictions.
    texts : optional input texts, saved with the predictions for error analysis.
    notes : free-text description of the configuration (features, hyper-parameters).
    """
    metrics = compute_metrics(y_true, y_pred)
    record = {
        "model": model_name,
        "split": split,
        "notes": notes or "",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metrics,
    }
    if save:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_DIR / f"{model_name}__{split}.json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        if texts is not None:
            save_predictions(model_name, split, texts, y_true, y_pred, y_proba)
    if verbose:
        print(
            f"{model_name} [{split}]  accuracy={metrics['accuracy']:.4f}  "
            f"macro-F1={metrics['macro_f1']:.4f}  weighted-F1={metrics['weighted_f1']:.4f}"
        )
    return record


def save_predictions(
    model_name: str,
    split: str,
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: np.ndarray | None = None,
) -> pd.DataFrame:
    """Write one row per example: text, gold, prediction and class probabilities."""
    df = pd.DataFrame(
        {
            "idx": np.arange(len(texts)),
            "text": list(texts),
            "true": np.asarray(y_true, dtype=int),
            "pred": np.asarray(y_pred, dtype=int),
        }
    )
    df["true_label"] = df["true"].map(dict(enumerate(LABELS)))
    df["pred_label"] = df["pred"].map(dict(enumerate(LABELS)))
    df["correct"] = df["true"] == df["pred"]
    if y_proba is not None:
        proba = np.asarray(y_proba)
        for i, name in enumerate(LABELS):
            df[f"p_{name}"] = proba[:, i]
        df["confidence"] = proba.max(axis=1)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_DIR / f"{model_name}__{split}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def per_class_table(record: dict) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support as a DataFrame."""
    return pd.DataFrame(record["per_class"]).T[["precision", "recall", "f1", "support"]]


def load_record(model_name: str, split: str = "test") -> dict:
    with open(METRICS_DIR / f"{model_name}__{split}.json", encoding="utf-8") as fh:
        return json.load(fh)


def summary_table(model_names: Iterable[str] | None = None, split: str = "test") -> pd.DataFrame:
    """One row per model with the headline metrics and per-class F1.

    If ``model_names`` is None, every saved result for ``split`` is included.
    """
    if model_names is None:
        files = sorted(METRICS_DIR.glob(f"*__{split}.json"))
        model_names = [f.name.split("__")[0] for f in files]
    rows = []
    for name in model_names:
        rec = load_record(name, split)
        row = {
            "model": name,
            "accuracy": rec["accuracy"],
            "macro_f1": rec["macro_f1"],
            "weighted_f1": rec["weighted_f1"],
        }
        row.update({f"f1_{c}": rec["per_class"][c]["f1"] for c in LABELS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    title: str = "",
    normalize: bool = True,
    ax: plt.Axes | None = None,
    save_as: str | None = None,
) -> plt.Axes:
    """Heat map of the confusion matrix (rows = gold label, columns = prediction).

    With ``normalize=True`` each row sums to 1, so the diagonal equals recall
    and the off-diagonal cells show where each emotion is misclassified.
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS).astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.grid(False)  # the seaborn whitegrid style would otherwise draw lines through the cells
    ax.set_xticks(LABEL_IDS, LABELS, rotation=35, ha="right")
    ax.set_yticks(LABEL_IDS, LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in LABEL_IDS:
        for j in LABEL_IDS:
            value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > threshold else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(FIG_DIR / save_as, dpi=200, bbox_inches="tight")
    return ax


def plot_per_class_f1(table: pd.DataFrame, title: str = "", save_as: str | None = None) -> plt.Axes:
    """Grouped bar chart of per-class F1 for the models in ``summary_table``."""
    f1 = table[[f"f1_{c}" for c in LABELS]].copy()
    f1.columns = LABELS
    ax = f1.T.plot(kind="bar", figsize=(9, 4.2), width=0.8)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(FIG_DIR / save_as, dpi=200, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Significance testing for controlled comparisons
# ---------------------------------------------------------------------------
def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]) -> dict:
    """McNemar's test on the examples where exactly one of two models is correct.

    b = A correct and B wrong, c = A wrong and B correct. The exact binomial
    version is used when b + c < 25, otherwise the chi-squared version with
    continuity correction.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)), "method": "chi2"}


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)


def paired_bootstrap(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float] = _macro_f1,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap the difference metric(B) - metric(A) on resampled test sets.

    Both models are scored on the same resample in every round, so the interval
    reflects the uncertainty of the comparison rather than of each model alone.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = metric(y_true, pred_b) - metric(y_true, pred_a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_b_not_better": float(np.mean(deltas <= 0)),
    }


def emotion_palette(labels: Iterable[str] = LABELS) -> list[str]:
    """Colours in label order, for consistent figures across notebooks."""
    return [EMOTION_COLORS[name] for name in labels]
