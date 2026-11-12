"""Error coding and word-level explanations for 08_error_analysis.

* ``ERROR_CATEGORIES``  the coding scheme: one row per category with its linguistic
                        level and the rule that assigns it.
* ``code_error``        assigns one primary category to a misclassified text by
                        applying the rules in a fixed order of precedence.
* ``lr_contributions``  for a linear model on TF-IDF features (Lecture 5-6): the
                        contribution x_j (w_pred,j - w_gold,j) of every word to the
                        decision between the predicted and the gold class.
* ``occlusion``         for any model: the drop in the probability of a class when
                        one word is deleted (a model-agnostic perturbation explanation).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import LABEL2ID, LABELS
from .keywords import FEEL_VERBS, KeywordLexicon, feel_complement

# Negation cues and irrealis markers (modality, conditionals, wishes) whose scope can
# reverse or suspend the emotion named by the keyword.
SCOPE_MARKERS = frozenset(
    """not no never nothing nobody cannot cant dont didnt doesnt isnt wasnt wont wouldnt couldnt shouldnt
    havent hasnt nolonger t if wish hope hoping would should could might want wanted wanting trying try""".split()
)
SCOPE_WINDOW = 5  # tokens before the keyword (or the feel verb) that are searched for a marker

ERROR_CATEGORIES = pd.DataFrame(
    [
        ("A1", "Annotation", "Split keyword",
         "The text contains a split keyword: its training texts are divided between two labels, "
         "and the context does not predict which (§4.2)."),
        ("A2", "Annotation", "Gold label contradicts a reliable keyword",
         "The keyword has one dominant training label, the prediction follows it, the gold label does not, "
         "and no negation, modality or competing cue explains the difference."),
        ("B", "Pragmatics", "Implicit emotion",
         "The text contains no emotion keyword: the emotion must be inferred from the event or situation "
         "described (its appraisal), which requires world knowledge."),
        ("C", "Discourse", "Competing emotion cues",
         "Reliable keywords of two or more classes occur in the text and the model chose a different one "
         "than the annotator (mixed emotions, contrast, emotions of other people)."),
        ("D", "Syntax-semantics", "Negation or modality scope",
         f"A negation cue or an irrealis marker (if, hope, would, should, want ...) occurs within "
         f"{SCOPE_WINDOW} tokens before the keyword or the feel verb."),
        ("E", "Lexicon", "Rare or unseen keyword",
         "The word after feel is not in the keyword lexicon (fewer than 10 training occurrences), "
         "so the model must generalise from its form or its context."),
        ("F", "Other", "Other",
         "A reliable keyword, no competing cue and no scope marker: the model is simply wrong."),
    ],
    columns=["code", "level", "category", "rule"],
).set_index("code")


def _scope_marker_before(tokens: list[str], position: int) -> str | None:
    window = tokens[max(0, position - SCOPE_WINDOW):position]
    return next((tok for tok in reversed(window) if tok in SCOPE_MARKERS), None)


def code_error(text: str, gold: str, pred: str, lexicon: KeywordLexicon) -> dict:
    """Primary error category of one misclassified text (rules applied in the order A1, B, C, D, E, A2, F)."""
    tokens = text.split()
    keyword = lexicon.keyword(text)
    complement = feel_complement(text)
    split_word = lexicon.split_word_in(text)
    cue_classes = lexicon.cue_classes(text)

    # Position used for the scope check: the keyword if present, otherwise the first feel verb.
    anchor = keyword if keyword in tokens else next((t for t in tokens if t in FEEL_VERBS), None)
    marker = _scope_marker_before(tokens, tokens.index(anchor)) if anchor else None
    keyword_majority = lexicon.table_.at[keyword, "majority"] if keyword is not None else None

    if split_word is not None:
        code = "A1"
    elif keyword is None and complement is None:
        code = "B"
    elif len(cue_classes) >= 2 and pred in cue_classes:
        code = "C"
    elif marker is not None:
        code = "D"
    elif complement is not None and complement not in lexicon.table_.index:
        code = "E"
    elif keyword is not None and keyword in lexicon.reliable_words and gold != keyword_majority and pred == keyword_majority:
        code = "A2"
    else:
        code = "F"
    return {
        "code": code,
        "keyword": keyword,
        "split_word": split_word,
        "keyword_majority": keyword_majority,
        "scope_marker": marker,
        "cue_classes": ", ".join(sorted(cue_classes)),
    }


def code_errors(frame: pd.DataFrame, lexicon: KeywordLexicon, text_col: str = "text",
                gold_col: str = "true_label", pred_col: str = "pred_label") -> pd.DataFrame:
    """Apply ``code_error`` to every row of a predictions frame (normally its wrong predictions)."""
    coded = [code_error(t, g, p, lexicon) for t, g, p in zip(frame[text_col], frame[gold_col], frame[pred_col])]
    out = pd.DataFrame(coded, index=frame.index)
    out["level"] = out["code"].map(ERROR_CATEGORIES["level"])
    out["category"] = out["code"].map(ERROR_CATEGORIES["category"])
    return out


# ---------------------------------------------------------------------------
# Word-level explanations
# ---------------------------------------------------------------------------
def lr_contributions(pipeline, text: str, class_a: str, class_b: str) -> pd.Series:
    """Contribution of every word to the score difference between class_a and class_b.

    For a softmax (multinomial logistic) regression the log-odds of class a against
    class b is (w_a - w_b) . x + (b_a - b_b); the term of feature j is
    x_j (w_a,j - w_b,j). Positive values push the decision towards class_a.
    """
    vectorizer, model = pipeline[0], pipeline[-1]
    x = vectorizer.transform([text])
    a, b = LABEL2ID[class_a], LABEL2ID[class_b]
    diff = model.coef_[a] - model.coef_[b]
    cols = x.indices
    vocab = vectorizer.get_feature_names_out()
    values = x.data * diff[cols]
    return pd.Series(values, index=vocab[cols]).sort_values(ascending=False)


def occlusion(predict_proba: Callable[[Sequence[str]], np.ndarray], text: str, label: str) -> pd.Series:
    """Drop in P(label) when each word is deleted from the text (larger = more important)."""
    tokens = text.split()
    variants = [text] + [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    proba = predict_proba(variants)[:, LABEL2ID[label]]
    drops = proba[0] - proba[1:]
    index = [f"{tok}@{i}" if tokens.count(tok) > 1 else tok for i, tok in enumerate(tokens)]
    return pd.Series(drops, index=index).sort_values(ascending=False)


__all__ = ["ERROR_CATEGORIES", "SCOPE_MARKERS", "SCOPE_WINDOW", "code_error", "code_errors",
           "lr_contributions", "occlusion"]
