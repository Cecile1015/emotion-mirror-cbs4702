"""Diagnostic slices of the evaluation data, shared by all modelling notebooks.

The definitions are those of ``01_eda`` (§4 and §8.1), moved here so that every
model is analysed on exactly the same subsets:

* ``template_slice``     "starts with 'i feel'" / "feel elsewhere" / "no form of feel"
* ``negation_slice``     whether the text contains a negation cue
* ``unseen_word_mask``   whether the text contains at least one word that never
                         occurs in the training split (out-of-vocabulary, OOV)
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from .data import tokenize

FEEL_RE = re.compile(r"\b(?:feel|feels|feeling|felt)\b")

# Negation cues. The dataset has no apostrophes, so contractions appear either
# joined ("dont") or split ("don t"); both spellings are matched.
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|nothing|nobody|none|nor|without|hardly|cannot|cant|dont|didnt|doesnt|isnt|wasnt|arent|werent"
    r"|wont|wouldnt|couldnt|shouldnt|havent|hasnt|hadnt|aint)\b"
    r"|\b(?:don|didn|doesn|isn|wasn|aren|weren|won|wouldn|couldn|shouldn|haven|hasn|hadn|can|ain)\s+t\b"
)

TEMPLATE_LEVELS = ["starts with 'i feel'", "feel elsewhere", "no form of feel"]
NEGATION_LEVELS = ["without negation cue", "with negation cue"]
OOV_LEVELS = ["all words seen in training", ">= 1 unseen word"]


def template_slice(texts: Sequence[str]) -> np.ndarray:
    """Label each text by how it uses the dominant 'feel' template."""
    texts = list(texts)
    starts = np.array([t.startswith("i feel") for t in texts])
    has_feel = np.array([bool(FEEL_RE.search(t)) for t in texts])
    return np.select([starts, has_feel], TEMPLATE_LEVELS[:2], default=TEMPLATE_LEVELS[2])


def has_negation(texts: Sequence[str]) -> np.ndarray:
    """Boolean mask: True if the text contains a negation cue."""
    return np.array([bool(NEGATION_RE.search(t)) for t in texts])


def negation_slice(texts: Sequence[str]) -> np.ndarray:
    return np.where(has_negation(texts), NEGATION_LEVELS[1], NEGATION_LEVELS[0])


def training_vocabulary(train_texts: Iterable[str]) -> set[str]:
    """Word types of the training split, using the descriptive tokenizer of 01_eda."""
    return {tok for text in train_texts for tok in tokenize(text)}


def unseen_words(text: str, vocab: set[str]) -> list[str]:
    """Words of ``text`` that never occur in the training split."""
    return [tok for tok in tokenize(text) if tok not in vocab]


def unseen_word_mask(texts: Sequence[str], vocab: set[str]) -> np.ndarray:
    """Boolean mask: True if the text contains at least one unseen word."""
    return np.array([len(unseen_words(t, vocab)) > 0 for t in texts])


def oov_slice(texts: Sequence[str], vocab: set[str]) -> np.ndarray:
    return np.where(unseen_word_mask(texts, vocab), OOV_LEVELS[1], OOV_LEVELS[0])
