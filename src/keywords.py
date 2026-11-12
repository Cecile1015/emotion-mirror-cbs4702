"""The emotion keyword of a text and the consistency of the gold labels (used in 08_error_analysis).

Almost every text of dair-ai/emotion contains a form of *feel* (97.3% of the
training texts, 01_eda), and the word that follows it usually names the emotion:
"i feel *terrified* ...", "im feeling *blessed* ...". This module extracts that
word, tabulates how the training split labels each keyword, and flags the
keywords whose label is inconsistent.

* ``feel_complement``   the first content word after feel / feels / feeling / felt,
                        skipping intensifiers and hedges ("so", "a little", "kind of").
* ``KeywordLexicon``    label distribution of every keyword in the training split,
                        a one-word look-up classifier, the *split keywords*
                        (words whose training texts are divided between two
                        classes) and the *reliable keywords* (one dominant class).

A split keyword is a sign of annotation noise when the context does not tell
the two labels apart; ``context_signal`` in 08_error_analysis tests exactly that.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import LABELS

FEEL_VERBS = frozenset({"feel", "feels", "feeling", "felt"})

# Degree adverbs, hedges and determiners that may stand between "feel" and the emotion word.
KEYWORD_MODIFIERS = frozenset(
    """so very really quite a little bit kind of kinda sort sorta pretty too more less rather extremely
    incredibly totally slightly somewhat all just most much even absolutely completely utterly super truly
    fairly terribly increasingly especially particularly genuinely deeply awfully strangely oddly almost
    already still enough mildly overly soo sooo emotionally physically mentally strongly tad damn fucking such""".split()
)

# Words after "feel" that do not name an emotion (pronouns, prepositions, auxiliaries, "like", "that").
KEYWORD_STOPWORDS = frozenset(
    """like that i it the to for about and in when with my me you is what was this of as if myself im need
    though he way they its being but at there we your some she lot not his an than any are people her them
    their our those these on or because how who have has had be from by up out one something someone anything
    everything nothing would will can could should do does did ive id youre theyre dont don t s m somehow
    after far however good bad better worse right ok okay fine no left today sense""".split()
)


def feel_complement(text: str) -> str | None:
    """First content word after a form of *feel*, or None if there is none."""
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token in FEEL_VERBS:
            j = i + 1
            while j < len(tokens) and tokens[j] in KEYWORD_MODIFIERS:
                j += 1
            if j < len(tokens) and tokens[j] not in KEYWORD_STOPWORDS:
                return tokens[j]
    return None


@dataclass
class KeywordLexicon:
    """Label distribution of each emotion keyword in the training split.

    Parameters
    ----------
    min_count : a keyword is described by the texts in which it follows *feel* if
        there are at least this many of them.
    anywhere_min_count : otherwise (a word that follows *feel* only 2-9 times) it is
        described by all training texts that contain it, if there are at least this many;
        such a word is kept only if it is a split keyword (e.g. *supporting*, *beloved*).
    split_max_purity, split_min_top2 : a keyword is *split* if its most frequent
        label covers at most ``split_max_purity`` of its texts while its two most
        frequent labels together cover at least ``split_min_top2``.
    reliable_min_purity : a keyword is *reliable* if one label covers at least this share.
    """

    min_count: int = 10
    anywhere_min_count: int = 20
    split_max_purity: float = 0.75
    split_min_top2: float = 0.95
    reliable_min_purity: float = 0.80
    table_: pd.DataFrame = field(default=None, init=False, repr=False)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "KeywordLexicon":
        texts, labels = list(texts), list(labels)
        # 1. Candidate emotion words: content words that follow "feel" at least twice.
        after_feel: dict[str, Counter] = defaultdict(Counter)
        for text, label in zip(texts, labels):
            word = feel_complement(text)
            if word is not None:
                after_feel[word][label] += 1
        candidates = {w for w, c in after_feel.items() if sum(c.values()) >= 2}
        # 2. The same words counted anywhere in a text (one count per text).
        anywhere: dict[str, Counter] = defaultdict(Counter)
        for text, label in zip(texts, labels):
            for token in set(text.split()) & candidates:
                anywhere[token][label] += 1
        # 3. Statistics after "feel" when frequent enough there, otherwise over all occurrences.
        rows = {}
        for word in candidates:
            if sum(after_feel[word].values()) >= self.min_count:
                rows[word] = {**after_feel[word], "source": "after feel"}
            elif sum(anywhere[word].values()) >= self.anywhere_min_count:
                rows[word] = {**anywhere[word], "source": "anywhere"}
        table = pd.DataFrame.from_dict(rows, orient="index")
        source = table.pop("source")
        table = table.reindex(columns=LABELS).fillna(0).astype(int)
        n = table.sum(axis=1)
        shares = table.div(n, axis=0).to_numpy()
        order = np.argsort(-shares, axis=1, kind="stable")
        sorted_shares = np.take_along_axis(shares, order, axis=1)
        table = table.assign(
            n=n,
            majority=[LABELS[i] for i in order[:, 0]],
            second=[LABELS[i] for i in order[:, 1]],
            purity=sorted_shares[:, 0],
            top2=sorted_shares[:, :2].sum(axis=1),
            source=source,
        )
        table["kind"] = np.select(
            [
                (table.purity <= self.split_max_purity) & (table.top2 >= self.split_min_top2),
                table.purity >= self.reliable_min_purity,
            ],
            ["split", "reliable"],
            default="mixed",
        )
        # A word described by all its occurrences is kept only if it is split: most such words
        # (pronouns, adverbs) are not emotion words, but a two-way split identifies one that is.
        table = table[(table.source == "after feel") | (table.kind == "split")]
        self.table_ = table.sort_values(["n", "purity"], ascending=[False, False])
        return self

    # -- look-ups -------------------------------------------------------------
    @property
    def split_words(self) -> set[str]:
        return set(self.table_.index[self.table_.kind == "split"])

    @property
    def reliable_words(self) -> set[str]:
        return set(self.table_.index[self.table_.kind == "reliable"])

    def keyword(self, text: str) -> str | None:
        """The keyword of a text: its feel-complement if listed, else the first listed word in it."""
        word = feel_complement(text)
        if word in self.table_.index:
            return word
        for token in text.split():
            if token in self.table_.index:
                return token
        return None

    def keywords(self, texts: Iterable[str]) -> list[str | None]:
        return [self.keyword(t) for t in texts]

    def split_word_in(self, text: str) -> str | None:
        """The first split keyword occurring anywhere in the text (None if there is none)."""
        split = self.split_words
        return next((tok for tok in text.split() if tok in split), None)

    def cue_classes(self, text: str) -> set[str]:
        """Majority classes of all reliable keywords that occur anywhere in the text."""
        reliable = self.reliable_words
        return {self.table_.at[tok, "majority"] for tok in set(text.split()) if tok in reliable}

    def predict(self, texts: Iterable[str]) -> list[str | None]:
        """One-word look-up classifier: the training majority label of each text's keyword."""
        return [self.table_.at[k, "majority"] if k is not None else None for k in self.keywords(texts)]


__all__ = ["FEEL_VERBS", "KEYWORD_MODIFIERS", "KEYWORD_STOPWORDS", "KeywordLexicon", "feel_complement"]
