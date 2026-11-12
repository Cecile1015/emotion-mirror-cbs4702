"""Pre-trained GloVe word vectors and averaged sentence vectors (Lecture 6).

GloVe (Pennington et al., 2014) learns one dense, static vector per word type
from global word-word co-occurrence counts. Two pre-trained variants are used:

* ``twitter200``  2B tweets (27B tokens, 1.2M word types), 200 dimensions;
                  informal, user-generated text, close to the blog sentences
                  of dair-ai/emotion.
* ``wiki300``     Wikipedia 2014 + Gigaword 5 (6B tokens, 400K word types),
                  300 dimensions; edited encyclopaedic and news text.

Both files are the official Stanford vectors as redistributed by the
``gensim-data`` project (one gzip file per variant, word2vec text format).
They are downloaded once into ``data/embeddings/`` and converted to a NumPy
cache. Only word types that the project tokenizer (``src.data.tokenize``) can
produce are kept, so the cache is lossless for this dataset and loads in about
a second.

``MeanEmbeddingVectorizer`` turns a text into the average of its word vectors.
It is a scikit-learn transformer, so ``MeanEmbeddingVectorizer -> StandardScaler
-> LogisticRegression`` is a normal ``Pipeline`` that accepts raw strings.
The GloVe table is not stored inside the pickled pipeline; it is re-attached
from the local cache when the pipeline is loaded.
"""

from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .config import DATA_DIR
from .data import tokenize

EMBED_DIR = DATA_DIR / "embeddings"
_GENSIM_DATA = "https://github.com/RaRe-Technologies/gensim-data/releases/download"
# A token the project tokenizer can emit: lower-case letters, optionally one apostrophe group.
_TOKEN_FULL_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class GloVeVariant:
    """Metadata of one pre-trained GloVe file."""

    name: str
    file: str
    dim: int
    corpus: str
    vocab_size: int
    n_bytes: int

    @property
    def url(self) -> str:
        return f"{_GENSIM_DATA}/{self.file[:-3]}/{self.file}"


GLOVE_VARIANTS = {
    "twitter200": GloVeVariant(
        name="twitter200",
        file="glove-twitter-200.gz",
        dim=200,
        corpus="Twitter: 2B tweets, 27B tokens",
        vocab_size=1_193_514,
        n_bytes=795_373_100,
    ),
    "wiki300": GloVeVariant(
        name="wiki300",
        file="glove-wiki-gigaword-300.gz",
        dim=300,
        corpus="Wikipedia 2014 + Gigaword 5: 6B tokens",
        vocab_size=400_000,
        n_bytes=394_362_229,
    ),
}


# ---------------------------------------------------------------------------
# Download and cache
# ---------------------------------------------------------------------------
def download_glove(name: str) -> Path:
    """Download the gzip file of one GloVe variant (about 0.4-0.8 GB) if needed."""
    variant = GLOVE_VARIANTS[name]
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    target = EMBED_DIR / variant.file
    if target.exists() and target.stat().st_size == variant.n_bytes:
        return target

    from tqdm.auto import tqdm

    partial = target.with_suffix(".part")
    with urllib.request.urlopen(variant.url) as response, open(partial, "wb") as fh:
        with tqdm(total=variant.n_bytes, unit="B", unit_scale=True, desc=variant.file) as bar:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    if partial.stat().st_size != variant.n_bytes:
        raise IOError(f"Incomplete download of {variant.file}; delete {partial} and retry.")
    partial.rename(target)
    return target


def _cache_paths(name: str) -> tuple[Path, Path]:
    return EMBED_DIR / f"{name}.vectors.npy", EMBED_DIR / f"{name}.words.txt"


def _build_cache(name: str) -> None:
    """Parse the gzip text file once and save words + vectors in NumPy format."""
    variant = GLOVE_VARIANTS[name]
    source = download_glove(name)
    words, rows, seen = [], [], set()
    with gzip.open(source, "rt", encoding="utf-8") as fh:
        next(fh)  # header line: "<vocabulary size> <dimension>"
        for line in fh:
            word, _, values = line.rstrip("\n").partition(" ")
            # Keep the first (most frequent) entry of each type the tokenizer can produce.
            if word in seen or not _TOKEN_FULL_RE.fullmatch(word):
                continue
            vector = np.array(values.split(" "), dtype=np.float32)
            if vector.shape[0] != variant.dim:
                continue
            seen.add(word)
            words.append(word)
            rows.append(vector)
    vectors_path, words_path = _cache_paths(name)
    np.save(vectors_path, np.vstack(rows))
    words_path.write_text("\n".join(words), encoding="utf-8")


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------
class GloVe:
    """A word -> vector lookup table with a few helper queries.

    Rows are ordered by corpus frequency (the order of the original file), so
    a smaller row index means a more frequent word.
    """

    def __init__(self, name: str, words: Sequence[str], vectors: np.ndarray):
        self.name = name
        self.words = list(words)
        self.vectors = vectors
        self.index = {word: i for i, word in enumerate(self.words)}
        self._unit = None  # row-normalised copy, built on first similarity query

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]

    def _unit_vectors(self) -> np.ndarray:
        if self._unit is None:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self._unit = self.vectors / np.clip(norms, 1e-8, None)
        return self._unit

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of two words."""
        unit = self._unit_vectors()
        return float(unit[self.index[a]] @ unit[self.index[b]])

    def most_similar(self, word: str, k: int = 8, restrict: int | None = 50_000) -> list[tuple[str, float]]:
        """The k nearest neighbours by cosine similarity among the ``restrict`` most frequent words."""
        unit = self._unit_vectors()
        pool = unit if restrict is None else unit[:restrict]
        scores = pool @ unit[self.index[word]]
        order = np.argsort(-scores)
        return [(self.words[i], float(scores[i])) for i in order if self.words[i] != word][:k]


@lru_cache(maxsize=2)
def load_glove(name: str) -> GloVe:
    """Return one GloVe variant, downloading and caching it on first use."""
    if name not in GLOVE_VARIANTS:
        raise ValueError(f"Unknown GloVe variant '{name}'. Choose from {list(GLOVE_VARIANTS)}.")
    vectors_path, words_path = _cache_paths(name)
    if not (vectors_path.exists() and words_path.exists()):
        _build_cache(name)
    words = words_path.read_text(encoding="utf-8").split("\n")
    vectors = np.load(vectors_path)
    return GloVe(name, words, vectors)


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------
def coverage(token_lists: Iterable[Sequence[str]], glove: GloVe) -> dict:
    """Share of word types and running tokens that have a GloVe vector."""
    counts: dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    covered = {w for w in counts if w in glove}
    n_tokens = sum(counts.values())
    return {
        "types": len(counts),
        "type_coverage": len(covered) / max(len(counts), 1),
        "tokens": n_tokens,
        "token_coverage": sum(counts[w] for w in covered) / max(n_tokens, 1),
        "missing": {w: c for w, c in counts.items() if w not in covered},
    }


# ---------------------------------------------------------------------------
# Averaged sentence vectors
# ---------------------------------------------------------------------------
class MeanEmbeddingVectorizer(TransformerMixin, BaseEstimator):
    """Represent a text by the mean of the GloVe vectors of its tokens.

    Tokens without a vector are skipped; a text with no known token is mapped
    to the zero vector. The representation is order-free (a "bag of
    embeddings"): permuting the words of a text leaves it unchanged.
    """

    def __init__(self, glove_name: str = "wiki300"):
        self.glove_name = glove_name

    @property
    def glove(self) -> GloVe:
        return load_glove(self.glove_name)

    def fit(self, X: Iterable[str], y=None):
        self.n_features_in_ = 1
        self.dim_ = self.glove.dim
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        glove = self.glove
        out = np.zeros((len(X), glove.dim), dtype=np.float32)
        for i, text in enumerate(X):
            rows = [glove.index[t] for t in tokenize(text) if t in glove.index]
            if rows:
                out[i] = glove.vectors[rows].mean(axis=0)
        return out

    def known_token_share(self, X: Iterable[str]) -> np.ndarray:
        """Per text: the share of its tokens that received a vector."""
        glove = self.glove
        shares = []
        for text in X:
            tokens = tokenize(text)
            shares.append(np.mean([t in glove.index for t in tokens]) if tokens else 0.0)
        return np.asarray(shares)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray([f"{self.glove_name}_{j}" for j in range(self.glove.dim)], dtype=object)
