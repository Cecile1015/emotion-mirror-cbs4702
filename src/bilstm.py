"""Bidirectional LSTM text classifier with GloVe initialisation (Lecture 7).

The module follows the structure of the PyTorch examples in Lecture 7:

* ``Vocabulary``          word <-> index mapping with ``<pad>`` = 0 and ``<unk>`` = 1.
* ``EmotionDataset``      a ``torch.utils.data.Dataset`` of index sequences and labels.
* ``collate_batch``       pads a mini-batch to its longest sequence (dynamic padding).
* ``BiLSTMClassifier``    ``nn.Embedding`` -> ``nn.LSTM(bidirectional=True)`` -> ``nn.Linear``.
* ``train_epoch`` / ``evaluate_epoch`` / ``fit``
                          the training loop with early stopping on validation macro-F1.
* ``save_checkpoint`` / ``load_checkpoint`` / ``predict_proba``
                          persistence and raw-text inference for the web service.

Padding is handled with ``pack_padded_sequence``: the LSTM only runs over the
real tokens of each sequence, so the final hidden state of the forward
direction is the state after the last real word (not after padding).
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import NUM_LABELS, SEED, set_seed
from .data import tokenize

PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1
MAX_LEN = 66  # longest training text in word tokens (01_eda); longer inputs are truncated


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(prefer: str | None = None) -> torch.device:
    """Apple-silicon GPU (MPS) if available, then CUDA, then CPU."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Vocabulary and embedding matrix
# ---------------------------------------------------------------------------
class Vocabulary:
    """Word <-> index mapping; index 0 is padding and index 1 the unknown word."""

    def __init__(self, words: Iterable[str]):
        self.itos = [PAD, UNK] + [w for w in dict.fromkeys(words) if w not in (PAD, UNK)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        """Tokenise a text and map each token to its index (unknown words -> 1)."""
        ids = [self.stoi.get(token, UNK_IDX) for token in tokenize(text)[:max_len]]
        return ids or [UNK_IDX]  # an empty text becomes a single <unk> token


def build_vocabulary(train_texts: Sequence[str], extra_words: Iterable[str] = ()) -> Vocabulary:
    """Training-set word types (most frequent first), followed by ``extra_words``.

    ``extra_words`` are pre-trained words that do not occur in the training
    set. Their embedding rows never receive a gradient (they never appear in
    a training batch), so they keep their GloVe values in both the frozen and
    the fine-tuned model.
    """
    counts = Counter(token for text in train_texts for token in tokenize(text))
    train_words = [w for w, _ in counts.most_common()]
    seen = set(train_words)
    return Vocabulary(train_words + [w for w in extra_words if w not in seen])


def build_embedding_matrix(vocab: Vocabulary, glove, seed: int = SEED) -> tuple[torch.Tensor, np.ndarray]:
    """Rows initialised from GloVe; words without a vector get small random values.

    Returns the (|V|, d) matrix and a boolean mask of rows taken from GloVe.
    Random rows are drawn from N(0, sigma^2) with sigma = the standard
    deviation of the GloVe values, so they have the same scale as the
    pre-trained rows. The padding row is all zeros.
    """
    rng = np.random.default_rng(seed)
    sigma = float(glove.vectors.std())
    matrix = rng.normal(0.0, sigma, size=(len(vocab), glove.dim)).astype(np.float32)
    in_glove = np.zeros(len(vocab), dtype=bool)
    for i, word in enumerate(vocab.itos):
        j = glove.index.get(word)
        if j is not None:
            matrix[i] = glove.vectors[j]
            in_glove[i] = True
    matrix[PAD_IDX] = 0.0
    return torch.from_numpy(matrix), in_glove


# ---------------------------------------------------------------------------
# Dataset and mini-batches
# ---------------------------------------------------------------------------
class EmotionDataset(Dataset):
    """Index sequences and labels of one split."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None, vocab: Vocabulary):
        self.sequences = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.labels = None if labels is None else torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        label = -1 if self.labels is None else self.labels[i]
        return self.sequences[i], label


def collate_batch(batch):
    """Pad the sequences of one batch to the same length.

    Returns ``text`` (batch, max_len), ``text_lengths`` (batch,) and ``labels``.
    The lengths stay on the CPU, as ``pack_padded_sequence`` requires.
    """
    sequences, labels = zip(*batch)
    text_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    text = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return text, text_lengths, torch.as_tensor(labels, dtype=torch.long)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int = SEED) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                      generator=generator)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM -> final hidden states -> linear softmax layer.

    The sentence representation is the concatenation of the last hidden state
    of the forward LSTM (which has read the text left to right) and the last
    hidden state of the backward LSTM (which has read it right to left), as in
    the Lecture 7 example. The linear layer outputs one score (logit) per
    emotion; softmax is applied inside ``nn.CrossEntropyLoss``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int = NUM_LABELS,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = PAD_IDX,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        # Frozen: the GloVe vectors stay fixed. Fine-tuned: they are updated by back-propagation.
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.bidirectional = bidirectional

    def encode(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """Sentence vector of shape (batch, hidden_dim * num_directions)."""
        embedded = self.dropout(self.embedding(text))                  # (batch, seq_len, emb_dim)
        packed = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)              # hidden: (layers * dirs, batch, hid)
        if self.bidirectional:
            # hidden[-2] = last forward state, hidden[-1] = last backward state (top layer).
            return torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden[-1]

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        sentence = self.dropout(self.encode(text, text_lengths))
        return self.fc(sentence)                                        # (batch, output_dim) logits


def count_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyper-parameters of one training run (saved with the checkpoint)."""

    glove: str = "wiki300"
    hidden_dim: int = 128
    n_layers: int = 1
    dropout: float = 0.5
    freeze_embeddings: bool = False
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 30
    patience: int = 4
    clip_norm: float = 5.0
    class_weights: bool = True
    seed: int = SEED
    notes: str = field(default="")


def balanced_class_weights(labels: Sequence[int], n_classes: int = NUM_LABELS) -> torch.Tensor:
    """n / (k * n_c) for each class c, the same formula as scikit-learn's class_weight='balanced'."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device, clip_norm: float | None = 5.0) -> float:
    """One pass over the training data; returns the mean training loss."""
    model.train()
    total_loss, n = 0.0, 0
    for text, text_lengths, labels in loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(text, text_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # guards against exploding gradients
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device) -> dict:
    """Loss, predictions and class probabilities on one split (no gradient, dropout off)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_proba, all_labels = [], []
    for text, text_lengths, labels in loader:
        text, labels = text.to(device), labels.to(device)
        logits = model(text, text_lengths)
        if criterion is not None and labels.min() >= 0:
            total_loss += criterion(logits, labels).item() * len(labels)
        n += len(labels)
        all_proba.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels.cpu())
    proba = torch.cat(all_proba).numpy()
    y = torch.cat(all_labels).numpy()
    pred = proba.argmax(axis=1)
    out = {"loss": total_loss / n, "proba": proba, "pred": pred}
    if y.min() >= 0:
        out["macro_f1"] = float(f1_score(y, pred, average="macro"))
        out["accuracy"] = float(np.mean(pred == y))
    return out


def fit(
    config: TrainConfig,
    vocab: Vocabulary,
    embedding_matrix: torch.Tensor,
    train_data: tuple[Sequence[str], Sequence[int]],
    val_data: tuple[Sequence[str], Sequence[int]],
    device: torch.device,
    verbose: bool = True,
) -> tuple[BiLSTMClassifier, pd.DataFrame]:
    """Train with early stopping on validation macro-F1 and restore the best epoch.

    After every epoch the model is scored on the validation split. Training
    stops when macro-F1 has not improved for ``patience`` epochs, and the
    weights of the best epoch are returned with the per-epoch history.
    """
    set_seed(config.seed)
    train_ds = EmotionDataset(*train_data, vocab)
    val_ds = EmotionDataset(*val_data, vocab)
    train_loader = make_loader(train_ds, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_loader(val_ds, 256, shuffle=False)
    train_eval_loader = make_loader(train_ds, 256, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_matrix.shape[1],
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=config.freeze_embeddings,
    ).to(device)

    weight = balanced_class_weights(train_data[1]).to(device) if config.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    # Plain Adam without weight decay: rows of the embedding matrix that never
    # receive a gradient are left exactly unchanged.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)

    history, best_f1, best_state, best_epoch, stale = [], -1.0, None, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config.clip_norm)
        train_eval = evaluate_epoch(model, train_eval_loader, criterion, device)
        val_eval = evaluate_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,                  # mean loss during the epoch (dropout on)
            "train_loss_eval": train_eval["loss"],     # loss at the end of the epoch (dropout off)
            "val_loss": val_eval["loss"],
            "train_macro_f1": train_eval["macro_f1"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_accuracy": val_eval["accuracy"],
            "seconds": time.perf_counter() - start,
        })
        improved = val_eval["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = val_eval["macro_f1"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if verbose:
            row = history[-1]
            print(f"epoch {epoch:>2}  train loss {row['train_loss']:.4f}  val loss {row['val_loss']:.4f}  "
                  f"val macro-F1 {row['val_macro_f1']:.4f}{'  *' if improved else ''}  ({row['seconds']:.0f}s)")
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    history["best_epoch"] = best_epoch
    return model, history


# ---------------------------------------------------------------------------
# Inference and persistence
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: BiLSTMClassifier, vocab: Vocabulary, texts: Sequence[str],
                  device: torch.device | None = None, batch_size: int = 256) -> np.ndarray:
    """Class probabilities (n, 6) for raw input strings."""
    device = device or next(model.parameters()).device
    loader = make_loader(EmotionDataset(texts, None, vocab), batch_size, shuffle=False)
    return evaluate_epoch(model, loader, None, device)["proba"]


def save_checkpoint(path: Path, model: BiLSTMClassifier, vocab: Vocabulary, config: TrainConfig,
                    extra: dict | None = None) -> None:
    """Save weights, vocabulary and training configuration in one file."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "itos": vocab.itos, "config": asdict(config), "extra": extra or {}}, path)


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[BiLSTMClassifier, Vocabulary, dict]:
    """Rebuild a ``BiLSTMClassifier`` and its vocabulary from ``save_checkpoint`` output."""
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    vocab = Vocabulary(ckpt["itos"][2:])
    weight = ckpt["state_dict"]["embedding.weight"]
    model = BiLSTMClassifier(
        vocab_size=weight.shape[0],
        embedding_dim=weight.shape[1],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), vocab, ckpt
