"""
Classification probe for zolt hidden-state representations.

Supports a hybrid label set: a predefined default intent taxonomy plus
user-configurable extensions. The backbone (ZoltForCausalLM) is always frozen;
only the probe head is trained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Predefined intent taxonomy ──────────────────────────────────────────────

DEFAULT_INTENT_LABELS: List[str] = [
    "code_generation",
    "code_review",
    "refactor",
    "explain",
    "question",
    "debug",
    "test_writing",
    "documentation",
]

DEFAULT_LANG_LABELS: List[str] = [
    "python",
    "typescript",
    "javascript",
    "sql",
    "other",
]


def build_label_set(
    predefined: List[str],
    extra: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge a predefined label list with optional user-supplied extensions.
    Duplicates are removed while preserving order (predefined labels first).
    """
    seen: Dict[str, int] = {}
    result: List[str] = []
    for label in predefined + (extra or []):
        if label not in seen:
            seen[label] = len(result)
            result.append(label)
    return result


# ── Probe architectures ──────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear classification head."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop(x))


class MLPProbe(nn.Module):
    """Two-layer MLP classification head with ReLU activation."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Public API ───────────────────────────────────────────────────────────────

class ClassificationProbe:
    """
    Trainable classification probe over zolt hidden-state representations.

    Label set is hybrid: a predefined set (DEFAULT_INTENT_LABELS or
    DEFAULT_LANG_LABELS) merged with any user-supplied custom labels.
    """

    def __init__(
        self,
        input_dim: int,
        predefined_labels: Optional[List[str]] = None,
        extra_labels: Optional[List[str]] = None,
        arch: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            input_dim: Embedding dimension (1024 for 250M, 768 for 125M).
            predefined_labels: Base label list. Defaults to DEFAULT_INTENT_LABELS.
            extra_labels: Additional user labels appended to predefined set.
            arch: 'linear' or 'mlp'.
            hidden_dim: Hidden size for MLP arch.
            dropout: Dropout rate.
            device: Torch device.
        """
        self.labels = build_label_set(
            predefined_labels if predefined_labels is not None else DEFAULT_INTENT_LABELS,
            extra_labels,
        )
        self.label2id: Dict[str, int] = {l: i for i, l in enumerate(self.labels)}
        self.id2label: Dict[int, str] = {i: l for i, l in enumerate(self.labels)}
        self.n_classes = len(self.labels)
        self.device = device or torch.device("cpu")

        if arch == "mlp":
            self.model = MLPProbe(input_dim, self.n_classes, hidden_dim, dropout)
        else:
            self.model = LinearProbe(input_dim, self.n_classes, dropout)
        self.model.to(self.device)

    def fit(
        self,
        embeddings: torch.Tensor,
        labels: Union[List[str], List[int], torch.Tensor],
        n_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the probe head on labeled embeddings.
        Returns per-epoch cross-entropy loss.
        """
        # Encode string labels
        if isinstance(labels, list) and labels and isinstance(labels[0], str):
            label_ids = torch.tensor([self.label2id[l] for l in labels], dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
            label_ids = labels.long()
        else:
            label_ids = torch.tensor(labels, dtype=torch.long)

        embeddings = embeddings.to(self.device).detach()
        label_ids = label_ids.to(self.device)
        n = embeddings.shape[0]

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        losses = []

        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            steps = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                logits = self.model(embeddings[idx])
                loss = F.cross_entropy(logits, label_ids[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            losses.append(avg_loss)
            if verbose:
                print(f"[probe-classify] epoch {epoch + 1}/{n_epochs} loss={avg_loss:.4f}")

        self.model.eval()
        return losses

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> List[str]:
        """Return predicted label strings for a batch of embeddings."""
        self.model.eval()
        logits = self.model(embeddings.to(self.device))
        ids = logits.argmax(dim=-1).tolist()
        return [self.id2label[i] for i in ids]

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return softmax probability distribution over labels."""
        self.model.eval()
        logits = self.model(embeddings.to(self.device))
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def accuracy(self, embeddings: torch.Tensor, labels: Union[List[str], torch.Tensor]) -> float:
        """Compute accuracy on a labeled evaluation set."""
        preds = self.predict(embeddings)
        if isinstance(labels, torch.Tensor):
            truth = [self.id2label[i.item()] for i in labels]
        else:
            truth = list(labels)
        correct = sum(p == t for p, t in zip(preds, truth))
        return correct / max(len(truth), 1)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize probe weights and label map to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "labels": self.labels,
                "arch": self.model.__class__.__name__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], input_dim: int, device: Optional[torch.device] = None) -> "ClassificationProbe":
        """Load a previously saved probe from disk."""
        data = torch.load(path, map_location="cpu")
        arch = "mlp" if data["arch"] == "MLPProbe" else "linear"
        probe = cls(
            input_dim=input_dim,
            predefined_labels=data["labels"],
            arch=arch,
            device=device,
        )
        probe.model.load_state_dict(data["state_dict"])
        probe.model.eval()
        return probe
