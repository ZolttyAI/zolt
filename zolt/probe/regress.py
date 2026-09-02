"""
Regression probes for zolt hidden-state representations.

Supports two supervised targets:
  - quality_score: real-valued output of compute_textbook_quality_score()
  - complexity:    real-valued output of estimate_code_complexity()

Each head is a separate module with its own weights and training loop.
The backbone is always frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Registered target names with their expected value ranges
REGRESSION_TARGETS: dict[str, dict] = {
    "quality_score": {"min": 0.0, "max": 1.0, "description": "Textbook quality heuristic score"},
    "complexity": {"min": 0.0, "max": None, "description": "AST-based code complexity estimate"},
}


class RegressionHead(nn.Module):
    """Linear regression head mapping embeddings to a scalar or vector output."""

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop(x))


class MLPRegressionHead(nn.Module):
    """Two-layer MLP regression head with ReLU."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionProbe:
    """
    Trainable regression probe over zolt hidden-state embeddings.

    target_name selects which supervised signal to train against:
      'quality_score' -- output of compute_textbook_quality_score()
      'complexity'    -- output of estimate_code_complexity()
    Custom targets are also supported by passing target_name='custom'.
    """

    def __init__(
        self,
        input_dim: int,
        target_name: str = "quality_score",
        output_dim: int = 1,
        arch: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        loss: str = "mse",
        device: torch.device | None = None,
    ):
        """
        Args:
            input_dim:   Embedding dimension.
            target_name: 'quality_score', 'complexity', or 'custom'.
            output_dim:  1 for scalar regression; >1 for multi-target.
            arch:        'linear' or 'mlp'.
            hidden_dim:  Hidden size for MLP arch.
            dropout:     Dropout rate.
            loss:        'mse' or 'huber'.
            device:      Torch device.
        """
        if target_name not in REGRESSION_TARGETS and target_name != "custom":
            raise ValueError(
                f"Unknown target '{target_name}'. "
                f"Known targets: {list(REGRESSION_TARGETS.keys())} or 'custom'."
            )
        self.target_name = target_name
        self.output_dim = output_dim
        self.loss_name = loss
        self.device = device or torch.device("cpu")

        self.model: nn.Module
        if arch == "mlp":
            self.model = MLPRegressionHead(input_dim, output_dim, hidden_dim, dropout)
        else:
            self.model = RegressionHead(input_dim, output_dim, dropout)
        self.model.to(self.device)

    def _loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "huber":
            return F.huber_loss(preds, targets)
        return F.mse_loss(preds, targets)

    def fit(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        n_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train the regression head on labeled embeddings.
        Returns per-epoch loss values.
        """
        embeddings = embeddings.to(self.device).detach()
        targets = targets.float().to(self.device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
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
                preds = self.model(embeddings[idx])
                loss = self._loss_fn(preds, targets[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            losses.append(avg_loss)
            if verbose:
                print(
                    f"[probe-regress:{self.target_name}] epoch {epoch + 1}/{n_epochs} loss={avg_loss:.6f}"
                )

        self.model.eval()
        return losses

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted scalar/vector values for a batch of embeddings."""
        self.model.eval()
        return self.model(embeddings.to(self.device)).squeeze(-1)

    @torch.no_grad()
    def pearson_r(self, embeddings: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient between predictions and targets."""
        preds = self.predict(embeddings).cpu().float()
        targets = targets.float().cpu()
        if preds.dim() > 1:
            preds = preds[:, 0]
        if targets.dim() > 1:
            targets = targets[:, 0]
        vp = preds - preds.mean()
        vt = targets - targets.mean()
        denom = (vp.norm() * vt.norm()).item()
        if denom < 1e-12:
            return 0.0
        return (vp * vt).sum().item() / denom

    def save(self, path: str | Path) -> None:
        """Serialize probe weights and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "target_name": self.target_name,
                "output_dim": self.output_dim,
                "arch": self.model.__class__.__name__,
                "loss": self.loss_name,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, input_dim: int, device: torch.device | None = None
    ) -> RegressionProbe:
        """Load a previously saved regression probe from disk."""
        data = torch.load(path, map_location="cpu")
        arch = "mlp" if "MLP" in data["arch"] else "linear"
        probe = cls(
            input_dim=input_dim,
            target_name=data["target_name"],
            output_dim=data["output_dim"],
            arch=arch,
            loss=data.get("loss", "mse"),
            device=device,
        )
        probe.model.load_state_dict(data["state_dict"])
        probe.model.eval()
        return probe


def build_regression_suite(
    input_dim: int,
    device: torch.device | None = None,
) -> dict[str, RegressionProbe]:
    """
    Instantiate both standard regression probes (quality_score and complexity).
    Returns a dict keyed by target name for convenient batch usage.
    """
    return {
        "quality_score": RegressionProbe(
            input_dim=input_dim,
            target_name="quality_score",
            arch="linear",
            loss="mse",
            device=device,
        ),
        "complexity": RegressionProbe(
            input_dim=input_dim,
            target_name="complexity",
            arch="linear",
            loss="huber",  # Huber is more robust for unbounded complexity values
            device=device,
        ),
    }
