"""
Central configuration for the multi-task BERT model.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = "data/processed/data_augmented.csv"
    text_column: str = "text"
    sentiment_column: str = "sentiment_label"
    star_column: str = "corrected_star"
    star_input_column: str = "stars"

    # ── Splits ────────────────────────────────────────────────────────────
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "indobenchmark/indobert-base-p1"
    max_len: int = 128
    num_sentiment_classes: int = 3
    num_star_classes: int = 5
    dropout: float = 0.3
    star_embed_dim: int = 16
    fusion_dim: int = 256

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # ── Multi-task loss weights ───────────────────────────────────────────
    alpha: float = 0.7  # sentiment loss weight
    beta: float = 0.3   # star loss weight

    # ── Early stopping ────────────────────────────────────────────────────
    patience: int = 3

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "outputs"

    # ── Optional: limit data for smoke tests ──────────────────────────────
    sample_size: int | None = None

    # ── Label maps ────────────────────────────────────────────────────────
    sentiment_labels: list[str] = field(
        default_factory=lambda: ["Negative", "Neutral", "Positive"]
    )
    star_labels: list[str] = field(
        default_factory=lambda: ["1", "2", "3", "4", "5"]
    )

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
