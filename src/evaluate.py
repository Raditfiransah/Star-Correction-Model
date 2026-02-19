"""
Evaluation utilities: metrics computation and confusion matrix plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference and collect all predictions + true labels.

    Returns:
        dict with keys:
            'sentiment_true', 'sentiment_pred',
            'star_true', 'star_pred'
    """
    model.eval()
    results = {
        "sentiment_true": [],
        "sentiment_pred": [],
        "star_true": [],
        "star_pred": [],
    }

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        sentiment_logits, star_logits = model(input_ids, attention_mask)

        results["sentiment_pred"].extend(
            sentiment_logits.argmax(dim=1).cpu().numpy()
        )
        results["star_pred"].extend(star_logits.argmax(dim=1).cpu().numpy())
        results["sentiment_true"].extend(batch["sentiment_label"].numpy())
        results["star_true"].extend(batch["star_label"].numpy())

    return {k: np.array(v) for k, v in results.items()}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute accuracy, macro-F1, and weighted-F1 for one task.

    Also prints a full classification report.

    Returns:
        dict with 'accuracy', 'macro_f1', 'weighted_f1'.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {task_name} Metrics")
    print(f"{'─'*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print()
    print(
        classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        )
    )

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def full_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
    prefix: str = "test",
) -> dict[str, float]:
    """End-to-end evaluation: metrics + confusion matrices for both tasks.

    Returns:
        dict with all metric values, keyed like
        '{prefix}_sentiment_accuracy', '{prefix}_star_weighted_f1', etc.
    """
    preds = get_predictions(model, dataloader, device)

    # ── Sentiment metrics ─────────────────────────────────────────────────
    sent_metrics = compute_metrics(
        preds["sentiment_true"],
        preds["sentiment_pred"],
        "Sentiment",
        label_names=config.sentiment_labels,
    )
    plot_confusion_matrix(
        preds["sentiment_true"],
        preds["sentiment_pred"],
        labels=config.sentiment_labels,
        title=f"Sentiment Confusion Matrix ({prefix})",
        save_path=Path(config.output_dir) / f"{prefix}_sentiment_cm.png",
    )

    # ── Star metrics ──────────────────────────────────────────────────────
    star_metrics = compute_metrics(
        preds["star_true"],
        preds["star_pred"],
        "Star Rating",
        label_names=config.star_labels,
    )
    plot_confusion_matrix(
        preds["star_true"],
        preds["star_pred"],
        labels=config.star_labels,
        title=f"Star Rating Confusion Matrix ({prefix})",
        save_path=Path(config.output_dir) / f"{prefix}_star_cm.png",
    )

    # Flatten into a single dict
    all_metrics = {}
    for key, val in sent_metrics.items():
        all_metrics[f"{prefix}_sentiment_{key}"] = val
    for key, val in star_metrics.items():
        all_metrics[f"{prefix}_star_{key}"] = val

    return all_metrics
