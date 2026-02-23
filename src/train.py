"""
Training loop with AdamW, linear warmup scheduler, early stopping,
and gradient clipping for the multi-task BERT model.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    sentiment_criterion: nn.CrossEntropyLoss,
    star_criterion: nn.CrossEntropyLoss,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch.

    Returns:
        dict with 'total_loss', 'sentiment_loss', 'star_loss' (epoch averages).
    """
    model.train()
    running = {"total": 0.0, "sentiment": 0.0, "star": 0.0}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)
        star_labels = batch["star_label"].to(device)
        star_input = batch["star_input"].to(device)

        optimizer.zero_grad()

        sentiment_logits, star_logits = model(input_ids, attention_mask, star_input)

        loss_sent = sentiment_criterion(sentiment_logits, sentiment_labels)
        loss_star = star_criterion(star_logits, star_labels)
        loss = config.alpha * loss_sent + config.beta * loss_star

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        running["total"] += loss.item()
        running["sentiment"] += loss_sent.item()
        running["star"] += loss_star.item()
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    sentiment_criterion: nn.CrossEntropyLoss,
    star_criterion: nn.CrossEntropyLoss,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return average losses.

    Returns:
        dict with 'total_loss', 'sentiment_loss', 'star_loss'.
    """
    model.eval()
    running = {"total": 0.0, "sentiment": 0.0, "star": 0.0}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)
        star_labels = batch["star_label"].to(device)
        star_input = batch["star_input"].to(device)

        sentiment_logits, star_logits = model(input_ids, attention_mask, star_input)

        loss_sent = sentiment_criterion(sentiment_logits, sentiment_labels)
        loss_star = star_criterion(star_logits, star_labels)
        loss = config.alpha * loss_sent + config.beta * loss_star

        running["total"] += loss.item()
        running["sentiment"] += loss_sent.item()
        running["star"] += loss_star.item()
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


def train_model(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sentiment_weight: torch.FloatTensor,
    star_weight: torch.FloatTensor,
    device: torch.device,
) -> dict:
    """Full training loop with early stopping.

    Args:
        config: training configuration.
        model: MultiTaskBERT model (already on device).
        train_loader: training DataLoader.
        val_loader: validation DataLoader.
        sentiment_weight: class weight tensor for sentiment loss.
        star_weight: class weight tensor for star loss.
        device: torch device.

    Returns:
        dict with:
            'history': {epoch -> {train_*, val_*}}
            'best_val_loss': float
            'best_model_state': state_dict of best model
    """
    # ── Loss functions with class weights ─────────────────────────────────
    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weight.to(device))
    star_criterion = nn.CrossEntropyLoss(weight=star_weight.to(device))

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── Scheduler (linear warmup) ─────────────────────────────────────────
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history: dict[int, dict] = {}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            sentiment_criterion, star_criterion, config, device,
        )

        val_metrics = validate(
            model, val_loader,
            sentiment_criterion, star_criterion, config, device,
        )

        history[epoch] = {
            "train_total_loss": train_metrics["total"],
            "train_sentiment_loss": train_metrics["sentiment"],
            "train_star_loss": train_metrics["star"],
            "val_total_loss": val_metrics["total"],
            "val_sentiment_loss": val_metrics["sentiment"],
            "val_star_loss": val_metrics["star"],
            "learning_rate": scheduler.get_last_lr()[0],
        }

        print(f"  Train Loss: {train_metrics['total']:.4f}")
        print(f"  Val   Loss: {val_metrics['total']:.4f}")

        # ── Early stopping ────────────────────────────────────────────────
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                print("  ⚠ Early stopping triggered.")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_model_state": best_model_state,
    }
