"""
Main entrypoint for the multi-task BERT pipeline.

Usage:
    # Single training run with default config
    python -m src.main --mode train

    # Optuna hyperparameter tuning
    python -m src.main --mode tune --n_trials 20

    # Smoke test (1 epoch, small subset)
    python -m src.main --mode train --epochs 1 --max_len 32 --sample_size 200
"""

import argparse
from pathlib import Path

import mlflow
import torch

from .config import Config
from .dataset import build_dataloaders, compute_weights, load_and_split_data
from .evaluate import full_evaluation
from .model import MultiTaskBERT
from .optuna_tuning import run_optuna_study
from .train import train_model
from .utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Task BERT Training Pipeline")
    p.add_argument("--mode", choices=["train", "tune"], default="train")

    # Data
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--sample_size", type=int, default=None)

    # Model
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)

    # Multi-task
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)

    # Optuna
    p.add_argument("--n_trials", type=int, default=20)

    # Output
    p.add_argument("--output_dir", type=str, default=None)

    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build Config, overriding defaults with any CLI args that were set."""
    config = Config()
    for field_name in vars(config):
        cli_val = getattr(args, field_name, None)
        if cli_val is not None:
            setattr(config, field_name, cli_val)
    return config


def run_single_train(config: Config) -> None:
    """Single training run with full evaluation and MLflow logging."""
    device = get_device()
    set_seed(config.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_and_split_data(config)
    loaders = build_dataloaders(config, train_df, val_df, test_df)

    sentiment_weight = compute_weights(
        train_df["sentiment_idx"].values, config.num_sentiment_classes
    )
    star_weight = compute_weights(
        train_df["star_idx"].values, config.num_star_classes
    )

    print(f"\nSentiment class weights: {sentiment_weight.tolist()}")
    print(f"Star class weights:     {star_weight.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MultiTaskBERT(
        model_name=config.model_name,
        num_sentiment_classes=config.num_sentiment_classes,
        num_star_classes=config.num_star_classes,
        dropout=config.dropout,
        star_embed_dim=config.star_embed_dim,
        fusion_dim=config.fusion_dim,
    ).to(device)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment("multitask-bert-single")

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config.model_name,
            "max_len": config.max_len,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "alpha": config.alpha,
            "beta": config.beta,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "epochs": config.epochs,
            "patience": config.patience,
            "max_grad_norm": config.max_grad_norm,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
            "sentiment_class_weights": sentiment_weight.tolist(),
            "star_class_weights": star_weight.tolist(),
        })

        # ── Train ─────────────────────────────────────────────────────────
        result = train_model(
            config, model, loaders["train"], loaders["val"],
            sentiment_weight, star_weight, device,
        )

        # Log per-epoch metrics
        for epoch, metrics in result["history"].items():
            mlflow.log_metrics(
                {
                    "train_loss": metrics["train_total_loss"],
                    "val_loss": metrics["val_total_loss"],
                    "train_sentiment_loss": metrics["train_sentiment_loss"],
                    "train_star_loss": metrics["train_star_loss"],
                    "val_sentiment_loss": metrics["val_sentiment_loss"],
                    "val_star_loss": metrics["val_star_loss"],
                },
                step=epoch,
            )

        mlflow.log_metric("best_val_loss", result["best_val_loss"])

        # ── Evaluate on test set ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)
        test_metrics = full_evaluation(
            model, loaders["test"], device, config, prefix="test"
        )
        mlflow.log_metrics(test_metrics)

        # ── Save model ────────────────────────────────────────────────────
        model_path = Path(config.output_dir) / "best_model.pt"
        torch.save(result["best_model_state"], model_path)
        mlflow.log_artifact(str(model_path))
        print(f"\nModel saved → {model_path}")

        # Log confusion matrix images
        for img in Path(config.output_dir).glob("*.png"):
            mlflow.log_artifact(str(img))

    print("\n✓ Training complete. Check MLflow UI for full details.")


def run_tune(config: Config, n_trials: int) -> None:
    """Run Optuna study, then retrain with best params on full data."""
    set_seed(config.seed)

    study = run_optuna_study(n_trials=n_trials, config=config)

    # ── Retrain with best params ──────────────────────────────────────────
    best_params = study.best_trial.params
    print(f"\nRetraining with best params: {best_params}")

    config.learning_rate = best_params["learning_rate"]
    config.batch_size = best_params["batch_size"]
    config.alpha = best_params["alpha"]
    config.beta = round(1.0 - config.alpha, 2)
    config.weight_decay = best_params["weight_decay"]
    config.dropout = best_params["dropout"]

    run_single_train(config)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print(f"Mode: {args.mode}")
    print(f"Config: {config}")
    print()

    if args.mode == "train":
        run_single_train(config)
    elif args.mode == "tune":
        run_tune(config, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
