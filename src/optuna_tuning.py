"""
Optuna hyperparameter tuning with MLflow experiment tracking.
"""

import numpy as np
import mlflow
import optuna
import torch
import torch.nn as nn

from .config import Config
from .dataset import build_dataloaders, compute_weights, load_and_split_data
from .evaluate import full_evaluation, get_predictions
from .model import MultiTaskBERT
from .train import train_model
from .utils import set_seed, get_device


def objective(trial: optuna.Trial, config: Config | None = None) -> float:
    """Optuna objective: train a model with suggested hyperparams, return score.

    The objective maximises a weighted combination of validation
    weighted-F1 for both tasks:
        score = 0.5 * sentiment_weighted_f1 + 0.5 * star_weighted_f1

    Every trial is logged as an MLflow run.
    """
    if config is None:
        config = Config()

    # ── Suggest hyperparameters ───────────────────────────────────────────
    config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    config.alpha = trial.suggest_float("alpha", 0.2, 0.8, step=0.1)
    config.beta = round(1.0 - config.alpha, 2)
    config.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    config.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    device = get_device()
    set_seed(config.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, _ = load_and_split_data(config)
    loaders = build_dataloaders(config, train_df, val_df)

    sentiment_weight = compute_weights(
        train_df["sentiment_idx"].values, config.num_sentiment_classes
    )
    star_weight = compute_weights(
        train_df["star_idx"].values, config.num_star_classes
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = MultiTaskBERT(
        model_name=config.model_name,
        num_sentiment_classes=config.num_sentiment_classes,
        num_star_classes=config.num_star_classes,
        dropout=config.dropout,
    ).to(device)

    # ── MLflow tracking ───────────────────────────────────────────────────
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "alpha": config.alpha,
            "beta": config.beta,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "max_len": config.max_len,
            "epochs": config.epochs,
            "patience": config.patience,
        })

        # Log class weights
        mlflow.log_params({
            "sentiment_class_weights": sentiment_weight.tolist(),
            "star_class_weights": star_weight.tolist(),
        })

        # ── Train ─────────────────────────────────────────────────────────
        result = train_model(
            config, model, loaders["train"], loaders["val"],
            sentiment_weight, star_weight, device,
        )

        # Log training history
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

        # ── Evaluate on validation set ────────────────────────────────────
        val_metrics = full_evaluation(
            model, loaders["val"], device, config, prefix="val"
        )
        mlflow.log_metrics(val_metrics)

        # Log best val loss
        mlflow.log_metric("best_val_loss", result["best_val_loss"])

        # ── Score ─────────────────────────────────────────────────────────
        score = (
            0.5 * val_metrics["val_sentiment_weighted_f1"]
            + 0.5 * val_metrics["val_star_weighted_f1"]
        )
        mlflow.log_metric("objective_score", score)

    return score


def run_optuna_study(
    n_trials: int = 20,
    config: Config | None = None,
) -> optuna.Study:
    """Create and run an Optuna study for hyperparameter tuning.

    Args:
        n_trials: number of trials.
        config: base config (trial will override tunable params).

    Returns:
        The completed Optuna study.
    """
    if config is None:
        config = Config()

    mlflow.set_experiment("multitask-bert-optuna")

    study = optuna.create_study(
        direction="maximize",
        study_name="multitask-bert-tuning",
        pruner=optuna.pruners.MedianPruner(),
    )

    def _objective(trial):
        return objective(trial, config)

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    # ── Report best trial ─────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best Trial #{best.number}")
    print(f"  Score: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")

    return study
