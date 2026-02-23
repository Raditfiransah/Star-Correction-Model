"""
Dataset loading, splitting, and class weight computation for multi-task BERT.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .config import Config


# ── Label Mapping ─────────────────────────────────────────────────────────────

SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def star_to_index(star: float) -> int:
    """Convert corrected_star (1.0–5.0) → index (0–4)."""
    return int(star) - 1


# ── Dataset Class ─────────────────────────────────────────────────────────────


class MultiTaskDataset(Dataset):
    """PyTorch dataset for multi-task sentiment + star classification."""

    def __init__(
        self,
        texts: list[str],
        sentiment_labels: list[int],
        star_labels: list[int],
        star_inputs: list[int],
        tokenizer: BertTokenizer,
        max_len: int,
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.star_labels = star_labels
        self.star_inputs = star_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(
                self.sentiment_labels[idx], dtype=torch.long
            ),
            "star_label": torch.tensor(self.star_labels[idx], dtype=torch.long),
            "star_input": torch.tensor(self.star_inputs[idx], dtype=torch.long),
        }


# ── Data Loading & Splitting ─────────────────────────────────────────────────


def load_and_split_data(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV, clean, map labels, and perform stratified train/val/test split.

    Returns:
        (train_df, val_df, test_df) each with columns:
            text, sentiment_idx, star_idx
    """
    df = pd.read_csv(config.data_path)

    # Drop rows with missing target columns
    df = df.dropna(subset=[
        config.text_column, config.sentiment_column,
        config.star_column, config.star_input_column,
    ])

    # Keep only known sentiment labels
    df = df[df[config.sentiment_column].isin(SENTIMENT_MAP.keys())].copy()

    # Map labels to indices
    df["sentiment_idx"] = df[config.sentiment_column].map(SENTIMENT_MAP)
    df["star_idx"] = df[config.star_column].apply(star_to_index)
    df["star_input_idx"] = df[config.star_input_column].apply(star_to_index)

    # Validate star range
    df = df[df["star_idx"].between(0, 4) & df["star_input_idx"].between(0, 4)].copy()

    # Optional sample for smoke tests
    if config.sample_size is not None:
        df = df.sample(n=min(config.sample_size, len(df)), random_state=config.seed)

    # Create a combined stratification key
    df["strat_key"] = df["sentiment_idx"].astype(str) + "_" + df["star_idx"].astype(str)

    # First split: train vs (val + test)
    val_test_ratio = config.val_ratio + config.test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=config.seed,
        stratify=df["strat_key"],
    )

    # Second split: val vs test
    relative_test_ratio = config.test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=config.seed,
        stratify=temp_df["strat_key"],
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ── Class Weight Computation ─────────────────────────────────────────────────


def compute_weights(labels: np.ndarray, num_classes: int) -> torch.FloatTensor:
    """Compute balanced class weights from label distribution.

    Args:
        labels: 1-D array of integer class indices.
        num_classes: total number of classes.

    Returns:
        FloatTensor of shape (num_classes,) with balanced weights.
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.FloatTensor(weights)


# ── DataLoader Builder ────────────────────────────────────────────────────────


def build_dataloaders(
    config: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
) -> dict:
    """Build DataLoaders from split DataFrames.

    Returns:
        dict with keys 'train', 'val', and optionally 'test'.
    """
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    def _make_dataset(df: pd.DataFrame) -> MultiTaskDataset:
        return MultiTaskDataset(
            texts=df[config.text_column].tolist(),
            sentiment_labels=df["sentiment_idx"].tolist(),
            star_labels=df["star_idx"].tolist(),
            star_inputs=df["star_input_idx"].tolist(),
            tokenizer=tokenizer,
            max_len=config.max_len,
        )

    loaders = {
        "train": torch.utils.data.DataLoader(
            _make_dataset(train_df),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            _make_dataset(val_df),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }

    if test_df is not None:
        loaders["test"] = torch.utils.data.DataLoader(
            _make_dataset(test_df),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    return loaders
