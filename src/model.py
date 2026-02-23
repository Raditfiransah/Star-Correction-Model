"""
Multi-task BERT model with star input embedding and feature fusion.

Architecture:
    Text → IndoBERT → [CLS] (768-dim)
    Star → Embedding(5, 16) → (16-dim)
    Concat(768 + 16) → Linear(784, 256) → ReLU → Dropout
    → sentiment_head: Linear(256 → 3)
    → star_head:      Linear(256 → 5)
"""

import torch
import torch.nn as nn
from transformers import BertModel


class MultiTaskBERT(nn.Module):
    """BERT with star embedding fusion and two classification heads.

    The original user star rating is embedded and concatenated with the
    BERT [CLS] representation before being passed through a fusion layer
    and the two task-specific classification heads.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiment_classes: int = 3,
        num_star_classes: int = 5,
        dropout: float = 0.3,
        star_embed_dim: int = 16,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # Star embedding: index 0–4 → star_embed_dim vector
        self.star_embedding = nn.Embedding(num_star_classes, star_embed_dim)

        # Fusion: concat(BERT [CLS], star_embed) → fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + star_embed_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.sentiment_head = nn.Linear(fusion_dim, num_sentiment_classes)
        self.star_head = nn.Linear(fusion_dim, num_star_classes)

    def forward(self, input_ids, attention_mask, star_input):
        """Forward pass through shared encoder + fusion + both heads.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)
            star_input:      (batch,) int tensor with values 0–4

        Returns:
            sentiment_logits: (batch, num_sentiment_classes)
            star_logits:      (batch, num_star_classes)
        """
        # BERT encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)  # (batch, 768)

        # Star embedding
        star_embed = self.star_embedding(star_input)  # (batch, star_embed_dim)

        # Feature fusion
        fused = torch.cat([pooled, star_embed], dim=1)  # (batch, 784)
        fused = self.fusion(fused)  # (batch, fusion_dim)

        return self.sentiment_head(fused), self.star_head(fused)
