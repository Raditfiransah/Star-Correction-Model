"""
Multi-task BERT model with shared encoder and two classification heads.
"""

import torch.nn as nn
from transformers import BertModel


class MultiTaskBERT(nn.Module):
    """BERT-base with two independent linear heads for multi-task learning.

    Heads:
        - sentiment_head: 768 → num_sentiment_classes
        - star_head:      768 → num_star_classes
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiment_classes: int = 3,
        num_star_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiment_classes)
        self.star_head = nn.Linear(self.bert.config.hidden_size, num_star_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through shared encoder + both classification heads.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)

        Returns:
            sentiment_logits: (batch, num_sentiment_classes)
            star_logits:      (batch, num_star_classes)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.sentiment_head(pooled), self.star_head(pooled)
