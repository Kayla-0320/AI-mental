"""
模型定义 —— 基于 Chinese RoBERTa 的情感分类模型

使用 hfl/chinese-roberta-wwm-ext 预训练模型，
在 [CLS] 输出上接一个分类头做 5 类情感分类。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from perception.text_emotion.config import MODEL_NAME, NUM_LABELS


class TextEmotionClassifier(nn.Module):
    """中文文本情感分类模型

    结构：RoBERTa → [CLS] pooling → Dropout → Linear → softmax

    Args:
        num_labels: 分类类别数（默认 5）
        dropout_prob: Dropout 概率（默认 0.1）
    """

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播

        Args:
            input_ids: 形状 (batch_size, seq_len)
            attention_mask: 形状 (batch_size, seq_len)

        Returns:
            logits: 形状 (batch_size, num_labels)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled)
        return logits
