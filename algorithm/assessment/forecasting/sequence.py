"""小样本情绪序列的残差 GRU 对照模型。"""
from __future__ import annotations

import torch
from torch import nn


class SmallEmotionGRU(nn.Module):
    """消费过去八个正负情绪观测，输出下一目标时刻的两维自评。"""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 2)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """从最后隐状态学习末值预测的修正量。"""
        _, hidden = self.gru(sequence)
        return sequence[:, -1] + self.head(hidden[-1])
