"""对缺失和质量显式建模的可学习多专家分类头。"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping

import torch
from torch import Tensor, nn


class QualityAwareMoE(nn.Module):
    """训练单模态、两两交互及联合专家；输入为冻结编码器的特征。

    top-k 仅限推理稀疏化权重，当前仍计算全部专家，不宣称加速；训练
    使用可微的 soft 路由，避免 top-1 归一化后路由梯度静默为零。知识分支
    可选，未检索到条目时其专家不可用；不以全零或中性向量冒充观测。
    """

    def __init__(self, dimensions: Mapping[str, int], classes: int, hidden: int = 32,
                 knowledge_dim: int = 0, top_k: int = 0) -> None:
        super().__init__()
        if not dimensions or any(type(d) is not int or d < 1 for d in dimensions.values()):
            raise ValueError("必须提供正整数特征维数")
        if classes < 2 or hidden < 2 or knowledge_dim < 0 or top_k < 0:
            raise ValueError("非法模型维度或路由参数")
        self.modalities = list(dimensions)
        self.dimensions = dict(dimensions)
        self.classes, self.hidden = classes, hidden
        self.knowledge_dim, self.top_k = knowledge_dim, top_k
        self.projections = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(dimensions[name], hidden), nn.LayerNorm(hidden), nn.GELU()
        ) for name in self.modalities})
        self.requirements = [(i,) for i in range(len(self.modalities))]
        self.requirements += list(combinations(range(len(self.modalities)), 2))
        if len(self.modalities) > 2:
            self.requirements.append(tuple(range(len(self.modalities))))
        self.expert_names = ["+".join(self.modalities[i] for i in req) for req in self.requirements]
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(len(req) * hidden, hidden), nn.GELU(), nn.Linear(hidden, classes))
            for req in self.requirements
        ])
        if knowledge_dim:
            self.knowledge_projection = nn.Sequential(nn.Linear(knowledge_dim, hidden), nn.GELU())
            self.knowledge_expert = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, classes))
            self.expert_names.append("candidate_knowledge")
        gate_dim = len(self.modalities) * (hidden + 2) + (hidden + 1 if knowledge_dim else 0)
        self.router = nn.Sequential(nn.Linear(gate_dim, hidden), nn.GELU(), nn.Linear(hidden, len(self.expert_names)))

    def forward(self, features: Mapping[str, Tensor], observed: Tensor, quality: Tensor,
                knowledge: Tensor | None = None, knowledge_present: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        """返回融合 logits、路由权重及可微负载正则；全缺失时拒绝预测。"""
        if self.training and self.top_k:
            raise ValueError("top-k 仅用于推理；请使用 top_k=0 的可微软路由训练")
        if observed.ndim != 2 or observed.shape != quality.shape or observed.shape[1] != len(self.modalities):
            raise ValueError("模态掩码和质量的形状不匹配")
        if observed.dtype != torch.bool or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("掩码必须是布尔值，质量必须在[0,1]")
        mask = observed & (quality > 0)
        if not mask.any(dim=1).all():
            raise ValueError("没有有效模态，必须拒绝或显式弃权")
        embeddings: list[Tensor] = []
        for i, name in enumerate(self.modalities):
            x = features[name]
            if x.shape != (observed.shape[0], self.dimensions[name]):
                raise ValueError(f"{name} 特征维数不匹配")
            if not torch.isfinite(x[mask[:, i]]).all():
                raise ValueError("有效观测包含非有限值")
            clean = torch.where(mask[:, i:i+1], x, torch.zeros_like(x))
            h = self.projections[name](clean) * mask[:, i:i+1] * quality[:, i:i+1]
            embeddings.append(h)
        eligible = [mask[:, req].all(dim=1) for req in self.requirements]
        expert_logits = [expert(torch.cat([embeddings[i] for i in req], dim=1))
                         for req, expert in zip(self.requirements, self.experts)]
        gate_input = [*embeddings, mask.to(quality.dtype), quality * mask]
        if self.knowledge_dim:
            if knowledge is None or knowledge_present is None:
                raise ValueError("启用知识分支时必须显式提供知识及命中掩码")
            if knowledge.shape != (mask.shape[0], self.knowledge_dim) or knowledge_present.shape != (mask.shape[0],):
                raise ValueError("知识输入维数不匹配")
            if knowledge_present.dtype != torch.bool or not torch.isfinite(knowledge).all():
                raise ValueError("非法知识观测")
            kh = self.knowledge_projection(knowledge) * knowledge_present[:, None]
            expert_logits.append(self.knowledge_expert(kh))
            eligible.append(knowledge_present)
            gate_input.extend([kh, knowledge_present[:, None].to(quality.dtype)])
        can_use = torch.stack(eligible, dim=1)
        gate_logits = self.router(torch.cat(gate_input, dim=1)).masked_fill(~can_use, -torch.inf)
        if self.top_k and self.top_k < gate_logits.shape[1]:
            indices = gate_logits.topk(self.top_k, dim=1).indices
            selected = torch.zeros_like(can_use).scatter_(1, indices, True) & can_use
            gate_logits = gate_logits.masked_fill(~selected, -torch.inf)
        weights = torch.softmax(gate_logits, dim=1)
        logits = (weights[:, :, None] * torch.stack(expert_logits, dim=1)).sum(dim=1)
        # 仅针对可用专家的边际机会均衡；缺失专家不被强迫接单。
        target = (can_use.to(weights.dtype) / can_use.sum(dim=1, keepdim=True)).mean(dim=0)
        load_balance = ((weights.mean(dim=0) - target).square()).sum() * weights.shape[1]
        return logits, weights, load_balance


class ConcatenationHead(nn.Module):
    """与 MoE 使用同一隐藏维数和掩码信息的拼接基线。"""

    def __init__(self, dimensions: Mapping[str, int], classes: int, hidden: int = 32,
                 knowledge_dim: int = 0) -> None:
        super().__init__()
        self.modalities = list(dimensions)
        self.dimensions = dict(dimensions)
        self.classes = classes
        self.knowledge_dim = knowledge_dim
        self.net = nn.Sequential(
            nn.Linear(sum(dimensions.values()) + len(dimensions) * 2 + knowledge_dim + bool(knowledge_dim), hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, classes)
        )

    def forward(self, features: Mapping[str, Tensor], observed: Tensor, quality: Tensor,
                knowledge: Tensor | None = None, knowledge_present: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        """遮蔽缺失值后拼接；返回与 MoE 相同结构供统一实验使用。"""
        if observed.ndim != 2 or observed.shape != quality.shape or observed.shape[1] != len(self.modalities):
            raise ValueError("模态掩码和质量的形状不匹配")
        if observed.dtype != torch.bool or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("掩码必须是布尔值，质量必须在[0,1]")
        mask = observed & (quality > 0)
        if not mask.any(dim=1).all():
            raise ValueError("全缺失样本不允许返回有效预测")
        for i, name in enumerate(self.modalities):
            if features[name].shape != (observed.shape[0], self.dimensions[name]):
                raise ValueError("特征维数不匹配")
            if not torch.isfinite(features[name][mask[:, i]]).all():
                raise ValueError("有效观测包含非有限值")
        values = [torch.where(mask[:, i:i+1], features[name], torch.zeros_like(features[name])) * quality[:, i:i+1]
                  for i, name in enumerate(self.modalities)]
        values += [mask.to(quality.dtype), quality * mask]
        if self.knowledge_dim:
            if knowledge is None or knowledge_present is None:
                raise ValueError("缺少知识输入")
            if knowledge.shape != (mask.shape[0], self.knowledge_dim) or knowledge_present.shape != (mask.shape[0],):
                raise ValueError("知识输入维数不匹配")
            if knowledge_present.dtype != torch.bool or not torch.isfinite(knowledge).all():
                raise ValueError("非法知识观测")
            values += [knowledge * knowledge_present[:, None], knowledge_present[:, None].to(quality.dtype)]
        logits = self.net(torch.cat(values, dim=1))
        return logits, torch.ones((mask.shape[0], 1), device=logits.device), logits.sum() * 0

