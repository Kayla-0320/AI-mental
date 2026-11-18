"""实际按样本分派的稀疏融合头；编码器仍独立执行，不虚报整体加速。"""
from __future__ import annotations
from typing import Mapping
import torch
from torch import Tensor
from perception.knowledge_moe.model import QualityAwareMoE


class SparseQualityAwareMoE(QualityAwareMoE):
    """只执行每个样本被选中的专家；训练使用top-2和稠密负载辅助梯度。"""

    def __init__(self, dimensions: Mapping[str, int], classes: int, hidden: int = 32,
                 knowledge_dim: int = 0, top_k: int = 2) -> None:
        if top_k < 1:
            raise ValueError("稀疏分派至少选择一个专家")
        super().__init__(dimensions, classes, hidden, knowledge_dim, top_k=0)
        self.dispatch_k = top_k
        self.last_dispatch_counts: list[int] = []

    def forward(self, features: Mapping[str, Tensor], observed: Tensor, quality: Tensor,
                knowledge: Tensor | None = None, knowledge_present: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather被选样本、执行专家并scatter加权输出，未选专家不运行。"""
        if self.training and self.dispatch_k < 2:
            raise ValueError("top-1训练无分类路由梯度；请使用top-2训练")
        if observed.ndim != 2 or observed.shape != quality.shape or observed.shape[1] != len(self.modalities):
            raise ValueError("模态掩码/质量维数不匹配")
        if observed.dtype != torch.bool or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("非法观测或质量")
        mask = observed & (quality > 0)
        if not mask.any(dim=1).all():
            raise ValueError("没有有效模态，必须显式弃权")
        embeddings = []
        for j, name in enumerate(self.modalities):
            x = features[name]
            if x.shape != (len(mask), self.dimensions[name]) or not torch.isfinite(x[mask[:, j]]).all():
                raise ValueError("非法特征维度或非有限观测")
            clean = torch.where(mask[:, j:j+1], x, torch.zeros_like(x))
            embeddings.append(self.projections[name](clean) * mask[:, j:j+1] * quality[:, j:j+1])
        eligible = [mask[:, req].all(dim=1) for req in self.requirements]
        gate_input = [*embeddings, mask.to(quality.dtype), quality*mask]
        kh = None
        if self.knowledge_dim:
            if knowledge is None or knowledge_present is None or knowledge.shape != (len(mask), self.knowledge_dim) or knowledge_present.shape != (len(mask),):
                raise ValueError("缺少对齐的知识输入")
            if knowledge_present.dtype != torch.bool or not torch.isfinite(knowledge).all():
                raise ValueError("非法知识观测")
            kh = self.knowledge_projection(knowledge) * knowledge_present[:, None]
            eligible.append(knowledge_present)
            gate_input += [kh, knowledge_present[:, None].to(quality.dtype)]
        can_use = torch.stack(eligible, dim=1)
        scores = self.router(torch.cat(gate_input, dim=1)).masked_fill(~can_use, -torch.inf)
        dense_weights = torch.softmax(scores, dim=1)
        k = min(self.dispatch_k, len(self.expert_names))
        selected = torch.zeros_like(can_use).scatter_(1, scores.topk(k, dim=1).indices, True) & can_use
        weights = torch.softmax(scores.masked_fill(~selected, -torch.inf), dim=1)
        result = torch.zeros((len(mask), self.classes), device=quality.device, dtype=quality.dtype)
        counts = []
        for j, req in enumerate(self.requirements):
            rows = torch.where(selected[:, j])[0]
            counts.append(len(rows))
            if len(rows):
                # 关键：没有为其它行构造专家输入或执行网络。
                z = torch.cat([embeddings[i][rows] for i in req], dim=1)
                result = result.index_add(0, rows, self.experts[j](z) * weights[rows, j:j+1])
        if self.knowledge_dim:
            rows = torch.where(selected[:, -1])[0]
            counts.append(len(rows))
            if len(rows) and kh is not None:
                result = result.index_add(0, rows, self.knowledge_expert(kh[rows]) * weights[rows, -1:])
        self.last_dispatch_counts = counts
        target = (can_use.to(quality.dtype) / can_use.sum(dim=1, keepdim=True)).mean(dim=0)
        balance = (dense_weights.mean(dim=0) - target).square().sum() * len(self.expert_names)
        return result, weights, balance
