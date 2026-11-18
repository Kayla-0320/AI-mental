"""第十二轮：显式缺失、知识/记忆专家与真正按样本分派的稀疏融合。"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import Tensor, nn


def evidence_inputs(evidence: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只取T/A/V的真实OOF证据，避免把干净联合专家用于缺失输入。"""
    if evidence.ndim != 2 or evidence.shape[1] != 40:
        raise ValueError("需要三分类40维专家证据")
    p = evidence[:, :12].reshape(-1, 4, 3)[:, 1:]
    active = evidence[:, 25:28] > 0
    features = np.concatenate((p, evidence[:, 13:16, None], evidence[:, 17:20, None],
                               evidence[:, 21:24, None], quality[:, :, None]), axis=2)
    return np.where(active[:, :, None], features, 0.).astype(np.float32), active


def change_inputs(x: np.ndarray, mask: np.ndarray, alignment: np.ndarray, knowledge: np.ndarray,
                  memory: np.ndarray, condition: str, donors: np.ndarray | None = None,
                  mismatch_alignment: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """修改真实可用性并物理移除缺失值；错配只移动音频，不触碰目标。"""
    x, mask, alignment, knowledge, memory = (v.copy() for v in (x, mask, alignment, knowledge, memory))
    missing = {"missing_T": (0,), "missing_A": (1,), "missing_V": (2,),
               "text_only": (1, 2), "all_missing": (0, 1, 2)}.get(condition, ())
    for j in missing:
        mask[:, j] = False
        x[:, j] = 0
    if condition == "audio_mismatch":
        if donors is None or mismatch_alignment is None:
            raise ValueError("错配必须提供供体与重新计算的对齐代理")
        x[:, 1], mask[:, 1], alignment = x[donors, 1].copy(), mask[donors, 1].copy(), mismatch_alignment.copy()
    elif condition not in ("clean", "missing_T", "missing_A", "missing_V", "text_only", "all_missing"):
        raise ValueError("未知压力条件")
    alignment *= (mask[:, 0] & mask[:, 1])[:, None]
    knowledge *= mask[:, :1]
    memory *= mask[:, :1]
    return x, mask, alignment, knowledge, memory


def cross_source_memory(query: np.ndarray, bank: np.ndarray, labels: np.ndarray,
                        query_groups: np.ndarray, bank_groups: np.ndarray,
                        query_texts: np.ndarray, bank_texts: np.ndarray, k: int = 5
                        ) -> tuple[np.ndarray, np.ndarray]:
    """仅检索训练来源，排除同来源与同原文；输出监督记忆而非外部知识图谱。"""
    if k < 1 or len(bank) != len(labels) or len(bank) != len(bank_groups):
        raise ValueError("记忆索引不完整")
    q = np.asarray(query, np.float64)
    b = np.asarray(bank, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    scores = q @ b.T
    same_text = query_texts[:, None] == bank_texts[None, :]
    scores[(query_groups[:, None] == bank_groups[None, :]) | same_text] = -np.inf
    neighbors = np.full((len(q), k), -1, dtype=int)
    result = np.zeros((len(q), 6), np.float32)
    for i in range(len(q)):
        selected = np.argsort(-scores[i], kind="stable")[:k]
        selected = selected[np.isfinite(scores[i, selected])]
        if not len(selected):
            continue
        neighbors[i, :len(selected)] = selected
        weights = np.exp((scores[i, selected] - scores[i, selected].max()) / .1)
        weights /= weights.sum()
        p = np.bincount(labels[selected], weights=weights, minlength=3)
        result[i] = np.r_[p, scores[i, selected].max(), -(p * np.log(np.maximum(p, 1e-12))).sum(), 1.]
    return result, neighbors


class EvidenceHead(nn.Module):
    """整体M分类头与原T/A/V专家任务分开；Top-2只执行被选中的头。"""

    def __init__(self, config: dict[str, Any], knowledge_dim: int, prior: np.ndarray) -> None:
        super().__init__()
        self.config = config
        self.use_k, self.use_r = bool(config.get("knowledge")), bool(config.get("memory"))
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
        self.input_dim = 21 + 3 + 3 + (knowledge_dim if self.use_k else 0) + (6 if self.use_r else 0)
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, 48), nn.GELU(), nn.Dropout(.1))
        self.classifier = nn.Linear(48, 3)
        count = 4 + int(self.use_k) + int(self.use_r)
        self.router = nn.Sequential(nn.Linear(self.input_dim, 32), nn.GELU(), nn.Linear(32, count))
        dims = [7, 7, 7, self.input_dim] + ([knowledge_dim] if self.use_k else []) + ([6] if self.use_r else [])
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 3)) for d in dims])

    def forward(self, x: Tensor, mask: Tensor, alignment: Tensor, knowledge: Tensor,
                memory: Tensor) -> tuple[Tensor, Tensor]:
        """缺失输入严格归零，全缺失返回训练先验，不生成中性伪观测。"""
        x = torch.where(mask[:, :, None], x, 0.)
        alignment = alignment * (mask[:, 0] & mask[:, 1])[:, None]
        knowledge, memory = knowledge * mask[:, :1], memory * mask[:, :1]
        parts = [x.flatten(1), mask.float(), alignment]
        if self.use_k:
            parts.append(knowledge)
        if self.use_r:
            parts.append(memory)
        features = torch.cat(parts, 1)
        any_input = mask.any(1)
        if self.config["mode"] == "mlp":
            p = self.classifier(self.encoder(features)).softmax(1)
            gate = torch.ones((len(x), 1), device=x.device)
        else:
            # 全缺失暂时开放joint，最终被训练先验覆盖，避免全-inf softmax。
            eligible = [mask, torch.ones((len(x), 1), dtype=torch.bool, device=x.device)]
            inputs = [x[:, j] for j in range(3)] + [features]
            if self.use_k:
                eligible.append((mask[:, 0] & (knowledge.abs().sum(1) > 0))[:, None])
                inputs.append(knowledge)
            if self.use_r:
                eligible.append((mask[:, 0] & (memory[:, -1] > 0))[:, None])
                inputs.append(memory)
            allowed = torch.cat(eligible, 1)
            scores = self.router(features).masked_fill(~allowed, -torch.inf)
            if self.config["mode"] == "top2":
                selected = torch.zeros_like(allowed).scatter_(1, scores.topk(2, 1).indices, True)
                scores = scores.masked_fill(~selected, -torch.inf)
            gate = scores.softmax(1)
            p = torch.zeros((len(x), 3), device=x.device)
            for j, expert in enumerate(self.experts):
                rows = torch.where(gate[:, j] > 0)[0]
                if len(rows):
                    p = p.index_add(0, rows, gate[rows, j:j+1] * expert(inputs[j][rows]).softmax(1))
        p = torch.where(any_input[:, None], p, self.prior[None])
        gate = gate * any_input[:, None]
        return p.clamp_min(1e-9).log(), gate
