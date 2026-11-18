"""第九轮·增补实验一：I²MoE / FuseMoE 对照 + 柔性判定 + 推理耗时。

对应 PPT 批注（cla，2026-11-15）：
- 00:11:40 "参考 I2MOE，FUSEMOE 模型……提升部署推理速度和预测准确性"
- 00:12:46 "阈值判定的方法比较差，可以考虑多专家投票和 top-k 的路由机制，做柔性判定"

协议（与第三轮九臂对齐，差异只在架构集合）：
- 数据：train 2,555 / selection 321（选隐藏维与早停）/ test 821（已被前几轮查看，仅作同协议诊断）
- 隐藏维：{32, 64} 逐臂逐种子搜索，选择集损失最小者胜
- 五个种子：42/43/44/45/46；优化器、类别权重、模态丢弃与加噪、早停与九臂完全一致
- 结果以均值±标准差报告；臂间比较用同种子配对差，不把不同种子混成一次 bootstrap

设计边界（必须连同结果一起读）：
1. I²MoE 与 FuseMoE 是按论文思路的**简化重实现**，不是官方代码，不与论文数字比较。
2. 输入是既有冻结特征（text 512 / audio 46 / visual 512），不重新编码。
3. 本轮不启用知识分支（knowledge_dim=0），知识增量由第四轮对照单独负责。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from experiments.round2_moe_train import load_split, metrics, seed_all, tensors
from perception.knowledge_moe.features import FeatureStandardizer
from perception.knowledge_moe.model import ConcatenationHead, QualityAwareMoE
from perception.knowledge_moe.sparse import SparseQualityAwareMoE

BUNDLE = Path(os.environ.get("ROUND9_BUNDLE", r"D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2"))
OUTPUT = Path(os.environ.get("ROUND9_OUTPUT", r"D:\Develop\AIC\_国赛优化\round9_additions"))
SEEDS = tuple(int(x) for x in os.environ.get("ROUND9_SEEDS", "42,43,44,45,46").split(","))
TAG = os.environ.get("ROUND9_TAG", "")
HIDDENS = tuple(int(x) for x in os.environ.get("ROUND9_HIDDENS", "32,64").split(","))
# 第五轮 text_features 实际来自冻结预训练编码器，不是监督微调的 finetune 目录。
# FIXED_EPOCHS 表示严格采用最后一轮；未设置时沿用独立selection选择。
FIXED_EPOCHS = int(os.environ.get("ROUND9_FIXED_EPOCHS", "0"))
if FIXED_EPOCHS and "ROUND9_HIDDENS" not in os.environ:
    HIDDENS = (64,)
ARMS = tuple(a for a in os.environ.get("ROUND9_ARMS", "concat,moe,sparse_moe,i2moe,fusemoe").split(",") if a)
SCENARIOS = ("clean", "without_text", "without_audio", "without_visual", "random_missing")
UNKNOWN_LABEL = -1
CANONICAL = ("text", "audio", "visual", "face")


def modality_names(data: dict) -> list[str]:
    """按固定顺序取本特征包实际具备的模态，三模态包与四模态包共用同一套训练/评测。"""
    return [name for name in CANONICAL if name in data["train"]]


def _projections(dimensions: Mapping[str, int], hidden: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        name: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        for name, dim in dimensions.items()
    })


class I2MoE(nn.Module):
    """单模态专家 + 两两交互专家 + 学习门控 + 缺失掩码（简化重实现）。

    与 `QualityAwareMoE` 的差别：**不含三模态联合专家**，跨模态能力全部由
    两两交互专家承担，与参考实现的交互专家设计同向。
    """

    def __init__(self, dimensions: Mapping[str, int], classes: int, hidden: int = 32,
                 knowledge_dim: int = 0, top_k: int = 0) -> None:
        super().__init__()
        if classes < 2 or hidden < 2:
            raise ValueError("非法模型维度")
        self.modalities = list(dimensions)
        self.classes, self.hidden, self.top_k = classes, hidden, top_k
        self.projections = _projections(dimensions, hidden)
        self.requirements = [(i,) for i in range(len(self.modalities))]
        self.requirements += list(combinations(range(len(self.modalities)), 2))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(len(req) * hidden, hidden), nn.GELU(), nn.Linear(hidden, classes))
            for req in self.requirements
        ])
        gate_dim = len(self.modalities) * (hidden + 2)
        self.router = nn.Sequential(nn.Linear(gate_dim, hidden), nn.GELU(), nn.Linear(hidden, len(self.requirements)))

    def forward(self, features: Mapping[str, Tensor], observed: Tensor, quality: Tensor,
                knowledge: Tensor | None = None, knowledge_present: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        if observed.ndim != 2 or observed.shape != quality.shape or observed.shape[1] != len(self.modalities):
            raise ValueError("模态掩码和质量的形状不匹配")
        if observed.dtype != torch.bool or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("掩码必须是布尔值，质量必须有限且在[0,1]")
        mask = observed & (quality > 0)
        if not mask.any(dim=1).all():
            raise ValueError("没有有效模态，必须拒绝或显式弃权")
        emb, gate_in = [], []
        for i, name in enumerate(self.modalities):
            if features[name].shape != (observed.shape[0], self.projections[name][0].in_features):
                raise ValueError(f"{name} 特征维数不匹配")
            if not torch.isfinite(features[name][mask[:, i]]).all():
                raise ValueError("有效观测含非有限值")
            x = torch.where(mask[:, i:i + 1], features[name], torch.zeros_like(features[name]))
            h = self.projections[name](x) * mask[:, i:i + 1] * quality[:, i:i + 1]
            emb.append(h)
            gate_in.extend([h, mask[:, i:i + 1].to(quality.dtype), (quality * mask)[:, i:i + 1]])
        eligible = torch.stack([mask[:, req].all(dim=1) for req in self.requirements], dim=1)
        logits = [expert(torch.cat([emb[i] for i in req], dim=1))
                  for req, expert in zip(self.requirements, self.experts)]
        gate_logits = self.router(torch.cat(gate_in, dim=1)).masked_fill(~eligible, -torch.inf)
        weights = torch.softmax(gate_logits, dim=1)
        fused = (weights[:, :, None] * torch.stack(logits, dim=1)).sum(dim=1)
        target = (eligible.to(weights.dtype) / eligible.sum(dim=1, keepdim=True)).mean(dim=0)
        balance = ((weights.mean(dim=0) - target).square()).sum() * weights.shape[1]
        return fused, weights, balance


class FuseMoE(nn.Module):
    """Laplace 门控的专家混合（简化重实现）。

    对MLP输出构造Laplace形状的logit ``-|a_i|/b_i``，再稳定归一化。
    ``b`` 逐专家可学。这不等同于原论文的输入至专家原型距离门控。
    """

    def __init__(self, dimensions: Mapping[str, int], classes: int, hidden: int = 32,
                 knowledge_dim: int = 0, top_k: int = 0) -> None:
        super().__init__()
        if classes < 2 or hidden < 2:
            raise ValueError("非法模型维度")
        self.modalities = list(dimensions)
        self.classes, self.hidden, self.top_k = classes, hidden, top_k
        self.projections = _projections(dimensions, hidden)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, classes))
            for _ in self.modalities
        ])
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, classes))
        self.router = nn.Sequential(nn.Linear(len(self.modalities) * (hidden + 2), hidden), nn.GELU(),
                                    nn.Linear(hidden, len(self.modalities) + 1))
        self.log_b = nn.Parameter(torch.zeros(len(self.modalities) + 1))

    def forward(self, features: Mapping[str, Tensor], observed: Tensor, quality: Tensor,
                knowledge: Tensor | None = None, knowledge_present: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        if observed.ndim != 2 or observed.shape != quality.shape or observed.shape[1] != len(self.modalities):
            raise ValueError("模态掩码和质量的形状不匹配")
        if observed.dtype != torch.bool or not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("掩码必须是布尔值，质量必须有限且在[0,1]")
        mask = observed & (quality > 0)
        if not mask.any(dim=1).all():
            raise ValueError("没有有效模态，必须拒绝或显式弃权")
        emb, gate_in = [], []
        for i, name in enumerate(self.modalities):
            if features[name].shape != (observed.shape[0], self.projections[name][0].in_features):
                raise ValueError(f"{name} 特征维数不匹配")
            if not torch.isfinite(features[name][mask[:, i]]).all():
                raise ValueError("有效观测含非有限值")
            x = torch.where(mask[:, i:i + 1], features[name], torch.zeros_like(features[name]))
            h = self.projections[name](x) * mask[:, i:i + 1] * quality[:, i:i + 1]
            emb.append(h)
            gate_in.extend([h, mask[:, i:i + 1].to(quality.dtype), (quality * mask)[:, i:i + 1]])
        logits = torch.stack([head(emb[i]) for i, head in enumerate(self.heads)] +
                             [self.shared(torch.stack(emb, dim=1).sum(dim=1))], dim=1)
        eligible = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        a = self.router(torch.cat(gate_in, dim=1))
        scale = torch.exp(self.log_b).clamp(min=1e-2)
        # 在对数域归一化，避免有限但较大的路由值使所有指数下溢为零。
        weights = torch.softmax((-a.abs() / scale).masked_fill(~eligible, -torch.inf), dim=1)
        fused = (weights[:, :, None] * logits).sum(dim=1)
        target = (eligible.to(weights.dtype) / eligible.sum(dim=1, keepdim=True)).mean(dim=0)
        balance = ((weights.mean(dim=0) - target).square()).sum() * weights.shape[1]
        return fused, weights, balance


def model_for(arm: str, dimensions: Mapping[str, int], classes: int, hidden: int) -> nn.Module:
    """本轮五臂：拼接、联合专家、稀疏专家、I²MoE、FuseMoE。"""
    if arm == "concat":
        return ConcatenationHead(dimensions, classes, hidden)
    if arm == "moe":
        return QualityAwareMoE(dimensions, classes, hidden)
    if arm == "sparse_moe":
        return SparseQualityAwareMoE(dimensions, classes, hidden, 0, top_k=2)
    if arm == "i2moe":
        return I2MoE(dimensions, classes, hidden)
    if arm == "fusemoe":
        return FuseMoE(dimensions, classes, hidden)
    raise ValueError(f"未知实验臂 {arm}")


def forward(model: nn.Module, batch: tuple, indices: Tensor, augment: bool = False,
            scenario: str = "clean") -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """与九臂同构：同一套模态丢弃 / 加噪 / 缺失场景。"""
    x, mask, q, _kg, _km, _y = batch
    names = list(x)
    x = {k: v[indices].clone() for k, v in x.items()}
    mask, q = mask[indices].clone(), q[indices].clone()
    if augment:
        drop = torch.rand_like(q) < .15
        new = mask & ~drop
        new[~new.any(dim=1)] = mask[~new.any(dim=1)]
        mask = new
        for j, name in enumerate(names):
            corrupt = (torch.rand(len(indices)) < .1) & mask[:, j]
            x[name][corrupt] += torch.randn_like(x[name][corrupt]) * .5
            q[corrupt, j] = .25
    elif scenario.startswith("without_"):
        missing = scenario.removeprefix("without_")
        if missing in names:
            mask[:, names.index(missing)] = False
    elif scenario.startswith("noisy_") and scenario.endswith("_known_quality"):
        target = scenario.removeprefix("noisy_").removesuffix("_known_quality")
        if target in names:
            j = names.index(target)
            rng = np.random.default_rng(42)
            noise = torch.tensor(rng.normal(0, 2, tuple(x[target].shape)), dtype=torch.float32)
            x[target] += noise
            q[:, j] = .25
    elif scenario == "random_missing":
        rng = np.random.default_rng(42)
        mask &= torch.tensor(rng.random(tuple(mask.shape)) >= .4)
    valid = (mask & (q > 0)).any(dim=1)
    if not valid.any():
        return torch.empty(0, model.classes), torch.empty(0, 0), torch.tensor(0.), valid
    logits, gates, balance = model({k: v[valid] for k, v in x.items()}, mask[valid], q[valid])
    return logits, gates, balance, valid


class _NullKnowledge:
    """本轮不启用知识分支；提供与 tensors() 兼容的空知识接口。"""

    dimensions = 0

    def __init__(self, _dimensions: Mapping[str, int] | None = None) -> None:
        self.dimensions = 0

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(texts), 1), dtype=np.float32), np.zeros(len(texts), dtype=bool)


def _uncertainty(probs: np.ndarray, gates: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """两种不确定性：预测置信度（现行做法）与门控熵（柔性判定）。"""
    confidence = probs.max(axis=1)
    if gates is not None and gates.size and gates.shape[1] > 1 and len(gates) == len(probs):
        g = np.clip(gates, 1e-12, 1.0)
        entropy = -(g * np.log(g)).sum(axis=1)
    else:
        p = np.clip(probs, 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=1)
    return confidence, entropy


def _coverage_curve(labels: np.ndarray, probs: np.ndarray, uncertainty: np.ndarray,
                    higher_is_more_uncertain: bool) -> list[dict]:
    order = np.argsort(uncertainty if higher_is_more_uncertain else -uncertainty)
    curve = []
    for coverage in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = max(1, int(round(coverage * len(labels))))
        ix = order[:k]
        pred = probs[ix].argmax(axis=1)
        curve.append({"coverage": round(k / len(labels), 3),
                      "accuracy": float((pred == labels[ix]).mean())})
    return curve


def evaluate(model: nn.Module, batch: tuple, scenario: str, labels: np.ndarray) -> dict:
    """单一条件评测：指标 + 两种不确定性下的覆盖率-准确率曲线。"""
    model.eval()
    with torch.inference_mode():
        logits, gates, _, valid = forward(model, batch, torch.arange(len(labels)), scenario=scenario)
    y = labels[valid.numpy()]
    if len(y) == 0:
        return {"scenario": scenario, "n": 0}
    probs = torch.softmax(logits, dim=1).numpy()
    result = {"scenario": scenario, **metrics(y, logits.numpy(), model.classes)}
    result.pop("nll", None), result.pop("ece", None), result.pop("brier", None)
    confidence, gate_entropy = _uncertainty(probs, gates.numpy() if gates.numel() else None)
    result["confidence_curve"] = _coverage_curve(y, probs, confidence, higher_is_more_uncertain=False)
    result["gate_entropy_curve"] = _coverage_curve(y, probs, gate_entropy, higher_is_more_uncertain=True)
    return result


def _tensors(data: dict, names: list[str], scaler: FeatureStandardizer) -> tuple:
    """把本特征包转成统一张量；支持三模态与四模态（含 face）。

    第二轮 `tensors()` 的 canonical 只认 text/audio/visual，这里按 CANONICAL 取列，
    避免"人脸作为第四路"时列索引错位。
    """
    columns = [CANONICAL.index(name) for name in names]
    mask = np.asarray(data["modality_mask"][:, columns], dtype=bool)
    quality = data["quality"][:, columns].astype(np.float32)
    x = scaler.transform({name: data[name] for name in names}, mask)
    labels = torch.tensor(data["labels"].reshape(-1), dtype=torch.long)
    empty_knowledge = torch.zeros(len(labels), 1)
    empty_present = torch.zeros(len(labels), dtype=torch.bool)
    return ({name: torch.tensor(value) for name, value in x.items()},
            torch.tensor(mask), torch.tensor(quality), empty_knowledge, empty_present, labels)


def train_once(arm: str, seed: int, hidden: int, data: dict, classes: int,
               epochs: int = 60, patience: int = 10,
               use_last_checkpoint: bool = False) -> tuple[nn.Module, dict, dict]:
    """单次训练：同种子、同优化器、同早停、同类别权重。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = modality_names(data)
    dimensions = {name: data["train"][name].shape[1] for name in names}
    columns = [CANONICAL.index(name) for name in names]
    scaler = FeatureStandardizer().fit({name: data["train"][name] for name in names},
                                      data["train"]["modality_mask"][:, columns])
    batches = {split: _tensors(d, names, scaler) for split, d in data.items()}
    model = model_for(arm, dimensions, classes, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    y = batches["train"][-1]
    counts = torch.bincount(y, minlength=classes).clamp(min=1)
    criterion = nn.CrossEntropyLoss(weight=len(y) / (classes * counts.float()))
    best = {"loss": float("inf"), "state": None, "epoch": 0, "stale": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            ix = order[i:i + 64]
            logits, _, balance, valid = forward(model, batches["train"], ix, augment=True)
            if not valid.any():
                continue
            loss = criterion(logits, y[ix][valid]) + .01 * balance
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            logits, _, _, valid = forward(model, batches["selection"],
                                          torch.arange(len(data["selection"]["labels"])))
            selection_loss = float(nn.functional.cross_entropy(logits, batches["selection"][-1][valid]))
        if use_last_checkpoint or selection_loss < best["loss"] - 1e-5:
            best = {"loss": selection_loss, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch, "stale": 0}
        else:
            best["stale"] += 1
            if best["stale"] >= patience:
                break
    if best["state"] is None:
        raise RuntimeError(f"{arm} seed={seed} hidden={hidden} 训练未产出有效模型")
    model.load_state_dict(best["state"])
    return model, {"seed": seed, "hidden": hidden, "epoch": best["epoch"],
                   "selection_loss": round(best["loss"], 5)}, batches


def _infer_seconds(model: nn.Module, batch: tuple, repeats: int = 3) -> float:
    """单次全测试集前向耗时（毫秒，中位数），对应批注里的"部署推理速度"。"""
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter()
            forward(model, batch, torch.arange(len(batch[-1]), dtype=torch.long))
            times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 2)


def main() -> None:
    if (OUTPUT / f"moe_baselines{TAG}.json").exists():
        raise ValueError("拒绝覆盖已完成实验；请设置新的ROUND9_OUTPUT或ROUND9_TAG")
    if FIXED_EPOCHS and len(HIDDENS) != 1:
        raise ValueError("固定轮次实验要求显式单一隐藏维")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {split: load_split(BUNDLE / f"{split}.npz") for split in ("train", "selection", "test")}
    classes = 3
    labels = data["test"]["labels"]
    names = modality_names(data)
    scenarios = ("clean",) + tuple(f"without_{name}" for name in names) + ("random_missing",)
    report: dict[str, Any] = {
        "schema": "round9_moe_baselines/2",
        "bundle": str(BUNDLE),
        "classes": ["negative", "neutral", "positive"],
        "seeds": list(SEEDS),
        "protocol": {
            "train": len(data["train"]["labels"]),
            "selection": len(data["selection"]["labels"]),
            "test": len(labels),
            "hidden_grid": list(HIDDENS),
            "selection_rule": "固定最后一轮" if FIXED_EPOCHS else "selection 交叉熵最小",
            "optimizer": "AdamW lr=1e-3 wd=1e-3",
            "epochs_max": FIXED_EPOCHS or 60,
            "patience": None if FIXED_EPOCHS else 10,
            "augment": "15% 模态丢弃 + 10% 特征加噪（与九臂一致）",
            "knowledge": "本轮关闭（knowledge_dim=0）",
            "modalities": names,
        },
        "limitations": [
            "I²MoE 与 FuseMoE 为按论文思路的简化重实现，不是官方代码，不与论文数字比较",
            "测试集已被前几轮查看，属同协议诊断对照，不是全新外部验证",
            "视觉为冻结 CLIP 三帧均值，声音为 46 维手工特征，不是端到端 VLM",
            "五个种子只覆盖初始化与批次顺序，不覆盖数据划分",
        ],
        "arms": {},
    }
    per_seed: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        report["arms"][arm] = {"runs": [], "clean": {}, "scenarios": {}, "infer_ms": {}}
        for seed in SEEDS:
            best = None
            for hidden in HIDDENS:
                seed_all()
                model, info, batches = train_once(arm, seed, hidden, data, classes,
                                                  epochs=FIXED_EPOCHS or 60,
                                                  patience=10 ** 9 if FIXED_EPOCHS else 10,
                                                  use_last_checkpoint=bool(FIXED_EPOCHS))
                if best is None or info["selection_loss"] < best[1]["selection_loss"]:
                    best = (model, info, batches)
            model, info, batches = best
            scenario_results = {s: evaluate(model, batches["test"], s, labels) for s in scenarios}
            infer_ms = _infer_seconds(model, batches["test"])
            params = round(sum(p.numel() for p in model.parameters()) / 1000, 1)
            clean = scenario_results["clean"]
            report["arms"][arm]["runs"].append({
                **info, "params_k": params, "infer_ms": infer_ms,
                "clean_macro_f1": clean["macro_f1"], "clean_accuracy": clean["accuracy"],
                "clean_per_class_recall": clean["per_class_recall"],
                "scenarios": {s: {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"]}
                              for s, r in scenario_results.items()},
                "confidence_curve": clean["confidence_curve"],
                "gate_entropy_curve": clean["gate_entropy_curve"],
            })
            per_seed[arm].append({"seed": seed, "macro_f1": clean["macro_f1"],
                                  "accuracy": clean["accuracy"]})
        runs = report["arms"][arm]["runs"]
        f1s = [r["clean_macro_f1"] for r in runs]
        accs = [r["clean_accuracy"] for r in runs]
        report["arms"][arm]["clean"] = {
            "macro_f1_mean": round(statistics.mean(f1s), 4),
            "macro_f1_std": round(statistics.pstdev(f1s), 4),
            "accuracy_mean": round(statistics.mean(accs), 4),
            "accuracy_std": round(statistics.pstdev(accs), 4),
            "macro_f1_values": [round(x, 4) for x in f1s],
        }
        report["arms"][arm]["infer_ms"] = {
            "median": round(statistics.median([r["infer_ms"] for r in runs]), 2),
            "values": [r["infer_ms"] for r in runs],
        }
        for scenario in scenarios:
            vals = [r["scenarios"][scenario]["macro_f1"] for r in runs]
            report["arms"][arm]["scenarios"][scenario] = {
                "macro_f1_mean": round(statistics.mean(vals), 4),
                "macro_f1_std": round(statistics.pstdev(vals), 4),
            }
        print(f"{arm:11s} macroF1={report['arms'][arm]['clean']['macro_f1_mean']:.4f}"
              f"±{report['arms'][arm]['clean']['macro_f1_std']:.4f} "
              f"acc={report['arms'][arm]['clean']['accuracy_mean']:.4f} "
              f"infer={report['arms'][arm]['infer_ms']['median']}ms")

    # 同种子配对差（不混种子做 bootstrap）
    base = {r["seed"]: r["macro_f1"] for r in per_seed["concat"]}
    report["comparisons"] = {}
    for arm in ARMS[1:]:
        deltas = [round(r["macro_f1"] - base[r["seed"]], 4) for r in per_seed[arm]]
        report["comparisons"][f"{arm}_minus_concat"] = {
            "mean_delta": round(statistics.mean(deltas), 4),
            "per_seed": deltas,
            "wins": sum(1 for d in deltas if d > 0),
            "n": len(deltas),
        }
    (OUTPUT / f"moe_baselines{TAG}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    (OUTPUT / "source_shas.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": list(SEEDS),
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nwritten -> {OUTPUT}/moe_baselines{TAG}.json")


if __name__ == "__main__":
    main()
