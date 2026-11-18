"""模态替换的路由软目标正则；合成样本没有整体情绪硬金标。

第十轮修正：删除“捐赠者整体M标签就是混合样本标签”的断言。
教师分别观察源完整样本和捐赠者被替换的单模态，用停止梯度的路由权重混合
两份软预测，再约束混合输入输出。它是有假设的自蒸馏，不是已识别的因果效应。
历史第九轮硬目标实验仍保留原存档，本实现不用于重新解释其旧数字。
"""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from experiments.round2_moe_train import load_split, metrics, seed_all, tensors
from experiments.round9_moe_baselines import train_once as train_clean
from experiments.round9_moe_baselines import (
    OUTPUT, _NullKnowledge, _infer_seconds, _projections, model_for,
)
from perception.knowledge_moe.features import FeatureStandardizer

BUNDLE = Path(r"D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2")
SEEDS = (42, 43, 44)
HIDDENS = (32, 64)
MODALITIES = ("text", "audio", "visual")
ARMS = (("moe", False), ("moe", True), ("i2moe", False), ("i2moe", True))


def _eligibility(requirements: list[tuple[int, ...]], modality: int) -> Tensor:
    """哪些专家"包含"该模态：用于计算路由给这一路的权重和。"""
    return torch.tensor([modality in req for req in requirements], dtype=torch.bool)


def _gate_mass(model: nn.Module, gates: Tensor, modality: int) -> Tensor:
    """逐样本：路由到"包含该模态的专家"的权重和。"""
    if not hasattr(model, "requirements"):
        return torch.zeros(gates.shape[0])
    keep = _eligibility(model.requirements, modality).to(gates.device)
    return gates[:, keep].sum(dim=1)


class CounterfactualSampler:
    """同来源组、不同标签的捐赠者抽样；找不到就不造反事实（不跨组建假对）。"""

    def __init__(self, data: dict, seed: int) -> None:
        self.groups = data["train"]["source_groups"]
        self.labels = data["train"]["labels"]
        self.rng = np.random.default_rng(seed)
        by_group: dict[str, dict[int, list[int]]] = {}
        for index, (group, label) in enumerate(zip(self.groups, self.labels)):
            by_group.setdefault(group, {}).setdefault(int(label), []).append(index)
        self.pairs: dict[int, list[int]] = {}
        for group, per_label in by_group.items():
            if len(per_label) < 2:
                continue
            for label, members in per_label.items():
                others = [i for other, idx in per_label.items() if other != label for i in idx]
                if others:
                    for member in members:
                        self.pairs[member] = others

    def sample(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """返回（捐赠者索引, 被替换模态, 是否构造成对）。"""
        donors = torch.full_like(indices, -1)
        modality = torch.randint(0, len(MODALITIES), (len(indices),), generator=None)
        ok = torch.zeros(len(indices), dtype=torch.bool)
        for position, index in enumerate(indices.tolist()):
            candidates = self.pairs.get(index)
            if not candidates:
                continue
            donors[position] = int(self.rng.choice(candidates))
            ok[position] = True
        return donors, modality, ok


def forward_with_swap(model: nn.Module, batch: tuple, indices: Tensor, donors: Tensor,
                      modality: Tensor, ok: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """只交换双方均可观测的模态，质量值随捐赠者观测一起转移。"""
    x, mask, q, _kg, _km, _y = batch
    names = list(x)
    cf = {name: value[indices].clone() for name, value in x.items()}
    cf_quality = q[indices].clone()
    for position, (donor, m) in enumerate(zip(donors.tolist(), modality.tolist())):
        if donor < 0:
            continue
        name = names[m]
        cf[name][position] = x[name][donor]
        cf_quality[position, m] = q[donor, m]
    valid = (mask[indices] & (q[indices] > 0)).any(dim=1) & ok & (donors >= 0)
    for position in torch.where(valid)[0].tolist():
        m, donor = int(modality[position]), int(donors[position])
        valid[position] &= mask[indices[position], m] & mask[donor, m] & (q[indices[position], m] > 0) & (q[donor, m] > 0)
    if not valid.any():
        return torch.empty(0, model.classes), torch.empty(0, 0), torch.empty(0), valid
    logits, gates, _ = model({k: v[valid] for k, v in cf.items()}, mask[indices][valid], cf_quality[valid])
    return logits, gates, modality[valid], valid


def soft_counterfactual_target(source_logits: Tensor, donor_logits: Tensor, gate_mass: Tensor) -> Tensor:
    """源与捐赠者单模态教师形成软代理目标，教师和路由均停止梯度。"""
    if source_logits.shape != donor_logits.shape or gate_mass.shape != (len(source_logits),):
        raise ValueError("软目标教师、路由或行数不一致")
    weight = gate_mass.detach().clamp(0., 1.)[:, None]
    return (1-weight)*source_logits.detach().softmax(1)+weight*donor_logits.detach().softmax(1)


def unimodal_teacher(model: nn.Module, batch: tuple, donors: Tensor, modality: Tensor) -> Tensor:
    """捐赠教师只可见被替换模态，不能偷看捐赠者的其他情绪线索。"""
    x, mask, quality, _kg, _km, _labels = batch
    observed = torch.zeros_like(mask[donors])
    observed.scatter_(1, modality[:, None], True)
    observed &= mask[donors] & (quality[donors] > 0)
    if not observed.any(1).all():
        raise ValueError("单模态教师含未观测输入")
    q = torch.where(observed, quality[donors], torch.zeros_like(quality[donors]))
    logits, _, _ = model({name: value[donors] for name, value in x.items()}, observed, q)
    return logits


def train_once(arm: str, cf: bool, seed: int, hidden: int, data: dict, classes: int,
               cf_weight: float = 0.3, epochs: int = 60) -> tuple[nn.Module, dict, dict]:
    """单次训练；反事实臂只多一项正则，其余配方完全一致。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = list(MODALITIES)
    dimensions = {name: data["train"][name].shape[1] for name in names}
    columns = [names.index(name) for name in names]
    scaler = FeatureStandardizer().fit({name: data["train"][name] for name in names},
                                      data["train"]["modality_mask"][:, columns])
    batches = {split: tensors(d, names, scaler, _NullKnowledge(dimensions)) for split, d in data.items()}
    model = model_for(arm, dimensions, classes, hidden)
    sampler = CounterfactualSampler(data, seed) if cf else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    y = batches["train"][-1]
    counts = torch.bincount(y, minlength=classes).clamp(min=1)
    criterion = nn.CrossEntropyLoss(weight=len(y) / (classes * counts.float()))
    best = {"loss": float("inf"), "state": None, "epoch": 0, "stale": 0}
    cf_stats = {"pairs": 0, "w_mean": 0.0}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y))
        for start in range(0, len(y), 64):
            ix = order[start:start + 64]
            from experiments.round9_moe_baselines import forward
            logits, gates, balance, valid = forward(model, batches["train"], ix, augment=True)
            if not valid.any():
                continue
            loss = criterion(logits, y[ix][valid]) + .01 * balance
            if cf and sampler is not None:
                donors, modality, ok = sampler.sample(ix)
                if ok.any():
                    cf_logits, cf_gates, cf_modality, cf_valid = forward_with_swap(
                        model, batches["train"], ix, donors, modality, ok)
                    if cf_valid.any():
                        w = _gate_mass(model, cf_gates, 0).new_zeros(len(cf_logits))
                        for m in range(len(MODALITIES)):
                            selector = cf_modality == m
                            if selector.any():
                                w[selector] = _gate_mass(model, cf_gates[selector], m)
                        was_training = model.training
                        model.eval()
                        with torch.no_grad():
                            teacher, _, _, teacher_valid = _forward_clean(model, batches["train"], ix[cf_valid])
                            donor_teacher = unimodal_teacher(model, batches["train"], donors[cf_valid], cf_modality)
                        model.train(was_training)
                        if not teacher_valid.all():
                            raise ValueError("反事实教师与有效配对未对齐")
                        target = soft_counterfactual_target(teacher, donor_teacher, w)
                        regularizer = nn.functional.kl_div(cf_logits.log_softmax(1), target, reduction="batchmean")
                        loss = loss + cf_weight * regularizer
                        cf_stats["pairs"] += int(cf_valid.sum())
                        cf_stats["w_mean"] += float(w.detach().mean()) * int(cf_valid.sum())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            logits, _, _, valid = _forward_clean(model, batches["selection"],
                                                 torch.arange(len(data["selection"]["labels"])))
            selection_loss = float(nn.functional.cross_entropy(logits, batches["selection"][-1][valid]))
        if selection_loss < best["loss"] - 1e-5:
            best = {"loss": selection_loss, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch, "stale": 0}
        else:
            best["stale"] += 1
            if best["stale"] >= 10:
                break
    if best["state"] is None:
        raise RuntimeError("训练未产出有效模型")
    model.load_state_dict(best["state"])
    info = {"seed": seed, "hidden": hidden, "epoch": best["epoch"],
            "selection_loss": round(best["loss"], 5)}
    if cf and cf_stats["pairs"]:
        info["cf_pairs"] = cf_stats["pairs"]
        info["cf_gate_mass_mean"] = round(cf_stats["w_mean"] / cf_stats["pairs"], 4)
    return model, info, batches


def _forward_clean(model: nn.Module, batch: tuple, indices: Tensor):
    from experiments.round9_moe_baselines import forward
    return forward(model, batch, indices)


def evaluate(model: nn.Module, batch: tuple, scenario: str, labels: np.ndarray,
             modality: int | None = None) -> dict:
    """干净/损坏场景；损坏场景同时报告"路由给该模态的权重和"。"""
    from experiments.round9_moe_baselines import forward
    model.eval()
    with torch.inference_mode():
        logits, gates, _, valid = forward(model, batch, torch.arange(len(labels)), scenario=scenario)
    y = labels[valid.numpy()]
    result = {"n": int(len(y))}
    if len(y) == 0:
        return result
    result.update({k: metrics(y, logits.numpy(), model.classes)[k]
                   for k in ("accuracy", "macro_f1", "per_class_recall")})
    if modality is not None and gates.numel():
        result["gate_mass"] = round(float(_gate_mass(model, gates, modality).mean()), 4)
    return result


def main() -> None:
    if (OUTPUT / "counterfactual.json").exists():
        raise ValueError("拒绝覆盖旧反事实结果；请设置新的ROUND9_OUTPUT")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = {split: load_split(BUNDLE / f"{split}.npz") for split in ("train", "selection", "test")}
    classes = 3
    labels = data["test"]["labels"]
    report: dict[str, Any] = {
        "schema": "round10_counterfactual_soft_teacher/2",
        "seeds": list(SEEDS),
        "cf_weight": 0.3,
        "supervision": "route-detached source/full and donor/single-modality soft self-distillation; no synthetic overall hard label",
        "protocol": "与 round9_moe_baselines 相同：seed×hidden 搜索、选择集早停、同划分",
        "limitations": [
            "反事实对只在同来源组、不同标签之间构造；构造不出的样本不造对",
            "损坏场景使用测试集自带的质量标记（visual/audio 置 0.25 并加噪），不是真实噪声金标",
            "测试集已被前轮查看，属同协议诊断",
        ],
        "arms": {},
    }
    for arm, cf in ARMS:
        name = f"{arm}{'+cf' if cf else ''}"
        runs = []
        for seed in SEEDS:
            best = None
            for hidden in HIDDENS:
                seed_all()
                torch.manual_seed(seed)
                np.random.seed(seed)
                # 对照臂直接复用基线脚本的训练入口，保证与基线表逐位一致；
                # 反事实臂走本地循环，两者差别只有多出来的那一项正则。
                if cf:
                    model, info, batches = train_once(arm, cf, seed, hidden, data, classes)
                else:
                    model, info, batches = train_clean(arm, seed, hidden, data, classes)
                if best is None or info["selection_loss"] < best[1]["selection_loss"]:
                    best = (model, info, batches)
            model, info, batches = best
            clean = evaluate(model, batches["test"], "clean", labels)
            noisy_visual = evaluate(model, batches["test"], "noisy_visual_known_quality", labels, modality=2)
            no_text = evaluate(model, batches["test"], "without_text", labels)
            runs.append({**info, "clean": clean, "noisy_visual": noisy_visual, "without_text": no_text,
                         "infer_ms": _infer_seconds(model, batches["test"])})
        report["arms"][name] = {
            "runs": runs,
            "clean_macro_f1_mean": round(statistics.mean(r["clean"]["macro_f1"] for r in runs), 4),
            "clean_macro_f1_std": round(statistics.pstdev([r["clean"]["macro_f1"] for r in runs]), 4),
            "clean_accuracy_mean": round(statistics.mean(r["clean"]["accuracy"] for r in runs), 4),
            "noisy_visual_macro_f1_mean": round(statistics.mean(r["noisy_visual"]["macro_f1"] for r in runs), 4),
            "noisy_visual_gate_mass_mean": round(statistics.mean(
                r["noisy_visual"].get("gate_mass", float("nan")) for r in runs), 4),
            "without_text_macro_f1_mean": round(statistics.mean(r["without_text"]["macro_f1"] for r in runs), 4),
        }
        a = report["arms"][name]
        print(f"{name:14s} clean={a['clean_macro_f1_mean']:.4f}±{a['clean_macro_f1_std']:.4f} "
              f"noisyVis={a['noisy_visual_macro_f1_mean']:.4f} "
              f"gateVis={a['noisy_visual_gate_mass_mean']:.3f} "
              f"noText={a['without_text_macro_f1_mean']:.4f}")

    # 配对差：同种子 cf − 无 cf
    report["comparisons"] = {}
    for arm, _cf in (("moe", False), ("i2moe", False)):
        base = report["arms"][arm]["runs"]
        treat = report["arms"][f"{arm}+cf"]["runs"]
        deltas = [round(t["clean"]["macro_f1"] - b["clean"]["macro_f1"], 4) for t, b in zip(treat, base)]
        report["comparisons"][f"{arm}+cf_minus_{arm}"] = {
            "clean_mean_delta": round(statistics.mean(deltas), 4),
            "per_seed": deltas,
            "wins": sum(1 for d in deltas if d > 0),
        }
    (OUTPUT / "counterfactual.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nwritten -> {OUTPUT/'counterfactual.json'}")


if __name__ == "__main__":
    main()
