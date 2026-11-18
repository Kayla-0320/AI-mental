"""真实配对特征的固定配置七臂消融，独立选择/校准/测试划分。

此脚本不训练 VLM 编码器。预提取特征身份随数据清单保留；未经专家
审核的知识仅在显式候选消融中使用，不作为临床或线上知识。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import Tensor, nn

from perception.knowledge_moe.features import FeatureStandardizer, save_json, sha256_file
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.model import ConcatenationHead, QualityAwareMoE

SEED = 42


def seed_all() -> None:
    """固定所有实际使用的随机源。"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def load_split(path: Path) -> dict[str, np.ndarray]:
    """只加载非对象 NPZ；池化必须由数据接入器按真实有效长度完成。"""
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    n = len(result["labels"])
    for key in ["text", "audio", "visual"]:
        if result[key].ndim != 2 or len(result[key]) != n:
            raise ValueError("需要已审计的等长池化特征")
    if "modality_mask" not in result:
        raise ValueError("特征清单缺少模态观测掩码")
    if "quality" not in result:
        raise ValueError("特征清单缺少质量契约")
    return result


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """保持所有样本字段共同切片，防止标签与模态错位。"""
    return {k: v[indices] for k, v in data.items() if len(v) == len(data["labels"])}


def metrics(labels: np.ndarray, logits: np.ndarray, classes: int, temperature: float = 1.) -> dict[str, Any]:
    """同时报告分类、校准及逐类召回；不把覆盖率下降隐藏在准确率里。"""
    if len(labels) == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "nll": None, "ece": None, "brier": None, "per_class_recall": []}
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        ix = (confidence > low) & (confidence <= high)
        if ix.any():
            ece += ix.mean() * abs(confidence[ix].mean() - (pred[ix] == labels[ix]).mean())
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=list(range(classes)), average="macro", zero_division=0)),
            "nll": float(-np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-15)).mean()),
            "ece": float(ece), "brier": float(((p - np.eye(classes)[labels]) ** 2).sum(axis=1).mean()),
            "per_class_recall": recall_score(labels, pred, labels=list(range(classes)), average=None, zero_division=0).tolist()}


def temperature_fit(labels: np.ndarray, logits: np.ndarray, classes: int) -> float:
    """只以校准集 NLL 拟合一个标量，不读取测试标签。"""
    objective = lambda log_t: metrics(labels, logits, classes, math.exp(log_t))["nll"]
    result = minimize_scalar(objective, bounds=(math.log(.1), math.log(10.)), method="bounded")
    return float(math.exp(result.x))


def tensors(data: dict[str, np.ndarray], names: list[str], scaler: FeatureStandardizer,
            knowledge_index: KnowledgeIndex) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor]:
    """转换为统一张量，知识只从本条原文检索。"""
    canonical = ["text", "audio", "visual"]
    columns = [canonical.index(name) for name in names]
    mask = np.asarray(data["modality_mask"][:, columns], dtype=bool)
    quality = data["quality"][:, columns].astype(np.float32)
    x = scaler.transform({name: data[name] for name in names}, mask)
    kg, kg_mask = knowledge_index.encode(data["raw_text"].astype(str).tolist())
    return ({name: torch.tensor(values) for name, values in x.items()}, torch.tensor(mask), torch.tensor(quality),
            torch.tensor(kg), torch.tensor(kg_mask), torch.tensor(data["labels"].reshape(-1), dtype=torch.long))


def forward(model: nn.Module, batch: tuple, indices: Tensor, augment: bool = False,
            scenario: str = "clean") -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """训练与各臂使用相同受控干扰；测试质量标记是真值辅助压力诊断。"""
    x, mask, q, kg, km, y = batch
    names = list(x)
    x = {name: value[indices].clone() for name, value in x.items()}
    mask, q, kg, km = mask[indices].clone(), q[indices].clone(), kg[indices], km[indices]
    if augment and mask.shape[1] > 1:
        # 同预算各臂共用规则；保留至少一条真实观测。
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
    elif scenario == "random_missing":
        rng = np.random.default_rng(SEED)
        mask &= torch.tensor(rng.random(mask.shape) >= .4)
    elif scenario == "noisy_visual_known_quality" and "visual" in names:
        rng = np.random.default_rng(SEED)
        noise = torch.tensor(rng.normal(0, 2, x["visual"].shape), dtype=torch.float32)
        x["visual"] += noise
        q[:, names.index("visual")] = .25
    # 知识依赖当前文本，文字模态缺失时知识也必须缺失，防止绕过掩码。
    km = km & mask[:, names.index("text")] & (q[:, names.index("text")] > 0) if "text" in names else torch.zeros_like(km)
    valid = (mask & (q > 0)).any(dim=1)
    if not valid.any():
        return torch.empty(0, model.classes if hasattr(model, "classes") else 0), torch.empty(0, 0), torch.tensor(0.), valid
    logits, gates, balance = model({name: value[valid] for name, value in x.items()}, mask[valid], q[valid], kg[valid], km[valid])
    return logits, gates, balance, valid


def grouped_delta_ci(labels: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray,
                     classes: int, repeats: int = 1000) -> list[float]:
    """按来源视频成簇的配对 bootstrap，不把短片段当独立参与者。"""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    units = [np.flatnonzero(groups == g) for g in unique]
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        fa = f1_score(labels[indices], a[indices], labels=list(range(classes)), average="macro", zero_division=0)
        fb = f1_score(labels[indices], b[indices], labels=list(range(classes)), average="macro", zero_division=0)
        differences.append(float(fa - fb))
    return np.quantile(differences, [.025, .975]).tolist()


def run(bundle: Path, destination: Path, epochs: int = 60) -> dict[str, Any]:
    """固定配置完成七臂实验，保存可复验权重、划分与最终测试预测。"""
    seed_all()
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle / "metadata.json"
    if not metadata_path.exists():
        metadata_path = bundle / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = {split: load_split(bundle / f"{split}.npz") for split in ["train", "valid", "test"]}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if set(data[a]["source_groups"]) & set(data[b]["source_groups"]):
            raise ValueError("来源视频组跨划分，拒绝用泄漏划分报告主结果")
        if set(data[a]["sample_ids"]) & set(data[b]["sample_ids"]):
            raise ValueError("样本ID跨划分")
    unique = np.unique(data["valid"]["source_groups"])
    if len(unique) < 4:
        raise ValueError("验证集至少需要4个来源组，才能独立选择与校准")
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    calibration_groups = set(unique[:len(unique) // 2])
    calibration_mask = np.array([g in calibration_groups for g in data["valid"]["source_groups"]])
    data["calibration"] = subset(data["valid"], np.flatnonzero(calibration_mask))
    data["selection"] = subset(data["valid"], np.flatnonzero(~calibration_mask))
    class_names = metadata["class_names"]
    classes = len(class_names)
    knowledge_path = Path(__file__).resolve().parents[1] / "perception/knowledge_moe/knowledge_seed.json"
    knowledge_index = KnowledgeIndex(knowledge_path, allow_candidates=True)
    report: dict[str, Any] = {"schema_version": "round2_moe_experiment.v1", "seed": SEED,
                              "task": metadata["task"], "class_names": class_names, "data": str(bundle),
                              "feature_manifest_sha256": sha256_file(metadata_path), "model_kind": "frozen_feature_fusion_not_vlm",
                              "knowledge_status": "engineering_candidate_lexical_graph_not_expert_reviewed",
                              "split_counts": {s: len(d["labels"]) for s, d in data.items()},
                              "group_counts": {s: len(np.unique(d["source_groups"])) for s, d in data.items()},
                              "configuration": {"epochs_max": epochs, "hidden": 32, "lr": .001,
                                                "weight_decay": .001, "modality_dropout": .15, "balance_coefficient": .01},
                              "limitations": ["官方原划分被来源视频互斥划分替代，不能与官方排行榜直接比较",
                                              "知识为工程候选词汇与关系，不是专家审核完成的知识图谱",
                                              "质量为已知模拟干扰强度，不证明能自动识别真实噪声",
                                              "仅一个固定随机种子；模型参数量不相等，未做等参数消融",
                                              "测试集用于一次预定实验和干扰诊断，不是外部目标人群验证"],
                              "arms": {}}
    save_json(destination / "split_protocol.json", {"selection_sample_ids": data["selection"]["sample_ids"].tolist(),
                                                    "calibration_sample_ids": data["calibration"]["sample_ids"].tolist(),
                                                    "test_sample_ids": data["test"]["sample_ids"].tolist(),
                                                    "source_manifest_sha256": sha256_file(metadata_path)})
    arms = ["text", "audio", "visual", "concat", "concat_knowledge", "moe", "moe_knowledge"]
    predictions = {}
    for arm in arms:
        seed_all()
        start = time.monotonic()
        names = [arm] if arm in ["text", "audio", "visual"] else ["text", "audio", "visual"]
        dimensions = {name: data["train"][name].shape[1] for name in names}
        columns = [["text", "audio", "visual"].index(name) for name in names]
        scaler = FeatureStandardizer().fit({name: data["train"][name] for name in names},
                                           data["train"]["modality_mask"][:, columns])
        batches = {s: tensors(d, names, scaler, knowledge_index) for s, d in data.items() if s != "valid"}
        knowledge_dim = knowledge_index.dimensions if "knowledge" in arm else 0
        architecture = "moe" if arm.startswith("moe") else "concat"
        model = (QualityAwareMoE if architecture == "moe" else ConcatenationHead)(dimensions, classes, 32, knowledge_dim)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        y = batches["train"][-1]
        counts = torch.bincount(y, minlength=classes).clamp(min=1)
        class_weights = len(y) / (classes * counts.float())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(y))
            for i in range(0, len(y), 64):
                ix = order[i:i+64]
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
                logits, _, _, valid = forward(model, batches["selection"], torch.arange(len(data["selection"]["labels"])))
                selection_loss = float(nn.functional.cross_entropy(logits, batches["selection"][-1][valid]))
            if selection_loss < best_loss - 1e-5:
                best_loss, best_state, best_epoch, stale = selection_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
                if stale >= 10:
                    break
        if best_state is None:
            raise ValueError("训练未产出有效选择集模型")
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            cal_logits, _, _, cal_valid = forward(model, batches["calibration"], torch.arange(len(data["calibration"]["labels"])))
        temperature = temperature_fit(data["calibration"]["labels"].reshape(-1)[cal_valid.numpy()], cal_logits.numpy(), classes)
        arm_dir = destination / arm
        arm_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), arm_dir / "weights.pt")
        encoders = metadata["encoder_fingerprints"]
        artifact = {"schema_version": "knowledge_moe.v1", "task": metadata["task"], "class_names": class_names,
                    "dimensions": dimensions, "hidden": 32, "architecture": architecture, "top_k": 0,
                    "knowledge_dim": knowledge_dim, "knowledge_sha256": knowledge_index.sha256 if knowledge_dim else None,
                    "knowledge_review_mode": knowledge_index.review_mode if knowledge_dim else "disabled",
                    "checkpoint_sha256": sha256_file(arm_dir / "weights.pt"), "encoder_fingerprints": {k: encoders[k] for k in names},
                    "standardizer": scaler.parameters, "temperature": temperature,
                    "calibration": {"status": "calibrated", "fit_split": "calibration", "n": int(cal_valid.sum()),
                                    "sample_ids_sha256": hashlib.sha256("\n".join(data["calibration"]["sample_ids"]).encode()).hexdigest()},
                    "training_manifest_sha256": sha256_file(metadata_path), "seed": SEED}
        save_json(arm_dir / "manifest.json", artifact)
        arm_result: dict[str, Any] = {"parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
                                      "temperature": temperature, "calibration_raw": metrics(data["calibration"]["labels"].reshape(-1)[cal_valid.numpy()], cal_logits.numpy(), classes),
                                      "calibration_after": metrics(data["calibration"]["labels"].reshape(-1)[cal_valid.numpy()], cal_logits.numpy(), classes, temperature),
                                      "conditions": {}}
        for scenario in ["clean", "without_text", "without_audio", "without_visual", "random_missing", "noisy_visual_known_quality"]:
            with torch.inference_mode():
                logits, gates, _, valid = forward(model, batches["test"], torch.arange(len(data["test"]["labels"])), scenario=scenario)
            y_test = data["test"]["labels"].reshape(-1)[valid.numpy()]
            result = metrics(y_test, logits.numpy(), classes, temperature)
            result["coverage"] = float(valid.float().mean())
            result["mean_expert_weights"] = gates.mean(dim=0).tolist() if len(gates) else []
            arm_result["conditions"][scenario] = result
            if scenario == "clean":
                arm_result["test_uncalibrated"] = metrics(y_test, logits.numpy(), classes)
                predictions[arm] = logits.argmax(dim=1).numpy()
                np.savez_compressed(arm_dir / "test_predictions.npz", sample_ids=data["test"]["sample_ids"][valid.numpy()],
                                    logits=logits.numpy(), labels=y_test, expert_weights=gates.numpy())
        arm_result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        report["arms"][arm] = arm_result
        save_json(destination / "experiment.json", report)
        print(json.dumps({"arm": arm, "macro_f1": arm_result["conditions"]["clean"]["macro_f1"],
                          "ece": arm_result["conditions"]["clean"]["ece"], "seconds": arm_result["elapsed_seconds"]}), flush=True)
    labels = data["test"]["labels"].reshape(-1)
    report["paired_group_bootstrap_macro_f1"] = {}
    for a, b in [("moe", "concat"), ("moe_knowledge", "moe"), ("moe_knowledge", "concat")]:
        if len(predictions[a]) != len(labels) or len(predictions[b]) != len(labels):
            continue
        delta = report["arms"][a]["conditions"]["clean"]["macro_f1"] - report["arms"][b]["conditions"]["clean"]["macro_f1"]
        ci = grouped_delta_ci(labels, predictions[a], predictions[b], data["test"]["source_groups"], classes)
        report["paired_group_bootstrap_macro_f1"][f"{a}_minus_{b}"] = {"difference": delta, "ci95": ci, "replicates": 1000}
    save_json(destination / "experiment.json", report)
    return report


def main() -> None:
    """运行独立研究任务，不修改线上模型或阈值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    run(args.bundle, args.output, args.epochs)


if __name__ == "__main__":
    main()
