"""官方单模态监督专家与严格嵌套交叉拟合的非线性证据读出。"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier

from experiments.round10_fusion import OUTPUT, folds, take, save
from experiments import round10_represented_fusion as represented
from experiments import round10_llm_fusion as llm
from experiments.round10_domain import weights_for, bias_select
from experiments.round7_decision_oof import metric, probability_metrics
from experiments.round10_final import bootstrap

VIEWS = ("M", "T", "A", "V")
EXPERT_CLASSES = 3
PACKS = {
    "view_M": ("llm_last", "aux_mean", "bge_cls", "face_summary", "audio", "visual"),
    "view_T": ("llm_last", "aux_mean", "bge_cls"),
    "view_A": ("audio",),
    "view_V": ("face_summary", "deep_face", "visual"),
}
represented.PACKS.update(PACKS)


def data_for(splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """推理数据不加载单模态金标；监督标注只由训练入口读取。"""
    data = llm.data_for(splits)
    deep = represented.data_for(splits, True)
    data["deep_face"], data["coverage"] = deep["deep_face"], deep["coverage"]
    return data


def supervision(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """按官方ID及一条可核验的拼写别名对齐，M标签必须与原镜像相同。"""
    rows = list(csv.DictReader((OUTPUT / "official_meta.csv").open(encoding="utf-8")))
    lookup = {row["video_id"]+"_"+row["clip_id"]: row for row in rows}
    continuous, classes = [], []
    for identifier, target in zip(data["sample_ids"], data["labels"]):
        canonical = str(identifier).replace("csbdgs2_", "csbgs2_")
        if canonical not in lookup:
            raise ValueError(f"官方标注缺ID: {identifier}")
        row = lookup[canonical]
        if {"Negative": 0, "Neutral": 1, "Positive": 2}[row["annotation"]] != target:
            raise ValueError("官方M标签与冻结任务不一致，禁止静默改标")
        values = np.array([float(row[name]) for name in ("label", "label_T", "label_A", "label_V")])
        continuous.append(values)
        classes.append(np.sign(values).astype(int)+1)
    return np.stack(classes), np.stack(continuous)


def visibility(data: dict[str, np.ndarray], view: str) -> np.ndarray:
    """缺失专家返回带mask的空证据，不能当作中性观测。"""
    if view in ("M", "T"):
        return data["modality_mask"][:, 0] & (data["quality"][:, 0] > 0)
    if view == "A":
        return data["modality_mask"][:, 1] & (data["quality"][:, 1] > 0)
    return (data["face_summary"][:, 24] > 0) | (data["modality_mask"][:, 2] & (data["quality"][:, 2] > 0))


def fit_experts(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """四个专家分别拟合其真实M/T/A/V标签，不把整体标签强加给各模态。"""
    classes, values = supervision(data)
    if EXPERT_CLASSES == 5:
        classes = np.digitize(values, [-.5, -.1, .1, .5])
    result = {}
    for j, view in enumerate(VIEWS):
        config = {"pack": "view_"+view, "face_weight": 1., "components": 32}
        x, scaler = represented.transform(data, config)
        observed = visibility(data, view)
        weight = weights_for(classes[observed, j], data["source_groups"][observed], 0., .5)
        model = LogisticRegression(C=(.03 if view == "M" else .1)*2000/observed.sum(), max_iter=500, random_state=42)
        model.fit(x[observed], classes[observed, j], sample_weight=weight)
        regression = Ridge(alpha=10.).fit(x[observed], values[observed, j], sample_weight=weight)
        result[view] = {"model": model, "regression": regression, "scaler": scaler, "config": config}
    return result


def expert_evidence(models: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """证据为四路概率、强度、熵、差异和可用性；没有金标输入。"""
    probabilities, regressions, masks = [], [], []
    safe = dict(data)
    safe["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    for view in VIEWS:
        member = models[view]
        x, _ = represented.transform(safe, member["config"], member["scaler"])
        present = visibility(safe, view)
        p = member["model"].predict_proba(x)
        p[~present] = 0.
        probabilities.append(p)
        values = member["regression"].predict(x)
        values[~present] = 0.
        regressions.append(values)
        masks.append(present)
    probability = np.stack(probabilities, axis=1)
    mask = np.stack(masks, axis=1)
    entropy = -(probability*np.log(np.maximum(probability, 1e-12))).sum(2)
    expected = probability @ np.linspace(-1., 1., probability.shape[2])
    relations = []
    for a, b in ((1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3)):
        p, q = probability[:, a], probability[:, b]
        mean = .5*(p+q)
        js = .5*((p*np.log(np.maximum(p, 1e-12)/np.maximum(mean, 1e-12))).sum(1)
                  +(q*np.log(np.maximum(q, 1e-12)/np.maximum(mean, 1e-12))).sum(1))
        relations.extend([js*(mask[:, a]&mask[:, b]), abs(expected[:, a]-expected[:, b])*(mask[:, a]&mask[:, b])])
    return np.column_stack([probability.reshape(len(mask), -1), entropy, expected,
                            np.stack(regressions, 1), mask.astype(float), *relations])


def crossfit(data: dict[str, np.ndarray], n_splits: int = 3) -> np.ndarray:
    """训练读出器的每条证据均来自未见该来源的内层专家。"""
    result = None
    for tr, va in folds(data, n_splits):
        models = fit_experts(take(data, tr))
        features = expert_evidence(models, take(data, va))
        if result is None:
            result = np.full((len(data["labels"]), features.shape[1]), np.nan)
        result[va] = features
    if not np.isfinite(result).all():
        raise ValueError("专家交叉拟合覆盖不完整")
    return result


def options() -> list[dict[str, Any]]:
    """有限深度与类别损失候选，固定轮数，禁止用外层早停。"""
    return [{"id": f"evidence_{i}", "depth": depth, "iterations": 300, "learning_rate": .03,
             "l2_leaf_reg": reg, "class_power": cp}
            for i, (depth, reg, cp) in enumerate(__import__("itertools").product((3, 4, 5), (3., 10.), (.5, 1.)))]


def meta(config: dict[str, Any]) -> CatBoostClassifier:
    """非线性读出器，不修改模态专家或冻结编码器。"""
    return CatBoostClassifier(iterations=config["iterations"], depth=config["depth"], learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"], loss_function="MultiClass", random_seed=42,
        thread_count=2, verbose=False, allow_writing_files=False)


def search(output: Path) -> None:
    """内层生成元训练证据，外层整来源验证，避免普通堆叠的间接泄漏。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖证据模型选择")
    data = data_for(("train", "selection"))
    grid = options()
    save(output / "preregister.json", {"phase": 6, "expert_classes": EXPERT_CLASSES,
        "five_class_bins_if_used": [-.5, -.1, .1, .5], "source": "official meta.csv M/T/A/V labels, final target M3 unchanged",
        "protocol": "4 outer source folds; within each outer training,3 inner source folds produce all meta-training features",
        "configs": grid, "features": "M/T/A/V probabilities, regression intensity, entropy, disagreement, availability",
        "final_refit": "meta on full3579 crossfit evidence, experts on3579; no test-derived settings",
        "unimodal_test_labels_used_as_input": False, "historical": "adaptive repeated diagnostic"})
    predictions = np.full((len(grid), len(data["labels"]), 3), np.nan)
    for fold, (tr, va) in enumerate(folds(data)):
        training, validation = take(data, tr), take(data, va)
        inner = crossfit(training)
        experts = fit_experts(training)
        outer = expert_evidence(experts, validation)
        for i, config in enumerate(grid):
            weights = weights_for(training["labels"], training["source_groups"], 0., config["class_power"])
            model = meta(config).fit(inner, training["labels"], sample_weight=weights)
            predictions[i, va] = np.log(np.maximum(model.predict_proba(outer), 1e-12))
        np.savez_compressed(output / f"fold{fold}_evidence.npz", train_sample_ids=training["sample_ids"], validation_sample_ids=validation["sample_ids"],
                            inner_evidence=inner, outer_evidence=outer, train_labels=training["labels"], validation_labels=validation["labels"])
        print(json.dumps({"outer_fold": fold, "train": len(tr), "valid": len(va), "evidence_dimensions": inner.shape[1]}), flush=True)
    results = []
    for config, logits in zip(grid, predictions):
        results.append({"config": config, **bias_select(data, logits)})
    winner = max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "search_complete.json", {"results": results})
    save(output / "selection_locked.json", winner)
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps({"winner": winner}), flush=True)


def refit(output: Path) -> None:
    """最终训练的元特征仍使用整来源交叉拟合。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖证据模型")
    data = data_for(("train", "selection", "calibration"))
    winner = json.loads((output / "selection_locked.json").read_text("utf-8"))
    features = crossfit(data, 4)
    weights = weights_for(data["labels"], data["source_groups"], 0., winner["config"]["class_power"])
    model = meta(winner["config"]).fit(features, data["labels"], sample_weight=weights)
    experts = fit_experts(data)
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"experts": experts, "meta": model, "selection": winner,
                     "training_groups": np.unique(data["source_groups"])}, handle)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """最终推理只读取观测特征；单模态标签只在fit_experts中使用。"""
    evidence = expert_evidence(bundle["experts"], data)
    logits = np.log(np.maximum(bundle["meta"].predict_proba(evidence), 1e-12))+np.asarray(bundle["selection"]["bias"])
    return softmax(logits, axis=1)


def evaluate(output: Path) -> None:
    """固定后的一次历史诊断，保留此前所有未达标阶段。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖证据历史诊断")
    data = data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if set(bundle["training_groups"]) & set(data["source_groups"]):
        raise ValueError("来源交叉")
    p = predict(bundle, data)
    baseline = np.load(OUTPUT / "final" / "historical_predictions.npz")["baseline"]
    result = {**metric(data["labels"], p.argmax(1)), **probability_metrics(data["labels"], p),
        "relative_gain": metric(data["labels"], p.argmax(1))["macro_f1"]/.6388-1,
        "target_achieved": metric(data["labels"], p.argmax(1))["macro_f1"] >= .70268,
        "bootstrap": bootstrap(data["labels"], p, baseline, data["source_groups"]), "coverage": 1., "adaptive_historical_diagnostic": True}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    print(json.dumps(result), flush=True)


def main() -> None:
    """选型、拟合和评测分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    global EXPERT_CLASSES
    if args.fine:
        EXPERT_CLASSES = 5
    output = OUTPUT / ("evidence5" if args.fine else "evidence")
    output.mkdir(exist_ok=True)
    {"search": search, "refit": refit, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
