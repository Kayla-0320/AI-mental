"""用官方连续强度监督低维证据读出，三类验收标签保持原样。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pickle
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from catboost import CatBoostRegressor
from experiments.round10_fusion import OUTPUT, load_npz, take, save
from experiments import round10_evidence as evidence
from experiments.round10_domain import weights_for, weighted_f1
from experiments.round7_decision_oof import metric
from experiments.round10_final import bootstrap


def configs() -> list[dict[str, Any]]:
    """仅比较读出损失及有限树复杂度。"""
    return [{"id": f"intensity_{i}", "depth": depth, "iterations": 500, "learning_rate": .03,
             "l2_leaf_reg": reg, "class_power": cp}
        for i, (depth, reg, cp) in enumerate(itertools.product((4, 6), (3., 10.), (0., .5, 1.)))]


def model_for(config: dict[str, Any]) -> CatBoostRegressor:
    """固定轮次的连续强度回归，不在外层留出上早停。"""
    return CatBoostRegressor(iterations=config["iterations"], depth=config["depth"], learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"], loss_function="RMSE", random_seed=42, verbose=False,
        thread_count=2, allow_writing_files=False)


def select_cut(data: dict[str, np.ndarray], values: np.ndarray) -> dict[str, Any]:
    """三类映射切点仅由来源OOF选择，原三类标签没有变化。"""
    results = []
    for lo, hi in itertools.product((-.2, -.15, -.1, -.05, -.025), (.025, .05, .1, .15, .2)):
        pred = np.where(values < lo, 0, np.where(values > hi, 2, 1))
        metrics = metric(data["labels"], pred)
        source = weighted_f1(data["labels"], pred, data["source_groups"])
        results.append({"lo": lo, "hi": hi, "metrics": metrics, "source_balanced_f1": source,
                        "selection_score": .5*(source+metrics["macro_f1"])})
    return max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))


def search(output: Path) -> None:
    """复用已严格交叉拟合的专家证据，外层标签只用于最终开发评价。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖强度读出选择")
    data = evidence.data_for(("train", "selection"))
    _, continuous = evidence.supervision(data)
    lookup = {identifier: i for i, identifier in enumerate(data["sample_ids"])}
    grid = configs()
    save(output / "preregister.json", {"phase": 7, "configs": grid, "target_training": "official continuous M intensity",
         "acceptance": "unchanged original negative/neutral/positive labels and full821 coverage",
         "selection": "4 source-fold OOF, train+selection only; .5 pooled+.5 source F1",
         "evidence": "reuses3-inner/4-outer crossfit M/T/A/V evidence; no test annotation input",
         "historical": "adaptive repeated diagnostics, not new independent confirmation"})
    predictions = np.full((len(grid), len(data["labels"])), np.nan)
    for fold in range(4):
        rows = load_npz(OUTPUT / "evidence" / f"fold{fold}_evidence.npz")
        tr = np.array([lookup[sid] for sid in rows["train_sample_ids"]])
        va = np.array([lookup[sid] for sid in rows["validation_sample_ids"]])
        if set(data["source_groups"][tr]) & set(data["source_groups"][va]):
            raise ValueError("来源交叉")
        for i, config in enumerate(grid):
            weight = weights_for(data["labels"][tr], data["source_groups"][tr], 0., config["class_power"])
            model = model_for(config).fit(rows["inner_evidence"], continuous[tr, 0], sample_weight=weight)
            predictions[i, va] = model.predict(rows["outer_evidence"]).clip(-1, 1)
        print(json.dumps({"intensity_outer_fold": fold}), flush=True)
    results = [{"config": config, **select_cut(data, values)} for config, values in zip(grid, predictions)]
    winner = max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "selection_locked.json", winner)
    save(output / "search_complete.json", {"results": results})
    np.savez_compressed(output / "candidates_oof.npz", values=predictions, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps(winner), flush=True)


def refit(output: Path) -> None:
    """最终元训练依然使用来源交叉拟合证据。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖强度模型")
    data = evidence.data_for(("train", "selection", "calibration"))
    _, continuous = evidence.supervision(data)
    selected = json.loads((output / "selection_locked.json").read_text("utf-8"))
    features = evidence.crossfit(data, 4)
    weight = weights_for(data["labels"], data["source_groups"], 0., selected["config"]["class_power"])
    model = model_for(selected["config"]).fit(features, continuous[:, 0], sample_weight=weight)
    experts = evidence.fit_experts(data)
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"meta": model, "experts": experts, "selection": selected,
                     "training_groups": np.unique(data["source_groups"])}, handle)


def evaluate(output: Path) -> None:
    """真实测试类别及样本数不变，连续值不被包装为校准概率。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖历史结果")
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    data = evidence.data_for(("test",))
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("来源交叉")
    features = evidence.expert_evidence(bundle["experts"], data)
    values = bundle["meta"].predict(features).clip(-1, 1)
    lo, hi = bundle["selection"]["lo"], bundle["selection"]["hi"]
    pred = np.where(values < lo, 0, np.where(values > hi, 2, 1))
    baseline = load_npz(OUTPUT / "final" / "historical_predictions.npz")["baseline"]
    result = {**metric(data["labels"], pred), "relative_gain": metric(data["labels"], pred)["macro_f1"]/.6388-1,
        "target_achieved": metric(data["labels"], pred)["macro_f1"] >= .70268,
        "bootstrap": bootstrap(data["labels"], np.eye(3)[pred], baseline, data["source_groups"]),
        "coverage": 1., "adaptive_historical_diagnostic": True}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", values=values, pred=pred, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps(result), flush=True)


def main() -> None:
    """开发、拟合和历史诊断入口分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    args = parser.parse_args()
    output = OUTPUT / "intensity"
    output.mkdir(exist_ok=True)
    {"search": search, "refit": refit, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
