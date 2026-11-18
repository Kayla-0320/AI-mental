"""独立训练的有序情感轴与中性区间；仅由来源OOF选择切点。"""
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
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.special import expit
from experiments.round10_fusion import OUTPUT, folds, take, save
from experiments.round10_represented_fusion import data_for, transform
from experiments.round10_domain import weights_for, weighted_f1
from experiments.round7_decision_oof import metric
from experiments.round10_final import bootstrap


def configs() -> list[dict[str, Any]]:
    """训练有序目标，而非在原三类logit上事后改名。"""
    return [{"id": f"ordinal_{i}", "pack": pack, "components": 128, "face_weight": 2. if pack == "aux_full" else 1.,
             "kind": kind, "C": c, "class_power": cp}
        for i, (pack, kind, c, cp) in enumerate(itertools.product(("aux_full", "deep_combined"),
              ("ridge", "cumulative"), (.01, .1, 1., 10.), (.5, 1.)))]


def fit_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray, config: dict[str, Any]) -> Any:
    """岭回归预测有序标签，累计分类预测两个累积事件。"""
    weights = weights_for(y, groups, 0., config["class_power"])
    if config["kind"] == "ridge":
        return Ridge(alpha=1/config["C"]).fit(x, y-1., sample_weight=weights)
    return [LogisticRegression(C=config["C"]*2000/len(y), max_iter=500, random_state=42).fit(x, (y > k).astype(int), sample_weight=weights) for k in (0, 1)]


def raw_predict(model: Any, x: np.ndarray, kind: str) -> np.ndarray:
    """累计概率的和减1是有序标签的期望。"""
    if kind == "ridge":
        return model.predict(x)
    p = [member.predict_proba(x)[:, 1] for member in model]
    return p[0]+p[1]-1


def choose(y: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """切点只看来源OOF，固定网格不读取历史中性比例。"""
    options = []
    for lo, hi in itertools.product(np.arange(-.8, -.049, .05), np.arange(.05, .801, .05)):
        pred = np.where(values < lo, 0, np.where(values > hi, 2, 1))
        result = metric(y, pred)
        options.append({"lo": float(lo), "hi": float(hi), "metrics": result})
    return max(options, key=lambda row: (row["metrics"]["macro_f1"], row["metrics"]["accuracy"]))


def search(output: Path) -> None:
    """新有序训练假设下的开发研究；先锁定，再拟合或评测。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖有序模型选择")
    data = data_for(("train", "selection"), True)
    grid = configs()
    save(output / "preregister.json", {"phase": 3, "configs": grid, "objective": "pooled OOF macro-F1",
         "selection": "source-group train+selection only", "horizon": "current clip classification, not forecasting",
         "final_training": "fixed four source-fold bag on nonhistorical3579", "historical_status": "adaptive diagnostics"})
    cache, results, predictions = {}, [], []
    for config in grid:
        oof = np.full(len(data["labels"]), np.nan)
        for fold, (tr, va) in enumerate(folds(data)):
            key = config["pack"], fold
            if key not in cache:
                x, scaler = transform(take(data, tr), config)
                v, _ = transform(take(data, va), config, scaler)
                cache[key] = x, v
            x, v = cache[key]
            model = fit_model(x, data["labels"][tr], data["source_groups"][tr], config)
            oof[va] = raw_predict(model, v, config["kind"])
        selected = choose(data["labels"], oof)
        results.append({"config": config, **selected})
        predictions.append(oof)
        if len(results) % 8 == 0:
            best = max(results, key=lambda row: row["metrics"]["macro_f1"])
            print(json.dumps({"candidates": len(results), "total": len(grid), "best_oof": best["metrics"]["macro_f1"], "winner": best["config"]["id"]}), flush=True)
    best = max(results, key=lambda row: (row["metrics"]["macro_f1"], row["metrics"]["accuracy"]))
    save(output / "search_complete.json", {"results": results})
    save(output / "selection_locked.json", best)
    np.savez_compressed(output / "candidates_oof.npz", values=np.stack(predictions), labels=data["labels"], sample_ids=data["sample_ids"])


def refit(output: Path) -> None:
    """固定来源袋装，不根据历史结果挑成员或训练规模。"""
    selected = json.loads((output / "selection_locked.json").read_text("utf-8"))
    config = selected["config"]
    data = data_for(("train", "selection", "calibration"), True)
    models = []
    for tr, _ in folds(data):
        sub = take(data, tr)
        x, scaler = transform(sub, config)
        models.append({"scaler": scaler, "model": fit_model(x, sub["labels"], sub["source_groups"], config)})
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "selection": selected, "training_groups": np.unique(data["source_groups"])}, handle)


def evaluate(output: Path) -> None:
    """固定有序配方的历史诊断，保留先前所有结果。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖有序历史结果")
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    data = data_for(("test",), True)
    if set(bundle["training_groups"]) & set(data["source_groups"]):
        raise ValueError("来源交叉")
    selected = bundle["selection"]
    values = []
    for model in bundle["models"]:
        x, _ = transform(data, selected["config"], model["scaler"])
        values.append(raw_predict(model["model"], x, selected["config"]["kind"]))
    values = np.mean(values, 0)
    pred = np.where(values < selected["lo"], 0, np.where(values > selected["hi"], 2, 1))
    result = {**metric(data["labels"], pred), "relative_gain": metric(data["labels"], pred)["macro_f1"]/.6388-1,
              "target_achieved": metric(data["labels"], pred)["macro_f1"] >= .70268, "coverage": 1., "adaptive_diagnostic": True}
    old = np.load(OUTPUT / "final" / "historical_predictions.npz")
    result["bootstrap"] = bootstrap(data["labels"], np.eye(3)[pred], old["baseline"], data["source_groups"])
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", values=values, pred=pred, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps(result), flush=True)


def main() -> None:
    """分离选型、拟合和评测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    args = parser.parse_args()
    {"search": search, "refit": refit, "evaluate": evaluate}[args.action](OUTPUT / "ordinal")


if __name__ == "__main__":
    main()
