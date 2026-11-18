"""来源均衡经验风险、来源风格惩罚和固定来源袋装的第二阶段实验。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from experiments.round10_fusion import OUTPUT, save, folds, take
from experiments.round10_represented_fusion import data_for, transform
from experiments.round7_decision_oof import metric, probability_metrics
from experiments.round10_final import bootstrap


def weighted_f1(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    """来源等权的汇总混淆矩阵，避免大视频主导开发选择。"""
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1/count[inverse]
    matrix = np.bincount(3*y+pred, weights=weight, minlength=9).reshape(3, 3)
    den = matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix), den, out=np.zeros(3), where=den > 0).mean())


def weights_for(y: np.ndarray, groups: np.ndarray, gp: float, cp: float) -> np.ndarray:
    """来源权重与类别权重分别定义，按有效样本权重重新计算类别频率。"""
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    weight = (len(y)/len(count)/count[inverse])**gp
    class_count = np.bincount(y, weights=weight, minlength=3)
    weight *= (weight.sum()/(3*class_count[y]))**cp
    return weight/weight.mean()


def style_fit(x: np.ndarray, groups: np.ndarray, strength: float) -> dict[str, np.ndarray]:
    """只用训练来源均值构建低秩风格惩罚，预测无需来源ID。"""
    if not strength:
        return {"directions": np.empty((0, x.shape[1])), "factors": np.empty(0)}
    means = np.stack([x[groups == name].mean(0) for name in np.unique(groups)])
    means = (means - x.mean(0))/np.sqrt(len(means))
    _, singular, directions = np.linalg.svd(means, full_matrices=False)
    return {"directions": directions, "factors": 1/np.sqrt(1+strength*singular**2)-1}


def style_apply(x: np.ndarray, projection: dict[str, np.ndarray]) -> np.ndarray:
    """把依赖来源风格的参数方向施加更强L2惩罚，不读测试来源均值。"""
    directions = projection["directions"]
    return x + ((x@directions.T)*projection["factors"])@directions


def bias_select(data: dict[str, np.ndarray], logits: np.ndarray) -> dict[str, Any]:
    """兼顾逐样本macro-F1与来源等权macro-F1，网格预先固定。"""
    options = []
    for neg, neutral in itertools.product((-.25, 0., .25), (-.25, 0., .25, .5)):
        bias = [neg, neutral, 0.]
        pred = (logits+np.array(bias)).argmax(1)
        result = metric(data["labels"], pred)
        source = weighted_f1(data["labels"], pred, data["source_groups"])
        options.append({"bias": bias, "metrics": result, "source_balanced_f1": source,
                        "selection_score": .5*(source+result["macro_f1"])})
    return max(options, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"],
                                         -sum(abs(v) for v in row["bias"])))


def configs() -> list[dict[str, Any]]:
    """第二阶段按来源泛化假设独立预注册，历史分数不参与选择。"""
    result = []
    for pack, gp, cp, strength, c in itertools.product(("aux_full", "deep_combined"), (0., .5, 1.),
                                                        (.5, 1.), (0., 1., 10.), (.01, .03, .1, .3)):
        result.append({"id": f"domain_{len(result)}", "pack": pack, "components": 128,
            "face_weight": 2. if pack == "aux_full" else 1., "group_power": gp, "class_power": cp,
            "style_strength": strength, "C_reference2000": c})
    return result


def search(output: Path) -> None:
    """第二阶段仅从训练来源OOF评估与锁定配方；第一阶段负结果不覆盖。"""
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖来源泛化搜索")
    data = data_for(("train", "selection"), True)
    grid = configs()
    save(output / "preregister.json", {"phase": 2, "previous_phase_target_achieved": False,
         "configs": grid, "selection": ".5 pooled macro-F1 + .5 source-balanced macro-F1, train+selection OOF only",
         "final_refit": "four equal-weight source-fold models on nonhistorical3579; C scaled as reference2000/n_train",
         "historical_exposure": "phase1 already evaluated; phase2 historical result is adaptive repeated diagnostic, not independent confirmation"})
    predictions = np.full((len(grid), len(data["labels"]), 3), np.nan)
    cache, results = {}, []
    start = time.monotonic()
    for i, config in enumerate(grid):
        for fold, (tr, va) in enumerate(folds(data)):
            key = (config["pack"], config["style_strength"], fold)
            if key not in cache:
                base_key = (config["pack"], "base", fold)
                if base_key not in cache:
                    x, scaler = transform(take(data, tr), config)
                    v, _ = transform(take(data, va), config, scaler)
                    cache[base_key] = x, v
                x, v = cache[base_key]
                projection = style_fit(x, data["source_groups"][tr], config["style_strength"])
                cache[key] = style_apply(x, projection), style_apply(v, projection)
            x, v = cache[key]
            weight = weights_for(data["labels"][tr], data["source_groups"][tr], config["group_power"], config["class_power"])
            model = LogisticRegression(C=config["C_reference2000"]*2000/len(tr), max_iter=500, random_state=42)
            model.fit(x, data["labels"][tr], sample_weight=weight)
            predictions[i, va] = model.decision_function(v)
        scale = max(.1, float(predictions[i].std(1).mean()))
        chosen = bias_select(data, predictions[i]/scale)
        results.append({"config": config, "logit_scale": scale, **chosen})
        if (i+1) % 12 == 0:
            winner = max(results, key=lambda row: row["selection_score"])
            save(output / "progress.json", results)
            print(json.dumps({"candidates": i+1, "total": len(grid), "winner": winner["config"]["id"],
                  "oof_f1": winner["metrics"]["macro_f1"], "source_f1": winner["source_balanced_f1"],
                  "seconds": round(time.monotonic()-start, 1)}), flush=True)
    winner = max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "selection_locked.json", winner)
    save(output / "search_complete.json", {"results": results})
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"])


def refit(output: Path) -> None:
    """固定四折袋装；没有根据历史分数选择训练规模或集成成员。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖来源模型")
    chosen = json.loads((output / "selection_locked.json").read_text("utf-8"))
    config = chosen["config"]
    data = data_for(("train", "selection", "calibration"), True)
    models = []
    for train, _ in folds(data):
        subset = take(data, train)
        x, scaler = transform(subset, config)
        projection = style_fit(x, subset["source_groups"], config["style_strength"])
        x = style_apply(x, projection)
        weight = weights_for(subset["labels"], subset["source_groups"], config["group_power"], config["class_power"])
        model = LogisticRegression(C=config["C_reference2000"]*2000/len(train), max_iter=500, random_state=42).fit(x, subset["labels"], sample_weight=weight)
        models.append({"model": model, "scalers": scaler, "projection": projection})
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "selection": chosen, "training_groups": np.unique(data["source_groups"])}, handle)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """来源风格方向在训练时冻结，推理不读取来源ID或真实标签。"""
    chosen = bundle["selection"]
    result = []
    for model in bundle["models"]:
        x, _ = transform(data, chosen["config"], model["scalers"])
        z = model["model"].decision_function(style_apply(x, model["projection"])) / chosen["logit_scale"]
        result.append(softmax(z, axis=1))
    return softmax(np.log(np.maximum(np.mean(result, axis=0), 1e-12))+np.asarray(chosen["bias"]), axis=1)


def evaluate(output: Path) -> None:
    """锁定后的第二阶段历史诊断，完整保留第一阶段负结果。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖历史诊断")
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    data = data_for(("test",), True)
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("来源交叉")
    p = predict(bundle, data)
    old = np.load(OUTPUT / "final" / "historical_predictions.npz")
    report = {**metric(data["labels"], p.argmax(1)), **probability_metrics(data["labels"], p),
        "relative_gain": metric(data["labels"], p.argmax(1))["macro_f1"]/.6388-1,
        "target_achieved": metric(data["labels"], p.argmax(1))["macro_f1"] >= .70268,
        "bootstrap": bootstrap(data["labels"], p, old["baseline"], data["source_groups"]),
        "adaptive_historical_diagnostic": True, "coverage": 1.}
    save(output / "results.json", report)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    print(json.dumps(report), flush=True)


def main() -> None:
    """搜索、拟合与评测分开执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    args = parser.parse_args()
    {"search": search, "refit": refit, "evaluate": evaluate}[args.action](OUTPUT / "domain")


if __name__ == "__main__":
    main()
