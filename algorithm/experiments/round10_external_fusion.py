"""公开情感表征与跨模态交互特征，只由来源OOF选择。"""
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
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from experiments.round10_fusion import OUTPUT, load_npz, folds, take, save
from experiments import round10_represented_fusion as represented
from experiments.round10_domain import weights_for, bias_select
from experiments.round7_decision_oof import metric, probability_metrics
from experiments.round10_final import bootstrap

PACKS = {
    "sentface": ("sent_mean", "bge_cls", "face_summary", "sent_confidence"),
    "sentfull": ("sent_mean", "sent_cls", "bge_cls", "face_summary", "visual", "audio", "sent_confidence"),
    "sentaux": ("sent_mean", "aux_mean", "bge_cls", "face_summary", "visual", "audio", "sent_confidence"),
}
represented.PACKS.update(PACKS)
FEATURE_DIRECTORY = OUTPUT / "external_features"
PHASE = 4


def data_for(splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """身份严格对齐，二类置信度作为特征而非三类金标。"""
    data = represented.data_for(splits, False)
    parts = [load_npz(FEATURE_DIRECTORY / f"{split}.npz") for split in splits]
    ids = np.concatenate([part["sample_ids"] for part in parts])
    if not np.array_equal(ids, data["sample_ids"]):
        raise ValueError("公开特征ID未对齐")
    for name in ("sent_mean", "sent_cls", "sent_scores"):
        data[name] = np.concatenate([part[name] for part in parts])
    delta = data["sent_scores"][:, 1]-data["sent_scores"][:, 0]
    p = softmax(data["sent_scores"], axis=1)
    data["sent_confidence"] = np.column_stack([delta, np.abs(delta), delta**2, p[:, 1],
                                               -(p*np.log(np.maximum(p, 1e-12))).sum(1)])
    return data


def transform(data: dict[str, np.ndarray], config: dict[str, Any], scalers: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """可选的交互只使用同一样本当前文字和有效表情。"""
    values = dict(data)
    if config["cross"]:
        text_probability = softmax(data["sent_scores"], axis=1)[:, 1]
        face_p = softmax(data["face_summary"][:, :8], axis=1)
        face_valence = face_p[:, 4]-face_p[:, [0, 1, 2, 3, 6]].sum(1)
        text_valence = 2*text_probability-1
        cross = np.column_stack([text_valence*face_valence, abs(text_valence-face_valence),
                                 face_valence**2, text_valence**2, face_p[:, 5]])
        cross *= (data["face_summary"][:, 24] > 0)[:, None]
        values["face_summary"] = np.column_stack([data["face_summary"], cross])
    return represented.transform(values, config, scalers)


def configs() -> list[dict[str, Any]]:
    """有限外部表征候选，不从历史成绩扫描权重或切点。"""
    return [{"id": f"external_{i}", "pack": pack, "cross": cross, "components": 0,
             "face_weight": fw, "C": c, "class_power": power}
        for i, (pack, cross, fw, c, power) in enumerate(itertools.product(PACKS, (False, True),
              (1., 2.), (.01, .03, .1), (.5, 1.)))]


def search(output: Path) -> None:
    """公开模型特征选型；上游训练重合情况显式保留未核实状态。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖公开表征选择")
    data = data_for(("train", "selection"))
    grid = configs()
    save(output / "preregister.json", {"phase": PHASE, "configs": grid,
         "selection": "source-group train+selection OOF; .5 pooled F1 + .5 source-balanced F1",
         "final_refit": "four fixed source-fold models on nonhistorical3579, equal probability voting",
         "upstream_target_overlap_verified": False, "historical": "adaptive repeated diagnostic only"})
    cache, results, predictions = {}, [], []
    for config in grid:
        oof = np.full((len(data["labels"]), 3), np.nan)
        for fold, (tr, va) in enumerate(folds(data)):
            key = config["pack"], config["cross"], config["face_weight"], fold
            if key not in cache:
                x, scalers = transform(take(data, tr), config)
                v, _ = transform(take(data, va), config, scalers)
                cache[key] = x, v
            x, v = cache[key]
            weight = weights_for(data["labels"][tr], data["source_groups"][tr], 0., config["class_power"])
            model = LogisticRegression(C=config["C"]*2000/len(tr), max_iter=500, random_state=42).fit(x, data["labels"][tr], sample_weight=weight)
            oof[va] = model.decision_function(v)
        scale = max(.1, float(oof.std(1).mean()))
        selected = bias_select(data, oof/scale)
        results.append({"config": config, "scale": scale, **selected})
        predictions.append(oof)
        if len(results) % 12 == 0:
            best = max(results, key=lambda row: row["selection_score"])
            save(output / "progress.json", results)
            print(json.dumps({"candidates": len(results), "total": len(grid), "best_oof": best["metrics"]["macro_f1"],
                              "source_f1": best["source_balanced_f1"], "winner": best["config"]["id"]}), flush=True)
    best = max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "selection_locked.json", best)
    save(output / "search_complete.json", {"results": results})
    np.savez_compressed(output / "candidates_oof.npz", logits=np.stack(predictions), labels=data["labels"], sample_ids=data["sample_ids"])


def refit(output: Path) -> None:
    """固定来源袋装，不读取历史标签。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖模型")
    selected = json.loads((output / "selection_locked.json").read_text("utf-8"))
    config = selected["config"]
    data = data_for(("train", "selection", "calibration"))
    models = []
    for tr, _ in folds(data):
        subset = take(data, tr)
        x, scalers = transform(subset, config)
        weight = weights_for(subset["labels"], subset["source_groups"], 0., config["class_power"])
        model = LogisticRegression(C=config["C"]*2000/len(tr), max_iter=500, random_state=42).fit(x, subset["labels"], sample_weight=weight)
        models.append({"model": model, "scalers": scalers})
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "selection": selected, "training_groups": np.unique(data["source_groups"])}, handle)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """推理不读取金标或来源ID。"""
    data = dict(data)
    data["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    selected = bundle["selection"]
    probability = []
    for model in bundle["models"]:
        x, _ = transform(data, selected["config"], model["scalers"])
        probability.append(softmax(model["model"].decision_function(x)/selected["scale"], axis=1))
    return softmax(np.log(np.maximum(np.mean(probability, axis=0), 1e-12))+np.asarray(selected["bias"]), axis=1)


def evaluate(output: Path) -> None:
    """锁定后的历史诊断，不把公开预训练数据未知重合包装成无偏确认。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖历史结果")
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    data = data_for(("test",))
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("来源交叉")
    p = predict(bundle, data)
    baseline = np.load(OUTPUT / "final" / "historical_predictions.npz")["baseline"]
    result = {**metric(data["labels"], p.argmax(1)), **probability_metrics(data["labels"], p),
        "relative_gain": metric(data["labels"], p.argmax(1))["macro_f1"]/.6388-1,
        "target_achieved": metric(data["labels"], p.argmax(1))["macro_f1"] >= .70268,
        "bootstrap": bootstrap(data["labels"], p, baseline, data["source_groups"]),
        "upstream_target_overlap_verified": False, "adaptive_historical_diagnostic": True, "coverage": 1.}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    print(json.dumps(result), flush=True)


def main() -> None:
    """来源OOF选型与最终评测分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    global FEATURE_DIRECTORY
    if args.large:
        FEATURE_DIRECTORY = OUTPUT / "external_large_features"
    {"search": search, "refit": refit, "evaluate": evaluate}[args.action](OUTPUT / ("external_large_fusion" if args.large else "external_fusion"))


if __name__ == "__main__":
    main()
