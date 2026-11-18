"""按开发OOF锁定的两专家按需调用与软投票，保存真实路由覆盖。"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from experiments.round10_fusion import OUTPUT, load_npz, load_data, folds, take, save
from experiments.round10_domain import bias_select, weights_for
from experiments import round10_llm_fusion as llm
from experiments import round10_external_fusion as external
from experiments.round10_represented_fusion import transform as base_transform
from experiments.round7_decision_oof import metric, probability_metrics
from experiments.round10_final import bootstrap


def select(output: Path) -> None:
    """只使用来源OOF，硬路由阈值与投票权重均在测试前锁定。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖最终级联选择")
    llm.configure()
    data = load_data()
    probabilities, components = [], []
    for stage in ("llm_fusion", "aux_fusion"):
        rows = json.loads((OUTPUT / stage / "search_complete.json").read_text("utf-8"))["results"]
        i = max(range(len(rows)), key=lambda i: rows[i].get("selection_score", rows[i]["metrics"]["macro_f1"]))
        source = load_npz(OUTPUT / stage / "candidates_oof.npz")
        if not np.array_equal(source["sample_ids"], data["sample_ids"]):
            raise ValueError("专家OOF未对齐")
        z = source["logits"][i]
        scale = max(.1, float(z.std(1).mean()))
        probabilities.append(softmax(z/scale, axis=1))
        components.append({"stage": stage, "index": i, "config": rows[i]["config"], "scale": scale})
    save(output / "preregister.json", {"phase": 5, "members": components,
         "weights": [.25, .5, .75, 1.], "route_thresholds": [.7, .8, .9, .95, 1.01],
         "selection": ".5 pooled OOF F1 + .5 source-balanced F1; lower costly coverage breaks ties",
         "refit": "four source folds on3579; LR regularization normalized to OOF training size",
         "status": "adaptive historical diagnostic; all earlier negative results retained"})
    candidates, best = [], None
    for weight, threshold in itertools.product((.25, .5, .75, 1.), (.7, .8, .9, .95, 1.01)):
        needed = probabilities[1].max(1) < threshold
        p = probabilities[1].copy()
        p[needed] = (1-weight)*p[needed]+weight*probabilities[0][needed]
        result = bias_select(data, np.log(np.maximum(p, 1e-12)))
        row = {"llm_weight": weight, "route_threshold": threshold, "llm_fraction": float(needed.mean()), **result}
        candidates.append(row)
        key = (row["selection_score"], row["metrics"]["macro_f1"], -row["llm_fraction"])
        if best is None or key > best[0]:
            best = key, row, p, needed
    lock = {"components": components, "winner": best[1]}
    save(output / "selection_locked.json", lock)
    save(output / "candidates.json", candidates)
    np.savez_compressed(output / "selected_oof.npz", probabilities=best[2], llm_needed=best[3], labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps(best[1]), flush=True)


def refit(output: Path) -> None:
    """训练阶段不读取历史标签，低置信样本调用语义专家的规则保持锁定。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖级联模型")
    llm.configure()
    lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
    data = llm.data_for(("train", "selection", "calibration"))
    fitted = []
    for member in lock["components"]:
        config = member["config"]
        members = []
        for tr, _ in folds(data):
            subset = take(data, tr)
            if member["stage"] == "llm_fusion":
                x, scaler = external.transform(subset, config)
                c = config["C"]*2000/len(tr)
            else:
                x, scaler = base_transform(subset, config)
                c = config["C"]*(.75*2876)/len(tr)
            weights = weights_for(subset["labels"], subset["source_groups"], 0., 1.)
            model = LogisticRegression(C=c, max_iter=500, random_state=42).fit(x, subset["labels"], sample_weight=weights)
            members.append({"model": model, "scaler": scaler})
        fitted.append({**member, "models": members})
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"members": fitted, "selection": lock["winner"], "training_groups": np.unique(data["source_groups"])}, handle)


def member_predict(member: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """训练统计冻结，目标标签不进入推理。"""
    data = dict(data)
    data["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    predictions = []
    for model in member["models"]:
        transform = external.transform if member["stage"] == "llm_fusion" else base_transform
        x, _ = transform(data, member["config"], model["scaler"])
        predictions.append(softmax(model["model"].decision_function(x)/member["scale"], axis=1))
    return np.mean(predictions, axis=0)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """先执行轻量专家，只在需要行上执行昂贵表征的分类头。"""
    llm.configure()
    valid = data["modality_mask"][:, 0] & (data["quality"][:, 0] > 0)
    if not valid.all():
        raise ValueError("当前研究配方需要真实可用文字")
    expensive, cheap = bundle["members"]
    probability = member_predict(cheap, data)
    needed = probability.max(1) < bundle["selection"]["route_threshold"]
    if needed.any():
        if "llm_last" not in data:
            raise ValueError("需要语义专家的行缺少表征，不能用默认值代替")
        expensive_p = member_predict(expensive, take(data, np.flatnonzero(needed)))
        weight = bundle["selection"]["llm_weight"]
        probability[needed] = (1-weight)*probability[needed]+weight*expensive_p
    adjusted = np.log(np.maximum(probability, 1e-12))+np.asarray(bundle["selection"]["bias"])
    return softmax(adjusted, axis=1), needed


def evaluate(output: Path) -> None:
    """一次锁定评测，报告全覆盖成绩及实际语义专家调用比例。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖级联评测")
    llm.configure()
    data = llm.data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("来源跨划分")
    p, needed = predict(bundle, data)
    baseline = load_npz(OUTPUT / "final" / "historical_predictions.npz")["baseline"]
    result = {**metric(data["labels"], p.argmax(1)), **probability_metrics(data["labels"], p),
        "relative_gain": metric(data["labels"], p.argmax(1))["macro_f1"]/.6388-1,
        "target_achieved": metric(data["labels"], p.argmax(1))["macro_f1"] >= .70268,
        "llm_fraction": float(needed.mean()), "coverage": 1., "adaptive_historical_diagnostic": True,
        "bootstrap": bootstrap(data["labels"], p, baseline, data["source_groups"])}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, llm_needed=needed,
                        labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    print(json.dumps(result), flush=True)


def main() -> None:
    """明确分离最终配方的选择、拟合和评测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select", "refit", "evaluate"))
    args = parser.parse_args()
    {"select": select, "refit": refit, "evaluate": evaluate}[args.action](OUTPUT / "cascade")


if __name__ == "__main__":
    main()
