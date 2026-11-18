"""保留703条来源隔离样本做输出适配，避免最终重拟合置信分布漂移。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments import round10_robust_final as fusion
from experiments import round10_large_speech as large
from experiments.round10_domain import weights_for, weighted_f1
from experiments.round10_fusion import OUTPUT, save, load_npz, folds, take
from experiments.round10_represented_fusion import transform
from experiments.round10_speech_final import paired_interval
from experiments.round7_decision_oof import metric, probability_metrics


def calibrator(config: dict[str, float], n: int) -> LogisticRegression:
    """小维数有监督读出器；正则化按样本数归一，不冒充无偏概率校准。"""
    return LogisticRegression(C=config["C"]*703/n, max_iter=500, random_state=42)


def adaptation_oof(features: np.ndarray, data: dict[str, np.ndarray],
                   config: dict[str, float]) -> np.ndarray:
    """适配器的每条选型预测来自未见该来源的适配训练集。"""
    result = np.full_like(features, np.nan)
    for tr, va in folds(data, 3):
        weights = weights_for(data["labels"][tr], data["source_groups"][tr], 0., config["class_power"])
        for i in range(len(features)):
            model = calibrator(config, len(tr)).fit(features[i, tr], data["labels"][tr], sample_weight=weights)
            result[i, va] = model.predict_proba(features[i, va])
    if not np.isfinite(result).all():
        raise ValueError("适配来源OOF覆盖不完整")
    return result


def fit_select(output: Path) -> None:
    """基础模型不见703标签；读出适配再用三折来源OOF选型。"""
    if (output / "model.pkl").exists() or (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖保留来源适配阶段")
    large.configure()
    lock = json.loads((OUTPUT / "large_final" / "selection_locked.json").read_text("utf-8"))
    grid = [{"C": c, "class_power": power} for c, power in itertools.product((.1, 1., 10.), (0., .5, 1.))]
    save(output / "preregister.json", {"base_recipe": lock, "base_training_splits": ["train", "selection"],
        "readout_adaptation_split": "calibration703, excluded from every base/cheap expert and meta classifier",
        "readout_selection": "3 source folds on703, five-seed mean of .5 pooled +.5 source-balanced F1",
        "configs": grid, "acceptance": "five seed mean macro-F1 >=0.70268, unchanged821 full coverage",
        "protocol_change": "prior stages merged703 into base training; this stage reserves703 for readout adaptation",
        "previous_historical_mean": .6943160592005289, "adaptive_historical_diagnostic": True,
        "readout_probability_calibration_not_guaranteed": True})
    training = large.data_for(("train", "selection"))
    calibration = large.data_for(("calibration",))
    if set(training["source_groups"]) & set(calibration["source_groups"]):
        raise ValueError("基础训练与适配来源交叉")
    families = [fusion.fit_family(training, member["config"], classes, lock["seeds"])
                for member, classes in zip(lock["members"][:2], (3, 5))]
    cheap_recipe = lock["members"][2]
    models = []
    for tr, _ in folds(training):
        subset = take(training, tr)
        x, scaler = transform(subset, cheap_recipe["config"])
        weights = weights_for(subset["labels"], subset["source_groups"], 0., 1.)
        model = LogisticRegression(C=cheap_recipe["config"]["C"]*2157/len(tr), max_iter=500, random_state=42)
        model.fit(x, subset["labels"], sample_weight=weights)
        models.append({"model": model, "scaler": scaler})
    base = {"families": families, "cheap": {**cheap_recipe, "models": models}, "selection": lock}
    p, _ = fusion.predict(base, calibration)
    features = np.log(np.maximum(p, 1e-12))
    candidates, all_oof = [], []
    for config in grid:
        oof = adaptation_oof(features, calibration, config)
        scores = [metric(calibration["labels"], row.argmax(1))["macro_f1"] for row in oof]
        source = [weighted_f1(calibration["labels"], row.argmax(1), calibration["source_groups"]) for row in oof]
        candidates.append({"config": config, "macro_f1": float(np.mean(scores)), "source_balanced_f1": float(np.mean(source)),
                           "selection_score": .5*float(np.mean(scores)+np.mean(source))})
        all_oof.append(oof)
    index = max(range(len(grid)), key=lambda i: (candidates[i]["selection_score"], candidates[i]["macro_f1"]))
    chosen = candidates[index]
    save(output / "selection_locked.json", {"base_recipe": lock, "adaptation": chosen, "seeds": lock["seeds"]})
    save(output / "search_complete.json", {"results": candidates})
    np.savez_compressed(output / "calibration_oof.npz", probabilities=np.stack(all_oof), unadapted_probabilities=p,
                        sample_ids=calibration["sample_ids"], source_groups=calibration["source_groups"], labels=calibration["labels"])
    weights = weights_for(calibration["labels"], calibration["source_groups"], 0., chosen["config"]["class_power"])
    adapters = [calibrator(chosen["config"], 703).fit(row, calibration["labels"], sample_weight=weights) for row in features]
    bundle = {"base": base, "adapters": adapters, "selection": chosen, "seeds": lock["seeds"],
              "training_groups": np.unique(training["source_groups"]), "training_sample_ids": training["sample_ids"],
              "adaptation_groups": np.unique(calibration["source_groups"]), "adaptation_sample_ids": calibration["sample_ids"]}
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    save(output / "model_manifest.json", {"base_training_n": 2876, "adaptation_n": 703, "seeds": lock["seeds"],
        "model_sha256": hashlib.sha256((output / "model.pkl").read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256((output / "selection_locked.json").read_bytes()).hexdigest(),
        "test_labels_used_in_fit_or_inference": False})
    print(json.dumps({"adaptation_selection": chosen}), flush=True)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """适配器只读取模型概率，不读取推理样本真实标签和来源。"""
    large.configure()
    p, needed = fusion.predict(bundle["base"], data)
    return np.stack([adapter.predict_proba(np.log(np.maximum(row, 1e-12)))
                     for row, adapter in zip(p, bundle["adapters"])]), needed


def evaluate(output: Path) -> None:
    """独立保存新协议的一次历史诊断，不覆盖此前失败结果。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖保留来源适配评测")
    data = large.data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if (set(bundle["training_groups"]) | set(bundle["adaptation_groups"])) & set(data["source_groups"]):
        raise ValueError("训练或适配与历史来源交叉")
    old = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
        raise ValueError("历史样本或标签改变")
    p, needed = predict(bundle, data)
    runs = [{"seed": seed, **metric(data["labels"], row.argmax(1)), **probability_metrics(data["labels"], row)}
            for seed, row in zip(bundle["seeds"], p)]
    mean = float(np.mean([row["macro_f1"] for row in runs]))
    ensemble = p.mean(0)
    result = {"primary_metric": "five seed mean macro-F1", "mean_macro_f1": mean,
        "std_macro_f1": float(np.std([row["macro_f1"] for row in runs])), "runs": runs,
        "ensemble": {**metric(data["labels"], ensemble.argmax(1)), **probability_metrics(data["labels"], ensemble)},
        "baseline_reported": .6388, "relative_gain_vs_reported_baseline": mean/.6388-1,
        "required_mean_macro_f1": .70268, "target_achieved": mean >= .70268, "coverage": 1., "n": len(needed),
        "costly_fraction": float(needed.mean()), "adaptive_historical_diagnostic": True,
        "bootstrap": paired_interval(data["labels"], p, old["baseline"], data["source_groups"])}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, ensemble=ensemble,
                        labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"],
                        costly_needed=needed, seeds=np.asarray(bundle["seeds"]))
    print(json.dumps(result), flush=True)


def main() -> None:
    """拟合适配和历史诊断分开执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit_select", "evaluate"))
    args = parser.parse_args()
    output = OUTPUT / "holdout_final"
    output.mkdir(exist_ok=True)
    {"fit_select": fit_select, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
