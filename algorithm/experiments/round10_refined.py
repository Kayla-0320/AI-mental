"""来源开发最优偏置位于原网格边界，扩大范围并提高分辨率的对照。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.special import softmax

from experiments import round10_large_speech as large
from experiments import round10_robust_final as legacy
from experiments.round10_domain import weighted_f1
from experiments.round10_fusion import OUTPUT, load_data, load_npz, save
from experiments.round10_speech_final import paired_interval
from experiments.round7_decision_oof import metric, probability_metrics


def select(output: Path) -> None:
    """只用原来源OOF选两项截距，固定其他权重、专家和训练轮数。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖细网格类别校正")
    data = load_data()
    old_lock = json.loads((OUTPUT / "large_final" / "selection_locked.json").read_text("utf-8"))
    source = load_npz(OUTPUT / "large_final" / "selected_oof.npz")
    if not np.array_equal(source["sample_ids"], data["sample_ids"]) or not np.array_equal(source["labels"], data["labels"]):
        raise ValueError("来源OOF样本或标签不一致")
    grid = np.round(np.arange(-.5, .750001, .05), 8)
    save(output / "preregister.json", {"hypothesis": "large_final negative intercept optimum0.25 reaches prior grid upper boundary0.25",
        "absolute_bias_grid": grid.tolist(), "positive_reference_bias": 0., "parameters": 2,
        "selection": ".5 pooled source OOF F1+.5 source-balanced F1, norm toward0 breaks exact ties",
        "base_model": "large_final, unchanged weights and all five seeds", "training_scope_unchanged": 3579,
        "primary_acceptance": "five seed mean macro-F1 >=0.70268 on unchanged821,100%coverage",
        "test_parameters_chosen": False, "adaptive_historical_diagnostic": True})
    z = np.log(np.maximum(source["probabilities"], 1e-12))
    rows = []
    for negative, neutral in itertools.product(grid, grid):
        bias = [float(negative), float(neutral), 0.]
        pred = (z+np.asarray(bias)).argmax(1)
        m = metric(data["labels"], pred)
        s = weighted_f1(data["labels"], pred, data["source_groups"])
        rows.append({"bias": bias, "metrics": m, "source_balanced_f1": s, "selection_score": .5*(m["macro_f1"]+s)})
    chosen = max(rows, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"], -sum(abs(v) for v in row["bias"])))
    promoted = chosen["selection_score"] > old_lock["winner"]["selection_score"]+1e-12
    lock = {"winner": chosen, "old_bias": old_lock["winner"]["bias"], "promoted": promoted,
            "seeds": [42, 43, 44, 45, 46], "base_recipe": old_lock, "candidate_count": len(rows)}
    save(output / "selection_locked.json", lock)
    save(output / "candidates.json", rows)
    np.savez_compressed(output / "selected_oof.npz", probabilities=softmax(z+chosen["bias"], axis=1),
                        labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps({"winner": chosen, "promoted": promoted}), flush=True)


def fit(output: Path) -> None:
    """复用完全相同的五种子模型，只新增已在来源OOF锁定的截距。"""
    target = output / "model.pkl"
    if target.exists():
        raise ValueError("拒绝覆盖细化截距模型")
    lock_path = output / "selection_locked.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    if not lock["promoted"]:
        raise ValueError("来源开发没有收益，禁止追加历史选型")
    source = OUTPUT / "large_final" / "model.pkl"
    with source.open("rb") as handle:
        earlier = pickle.load(handle)
    bundle = {"model": earlier, "old_bias": lock["old_bias"], "new_bias": lock["winner"]["bias"],
              "training_sample_ids": earlier["training_sample_ids"], "training_groups": earlier["training_groups"]}
    with target.open("wb") as handle:
        pickle.dump(bundle, handle)
    save(output / "model_manifest.json", {"training_n": len(bundle["training_sample_ids"]), "base_model": str(source),
        "base_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "seeds": lock["seeds"],
        "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(), "only_new_parameters": lock["winner"]["bias"],
        "test_labels_used_in_fit_or_inference": False})


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """固定校正作用于每个种子，真实标签不参与预测。"""
    probability, needed = legacy.predict(bundle["model"], data)
    z = np.log(np.maximum(probability, 1e-12))-np.asarray(bundle["old_bias"])+np.asarray(bundle["new_bias"])
    return softmax(z, axis=-1), needed


def evaluate(output: Path) -> None:
    """只评测已锁定的唯一偏置，不在历史集上挑网格点。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖细化截距历史诊断")
    data = large.data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if set(bundle["training_groups"]) & set(data["source_groups"]):
        raise ValueError("来源交叉")
    old = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
        raise ValueError("历史样本或标签改变")
    p, needed = predict(bundle, data)
    runs = [{"seed": seed, **metric(data["labels"], row.argmax(1)), **probability_metrics(data["labels"], row)}
            for seed, row in zip(range(42, 47), p)]
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
                        costly_needed=needed, seeds=np.arange(42, 47))
    print(json.dumps(result), flush=True)


def main() -> None:
    """来源选择、落盘、唯一历史核验分开执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select", "fit", "evaluate"))
    args = parser.parse_args()
    output = OUTPUT / "refined_final"
    output.mkdir(exist_ok=True)
    {"select": select, "fit": fit, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
