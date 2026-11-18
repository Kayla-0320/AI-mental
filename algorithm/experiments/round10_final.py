"""锁定配方的最终拟合、完整覆盖评测、压力诊断和来源组区间。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
from scipy.special import softmax

from experiments.round10_fusion import OUTPUT, ROOT, save, make_model, load_data
from experiments.round10_represented_fusion import data_for, transform
from experiments.round7_decision_oof import metric, probability_metrics


def fit(scope: str, splits: tuple[str, ...]) -> None:
    """超参数已锁定；703仅在最终训练scope中作为新增监督样本。"""
    output = OUTPUT / "final"
    target = output / f"{scope}.pkl"
    if target.exists():
        raise ValueError("拒绝覆盖最终模型")
    lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
    data = data_for(splits, True)
    components = []
    for weight, component in zip(lock["winner"]["weights"], lock["components"]):
        if not weight:
            continue
        if component["stage"] not in ("aux_fusion", "deep_fusion"):
            raise ValueError("当前锁定配方包含未实现的最终拟合族")
        config = component["candidate"]["config"]
        x, scalers = transform(data, config)
        model = make_model(config).fit(x, data["labels"])
        components.append({"weight": weight, "config": config, "scalers": scalers,
                           "model": model, "logit_scale": component["logit_scale"]})
    bundle = {"components": components, "bias": lock["winner"]["bias"],
              "selection_sha256": hashlib.sha256((output / "selection_locked.json").read_bytes()).hexdigest(),
              "training_sample_ids": data["sample_ids"], "training_groups": np.unique(data["source_groups"]),
              "scope": scope, "uncalibrated": True}
    with target.open("wb") as handle:
        pickle.dump(bundle, handle)
    save(output / f"{scope}_manifest.json", {"training_n": len(data["labels"]), "training_splits": splits,
         "groups": len(bundle["training_groups"]), "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
         "selection_sha256": bundle["selection_sha256"], "scope": scope,
         "label_digest": hashlib.sha256(data["labels"].tobytes()).hexdigest(), "uncalibrated": True})
    print(json.dumps({"fitted": scope, "n": len(data["labels"])}), flush=True)


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """预测与真实标签无关；推理调用方可以完全不提供labels。"""
    values = dict(data)
    values["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    probabilities = np.zeros((len(data["sample_ids"]), 3))
    for component in bundle["components"]:
        x, _ = transform(values, component["config"], component["scalers"])
        logits = component["model"].decision_function(x)
        probabilities += component["weight"] * softmax(logits / component["logit_scale"], axis=1)
    adjusted = np.log(np.maximum(probabilities, 1e-12)) + np.asarray(bundle["bias"])
    return softmax(adjusted, axis=1)


def bootstrap(y: np.ndarray, new: np.ndarray, old: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """按来源视频配对重采样，旧分数为五种子F1的均值而非概率集成。"""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=10000)
    samples = {}
    for name, predictions in (("new", new[None].argmax(-1)), ("old", old.argmax(-1))):
        cm = np.zeros((len(predictions), len(unique), 3, 3), int)
        for k, pred in enumerate(predictions):
            np.add.at(cm[k], (inverse, y, pred), 1)
        matrix = np.einsum("bg,kgij->bkij", draws, cm)
        tp = np.diagonal(matrix, axis1=2, axis2=3)
        denominator = matrix.sum(2) + matrix.sum(3)
        f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0).mean(2)
        samples[name] = f1.mean(1)
    return {"unit": "source video", "groups": len(unique), "replicates": 10000,
            "new_f1_ci95": np.quantile(samples["new"], [.025, .975]).tolist(),
            "delta_ci95": np.quantile(samples["new"]-samples["old"], [.025, .975]).tolist(),
            "relative_gain_ci95": np.quantile(samples["new"]/samples["old"]-1, [.025, .975]).tolist(),
            "limitation": "Conditional diagnostic intervals do not remove prior repeated historical-test exposure"}


def stress(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> dict[str, Any]:
    """保持全部测试样本，不用丢弃困难样本换取鲁棒性分数。"""
    result = {}
    for scenario in ("without_face", "without_audio", "without_visual", "face_swapped", "face_noise"):
        part = dict(data)
        if scenario == "without_face":
            part["face_summary"] = np.zeros_like(data["face_summary"])
            part["deep_face"] = np.zeros_like(data["deep_face"])
            part["coverage"] = np.zeros_like(data["coverage"])
        elif scenario.startswith("without_"):
            j = ("text", "audio", "visual").index(scenario.removeprefix("without_"))
            part["modality_mask"] = data["modality_mask"].copy()
            part["quality"] = data["quality"].copy()
            part["modality_mask"][:, j] = False
            part["quality"][:, j] = 0
        elif scenario == "face_swapped":
            order = np.random.default_rng(42).permutation(len(data["labels"]))
            for name in ("face_summary", "deep_face", "coverage"):
                part[name] = data[name][order]
        else:
            rng = np.random.default_rng(42)
            part["face_summary"] = data["face_summary"].copy()
            scale = bundle["components"][0]["scalers"]["face_summary"]["scaler"].scale_[:24]
            part["face_summary"][:, :24] += rng.normal(size=(len(data["labels"]), 24)) * scale * .5
        prediction = predict(bundle, part)
        result[scenario] = metric(data["labels"], prediction.argmax(1))
    return result


def evaluate() -> None:
    """选型与模型文件存在后，才执行一次历史集评测。"""
    output = OUTPUT / "final"
    if (output / "results.json").exists():
        raise ValueError("历史评测已完成，禁止反复覆盖或改选")
    data = data_for(("test",), True)
    bundles = {}
    for scope in ("development2876", "final3579"):
        with (output / f"{scope}.pkl").open("rb") as handle:
            bundles[scope] = pickle.load(handle)
        if set(bundles[scope]["training_groups"]) & set(data["source_groups"]):
            raise ValueError("训练与测试来源交叉")
    old_path = ROOT.parent / "output" / "round9_review_20261116" / "replay_predictions.npz"
    with np.load(old_path) as old:
        if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
            raise ValueError("历史标签或顺序发生变化")
        reference = old["moe"].copy()
    baseline = float(np.mean([metric(data["labels"], p.argmax(1))["macro_f1"] for p in reference]))
    report = {"baseline_actual_five_seed_mean": baseline, "baseline_reported": .6388,
              "required_score": .70268, "coverage": 1., "selected_before_evaluation": True,
              "historical_test_repeatedly_observed_in_prior_rounds": True, "models": {}}
    predictions = {}
    for scope, bundle in bundles.items():
        probabilities = predict(bundle, data)
        predictions[scope] = probabilities
        score = metric(data["labels"], probabilities.argmax(1))
        report["models"][scope] = {**score, **probability_metrics(data["labels"], probabilities),
                                  "relative_gain_vs_reported_baseline": score["macro_f1"]/.6388-1}
    final = predictions["final3579"]
    report["target_point_estimate_achieved"] = report["models"]["final3579"]["macro_f1"] >= .70268
    report["bootstrap"] = bootstrap(data["labels"], final, reference, data["source_groups"])
    report["stress"] = stress(bundles["final3579"], data)
    report["selection_curve"] = []
    for coverage in (1., .9, .8, .7):
        order = np.argsort(-final.max(1))[:round(len(final)*coverage)]
        report["selection_curve"].append({"coverage": len(order)/len(final),
            **metric(data["labels"][order], final[order].argmax(1)),
            "class_coverage": [float(np.mean(np.isin(np.flatnonzero(data["labels"] == k), order))) for k in range(3)]})
    np.savez_compressed(output / "historical_predictions.npz", **predictions, labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"], baseline=reference)
    save(output / "results.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def matched_baseline() -> None:
    """额外训练规模对照固定三轮，不用历史结果选轮次或种子。"""
    import torch
    from experiments.round9_moe_baselines import train_once, seed_all, forward
    from experiments.round2_moe_train import metrics as old_metrics
    output = OUTPUT / "final"
    if (output / "matched_moe.json").exists():
        raise ValueError("拒绝覆盖匹配对照")

    def convert(part: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(part)
        result["text"] = part["roberta_mean"]
        result["face"] = part["face_summary"]
        present = part["face_summary"][:, 24:25] > 0
        result["modality_mask"] = np.column_stack([part["modality_mask"], present])
        result["quality"] = np.column_stack([part["quality"], present.astype(np.float32)])
        return result

    data = {"train": convert(load_data(("train", "selection", "calibration"))),
            "selection": convert(load_data(("selection",))), "test": convert(load_data(("test",)))}
    results, predictions = [], []
    for seed in range(42, 47):
        seed_all()
        model, info, batches = train_once("moe", seed, 64, data, 3, epochs=3, use_last_checkpoint=True)
        model.eval()
        with torch.inference_mode():
            logits, _, _, valid = forward(model, batches["test"], torch.arange(len(data["test"]["labels"])))
        if not valid.all():
            raise ValueError("匹配基线丢弃样本")
        probabilities = logits.softmax(1).numpy()
        results.append({"seed": seed, **metric(data["test"]["labels"], probabilities.argmax(1))})
        predictions.append(probabilities)
        torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": 3}, output / f"matched_moe_seed{seed}.pt")
    save(output / "matched_moe.json", {"training_n": len(data["train"]["labels"]), "epochs": 3,
         "selection_used_for_checkpoint": False, "runs": results,
         "mean_macro_f1": float(np.mean([row["macro_f1"] for row in results])),
         "limitation": "Fixed-three-epoch data-size control, not a claim of each architecture's independently optimized maximum"})
    np.savez_compressed(output / "matched_moe_predictions.npz", probabilities=np.stack(predictions),
                        sample_ids=data["test"]["sample_ids"], labels=data["test"]["labels"])


def main() -> None:
    """最终拟合和一次性评测分离执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "evaluate", "matched"))
    args = parser.parse_args()
    if args.action == "fit":
        fit("development2876", ("train", "selection"))
        fit("final3579", ("train", "selection", "calibration"))
    elif args.action == "matched":
        matched_baseline()
    else:
        evaluate()


if __name__ == "__main__":
    main()
