"""锁定语音证据融合的五种子最终训练和全覆盖历史核验。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.special import softmax

from experiments import round10_evidence as evidence
from experiments import round10_speech_evidence as speech
from experiments.round10_cascade import member_predict
from experiments.round10_domain import weights_for
from experiments.round10_fusion import OUTPUT, load_npz, save, take
from experiments.round7_decision_oof import metric, probability_metrics


def refit(output: Path) -> None:
    """只用锁定配方训练，五个种子全部保留，不按历史成绩挑选。"""
    target = output / "model.pkl"
    if target.exists():
        raise ValueError("拒绝覆盖最终语音融合模型")
    lock_path = output / "selection_locked.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    speech.configure()
    data = speech.data_for(("train", "selection", "calibration"))
    inner = evidence.crossfit(data, 4)
    config = lock["members"][0]["config"]
    weights = weights_for(data["labels"], data["source_groups"], 0., config["class_power"])
    models = []
    for seed in lock["seeds"]:
        model = evidence.meta(config).set_params(random_seed=seed)
        model.fit(inner, data["labels"], sample_weight=weights)
        models.append(model)
        print(json.dumps({"meta_seed_trained": seed}), flush=True)
    experts = evidence.fit_experts(data)
    # 复用已按相同3579条训练数据拟合的固定线性专家，核对配方和温度。
    with (OUTPUT / "cascade" / "model.pkl").open("rb") as handle:
        earlier = pickle.load(handle)
    other = []
    for selected in lock["members"][1:]:
        member = next(m for m in earlier["members"] if m["stage"] == selected["stage"])
        if member["config"] != selected["config"] or member["scale"] != selected["scale"]:
            raise ValueError("复用专家与锁定配方不一致")
        other.append(member)
    if not np.array_equal(np.unique(data["source_groups"]), earlier["training_groups"]):
        raise ValueError("复用专家训练来源不一致")
    bundle = {"experts": experts, "meta_models": models, "other_members": other,
              "selection": lock, "training_groups": np.unique(data["source_groups"]),
              "training_sample_ids": data["sample_ids"]}
    with target.open("wb") as handle:
        pickle.dump(bundle, handle)
    np.savez_compressed(output / "final_crossfit_evidence.npz", evidence=inner,
                        sample_ids=data["sample_ids"], labels=data["labels"])
    save(output / "model_manifest.json", {"training_n": len(inner), "training_groups": len(bundle["training_groups"]),
         "splits": ["train", "selection", "calibration"], "seeds": lock["seeds"],
         "selection_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
         "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
         "meta_training": "four-fold source crossfit, no in-sample base predictions",
         "test_supervision_used_in_fit_or_inference": False})


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """返回全部种子预测；仅观测输入决定路由，真实标签可以完全省略。"""
    speech.configure()
    safe = dict(data)
    safe["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    selected = bundle["selection"]["winner"]
    weights = np.asarray(selected["weights"])
    cheap = member_predict(bundle["other_members"][1], safe)
    needed = cheap.max(1) < selected["route"]
    result = np.repeat(cheap[None], len(bundle["meta_models"]), axis=0)
    if needed.any():
        part = take(safe, np.flatnonzero(needed))
        features = evidence.expert_evidence(bundle["experts"], part) if weights[0] else None
        fixed = weights[2]*cheap[needed]
        if weights[1]:
            fixed = fixed + weights[1]*member_predict(bundle["other_members"][0], part)
        for i, model in enumerate(bundle["meta_models"]):
            result[i, needed] = fixed + (weights[0]*model.predict_proba(features) if weights[0] else 0.)
    adjusted = np.log(np.maximum(result, 1e-12))+np.asarray(selected["bias"])
    return softmax(adjusted, axis=-1), needed


def paired_interval(y: np.ndarray, new: np.ndarray, old: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """按来源配对重采样，比较双方五种子F1均值，保留自适应评测局限。"""
    unique, inverse = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(42).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=10000)
    scores = {}
    for name, probability in (("new", new), ("old", old)):
        matrices = np.zeros((len(probability), len(unique), 3, 3), dtype=int)
        for k, predictions in enumerate(probability.argmax(-1)):
            np.add.at(matrices[k], (inverse, y, predictions), 1)
        sampled = np.einsum("bg,kgij->bkij", draws, matrices)
        numerator = 2*np.diagonal(sampled, axis1=2, axis2=3)
        denominator = sampled.sum(2)+sampled.sum(3)
        values = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)
        scores[name] = values.mean((1, 2))
    return {"unit": "source video", "groups": len(unique), "replicates": 10000,
            "new_mean_f1_ci95": np.quantile(scores["new"], [.025, .975]).tolist(),
            "delta_ci95": np.quantile(scores["new"]-scores["old"], [.025, .975]).tolist(),
            "relative_gain_ci95": np.quantile(scores["new"]/scores["old"]-1, [.025, .975]).tolist(),
            "limitation": "Conditional intervals do not correct repeated adaptive historical-test exposure"}


def evaluate(output: Path) -> None:
    """锁定后核验原始821条，主验收指标固定为五种子F1均值。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖最终语音融合历史评测")
    speech.configure()
    data = speech.data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("训练和测试来源交叉")
    old = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
        raise ValueError("历史样本、顺序或标签发生变化")
    probability, needed = predict(bundle, data)
    runs = [{"seed": seed, **metric(data["labels"], p.argmax(1)), **probability_metrics(data["labels"], p)}
            for seed, p in zip(bundle["selection"]["seeds"], probability)]
    mean = float(np.mean([r["macro_f1"] for r in runs]))
    ensemble = probability.mean(0)
    result = {"primary_metric": "mean of five seed macro-F1 scores", "mean_macro_f1": mean,
              "std_macro_f1": float(np.std([r["macro_f1"] for r in runs])), "runs": runs,
              "ensemble": {**metric(data["labels"], ensemble.argmax(1)), **probability_metrics(data["labels"], ensemble)},
              "relative_gain_vs_reported_baseline": mean/.6388-1, "baseline_reported": .6388,
              "required_mean_macro_f1": .70268, "target_achieved": mean >= .70268,
              "coverage": 1., "costly_fraction": float(needed.mean()), "n": len(needed),
              "adaptive_historical_diagnostic": True,
              "bootstrap": paired_interval(data["labels"], probability, old["baseline"], data["source_groups"])}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=probability, ensemble=ensemble,
                        labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"],
                        costly_needed=needed, seeds=np.asarray(bundle["selection"]["seeds"]))
    print(json.dumps(result), flush=True)


def main() -> None:
    """分开训练和评测；已有产物禁止覆盖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("refit", "evaluate"))
    args = parser.parse_args()
    {"refit": refit, "evaluate": evaluate}[args.action](OUTPUT / "speech_final")


if __name__ == "__main__":
    main()
