"""组合三档与五档单模态证据，所有融合设置只从来源OOF选择。"""
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

from experiments import round10_evidence as evidence
from experiments import round10_speech_evidence as speech
from experiments.round10_cascade import member_predict
from experiments.round10_domain import bias_select, weights_for
from experiments.round10_fusion import OUTPUT, load_npz, save, take
from experiments.round10_speech_final import paired_interval
from experiments.round7_decision_oof import metric, probability_metrics

FAMILY_STAGES = ("speech_evidence", "speech_evidence5")
FINAL_DIRECTORY = "robust_final"
REUSE_COARSE: Path | None = OUTPUT / "speech_final" / "model.pkl"
data_for = speech.data_for
configure = speech.configure


def select(output: Path) -> None:
    """细粒度标签仅改变专家监督，不改变主任务三类标签或历史样本。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖强度证据融合选择")
    configure()
    data = data_for(("train", "selection"))
    members, probabilities = [], []
    for stage in (*FAMILY_STAGES, "aux_fusion"):
        rows = json.loads((OUTPUT / stage / "search_complete.json").read_text("utf-8"))["results"]
        index = max(range(len(rows)), key=lambda j: rows[j].get("selection_score", rows[j]["metrics"]["macro_f1"]))
        source = load_npz(OUTPUT / stage / "candidates_oof.npz")
        if not np.array_equal(source["sample_ids"], data["sample_ids"]):
            raise ValueError("来源OOF顺序不一致")
        z = source["logits"][index]
        scale = max(.1, float(z.std(1).mean())) if stage == "aux_fusion" else 1.
        probabilities.append(softmax(z/scale, axis=1))
        members.append({"stage": stage, "index": index, "config": rows[index]["config"], "scale": scale})
    grid = [np.array([a, b, 4-a-b])/4 for a in range(5) for b in range(5-a) if a+b]
    save(output / "preregister.json", {"members": members, "quarter_step_weights": [w.tolist() for w in grid],
        "routes": [.7, .8, .9, .95, 1.01], "seeds": [42, 43, 44, 45, 46],
        "selection": ".5 pooled source OOF F1 + .5 source-balanced F1; lower invocation breaks ties",
        "acceptance": "five seed mean macro-F1 >=0.70268 on unchanged821 full coverage",
        "prior_historical_exposure": True, "fine_bins": [-.5, -.1, .1, .5]})
    candidates, best = [], None
    for weight, threshold in itertools.product(grid, (.7, .8, .9, .95, 1.01)):
        needed = probabilities[2].max(1) < threshold
        p = probabilities[2].copy()
        p[needed] = sum(w*proba[needed] for w, proba in zip(weight, probabilities))
        row = {"weights": weight.tolist(), "route": threshold, "costly_fraction": float(needed.mean()),
               **bias_select(data, np.log(np.maximum(p, 1e-12)))}
        candidates.append(row)
        key = row["selection_score"], row["metrics"]["macro_f1"], -row["costly_fraction"]
        if best is None or key > best[0]:
            best = key, row, p
    save(output / "selection_locked.json", {"members": members, "winner": best[1], "seeds": [42, 43, 44, 45, 46]})
    save(output / "candidates.json", candidates)
    np.savez_compressed(output / "selected_oof.npz", probabilities=best[2], sample_ids=data["sample_ids"], labels=data["labels"])
    print(json.dumps({"locked": best[1]}), flush=True)


def fit_family(data: dict[str, np.ndarray], config: dict[str, Any], classes: int,
               seeds: list[int]) -> dict[str, Any]:
    """每个专家监督粒度独立交叉拟合，主任务始终三类。"""
    evidence.EXPERT_CLASSES = classes
    try:
        inner = evidence.crossfit(data, 4)
        weights = weights_for(data["labels"], data["source_groups"], 0., config["class_power"])
        models = []
        for seed in seeds:
            models.append(evidence.meta(config).set_params(random_seed=seed).fit(inner, data["labels"], sample_weight=weights))
            print(json.dumps({"expert_classes": classes, "meta_seed": seed}), flush=True)
        return {"experts": evidence.fit_experts(data), "meta_models": models}
    finally:
        evidence.EXPERT_CLASSES = 3


def refit(output: Path) -> None:
    """复用相同三档专家，新增五档专家并保留全部五个种子。"""
    target = output / "model.pkl"
    if target.exists():
        raise ValueError("拒绝覆盖强度证据融合模型")
    lock_path = output / "selection_locked.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    configure()
    data = data_for(("train", "selection", "calibration"))
    if REUSE_COARSE is not None:
        with REUSE_COARSE.open("rb") as handle:
            earlier = pickle.load(handle)
        if earlier["selection"]["members"][0]["config"] != lock["members"][0]["config"]:
            raise ValueError("三档模型配方不一致")
        coarse = {"experts": earlier["experts"], "meta_models": earlier["meta_models"]}
    else:
        coarse = fit_family(data, lock["members"][0]["config"], 3, lock["seeds"])
    fine = fit_family(data, lock["members"][1]["config"], 5, lock["seeds"])
    with (OUTPUT / "cascade" / "model.pkl").open("rb") as handle:
        cheap_source = pickle.load(handle)
    cheap = next(member for member in cheap_source["members"] if member["stage"] == "aux_fusion")
    if cheap["config"] != lock["members"][2]["config"] or cheap["scale"] != lock["members"][2]["scale"]:
        raise ValueError("轻量专家配方不一致")
    if not np.array_equal(np.unique(data["source_groups"]), cheap_source["training_groups"]):
        raise ValueError("轻量专家训练来源不一致")
    bundle = {"families": [coarse, fine], "cheap": cheap, "selection": lock,
              "training_groups": np.unique(data["source_groups"]), "training_sample_ids": data["sample_ids"]}
    with target.open("wb") as handle:
        pickle.dump(bundle, handle)
    save(output / "model_manifest.json", {"training_n": len(data["labels"]), "groups": len(bundle["training_groups"]),
        "seeds": lock["seeds"], "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(), "fine_bins": [-.5, -.1, .1, .5],
        "training_splits": ["train", "selection", "calibration"], "test_labels_used_in_fit_or_inference": False})


def predict(bundle: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """无标签预测保持全部样本，按置信路由执行正权重的证据读出器。"""
    configure()
    safe = dict(data)
    safe["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    selected = bundle["selection"]["winner"]
    cheap = member_predict(bundle["cheap"], safe)
    needed = cheap.max(1) < selected["route"]
    result = np.repeat(cheap[None], len(bundle["selection"]["seeds"]), axis=0)
    if needed.any():
        part = take(safe, np.flatnonzero(needed))
        result[:, needed] = selected["weights"][2]*cheap[needed]
        for weight, family in zip(selected["weights"][:2], bundle["families"]):
            if not weight:
                continue
            features = evidence.expert_evidence(family["experts"], part)
            if bundle.get("raw_foundation_evidence", False):
                from experiments.round10_direct_evidence import raw_evidence
                features = np.column_stack([features, raw_evidence(part)])
            for i, model in enumerate(family["meta_models"]):
                result[i, needed] += weight*model.predict_proba(features)
    return softmax(np.log(np.maximum(result, 1e-12))+np.asarray(selected["bias"]), axis=-1), needed


def evaluate(output: Path) -> None:
    """使用已锁定的主验收标准，集成成绩单列而非替代五种子均值。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖强度证据历史核验")
    configure()
    data = data_for(("test",))
    with (output / "model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if set(data["source_groups"]) & set(bundle["training_groups"]):
        raise ValueError("来源交叉")
    old = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
        raise ValueError("历史ID或标签发生变化")
    p, needed = predict(bundle, data)
    runs = [{"seed": seed, **metric(data["labels"], row.argmax(1)), **probability_metrics(data["labels"], row)}
            for seed, row in zip(bundle["selection"]["seeds"], p)]
    mean = float(np.mean([row["macro_f1"] for row in runs]))
    ensemble = p.mean(0)
    result = {"primary_metric": "five seed mean macro-F1", "mean_macro_f1": mean,
        "std_macro_f1": float(np.std([row["macro_f1"] for row in runs])), "runs": runs,
        "ensemble": {**metric(data["labels"], ensemble.argmax(1)), **probability_metrics(data["labels"], ensemble)},
        "baseline_reported": .6388, "relative_gain_vs_reported_baseline": mean/.6388-1,
        "required_mean_macro_f1": .70268, "target_achieved": mean >= .70268,
        "coverage": 1., "n": len(needed), "costly_fraction": float(needed.mean()), "adaptive_historical_diagnostic": True,
        "bootstrap": paired_interval(data["labels"], p, old["baseline"], data["source_groups"])}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, ensemble=ensemble,
                        labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"],
                        costly_needed=needed, seeds=np.asarray(bundle["selection"]["seeds"]))
    print(json.dumps(result), flush=True)


def main() -> None:
    """开发选型、最终训练与历史评测分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select", "refit", "evaluate"))
    args = parser.parse_args()
    output = OUTPUT / FINAL_DIRECTORY
    output.mkdir(exist_ok=True)
    {"select": select, "refit": refit, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
