"""用内层来源折专家袋装对齐元模型的训练和推理证据分布。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.special import softmax

from experiments import round10_evidence as evidence
from experiments import round10_speech_evidence as speech
from experiments.round10_domain import bias_select, weights_for
from experiments.round10_fusion import OUTPUT, folds, take, save


def bag_evidence(models: list[dict[str, Any]], data: dict[str, np.ndarray]) -> np.ndarray:
    """在专家证据层等权平均，所有成员都必须未见被评分来源。"""
    return np.mean([evidence.expert_evidence(model, data) for model in models], axis=0)


def search(output: Path) -> None:
    """整来源外层验证选择袋装比例，不触碰历史测试标签。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖袋装证据选择")
    speech.configure()
    data = speech.data_for(("train", "selection"))
    grid = [{**config, "full_expert_weight": weight, "id": f"bag_{i}_{weight}"}
            for i, config in enumerate(evidence.options()) for weight in (0., .5, 1.)]
    save(output / "preregister.json", {"hypothesis": "reduce single full-refit expert confidence shift using inner-fold ensemble evidence",
        "protocol": "4 outer source folds,3 inner source folds; every scored source excluded from all fitted experts",
        "configs": grid, "final_seeds": [42, 43, 44, 45, 46], "expert_classes": evidence.EXPERT_CLASSES,
        "prior_historical_mean": .6937360668199399, "adaptive_historical_diagnostic": True})
    predictions = np.full((len(grid), len(data["labels"]), 3), np.nan)
    for fold, (tr, va) in enumerate(folds(data)):
        training, validation = take(data, tr), take(data, va)
        inner = None
        members = []
        for it, iv in folds(training, 3):
            member = evidence.fit_experts(take(training, it))
            observed = evidence.expert_evidence(member, take(training, iv))
            if inner is None:
                inner = np.full((len(tr), observed.shape[1]), np.nan)
            inner[iv] = observed
            members.append(member)
        full = evidence.expert_evidence(evidence.fit_experts(training), validation)
        bag = bag_evidence(members, validation)
        if not np.isfinite(inner).all():
            raise ValueError("交叉拟合不完整")
        for i, config in enumerate(grid):
            if i % 3 == 0:
                weights = weights_for(training["labels"], training["source_groups"], 0., config["class_power"])
                model = evidence.meta(config).fit(inner, training["labels"], sample_weight=weights)
            mixture = config["full_expert_weight"]*full+(1-config["full_expert_weight"])*bag
            predictions[i, va] = np.log(np.maximum(model.predict_proba(mixture), 1e-12))
        np.savez_compressed(output / f"fold{fold}_evidence.npz", train_sample_ids=training["sample_ids"],
                            validation_sample_ids=validation["sample_ids"], inner_evidence=inner,
                            full_outer_evidence=full, bag_outer_evidence=bag)
        print(json.dumps({"bag_outer_fold": fold}), flush=True)
    rows = [{"config": config, **bias_select(data, z)} for config, z in zip(grid, predictions)]
    winner = max(rows, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "search_complete.json", {"results": rows})
    save(output / "selection_locked.json", winner)
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps({"winner": winner}), flush=True)


def refit(output: Path) -> None:
    """元训练仍交叉拟合，推理专家的袋装比例在外层OOF锁定。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖袋装证据模型")
    speech.configure()
    data = speech.data_for(("train", "selection", "calibration"))
    lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
    inner, members = None, []
    for tr, va in folds(data):
        member = evidence.fit_experts(take(data, tr))
        observed = evidence.expert_evidence(member, take(data, va))
        if inner is None:
            inner = np.full((len(data["labels"]), observed.shape[1]), np.nan)
        inner[va] = observed
        members.append(member)
    weights = weights_for(data["labels"], data["source_groups"], 0., lock["config"]["class_power"])
    models = [evidence.meta(lock["config"]).set_params(random_seed=seed).fit(inner, data["labels"], sample_weight=weights)
              for seed in range(42, 47)]
    full = evidence.fit_experts(data) if lock["config"]["full_expert_weight"] else None
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"members": members, "full_experts": full, "meta_models": models, "selection": lock,
                     "training_groups": np.unique(data["source_groups"]), "training_sample_ids": data["sample_ids"]}, handle)


def main() -> None:
    """先开发选择，再按锁定配方训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit"))
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    if args.fine:
        evidence.EXPERT_CLASSES = 5
    output = OUTPUT / ("speech_bag5" if args.fine else "speech_bag")
    output.mkdir(exist_ok=True)
    {"search": search, "refit": refit}[args.action](output)


if __name__ == "__main__":
    main()
