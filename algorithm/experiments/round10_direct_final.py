"""保留冻结原始情绪输出的最终融合，沿用同一五种子主验收。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from experiments import round10_robust_final as final
from experiments import round10_evidence as evidence
from experiments import round10_large_speech as large
from experiments.round10_direct_evidence import raw_evidence
from experiments.round10_domain import weights_for
from experiments.round10_fusion import OUTPUT, save


def configure() -> None:
    """固定来源选型规则，只改变元模型可见的真实观测信息。"""
    final.FAMILY_STAGES = ("direct_evidence", "direct_evidence5")
    final.FINAL_DIRECTORY = "direct_final"
    final.configure = large.configure
    final.data_for = large.data_for


def refit(output: Path) -> None:
    """基专家复用同一3579训练集；元训练仍来自严格来源交叉拟合。"""
    if (output / "model.pkl").exists():
        raise ValueError("拒绝覆盖原始输出融合模型")
    large.configure()
    data = large.data_for(("train", "selection", "calibration"))
    lock_path = output / "selection_locked.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    with (OUTPUT / "large_final" / "model.pkl").open("rb") as handle:
        previous = pickle.load(handle)
    if not np.array_equal(previous["training_sample_ids"], data["sample_ids"]):
        raise ValueError("复用基专家的训练ID不一致")
    raw = raw_evidence(data)
    families = []
    for member, base, classes in zip(lock["members"][:2], previous["families"], (3, 5)):
        evidence.EXPERT_CLASSES = classes
        try:
            features = np.column_stack([evidence.crossfit(data, 4), raw])
        finally:
            evidence.EXPERT_CLASSES = 3
        config = member["config"]
        weights = weights_for(data["labels"], data["source_groups"], 0., config["class_power"])
        models = []
        for seed in lock["seeds"]:
            models.append(evidence.meta(config).set_params(random_seed=seed).fit(features, data["labels"], sample_weight=weights))
            print(json.dumps({"direct_classes": classes, "meta_seed": seed}), flush=True)
        families.append({"experts": base["experts"], "meta_models": models})
    bundle = {"families": families, "cheap": previous["cheap"], "selection": lock,
              "raw_foundation_evidence": True, "training_groups": previous["training_groups"],
              "training_sample_ids": previous["training_sample_ids"]}
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    save(output / "model_manifest.json", {"training_n": 3579, "groups": len(bundle["training_groups"]),
        "seeds": lock["seeds"], "raw_target_independent_dimensions": raw.shape[1],
        "selection_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "model.pkl").read_bytes()).hexdigest(),
        "test_labels_used_in_fit_or_inference": False})


def main() -> None:
    """所有参数先在开发来源锁定，训练后才执行历史诊断。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select", "refit", "evaluate"))
    args = parser.parse_args()
    configure()
    output = OUTPUT / "direct_final"
    output.mkdir(exist_ok=True)
    {"select": final.select, "refit": refit, "evaluate": final.evaluate}[args.action](output)


if __name__ == "__main__":
    main()
