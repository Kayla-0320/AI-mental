"""将冻结基础模型的原始情绪输出保留给读出器，减少线性专家信息压缩。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from experiments import round10_evidence as evidence
from experiments import round10_large_speech as large
from experiments.round10_domain import bias_select, weights_for
from experiments.round10_fusion import OUTPUT, save, load_npz, folds, take


def raw_evidence(data: dict[str, np.ndarray]) -> np.ndarray:
    """只读冻结模型输出和可用性，不读取标签、来源或跨样本统计。"""
    p = softmax(data["llm_scores"], axis=1)
    entropy = -(p*np.log(np.maximum(p, 1e-12))).sum(1)
    voice_mask = data["speech_available"] & data["modality_mask"][:, 1] & (data["quality"][:, 1] > 0)
    voice = np.where(voice_mask[:, None], data["speech_emotion"], 0.)
    face_mask = data["face_summary"][:, 24] > 0
    face = np.where(face_mask[:, None], data["face_summary"][:, :24], 0.)
    result = np.column_stack([p, entropy, voice, voice_mask, face, face_mask])
    if not np.isfinite(result).all():
        raise ValueError("冻结原始情绪证据含非有限有效输入")
    return result


def search(output: Path, fine: bool) -> None:
    """复用同一严格嵌套专家OOF，新增特征本身不含任何目标监督。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖原始证据选择")
    data = large.data_for(("train", "selection"))
    stage = "large_speech5" if fine else "large_speech"
    raw = raw_evidence(data)
    grid = evidence.options()
    save(output / "preregister.json", {"source_nested_evidence": stage,
        "extra_features": "frozen LLM three probabilities and entropy,3 speech emotion dimensions and mask,24 expression stats and mask",
        "features_target_independent": True, "configs": grid, "seeds": [42, 43, 44, 45, 46],
        "protocol": "same4 outer/3 inner source folds; frozen features appended to both inner and outer observations",
        "previous_historical_target_achieved": False, "adaptive_historical_diagnostic": True})
    predictions = np.full((len(grid), len(data["labels"]), 3), np.nan)
    for fold, (tr, va) in enumerate(folds(data)):
        cached = load_npz(OUTPUT / stage / f"fold{fold}_evidence.npz")
        if not np.array_equal(cached["train_sample_ids"], data["sample_ids"][tr]) or not np.array_equal(cached["validation_sample_ids"], data["sample_ids"][va]):
            raise ValueError("复用的嵌套证据与本轮来源折不一致")
        x = np.column_stack([cached["inner_evidence"], raw[tr]])
        v = np.column_stack([cached["outer_evidence"], raw[va]])
        for i, config in enumerate(grid):
            weights = weights_for(data["labels"][tr], data["source_groups"][tr], 0., config["class_power"])
            model = evidence.meta(config).fit(x, data["labels"][tr], sample_weight=weights)
            predictions[i, va] = np.log(np.maximum(model.predict_proba(v), 1e-12))
        print(json.dumps({"direct_outer_fold": fold, "dimensions": x.shape[1]}), flush=True)
    rows = [{"config": config, **bias_select(data, z)} for config, z in zip(grid, predictions)]
    winner = max(rows, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    save(output / "search_complete.json", {"results": rows})
    save(output / "selection_locked.json", winner)
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps({"winner": winner}), flush=True)


def main() -> None:
    """两个专家粒度仅在开发来源中比较。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    output = OUTPUT / ("direct_evidence5" if args.fine else "direct_evidence")
    output.mkdir(exist_ok=True)
    search(output, args.fine)


if __name__ == "__main__":
    main()
