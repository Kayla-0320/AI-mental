"""在真实配对特征上评测已知故障元数据处理，故障注入不冒充实测噪声真值。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from data_quality import assess_modality
from shared import ModalityObservation

MODALITIES = ("text", "audio", "visual")
SCENARIOS = ("clean", "audio_missing_50pct", "visual_stale_50pct", "audio_misaligned_50pct", "audio_nonfinite_50pct", "random_modality_missing_50pct", "all_missing")


def metrics(labels: np.ndarray, probabilities: np.ndarray, covered: np.ndarray) -> dict[str, Any]:
    """报告全样本和保留样本指标，避免通过丢弃困难样本制造提升。"""
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    result: dict[str, Any] = {
        "n": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "coverage": float(covered.mean()), "covered_n": int(covered.sum()),
        "covered_macro_f1": None,
    }
    if covered.any():
        result["covered_macro_f1"] = float(f1_score(labels[covered], predictions[covered], labels=[0, 1, 2], average="macro", zero_division=0))
    return result


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """固定模型与测试集，比较使用/忽略已知模态故障元数据的行为。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        train = {key: archive[key] for key in archive.files}
    with np.load(data_dir / "test.npz", allow_pickle=False) as archive:
        test = {key: archive[key] for key in archive.files}
    models: list[Any] = []
    for name in MODALITIES:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        model.fit(train[name], train["labels"])
        models.append(model)
    prior = np.bincount(train["labels"], minlength=3).astype(float)
    prior /= prior.sum()
    count = len(test["labels"])
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        rng = np.random.default_rng(42)
        affected = np.zeros(count, dtype=bool)
        affected[rng.choice(count, count // 2, replace=False)] = True
        branch_probs: list[np.ndarray] = []
        reliability = np.zeros((count, 3), dtype=float)
        reasons: Counter[str] = Counter()
        for j, name in enumerate(MODALITIES):
            sanitized = np.zeros_like(test[name])
            naive = np.nan_to_num(test[name].copy())
            for i in range(count):
                features = test[name][i].astype(float).tolist()
                available = bool(test["modality_mask"][i, j])
                quality = float(test["quality"][i, j])
                observed_at = 100.0
                start, end = 1.0, 2.0
                damaged = affected[i]
                if scenario == "all_missing" or (scenario == "audio_missing_50pct" and name == "audio" and damaged) or (scenario == "random_modality_missing_50pct" and rng.random() < 0.5):
                    available = False
                    features = [0.0] * len(features)
                    naive[i] = 0
                if scenario == "visual_stale_50pct" and name == "visual" and damaged:
                    observed_at = 50.0
                    features = test[name][(i + 1) % count].astype(float).tolist()
                    naive[i] = features
                if scenario == "audio_misaligned_50pct" and name == "audio" and damaged:
                    start, end = 3.0, 4.0
                    features = test[name][(i + 1) % count].astype(float).tolist()
                    naive[i] = features
                if scenario == "audio_nonfinite_50pct" and name == "audio" and damaged:
                    features[0] = float("nan")
                    naive[i, 0] = 0.0
                report = assess_modality(ModalityObservation(modality=name, features=features, available=available, quality_score=quality, observed_at=observed_at, segment_start=start, segment_end=end), now=102.0, reference_start=1.0, reference_end=2.0)
                sanitized[i] = report.sanitized_features
                reliability[i, j] = report.reliability
                reasons.update(report.reason_codes)
            # 仅根据质量mask决定分支是否参与；分支模型与融合系数完全固定。
            branch_probs.append(models[j].predict_proba(naive))
        probability_stack = np.stack(branch_probs, axis=1)
        unmasked = probability_stack.mean(axis=1)
        covered = reliability.sum(axis=1) > 0
        masked = np.tile(prior, (count, 1))
        masked[covered] = (probability_stack[covered] * reliability[covered, :, None]).sum(axis=1) / reliability[covered].sum(axis=1, keepdims=True)
        scenarios[scenario] = {
            "ignoring_fault_metadata": metrics(test["labels"], unmasked, np.ones(count, dtype=bool)),
            "quality_aware": metrics(test["labels"], masked, covered),
            "quality_reason_counts": dict(reasons),
        }
    result = {
        "seed": 42, "data_dir": str(data_dir), "dataset": "CH-SIMS-v1 source-video holdout",
        "feature_inputs": "author frozen features, not VLM", "train_n": len(train["labels"]), "test_n": count,
        "model": "three train-only standardized LogisticRegression branches; equal weighted probability fusion",
        "tuning": "fixed C=1; no test model selection or threshold tuning; no test calibration",
        "faults": "synthetic missing/stale/misaligned/non-finite faults with supplied fault metadata; NOT real-world fault detection accuracy",
        "all_missing_policy": "abstain; report train-prior fallback only for diagnostic scoring; coverage=0; no emotion inference",
        "scenarios": scenarios,
    }
    (output_dir / "quality_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
